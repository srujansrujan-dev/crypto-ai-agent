"""
main.py — Entry point for the Crypto AI Agent.

Loop:
  1. Fetch market data (500+ coins via CoinGecko)
  2. Calculate indicators for each coin
  3. Run pump detection scoring
  4. Detect multi-cycle trends
  5. AI-evaluate top opportunities (Gemini)
  6. Save signals to SQLite
  7. Check outcomes of past signals → label WIN/LOSS/NEUTRAL
  8. Adjust scoring weights (self-learning)
  9. Sleep 5 minutes → repeat
"""

import logging
import os
import time
from datetime import datetime, timezone

from config import (
    ALLOWED_SIGNAL_ACTIONS,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TARGET_PCT,
    LOG_FILE,
    MAX_AI_EVALUATIONS_PER_CYCLE,
    MAX_SIGNALS_PER_CYCLE,
    MIN_SIGNAL_CONFIDENCE,
    MIN_SIGNAL_PUMP_SCORE,
    MIN_SIGNAL_QUALITY_SCORE,
    MIN_SIGNAL_TREND_SCORE,
    SCAN_INTERVAL_SECONDS,
)
from scanner        import fetch_market_data
from indicators     import calculate_indicators
from pump_detector  import PumpDetector
from trend_detector import TrendDetector
from ai_engine      import analyse
from learning_engine import (
    adjust_weights,
    check_outcomes,
    get_stats,
    has_recent_pending_signal,
    init_db,
    load_weights,
    save_signal,
)
from signals  import TradingSignal
from alerts   import send_signal, send_cycle_summary, send_startup_banner
from dashboard import start_dashboard


# ── Logging setup ──────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


# ── Agent ─────────────────────────────────────────────────────────────────────

def build_signal(
    snap,
    ind,
    pump_score,
    ai_action,
    ai_confidence,
    ai_reason,
    quality_score,
    trend_score,
) -> TradingSignal:
    entry = snap.current_price
    return TradingSignal(
        asset_id     = snap.id,
        coin         = snap.name,
        symbol       = snap.symbol,
        timestamp    = datetime.now(timezone.utc).isoformat(),
        entry_price  = entry,
        target_price = round(entry * (1 + DEFAULT_TARGET_PCT), 8),
        stop_loss    = round(entry * (1 - DEFAULT_STOP_LOSS_PCT), 8),
        buy_zone_low = round(entry * 0.99, 8),
        buy_zone_high= round(entry * 1.01, 8),
        confidence   = ai_confidence,
        ai_action    = ai_action,
        ai_reason    = ai_reason,
        pump_score   = pump_score.total_score,
        quality_score= quality_score,
        trend_score  = trend_score,
        volume_ratio = ind.volume_ratio,
        momentum     = ind.momentum,
        rsi          = float(ind.rsi or 50.0),
        market_cap   = snap.market_cap,
        price_change_24h = snap.price_change_24h,
    )


def rank_opportunity(snap, ind, pump_score, trend_info) -> float:
    trend_score = trend_info.get("trend_score", 0.0) if trend_info else 0.0
    rsi = ind.rsi or 50.0
    above_mas = (
        snap.current_price >= (ind.ma20 or snap.current_price)
        and snap.current_price >= (ind.ma50 or snap.current_price)
    )

    score = pump_score.total_score
    score += min(ind.volume_ratio, 6.0) * 2.5
    score += min(max(ind.momentum, 0.0), 20.0)
    score += min(trend_score, 60.0) * 0.2
    if above_mas:
        score += 6
    if 45 <= rsi <= 68:
        score += 6
    elif rsi > 78:
        score -= 18
    elif rsi > 72:
        score -= 10
    score -= min(max(snap.price_change_24h - 18.0, 0.0), 12.0) * 0.8
    return round(score, 2)


def calculate_signal_quality(
    snap,
    ind,
    pump_score,
    ai_action,
    ai_confidence,
    trend_info,
) -> float:
    trend_score = trend_info.get("trend_score", 0.0) if trend_info else 0.0
    rsi = ind.rsi or 50.0
    above_mas = (
        snap.current_price >= (ind.ma20 or snap.current_price)
        and snap.current_price >= (ind.ma50 or snap.current_price)
    )

    quality = pump_score.total_score * 0.42 + ai_confidence * 0.33
    quality += min(ind.volume_ratio / 5.0, 1.0) * 10
    quality += min(max(ind.momentum, 0.0) / 18.0, 1.0) * 10
    quality += min(trend_score / 60.0, 1.0) * 12
    quality += {"BUY": 12, "HOLD": -8, "AVOID": -20}.get(ai_action, 0)

    if above_mas:
        quality += 6

    if rsi > 78:
        quality -= 18
    elif rsi > 72:
        quality -= 10
    elif rsi < 25:
        quality -= 8
    elif 45 <= rsi <= 68:
        quality += 6

    quality -= min(max(snap.price_change_24h - 15.0, 0.0), 15.0) * 0.6
    if snap.market_cap <= 0:
        quality -= 4

    return round(max(0.0, min(100.0, quality)), 2)


def should_emit_signal(
    snap,
    pump_score,
    ai_action,
    ai_confidence,
    quality_score,
    trend_info,
) -> bool:
    trend_score = trend_info.get("trend_score", 0.0) if trend_info else 0.0

    if ai_action not in ALLOWED_SIGNAL_ACTIONS:
        logger.info("Rejecting %s because action %s is below the signal bar", snap.symbol, ai_action)
        return False

    if ai_confidence < MIN_SIGNAL_CONFIDENCE:
        logger.info(
            "Rejecting %s because confidence %.1f is below %.1f",
            snap.symbol, ai_confidence, MIN_SIGNAL_CONFIDENCE,
        )
        return False

    if pump_score.total_score < MIN_SIGNAL_PUMP_SCORE:
        logger.info(
            "Rejecting %s because pump score %.1f is below %.1f",
            snap.symbol, pump_score.total_score, MIN_SIGNAL_PUMP_SCORE,
        )
        return False

    if quality_score < MIN_SIGNAL_QUALITY_SCORE:
        logger.info(
            "Rejecting %s because quality %.1f is below %.1f",
            snap.symbol, quality_score, MIN_SIGNAL_QUALITY_SCORE,
        )
        return False

    if trend_info and trend_score < MIN_SIGNAL_TREND_SCORE and pump_score.total_score < 95:
        logger.info(
            "Rejecting %s because trend score %.1f is below %.1f",
            snap.symbol, trend_score, MIN_SIGNAL_TREND_SCORE,
        )
        return False

    return True


def run_cycle(
    cycle:    int,
    detector: PumpDetector,
    trends:   TrendDetector,
) -> None:
    logger.info("── Cycle %d starting ──", cycle)

    # ── 1. Fetch market data ───────────────────────────────────────────────
    snapshots = fetch_market_data()
    if not snapshots:
        logger.warning("No market data returned — skipping cycle.")
        return

    # ── 2. Build price-map for outcome checking ────────────────────────────
    price_map = {s.id: s.current_price for s in snapshots}

    # ── 3. Calculate indicators ────────────────────────────────────────────
    indicators = [calculate_indicators(snap) for snap in snapshots]

    # ── 4. Update trend detector history ──────────────────────────────────
    trends.update(snapshots)

    # ── 5. Pump detection scan ─────────────────────────────────────────────
    opportunities = detector.scan(snapshots, indicators)
    logger.info("Opportunities flagged: %d", len(opportunities))
    ranked_opportunities = []
    for snap, ind, pump in opportunities:
        trend_info = trends.evaluate(snap.id)
        ranked_opportunities.append(
            (rank_opportunity(snap, ind, pump, trend_info), snap, ind, pump, trend_info)
        )
    ranked_opportunities.sort(key=lambda item: item[0], reverse=True)

    # ── 6. AI evaluation of top 5 opportunities ────────────────────────────
    signals_this_cycle = 0
    for _, snap, ind, pump, trend_info in ranked_opportunities[:MAX_AI_EVALUATIONS_PER_CYCLE]:
        if has_recent_pending_signal(snap.id, snap.symbol):
            logger.info("Skipping %s — recent pending signal already exists", snap.symbol)
            continue

        trend_info = trends.evaluate(snap.id)
        action, confidence, reason = analyse(snap, ind, pump, trend_info)
        quality_score = calculate_signal_quality(
            snap, ind, pump, action, confidence, trend_info,
        )

        if not should_emit_signal(
            snap, pump, action, confidence, quality_score, trend_info,
        ):
            continue

        if action == "AVOID":
            logger.info("Skipping %s — AI says AVOID", snap.symbol)
            continue

        signal = build_signal(
            snap,
            ind,
            pump,
            action,
            confidence,
            reason,
            quality_score,
            trend_info.get("trend_score", 0.0) if trend_info else 0.0,
        )
        save_signal(signal)
        send_signal(signal)
        signals_this_cycle += 1
        if signals_this_cycle >= MAX_SIGNALS_PER_CYCLE:
            break

    # ── 7. Check past signal outcomes ─────────────────────────────────────
    check_outcomes(price_map)

    # ── 8. Self-learning: adjust weights ──────────────────────────────────
    new_weights = adjust_weights()
    detector.set_weights(new_weights)

    # ── 9. Cycle summary ───────────────────────────────────────────────────
    send_cycle_summary(cycle, len(snapshots), len(opportunities), signals_this_cycle)
    stats = get_stats()
    logger.info(
        "Stats — Total: %d | Win rate: %.1f%% | Pending: %d",
        stats["total"], stats["win_rate"], stats["pending"],
    )


def main() -> None:
    send_startup_banner()

    # Initialise DB
    init_db()

    # Load adaptive weights
    weights  = load_weights()
    detector = PumpDetector()
    detector.set_weights(weights)
    trends   = TrendDetector()

    # Start web dashboard (background thread)
    start_dashboard()

    cycle = 1
    while True:
        try:
            run_cycle(cycle, detector, trends)
        except KeyboardInterrupt:
            logger.info("Shutdown requested — exiting.")
            break
        except Exception as exc:
            logger.exception("Unhandled error in cycle %d: %s", cycle, exc)

        cycle += 1
        logger.info("Sleeping %ds until next cycle …", SCAN_INTERVAL_SECONDS)
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
