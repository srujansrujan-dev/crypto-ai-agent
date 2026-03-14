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
    SCAN_INTERVAL_SECONDS,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TARGET_PCT,
    LOG_FILE,
)
from scanner        import fetch_market_data
from indicators     import calculate_indicators
from pump_detector  import PumpDetector
from trend_detector import TrendDetector
from ai_engine      import analyse
from learning_engine import (
    init_db, save_signal, check_outcomes,
    adjust_weights, load_weights, get_stats,
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
    snap, ind, pump_score, ai_action, ai_confidence, ai_reason
) -> TradingSignal:
    entry = snap.current_price
    return TradingSignal(
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
    )


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
    price_map = {s.symbol: s.current_price for s in snapshots}

    # ── 3. Calculate indicators ────────────────────────────────────────────
    indicators = [calculate_indicators(snap) for snap in snapshots]

    # ── 4. Update trend detector history ──────────────────────────────────
    trends.update(snapshots)

    # ── 5. Pump detection scan ─────────────────────────────────────────────
    opportunities = detector.scan(snapshots, indicators)
    logger.info("Opportunities flagged: %d", len(opportunities))

    # ── 6. AI evaluation of top 5 opportunities ────────────────────────────
    signals_this_cycle = 0
    for snap, ind, pump in opportunities[:5]:
        trend_info = trends.evaluate(snap.id)
        action, confidence, reason = analyse(snap, ind, pump, trend_info)

        if action == "AVOID":
            logger.info("Skipping %s — AI says AVOID", snap.symbol)
            continue

        signal = build_signal(snap, ind, pump, action, confidence, reason)
        save_signal(signal)
        send_signal(signal)
        signals_this_cycle += 1

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
