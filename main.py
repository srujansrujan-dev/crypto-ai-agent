"""
main.py - Entry point for the Crypto AI Agent.

Pipeline:
1. Fetch broad market data
2. Build indicators and trend history
3. Classify the current market regime
4. Run a rough scan to create an internal watchlist
5. Deep-scan only shortlisted coins
6. Final-evaluate the strongest candidates
7. Save only the strongest publishable signals
8. Check outcomes and update learning weights
"""

import logging
import os
import time
from datetime import datetime, timezone

from ai_engine import analyse
from alerts import send_cycle_summary, send_signal, send_startup_banner
from config import (
    ALLOWED_SIGNAL_ACTIONS,
    COINDCX_FUTURES_ONLY,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TARGET_PCT,
    LOG_FILE,
    MAX_AI_EVALUATIONS_PER_CYCLE,
    MAX_SIGNALS_PER_CYCLE,
    MAX_SIGNAL_RISK_SCORE,
    MIN_FUTURES_CONFIRMATION,
    MIN_SIGNAL_CONFIDENCE,
    MIN_SIGNAL_FINAL_SCORE,
    MIN_SIGNAL_LIQUIDITY_SCORE,
    MIN_SIGNAL_PUMP_SCORE,
    MIN_SIGNAL_QUALITY_SCORE,
    MIN_SIGNAL_TREND_SCORE,
    SCAN_INTERVAL_SECONDS,
    WATCHLIST_DEEP_LIMIT,
    WATCHLIST_LOG_LIMIT,
    WATCHLIST_ROUGH_LIMIT,
)
from dashboard import start_dashboard
from indicators import calculate_indicators
from learning_engine import (
    adjust_weights,
    check_outcomes,
    get_stats,
    has_recent_pending_signal,
    init_db,
    load_weights,
    save_signal,
    update_signal_marks,
)
from market_intelligence import (
    analyse_deep_context,
    calculate_final_publish_score,
    classify_market_regime,
    score_watchlist_candidate,
    summarize_futures_context,
)
from pump_detector import PumpDetector
from scanner import fetch_coin_market_chart, fetch_derivatives_tickers, fetch_market_data
from signals import TradingSignal
from trend_detector import TrendDetector


os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger(__name__)


def _direction_from_action(action: str, fallback: str = "LONG") -> str:
    return "SHORT" if str(action or fallback).upper() == "SHORT" else "LONG"


def _trend_score_for_direction(trend_info, direction: str) -> float:
    if not trend_info:
        return 0.0
    if direction == "SHORT":
        return float(trend_info.get("bearish_trend_score", trend_info.get("trend_score", 0.0)) or 0.0)
    return float(trend_info.get("bullish_trend_score", trend_info.get("trend_score", 0.0)) or 0.0)


def _directional_deep_score(deep_context, direction: str) -> float:
    if direction == "SHORT":
        return float(deep_context.get("short_deep_score", deep_context.get("deep_score", 0.0)) or 0.0)
    return float(deep_context.get("long_deep_score", deep_context.get("deep_score", 0.0)) or 0.0)


def _momentum_strength(ind, direction: str) -> float:
    return max(ind.momentum, 0.0) if direction == "LONG" else abs(min(ind.momentum, 0.0))


def _ma_alignment(snap, ind, direction: str) -> bool:
    ma20 = ind.ma20 or snap.current_price
    ma50 = ind.ma50 or snap.current_price
    if direction == "SHORT":
        return snap.current_price <= ma20 and snap.current_price <= ma50
    return snap.current_price >= ma20 and snap.current_price >= ma50


def build_signal(
    snap,
    ind,
    pump_score,
    ai_action,
    ai_confidence,
    ai_reason,
    quality_score,
    final_score,
    deep_context,
    futures_context,
    regime,
    trend_score,
) -> TradingSignal:
    entry = snap.current_price
    direction = _direction_from_action(ai_action, futures_context.get("trade_bias", "LONG"))
    is_short = direction == "SHORT"
    target_price = round(entry * (1 - DEFAULT_TARGET_PCT), 8) if is_short else round(entry * (1 + DEFAULT_TARGET_PCT), 8)
    stop_loss = round(entry * (1 + DEFAULT_STOP_LOSS_PCT), 8) if is_short else round(entry * (1 - DEFAULT_STOP_LOSS_PCT), 8)
    zone_low = round(entry * 0.99, 8)
    zone_high = round(entry * 1.01, 8)

    context_suffix = (
        f" Regime: {regime.get('label', 'UNKNOWN')} ({regime.get('score', 50):.0f}/100)."
        f" Deep score: {_directional_deep_score(deep_context, direction):.0f}/100."
        f" Liquidity: {deep_context.get('liquidity_score', 0):.0f}/100."
        f" Risk proxy: {deep_context.get('risk_score', 0):.0f}/100."
    )
    risk_notes = deep_context.get("risk_notes") or []
    if risk_notes:
        context_suffix += f" Note: {risk_notes[0]}."
    if futures_context.get("has_data"):
        context_suffix += (
            f" CoinDCX futures setup: {futures_context.get('trade_bias', 'UNAVAILABLE')},"
            f" recommended {futures_context.get('leverage_hint', '1x')} leverage,"
            f" funding {futures_context.get('funding_rate', 0.0):.4f},"
            f" OI {futures_context.get('open_interest', 0.0):,.0f},"
            f" execution matrix {futures_context.get('futures_score', 0.0):.0f}/100."
        )

    return TradingSignal(
        asset_id=snap.id,
        coin=snap.name,
        symbol=snap.symbol,
        timestamp=datetime.now(timezone.utc).isoformat(),
        entry_price=entry,
        target_price=target_price,
        stop_loss=stop_loss,
        buy_zone_low=zone_low,
        buy_zone_high=zone_high,
        confidence=ai_confidence,
        ai_action=ai_action,
        ai_reason=f"{ai_reason}{context_suffix}",
        pump_score=pump_score.total_score,
        quality_score=quality_score,
        aggregate_score=final_score,
        deep_score=_directional_deep_score(deep_context, direction),
        liquidity_score=deep_context.get("liquidity_score", 0.0),
        risk_score=deep_context.get("risk_score", 0.0),
        market_regime=regime.get("label", "UNKNOWN"),
        regime_score=regime.get("score", 50.0),
        trend_score=trend_score,
        volume_ratio=ind.volume_ratio,
        momentum=ind.momentum,
        rsi=float(ind.rsi or 50.0),
        market_cap=snap.market_cap,
        price_change_24h=snap.price_change_24h,
        futures_bias=futures_context.get("trade_bias", "UNAVAILABLE"),
        leverage_hint=futures_context.get("leverage_hint", "Unavailable"),
        futures_exchange=futures_context.get("futures_exchange", ""),
        futures_symbol=futures_context.get("futures_symbol", ""),
        funding_rate=futures_context.get("funding_rate", 0.0),
        open_interest=futures_context.get("open_interest", 0.0),
        basis=futures_context.get("basis", 0.0),
        spread=futures_context.get("spread", 0.0),
        futures_volume_24h=futures_context.get("futures_volume_24h", 0.0),
        futures_score=futures_context.get("futures_score", 0.0),
    )


def rank_opportunity(snap, ind, pump_score, trend_info) -> float:
    direction = _direction_from_action(getattr(pump_score, "direction", "LONG"))
    trend_score = _trend_score_for_direction(trend_info, direction)
    rsi = ind.rsi or 50.0

    score = pump_score.total_score
    score += min(ind.volume_ratio, 6.0) * 2.5
    score += min(_momentum_strength(ind, direction), 20.0)
    score += min(trend_score, 60.0) * 0.2
    if _ma_alignment(snap, ind, direction):
        score += 6

    if direction == "LONG":
        if 45 <= rsi <= 68:
            score += 6
        elif rsi > 78:
            score -= 18
        elif rsi > 72:
            score -= 10
        score -= min(max(snap.price_change_24h - 18.0, 0.0), 12.0) * 0.8
    else:
        if 34 <= rsi <= 58:
            score += 6
        elif rsi < 22:
            score -= 18
        elif rsi < 28:
            score -= 10
        score -= min(max(abs(min(snap.price_change_24h, 0.0)) - 18.0, 0.0), 12.0) * 0.8

    return round(score, 2)


def calculate_signal_quality(
    snap,
    ind,
    pump_score,
    ai_action,
    ai_confidence,
    trend_info,
) -> float:
    direction = _direction_from_action(ai_action, getattr(pump_score, "direction", "LONG"))
    trend_score = _trend_score_for_direction(trend_info, direction)
    rsi = ind.rsi or 50.0

    quality = pump_score.total_score * 0.42 + ai_confidence * 0.33
    quality += min(ind.volume_ratio / 5.0, 1.0) * 10
    quality += min(_momentum_strength(ind, direction) / 18.0, 1.0) * 10
    quality += min(trend_score / 60.0, 1.0) * 12
    quality += {"BUY": 12, "SHORT": 12, "HOLD": -8, "AVOID": -20}.get(ai_action, 0)

    if _ma_alignment(snap, ind, direction):
        quality += 6

    if direction == "LONG":
        if rsi > 78:
            quality -= 18
        elif rsi > 72:
            quality -= 10
        elif rsi < 25:
            quality -= 8
        elif 45 <= rsi <= 68:
            quality += 6
        quality -= min(max(snap.price_change_24h - 15.0, 0.0), 15.0) * 0.6
    else:
        if rsi < 22:
            quality -= 18
        elif rsi < 28:
            quality -= 10
        elif rsi > 75:
            quality -= 6
        elif 34 <= rsi <= 58:
            quality += 6
        quality -= min(max(abs(min(snap.price_change_24h, 0.0)) - 15.0, 0.0), 15.0) * 0.6

    if snap.market_cap <= 0:
        quality -= 4

    return round(max(0.0, min(100.0, quality)), 2)


def calculate_final_base_score(
    snap,
    ind,
    pump_score,
    ai_action,
    ai_confidence,
    quality_score,
    trend_info,
) -> float:
    direction = _direction_from_action(ai_action, getattr(pump_score, "direction", "LONG"))
    trend_score = _trend_score_for_direction(trend_info, direction)
    rsi = ind.rsi or 50.0

    final_score = pump_score.total_score * 0.28
    final_score += quality_score * 0.34
    final_score += ai_confidence * 0.18
    final_score += min(trend_score, 60.0) * 0.10
    final_score += min(ind.volume_ratio, 6.0) * 2.2
    final_score += min(_momentum_strength(ind, direction), 20.0) * 0.55
    final_score += {"BUY": 8, "SHORT": 8, "HOLD": -10, "AVOID": -24}.get(ai_action, -8)

    if _ma_alignment(snap, ind, direction):
        final_score += 4

    if direction == "LONG":
        if 45 <= rsi <= 68:
            final_score += 4
        elif rsi > 78:
            final_score -= 18
        elif rsi > 72:
            final_score -= 10
        elif rsi < 25:
            final_score -= 6

        if snap.price_change_24h >= 18:
            final_score -= 8
        elif snap.price_change_24h >= 12:
            final_score -= 4
    else:
        if 34 <= rsi <= 58:
            final_score += 4
        elif rsi < 22:
            final_score -= 18
        elif rsi < 28:
            final_score -= 10
        elif rsi > 75:
            final_score -= 4

        if snap.price_change_24h <= -18:
            final_score -= 8
        elif snap.price_change_24h <= -12:
            final_score -= 4

    return round(max(0.0, min(100.0, final_score)), 2)


def should_emit_signal(
    snap,
    pump_score,
    ai_action,
    ai_confidence,
    quality_score,
    final_score,
    deep_context,
    futures_context,
    regime,
    trend_info,
) -> bool:
    direction = _direction_from_action(ai_action, getattr(pump_score, "direction", "LONG"))
    trend_score = _trend_score_for_direction(trend_info, direction)
    liquidity_score = deep_context.get("liquidity_score", 0.0)
    risk_score = deep_context.get("risk_score", 0.0)
    market_regime = regime.get("label", "UNKNOWN")
    futures_bias = futures_context.get("trade_bias", "UNAVAILABLE")
    futures_score = futures_context.get("futures_score", 0.0)

    if ai_action not in ALLOWED_SIGNAL_ACTIONS:
        logger.info("Rejecting %s because action %s is below the signal bar", snap.symbol, ai_action)
        return False

    if ai_confidence < MIN_SIGNAL_CONFIDENCE:
        logger.info(
            "Rejecting %s because confidence %.1f is below %.1f",
            snap.symbol,
            ai_confidence,
            MIN_SIGNAL_CONFIDENCE,
        )
        return False

    if pump_score.total_score < MIN_SIGNAL_PUMP_SCORE:
        logger.info(
            "Rejecting %s because opportunity score %.1f is below %.1f",
            snap.symbol,
            pump_score.total_score,
            MIN_SIGNAL_PUMP_SCORE,
        )
        return False

    if quality_score < MIN_SIGNAL_QUALITY_SCORE:
        logger.info(
            "Rejecting %s because quality %.1f is below %.1f",
            snap.symbol,
            quality_score,
            MIN_SIGNAL_QUALITY_SCORE,
        )
        return False

    if final_score < MIN_SIGNAL_FINAL_SCORE:
        logger.info(
            "Rejecting %s because final score %.1f is below %.1f",
            snap.symbol,
            final_score,
            MIN_SIGNAL_FINAL_SCORE,
        )
        return False

    if trend_info and trend_score < MIN_SIGNAL_TREND_SCORE and pump_score.total_score < 95:
        logger.info(
            "Rejecting %s because directional trend score %.1f is below %.1f",
            snap.symbol,
            trend_score,
            MIN_SIGNAL_TREND_SCORE,
        )
        return False

    if liquidity_score < MIN_SIGNAL_LIQUIDITY_SCORE:
        logger.info(
            "Rejecting %s because liquidity %.1f is below %.1f",
            snap.symbol,
            liquidity_score,
            MIN_SIGNAL_LIQUIDITY_SCORE,
        )
        return False

    if risk_score > MAX_SIGNAL_RISK_SCORE:
        logger.info(
            "Rejecting %s because risk proxy %.1f is above %.1f",
            snap.symbol,
            risk_score,
            MAX_SIGNAL_RISK_SCORE,
        )
        return False

    if direction == "LONG" and market_regime == "PANIC" and snap.symbol not in {"BTC", "ETH", "SOL"}:
        logger.info("Rejecting %s because market regime is PANIC for longs", snap.symbol)
        return False
    if direction == "SHORT" and market_regime == "RISK_ON" and snap.symbol not in {"BTC", "ETH", "SOL"}:
        logger.info("Rejecting %s because shorting into a strong risk-on regime is lower quality", snap.symbol)
        return False

    if not futures_context.get("has_data"):
        logger.info("Rejecting %s because CoinDCX futures data is unavailable", snap.symbol)
        return False

    if (direction == "LONG" and futures_bias != "LONG") or (direction == "SHORT" and futures_bias != "SHORT"):
        logger.info(
            "Rejecting %s because action %s does not align with CoinDCX futures setup %s",
            snap.symbol,
            ai_action,
            futures_bias,
        )
        return False

    if futures_score < MIN_FUTURES_CONFIRMATION:
        logger.info(
            "Rejecting %s because futures execution score %.1f is below %.1f",
            snap.symbol,
            futures_score,
            MIN_FUTURES_CONFIRMATION,
        )
        return False

    return True


def run_cycle(
    cycle: int,
    detector: PumpDetector,
    trends: TrendDetector,
) -> None:
    logger.info("----- Cycle %d starting -----", cycle)

    snapshots = fetch_market_data()
    if not snapshots:
        logger.warning("No market data returned - skipping cycle.")
        return

    price_map = {snap.id: snap.current_price for snap in snapshots}
    indicators = [calculate_indicators(snap) for snap in snapshots]
    trends.update(snapshots)

    market_regime = classify_market_regime(snapshots)
    logger.info(
        "Market regime - %s score=%.1f breadth24=%.2f median24h=%.2f%%",
        market_regime["label"],
        market_regime["score"],
        market_regime["breadth_24h"],
        market_regime["median_change_24h"],
    )

    opportunities = detector.scan(snapshots, indicators)
    logger.info("Opportunities flagged: %d", len(opportunities))

    rough_candidates = []
    for snap, ind, pump in opportunities:
        if COINDCX_FUTURES_ONLY and not snap.coindcx_has_futures:
            continue
        trend_info = trends.evaluate(snap.id)
        rough_candidates.append(
            {
                "rough_score": rank_opportunity(snap, ind, pump, trend_info),
                "snapshot": snap,
                "indicators": ind,
                "pump": pump,
                "trend_info": trend_info,
            }
        )

    if COINDCX_FUTURES_ONLY:
        logger.info("CoinDCX futures-eligible opportunities: %d", len(rough_candidates))
    rough_candidates.sort(key=lambda item: item["rough_score"], reverse=True)

    watchlist = []
    for index, candidate in enumerate(rough_candidates[:WATCHLIST_ROUGH_LIMIT]):
        snap = candidate["snapshot"]
        history = fetch_coin_market_chart(snap.id, days=7) if index < WATCHLIST_DEEP_LIMIT else {}
        deep_context = analyse_deep_context(
            snap,
            candidate["indicators"],
            candidate["trend_info"],
            history,
            market_regime,
        )
        candidate["deep_context"] = deep_context
        candidate["watchlist_score"] = score_watchlist_candidate(
            snap,
            candidate["pump"],
            candidate["rough_score"],
            deep_context,
            market_regime,
        )
        watchlist.append(candidate)

    watchlist.sort(key=lambda item: item["watchlist_score"], reverse=True)
    if watchlist:
        top_watchlist = ", ".join(
            f"{item['snapshot'].symbol}:{item['watchlist_score']:.1f}"
            for item in watchlist[:WATCHLIST_LOG_LIMIT]
        )
        logger.info("Internal watchlist - %s", top_watchlist)

    derivatives = fetch_derivatives_tickers() if watchlist else []

    candidate_signals = []
    for candidate in watchlist[:MAX_AI_EVALUATIONS_PER_CYCLE]:
        snap = candidate["snapshot"]
        ind = candidate["indicators"]
        pump = candidate["pump"]
        trend_info = candidate["trend_info"]
        deep_context = candidate["deep_context"]
        futures_context = summarize_futures_context(
            snap,
            ind,
            trend_info,
            deep_context,
            market_regime,
            derivatives,
        )

        if has_recent_pending_signal(snap.id, snap.symbol):
            logger.info("Skipping %s - recent pending signal already exists", snap.symbol)
            continue

        action, confidence, reason = analyse(snap, ind, pump, trend_info, futures_context)
        quality_score = calculate_signal_quality(
            snap,
            ind,
            pump,
            action,
            confidence,
            trend_info,
        )
        base_final_score = calculate_final_base_score(
            snap,
            ind,
            pump,
            action,
            confidence,
            quality_score,
            trend_info,
        )
        final_score = calculate_final_publish_score(
            snap,
            action,
            confidence,
            quality_score,
            base_final_score,
            deep_context,
            market_regime,
            futures_context,
        )

        if not should_emit_signal(
            snap,
            pump,
            action,
            confidence,
            quality_score,
            final_score,
            deep_context,
            futures_context,
            market_regime,
            trend_info,
        ):
            continue

        signal = build_signal(
            snap,
            ind,
            pump,
            action,
            confidence,
            reason,
            quality_score,
            final_score,
            deep_context,
            futures_context,
            market_regime,
            _trend_score_for_direction(
                trend_info,
                _direction_from_action(action, getattr(pump, "direction", "LONG")),
            ),
        )
        candidate_signals.append(signal)

    candidate_signals.sort(
        key=lambda signal: (
            signal.aggregate_score,
            signal.deep_score,
            signal.quality_score,
            signal.confidence,
            signal.pump_score,
        ),
        reverse=True,
    )

    signals_this_cycle = 0
    for signal in candidate_signals[:MAX_SIGNALS_PER_CYCLE]:
        save_signal(signal)
        send_signal(signal)
        signals_this_cycle += 1

    update_signal_marks(price_map)
    check_outcomes(price_map)
    new_weights = adjust_weights()
    detector.set_weights(new_weights)

    send_cycle_summary(cycle, len(snapshots), len(opportunities), signals_this_cycle)
    stats = get_stats()
    logger.info(
        "Stats - Total: %d | Win rate: %.1f%% | Pending: %d",
        stats["total"],
        stats["win_rate"],
        stats["pending"],
    )


def main() -> None:
    send_startup_banner()
    init_db()

    weights = load_weights()
    detector = PumpDetector()
    detector.set_weights(weights)
    trends = TrendDetector()

    start_dashboard()

    cycle = 1
    while True:
        try:
            run_cycle(cycle, detector, trends)
        except KeyboardInterrupt:
            logger.info("Shutdown requested - exiting.")
            break
        except Exception as exc:
            logger.exception("Unhandled error in cycle %d: %s", cycle, exc)

        cycle += 1
        logger.info("Sleeping %ds until next cycle...", SCAN_INTERVAL_SECONDS)
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
