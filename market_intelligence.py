"""
market_intelligence.py - Higher-level market context and candidate scoring.

This layer adds:
- market regime classification
- deeper historical context for shortlisted coins
- liquidity and risk proxies
- final publish scoring

It is intentionally heuristic and data-source aware: if richer data is not
available for a token, it falls back to neutral assumptions instead of forcing
bad conclusions.
"""

from statistics import median
from typing import Dict, List, Optional

from config import COINDCX_RECOMMENDED_LEVERAGE_CAP
from scanner import fetch_coindcx_futures_prices, fetch_coindcx_instrument_details
from signals import IndicatorSet, MarketSnapshot, PumpScore

MAJOR_SYMBOLS = {"BTC", "ETH", "SOL", "BNB", "XRP"}
FUTURES_UNAVAILABLE = "UNAVAILABLE"
FUTURES_WAIT = "WAIT"
FUTURES_LONG = "LONG"
FUTURES_SHORT = "SHORT"


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _directional_trend_score(trend_info: Optional[dict], direction: str) -> float:
    if not trend_info:
        return 0.0
    if direction == FUTURES_SHORT:
        return _safe_float(
            trend_info.get("bearish_trend_score", trend_info.get("trend_score", 0.0)),
            0.0,
        )
    return _safe_float(
        trend_info.get("bullish_trend_score", trend_info.get("trend_score", 0.0)),
        0.0,
    )


def _directional_deep_score(deep_context: dict, direction: str) -> float:
    if direction == FUTURES_SHORT:
        return _safe_float(deep_context.get("short_deep_score", deep_context.get("deep_score", 0.0)), 0.0)
    return _safe_float(deep_context.get("long_deep_score", deep_context.get("deep_score", 0.0)), 0.0)


def _action_matches_bias(action: str, bias: str) -> bool:
    action_upper = str(action or "").upper()
    bias_upper = str(bias or "").upper()
    return (action_upper == "BUY" and bias_upper == FUTURES_LONG) or (
        action_upper == "SHORT" and bias_upper == FUTURES_SHORT
    )


def _weighted_average(rows: List[dict], value_key: str, weight_key: str) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for row in rows:
        weight = max(_safe_float(row.get(weight_key), 1.0), 1.0)
        value = _safe_float(row.get(value_key), 0.0)
        weighted_sum += value * weight
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


def _normalise_symbol(text: str) -> str:
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def classify_market_regime(snapshots: List[MarketSnapshot]) -> dict:
    """Classify the current market regime from broad cross-sectional behavior."""
    liquid = [snap for snap in snapshots if snap.current_price > 0 and snap.total_volume > 0]
    ranked = sorted(
        [snap for snap in liquid if snap.market_cap > 0],
        key=lambda snap: snap.market_cap,
        reverse=True,
    )
    top = ranked[:100] if ranked else liquid[:100]

    if not top:
        return {
            "label": "UNKNOWN",
            "score": 50.0,
            "breadth_24h": 0.5,
            "breadth_7d": 0.5,
            "median_change_24h": 0.0,
            "median_change_7d": 0.0,
            "alt_multiplier": 0.95,
            "notes": ["market breadth unavailable"],
        }

    breadth_24h = sum(1 for snap in top if snap.price_change_24h > 0) / len(top)
    breadth_7d = sum(1 for snap in top if snap.price_change_7d > 0) / len(top)
    median_24h = median(snap.price_change_24h for snap in top)
    median_7d = median(snap.price_change_7d for snap in top)

    btc = next((snap for snap in snapshots if snap.symbol == "BTC"), None)
    eth = next((snap for snap in snapshots if snap.symbol == "ETH"), None)
    btc_24h = btc.price_change_24h if btc else median_24h
    eth_24h = eth.price_change_24h if eth else median_24h

    notes: List[str] = []
    if breadth_24h >= 0.68 and median_24h >= 1.8 and btc_24h >= 0:
        label = "RISK_ON"
        score = 78 + min((breadth_24h - 0.68) * 40, 12)
        alt_multiplier = 1.08
        notes.append("broad positive breadth across liquid coins")
    elif breadth_24h <= 0.25 and median_24h <= -3.0 and btc_24h <= -1.5:
        label = "PANIC"
        score = 18 + max(median_24h, -12.0) * 1.5
        alt_multiplier = 0.72
        notes.append("broad market selloff and weak majors")
    elif breadth_24h <= 0.4 and median_24h <= -1.0:
        label = "RISK_OFF"
        score = 34 + breadth_24h * 20
        alt_multiplier = 0.84
        notes.append("market breadth is weak and defensive")
    elif abs(median_24h) <= 1.2 and 0.42 <= breadth_24h <= 0.58:
        label = "RANGE"
        score = 52
        alt_multiplier = 0.94
        notes.append("market looks mixed and range-bound")
    else:
        label = "ROTATION"
        score = 60 + max(min(median_7d, 8.0), -4.0)
        alt_multiplier = 1.0
        notes.append("selective strength rather than broad trend")

    if eth_24h > btc_24h + 1.5:
        notes.append("alts are outperforming majors")
    elif btc_24h > eth_24h + 1.5:
        notes.append("BTC leadership is dominant")

    return {
        "label": label,
        "score": round(max(0.0, min(100.0, score)), 2),
        "breadth_24h": round(breadth_24h, 4),
        "breadth_7d": round(breadth_7d, 4),
        "median_change_24h": round(median_24h, 3),
        "median_change_7d": round(median_7d, 3),
        "alt_multiplier": round(alt_multiplier, 3),
        "notes": notes,
    }


def analyse_deep_context(
    snapshot: MarketSnapshot,
    indicators: IndicatorSet,
    trend_info: Optional[dict],
    history: Optional[Dict[str, List[float]]],
    regime: dict,
) -> dict:
    """
    Build deeper per-coin context from recent price/volume history and current regime.
    """
    prices = list((history or {}).get("prices") or [])
    volumes = list((history or {}).get("volumes") or [])

    if not prices:
        synthetic_7d = (
            snapshot.current_price / (1 + snapshot.price_change_7d / 100)
            if snapshot.price_change_7d not in (-100, 0)
            else snapshot.current_price
        )
        synthetic_1d = (
            snapshot.current_price / (1 + snapshot.price_change_24h / 100)
            if snapshot.price_change_24h not in (-100, 0)
            else snapshot.current_price
        )
        prices = [synthetic_7d, synthetic_1d, snapshot.current_price]

    recent_high = max(prices) if prices else snapshot.high_24h or snapshot.current_price
    recent_low = min(prices) if prices else snapshot.low_24h or snapshot.current_price
    weekly_change = (
        ((prices[-1] - prices[0]) / prices[0] * 100.0)
        if prices and prices[0]
        else snapshot.price_change_7d
    )
    distance_from_high = (
        ((recent_high - snapshot.current_price) / recent_high * 100.0)
        if recent_high
        else 0.0
    )
    rebound_from_low = (
        ((snapshot.current_price - recent_low) / recent_low * 100.0)
        if recent_low
        else 0.0
    )

    liquidity_ratio = (
        snapshot.total_volume / snapshot.market_cap
        if snapshot.market_cap and snapshot.market_cap > 0
        else 0.0
    )
    liquidity_score = min(100.0, liquidity_ratio * 400.0)
    if snapshot.total_volume >= 100_000_000:
        liquidity_score += 15
    elif snapshot.total_volume >= 20_000_000:
        liquidity_score += 8
    elif snapshot.total_volume >= 5_000_000:
        liquidity_score += 4
    liquidity_score = max(0.0, min(100.0, liquidity_score))

    bullish_trend_score = _directional_trend_score(trend_info, FUTURES_LONG)
    bearish_trend_score = _directional_trend_score(trend_info, FUTURES_SHORT)
    ma20 = indicators.ma20 or snapshot.current_price
    ma50 = indicators.ma50 or snapshot.current_price
    rsi = indicators.rsi or 50.0
    above_mas = snapshot.current_price >= ma20 and snapshot.current_price >= ma50
    below_mas = snapshot.current_price <= ma20 and snapshot.current_price <= ma50

    long_extension_penalty = 0.0
    short_extension_penalty = 0.0
    risk_notes: List[str] = []
    if distance_from_high <= 2.5 and rsi > 72:
        long_extension_penalty += 12
        risk_notes.append("price is near recent highs with stretched RSI")
    if snapshot.price_change_24h >= 0.7 * max(weekly_change, 1.0) and snapshot.price_change_24h >= 10:
        long_extension_penalty += 8
        risk_notes.append("24h move already accounts for most of the weekly run")
    if rebound_from_low <= 2.5 and rsi < 30:
        short_extension_penalty += 12
        risk_notes.append("price is near recent lows with already-oversold RSI")
    if (
        snapshot.price_change_24h <= -10
        and abs(snapshot.price_change_24h) >= 0.7 * max(abs(min(weekly_change, 0.0)), 1.0)
    ):
        short_extension_penalty += 8
        risk_notes.append("24h drop already accounts for most of the weekly drawdown")
    if indicators.volatility >= 0.09:
        long_extension_penalty += 6
        short_extension_penalty += 6
        risk_notes.append("volatility is elevated")

    long_continuation = 32.0
    long_continuation += min(max(weekly_change, 0.0), 18.0) * 1.1
    long_continuation += min(max(indicators.momentum, 0.0), 15.0) * 1.2
    long_continuation += min(bullish_trend_score, 60.0) * 0.35
    long_continuation += min(indicators.volume_ratio, 5.0) * 3.2
    if above_mas:
        long_continuation += 8
    if 45 <= rsi <= 68:
        long_continuation += 5
    elif rsi > 78:
        long_continuation -= 10

    short_continuation = 32.0
    short_continuation += min(abs(min(weekly_change, 0.0)), 18.0) * 1.1
    short_continuation += min(abs(min(indicators.momentum, 0.0)), 15.0) * 1.2
    short_continuation += min(bearish_trend_score, 60.0) * 0.35
    short_continuation += min(indicators.volume_ratio, 5.0) * 3.0
    if below_mas:
        short_continuation += 8
    if 34 <= rsi <= 58:
        short_continuation += 5
    elif rsi < 22:
        short_continuation -= 10

    market_regime = regime.get("label", "UNKNOWN")
    if market_regime in {"PANIC", "RISK_OFF"} and snapshot.symbol not in MAJOR_SYMBOLS:
        long_continuation -= 8
        short_continuation += 4
        risk_notes.append("market regime is weak for alt continuation")
    elif market_regime == "RISK_ON":
        long_continuation += 4
        short_continuation -= 5

    long_deep_score = max(
        0.0,
        min(100.0, long_continuation - long_extension_penalty + liquidity_score * 0.12),
    )
    short_deep_score = max(
        0.0,
        min(100.0, short_continuation - short_extension_penalty + liquidity_score * 0.12),
    )
    deep_score = max(long_deep_score, short_deep_score)
    if long_deep_score >= short_deep_score + 5:
        structure_bias = FUTURES_LONG
    elif short_deep_score >= long_deep_score + 5:
        structure_bias = FUTURES_SHORT
    else:
        structure_bias = "MIXED"

    holder_risk_proxy = 20.0
    if snapshot.market_cap and snapshot.market_cap <= 75_000_000 and liquidity_score < 35:
        holder_risk_proxy = 74.0
        risk_notes.append("small-cap token with thin liquidity can be whale-driven")
    elif snapshot.market_cap and snapshot.market_cap <= 250_000_000 and liquidity_score < 45:
        holder_risk_proxy = 58.0
        risk_notes.append("mid-cap liquidity is still concentration-sensitive")
    elif snapshot.market_cap <= 0:
        holder_risk_proxy = 55.0
        risk_notes.append("market-cap data missing, holder concentration unknown")
    elif snapshot.symbol in MAJOR_SYMBOLS:
        holder_risk_proxy = 18.0

    return {
        "deep_score": round(deep_score, 2),
        "long_deep_score": round(long_deep_score, 2),
        "short_deep_score": round(short_deep_score, 2),
        "structure_bias": structure_bias,
        "liquidity_score": round(liquidity_score, 2),
        "risk_score": round(max(holder_risk_proxy, max(long_extension_penalty, short_extension_penalty) * 5.0), 2),
        "distance_from_high_pct": round(distance_from_high, 3),
        "rebound_from_low_pct": round(rebound_from_low, 3),
        "weekly_change_pct": round(weekly_change, 3),
        "extension_penalty": round(max(long_extension_penalty, short_extension_penalty), 2),
        "long_extension_penalty": round(long_extension_penalty, 2),
        "short_extension_penalty": round(short_extension_penalty, 2),
        "liquidity_ratio": round(liquidity_ratio, 5),
        "market_regime": market_regime,
        "market_regime_score": round(regime.get("score", 50.0), 2),
        "risk_notes": risk_notes[:3],
    }


def score_watchlist_candidate(
    snapshot: MarketSnapshot,
    pump: PumpScore,
    rough_score: float,
    deep_context: dict,
    regime: dict,
) -> float:
    """Internal shortlist score before the final publish decision."""
    preferred_direction = FUTURES_SHORT if str(pump.direction).upper() == FUTURES_SHORT else FUTURES_LONG
    score = rough_score * 0.48
    score += _directional_deep_score(deep_context, preferred_direction) * 0.32
    score += deep_context.get("liquidity_score", 0.0) * 0.12
    score += regime.get("score", 50.0) * 0.08

    if preferred_direction == FUTURES_LONG and regime.get("label") in {"PANIC", "RISK_OFF"} and snapshot.symbol not in MAJOR_SYMBOLS:
        score -= 8
    elif preferred_direction == FUTURES_SHORT and regime.get("label") == "RISK_ON" and snapshot.symbol not in MAJOR_SYMBOLS:
        score -= 6
    if deep_context.get("risk_score", 0.0) >= 72:
        score -= 10
    if pump.total_score >= 95:
        score += 3
    if deep_context.get("structure_bias") == preferred_direction:
        score += 4

    return round(max(0.0, min(100.0, score)), 2)


def calculate_final_publish_score(
    snapshot: MarketSnapshot,
    ai_action: str,
    ai_confidence: float,
    quality_score: float,
    prior_final_score: float,
    deep_context: dict,
    regime: dict,
    futures_context: Optional[dict] = None,
) -> float:
    """Final decision score used for the published signal only."""
    preferred_direction = FUTURES_SHORT if str(ai_action).upper() == "SHORT" else FUTURES_LONG
    score = prior_final_score * 0.42
    score += quality_score * 0.18
    score += ai_confidence * 0.12
    score += _directional_deep_score(deep_context, preferred_direction) * 0.18
    score += deep_context.get("liquidity_score", 0.0) * 0.08
    score += regime.get("score", 50.0) * 0.05

    if futures_context and futures_context.get("has_data"):
        score += futures_context.get("futures_score", 0.0) * 0.09
        if _action_matches_bias(ai_action, futures_context.get("trade_bias", "")):
            score += 4
        elif futures_context.get("trade_bias") == FUTURES_WAIT:
            score -= 8
        else:
            score -= 12

    if str(ai_action).upper() not in {"BUY", "SHORT"}:
        score -= 18
    if regime.get("label") == "PANIC" and str(ai_action).upper() == "BUY" and snapshot.symbol not in MAJOR_SYMBOLS:
        score -= 12
    elif regime.get("label") == "RISK_OFF" and str(ai_action).upper() == "BUY" and snapshot.symbol not in MAJOR_SYMBOLS:
        score -= 7
    elif regime.get("label") == "RISK_ON" and str(ai_action).upper() == "SHORT" and snapshot.symbol not in MAJOR_SYMBOLS:
        score -= 7
    elif regime.get("label") in {"PANIC", "RISK_OFF"} and str(ai_action).upper() == "SHORT":
        score += 3

    score -= min(deep_context.get("risk_score", 0.0), 100.0) * 0.08
    if futures_context and abs(futures_context.get("spread", 0.0)) >= 1.5:
        score -= 6
    return round(max(0.0, min(100.0, score)), 2)


def summarize_futures_context(
    snapshot: MarketSnapshot,
    indicators: IndicatorSet,
    trend_info: Optional[dict],
    deep_context: dict,
    regime: dict,
    derivatives: Optional[List[dict]],
) -> dict:
    """
    Summarise CoinDCX futures context for a spot symbol.

    The goal is to keep final suggestions tied to instruments the user can
    actually trade on CoinDCX futures, including a conservative leverage hint
    that never exceeds CoinDCX's published max leverage for that instrument.
    """
    rows = derivatives or []
    if not rows or not snapshot.coindcx_has_futures:
        return {
            "has_data": False,
            "trade_bias": FUTURES_UNAVAILABLE,
            "leverage_hint": "Unavailable",
            "futures_exchange": "CoinDCX",
            "futures_symbol": snapshot.coindcx_futures_instrument or "",
            "funding_rate": 0.0,
            "open_interest": 0.0,
            "basis": 0.0,
            "spread": 0.0,
            "futures_volume_24h": 0.0,
            "futures_score": 0.0,
            "matrix_score_long": 0.0,
            "matrix_score_short": 0.0,
            "confidence_label": "Unavailable",
            "notes": ["no active CoinDCX futures instrument found for this symbol"],
        }

    symbol_key = _normalise_symbol(snapshot.symbol)
    matches = [row for row in rows if _normalise_symbol(row.get("underlying_symbol", "")) == symbol_key]
    if not matches and snapshot.coindcx_futures_instrument:
        matches = [
            row
            for row in rows
            if str(row.get("instrument_name") or "") == snapshot.coindcx_futures_instrument
        ]

    if not matches:
        return {
            "has_data": False,
            "trade_bias": FUTURES_UNAVAILABLE,
            "leverage_hint": "Unavailable",
            "futures_exchange": "CoinDCX",
            "futures_symbol": snapshot.coindcx_futures_instrument or "",
            "funding_rate": 0.0,
            "open_interest": 0.0,
            "basis": 0.0,
            "spread": 0.0,
            "futures_volume_24h": 0.0,
            "futures_score": 0.0,
            "matrix_score_long": 0.0,
            "matrix_score_short": 0.0,
            "confidence_label": "Unavailable",
            "notes": ["CoinDCX futures universe does not include this symbol"],
        }

    preferred = None
    if snapshot.coindcx_futures_instrument:
        preferred = next(
            (
                row for row in matches
                if str(row.get("instrument_name") or "") == snapshot.coindcx_futures_instrument
            ),
            None,
        )
    lead = preferred or matches[0]
    instrument_name = str(lead.get("instrument_name") or snapshot.coindcx_futures_instrument or "")
    details = fetch_coindcx_instrument_details(instrument_name) if instrument_name else {}
    live_prices = fetch_coindcx_futures_prices([instrument_name]) if instrument_name else {}
    price_row = live_prices.get(instrument_name, {})

    futures_price = _safe_float(
        price_row.get("last_price")
        or details.get("last_price")
        or snapshot.current_price
    )
    basis = (
        ((futures_price - snapshot.current_price) / snapshot.current_price) * 100.0
        if snapshot.current_price
        else 0.0
    )
    spread = _safe_float(price_row.get("spread") or details.get("spread"))
    funding_rate = _safe_float(price_row.get("funding_rate") or details.get("funding_rate"))
    open_interest = _safe_float(price_row.get("open_interest") or details.get("open_interest"))
    futures_volume_24h = _safe_float(price_row.get("volume_24h"))
    max_long = _safe_float(details.get("max_leverage_long") or snapshot.coindcx_max_leverage_long or 1.0, 1.0)
    max_short = _safe_float(details.get("max_leverage_short") or snapshot.coindcx_max_leverage_short or max_long, max_long)

    score = 42.0
    notes: List[str] = []
    bullish_trend_score = _directional_trend_score(trend_info, FUTURES_LONG)
    bearish_trend_score = _directional_trend_score(trend_info, FUTURES_SHORT)
    risk_score = _safe_float(deep_context.get("risk_score"), 50.0)
    liquidity_score = _safe_float(deep_context.get("liquidity_score"), 0.0)
    extension_penalty = _safe_float(deep_context.get("extension_penalty"), 0.0)
    rsi = _safe_float(indicators.rsi, 50.0)
    ma20 = _safe_float(indicators.ma20, snapshot.current_price)
    ma50 = _safe_float(indicators.ma50, snapshot.current_price)
    volume_ratio = _safe_float(indicators.volume_ratio, 1.0)
    momentum = _safe_float(indicators.momentum, 0.0)
    bullish_ma_stack = snapshot.current_price >= ma20 >= ma50 if ma20 and ma50 else False
    bearish_ma_stack = snapshot.current_price <= ma20 <= ma50 if ma20 and ma50 else False

    if max_long >= 20:
        score += 8
    elif max_long >= 10:
        score += 5
    elif max_long >= 5:
        score += 3

    spread_abs = abs(spread)
    if spread_abs == 0:
        notes.append("live bid/ask spread not published by CoinDCX for this instrument")
    elif spread_abs <= 0.25:
        score += 8
    elif spread_abs <= 0.75:
        score += 4
    else:
        score -= 8
        notes.append("futures spread is wide")

    basis_abs = abs(basis)
    if basis_abs <= 0.35:
        score += 10
        notes.append("futures price is tracking spot closely")
    elif basis_abs <= 1.0:
        score += 4
    else:
        score -= 6
        notes.append("futures basis is stretched versus spot")

    if futures_volume_24h >= 100_000_000:
        score += 8
    elif futures_volume_24h >= 20_000_000:
        score += 4

    if open_interest >= 100_000_000:
        score += 8
    elif open_interest >= 10_000_000:
        score += 4

    if -0.02 <= funding_rate <= 0.02 and funding_rate != 0:
        score += 4
    elif abs(funding_rate) >= 0.10:
        score -= 6
        notes.append("funding looks crowded")

    regime_label = regime.get("label", "UNKNOWN")
    if regime_label == "RISK_ON" and snapshot.price_change_24h > 0:
        score += 4
    elif regime_label in {"PANIC", "RISK_OFF"} and snapshot.symbol not in MAJOR_SYMBOLS:
        score -= 6

    long_matrix = 0.0
    long_matrix += _clamp((volume_ratio - 1.0) / 3.0, 0.0, 1.0) * 16.0
    long_matrix += _clamp(momentum / 12.0, 0.0, 1.0) * 16.0
    long_matrix += _clamp(bullish_trend_score / 55.0, 0.0, 1.0) * 16.0
    long_matrix += _clamp(liquidity_score / 80.0, 0.0, 1.0) * 8.0
    long_matrix += 10.0 if bullish_ma_stack else 0.0
    long_matrix += 8.0 if 48.0 <= rsi <= 68.0 else 3.0 if 68.0 < rsi <= 74.0 else 0.0
    long_matrix += 8.0 if basis >= -0.25 else 3.0 if basis >= -0.60 else 0.0
    long_matrix += 8.0 if spread_abs <= 0.25 and spread_abs > 0 else 4.0 if spread_abs <= 0.70 else 0.0
    long_matrix += 6.0 if open_interest >= 10_000_000 else 0.0
    long_matrix += 4.0 if futures_volume_24h >= 20_000_000 else 0.0
    long_matrix += 4.0 if regime_label in {"RISK_ON", "ROTATION"} else 0.0
    long_matrix -= 10.0 if rsi > 78.0 else 4.0 if rsi > 72.0 else 0.0
    long_matrix -= _clamp(extension_penalty, 0.0, 12.0) * 0.8
    long_matrix -= _clamp((risk_score - 40.0) / 30.0, 0.0, 1.0) * 12.0
    long_matrix -= 5.0 if funding_rate >= 0.05 else 0.0

    short_matrix = 0.0
    short_matrix += _clamp(abs(min(momentum, 0.0)) / 10.0, 0.0, 1.0) * 16.0
    short_matrix += _clamp(bearish_trend_score / 55.0, 0.0, 1.0) * 16.0
    short_matrix += 12.0 if bearish_ma_stack else 0.0
    short_matrix += 10.0 if basis <= -0.25 else 4.0 if basis < 0 else 0.0
    short_matrix += 8.0 if regime_label in {"PANIC", "RISK_OFF"} else 3.0 if regime_label == "RANGE" else 0.0
    short_matrix += 6.0 if funding_rate >= 0.03 else 0.0
    short_matrix += 6.0 if spread_abs <= 0.35 and spread_abs > 0 else 0.0
    short_matrix += 6.0 if open_interest >= 10_000_000 else 0.0
    short_matrix += 6.0 if snapshot.price_change_24h <= -4.0 else 0.0
    short_matrix -= 8.0 if rsi < 32.0 else 0.0
    short_matrix -= 5.0 if basis_abs > 1.2 else 0.0
    short_matrix -= _clamp(_safe_float(deep_context.get("short_extension_penalty"), extension_penalty), 0.0, 12.0) * 0.7
    short_matrix -= 5.0 if funding_rate <= -0.05 else 0.0

    long_matrix = _clamp(long_matrix + score * 0.35, 0.0, 100.0)
    short_matrix = _clamp(short_matrix + max(0.0, 55.0 - score) * 0.10 + score * 0.12, 0.0, 100.0)

    if long_matrix >= 74.0 and long_matrix >= short_matrix + 8.0:
        trade_bias = FUTURES_LONG
    elif short_matrix >= 74.0 and short_matrix >= long_matrix + 8.0:
        trade_bias = FUTURES_SHORT
    else:
        trade_bias = FUTURES_WAIT

    execution_score = max(long_matrix, short_matrix)

    recommended_cap_long = max(1.0, min(max_long, COINDCX_RECOMMENDED_LEVERAGE_CAP))
    recommended_cap_short = max(1.0, min(max_short, COINDCX_RECOMMENDED_LEVERAGE_CAP))
    if trade_bias == FUTURES_LONG and execution_score >= 88 and risk_score <= 42 and spread_abs <= 0.30 and basis_abs <= 0.35:
        leverage_hint = f"{int(min(3.0, recommended_cap_long))}x (max {int(max_long)}x)"
    elif trade_bias == FUTURES_LONG and execution_score >= 78 and risk_score <= 58:
        leverage_hint = f"{int(min(2.0, recommended_cap_long))}x (max {int(max_long)}x)"
    elif trade_bias == FUTURES_LONG:
        leverage_hint = f"1x (max {int(max_long)}x)"
    elif trade_bias == FUTURES_SHORT and execution_score >= 88 and risk_score <= 42 and spread_abs <= 0.30 and basis_abs <= 0.45:
        leverage_hint = f"{int(min(3.0, recommended_cap_short))}x short (max {int(max_short)}x)"
    elif trade_bias == FUTURES_SHORT and execution_score >= 78 and risk_score <= 58:
        leverage_hint = f"{int(min(2.0, recommended_cap_short))}x short (max {int(max_short)}x)"
    elif trade_bias == FUTURES_SHORT:
        leverage_hint = f"1x short (max {int(max_short)}x)"
    else:
        leverage_hint = "Wait"

    if execution_score >= 88:
        confidence_label = "High"
    elif execution_score >= 78:
        confidence_label = "Medium"
    elif execution_score >= 68:
        confidence_label = "Watch"
    else:
        confidence_label = "Weak"

    return {
        "has_data": True,
        "trade_bias": trade_bias,
        "leverage_hint": leverage_hint,
        "futures_exchange": "CoinDCX",
        "futures_symbol": instrument_name or snapshot.symbol,
        "funding_rate": round(funding_rate, 6),
        "open_interest": round(open_interest, 2),
        "basis": round(basis, 4),
        "spread": round(spread, 4),
        "futures_volume_24h": round(futures_volume_24h, 2),
        "futures_score": round(execution_score, 2),
        "matrix_score_long": round(long_matrix, 2),
        "matrix_score_short": round(short_matrix, 2),
        "confidence_label": confidence_label,
        "notes": notes[:3],
    }
