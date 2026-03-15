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

from signals import IndicatorSet, MarketSnapshot, PumpScore

MAJOR_SYMBOLS = {"BTC", "ETH", "SOL", "BNB", "XRP"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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

    trend_score = trend_info.get("trend_score", 0.0) if trend_info else 0.0
    above_mas = snapshot.current_price >= (indicators.ma20 or snapshot.current_price) and snapshot.current_price >= (
        indicators.ma50 or snapshot.current_price
    )

    extension_penalty = 0.0
    risk_notes: List[str] = []
    if distance_from_high <= 2.5 and (indicators.rsi or 50.0) > 72:
        extension_penalty += 12
        risk_notes.append("price is near recent highs with stretched RSI")
    if snapshot.price_change_24h >= 0.7 * max(weekly_change, 1.0) and snapshot.price_change_24h >= 10:
        extension_penalty += 8
        risk_notes.append("24h move already accounts for most of the weekly run")
    if indicators.volatility >= 0.09:
        extension_penalty += 6
        risk_notes.append("volatility is elevated")

    continuation_score = 32.0
    continuation_score += min(max(weekly_change, 0.0), 18.0) * 1.1
    continuation_score += min(max(indicators.momentum, 0.0), 15.0) * 1.2
    continuation_score += min(trend_score, 60.0) * 0.35
    continuation_score += min(indicators.volume_ratio, 5.0) * 3.2
    if above_mas:
        continuation_score += 8
    if 45 <= (indicators.rsi or 50.0) <= 68:
        continuation_score += 5
    elif (indicators.rsi or 50.0) > 78:
        continuation_score -= 10

    market_regime = regime.get("label", "UNKNOWN")
    if market_regime in {"PANIC", "RISK_OFF"} and snapshot.symbol not in MAJOR_SYMBOLS:
        continuation_score -= 8
        risk_notes.append("market regime is weak for alt continuation")
    elif market_regime == "RISK_ON":
        continuation_score += 4

    deep_score = max(0.0, min(100.0, continuation_score - extension_penalty + liquidity_score * 0.12))

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
        "liquidity_score": round(liquidity_score, 2),
        "risk_score": round(max(holder_risk_proxy, extension_penalty * 5.0), 2),
        "distance_from_high_pct": round(distance_from_high, 3),
        "rebound_from_low_pct": round(rebound_from_low, 3),
        "weekly_change_pct": round(weekly_change, 3),
        "extension_penalty": round(extension_penalty, 2),
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
    score = rough_score * 0.48
    score += deep_context.get("deep_score", 0.0) * 0.32
    score += deep_context.get("liquidity_score", 0.0) * 0.12
    score += regime.get("score", 50.0) * 0.08

    if regime.get("label") in {"PANIC", "RISK_OFF"} and snapshot.symbol not in MAJOR_SYMBOLS:
        score -= 8
    if deep_context.get("risk_score", 0.0) >= 72:
        score -= 10
    if pump.total_score >= 95:
        score += 3

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
    score = prior_final_score * 0.42
    score += quality_score * 0.18
    score += ai_confidence * 0.12
    score += deep_context.get("deep_score", 0.0) * 0.18
    score += deep_context.get("liquidity_score", 0.0) * 0.08
    score += regime.get("score", 50.0) * 0.05

    if futures_context and futures_context.get("has_data"):
        score += futures_context.get("futures_score", 0.0) * 0.09
        if futures_context.get("trade_bias") == "LONG":
            score += 4
        elif futures_context.get("trade_bias") == "SHORT":
            score -= 12
        else:
            score -= 4

    if ai_action != "BUY":
        score -= 18
    if regime.get("label") == "PANIC" and snapshot.symbol not in MAJOR_SYMBOLS:
        score -= 12
    elif regime.get("label") == "RISK_OFF" and snapshot.symbol not in MAJOR_SYMBOLS:
        score -= 7

    score -= min(deep_context.get("risk_score", 0.0), 100.0) * 0.08
    if futures_context and abs(futures_context.get("spread", 0.0)) >= 1.5:
        score -= 6
    return round(max(0.0, min(100.0, score)), 2)


def summarize_futures_context(
    snapshot: MarketSnapshot,
    regime: dict,
    derivatives: Optional[List[dict]],
) -> dict:
    """
    Summarise live derivatives context for a spot symbol.

    This is a real futures-data layer driven by CoinGecko's derivatives feed.
    It does not force a decision when futures coverage is missing; instead it
    returns a neutral NO-DATA state so the rest of the system can degrade
    gracefully.
    """
    rows = derivatives or []
    if not rows:
        return {
            "has_data": False,
            "trade_bias": "NO-DATA",
            "leverage_hint": "1x",
            "futures_exchange": "",
            "futures_symbol": "",
            "funding_rate": 0.0,
            "open_interest": 0.0,
            "basis": 0.0,
            "spread": 0.0,
            "futures_volume_24h": 0.0,
            "futures_score": 0.0,
            "notes": ["no derivatives coverage found"],
        }

    symbol_key = _normalise_symbol(snapshot.symbol)
    matches = []
    for row in rows:
        if row.get("expired_at"):
            continue
        index_id = _normalise_symbol(row.get("index_id", ""))
        derivative_symbol = _normalise_symbol(row.get("symbol", ""))
        if index_id == symbol_key or derivative_symbol.startswith(symbol_key):
            matches.append(row)

    if not matches:
        return {
            "has_data": False,
            "trade_bias": "NO-DATA",
            "leverage_hint": "1x",
            "futures_exchange": "",
            "futures_symbol": "",
            "funding_rate": 0.0,
            "open_interest": 0.0,
            "basis": 0.0,
            "spread": 0.0,
            "futures_volume_24h": 0.0,
            "futures_score": 0.0,
            "notes": ["no active derivatives market found for this symbol"],
        }

    perpetuals = [
        row
        for row in matches
        if "perpetual" in str(row.get("contract_type", "")).lower()
    ]
    if perpetuals:
        matches = perpetuals

    matches.sort(key=lambda row: _safe_float(row.get("volume_24h")), reverse=True)
    top = matches[:5]

    futures_volume_24h = sum(max(_safe_float(row.get("volume_24h")), 0.0) for row in top)
    open_interest = sum(max(_safe_float(row.get("open_interest")), 0.0) for row in top)
    funding_rate = _weighted_average(top, "funding_rate", "volume_24h")
    basis = _weighted_average(top, "basis", "volume_24h")
    spread = _weighted_average(top, "spread", "volume_24h")
    futures_price_change = _weighted_average(top, "price_change_24h", "volume_24h")

    score = 44.0
    notes: List[str] = []

    if futures_volume_24h >= 500_000_000:
        score += 18
        notes.append("strong derivatives volume")
    elif futures_volume_24h >= 100_000_000:
        score += 12
        notes.append("healthy derivatives volume")
    elif futures_volume_24h >= 20_000_000:
        score += 6
        notes.append("moderate derivatives volume")

    if open_interest >= 250_000_000:
        score += 12
        notes.append("deep open interest")
    elif open_interest >= 50_000_000:
        score += 7
    elif open_interest <= 1_000_000:
        score -= 6
        notes.append("open interest is thin")

    spread_abs = abs(spread)
    if spread_abs <= 0.20:
        score += 8
    elif spread_abs <= 0.50:
        score += 4
    elif spread_abs >= 1.50:
        score -= 8
        notes.append("spread is wide")

    if basis >= 0 and snapshot.price_change_24h > 0:
        score += 5
        notes.append("basis supports bullish continuation")
    elif basis < 0 and snapshot.price_change_24h > 0:
        score -= 6
        notes.append("basis is weak versus spot strength")
    elif basis < 0 and snapshot.price_change_24h < 0:
        score += 3

    if -0.02 <= funding_rate <= 0.02:
        score += 6
    elif abs(funding_rate) >= 0.10:
        score -= 8
        notes.append("funding looks crowded")

    regime_label = regime.get("label", "UNKNOWN")
    if regime_label == "RISK_ON" and snapshot.price_change_24h > 0:
        score += 4
    elif regime_label in {"PANIC", "RISK_OFF"} and snapshot.symbol not in MAJOR_SYMBOLS:
        score -= 6

    if snapshot.price_change_24h > 0 and futures_price_change >= 0 and basis >= -0.10 and funding_rate > -0.03:
        trade_bias = "LONG"
    elif snapshot.price_change_24h < 0 and futures_price_change <= 0 and basis <= 0 and funding_rate < 0:
        trade_bias = "SHORT"
    else:
        trade_bias = "NO-TRADE"

    if trade_bias == "LONG" and score >= 78 and spread_abs <= 0.30:
        leverage_hint = "3x"
    elif trade_bias == "LONG" and score >= 62:
        leverage_hint = "2x"
    else:
        leverage_hint = "1x"

    lead = top[0]
    return {
        "has_data": True,
        "trade_bias": trade_bias,
        "leverage_hint": leverage_hint,
        "futures_exchange": str(lead.get("market") or ""),
        "futures_symbol": str(lead.get("symbol") or snapshot.symbol),
        "funding_rate": round(funding_rate, 6),
        "open_interest": round(open_interest, 2),
        "basis": round(basis, 4),
        "spread": round(spread, 4),
        "futures_volume_24h": round(futures_volume_24h, 2),
        "futures_score": round(max(0.0, min(100.0, score)), 2),
        "notes": notes[:3],
    }
