"""
indicators.py — Technical indicator calculations.

We build indicators from the market snapshot data available from CoinGecko
(current price, 24h change, 7d change, volume, high, low).
For multi-period indicators (RSI, MA20, MA50) we approximate using the
available data when full OHLCV history is not loaded, and compute precisely
when called from the backtester with full history.
"""

import math
import logging
from typing import List, Optional

from config import RSI_PERIOD, MA_SHORT, MA_LONG
from signals import MarketSnapshot, IndicatorSet

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rsi(prices: List[float], period: int = RSI_PERIOD) -> Optional[float]:
    """Standard Wilder RSI from a list of closing prices."""
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _sma(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period:
        return None
    return round(sum(prices[-period:]) / period, 8)


def _volatility(prices: List[float]) -> float:
    """Standard deviation of returns (annualised approx)."""
    if len(prices) < 2:
        return 0.0
    returns = [(prices[i] - prices[i - 1]) / prices[i - 1]
               for i in range(1, len(prices)) if prices[i - 1] != 0]
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    return round(math.sqrt(variance), 6)


# ── Main public function ───────────────────────────────────────────────────────

def calculate_indicators(
    snapshot: MarketSnapshot,
    price_history: Optional[List[float]] = None,
    volume_history: Optional[List[float]] = None,
) -> IndicatorSet:
    """
    Calculate indicators for a single coin.

    If price_history / volume_history are provided (from the backtester or a
    multi-cycle buffer) we use them for precise RSI / MA.
    Otherwise we estimate from snapshot fields.
    """
    ind = IndicatorSet(symbol=snapshot.symbol)

    # ── Approximate prices from snapshot when no history is available ──────
    if price_history and len(price_history) >= 2:
        prices = price_history
    else:
        # Build a synthetic mini-history: reconstruct ~3 price points
        p_now  = snapshot.current_price
        p_1d   = p_now / (1 + snapshot.price_change_24h / 100) if snapshot.price_change_24h != -100 else p_now
        p_7d   = p_now / (1 + snapshot.price_change_7d  / 100) if snapshot.price_change_7d  != -100 else p_now
        prices = [p_7d, p_1d, p_now]

    # ── RSI ────────────────────────────────────────────────────────────────
    ind.rsi = _rsi(prices)
    if ind.rsi is None:
        # Heuristic estimate when history is too short
        ch = snapshot.price_change_24h
        ind.rsi = max(0, min(100, 50 + ch * 2))

    # ── Moving averages ────────────────────────────────────────────────────
    ind.ma20 = _sma(prices, MA_SHORT) or snapshot.current_price
    ind.ma50 = _sma(prices, MA_LONG)  or snapshot.current_price

    # ── Momentum (% change over available window) ──────────────────────────
    if len(prices) >= 2 and prices[0] != 0:
        ind.momentum = round((prices[-1] - prices[0]) / prices[0] * 100, 4)
    else:
        ind.momentum = snapshot.price_change_24h

    # ── Volatility ─────────────────────────────────────────────────────────
    ind.volatility = _volatility(prices)

    # ── Volume ratio (current / average across history) ────────────────────
    if volume_history and len(volume_history) >= 3:
        avg_vol = sum(volume_history[:-1]) / (len(volume_history) - 1)
        ind.volume_ratio = round(snapshot.total_volume / avg_vol, 4) if avg_vol else 1.0
    else:
        # Estimate: high volume if volume > 5% of market cap (rough heuristic)
        if snapshot.market_cap and snapshot.market_cap > 0:
            ind.volume_ratio = round(snapshot.total_volume / (snapshot.market_cap * 0.05), 4)
        else:
            ind.volume_ratio = 1.0

    return ind


def calculate_indicators_from_history(
    symbol: str,
    prices: List[float],
    volumes: List[float],
) -> IndicatorSet:
    """Used by the backtester — full history available."""
    snap = MarketSnapshot(
        id=symbol, symbol=symbol, name=symbol,
        current_price   = prices[-1] if prices else 0,
        market_cap      = 0,
        total_volume    = volumes[-1] if volumes else 0,
        price_change_24h= ((prices[-1] - prices[-2]) / prices[-2] * 100)
                          if len(prices) >= 2 and prices[-2] else 0,
        price_change_7d = ((prices[-1] - prices[-8]) / prices[-8] * 100)
                          if len(prices) >= 8 and prices[-8] else 0,
    )
    return calculate_indicators(snap, prices, volumes)
