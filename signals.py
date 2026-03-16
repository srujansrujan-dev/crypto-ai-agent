"""
signals.py — Shared data models used across all modules.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketSnapshot:
    """Raw data returned by CoinGecko for one coin."""
    id:                  str
    symbol:              str
    name:                str
    current_price:       float
    market_cap:          float
    total_volume:        float
    price_change_24h:    float          # percentage
    price_change_7d:     float = 0.0   # percentage (may be missing)
    high_24h:            float = 0.0
    low_24h:             float = 0.0
    coindcx_spot_market: str = ""
    coindcx_has_futures: bool = False
    coindcx_futures_instrument: str = ""
    coindcx_margin_asset: str = ""
    coindcx_max_leverage_long: float = 0.0
    coindcx_max_leverage_short: float = 0.0


@dataclass
class IndicatorSet:
    """Calculated technical indicators for one coin."""
    symbol:       str
    rsi:          Optional[float] = None
    ma20:         Optional[float] = None
    ma50:         Optional[float] = None
    momentum:     float = 0.0
    volatility:   float = 0.0
    volume_ratio: float = 1.0          # current vol / average vol


@dataclass
class PumpScore:
    """Pump-detection scoring breakdown."""
    symbol:            str
    total_score:       float = 0.0
    volume_spike:      float = 0.0
    price_change:      float = 0.0
    momentum_breakout: float = 0.0
    small_cap:         float = 0.0
    is_opportunity:    bool  = False


@dataclass
class TradingSignal:
    """Full signal object — stored in DB and shown on dashboard."""
    asset_id:        str
    coin:            str
    symbol:          str
    timestamp:       str
    entry_price:     float
    target_price:    float
    stop_loss:       float
    buy_zone_low:    float
    buy_zone_high:   float
    confidence:      float             # 0-100
    ai_action:       str               # BUY / HOLD / AVOID
    ai_reason:       str
    pump_score:      float
    quality_score:   float
    aggregate_score: float
    deep_score:      float
    liquidity_score: float
    risk_score:      float
    market_regime:   str
    regime_score:    float
    trend_score:     float
    volume_ratio:    float
    momentum:        float
    rsi:             float
    market_cap:      float
    price_change_24h: float
    futures_bias:    str = "NO-DATA"
    leverage_hint:   str = "1x"
    futures_exchange: str = ""
    futures_symbol:  str = ""
    funding_rate:    float = 0.0
    open_interest:   float = 0.0
    basis:           float = 0.0
    spread:          float = 0.0
    futures_volume_24h: float = 0.0
    futures_score:   float = 0.0
    outcome:         str = "PENDING"   # PENDING / WIN / LOSS / NEUTRAL
    outcome_price:   Optional[float] = None
    outcome_checked: bool = False

    def to_dict(self) -> dict:
        return self.__dict__
