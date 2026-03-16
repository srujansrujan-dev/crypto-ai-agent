"""
trend_detector.py - Detects early trends across multiple scan cycles.

Stores a rolling window of snapshots per coin and identifies:
- Increasing volume trend
- Accelerating price momentum
- Momentum direction change
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from signals import MarketSnapshot

logger = logging.getLogger(__name__)

WINDOW_SIZE = 6


class TrendDetector:
    def __init__(self, window: int = WINDOW_SIZE):
        self.window = window
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, snapshots: List[MarketSnapshot]) -> None:
        """Feed the latest cycle's snapshots into the history buffer."""
        for snap in snapshots:
            self._history[snap.id].append(snap)

    def _volume_trend(self, snaps: List[MarketSnapshot]) -> float:
        if len(snaps) < 2:
            return 0.0
        vols = [snap.total_volume for snap in snaps]
        increasing = sum(1 for idx in range(1, len(vols)) if vols[idx] > vols[idx - 1])
        return increasing / (len(vols) - 1)

    def _price_acceleration(self, snaps: List[MarketSnapshot]) -> float:
        if len(snaps) < 3:
            return 0.0
        prices = [snap.current_price for snap in snaps]
        changes = [
            (prices[idx] - prices[idx - 1]) / prices[idx - 1] * 100
            for idx in range(1, len(prices))
            if prices[idx - 1]
        ]
        if len(changes) < 2:
            return 0.0
        avg_prev = sum(changes[:-1]) / len(changes[:-1])
        return changes[-1] - avg_prev

    def _momentum_change(self, snaps: List[MarketSnapshot]) -> float:
        if len(snaps) < 2:
            return 0.0
        return snaps[-1].price_change_24h - snaps[0].price_change_24h

    def evaluate(self, coin_id: str) -> Optional[dict]:
        """Return a trend summary for a coin, or None if not enough history."""
        history = list(self._history.get(coin_id, []))
        if len(history) < 2:
            return None

        vol_trend = self._volume_trend(history)
        price_accel = self._price_acceleration(history)
        mom_change = self._momentum_change(history)

        bullish_score = (
            vol_trend * 40
            + min(max(price_accel / 2, 0), 1) * 35
            + min(max(mom_change / 10, 0), 1) * 25
        )
        bearish_score = (
            vol_trend * 40
            + min(max((-price_accel) / 2, 0), 1) * 35
            + min(max((-mom_change) / 10, 0), 1) * 25
        )
        trend_score = max(bullish_score, bearish_score)

        if bullish_score >= bearish_score + 5:
            trend_bias = "UP"
        elif bearish_score >= bullish_score + 5:
            trend_bias = "DOWN"
        else:
            trend_bias = "FLAT"

        return {
            "coin_id": coin_id,
            "volume_trend": round(vol_trend, 4),
            "price_accel": round(price_accel, 4),
            "momentum_delta": round(mom_change, 4),
            "bullish_trend_score": round(bullish_score, 2),
            "bearish_trend_score": round(bearish_score, 2),
            "trend_score": round(trend_score, 2),
            "trend_bias": trend_bias,
            "is_early_trend": trend_score >= 50 and vol_trend >= 0.6,
        }

    def get_trending_coins(
        self,
        snapshots: List[MarketSnapshot],
    ) -> List[Tuple[MarketSnapshot, dict]]:
        """
        Return (snapshot, trend_info) pairs for coins showing early trends.
        Sorted by trend_score descending.
        """
        snap_map = {snap.id: snap for snap in snapshots}
        results = []

        for coin_id, snap in snap_map.items():
            trend = self.evaluate(coin_id)
            if trend and trend["is_early_trend"]:
                results.append((snap, trend))

        results.sort(key=lambda item: item[1]["trend_score"], reverse=True)
        if results:
            logger.info("Trend detection - %d early trends detected.", len(results))
        return results
