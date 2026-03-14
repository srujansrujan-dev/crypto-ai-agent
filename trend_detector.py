"""
trend_detector.py — Detects early trends across multiple scan cycles.

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

WINDOW_SIZE = 6   # keep last N cycles (~30 minutes at 5-min intervals)


class TrendDetector:
    def __init__(self, window: int = WINDOW_SIZE):
        self.window = window
        # coin_id → deque of MarketSnapshot (most recent last)
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, snapshots: List[MarketSnapshot]) -> None:
        """Feed the latest cycle's snapshots into the history buffer."""
        for snap in snapshots:
            self._history[snap.id].append(snap)

    def _volume_trend(self, snaps: List[MarketSnapshot]) -> float:
        """
        Returns positive score if volume is consistently increasing.
        Score range: 0 → 1.0
        """
        if len(snaps) < 2:
            return 0.0
        vols = [s.total_volume for s in snaps]
        increasing = sum(1 for i in range(1, len(vols)) if vols[i] > vols[i - 1])
        return increasing / (len(vols) - 1)

    def _price_acceleration(self, snaps: List[MarketSnapshot]) -> float:
        """
        Measures whether price changes are accelerating (each step > previous).
        Returns positive value if accelerating upward, negative if downward.
        """
        if len(snaps) < 3:
            return 0.0
        prices = [s.current_price for s in snaps]
        changes = [(prices[i] - prices[i - 1]) / prices[i - 1] * 100
                   for i in range(1, len(prices)) if prices[i - 1]]
        if len(changes) < 2:
            return 0.0
        # How much is the latest change bigger than the average of previous changes?
        avg_prev = sum(changes[:-1]) / len(changes[:-1])
        return changes[-1] - avg_prev

    def _momentum_change(self, snaps: List[MarketSnapshot]) -> float:
        """Difference between latest 24h % change and the earliest in the window."""
        if len(snaps) < 2:
            return 0.0
        return snaps[-1].price_change_24h - snaps[0].price_change_24h

    def evaluate(self, coin_id: str) -> Optional[dict]:
        """
        Return a trend summary for a coin, or None if not enough history.
        """
        history = list(self._history.get(coin_id, []))
        if len(history) < 2:
            return None

        vol_trend    = self._volume_trend(history)
        price_accel  = self._price_acceleration(history)
        mom_change   = self._momentum_change(history)

        # Composite trend score (0–100)
        trend_score = (
            vol_trend    * 40 +
            min(max(price_accel / 2, 0), 1) * 35 +
            min(max(mom_change   / 10, 0), 1) * 25
        )

        return {
            "coin_id":        coin_id,
            "volume_trend":   round(vol_trend,   4),
            "price_accel":    round(price_accel, 4),
            "momentum_delta": round(mom_change,  4),
            "trend_score":    round(trend_score, 2),
            "is_early_trend": trend_score >= 50 and vol_trend >= 0.6,
        }

    def get_trending_coins(
        self, snapshots: List[MarketSnapshot]
    ) -> List[Tuple[MarketSnapshot, dict]]:
        """
        Return (snapshot, trend_info) pairs for coins showing early trends.
        Sorted by trend_score descending.
        """
        snap_map = {s.id: s for s in snapshots}
        results = []

        for coin_id, snap in snap_map.items():
            trend = self.evaluate(coin_id)
            if trend and trend["is_early_trend"]:
                results.append((snap, trend))

        results.sort(key=lambda x: x[1]["trend_score"], reverse=True)
        if results:
            logger.info("Trend detection — %d early trends detected.", len(results))
        return results
