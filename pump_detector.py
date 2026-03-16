"""
pump_detector.py - Scores coins using weighted opportunity criteria.

Weights live in config.DEFAULT_WEIGHTS but are overridden at runtime by the
learning engine (stored in data/weights.json).
"""

import logging
from typing import List, Tuple

from config import OPPORTUNITY_SCORE_THRESHOLD
from signals import IndicatorSet, MarketSnapshot, PumpScore

logger = logging.getLogger(__name__)

VOLUME_SPIKE_RATIO = 3.0
PRICE_CHANGE_MIN = 5.0
MOMENTUM_BREAKOUT = 8.0
SMALL_CAP_LIMIT = 500_000_000


class PumpDetector:
    def __init__(self, threshold: float = OPPORTUNITY_SCORE_THRESHOLD):
        self.threshold = threshold
        self._weights: dict = {}

    def set_weights(self, weights: dict) -> None:
        self._weights = weights

    def _w(self, key: str, default: float) -> float:
        return self._weights.get(key, default)

    def score(
        self,
        snapshot: MarketSnapshot,
        indicators: IndicatorSet,
    ) -> PumpScore:
        ps = PumpScore(symbol=snapshot.symbol)

        w_vol = self._w("volume_spike", 40)
        ratio = indicators.volume_ratio
        volume_points = 0.0
        if ratio >= VOLUME_SPIKE_RATIO:
            volume_points = w_vol
        elif ratio >= VOLUME_SPIKE_RATIO * 0.6:
            volume_points = w_vol * 0.5

        w_price = self._w("price_change", 25)
        ch = snapshot.price_change_24h
        long_price_points = 0.0
        short_price_points = 0.0
        if ch >= PRICE_CHANGE_MIN:
            long_price_points = w_price
        elif ch >= PRICE_CHANGE_MIN * 0.6:
            long_price_points = w_price * (ch / PRICE_CHANGE_MIN)
        elif ch <= -PRICE_CHANGE_MIN:
            short_price_points = w_price
        elif ch <= -(PRICE_CHANGE_MIN * 0.6):
            short_price_points = w_price * (abs(ch) / PRICE_CHANGE_MIN)

        w_mom = self._w("momentum_breakout", 20)
        mom = indicators.momentum
        long_momentum_points = 0.0
        short_momentum_points = 0.0
        if mom >= MOMENTUM_BREAKOUT:
            long_momentum_points = w_mom
        elif mom >= MOMENTUM_BREAKOUT * 0.5:
            long_momentum_points = w_mom * 0.5
        elif mom <= -MOMENTUM_BREAKOUT:
            short_momentum_points = w_mom
        elif mom <= -(MOMENTUM_BREAKOUT * 0.5):
            short_momentum_points = w_mom * 0.5

        w_sc = self._w("small_cap", 15)
        small_cap_points = 0.0
        if 0 < snapshot.market_cap < SMALL_CAP_LIMIT:
            small_cap_points = w_sc
        elif snapshot.market_cap < SMALL_CAP_LIMIT * 2:
            small_cap_points = w_sc * 0.5

        long_total = volume_points + long_price_points + long_momentum_points + small_cap_points
        short_total = volume_points + short_price_points + short_momentum_points + small_cap_points

        ps.long_score = round(long_total, 2)
        ps.short_score = round(short_total, 2)
        ps.direction = "SHORT" if short_total > long_total else "LONG"
        ps.volume_spike = round(volume_points, 2)
        ps.price_change = round(
            short_price_points if ps.direction == "SHORT" else long_price_points,
            2,
        )
        ps.momentum_breakout = round(
            short_momentum_points if ps.direction == "SHORT" else long_momentum_points,
            2,
        )
        ps.small_cap = round(small_cap_points, 2)
        ps.total_score = round(max(long_total, short_total), 2)
        ps.is_opportunity = ps.total_score >= self.threshold
        return ps

    def scan(
        self,
        snapshots: List[MarketSnapshot],
        indicators: List[IndicatorSet],
    ) -> List[Tuple[MarketSnapshot, IndicatorSet, PumpScore]]:
        """
        Run scoring over all coins and return only flagged opportunities,
        sorted by score descending.
        """
        ind_map = {indicator.symbol: indicator for indicator in indicators}
        results = []

        for snap in snapshots:
            ind = ind_map.get(snap.symbol)
            if ind is None:
                continue
            ps = self.score(snap, ind)
            if ps.is_opportunity:
                results.append((snap, ind, ps))
                logger.debug(
                    "Opportunity: %s dir=%s score=%.1f vol_ratio=%.2fx ch=%.2f%%",
                    snap.symbol,
                    ps.direction,
                    ps.total_score,
                    ind.volume_ratio,
                    snap.price_change_24h,
                )

        results.sort(key=lambda item: item[2].total_score, reverse=True)
        logger.info("Opportunity scan complete - %d opportunities found.", len(results))
        return results
