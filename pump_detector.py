"""
pump_detector.py — Scores coins using weighted pump-detection criteria.

Weights live in config.DEFAULT_WEIGHTS but are overridden at runtime by the
learning engine (stored in data/weights.json).
"""

import logging
from typing import List, Tuple

from config import OPPORTUNITY_SCORE_THRESHOLD
from signals import MarketSnapshot, IndicatorSet, PumpScore

logger = logging.getLogger(__name__)

# Thresholds
VOLUME_SPIKE_RATIO  = 3.0    # volume_ratio >= this  → full volume_spike score
PRICE_CHANGE_MIN    = 5.0    # 24h % change >= this  → full price_change score
MOMENTUM_BREAKOUT   = 8.0    # momentum  >= this     → full momentum score
SMALL_CAP_LIMIT     = 500_000_000  # market cap < $500M → full small_cap score


class PumpDetector:
    def __init__(self, threshold: float = OPPORTUNITY_SCORE_THRESHOLD):
        self.threshold = threshold
        self._weights: dict = {}   # loaded from learning engine

    def set_weights(self, weights: dict) -> None:
        self._weights = weights

    def _w(self, key: str, default: float) -> float:
        return self._weights.get(key, default)

    def score(
        self,
        snapshot:   MarketSnapshot,
        indicators: IndicatorSet,
    ) -> PumpScore:
        ps = PumpScore(symbol=snapshot.symbol)

        # ── 1. Volume spike ────────────────────────────────────────────────
        w_vol = self._w("volume_spike", 40)
        ratio = indicators.volume_ratio
        if ratio >= VOLUME_SPIKE_RATIO:
            ps.volume_spike = w_vol
        elif ratio >= VOLUME_SPIKE_RATIO * 0.6:
            ps.volume_spike = w_vol * 0.5   # partial credit

        # ── 2. Price change ────────────────────────────────────────────────
        w_price = self._w("price_change", 25)
        ch = snapshot.price_change_24h
        if ch >= PRICE_CHANGE_MIN:
            ps.price_change = w_price
        elif ch >= PRICE_CHANGE_MIN * 0.6:
            ps.price_change = w_price * (ch / PRICE_CHANGE_MIN)

        # ── 3. Momentum breakout ───────────────────────────────────────────
        w_mom = self._w("momentum_breakout", 20)
        mom = indicators.momentum
        if mom >= MOMENTUM_BREAKOUT:
            ps.momentum_breakout = w_mom
        elif mom >= MOMENTUM_BREAKOUT * 0.5:
            ps.momentum_breakout = w_mom * 0.5

        # ── 4. Small cap (<$500M) ──────────────────────────────────────────
        w_sc = self._w("small_cap", 15)
        if 0 < snapshot.market_cap < SMALL_CAP_LIMIT:
            ps.small_cap = w_sc
        elif snapshot.market_cap < SMALL_CAP_LIMIT * 2:
            ps.small_cap = w_sc * 0.5

        ps.total_score  = round(
            ps.volume_spike + ps.price_change + ps.momentum_breakout + ps.small_cap, 2
        )
        ps.is_opportunity = ps.total_score >= self.threshold

        return ps

    def scan(
        self,
        snapshots:   List[MarketSnapshot],
        indicators:  List[IndicatorSet],
    ) -> List[Tuple[MarketSnapshot, IndicatorSet, PumpScore]]:
        """
        Run scoring over all coins and return only flagged opportunities,
        sorted by score descending.
        """
        ind_map = {i.symbol: i for i in indicators}
        results = []

        for snap in snapshots:
            ind = ind_map.get(snap.symbol)
            if ind is None:
                continue
            ps = self.score(snap, ind)
            if ps.is_opportunity:
                results.append((snap, ind, ps))
                logger.debug(
                    "Opportunity: %s  score=%.1f  vol_ratio=%.2fx  ch=%.2f%%",
                    snap.symbol, ps.total_score, ind.volume_ratio, snap.price_change_24h,
                )

        results.sort(key=lambda x: x[2].total_score, reverse=True)
        logger.info("Pump scan complete — %d opportunities found.", len(results))
        return results
