"""
backtester.py — Tests the signal strategy on historical OHLCV data.

Usage:
    python backtester.py --coins bitcoin,ethereum --days 30
"""

import argparse
import logging
import sys
from typing import List, Dict

from scanner import fetch_coin_history
from indicators import calculate_indicators_from_history
from pump_detector import PumpDetector
from learning_engine import load_weights
from config import DEFAULT_STOP_LOSS_PCT, DEFAULT_TARGET_PCT

logger = logging.getLogger(__name__)


def _build_price_windows(rows: List[dict], window: int = 50):
    """Yield (idx, price_window, volume_window) for each data point."""
    prices  = [r["close"]  for r in rows]
    volumes = [r["volume"] or 0 for r in rows]
    for i in range(window, len(prices)):
        yield i, prices[max(0, i - window): i + 1], volumes[max(0, i - window): i + 1]


def backtest_coin(coin_id: str, days: int = 30) -> Dict:
    """Run the pump-detection strategy on historical data for one coin."""
    rows = fetch_coin_history(coin_id, days)
    if len(rows) < 10:
        logger.warning("Not enough history for %s (%d rows).", coin_id, len(rows))
        return {}

    weights  = load_weights()
    detector = PumpDetector()
    detector.set_weights(weights)

    signals_generated = 0
    wins  = 0
    losses = 0
    returns: List[float] = []

    for i, price_win, vol_win in _build_price_windows(rows):
        current_price = price_win[-1]
        if current_price <= 0:
            continue

        ind = calculate_indicators_from_history(
            coin_id.upper(), price_win, vol_win
        )

        # Build a mock snapshot
        from signals import MarketSnapshot
        snap = MarketSnapshot(
            id=coin_id, symbol=coin_id.upper(), name=coin_id,
            current_price    = current_price,
            market_cap       = 0,
            total_volume     = vol_win[-1] if vol_win else 0,
            price_change_24h = ind.momentum,
            price_change_7d  = 0,
        )

        ps = detector.score(snap, ind)
        if not ps.is_opportunity:
            continue

        signals_generated += 1
        target_price = current_price * (1 + DEFAULT_TARGET_PCT)
        stop_loss    = current_price * (1 - DEFAULT_STOP_LOSS_PCT)

        # Look ahead: did price hit target or stop-loss first?
        future_prices = [rows[j]["close"] for j in range(i + 1, min(i + 15, len(rows)))]
        outcome = "NEUTRAL"
        ret     = 0.0

        for fp in future_prices:
            if fp is None:
                continue
            if fp >= target_price:
                outcome = "WIN"
                ret     = DEFAULT_TARGET_PCT
                break
            if fp <= stop_loss:
                outcome = "LOSS"
                ret     = -DEFAULT_STOP_LOSS_PCT
                break

        if outcome == "WIN":
            wins += 1
        elif outcome == "LOSS":
            losses += 1

        returns.append(ret)

    closed   = wins + losses
    win_rate = (wins / closed * 100) if closed else 0.0
    avg_ret  = (sum(returns) / len(returns) * 100) if returns else 0.0

    # Max drawdown
    equity = 1.0
    peak   = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1 + r)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd

    result = {
        "coin":               coin_id,
        "days":               days,
        "signals_generated":  signals_generated,
        "wins":               wins,
        "losses":             losses,
        "neutral":            signals_generated - closed,
        "win_rate":           round(win_rate, 1),
        "avg_return_pct":     round(avg_ret, 2),
        "max_drawdown_pct":   round(max_dd * 100, 2),
    }

    print(f"\n{'─'*50}")
    print(f"  Backtest: {coin_id.upper()}  ({days} days)")
    print(f"{'─'*50}")
    print(f"  Signals generated : {signals_generated}")
    print(f"  Win rate          : {win_rate:.1f}%  ({wins}W / {losses}L)")
    print(f"  Avg return        : {avg_ret:+.2f}%")
    print(f"  Max drawdown      : {max_dd * 100:.2f}%")
    print(f"{'─'*50}")

    return result


def run_backtest(coins: List[str], days: int = 30) -> None:
    all_results = []
    for coin in coins:
        r = backtest_coin(coin.strip().lower(), days)
        if r:
            all_results.append(r)

    if len(all_results) > 1:
        total_signals = sum(r["signals_generated"] for r in all_results)
        total_wins    = sum(r["wins"]   for r in all_results)
        total_losses  = sum(r["losses"] for r in all_results)
        closed        = total_wins + total_losses
        overall_wr    = (total_wins / closed * 100) if closed else 0.0
        avg_ret       = sum(r["avg_return_pct"] for r in all_results) / len(all_results)

        print(f"\n{'═'*50}")
        print("  OVERALL BACKTEST SUMMARY")
        print(f"{'═'*50}")
        print(f"  Coins tested      : {len(all_results)}")
        print(f"  Total signals     : {total_signals}")
        print(f"  Overall win rate  : {overall_wr:.1f}%")
        print(f"  Avg return/trade  : {avg_ret:+.2f}%")
        print(f"{'═'*50}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Crypto AI Agent Backtester")
    parser.add_argument(
        "--coins", default="bitcoin,ethereum,solana",
        help="Comma-separated CoinGecko coin IDs"
    )
    parser.add_argument("--days", type=int, default=30, help="Lookback days")
    args = parser.parse_args()

    coins = [c.strip() for c in args.coins.split(",")]
    run_backtest(coins, args.days)
