"""
backtester.py - Tests the signal strategy on historical OHLCV data.

Usage:
    python backtester.py --coins bitcoin,ethereum --days 30
"""

import argparse
import logging
from typing import Dict, List

from config import DEFAULT_STOP_LOSS_PCT, DEFAULT_TARGET_PCT
from indicators import calculate_indicators_from_history
from learning_engine import load_weights
from pump_detector import PumpDetector
from scanner import fetch_coin_history

logger = logging.getLogger(__name__)


def _build_price_windows(rows: List[dict], window: int = 50):
    """Yield (idx, price_window, volume_window) for each data point."""
    prices = [row["close"] for row in rows]
    volumes = [row["volume"] or 0 for row in rows]
    for idx in range(window, len(prices)):
        yield idx, prices[max(0, idx - window): idx + 1], volumes[max(0, idx - window): idx + 1]


def backtest_coin(coin_id: str, days: int = 30) -> Dict:
    """Run the opportunity strategy on historical data for one coin."""
    rows = fetch_coin_history(coin_id, days)
    if len(rows) < 10:
        logger.warning("Not enough history for %s (%d rows).", coin_id, len(rows))
        return {}

    weights = load_weights()
    detector = PumpDetector()
    detector.set_weights(weights)

    signals_generated = 0
    long_signals = 0
    short_signals = 0
    wins = 0
    losses = 0
    returns: List[float] = []

    for idx, price_window, volume_window in _build_price_windows(rows):
        current_price = price_window[-1]
        if current_price <= 0:
            continue

        ind = calculate_indicators_from_history(
            coin_id.upper(),
            price_window,
            volume_window,
        )

        from signals import MarketSnapshot

        snap = MarketSnapshot(
            id=coin_id,
            symbol=coin_id.upper(),
            name=coin_id,
            current_price=current_price,
            market_cap=0,
            total_volume=volume_window[-1] if volume_window else 0,
            price_change_24h=ind.momentum,
            price_change_7d=0,
        )

        ps = detector.score(snap, ind)
        if not ps.is_opportunity:
            continue

        direction = ps.direction or "LONG"
        signals_generated += 1

        if direction == "SHORT":
            short_signals += 1
            target_price = current_price * (1 - DEFAULT_TARGET_PCT)
            stop_loss = current_price * (1 + DEFAULT_STOP_LOSS_PCT)
        else:
            long_signals += 1
            target_price = current_price * (1 + DEFAULT_TARGET_PCT)
            stop_loss = current_price * (1 - DEFAULT_STOP_LOSS_PCT)

        future_prices = [rows[j]["close"] for j in range(idx + 1, min(idx + 15, len(rows)))]
        outcome = "NEUTRAL"
        ret = 0.0

        for future_price in future_prices:
            if future_price is None:
                continue
            if direction == "SHORT":
                if future_price <= target_price:
                    outcome = "WIN"
                    ret = DEFAULT_TARGET_PCT
                    break
                if future_price >= stop_loss:
                    outcome = "LOSS"
                    ret = -DEFAULT_STOP_LOSS_PCT
                    break
            else:
                if future_price >= target_price:
                    outcome = "WIN"
                    ret = DEFAULT_TARGET_PCT
                    break
                if future_price <= stop_loss:
                    outcome = "LOSS"
                    ret = -DEFAULT_STOP_LOSS_PCT
                    break

        if outcome == "WIN":
            wins += 1
        elif outcome == "LOSS":
            losses += 1

        returns.append(ret)

    closed = wins + losses
    win_rate = (wins / closed * 100) if closed else 0.0
    avg_return = (sum(returns) / len(returns) * 100) if returns else 0.0

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for trade_return in returns:
        equity *= (1 + trade_return)
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    result = {
        "coin": coin_id,
        "days": days,
        "signals_generated": signals_generated,
        "long_signals": long_signals,
        "short_signals": short_signals,
        "wins": wins,
        "losses": losses,
        "neutral": signals_generated - closed,
        "win_rate": round(win_rate, 1),
        "avg_return_pct": round(avg_return, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
    }

    print(f"\n{'-' * 50}")
    print(f"  Backtest: {coin_id.upper()} ({days} days)")
    print(f"{'-' * 50}")
    print(f"  Signals generated : {signals_generated}")
    print(f"  Direction split   : {long_signals} long / {short_signals} short")
    print(f"  Win rate          : {win_rate:.1f}% ({wins}W / {losses}L)")
    print(f"  Avg return        : {avg_return:+.2f}%")
    print(f"  Max drawdown      : {max_drawdown * 100:.2f}%")
    print(f"{'-' * 50}")

    return result


def run_backtest(coins: List[str], days: int = 30) -> None:
    all_results = []
    for coin in coins:
        result = backtest_coin(coin.strip().lower(), days)
        if result:
            all_results.append(result)

    if len(all_results) > 1:
        total_signals = sum(result["signals_generated"] for result in all_results)
        total_wins = sum(result["wins"] for result in all_results)
        total_losses = sum(result["losses"] for result in all_results)
        total_longs = sum(result["long_signals"] for result in all_results)
        total_shorts = sum(result["short_signals"] for result in all_results)
        closed = total_wins + total_losses
        overall_win_rate = (total_wins / closed * 100) if closed else 0.0
        avg_return = sum(result["avg_return_pct"] for result in all_results) / len(all_results)

        print(f"\n{'=' * 50}")
        print("  OVERALL BACKTEST SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Coins tested      : {len(all_results)}")
        print(f"  Total signals     : {total_signals}")
        print(f"  Direction split   : {total_longs} long / {total_shorts} short")
        print(f"  Overall win rate  : {overall_win_rate:.1f}%")
        print(f"  Avg return/trade  : {avg_return:+.2f}%")
        print(f"{'=' * 50}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Crypto AI Agent Backtester")
    parser.add_argument(
        "--coins",
        default="bitcoin,ethereum,solana",
        help="Comma-separated CoinGecko coin IDs",
    )
    parser.add_argument("--days", type=int, default=30, help="Lookback days")
    args = parser.parse_args()

    run_backtest([coin.strip() for coin in args.coins.split(",")], args.days)
