"""
alerts.py - Signal display and alerting.

Always prints to console.
Telegram is skipped (not configured in this setup).
"""

import logging
from datetime import datetime

from signals import TradingSignal

logger = logging.getLogger(__name__)

RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"


def _color_action(action: str) -> str:
    if action == "BUY":
        return f"{GREEN}{BOLD}{action}{RESET}"
    if action == "AVOID":
        return f"{RED}{BOLD}{action}{RESET}"
    return f"{YELLOW}{BOLD}{action}{RESET}"


def format_signal(signal: TradingSignal) -> str:
    bar = "=" * 55
    futures_line = ""
    if signal.futures_bias != "NO-DATA" or signal.futures_score > 0:
        futures_line = (
            f"  Futures     : {signal.futures_bias}  |  Lev {signal.leverage_hint}  |  "
            f"Funding {signal.funding_rate:.4f}  |  OI ${signal.open_interest:,.0f}\n"
        )

    return (
        f"\n{CYAN}{bar}{RESET}\n"
        f"  {BOLD}SIGNAL DETECTED{RESET}\n"
        f"{CYAN}{bar}{RESET}\n"
        f"  Coin        : {BOLD}{signal.coin} ({signal.symbol}){RESET}\n"
        f"  Action      : {_color_action(signal.ai_action)}\n"
        f"  Current Px  : ${signal.entry_price:>16,.6f}\n"
        f"  Buy Zone    : ${signal.buy_zone_low:,.6f} - ${signal.buy_zone_high:,.6f}\n"
        f"  Target      : ${signal.target_price:>16,.6f}  (+15%)\n"
        f"  Stop Loss   : ${signal.stop_loss:>16,.6f}  (-5%)\n"
        f"  Confidence  : {signal.confidence:.0f}/100\n"
        f"  Pump Score  : {signal.pump_score:.1f}/100\n"
        f"{futures_line}"
        f"  Reason      : {signal.ai_reason}\n"
        f"  Timestamp   : {signal.timestamp}\n"
        f"{CYAN}{bar}{RESET}\n"
    )


def send_signal(signal: TradingSignal) -> None:
    """Print signal to console."""
    print(format_signal(signal))
    logger.info(
        "Signal: %s | %s | confidence=%.0f | score=%.1f",
        signal.symbol,
        signal.ai_action,
        signal.confidence,
        signal.pump_score,
    )


def send_cycle_summary(cycle: int, scanned: int, opportunities: int, signals: int) -> None:
    print(
        f"\n{BOLD}[Cycle {cycle:04d}]{RESET}  "
        f"Scanned: {scanned}  |  "
        f"Opportunities: {opportunities}  |  "
        f"Signals generated: {signals}  |  "
        f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    )


def send_startup_banner() -> None:
    banner = f"""
{CYAN}{'=' * 60}{RESET}
{BOLD}   CRYPTO AI AGENT  -  Market Analysis System{RESET}
{CYAN}{'=' * 60}{RESET}
  DISCLAIMER: This tool provides analysis ONLY.
      It does NOT execute trades. Use at your own risk.
{'-' * 60}
  Data source : CoinGecko (free tier)
  AI engine   : Gemini 1.5 Flash
  Scan every  : 5 minutes
  Dashboard   : http://localhost:8080
{'-' * 60}
"""
    print(banner)
