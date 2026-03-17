"""
alerts.py - Signal display and alerting.

Always prints to console.
Telegram is skipped (not configured in this setup).
"""

import logging
from datetime import datetime

from config import DEFAULT_STOP_LOSS_PCT, DEFAULT_TARGET_PCT
from config import get_dashboard_display_url
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
    if action == "SHORT":
        return f"{RED}{BOLD}{action}{RESET}"
    if action == "AVOID":
        return f"{RED}{BOLD}{action}{RESET}"
    return f"{YELLOW}{BOLD}{action}{RESET}"


def format_signal(signal: TradingSignal) -> str:
    bar = "=" * 55
    is_short = signal.is_short
    target_pct = -DEFAULT_TARGET_PCT * 100 if is_short else DEFAULT_TARGET_PCT * 100
    stop_pct = DEFAULT_STOP_LOSS_PCT * 100 if is_short else -DEFAULT_STOP_LOSS_PCT * 100
    futures_line = ""
    if signal.futures_bias != "UNAVAILABLE" or signal.futures_score > 0:
        futures_line = (
            f"  CoinDCX Fut : {signal.futures_bias}  |  Lev {signal.leverage_hint}  |  "
            f"Funding {signal.funding_rate:.4f}  |  OI ${signal.open_interest:,.0f}\n"
        )

    return (
        f"\n{CYAN}{bar}{RESET}\n"
        f"  {BOLD}SIGNAL DETECTED{RESET}\n"
        f"{CYAN}{bar}{RESET}\n"
        f"  Coin        : {BOLD}{signal.coin} ({signal.symbol}){RESET}\n"
        f"  Action      : {_color_action(signal.ai_action)}\n"
        f"  Current Px  : ${signal.entry_price:>16,.6f}\n"
        f"  Entry Zone  : ${signal.buy_zone_low:,.6f} - ${signal.buy_zone_high:,.6f}\n"
        f"  Target      : ${signal.target_price:>16,.6f}  ({target_pct:+.0f}%)\n"
        f"  Stop Loss   : ${signal.stop_loss:>16,.6f}  ({stop_pct:+.0f}%)\n"
        f"  Confidence  : {signal.confidence:.0f}/100\n"
        f"  Opportunity : {signal.pump_score:.1f}/100\n"
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
    dashboard_url = get_dashboard_display_url()
    banner = f"""
{CYAN}{'=' * 60}{RESET}
{BOLD}   CRYPTO AI AGENT  -  Market Analysis System{RESET}
{CYAN}{'=' * 60}{RESET}
  DISCLAIMER: This tool provides analysis ONLY.
      It does NOT execute trades. Use at your own risk.
{'-' * 60}
  Data source : CoinGecko + CoinDCX public APIs
  Universe    : CoinDCX-listed assets, filtered to CoinDCX futures when enabled
  Signal mode : Directional analysis only (LONG and SHORT suggestions)
  AI engine   : Gemini
  Scan every  : 5 minutes
  Dashboard   : {dashboard_url}
{'-' * 60}
"""
    print(banner)
