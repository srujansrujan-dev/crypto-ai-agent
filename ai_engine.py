"""
ai_engine.py - Sends signal data to Gemini and parses its evaluation.

Returns: action (BUY/SHORT/HOLD/AVOID), confidence (0-100), reason (string).
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import google.generativeai as genai  # type: ignore

from config import (
    AI_MODEL_COOLDOWN_SECONDS,
    AI_QUOTA_COOLDOWN_SECONDS,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)
from signals import IndicatorSet, MarketSnapshot, PumpScore

logger = logging.getLogger(__name__)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set - AI analysis will use fallback heuristics.")

_MODEL = None
_AI_DISABLED_UNTIL: Optional[datetime] = None
_AI_DISABLE_REASON = ""


def _get_model():
    global _MODEL
    if _MODEL is None and GEMINI_API_KEY:
        _MODEL = genai.GenerativeModel(GEMINI_MODEL)
    return _MODEL


def _disable_ai(seconds: int, reason: str) -> None:
    global _AI_DISABLED_UNTIL, _AI_DISABLE_REASON
    _AI_DISABLED_UNTIL = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    _AI_DISABLE_REASON = reason
    logger.warning(
        "AI disabled until %s UTC: %s",
        _AI_DISABLED_UNTIL.isoformat(timespec="seconds"),
        reason,
    )


def _ai_disabled_reason() -> Optional[str]:
    if _AI_DISABLED_UNTIL is None:
        return None
    if datetime.now(timezone.utc) >= _AI_DISABLED_UNTIL:
        return None
    return _AI_DISABLE_REASON or "AI temporarily disabled."


def _build_prompt(
    snapshot: MarketSnapshot,
    indicators: IndicatorSet,
    pump: PumpScore,
    trend_info: Optional[dict] = None,
    futures_context: Optional[dict] = None,
) -> str:
    rsi = float(indicators.rsi or 50.0)
    ma20 = float(indicators.ma20 or snapshot.current_price)
    ma50 = float(indicators.ma50 or snapshot.current_price)
    bullish_trend = (trend_info or {}).get("bullish_trend_score", (trend_info or {}).get("trend_score", 0.0))
    bearish_trend = (trend_info or {}).get("bearish_trend_score", (trend_info or {}).get("trend_score", 0.0))
    trend_bias = (trend_info or {}).get("trend_bias", "FLAT")

    trend_block = ""
    if trend_info:
        trend_block = f"""
Trend Detection (last 6 cycles):
  Volume trend score  : {trend_info.get('volume_trend', 0):.2f}
  Price acceleration  : {trend_info.get('price_accel', 0):+.2f}%
  Momentum delta      : {trend_info.get('momentum_delta', 0):+.2f}%
  Bullish trend score : {bullish_trend:.1f}/100
  Bearish trend score : {bearish_trend:.1f}/100
  Trend bias          : {trend_bias}
"""

    futures_block = ""
    if futures_context and futures_context.get("has_data"):
        futures_block = f"""
=== COINDCX FUTURES CONTEXT ===
Instrument      : {futures_context.get('futures_symbol', snapshot.symbol)}
Exchange        : {futures_context.get('futures_exchange', 'CoinDCX')}
Trade Bias      : {futures_context.get('trade_bias', 'UNAVAILABLE')}
Leverage Hint   : {futures_context.get('leverage_hint', 'Unavailable')}
Funding Rate    : {futures_context.get('funding_rate', 0.0):.6f}
Open Interest   : {futures_context.get('open_interest', 0.0):,.0f}
Basis vs Spot   : {futures_context.get('basis', 0.0):+.2f}%
Execution Score : {futures_context.get('futures_score', 0.0):.1f}/100
Long Matrix     : {futures_context.get('matrix_score_long', 0.0):.1f}/100
Short Matrix    : {futures_context.get('matrix_score_short', 0.0):.1f}/100
"""

    return f"""
You are a professional cryptocurrency market analyst. Evaluate the following opportunity signal
and provide a structured JSON response. This system NEVER executes trades. It only suggests ideas.

=== MARKET DATA ===
Coin         : {snapshot.name} ({snapshot.symbol})
Price (USD)  : ${snapshot.current_price:,.6f}
Market Cap   : ${snapshot.market_cap:,.0f}
24h Volume   : ${snapshot.total_volume:,.0f}
24h Change   : {snapshot.price_change_24h:+.2f}%
7d Change    : {snapshot.price_change_7d:+.2f}%
High 24h     : ${snapshot.high_24h:,.6f}
Low 24h      : ${snapshot.low_24h:,.6f}

=== TECHNICAL INDICATORS ===
RSI          : {rsi:.1f}
MA20         : ${ma20:,.6f}
MA50         : ${ma50:,.6f}
Momentum     : {indicators.momentum:+.2f}%
Volatility   : {indicators.volatility:.4f}
Volume Ratio : {indicators.volume_ratio:.2f}x average

=== OPPORTUNITY SCORE ===
Selected bias : {pump.direction}
Total Score   : {pump.total_score:.1f} / 100
Long Score    : {pump.long_score:.1f}
Short Score   : {pump.short_score:.1f}
  Volume spike       : {pump.volume_spike}
  Price move         : {pump.price_change}
  Momentum breakout  : {pump.momentum_breakout}
  Small cap bonus    : {pump.small_cap}
{trend_block}
{futures_block}

=== YOUR TASK ===
Decide whether this is best treated as:
1. BUY for a long setup
2. SHORT for a bearish setup
3. HOLD if the setup is mixed and needs confirmation
4. AVOID if the setup is poor or too risky

Consider:
- RSI stretch versus continuation potential
- Price position versus MA20 and MA50
- Whether the move looks extended already
- Whether the CoinDCX futures matrix supports LONG or SHORT
- Risk/reward, liquidity, and crowding risk

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "action":     "BUY" | "SHORT" | "HOLD" | "AVOID",
  "confidence": <integer 0-100>,
  "reason":     "<2-3 sentence explanation>",
  "risk_level": "LOW" | "MEDIUM" | "HIGH"
}}
""".strip()


def _smart_fallback_analysis(
    snapshot: MarketSnapshot,
    indicators: IndicatorSet,
    pump: PumpScore,
    trend_info: Optional[dict] = None,
    futures_context: Optional[dict] = None,
) -> Tuple[str, float, str]:
    """Rule-based fallback when Gemini is unavailable."""
    rsi = indicators.rsi or 50.0
    volume_ratio = indicators.volume_ratio
    momentum = indicators.momentum
    bullish_trend = float((trend_info or {}).get("bullish_trend_score", (trend_info or {}).get("trend_score", 0.0)) or 0.0)
    bearish_trend = float((trend_info or {}).get("bearish_trend_score", (trend_info or {}).get("trend_score", 0.0)) or 0.0)
    ma20 = indicators.ma20 or snapshot.current_price
    ma50 = indicators.ma50 or snapshot.current_price
    above_mas = snapshot.current_price >= ma20 and snapshot.current_price >= ma50
    below_mas = snapshot.current_price <= ma20 and snapshot.current_price <= ma50
    futures_bias = (futures_context or {}).get("trade_bias", "UNAVAILABLE")
    futures_score = float((futures_context or {}).get("futures_score", 0.0) or 0.0)

    long_score = 18.0
    short_score = 18.0
    long_strengths = []
    short_strengths = []
    cautions = []

    long_score += min(float(pump.long_score or pump.total_score), 100.0) * 0.28
    short_score += min(float(pump.short_score or pump.total_score), 100.0) * 0.28

    if volume_ratio >= 5:
        long_score += 8
        short_score += 8
        long_strengths.append("volume is surging")
        short_strengths.append("volume is surging")
    elif volume_ratio >= 3:
        long_score += 5
        short_score += 5

    if momentum >= 12:
        long_score += 12
        long_strengths.append("momentum is accelerating upward")
    elif momentum >= 6:
        long_score += 7
    elif momentum <= -12:
        short_score += 12
        short_strengths.append("momentum is accelerating downward")
    elif momentum <= -6:
        short_score += 7

    long_score += min(bullish_trend, 60.0) * 0.18
    short_score += min(bearish_trend, 60.0) * 0.18

    if above_mas:
        long_score += 8
        long_strengths.append("price is above MA20 and MA50")
    else:
        long_score -= 4

    if below_mas:
        short_score += 8
        short_strengths.append("price is below MA20 and MA50")
    else:
        short_score -= 4

    if 45 <= rsi <= 68:
        long_score += 8
        long_strengths.append("RSI is constructive for continuation")
    elif rsi > 78:
        long_score -= 18
        cautions.append("RSI is heavily overbought")
    elif rsi > 72:
        long_score -= 10
        cautions.append("RSI is getting stretched for longs")

    if 34 <= rsi <= 58:
        short_score += 8
        short_strengths.append("RSI still leaves room for downside")
    elif rsi < 22:
        short_score -= 18
        cautions.append("RSI is already deeply oversold")
    elif rsi < 28:
        short_score -= 10
        cautions.append("RSI is stretched for fresh shorts")

    if snapshot.price_change_24h >= 10:
        long_score += 6
    if snapshot.price_change_24h >= 18:
        long_score -= 8
        cautions.append("the upside move is already extended")

    if snapshot.price_change_24h <= -4:
        short_score += 6
    if snapshot.price_change_24h <= -18:
        short_score -= 8
        cautions.append("the downside move is already extended")

    if futures_bias == "LONG":
        long_score += min(18.0, futures_score * 0.16)
        short_score -= 10
        long_strengths.append("CoinDCX futures matrix supports a long")
    elif futures_bias == "SHORT":
        short_score += min(18.0, futures_score * 0.16)
        long_score -= 10
        short_strengths.append("CoinDCX futures matrix supports a short")
    elif futures_bias == "WAIT":
        long_score -= 8
        short_score -= 8
        cautions.append("CoinDCX futures matrix says wait")
    else:
        long_score -= 12
        short_score -= 12
        cautions.append("CoinDCX futures confirmation is unavailable")

    long_score = max(5.0, min(95.0, long_score))
    short_score = max(5.0, min(95.0, short_score))

    if long_score >= 62 and long_score >= short_score + 6:
        action = "BUY"
        confidence = long_score
        reason = "Strengths: " + ", ".join(long_strengths[:3]) + "."
    elif short_score >= 62 and short_score >= long_score + 6:
        action = "SHORT"
        confidence = short_score
        reason = "Strengths: " + ", ".join(short_strengths[:3]) + "."
    elif max(long_score, short_score) >= 48:
        action = "HOLD"
        confidence = max(long_score, short_score) - 8
        reason = "Setup is mixed and needs more confirmation before taking directional exposure."
    else:
        action = "AVOID"
        confidence = 100.0 - max(long_score, short_score)
        reason = "Setup lacks enough aligned factors and the risk/reward is not attractive."

    if cautions:
        reason = f"{reason} Caution: {', '.join(cautions[:2])}."

    return action, round(max(5.0, min(95.0, confidence)), 1), reason


def analyse(
    snapshot: MarketSnapshot,
    indicators: IndicatorSet,
    pump: PumpScore,
    trend_info: Optional[dict] = None,
    futures_context: Optional[dict] = None,
    retries: int = 2,
) -> Tuple[str, float, str]:
    """
    Call Gemini and return (action, confidence, reason).
    Falls back to heuristics if API is unavailable.
    """
    disabled_reason = _ai_disabled_reason()
    if disabled_reason:
        logger.warning("AI unavailable for %s - using fallback: %s", snapshot.symbol, disabled_reason)
        return _smart_fallback_analysis(snapshot, indicators, pump, trend_info, futures_context)

    model = _get_model()
    if not model:
        logger.warning("No Gemini model - using fallback heuristics for %s.", snapshot.symbol)
        return _smart_fallback_analysis(snapshot, indicators, pump, trend_info, futures_context)

    prompt = _build_prompt(snapshot, indicators, pump, trend_info, futures_context)

    for attempt in range(1, retries + 1):
        text = ""
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.25,
                    max_output_tokens=320,
                ),
            )
            text = response.text.strip()
            text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\n?```$", "", text)

            parsed = json.loads(text)
            action = str(parsed.get("action", "HOLD")).upper()
            confidence = float(parsed.get("confidence", 50))
            reason = str(parsed.get("reason", "No reason provided."))

            if action not in {"BUY", "SHORT", "HOLD", "AVOID"}:
                action = "HOLD"

            logger.info(
                "Gemini -> %s %s confidence=%.0f",
                snapshot.symbol,
                action,
                confidence,
            )
            return action, max(0.0, min(100.0, confidence)), reason

        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error attempt %d: %s | raw: %s", attempt, exc, text[:200])
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            logger.error("Gemini API error attempt %d: %s", attempt, exc)

            if "quota" in lowered or "429" in lowered:
                _disable_ai(
                    AI_QUOTA_COOLDOWN_SECONDS,
                    "Gemini quota exceeded; switching to rule-based fallback.",
                )
                break

            if "404" in lowered and "model" in lowered:
                _disable_ai(
                    AI_MODEL_COOLDOWN_SECONDS,
                    f"Configured model '{GEMINI_MODEL}' is unavailable for this key/API version.",
                )
                break

            if attempt < retries:
                time.sleep(5)

    logger.warning("All Gemini attempts failed - using fallback for %s.", snapshot.symbol)
    return _smart_fallback_analysis(snapshot, indicators, pump, trend_info, futures_context)
