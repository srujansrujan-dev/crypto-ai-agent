"""
ai_engine.py — Sends signal data to Gemini and parses its evaluation.

Returns: action (BUY/HOLD/AVOID), confidence (0-100), reason (string).
"""

import json
import logging
import re
import time
from typing import Optional, Tuple

import google.generativeai as genai

from config import GEMINI_API_KEY, GEMINI_MODEL
from signals import MarketSnapshot, IndicatorSet, PumpScore

logger = logging.getLogger(__name__)

# Configure Gemini once at import time
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not set — AI analysis will use fallback heuristics.")

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None and GEMINI_API_KEY:
        _MODEL = genai.GenerativeModel(GEMINI_MODEL)
    return _MODEL


def _build_prompt(
    snapshot:   MarketSnapshot,
    indicators: IndicatorSet,
    pump:       PumpScore,
    trend_info: Optional[dict] = None,
) -> str:
    trend_block = ""
    if trend_info:
        trend_block = f"""
Trend Detection (last 6 cycles):
  Volume trend score : {trend_info.get('volume_trend', 0):.2f}  (1.0 = always rising)
  Price acceleration : {trend_info.get('price_accel', 0):.2f}%
  Momentum delta     : {trend_info.get('momentum_delta', 0):.2f}%
  Composite trend    : {trend_info.get('trend_score', 0):.1f}/100
"""

    return f"""
You are a professional cryptocurrency market analyst. Evaluate the following opportunity signal
and provide a structured JSON response.

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
RSI          : {indicators.rsi:.1f}
MA20         : ${indicators.ma20:,.6f}
MA50         : ${indicators.ma50:,.6f}
Momentum     : {indicators.momentum:+.2f}%
Volatility   : {indicators.volatility:.4f}
Volume Ratio : {indicators.volume_ratio:.2f}x average

=== PUMP DETECTION SCORE ===
Total Score  : {pump.total_score:.1f} / 100
  Volume spike       : {pump.volume_spike}
  Price change       : {pump.price_change}
  Momentum breakout  : {pump.momentum_breakout}
  Small cap bonus    : {pump.small_cap}
{trend_block}

=== YOUR TASK ===
Analyse this opportunity as a professional trader would. Consider:
1. Is the RSI overbought (>70) or oversold (<30)?
2. Is price above or below MA20/MA50?
3. Does the volume spike look organic or like a pump-and-dump?
4. Is the market cap small enough for significant upside?
5. What is the overall risk/reward?

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "action":     "BUY" | "HOLD" | "AVOID",
  "confidence": <integer 0-100>,
  "reason":     "<2-3 sentence explanation>",
  "risk_level": "LOW" | "MEDIUM" | "HIGH"
}}
""".strip()


def _fallback_analysis(pump: PumpScore, indicators: IndicatorSet) -> Tuple[str, float, str]:
    """Simple rule-based fallback when Gemini is not available."""
    score = pump.total_score
    rsi   = indicators.rsi or 50

    if score >= 80 and rsi < 65:
        return "BUY", 60.0, "High pump score with RSI not yet overbought — potential opportunity."
    elif score >= 70 and rsi > 70:
        return "HOLD", 45.0, "Strong pump signal but RSI is overbought — wait for pullback."
    elif rsi > 75:
        return "AVOID", 70.0, "RSI severely overbought — high reversal risk."
    else:
        return "HOLD", 40.0, "Moderate signal — insufficient evidence for strong conviction."


def analyse(
    snapshot:   MarketSnapshot,
    indicators: IndicatorSet,
    pump:       PumpScore,
    trend_info: Optional[dict] = None,
    retries:    int = 2,
) -> Tuple[str, float, str]:
    """
    Call Gemini and return (action, confidence, reason).
    Falls back to heuristics if API is unavailable.
    """
    model = _get_model()
    if not model:
        logger.warning("No Gemini model — using fallback heuristics for %s.", snapshot.symbol)
        return _fallback_analysis(pump, indicators)

    prompt = _build_prompt(snapshot, indicators, pump, trend_info)

    for attempt in range(1, retries + 1):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=300,
                ),
            )
            text = response.text.strip()

            # Strip markdown fences if present
            text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\n?```$", "", text)

            parsed = json.loads(text)
            action     = parsed.get("action", "HOLD").upper()
            confidence = float(parsed.get("confidence", 50))
            reason     = parsed.get("reason", "No reason provided.")

            if action not in ("BUY", "HOLD", "AVOID"):
                action = "HOLD"

            logger.info(
                "Gemini → %s  %s  confidence=%.0f",
                snapshot.symbol, action, confidence,
            )
            return action, confidence, reason

        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error attempt %d: %s | raw: %s", attempt, exc, text[:200])
        except Exception as exc:
            logger.error("Gemini API error attempt %d: %s", attempt, exc)
            if "quota" in str(exc).lower() or "429" in str(exc):
                logger.warning("Gemini quota hit — sleeping 60s.")
                time.sleep(60)
            elif attempt < retries:
                time.sleep(5)

    logger.warning("All Gemini attempts failed — using fallback for %s.", snapshot.symbol)
    return _fallback_analysis(pump, indicators)
