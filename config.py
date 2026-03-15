"""
config.py — Central configuration for crypto-ai-agent
All secrets come from environment variables or a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY      = os.getenv("GEMINI_API_KEY", "")
COINGECKO_API_KEY   = os.getenv("COINGECKO_API_KEY", "")
DATABASE_URL        = os.getenv("DATABASE_URL", "")

# ── CoinGecko ─────────────────────────────────────────────────────────────────
COINGECKO_BASE_URL  = "https://api.coingecko.com/api/v3"
COINS_PER_CYCLE     = 500
COINGECKO_PAGE_SIZE = 250

# ── Scan timing ───────────────────────────────────────────────────────────────
SCAN_INTERVAL_SECONDS = 300         # 5 minutes

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
MAX_AI_EVALUATIONS_PER_CYCLE = 5
MAX_SIGNALS_PER_CYCLE        = 3
AI_QUOTA_COOLDOWN_SECONDS    = 6 * 60 * 60
AI_MODEL_COOLDOWN_SECONDS    = 24 * 60 * 60

# ── Scoring weights (learning engine updates data/weights.json at runtime) ────
DEFAULT_WEIGHTS = {
    "volume_spike":      40,
    "price_change":      25,
    "momentum_breakout": 20,
    "small_cap":         15,
}
OPPORTUNITY_SCORE_THRESHOLD = 70
ALLOWED_SIGNAL_ACTIONS      = ("BUY",)
MIN_SIGNAL_CONFIDENCE       = 55.0
MIN_SIGNAL_PUMP_SCORE       = 85.0
MIN_SIGNAL_QUALITY_SCORE    = 75.0
MIN_SIGNAL_TREND_SCORE      = 25.0

# ── Signal risk defaults ──────────────────────────────────────────────────────
DEFAULT_STOP_LOSS_PCT = 0.05
DEFAULT_TARGET_PCT    = 0.15
SIGNAL_MAX_AGE_HOURS  = 48
SIGNAL_DEDUP_HOURS    = 12
LEARNING_MIN_CLOSED_SIGNALS = 12
LEARNING_LOOKBACK_SIGNALS   = 200

# ── Storage ───────────────────────────────────────────────────────────────────
DB_PATH      = "data/signals.db"
WEIGHTS_PATH = "data/weights.json"
LOG_FILE     = "logs/agent.log"

# ── Indicators ────────────────────────────────────────────────────────────────
RSI_PERIOD = 14
MA_SHORT   = 20
MA_LONG    = 50

# ── Web dashboard ─────────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = int(os.getenv("PORT", 8080))
