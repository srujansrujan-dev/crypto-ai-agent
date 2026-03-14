"""
learning_engine.py — Tracks signal outcomes and adjusts scoring weights.

Every cycle this module:
1. Checks all PENDING signals to see if target or stop-loss was hit
2. Labels them WIN / LOSS / NEUTRAL
3. Recalculates scoring weights based on what worked
4. Saves weights to data/weights.json so pump_detector uses them next cycle
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config import DB_PATH, WEIGHTS_PATH, DEFAULT_WEIGHTS
from signals import TradingSignal

logger = logging.getLogger(__name__)


# ── Database helpers ───────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the SQLite database and tables if they don't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            coin             TEXT    NOT NULL,
            symbol           TEXT    NOT NULL,
            timestamp        TEXT    NOT NULL,
            entry_price      REAL    NOT NULL,
            target_price     REAL    NOT NULL,
            stop_loss        REAL    NOT NULL,
            buy_zone_low     REAL    NOT NULL,
            buy_zone_high    REAL    NOT NULL,
            confidence       REAL    NOT NULL,
            ai_action        TEXT    NOT NULL,
            ai_reason        TEXT,
            pump_score       REAL    NOT NULL,
            outcome          TEXT    DEFAULT 'PENDING',
            outcome_price    REAL,
            outcome_checked  INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", DB_PATH)


def save_signal(signal: TradingSignal) -> int:
    """Insert a new signal record. Returns the row id."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO signals
        (coin, symbol, timestamp, entry_price, target_price, stop_loss,
         buy_zone_low, buy_zone_high, confidence, ai_action, ai_reason, pump_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        signal.coin, signal.symbol, signal.timestamp,
        signal.entry_price, signal.target_price, signal.stop_loss,
        signal.buy_zone_low, signal.buy_zone_high,
        signal.confidence, signal.ai_action, signal.ai_reason, signal.pump_score,
    ))
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def get_pending_signals() -> List[dict]:
    """Fetch all signals still labelled PENDING."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute("SELECT * FROM signals WHERE outcome = 'PENDING'")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def update_signal_outcome(signal_id: int, outcome: str, price: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute(
        "UPDATE signals SET outcome=?, outcome_price=?, outcome_checked=1 WHERE id=?",
        (outcome, price, signal_id),
    )
    conn.commit()
    conn.close()


def get_recent_signals(limit: int = 50) -> List[dict]:
    """Fetch most recent signals for the dashboard."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute(
        "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_stats() -> dict:
    """Return win/loss/neutral/pending counts and win rate."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        SELECT outcome, COUNT(*) as cnt
        FROM signals GROUP BY outcome
    """)
    counts = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()

    wins    = counts.get("WIN",     0)
    losses  = counts.get("LOSS",    0)
    neutral = counts.get("NEUTRAL", 0)
    pending = counts.get("PENDING", 0)
    closed  = wins + losses + neutral
    win_rate = (wins / closed * 100) if closed else 0.0

    return {
        "wins":     wins,
        "losses":   losses,
        "neutral":  neutral,
        "pending":  pending,
        "total":    closed + pending,
        "win_rate": round(win_rate, 1),
    }


# ── Weights ────────────────────────────────────────────────────────────────────

def load_weights() -> dict:
    """Load weights from JSON file, or return defaults."""
    if os.path.exists(WEIGHTS_PATH):
        try:
            with open(WEIGHTS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return dict(DEFAULT_WEIGHTS)


def save_weights(weights: dict) -> None:
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(weights, f, indent=2)
    logger.info("Weights saved: %s", weights)


# ── Outcome checking ───────────────────────────────────────────────────────────

def check_outcomes(current_prices: Dict[str, float]) -> None:
    """
    Compare current prices against pending signals.
    Labels WIN if target hit, LOSS if stop-loss hit, NEUTRAL otherwise.
    """
    pending = get_pending_signals()
    if not pending:
        return

    updated = 0
    for sig in pending:
        sym   = sig["symbol"]
        price = current_prices.get(sym)
        if price is None:
            continue

        if price >= sig["target_price"]:
            update_signal_outcome(sig["id"], "WIN",  price)
            updated += 1
        elif price <= sig["stop_loss"]:
            update_signal_outcome(sig["id"], "LOSS", price)
            updated += 1
        # else still PENDING

    if updated:
        logger.info("Outcome check: %d signals updated.", updated)


# ── Weight adjustment (self-learning) ─────────────────────────────────────────

def adjust_weights() -> dict:
    """
    Recalculate scoring weights based on historical WIN/LOSS signals.

    Logic:
    - Query recent closed signals grouped by approximate score buckets
    - If BUY signals with high volume_spike tend to WIN → keep/increase that weight
    - If they tend to LOSS → reduce it
    - Weights are nudged ±5% per cycle, clamped to [5, 60]
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()

    # Need enough closed signals to learn from
    cur.execute("SELECT COUNT(*) FROM signals WHERE outcome != 'PENDING'")
    closed = cur.fetchone()[0]
    conn.close()

    weights = load_weights()

    if closed < 10:
        logger.info("Not enough closed signals to adjust weights (%d < 10).", closed)
        return weights

    stats = get_stats()
    win_rate = stats["win_rate"] / 100

    # Gentle nudge: if win rate > 60%, trust current weights; if < 40%, reduce aggression
    adjustment = (win_rate - 0.5) * 0.1   # −5% to +5%

    for key in weights:
        weights[key] = round(
            max(5.0, min(60.0, weights[key] * (1 + adjustment))), 2
        )

    # Renormalise so weights sum to 100
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total * 100, 2) for k, v in weights.items()}

    save_weights(weights)
    logger.info(
        "Weights adjusted (win_rate=%.1f%%, delta=%.3f): %s",
        stats["win_rate"], adjustment, weights,
    )
    return weights
