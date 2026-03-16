"""
learning_engine.py - Tracks signal outcomes and adjusts scoring weights.

Storage behavior:
- Use Postgres when DATABASE_URL is configured and psycopg is available.
- Fall back to local SQLite otherwise.
- Persist adaptive weights in the database when possible, with a JSON file
  kept as a local fallback/cache for local runs.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional dependency for local fallback
    psycopg = None
    dict_row = None

from config import (
    DATABASE_URL,
    DB_PATH,
    DEFAULT_WEIGHTS,
    LEARNING_LOOKBACK_SIGNALS,
    LEARNING_MIN_CLOSED_SIGNALS,
    SIGNAL_DEDUP_HOURS,
    SIGNAL_MAX_AGE_HOURS,
    SIMULATION_STAKE_INR,
    WEIGHTS_PATH,
)
from signals import TradingSignal

logger = logging.getLogger(__name__)

_warned_missing_psycopg = False
_warned_postgres_connect = False

_SIGNAL_COLUMN_TYPES = {
    "asset_id": "TEXT",
    "coin": "TEXT NOT NULL",
    "symbol": "TEXT NOT NULL",
    "timestamp": "TEXT NOT NULL",
    "entry_price": "REAL NOT NULL",
    "target_price": "REAL NOT NULL",
    "stop_loss": "REAL NOT NULL",
    "buy_zone_low": "REAL NOT NULL",
    "buy_zone_high": "REAL NOT NULL",
    "confidence": "REAL NOT NULL",
    "ai_action": "TEXT NOT NULL",
    "ai_reason": "TEXT",
    "pump_score": "REAL NOT NULL",
    "quality_score": "REAL DEFAULT 0",
    "aggregate_score": "REAL DEFAULT 0",
    "deep_score": "REAL DEFAULT 0",
    "liquidity_score": "REAL DEFAULT 0",
    "risk_score": "REAL DEFAULT 0",
    "market_regime": "TEXT DEFAULT 'UNKNOWN'",
    "regime_score": "REAL DEFAULT 50",
    "trend_score": "REAL DEFAULT 0",
    "volume_ratio": "REAL DEFAULT 0",
    "momentum": "REAL DEFAULT 0",
    "rsi": "REAL DEFAULT 50",
    "market_cap": "REAL DEFAULT 0",
    "price_change_24h": "REAL DEFAULT 0",
    "futures_bias": "TEXT DEFAULT 'UNAVAILABLE'",
    "leverage_hint": "TEXT DEFAULT 'Unavailable'",
    "futures_exchange": "TEXT DEFAULT ''",
    "futures_symbol": "TEXT DEFAULT ''",
    "funding_rate": "REAL DEFAULT 0",
    "open_interest": "REAL DEFAULT 0",
    "basis": "REAL DEFAULT 0",
    "spread": "REAL DEFAULT 0",
    "futures_volume_24h": "REAL DEFAULT 0",
    "futures_score": "REAL DEFAULT 0",
    "shadow_action": "TEXT DEFAULT 'HOLD'",
    "shadow_bias": "TEXT DEFAULT 'WAIT'",
    "shadow_confidence": "REAL DEFAULT 0",
    "shadow_score": "REAL DEFAULT 0",
    "shadow_note": "TEXT DEFAULT ''",
    "outcome": "TEXT DEFAULT 'PENDING'",
    "outcome_price": "REAL",
    "outcome_checked": "INTEGER DEFAULT 0",
    "last_price": "REAL",
    "last_price_updated_at": "TEXT",
}


def _should_use_postgres() -> bool:
    global _warned_missing_psycopg

    if not DATABASE_URL:
        return False
    if psycopg is None:
        if not _warned_missing_psycopg:
            logger.warning(
                "DATABASE_URL is set but psycopg is not installed. Falling back to SQLite."
            )
            _warned_missing_psycopg = True
        return False
    return True


def _sqlite_connect(row_factory: bool = False):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    if row_factory:
        conn.row_factory = sqlite3.Row
    return conn


def _postgres_connect(row_factory: bool = False):
    kwargs = {}
    if row_factory and dict_row is not None:
        kwargs["row_factory"] = dict_row
    return psycopg.connect(DATABASE_URL, **kwargs)


def _connect(row_factory: bool = False):
    global _warned_postgres_connect

    if _should_use_postgres():
        try:
            return _postgres_connect(row_factory), "postgres"
        except Exception as exc:  # pragma: no cover - depends on environment
            if not _warned_postgres_connect:
                logger.warning(
                    "Postgres connection failed (%s). Falling back to SQLite.",
                    exc,
                )
                _warned_postgres_connect = True

    return _sqlite_connect(row_factory), "sqlite"


def _sqlite_add_missing_columns(conn) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(signals)")
    existing = {row[1] for row in cur.fetchall()}
    for column, column_type in _SIGNAL_COLUMN_TYPES.items():
        if column not in existing:
            cur.execute(f"ALTER TABLE signals ADD COLUMN {column} {column_type}")


def _ensure_sqlite_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id         TEXT,
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
            quality_score    REAL    DEFAULT 0,
            aggregate_score  REAL    DEFAULT 0,
            deep_score       REAL    DEFAULT 0,
            liquidity_score  REAL    DEFAULT 0,
            risk_score       REAL    DEFAULT 0,
            market_regime    TEXT    DEFAULT 'UNKNOWN',
            regime_score     REAL    DEFAULT 50,
            trend_score      REAL    DEFAULT 0,
            volume_ratio     REAL    DEFAULT 0,
            momentum         REAL    DEFAULT 0,
            rsi              REAL    DEFAULT 50,
            market_cap       REAL    DEFAULT 0,
            price_change_24h REAL    DEFAULT 0,
            futures_bias     TEXT    DEFAULT 'UNAVAILABLE',
            leverage_hint    TEXT    DEFAULT 'Unavailable',
            futures_exchange TEXT    DEFAULT '',
            futures_symbol   TEXT    DEFAULT '',
            funding_rate     REAL    DEFAULT 0,
            open_interest    REAL    DEFAULT 0,
            basis            REAL    DEFAULT 0,
            spread           REAL    DEFAULT 0,
            futures_volume_24h REAL  DEFAULT 0,
            futures_score    REAL    DEFAULT 0,
            shadow_action    TEXT    DEFAULT 'HOLD',
            shadow_bias      TEXT    DEFAULT 'WAIT',
            shadow_confidence REAL   DEFAULT 0,
            shadow_score     REAL    DEFAULT 0,
            shadow_note      TEXT    DEFAULT '',
            outcome          TEXT    DEFAULT 'PENDING',
            outcome_price    REAL,
            outcome_checked  INTEGER DEFAULT 0,
            last_price       REAL,
            last_price_updated_at TEXT
        )
        """
    )
    _sqlite_add_missing_columns(conn)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_state (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_signals_outcome ON signals(outcome)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)"
    )


def _ensure_postgres_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signals (
            id               BIGSERIAL PRIMARY KEY,
            asset_id         TEXT,
            coin             TEXT NOT NULL,
            symbol           TEXT NOT NULL,
            timestamp        TEXT NOT NULL,
            entry_price      DOUBLE PRECISION NOT NULL,
            target_price     DOUBLE PRECISION NOT NULL,
            stop_loss        DOUBLE PRECISION NOT NULL,
            buy_zone_low     DOUBLE PRECISION NOT NULL,
            buy_zone_high    DOUBLE PRECISION NOT NULL,
            confidence       DOUBLE PRECISION NOT NULL,
            ai_action        TEXT NOT NULL,
            ai_reason        TEXT,
            pump_score       DOUBLE PRECISION NOT NULL,
            quality_score    DOUBLE PRECISION DEFAULT 0,
            aggregate_score  DOUBLE PRECISION DEFAULT 0,
            deep_score       DOUBLE PRECISION DEFAULT 0,
            liquidity_score  DOUBLE PRECISION DEFAULT 0,
            risk_score       DOUBLE PRECISION DEFAULT 0,
            market_regime    TEXT DEFAULT 'UNKNOWN',
            regime_score     DOUBLE PRECISION DEFAULT 50,
            trend_score      DOUBLE PRECISION DEFAULT 0,
            volume_ratio     DOUBLE PRECISION DEFAULT 0,
            momentum         DOUBLE PRECISION DEFAULT 0,
            rsi              DOUBLE PRECISION DEFAULT 50,
            market_cap       DOUBLE PRECISION DEFAULT 0,
            price_change_24h DOUBLE PRECISION DEFAULT 0,
            futures_bias     TEXT DEFAULT 'UNAVAILABLE',
            leverage_hint    TEXT DEFAULT 'Unavailable',
            futures_exchange TEXT DEFAULT '',
            futures_symbol   TEXT DEFAULT '',
            funding_rate     DOUBLE PRECISION DEFAULT 0,
            open_interest    DOUBLE PRECISION DEFAULT 0,
            basis            DOUBLE PRECISION DEFAULT 0,
            spread           DOUBLE PRECISION DEFAULT 0,
            futures_volume_24h DOUBLE PRECISION DEFAULT 0,
            futures_score    DOUBLE PRECISION DEFAULT 0,
            shadow_action    TEXT DEFAULT 'HOLD',
            shadow_bias      TEXT DEFAULT 'WAIT',
            shadow_confidence DOUBLE PRECISION DEFAULT 0,
            shadow_score     DOUBLE PRECISION DEFAULT 0,
            shadow_note      TEXT DEFAULT '',
            outcome          TEXT DEFAULT 'PENDING',
            outcome_price    DOUBLE PRECISION,
            outcome_checked  BOOLEAN DEFAULT FALSE,
            last_price       DOUBLE PRECISION,
            last_price_updated_at TEXT
        )
        """
    )
    for column, column_type in _SIGNAL_COLUMN_TYPES.items():
        pg_type = column_type.replace("REAL", "DOUBLE PRECISION").replace(
            "INTEGER DEFAULT 0", "BOOLEAN DEFAULT FALSE"
        )
        cur.execute(
            f"ALTER TABLE signals ADD COLUMN IF NOT EXISTS {column} {pg_type}"
        )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_state (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_signals_outcome ON signals(outcome)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)"
    )


def init_db() -> None:
    """Create the active database schema if it does not exist."""
    conn, backend = _connect()
    try:
        if backend == "postgres":
            _ensure_postgres_schema(conn)
            logger.info("Database initialised using Postgres.")
        else:
            _ensure_sqlite_schema(conn)
            logger.info("Database initialised at %s", DB_PATH)
        conn.commit()
    finally:
        conn.close()


def _signal_payload(signal: TradingSignal):
    return (
        signal.asset_id,
        signal.coin,
        signal.symbol,
        signal.timestamp,
        signal.entry_price,
        signal.target_price,
        signal.stop_loss,
        signal.buy_zone_low,
        signal.buy_zone_high,
        signal.confidence,
        signal.ai_action,
        signal.ai_reason,
        signal.pump_score,
        signal.quality_score,
        signal.aggregate_score,
        signal.deep_score,
        signal.liquidity_score,
        signal.risk_score,
        signal.market_regime,
        signal.regime_score,
        signal.trend_score,
        signal.volume_ratio,
        signal.momentum,
        signal.rsi,
        signal.market_cap,
        signal.price_change_24h,
        signal.futures_bias,
        signal.leverage_hint,
        signal.futures_exchange,
        signal.futures_symbol,
        signal.funding_rate,
        signal.open_interest,
        signal.basis,
        signal.spread,
        signal.futures_volume_24h,
        signal.futures_score,
        signal.shadow_action,
        signal.shadow_bias,
        signal.shadow_confidence,
        signal.shadow_score,
        signal.shadow_note,
        signal.outcome,
        signal.outcome_price,
        bool(signal.outcome_checked),
    )


def save_signal(signal: TradingSignal) -> int:
    """Insert a new signal record. Returns the new row id."""
    conn, backend = _connect()
    try:
        cur = conn.cursor()
        columns = (
            "asset_id, coin, symbol, timestamp, entry_price, target_price, "
            "stop_loss, buy_zone_low, buy_zone_high, confidence, ai_action, "
            "ai_reason, pump_score, quality_score, aggregate_score, deep_score, "
            "liquidity_score, risk_score, market_regime, regime_score, trend_score, volume_ratio, "
            "momentum, rsi, market_cap, price_change_24h, futures_bias, leverage_hint, "
            "futures_exchange, futures_symbol, funding_rate, open_interest, basis, spread, "
            "futures_volume_24h, futures_score, shadow_action, shadow_bias, shadow_confidence, shadow_score, shadow_note, outcome, "
            "outcome_price, outcome_checked"
        )
        payload = _signal_payload(signal)
        if backend == "postgres":
            placeholders = ", ".join(["%s"] * len(payload))
            cur.execute(
                f"""
                INSERT INTO signals ({columns})
                VALUES ({placeholders})
                RETURNING id
                """,
                payload,
            )
            row_id = cur.fetchone()[0]
        else:
            placeholders = ", ".join(["?"] * len(payload))
            cur.execute(
                f"""
                INSERT INTO signals ({columns})
                VALUES ({placeholders})
                """,
                payload,
            )
            row_id = cur.lastrowid
        conn.commit()
        return row_id
    finally:
        conn.close()


def has_recent_pending_signal(
    asset_id: str,
    symbol: str,
    hours: int = SIGNAL_DEDUP_HOURS,
) -> bool:
    """Avoid duplicate open signals for the same asset within a short window."""
    cutoff_iso = (
        datetime.now(timezone.utc) - timedelta(hours=hours)
    ).isoformat()

    conn, backend = _connect()
    try:
        cur = conn.cursor()
        query = """
            SELECT 1
            FROM signals
            WHERE outcome = 'PENDING'
              AND timestamp >= {placeholder}
              AND (
                ((asset_id IS NOT NULL) AND asset_id <> '' AND asset_id = {placeholder})
                OR symbol = {placeholder}
              )
            LIMIT 1
        """
        placeholder = "%s" if backend == "postgres" else "?"
        cur.execute(
            query.format(placeholder=placeholder),
            (cutoff_iso, asset_id, symbol),
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def _rows_to_dicts(rows) -> List[dict]:
    return [dict(row) for row in rows]


def get_pending_signals() -> List[dict]:
    """Fetch all signals still labelled PENDING."""
    conn, backend = _connect(row_factory=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM signals WHERE outcome = 'PENDING' ORDER BY id ASC"
        )
        return _rows_to_dicts(cur.fetchall())
    finally:
        conn.close()


def update_signal_outcome(signal_id: int, outcome: str, price: float) -> None:
    conn, backend = _connect()
    try:
        cur = conn.cursor()
        updated_at = datetime.now(timezone.utc).isoformat()
        if backend == "postgres":
            cur.execute(
                """
                UPDATE signals
                SET outcome = %s,
                    outcome_price = %s,
                    outcome_checked = %s,
                    last_price = %s,
                    last_price_updated_at = %s
                WHERE id = %s
                """,
                (outcome, price, True, price, updated_at, signal_id),
            )
        else:
            cur.execute(
                """
                UPDATE signals
                SET outcome = ?,
                    outcome_price = ?,
                    outcome_checked = ?,
                    last_price = ?,
                    last_price_updated_at = ?
                WHERE id = ?
                """,
                (outcome, price, 1, price, updated_at, signal_id),
            )
        conn.commit()
    finally:
        conn.close()


def update_signal_marks(current_prices: Dict[str, float]) -> None:
    """Mark pending signals to market using the latest scan prices."""
    pending = get_pending_signals()
    if not pending:
        return

    conn, backend = _connect()
    try:
        cur = conn.cursor()
        updated = 0
        updated_at = datetime.now(timezone.utc).isoformat()
        for signal in pending:
            asset_id = signal.get("asset_id") or ""
            symbol = signal.get("symbol", "")
            price = current_prices.get(asset_id) if asset_id else None
            if price is None:
                price = current_prices.get(symbol)
            if price is None:
                continue

            if backend == "postgres":
                cur.execute(
                    """
                    UPDATE signals
                    SET last_price = %s, last_price_updated_at = %s
                    WHERE id = %s
                    """,
                    (price, updated_at, signal["id"]),
                )
            else:
                cur.execute(
                    """
                    UPDATE signals
                    SET last_price = ?, last_price_updated_at = ?
                    WHERE id = ?
                    """,
                    (price, updated_at, signal["id"]),
                )
            updated += 1

        if updated:
            conn.commit()
            logger.info("Marked %d pending signals to market.", updated)
    finally:
        conn.close()


def get_recent_signals(
    limit: int = 50,
    offset: int = 0,
    outcome: Optional[str] = None,
) -> List[dict]:
    """Fetch recent signals for the dashboard or history views."""
    conn, backend = _connect(row_factory=True)
    try:
        cur = conn.cursor()
        where_clause = ""
        params: List = []
        placeholder = "%s" if backend == "postgres" else "?"
        if outcome:
            where_clause = f"WHERE outcome = {placeholder}"
            params.append(outcome)

        if backend == "postgres":
            params.extend([limit, offset])
            cur.execute(
                f"SELECT * FROM signals {where_clause} ORDER BY id DESC LIMIT %s OFFSET %s",
                tuple(params),
            )
        else:
            params.extend([limit, offset])
            cur.execute(
                f"SELECT * FROM signals {where_clause} ORDER BY id DESC LIMIT ? OFFSET ?",
                tuple(params),
            )
        return _rows_to_dicts(cur.fetchall())
    finally:
        conn.close()


def get_stats() -> dict:
    """Return summary counts and win rate."""
    conn, backend = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT outcome, COUNT(*) AS cnt
            FROM signals
            GROUP BY outcome
            """
        )
        counts = {row[0]: row[1] for row in cur.fetchall()}
    finally:
        conn.close()

    wins = counts.get("WIN", 0)
    losses = counts.get("LOSS", 0)
    neutral = counts.get("NEUTRAL", 0)
    pending = counts.get("PENDING", 0)
    closed = wins + losses + neutral
    win_rate = (wins / closed * 100.0) if closed else 0.0

    return {
        "wins": wins,
        "losses": losses,
        "neutral": neutral,
        "pending": pending,
        "closed": closed,
        "total": closed + pending,
        "win_rate": round(win_rate, 1),
    }


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _signal_mark_price(signal: dict) -> float:
    outcome = str(signal.get("outcome") or "PENDING").upper()
    entry_price = _coerce_float(signal.get("entry_price"))
    if outcome != "PENDING":
        return _coerce_float(signal.get("outcome_price"), entry_price)
    return _coerce_float(signal.get("last_price"), entry_price)


def _signal_return_pct(signal: dict, exit_price: float) -> float:
    entry_price = _coerce_float(signal.get("entry_price"))
    if entry_price <= 0:
        return 0.0
    if str(signal.get("ai_action") or "").upper() == "SHORT":
        return ((entry_price - exit_price) / entry_price) * 100.0
    return ((exit_price - entry_price) / entry_price) * 100.0


def enrich_signal_simulation(signal: dict, stake_inr: float = SIMULATION_STAKE_INR) -> dict:
    """Add paper-trading metrics for a fixed INR stake per signal."""
    enriched = dict(signal)
    exit_price = _signal_mark_price(enriched)
    return_pct = _signal_return_pct(enriched, exit_price)
    value_inr = max(0.0, stake_inr * (1 + return_pct / 100.0))
    pnl_inr = value_inr - stake_inr

    created_at = _parse_timestamp(enriched.get("timestamp"))
    age_hours = 0.0
    if created_at is not None:
        age_hours = max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds() / 3600.0)

    enriched["simulation_stake_inr"] = round(stake_inr, 2)
    enriched["simulation_exit_price"] = round(exit_price, 8)
    enriched["simulation_return_pct"] = round(return_pct, 2)
    enriched["simulation_value_inr"] = round(value_inr, 2)
    enriched["simulation_pnl_inr"] = round(pnl_inr, 2)
    enriched["simulation_age_hours"] = round(age_hours, 2)
    enriched["simulation_is_closed"] = str(enriched.get("outcome") or "").upper() != "PENDING"
    return enriched


def get_simulation_stats(stake_inr: float = SIMULATION_STAKE_INR) -> dict:
    """Return portfolio-level paper-trading totals for all signals."""
    conn, backend = _connect(row_factory=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM signals ORDER BY id DESC")
        rows = _rows_to_dicts(cur.fetchall())
    finally:
        conn.close()

    if not rows:
        return {
            "stake_inr": round(stake_inr, 2),
            "total_trades": 0,
            "closed_trades": 0,
            "open_trades": 0,
            "total_invested_inr": 0.0,
            "current_value_inr": 0.0,
            "realized_pnl_inr": 0.0,
            "unrealized_pnl_inr": 0.0,
            "total_pnl_inr": 0.0,
            "roi_pct": 0.0,
        }

    enriched_rows = [enrich_signal_simulation(row, stake_inr) for row in rows]
    closed_rows = [row for row in enriched_rows if row["simulation_is_closed"]]
    open_rows = [row for row in enriched_rows if not row["simulation_is_closed"]]
    total_invested = stake_inr * len(enriched_rows)
    current_value = sum(row["simulation_value_inr"] for row in enriched_rows)
    realized_pnl = sum(row["simulation_pnl_inr"] for row in closed_rows)
    unrealized_pnl = sum(row["simulation_pnl_inr"] for row in open_rows)
    total_pnl = current_value - total_invested
    roi_pct = (total_pnl / total_invested * 100.0) if total_invested else 0.0

    return {
        "stake_inr": round(stake_inr, 2),
        "total_trades": len(enriched_rows),
        "closed_trades": len(closed_rows),
        "open_trades": len(open_rows),
        "total_invested_inr": round(total_invested, 2),
        "current_value_inr": round(current_value, 2),
        "realized_pnl_inr": round(realized_pnl, 2),
        "unrealized_pnl_inr": round(unrealized_pnl, 2),
        "total_pnl_inr": round(total_pnl, 2),
        "roi_pct": round(roi_pct, 2),
    }


def get_storage_backend_name() -> str:
    """Best-effort backend label for dashboard visibility."""
    if _should_use_postgres():
        try:
            conn = _postgres_connect()
            conn.close()
            return "Postgres"
        except Exception:
            pass
    return "SQLite"


def _load_weights_from_file() -> dict:
    if os.path.exists(WEIGHTS_PATH):
        try:
            with open(WEIGHTS_PATH, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Could not read %s: %s", WEIGHTS_PATH, exc)
    return dict(DEFAULT_WEIGHTS)


def _save_weights_to_file(weights: dict) -> None:
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(weights, handle, indent=2)


def load_weights() -> dict:
    """Load adaptive weights from database state or local file fallback."""
    if not _should_use_postgres():
        return _load_weights_from_file()

    conn, backend = _connect()
    try:
        if backend != "postgres":
            return _load_weights_from_file()

        cur = conn.cursor()
        cur.execute("SELECT value FROM agent_state WHERE key = %s", ("weights",))
        row = cur.fetchone()
        if not row:
            return _load_weights_from_file()
        return json.loads(row[0])
    except Exception as exc:
        logger.warning("Could not load weights from database: %s", exc)
        return _load_weights_from_file()
    finally:
        conn.close()


def save_weights(weights: dict) -> None:
    """Persist adaptive weights and keep the JSON cache in sync."""
    normalised = {key: round(float(value), 2) for key, value in weights.items()}
    _save_weights_to_file(normalised)

    conn, backend = _connect()
    try:
        cur = conn.cursor()
        payload = (
            "weights",
            json.dumps(normalised),
            datetime.now(timezone.utc).isoformat(),
        )
        if backend == "postgres":
            cur.execute(
                """
                INSERT INTO agent_state (key, value, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (key) DO UPDATE
                SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                """,
                payload,
            )
        else:
            cur.execute(
                """
                INSERT INTO agent_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                payload,
            )
        conn.commit()
        logger.info("Weights saved: %s", normalised)
    finally:
        conn.close()


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_signal_stale(timestamp: Optional[str]) -> bool:
    if not timestamp:
        return False

    try:
        created_at = datetime.fromisoformat(timestamp)
    except ValueError:
        return False

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    age = datetime.now(timezone.utc) - created_at
    return age >= timedelta(hours=SIGNAL_MAX_AGE_HOURS)


def check_outcomes(current_prices: Dict[str, float]) -> None:
    """
    Compare current prices against pending signals.
    Labels WIN if target hit, LOSS if stop-loss hit, and NEUTRAL if the
    signal ages out without resolving.
    """
    pending = get_pending_signals()
    if not pending:
        return

    updated = 0
    for signal in pending:
        asset_id = signal.get("asset_id") or ""
        symbol = signal.get("symbol", "")
        action = str(signal.get("ai_action") or "BUY").upper()
        price = current_prices.get(asset_id) if asset_id else None
        if price is None:
            price = current_prices.get(symbol)

        target_price = _coerce_float(signal.get("target_price"))
        stop_loss = _coerce_float(signal.get("stop_loss"))
        resolved = False
        if action == "SHORT":
            if price is not None and price <= target_price:
                update_signal_outcome(signal["id"], "WIN", price)
                updated += 1
                resolved = True
            elif price is not None and price >= stop_loss:
                update_signal_outcome(signal["id"], "LOSS", price)
                updated += 1
                resolved = True
        else:
            if price is not None and price >= target_price:
                update_signal_outcome(signal["id"], "WIN", price)
                updated += 1
                resolved = True
            elif price is not None and price <= stop_loss:
                update_signal_outcome(signal["id"], "LOSS", price)
                updated += 1
                resolved = True
        if not resolved and _is_signal_stale(signal.get("timestamp")):
            fallback_price = price
            if fallback_price is None:
                fallback_price = _coerce_float(signal.get("entry_price"))
            update_signal_outcome(signal["id"], "NEUTRAL", fallback_price)
            updated += 1

    if updated:
        logger.info("Outcome check: %d signals updated.", updated)


def _normalize_volume_ratio(volume_ratio: float) -> float:
    return max(0.0, min(1.0, (volume_ratio - 1.0) / 4.0))


def _normalize_price_change(price_change_24h: float, rsi: float, action: str) -> float:
    magnitude = abs(price_change_24h)
    base = max(0.0, min(1.0, (magnitude - 2.0) / 12.0))
    if str(action).upper() == "SHORT":
        if rsi < 22:
            base *= 0.35
        elif rsi < 28:
            base *= 0.6
        elif 34 <= rsi <= 58:
            base *= 1.0
        else:
            base *= 0.85
    else:
        if rsi > 78:
            base *= 0.35
        elif rsi > 72:
            base *= 0.6
        elif 45 <= rsi <= 68:
            base *= 1.0
        else:
            base *= 0.85
    return max(0.0, min(1.0, base))


def _normalize_momentum(momentum: float, trend_score: float, action: str) -> float:
    if str(action).upper() == "SHORT":
        base = max(0.0, min(1.0, abs(min(momentum, 0.0)) / 15.0))
    else:
        base = max(0.0, min(1.0, max(momentum, 0.0) / 15.0))
    trend = max(0.0, min(1.0, trend_score / 50.0))
    return max(0.0, min(1.0, base * 0.6 + trend * 0.4))


def _normalize_small_cap(market_cap: float) -> float:
    if market_cap <= 0:
        return 0.2
    if market_cap <= 50_000_000:
        return 1.0
    if market_cap <= 200_000_000:
        return 0.8
    if market_cap <= 1_000_000_000:
        return 0.55
    if market_cap <= 5_000_000_000:
        return 0.3
    return 0.1


def _feature_bias(rows: List[dict]) -> dict:
    outcome_scores = {"WIN": 1.0, "LOSS": -1.0, "NEUTRAL": -0.35}
    totals = {key: 0.0 for key in DEFAULT_WEIGHTS}
    strengths = {key: 0.0 for key in DEFAULT_WEIGHTS}

    for row in rows:
        outcome_score = outcome_scores.get(row.get("outcome"))
        if outcome_score is None:
            continue
        action = str(row.get("ai_action") or "BUY").upper()

        quality = _coerce_float(
            row.get("aggregate_score"),
            _coerce_float(
                row.get("quality_score"),
                _coerce_float(row.get("pump_score")),
            ),
        )
        importance = 0.5 + max(0.0, min(100.0, quality)) / 200.0

        features = {
            "volume_spike": _normalize_volume_ratio(
                _coerce_float(row.get("volume_ratio"), 1.0)
            ),
            "price_change": _normalize_price_change(
                _coerce_float(row.get("price_change_24h")),
                _coerce_float(row.get("rsi"), 50.0),
                action,
            ),
            "momentum_breakout": _normalize_momentum(
                _coerce_float(row.get("momentum")),
                _coerce_float(row.get("trend_score")),
                action,
            ),
            "small_cap": _normalize_small_cap(
                _coerce_float(row.get("market_cap"))
            ),
        }

        for key, strength in features.items():
            if strength <= 0:
                continue
            totals[key] += importance * strength
            strengths[key] += outcome_score * importance * strength

    bias = {}
    for key in DEFAULT_WEIGHTS:
        if totals[key] <= 0:
            bias[key] = 0.0
        else:
            bias[key] = strengths[key] / totals[key]
    return bias


def _renormalize_weights(weights: dict) -> dict:
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {
        key: round(value / total * 100.0, 2)
        for key, value in weights.items()
    }


def adjust_weights() -> dict:
    """
    Recalculate scoring weights based on recent resolved outcomes.

    The learning loop rewards feature patterns that appeared in winners and
    penalises those that showed up in losses or stale neutral outcomes.
    """
    weights = load_weights()

    conn, backend = _connect(row_factory=True)
    try:
        cur = conn.cursor()
        if backend == "postgres":
            cur.execute(
                """
                SELECT * FROM signals
                WHERE outcome != 'PENDING'
                ORDER BY id DESC
                LIMIT %s
                """,
                (LEARNING_LOOKBACK_SIGNALS,),
            )
        else:
            cur.execute(
                """
                SELECT * FROM signals
                WHERE outcome != 'PENDING'
                ORDER BY id DESC
                LIMIT ?
                """,
                (LEARNING_LOOKBACK_SIGNALS,),
            )
        rows = _rows_to_dicts(cur.fetchall())
    finally:
        conn.close()

    if len(rows) < LEARNING_MIN_CLOSED_SIGNALS:
        logger.info(
            "Not enough closed signals to adjust weights (%d < %d).",
            len(rows),
            LEARNING_MIN_CLOSED_SIGNALS,
        )
        return weights

    wins = sum(1 for row in rows if row.get("outcome") == "WIN")
    losses = sum(1 for row in rows if row.get("outcome") == "LOSS")
    neutrals = sum(1 for row in rows if row.get("outcome") == "NEUTRAL")
    closed = wins + losses + neutrals
    win_rate = (wins / closed) if closed else 0.0

    feature_bias = _feature_bias(rows)
    global_adjustment = (win_rate - 0.5) * 0.06

    updated = {}
    for key, current_value in weights.items():
        local_adjustment = max(
            -0.18,
            min(0.18, feature_bias.get(key, 0.0) * 0.12),
        )
        factor = max(0.5, 1.0 + global_adjustment + local_adjustment)
        updated[key] = round(
            max(5.0, min(60.0, float(current_value) * factor)),
            2,
        )

    updated = _renormalize_weights(updated)
    save_weights(updated)
    logger.info(
        "Weights adjusted from %d closed signals (win_rate=%.1f%%, feature_bias=%s): %s",
        len(rows),
        win_rate * 100.0,
        {key: round(value, 3) for key, value in feature_bias.items()},
        updated,
    )
    return updated
