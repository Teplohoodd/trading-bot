"""SQLite database: schema, connection management, CRUD repository."""

import json
import aiosqlite
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from typing import Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    figi TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    lots INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    stop_loss REAL,
    take_profit REAL,
    entry_order_id TEXT,
    exit_order_id TEXT,
    strategy TEXT NOT NULL,
    signal_confidence REAL,
    status TEXT NOT NULL DEFAULT 'open',
    pnl REAL,
    pnl_pct REAL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    exit_reason TEXT,
    lot_size INTEGER DEFAULT 1,
    instrument_kind TEXT DEFAULT 'share',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS candle_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    figi TEXT NOT NULL,
    interval TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    UNIQUE(figi, interval, ts)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    figi TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,
    confidence REAL NOT NULL,
    strategy TEXT NOT NULL,
    features TEXT,
    approved INTEGER DEFAULT 0,
    rejection_reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    figi TEXT,
    model_path TEXT NOT NULL,
    version INTEGER NOT NULL,
    accuracy REAL,
    f1_score REAL,
    train_samples INTEGER,
    feature_names TEXT,
    trained_at TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    tb_max_hold INTEGER DEFAULT 10
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL UNIQUE,
    realized_pnl REAL DEFAULT 0,
    unrealized_pnl REAL DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    portfolio_value REAL,
    max_drawdown_pct REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS custom_tickers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL UNIQUE,
    figi TEXT,
    name TEXT,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Engine state persisted across restarts.
--   position_peaks: trailing-stop peak/activated flag per open trade.  Without
--     this the trailing-stop machinery resets on every bot restart and
--     re-anchors to the current price as the new "peak" — wiping out any
--     lock-in that had accumulated.
--   cooldowns: last close time per FIGI.  Without this SAME_TICKER_COOLDOWN
--     was lost on restart and the bot would re-enter a freshly-stopped name.
CREATE TABLE IF NOT EXISTS position_peaks (
    trade_id INTEGER PRIMARY KEY,
    peak_price REAL NOT NULL,
    activated INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    -- Partial-TP state (FIX 4): remaining lots after scale-out, and whether
    -- the partial has fired.  Lets the engine close residual lots cleanly.
    partial_taken INTEGER NOT NULL DEFAULT 0,
    partial_price REAL
);

CREATE TABLE IF NOT EXISTS cooldowns (
    figi TEXT PRIMARY KEY,
    last_close_time TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_figi ON trades(figi);
CREATE INDEX IF NOT EXISTS idx_candle_cache_lookup ON candle_cache(figi, interval, ts);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at);
"""


@dataclass
class Trade:
    figi: str
    ticker: str
    direction: str  # "buy" or "sell"
    lots: int
    entry_price: float
    strategy: str
    signal_confidence: float
    entry_time: str
    lot_size: int = 1
    instrument_kind: str = "share"  # "share" | "future"
    id: Optional[int] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    status: str = "open"
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None


class Repository:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        # Lightweight migrations for columns added after initial release
        await self._migrate_add_column("trades", "instrument_kind", "TEXT DEFAULT 'share'")
        await self._migrate_add_column("model_registry", "tb_max_hold", "INTEGER DEFAULT 10")

    async def _migrate_add_column(self, table: str, column: str, coldef: str) -> None:
        """Add `column` to `table` if it doesn't exist yet (SQLite 3.x)."""
        try:
            cur = await self._db.execute(f"PRAGMA table_info({table})")
            cols = [row[1] for row in await cur.fetchall()]
            if column not in cols:
                await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coldef}")
                await self._db.commit()
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Migration ADD COLUMN {table}.{column} failed: {e}"
            )

    async def close(self):
        if self._db:
            await self._db.close()

    # --- Trades ---

    async def insert_trade(self, trade: Trade) -> int:
        cur = await self._db.execute(
            """INSERT INTO trades (figi, ticker, direction, lots, entry_price,
               stop_loss, take_profit, entry_order_id, strategy, signal_confidence,
               status, entry_time, lot_size, instrument_kind)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.figi,
                trade.ticker,
                trade.direction,
                trade.lots,
                trade.entry_price,
                trade.stop_loss,
                trade.take_profit,
                trade.entry_order_id,
                trade.strategy,
                trade.signal_confidence,
                trade.status,
                trade.entry_time,
                trade.lot_size,
                getattr(trade, "instrument_kind", "share"),
            ),
        )
        await self._db.commit()
        return cur.lastrowid

    async def update_trade(self, trade_id: int, **kwargs):
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values())
        vals.append(trade_id)
        await self._db.execute(f"UPDATE trades SET {sets} WHERE id = ?", vals)
        await self._db.commit()

    async def get_open_trades(self) -> list[dict]:
        cur = await self._db.execute("SELECT * FROM trades WHERE status = 'open'")
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_trades(self, from_dt: str, to_dt: str) -> list[dict]:
        cur = await self._db.execute(
            "SELECT * FROM trades WHERE entry_time >= ? AND entry_time <= ? ORDER BY entry_time DESC",
            (from_dt, to_dt),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_closed_trades(self, limit: int = 50) -> list[dict]:
        cur = await self._db.execute(
            "SELECT * FROM trades WHERE status = 'closed' ORDER BY exit_time DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # --- Candle Cache ---

    async def upsert_candles(self, figi: str, interval: str, candles: list[dict]):
        await self._db.executemany(
            """INSERT OR REPLACE INTO candle_cache (figi, interval, ts, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (figi, interval, c["ts"], c["open"], c["high"], c["low"], c["close"], c["volume"])
                for c in candles
            ],
        )
        await self._db.commit()

    async def get_cached_candles(
        self, figi: str, interval: str, from_dt: str, to_dt: str
    ) -> list[dict]:
        cur = await self._db.execute(
            """SELECT * FROM candle_cache WHERE figi = ? AND interval = ?
               AND ts >= ? AND ts <= ? ORDER BY ts""",
            (figi, interval, from_dt, to_dt),
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # --- Signals ---

    async def insert_signal(
        self,
        figi: str,
        ticker: str,
        direction: str,
        confidence: float,
        strategy: str,
        features: dict,
        approved: bool,
        rejection_reason: str = "",
    ) -> int:
        """Insert a signal row and return its rowid for later updates."""
        cur = await self._db.execute(
            """INSERT INTO signals (figi, ticker, direction, confidence, strategy,
               features, approved, rejection_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                figi,
                ticker,
                direction,
                confidence,
                strategy,
                json.dumps(features),
                int(approved),
                rejection_reason,
            ),
        )
        await self._db.commit()
        return int(cur.lastrowid) if cur.lastrowid is not None else 0

    async def update_signal_approval(
        self, signal_id: int, approved: bool, rejection_reason: str = ""
    ) -> None:
        """Update approval flag and rejection_reason on a signal post-hoc.

        The engine inserts every signal with approved=False because the risk
        check runs after the insert; this method patches the row once the
        decision is known so postmortem queries see ground truth.
        """
        if not signal_id:
            return
        await self._db.execute(
            "UPDATE signals SET approved = ?, rejection_reason = ? WHERE id = ?",
            (int(approved), rejection_reason, signal_id),
        )
        await self._db.commit()

    async def get_recent_signals(self, limit: int = 50) -> list[dict]:
        cur = await self._db.execute(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # --- Model Registry ---

    async def register_model(
        self,
        figi: str | None,
        model_path: str,
        version: int,
        accuracy: float,
        f1_score: float,
        train_samples: int,
        feature_names: list[str],
        tb_max_hold: int = 10,
    ) -> int:
        """Register a trained model, storing its label-horizon (tb_max_hold).

        tb_max_hold is stored so the rollback gate can detect when the horizon
        changed between retrain runs and skip the accuracy comparison (metrics
        at different horizons are not comparable — longer horizon = harder
        classification task = naturally lower accuracy).
        """
        # Deactivate previous models for this figi
        await self._db.execute(
            "UPDATE model_registry SET is_active = 0 WHERE figi IS ? OR figi = ?", (figi, figi)
        )
        cur = await self._db.execute(
            """INSERT INTO model_registry (figi, model_path, version, accuracy,
               f1_score, train_samples, feature_names, trained_at, is_active,
               tb_max_hold)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
            (
                figi,
                model_path,
                version,
                accuracy,
                f1_score,
                train_samples,
                json.dumps(feature_names),
                datetime.utcnow().isoformat(),
                tb_max_hold,
            ),
        )
        await self._db.commit()
        return cur.lastrowid

    async def get_active_model(self, figi: str | None = None) -> dict | None:
        cur = await self._db.execute(
            "SELECT * FROM model_registry WHERE (figi IS ? OR figi = ?) AND is_active = 1 ORDER BY version DESC LIMIT 1",
            (figi, figi),
        )
        row = await cur.fetchone()
        return dict(row) if row else None

    # --- Daily P&L ---

    async def upsert_daily_pnl(self, date: str, **kwargs):
        existing = await self._db.execute("SELECT id FROM daily_pnl WHERE date = ?", (date,))
        row = await existing.fetchone()
        if row:
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            vals = list(kwargs.values()) + [date]
            await self._db.execute(f"UPDATE daily_pnl SET {sets} WHERE date = ?", vals)
        else:
            cols = "date, " + ", ".join(kwargs.keys())
            placeholders = "?, " + ", ".join("?" for _ in kwargs)
            vals = [date] + list(kwargs.values())
            await self._db.execute(f"INSERT INTO daily_pnl ({cols}) VALUES ({placeholders})", vals)
        await self._db.commit()

    async def get_pnl_history(self, days: int = 30) -> list[dict]:
        cur = await self._db.execute("SELECT * FROM daily_pnl ORDER BY date DESC LIMIT ?", (days,))
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_daily_pnl_for_date(self, date: str) -> dict | None:
        """Fetch the P&L row for a specific date (YYYY-MM-DD), or None."""
        cur = await self._db.execute("SELECT * FROM daily_pnl WHERE date = ?", (date,))
        row = await cur.fetchone()
        return dict(row) if row else None

    # --- Custom Tickers ---

    async def add_custom_ticker(self, ticker: str, figi: str = "", name: str = ""):
        await self._db.execute(
            "INSERT OR REPLACE INTO custom_tickers (ticker, figi, name) VALUES (?, ?, ?)",
            (ticker.upper(), figi, name),
        )
        await self._db.commit()

    async def remove_custom_ticker(self, ticker: str):
        await self._db.execute("DELETE FROM custom_tickers WHERE ticker = ?", (ticker.upper(),))
        await self._db.commit()

    async def get_custom_tickers(self) -> list[dict]:
        cur = await self._db.execute("SELECT * FROM custom_tickers ORDER BY ticker")
        rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # --- Position peaks (trailing-stop persistence) ---

    async def upsert_position_peak(
        self,
        trade_id: int,
        peak_price: float,
        activated: bool,
        partial_taken: bool = False,
        partial_price: Optional[float] = None,
    ) -> None:
        await self._db.execute(
            """INSERT INTO position_peaks (trade_id, peak_price, activated,
                                            updated_at, partial_taken, partial_price)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
               ON CONFLICT(trade_id) DO UPDATE SET
                 peak_price = excluded.peak_price,
                 activated = excluded.activated,
                 partial_taken = excluded.partial_taken,
                 partial_price = excluded.partial_price,
                 updated_at = CURRENT_TIMESTAMP""",
            (trade_id, peak_price, int(activated), int(partial_taken), partial_price),
        )
        await self._db.commit()

    async def get_position_peaks(self) -> dict[int, dict]:
        cur = await self._db.execute(
            "SELECT trade_id, peak_price, activated, partial_taken, partial_price "
            "FROM position_peaks"
        )
        rows = await cur.fetchall()
        return {
            int(r["trade_id"]): {
                "peak_price": float(r["peak_price"]),
                "activated": bool(r["activated"]),
                "partial_taken": bool(r["partial_taken"]),
                "partial_price": (
                    float(r["partial_price"]) if r["partial_price"] is not None else None
                ),
            }
            for r in rows
        }

    async def delete_position_peak(self, trade_id: int) -> None:
        await self._db.execute("DELETE FROM position_peaks WHERE trade_id = ?", (trade_id,))
        await self._db.commit()

    # --- Cooldowns ---

    async def upsert_cooldown(self, figi: str, last_close_time: str) -> None:
        await self._db.execute(
            """INSERT INTO cooldowns (figi, last_close_time) VALUES (?, ?)
               ON CONFLICT(figi) DO UPDATE SET last_close_time = excluded.last_close_time""",
            (figi, last_close_time),
        )
        await self._db.commit()

    async def get_cooldowns(self) -> dict[str, str]:
        cur = await self._db.execute("SELECT figi, last_close_time FROM cooldowns")
        rows = await cur.fetchall()
        return {r["figi"]: r["last_close_time"] for r in rows}

    # --- Stats ---

    async def get_trade_stats(self, last_n: int = 50, direction: Optional[str] = None) -> dict:
        """Get win rate and avg win/loss from last N closed trades.

        Args:
            last_n: How many recent closed trades to use.
            direction: When set ("buy"/"sell"), restrict to same-direction
                trades.  Used by the risk manager so per-direction Kelly is
                computed from same-direction history — postmortem 2026-04-30
                showed Kelly f* = -0.62 on longs but +0.57 on shorts; pooled
                stats penalised short sizing because long losers dragged the
                average down.
        """
        if direction:
            cur = await self._db.execute(
                "SELECT pnl, pnl_pct FROM trades WHERE status = 'closed' "
                "AND pnl IS NOT NULL AND direction = ? "
                "ORDER BY exit_time DESC LIMIT ?",
                (direction, last_n),
            )
        else:
            cur = await self._db.execute(
                "SELECT pnl, pnl_pct FROM trades WHERE status = 'closed' AND pnl IS NOT NULL ORDER BY exit_time DESC LIMIT ?",
                (last_n,),
            )
        rows = await cur.fetchall()
        if not rows:
            return {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0, "total": 0, "kelly_f": 0.0}

        wins = [r["pnl_pct"] for r in rows if r["pnl"] > 0]
        losses = [abs(r["pnl_pct"]) for r in rows if r["pnl"] <= 0]

        win_rate = len(wins) / len(rows) if rows else 0.5
        avg_win = sum(wins) / len(wins) if wins else 1.0
        avg_loss = sum(losses) / len(losses) if losses else 1.0

        # Realised Kelly f* — included so callers can gate aggressive sizing
        # without re-deriving from win_rate/avg_win/avg_loss.  Negative f*
        # signals "no statistical edge" and is the trigger for the long-side
        # auto-pause downstream.
        if avg_loss > 0:
            b = avg_win / avg_loss
            kelly_f = (win_rate * b - (1 - win_rate)) / b
        else:
            kelly_f = 0.0

        return {
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total": len(rows),
            "kelly_f": kelly_f,
        }
