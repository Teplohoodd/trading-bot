"""Trend-bot DB schema.

trend_trades — every closed and open trend trade
trend_pos    — quick lookup of open positions (mirrors trend_trades.exit_time IS NULL)
"""

import asyncio
from datetime import datetime
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS trend_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    base TEXT NOT NULL,                    -- "GD"
    ticker TEXT NOT NULL,                  -- "GDM6"
    figi TEXT NOT NULL,
    direction TEXT NOT NULL,               -- buy / sell
    lots INTEGER NOT NULL,
    bb_n INTEGER NOT NULL,                 -- Bollinger lookback used (legacy)
    bb_k REAL NOT NULL,                    -- Bollinger σ multiplier (legacy)
    entry_price REAL NOT NULL,
    entry_upper REAL,                      -- band values at entry (debug)
    entry_lower REAL,
    entry_time TEXT NOT NULL,
    entry_order_id TEXT,
    exit_price REAL,
    exit_time TEXT,
    exit_reason TEXT,                      -- band_flip / pattern_stop / pattern_target / pattern_timeout / rollover / boot_reconcile / stop_kill / manual
    exit_order_id TEXT,
    pnl REAL,                              -- NET in RUB after commission
    pnl_pct REAL,                          -- % of notional at entry
    commission_rub REAL,
    rub_per_point REAL,
    lot_size INTEGER,
    paper INTEGER NOT NULL DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    -- Pattern-bot extension (NULL for legacy Bollinger entries) ─────────
    pattern_name TEXT,                     -- triple_top / triple_bottom / ...
    stop_price REAL,                       -- pattern invalidation level
    target_price REAL,                     -- measured-move target
    pattern_height_pct REAL                -- pattern height / entry price
);

CREATE INDEX IF NOT EXISTS idx_trend_trades_base ON trend_trades(base);
CREATE INDEX IF NOT EXISTS idx_trend_trades_entry_time ON trend_trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trend_trades_open ON trend_trades(exit_time);
"""

# Migration: ALTER TABLE for pre-existing DBs.  SQLite raises if column
# already exists, so we swallow that specific error.
_MIGRATIONS = [
    "ALTER TABLE trend_trades ADD COLUMN pattern_name TEXT",
    "ALTER TABLE trend_trades ADD COLUMN stop_price REAL",
    "ALTER TABLE trend_trades ADD COLUMN target_price REAL",
    "ALTER TABLE trend_trades ADD COLUMN pattern_height_pct REAL",
    # Exchange protective stop-loss order id (live).  Cancelled on any close.
    "ALTER TABLE trend_trades ADD COLUMN protective_stop_id TEXT",
]


class TrendDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        # Apply migrations (idempotent: ignore "duplicate column" errors)
        for stmt in _MIGRATIONS:
            try:
                await self._db.execute(stmt)
            except Exception as e:
                if "duplicate column" not in str(e).lower():
                    raise
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    async def insert_trade(self, **fields) -> int:
        cols = list(fields.keys())
        async with self._lock:
            cur = await self._db.execute(
                f"INSERT INTO trend_trades ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                [fields[c] for c in cols],
            )
            await self._db.commit()
            return cur.lastrowid

    async def close_trade(self, trade_id: int, **fields):
        sets = ", ".join(f"{k}=?" for k in fields)
        async with self._lock:
            await self._db.execute(
                f"UPDATE trend_trades SET {sets} WHERE id=?",
                [*fields.values(), trade_id],
            )
            await self._db.commit()

    async def open_trades(self) -> list[aiosqlite.Row]:
        async with self._lock:
            cur = await self._db.execute(
                "SELECT * FROM trend_trades WHERE exit_time IS NULL " "ORDER BY entry_time"
            )
            return await cur.fetchall()

    async def open_trade_for_base(self, base: str) -> aiosqlite.Row | None:
        async with self._lock:
            cur = await self._db.execute(
                "SELECT * FROM trend_trades WHERE base=? AND exit_time IS NULL",
                (base,),
            )
            return await cur.fetchone()

    async def daily_pnl_rub(self, date_iso: str) -> float:
        async with self._lock:
            cur = await self._db.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trend_trades "
                "WHERE date(exit_time)=? AND pnl IS NOT NULL",
                (date_iso,),
            )
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0

    async def contract_daily_pnl_rub(self, base: str, date_iso: str) -> float:
        async with self._lock:
            cur = await self._db.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trend_trades "
                "WHERE base=? AND date(exit_time)=? AND pnl IS NOT NULL",
                (base, date_iso),
            )
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0
