"""Scalp DB — separate from futbot.db so frequent trades don't bloat that one."""

import asyncio
from datetime import datetime
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS scalp_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    figi TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,           -- buy/sell
    lots INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    stop_loss REAL,
    take_profit REAL,
    score REAL,
    components TEXT,                   -- JSON snapshot of signal components
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    exit_reason TEXT,
    pnl REAL,
    pnl_pct REAL,
    paper INTEGER NOT NULL DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_scalp_figi ON scalp_trades(figi);
CREATE INDEX IF NOT EXISTS idx_scalp_entry_time ON scalp_trades(entry_time);
"""


class ScalpDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    async def insert_trade(
        self,
        *,
        figi,
        ticker,
        direction,
        lots,
        entry_price,
        stop_loss,
        take_profit,
        score,
        components_json,
        paper
    ) -> int:
        async with self._lock:
            cur = await self._db.execute(
                """INSERT INTO scalp_trades
                   (figi, ticker, direction, lots, entry_price,
                    stop_loss, take_profit, score, components,
                    entry_time, paper)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    figi,
                    ticker,
                    direction,
                    lots,
                    entry_price,
                    stop_loss,
                    take_profit,
                    score,
                    components_json,
                    datetime.utcnow().isoformat(),
                    1 if paper else 0,
                ),
            )
            await self._db.commit()
            return cur.lastrowid

    async def close_trade(self, trade_id: int, *, exit_price, exit_reason, pnl, pnl_pct):
        async with self._lock:
            await self._db.execute(
                """UPDATE scalp_trades
                   SET exit_price=?, exit_time=?, exit_reason=?,
                       pnl=?, pnl_pct=?
                   WHERE id=?""",
                (exit_price, datetime.utcnow().isoformat(), exit_reason, pnl, pnl_pct, trade_id),
            )
            await self._db.commit()

    async def open_trades(self) -> list[aiosqlite.Row]:
        async with self._lock:
            cur = await self._db.execute("SELECT * FROM scalp_trades WHERE exit_time IS NULL")
            return await cur.fetchall()

    async def daily_pnl(self, date_iso: str) -> float:
        async with self._lock:
            cur = await self._db.execute(
                "SELECT COALESCE(SUM(pnl),0) FROM scalp_trades "
                "WHERE date(exit_time)=? AND pnl IS NOT NULL",
                (date_iso,),
            )
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0

    async def trades_today(self, date_iso: str) -> int:
        async with self._lock:
            cur = await self._db.execute(
                "SELECT COUNT(*) FROM scalp_trades WHERE date(entry_time)=?",
                (date_iso,),
            )
            row = await cur.fetchone()
            return int(row[0]) if row else 0
