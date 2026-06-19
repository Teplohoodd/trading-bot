"""Pair-trade DB schema.  Separate from scalp/futbot DBs."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS pair_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pair TEXT NOT NULL,                  -- "LK-Si"
    base_y TEXT NOT NULL,                -- "LK"
    base_x TEXT NOT NULL,                -- "Si"
    figi_y TEXT NOT NULL,
    figi_x TEXT NOT NULL,
    direction INTEGER NOT NULL,          -- +1 (long-y short-x) or -1
    lots_y INTEGER NOT NULL,
    lots_x INTEGER NOT NULL,
    beta REAL NOT NULL,                  -- regression coef at entry
    entry_y_price REAL NOT NULL,
    entry_x_price REAL NOT NULL,
    entry_z REAL NOT NULL,
    entry_time TEXT NOT NULL,
    exit_y_price REAL,
    exit_x_price REAL,
    exit_z REAL,
    exit_time TEXT,
    exit_reason TEXT,                    -- mean_rev / horizon / stop / boot_reconcile
    spread_entry REAL,                   -- y_entry - β·x_entry
    spread_exit REAL,
    pnl REAL,                            -- in % of combined notional
    pnl_rub REAL,                        -- approx ruble P&L
    commission_rub REAL,
    paper INTEGER NOT NULL DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS pair_state (
    pair TEXT PRIMARY KEY,
    beta REAL,
    alpha REAL,
    adf_p REAL,
    spread_mean REAL,
    spread_std REAL,
    last_refit TEXT
);

CREATE INDEX IF NOT EXISTS idx_pair_trades_pair ON pair_trades(pair);
CREATE INDEX IF NOT EXISTS idx_pair_trades_entry ON pair_trades(entry_time);
"""


class PairsDB:
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

    # ── pair_trades ────────────────────────────────────────────────────
    async def insert_trade(self, **fields) -> int:
        cols = list(fields.keys())
        async with self._lock:
            cur = await self._db.execute(
                f"INSERT INTO pair_trades ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                [fields[c] for c in cols],
            )
            await self._db.commit()
            return cur.lastrowid

    async def close_trade(self, trade_id: int, **fields):
        sets = ", ".join(f"{k}=?" for k in fields)
        async with self._lock:
            await self._db.execute(
                f"UPDATE pair_trades SET {sets} WHERE id=?",
                [*fields.values(), trade_id],
            )
            await self._db.commit()

    async def open_trades(self) -> list[aiosqlite.Row]:
        async with self._lock:
            cur = await self._db.execute(
                "SELECT * FROM pair_trades WHERE exit_time IS NULL " "ORDER BY entry_time"
            )
            return await cur.fetchall()

    async def open_trade_for_pair(self, pair: str) -> aiosqlite.Row | None:
        async with self._lock:
            cur = await self._db.execute(
                "SELECT * FROM pair_trades WHERE pair=? AND exit_time IS NULL",
                (pair,),
            )
            return await cur.fetchone()

    async def daily_pnl_rub(self, date_iso: str) -> float:
        """Sum of NET P&L for trades closed on `date_iso`.

        Excludes trades tagged `*_OVERLEV` or `sizing_bug_*` — these were
        opened by the pre-2026-05-27 over-leveraged sizing bug and their
        P&L numbers don't reflect what the (correctly sized) strategy
        would have earned.  Keeping the rows in DB for audit but filtering
        from displays.
        """
        async with self._lock:
            cur = await self._db.execute(
                "SELECT COALESCE(SUM(pnl_rub), 0) FROM pair_trades "
                "WHERE date(exit_time)=? AND pnl_rub IS NOT NULL "
                "AND (exit_reason IS NULL "
                "     OR (exit_reason NOT LIKE '%_OVERLEV' "
                "         AND exit_reason NOT LIKE 'sizing_bug_%'))",
                (date_iso,),
            )
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0

    # ── pair_state (β cache) ───────────────────────────────────────────
    async def upsert_pair_state(
        self,
        *,
        pair: str,
        beta: float,
        alpha: float,
        adf_p: float,
        spread_mean: float,
        spread_std: float,
    ):
        async with self._lock:
            await self._db.execute(
                """INSERT INTO pair_state (pair, beta, alpha, adf_p,
                       spread_mean, spread_std, last_refit)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(pair) DO UPDATE SET
                       beta=excluded.beta, alpha=excluded.alpha,
                       adf_p=excluded.adf_p,
                       spread_mean=excluded.spread_mean,
                       spread_std=excluded.spread_std,
                       last_refit=excluded.last_refit""",
                (pair, beta, alpha, adf_p, spread_mean, spread_std, datetime.utcnow().isoformat()),
            )
            await self._db.commit()

    async def get_pair_state(self, pair: str) -> aiosqlite.Row | None:
        async with self._lock:
            cur = await self._db.execute("SELECT * FROM pair_state WHERE pair=?", (pair,))
            return await cur.fetchone()
