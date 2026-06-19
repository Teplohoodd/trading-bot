"""futbot's own SQLite schema.

Separate from trade_claude's trade_bot.db so paper / live experiments
don't pollute the main bot's trade history.  Same shape as the live
trades table where possible — that keeps postmortem / replay scripts
reusable.

Schema:
  trades              — every entry + exit (paper or live)
  decisions           — every pipeline run, including REJECTED candidates
                        (this is what makes "why didn't we trade X?"
                        answerable; in trade_claude only approved
                        candidates surface in `signals`)
  positions_state     — open-position bookkeeping (peak, trail, R-multiple)
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger("futbot.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    figi TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT NOT NULL,                -- buy / sell
    lots INTEGER NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    stop_loss REAL,
    take_profit REAL,
    initial_margin REAL,                    -- ГО per lot at entry
    rub_per_point REAL,
    entry_order_id TEXT,
    exit_order_id TEXT,
    pnl REAL,
    pnl_pct REAL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    exit_reason TEXT,
    paper INTEGER NOT NULL DEFAULT 1,       -- 1 = paper trade, 0 = real
    decision_id INTEGER,                    -- link to decisions.id
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    figi TEXT NOT NULL,
    ticker TEXT NOT NULL,
    proposed_direction TEXT,                -- buy / sell / NULL when all layers said hold
    approved INTEGER NOT NULL,              -- 1 if all layers + audit passed
    layer_trend TEXT,                       -- JSON of {vote, ema_fast, ema_slow, adx, ...}
    layer_regime TEXT,
    layer_setup TEXT,
    layer_trigger TEXT,
    layer_ml TEXT,
    layer_audit TEXT,
    rejected_at_layer TEXT,                 -- name of first layer that vetoed
    rejection_reason TEXT
);

CREATE TABLE IF NOT EXISTS positions_state (
    figi TEXT PRIMARY KEY,
    entry_time TEXT,
    direction TEXT,
    entry_price REAL,
    peak_price REAL,            -- highest high (long) / lowest low (short)
    trail_active INTEGER,       -- 0 until +1R, then 1
    initial_stop REAL,
    current_stop REAL,
    last_updated TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_figi ON trades(figi);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts);
CREATE INDEX IF NOT EXISTS idx_decisions_figi ON decisions(figi);
"""


class FutbotDB:
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
        logger.info(f"futbot DB initialized at {self.path}")

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None

    # ── Decisions ──────────────────────────────────────────────────────────
    async def insert_decision(
        self,
        *,
        figi: str,
        ticker: str,
        proposed_direction: str | None,
        approved: bool,
        layers: dict,
        rejected_at_layer: str | None,
        rejection_reason: str | None,
    ) -> int:
        """Record a full pipeline run.  Every layer's output is JSON-encoded
        so we can later answer "why didn't we trade SBRF at 14:00?"."""
        async with self._lock:
            cur = await self._db.execute(
                """INSERT INTO decisions
                   (ts, figi, ticker, proposed_direction, approved,
                    layer_trend, layer_regime, layer_setup, layer_trigger,
                    layer_ml, layer_audit,
                    rejected_at_layer, rejection_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    figi,
                    ticker,
                    proposed_direction,
                    1 if approved else 0,
                    json.dumps(layers.get("trend"), default=str),
                    json.dumps(layers.get("regime"), default=str),
                    json.dumps(layers.get("setup"), default=str),
                    json.dumps(layers.get("trigger"), default=str),
                    json.dumps(layers.get("ml"), default=str),
                    json.dumps(layers.get("audit"), default=str),
                    rejected_at_layer,
                    rejection_reason,
                ),
            )
            await self._db.commit()
            return cur.lastrowid

    # ── Trades ─────────────────────────────────────────────────────────────
    async def insert_trade(
        self,
        *,
        figi: str,
        ticker: str,
        direction: str,
        lots: int,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        initial_margin: float | None,
        rub_per_point: float | None,
        paper: bool,
        decision_id: int | None,
        entry_order_id: str | None,
    ) -> int:
        async with self._lock:
            cur = await self._db.execute(
                """INSERT INTO trades
                   (figi, ticker, direction, lots, entry_price,
                    stop_loss, take_profit, initial_margin, rub_per_point,
                    entry_time, entry_order_id, paper, decision_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    figi,
                    ticker,
                    direction,
                    lots,
                    entry_price,
                    stop_loss,
                    take_profit,
                    initial_margin,
                    rub_per_point,
                    datetime.utcnow().isoformat(),
                    entry_order_id,
                    1 if paper else 0,
                    decision_id,
                ),
            )
            await self._db.commit()
            return cur.lastrowid

    async def close_trade(
        self,
        trade_id: int,
        *,
        exit_price: float,
        exit_reason: str,
        pnl: float,
        pnl_pct: float,
        exit_order_id: str | None = None,
    ) -> None:
        async with self._lock:
            await self._db.execute(
                """UPDATE trades
                   SET exit_price=?, exit_time=?, exit_reason=?,
                       pnl=?, pnl_pct=?, exit_order_id=?
                   WHERE id=?""",
                (
                    exit_price,
                    datetime.utcnow().isoformat(),
                    exit_reason,
                    pnl,
                    pnl_pct,
                    exit_order_id,
                    trade_id,
                ),
            )
            await self._db.commit()

    async def open_trades(self) -> list[aiosqlite.Row]:
        async with self._lock:
            cur = await self._db.execute(
                "SELECT * FROM trades WHERE exit_time IS NULL ORDER BY entry_time"
            )
            return await cur.fetchall()

    async def open_trade_for_figi(self, figi: str) -> aiosqlite.Row | None:
        async with self._lock:
            cur = await self._db.execute(
                "SELECT * FROM trades WHERE figi=? AND exit_time IS NULL LIMIT 1",
                (figi,),
            )
            return await cur.fetchone()

    async def daily_pnl(self, date_iso: str) -> float:
        """Sum of realised P&L for trades closed on a given UTC date (YYYY-MM-DD)."""
        async with self._lock:
            cur = await self._db.execute(
                "SELECT COALESCE(SUM(pnl),0) FROM trades "
                "WHERE date(exit_time)=? AND pnl IS NOT NULL",
                (date_iso,),
            )
            row = await cur.fetchone()
            return float(row[0]) if row else 0.0

    # ── Positions state ────────────────────────────────────────────────────
    async def upsert_position_state(self, *, figi: str, **fields):
        cols = ["figi"] + list(fields.keys())
        placeholders = ", ".join("?" * len(cols))
        updates = ", ".join(f"{k}=excluded.{k}" for k in fields)
        async with self._lock:
            await self._db.execute(
                f"""INSERT INTO positions_state ({','.join(cols)})
                    VALUES ({placeholders})
                    ON CONFLICT(figi) DO UPDATE SET {updates},
                                                    last_updated=excluded.last_updated""",
                [figi] + [v for v in fields.values()],
            )
            await self._db.commit()

    async def get_position_state(self, figi: str) -> aiosqlite.Row | None:
        async with self._lock:
            cur = await self._db.execute("SELECT * FROM positions_state WHERE figi=?", (figi,))
            return await cur.fetchone()

    async def delete_position_state(self, figi: str):
        async with self._lock:
            await self._db.execute("DELETE FROM positions_state WHERE figi=?", (figi,))
            await self._db.commit()
