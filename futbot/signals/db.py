"""Signal log — every channel message, its interpretation, proposal, outcome.

Lets us build HONEST forward statistics on the "📈" channel: how often the
calls (correctly interpreted) actually win on Neo assets, going forward — not
the cherry-picked screenshots.
"""

import asyncio
import json
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    msg_id INTEGER,                 -- telegram message id (dedup)
    msg_time TEXT,
    raw_text TEXT,
    instrument TEXT,                -- BTC/ETH/SOL
    neo_ticker TEXT,
    direction INTEGER,              -- +1/-1/0
    actionable INTEGER,
    confidence TEXT,                -- high/low
    interpreted_by TEXT,            -- heuristic / llm / human
    status TEXT DEFAULT 'new',      -- new / proposed / approved / rejected / opened / closed / skipped
    proposed_entry REAL,
    proposed_lots INTEGER,
    fill_price REAL,
    exit_price REAL,
    pnl_rub REAL,
    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_msg ON signals(msg_id);
"""


class SignalDB:
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

    async def seen(self, msg_id: int) -> bool:
        async with self._lock:
            cur = await self._db.execute("SELECT 1 FROM signals WHERE msg_id=?", (msg_id,))
            return (await cur.fetchone()) is not None

    async def log(self, **fields) -> int:
        cols = list(fields.keys())
        async with self._lock:
            try:
                cur = await self._db.execute(
                    f"INSERT INTO signals ({','.join(cols)}) "
                    f"VALUES ({','.join('?'*len(cols))})",
                    [fields[c] for c in cols],
                )
                await self._db.commit()
                return cur.lastrowid
            except Exception:
                return -1  # duplicate msg_id

    async def update(self, sig_id: int, **fields):
        sets = ", ".join(f"{k}=?" for k in fields)
        async with self._lock:
            await self._db.execute(
                f"UPDATE signals SET {sets} WHERE id=?", [*fields.values(), sig_id]
            )
            await self._db.commit()

    async def recent(self, n: int = 20) -> list[aiosqlite.Row]:
        async with self._lock:
            cur = await self._db.execute("SELECT * FROM signals ORDER BY id DESC LIMIT ?", (n,))
            return await cur.fetchall()

    async def stats(self) -> dict:
        async with self._lock:
            cur = await self._db.execute(
                "SELECT COUNT(*) n, "
                "SUM(CASE WHEN pnl_rub>0 THEN 1 ELSE 0 END) wins, "
                "COALESCE(SUM(pnl_rub),0) total "
                "FROM signals WHERE status='closed' AND pnl_rub IS NOT NULL"
            )
            r = await cur.fetchone()
        return {"closed": r[0] or 0, "wins": r[1] or 0, "total_rub": r[2] or 0.0}
