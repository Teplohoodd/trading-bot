"""Tick recorder — saves GetLastTrades + order-book snapshots to SQLite
so we accumulate REAL microstructure history.

Why this exists:
  T-Invest API doesn't provide historical L2 or tick data older than
  the last hour.  To backtest scalp v2 honestly we need to GATHER our
  own microstructure history.  This recorder polls `get_last_trades`
  every TICK_POLL_SECONDS and writes the events to disk.

What we DO get per tick:
  price, quantity, direction (buyer/seller-aggressor), exchange timestamp.
  These are sufficient for computing TFI, CVD, and signed-volume features.

What we DON'T get (current SDK):
  per-tick open_interest.  The `with_open_interest` flag exists on
  SubscribeTradesRequest (streaming) but isn't on the REST get_last_trades
  endpoint, and Trade objects from REST don't carry an OI field.
  → The `open_interest` column is left NULL for REST-recorded ticks.
  → Periodic OI snapshots could be added via `get_market_values()`
    in a future iteration; for backtesting CVD/TFI we don't need OI.

Schema:
  ticks(figi, ticker, ts, price, qty, direction, open_interest, recorded_at)
  book_snapshots(figi, ticker, ts, bids_json, asks_json, recorded_at)

After ~1 week of running on the scalp universe, ~50-200k tick events per
contract — enough for proper microstructure backtesting replacing the
candle-proxy version.

Usage:
  python -m futbot.scripts.tick_recorder
  python -m futbot.scripts.tick_recorder BR GZ LK
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import Settings
from core.broker import BrokerClient
from tinkoff.invest import AsyncClient
from tinkoff.invest.utils import quotation_to_decimal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("tinkoff").setLevel(logging.WARNING)
logger = logging.getLogger("tick_rec")


DEFAULT_BASES = ["BR", "GZ", "LK", "GD"]
TICK_POLL_SECONDS = 30  # fetch last-trades every 30s; overlap is OK
BOOK_POLL_SECONDS = 60  # book snapshot every minute
DB_PATH = Path("data/ticks.db")


SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    figi TEXT NOT NULL,
    ticker TEXT NOT NULL,
    ts TEXT NOT NULL,                  -- ISO UTC when trade happened
    price REAL NOT NULL,
    qty INTEGER NOT NULL,
    direction INTEGER NOT NULL,        -- +1 buyer-aggressor, -1 seller-agg
    open_interest INTEGER,             -- contracts outstanding (futures)
    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(figi, ts, price, qty)
);
CREATE INDEX IF NOT EXISTS idx_ticks_figi_ts ON ticks(figi, ts);

CREATE TABLE IF NOT EXISTS book_snapshots (
    figi TEXT NOT NULL,
    ticker TEXT NOT NULL,
    ts TEXT NOT NULL,
    bids_json TEXT NOT NULL,           -- JSON [[price, qty], ...] top-N
    asks_json TEXT NOT NULL,
    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_book_figi_ts ON book_snapshots(figi, ts);
"""


async def _init_db() -> aiosqlite.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = await aiosqlite.connect(DB_PATH)
    await db.executescript(SCHEMA)
    await db.commit()
    return db


async def _resolve_front_month(broker, base: str):
    futs = await broker.get_all_futures()
    cands = []
    for f in futs:
        t = getattr(f, "ticker", "") or ""
        if t == base or (t.startswith(base) and len(t) == len(base) + 2):
            exp = getattr(f, "expiration_date", None)
            if exp is None:
                continue
            if hasattr(exp, "ToDatetime"):
                exp = exp.ToDatetime()
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            cands.append((f, exp))
    if not cands:
        return None
    cands.sort(key=lambda x: x[1])
    now = datetime.now(timezone.utc)
    for f, exp in cands:
        if (exp - now).days >= 14:
            return f
    return cands[0][0]


async def _record_trades(client, db, figi: str, ticker: str):
    """Fetch the last hour of anonymous trades and insert any we haven't seen."""
    from tinkoff.invest import TradeDirection

    try:
        now = datetime.now(timezone.utc)
        # Note: SDK doesn't expose TradeSourceType — default source is fine.
        # GetLastTrades API guarantees last hour of anonymous trades.
        resp = await client.market_data.get_last_trades(
            instrument_id=figi,
            from_=now - timedelta(minutes=2),  # 2 min window; we poll every 30s
            to=now,
        )
    except Exception as e:
        logger.warning(f"  {ticker}: get_last_trades failed: {e}")
        return 0

    inserted = 0
    for t in resp.trades:
        try:
            ts = t.time.isoformat() if hasattr(t.time, "isoformat") else str(t.time)
            price = float(quotation_to_decimal(t.price))
            qty = int(t.quantity)
            direction = +1 if t.direction == TradeDirection.TRADE_DIRECTION_BUY else -1
            # OI may be in `open_interest` attribute when requested
            oi = getattr(t, "open_interest", None)
            oi_val = int(oi) if oi is not None else None
            await db.execute(
                "INSERT OR IGNORE INTO ticks "
                "(figi, ticker, ts, price, qty, direction, open_interest) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (figi, ticker, ts, price, qty, direction, oi_val),
            )
            inserted += 1
        except Exception as e:
            logger.debug(f"  insert trade failed: {e}")
    await db.commit()
    return inserted


async def _record_book(client, db, figi: str, ticker: str, depth: int = 10):
    try:
        ob = await client.market_data.get_order_book(instrument_id=figi, depth=depth)
    except Exception as e:
        logger.warning(f"  {ticker}: get_order_book failed: {e}")
        return False
    try:
        bids = [[float(quotation_to_decimal(b.price)), int(b.quantity)] for b in (ob.bids or [])]
        asks = [[float(quotation_to_decimal(a.price)), int(a.quantity)] for a in (ob.asks or [])]
        await db.execute(
            "INSERT INTO book_snapshots (figi, ticker, ts, bids_json, asks_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (figi, ticker, datetime.utcnow().isoformat(), json.dumps(bids), json.dumps(asks)),
        )
        await db.commit()
        return True
    except Exception as e:
        logger.warning(f"  {ticker}: book insert failed: {e}")
        return False


async def main():
    bases = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_BASES
    settings = Settings()
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="tick-recorder",
    )
    await broker.connect()

    universe = []
    for b in bases:
        f = await _resolve_front_month(broker, b)
        if f is None:
            logger.warning(f"  {b}: no contract")
            continue
        universe.append((b, f.figi, f.ticker))
        logger.info(f"  {b}: tracking {f.ticker} ({f.figi})")
    await broker.disconnect()

    if not universe:
        logger.error("No contracts to record — exiting")
        return

    db = await _init_db()
    logger.info(f"DB: {DB_PATH}  contracts: {[t for _, _, t in universe]}")
    logger.info(f"Polling every {TICK_POLL_SECONDS}s (trades), {BOOK_POLL_SECONDS}s (book)")

    shutdown = asyncio.Event()

    def _sig(*_):
        logger.info("Shutdown received")
        shutdown.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(s, _sig)
        except NotImplementedError:
            pass

    async def trade_loop():
        async with AsyncClient(settings.T_INVEST_TOKEN, app_name="tick-rec") as client:
            while not shutdown.is_set():
                t0 = datetime.now(timezone.utc)
                total = 0
                for base, figi, ticker in universe:
                    n = await _record_trades(client, db, figi, ticker)
                    total += n
                logger.info(f"  +{total} new ticks")
                try:
                    await asyncio.wait_for(shutdown.wait(), timeout=TICK_POLL_SECONDS)
                except asyncio.TimeoutError:
                    pass

    async def book_loop():
        async with AsyncClient(settings.T_INVEST_TOKEN, app_name="tick-rec-book") as client:
            while not shutdown.is_set():
                snaps = 0
                for base, figi, ticker in universe:
                    if await _record_book(client, db, figi, ticker):
                        snaps += 1
                logger.info(f"  +{snaps} book snapshots")
                try:
                    await asyncio.wait_for(shutdown.wait(), timeout=BOOK_POLL_SECONDS)
                except asyncio.TimeoutError:
                    pass

    try:
        await asyncio.gather(trade_loop(), book_loop())
    finally:
        logger.info("Closing DB…")
        # Quick stats
        async with db.execute("SELECT COUNT(*) FROM ticks") as cur:
            r = await cur.fetchone()
            logger.info(f"  Total ticks recorded: {r[0]}")
        async with db.execute("SELECT COUNT(*) FROM book_snapshots") as cur:
            r = await cur.fetchone()
            logger.info(f"  Total book snapshots: {r[0]}")
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
