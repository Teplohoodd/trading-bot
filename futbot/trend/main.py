"""futbot.trend.main — multi-contract Bollinger breakout bot.

Hourly evaluation loop:
  1. Refresh contracts (front-month resolution every 6h).
  2. Daily kill check.
  3. For each contract in portfolio:
       * If no position open: fetch hourly candles, compute bands,
         place entry if breakout printed.
       * If position open: check for band-flip OR rollover deadline.
  4. Telegram alert on every open/close.

Paper mode by default.  Live requires explicit env flag.
"""

import asyncio
import json
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.broker import BrokerClient  # noqa: E402
from tinkoff.invest import CandleInterval, OrderDirection  # noqa: E402
from tinkoff.invest.utils import quotation_to_decimal  # noqa: E402

from futbot.trend.config import TrendSettings  # noqa: E402
from futbot.trend.db import TrendDB  # noqa: E402
from futbot.trend import strategy as strat  # noqa: E402
from futbot.trend.portfolio import PORTFOLIO, core_portfolio, TrendEntry  # noqa: E402
from futbot.utils import commissions as comm  # noqa: E402
from futbot.telegram_notifier import (  # noqa: E402
    TelegramNotifier,
    Msg,
    MsgType,
)


# ─────────────────────────────────────────────────────────────────────────────
# Resolved-contract bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ResolvedContract:
    base: str
    ticker: str
    figi: str
    instrument: object
    expiration: datetime
    rpp: float
    lot_size: int
    bb_n: int
    bb_k: float

    @property
    def days_to_expiry(self) -> int:
        return max(0, (self.expiration - datetime.now(timezone.utc)).days)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def setup_logging(settings: TrendSettings):
    settings.TREND_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.TREND_LOG_PATH, encoding="utf-8"),
        ],
    )
    for n in ("httpx", "telegram", "tinkoff", "grpc"):
        logging.getLogger(n).setLevel(logging.WARNING)


async def _resolve_front_month(broker, base: str, min_dte: int) -> tuple | None:
    """Return (instrument, expiration_dt) for base's front-month, or None.

    Handles weird Tinkoff edge cases: tickers that don't follow the
    base+monthcode+year convention (e.g. EURRUBF, USDRUBF) are matched
    by exact base equality with the full ticker.
    """
    futs = await broker.get_all_futures()
    cands = []
    now = datetime.now(timezone.utc)
    for f in futs:
        t = getattr(f, "ticker", "") or ""
        if len(t) < 3:
            continue
        # Either exact match or "base + 2 chars" suffix
        is_match = t == base or (t.startswith(base) and len(t) == len(base) + 2)
        if not is_match:
            continue
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
    for f, exp in cands:
        if (exp - now).days >= min_dte:
            return f, exp
    return cands[0]  # accept near-expiry as last resort


async def _resolve_portfolio(
    broker, entries: list[TrendEntry], min_dte: int
) -> list[ResolvedContract]:
    out = []
    for e in entries:
        try:
            r = await _resolve_front_month(broker, e.base, min_dte)
        except Exception as ex:
            logging.getLogger("futbot.trend").warning(f"  {e.base}: resolve failed ({ex})")
            continue
        if r is None:
            logging.getLogger("futbot.trend").warning(
                f"  {e.base}: no front-month with ≥{min_dte}d to expiry"
            )
            continue
        f, exp = r
        meta = broker.extract_futures_metadata(f)
        out.append(
            ResolvedContract(
                base=e.base,
                ticker=f.ticker,
                figi=f.figi,
                instrument=f,
                expiration=exp,
                rpp=float(meta.get("rub_per_point") or 1.0),
                lot_size=int(getattr(f, "lot", 1) or 1),
                bb_n=e.n,
                bb_k=e.k,
            )
        )
    return out


async def _fetch_hourly(broker, figi: str, days: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    try:
        candles = await broker.get_candles(
            figi,
            now - timedelta(days=days),
            now,
            interval=CandleInterval.CANDLE_INTERVAL_HOUR,
        )
    except Exception as e:
        logging.getLogger("futbot.trend").warning(f"  {figi}: fetch failed ({e})")
        return pd.DataFrame()
    rows = [
        {
            "time": c.time,
            "close": float(quotation_to_decimal(c.close)),
        }
        for c in candles
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


async def _place_market(
    broker, figi: str, direction: str, lots: int, paper: bool
) -> tuple[str, float]:
    last_p = float(await broker.get_last_price(figi))
    if paper:
        import time

        oid = f"paper-trend-{direction}-{figi[-6:]}-{int(time.time())}"
        return oid, last_p
    side = (
        OrderDirection.ORDER_DIRECTION_BUY
        if direction == "buy"
        else OrderDirection.ORDER_DIRECTION_SELL
    )
    resp = await broker.post_market_order(figi, lots, side)
    oid = getattr(resp, "order_id", "?")
    return oid, last_p


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    settings = TrendSettings()
    setup_logging(settings)
    logger = logging.getLogger("futbot.trend")

    mode = "PAPER" if settings.TREND_PAPER_MODE else "LIVE"
    logger.info(f"trend bot starting in {mode} mode")
    if not settings.TREND_PAPER_MODE:
        logger.warning("LIVE mode is ON. Stop now (Ctrl-C) if unintended.")

    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="futbot-trend",
    )
    await broker.connect()

    db = TrendDB(settings.TREND_DB_PATH)
    await db.initialize()

    notifier = TelegramNotifier(
        token=settings.TELEGRAM_BOT_TOKEN,
        chat_id=settings.TELEGRAM_CHAT_ID,
        paper=bool(settings.TREND_PAPER_MODE),
    )
    await notifier.start()

    # ── Resolve portfolio ─────────────────────────────────────────────
    entries = core_portfolio() if settings.TREND_UNIVERSE_MODE == "core" else PORTFOLIO
    logger.info(f"Universe mode: {settings.TREND_UNIVERSE_MODE} → {len(entries)} contracts")
    contracts = await _resolve_portfolio(broker, entries, settings.TREND_MIN_DAYS_TO_EXPIRY)
    if not contracts:
        logger.error("No contracts resolved — exiting")
        return
    by_base = {c.base: c for c in contracts}
    logger.info(
        f"Resolved {len(contracts)} contracts:\n  "
        + "\n  ".join(
            f"{c.base:<10} {c.ticker:<10} dte={c.days_to_expiry:>3}  " f"N={c.bb_n:<3} k={c.bb_k}"
            for c in contracts
        )
    )

    # ── Portfolio value ──────────────────────────────────────────────
    try:
        portfolio_value = float(await broker.get_portfolio_value())
    except Exception:
        portfolio_value = 100_000.0
    logger.info(f"Portfolio value: {portfolio_value:.0f} ₽")

    # ── Reconcile open trades from DB ────────────────────────────────
    open_rows = await db.open_trades()
    reconciled = 0
    force_closed = 0
    for r in open_rows:
        base = r["base"]
        contract = by_base.get(base)
        if contract is None:
            # Contract no longer in universe → force close at last price
            try:
                fill = float(await broker.get_last_price(r["figi"]))
            except Exception:
                fill = float(r["entry_price"])
            pnl, pnl_pct, _, comm_rub = comm.round_trip_pnl(
                direction=r["direction"],
                entry_price=r["entry_price"],
                exit_price=fill,
                lots=r["lots"],
                lot_size=int(r["lot_size"] or 1),
                rub_per_point=float(r["rub_per_point"] or 1.0),
                instrument_kind="future",
                base_ticker=base,
            )
            await db.close_trade(
                r["id"],
                exit_price=fill,
                exit_time=datetime.utcnow().isoformat(),
                exit_reason="boot_reconcile_unknown",
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission_rub=comm_rub,
            )
            logger.warning(
                f"  RECONCILE {base} (out of universe) — force-closed @ {fill:.4f}  pnl={pnl:+.2f}₽"
            )
            force_closed += 1
        else:
            # Re-hydrated to live tracking — actual position-state lives in DB,
            # main loop will read open trade row on next tick.
            logger.info(
                f"  RECONCILE {base} {r['direction']} entered "
                f"{r['entry_time'][:19]} — re-hydrated"
            )
            reconciled += 1

    if open_rows:
        logger.info(
            f"Reconcile finished: {reconciled} re-hydrated, " f"{force_closed} force-closed"
        )

    # ── Shutdown plumbing ───────────────────────────────────────────
    shutdown = asyncio.Event()

    def _sig(*_):
        logger.info("Shutdown received")
        shutdown.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(s, _sig)
        except NotImplementedError:
            pass

    notifier.push(
        Msg(
            MsgType.BOOT,
            f"Trend bot — {mode}",
            f"Universe: {len(contracts)} contracts (mode={settings.TREND_UNIVERSE_MODE})\n"
            f"Strategy: Bollinger band-flip on hourly bars\n"
            f"Reconciled: {reconciled} open, {force_closed} force-closed",
        )
    )

    # ── Main loop ───────────────────────────────────────────────────
    last_universe_refresh = datetime.now(timezone.utc)
    try:
        while not shutdown.is_set():
            t_start = datetime.now(timezone.utc)
            logger.info(f"── Tick {t_start.isoformat()[:19]} ──")

            # Refresh universe every 6h (handles rollovers)
            if (t_start - last_universe_refresh).total_seconds() > 6 * 3600:
                logger.info("Refreshing portfolio resolution…")
                contracts = await _resolve_portfolio(
                    broker,
                    entries,
                    settings.TREND_MIN_DAYS_TO_EXPIRY,
                )
                by_base = {c.base: c for c in contracts}
                last_universe_refresh = t_start

            # Daily kill check
            today_iso = t_start.date().isoformat()
            daily_pnl = await db.daily_pnl_rub(today_iso)
            loss_cap = portfolio_value * float(settings.TREND_DAILY_LOSS_PCT_LIMIT)
            kill_active = daily_pnl < -loss_cap
            if kill_active:
                logger.warning(
                    f"Daily kill active: P&L {daily_pnl:+.0f}₽ < -{loss_cap:.0f}₽ "
                    f"— no new entries"
                )

            # Count open positions for max-open cap
            open_now = await db.open_trades()
            n_open = len(open_now)

            # Walk every contract
            for c in contracts:
                # ── Check for rollover (close before expiry) ──
                open_t = await db.open_trade_for_base(c.base)
                if open_t is not None and c.days_to_expiry <= int(settings.TREND_ROLLOVER_DAYS):
                    fill_price = float(await broker.get_last_price(c.figi))
                    direction = open_t["direction"]
                    # Place opposite leg to close
                    exit_dir = "sell" if direction == "buy" else "buy"
                    oid, _ = await _place_market(
                        broker,
                        c.figi,
                        exit_dir,
                        open_t["lots"],
                        paper=bool(settings.TREND_PAPER_MODE),
                    )
                    pnl, pnl_pct, _, comm_rub = comm.round_trip_pnl(
                        direction=direction,
                        entry_price=open_t["entry_price"],
                        exit_price=fill_price,
                        lots=open_t["lots"],
                        lot_size=c.lot_size,
                        rub_per_point=c.rpp,
                        instrument_kind="future",
                        base_ticker=c.base,
                    )
                    await db.close_trade(
                        open_t["id"],
                        exit_price=fill_price,
                        exit_time=datetime.utcnow().isoformat(),
                        exit_reason="rollover",
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        commission_rub=comm_rub,
                        exit_order_id=oid,
                    )
                    logger.info(
                        f"  {c.base}: ROLLOVER close @ {fill_price:.4f} "
                        f"({c.days_to_expiry}d to expiry) pnl={pnl:+.2f}₽"
                    )
                    notifier.push(
                        Msg(
                            MsgType.TRADE_CLOSED,
                            f"TREND {c.base} ROLLOVER",
                            f"closed @ {fill_price:.4f} ({c.days_to_expiry}d to expiry)\n"
                            f"P&L: <b>{pnl:+.2f} ₽</b> ({pnl_pct:+.3f}%)",
                        )
                    )
                    continue

                # ── Fetch hourly bars ──
                df = await _fetch_hourly(broker, c.figi, settings.TREND_CANDLE_HISTORY_DAYS)
                if df.empty or len(df) < c.bb_n + 5:
                    logger.info(f"  {c.base}: insufficient bars ({len(df)})")
                    continue

                cur_pos = 1 if open_t and open_t["direction"] == "buy" else -1 if open_t else 0
                dec = strat.evaluate(
                    close=df["close"], n=c.bb_n, k=c.bb_k, current_position=cur_pos
                )

                if dec.action == "hold":
                    # Compact log; one line per contract
                    cur_close = float(df["close"].iloc[-1])
                    if dec.bands:
                        logger.info(
                            f"  {c.base:<10} HOLD  close={cur_close:.4f} "
                            f"[{dec.bands.lower:.4f}, {dec.bands.upper:.4f}] "
                            f"pos={cur_pos:+d}"
                        )
                    else:
                        logger.info(f"  {c.base:<10} HOLD  {dec.reason}")
                    continue

                if dec.action in ("open_long", "open_short"):
                    if kill_active:
                        logger.info(f"  {c.base}: signal SUPPRESSED (daily kill)")
                        continue
                    if n_open >= int(settings.TREND_MAX_OPEN_POSITIONS):
                        logger.info(f"  {c.base}: signal SUPPRESSED (max open {n_open})")
                        continue
                    direction = "buy" if dec.action == "open_long" else "sell"
                    lots = int(settings.TREND_LOTS_PER_TRADE)
                    oid, fill = await _place_market(
                        broker,
                        c.figi,
                        direction,
                        lots,
                        paper=bool(settings.TREND_PAPER_MODE),
                    )
                    bands = dec.bands
                    trade_id = await db.insert_trade(
                        base=c.base,
                        ticker=c.ticker,
                        figi=c.figi,
                        direction=direction,
                        lots=lots,
                        bb_n=c.bb_n,
                        bb_k=c.bb_k,
                        entry_price=fill,
                        entry_upper=bands.upper if bands else None,
                        entry_lower=bands.lower if bands else None,
                        entry_time=datetime.utcnow().isoformat(),
                        entry_order_id=oid,
                        rub_per_point=c.rpp,
                        lot_size=c.lot_size,
                        paper=int(bool(settings.TREND_PAPER_MODE)),
                    )
                    n_open += 1
                    logger.info(
                        f"  {c.base}: OPEN {direction.upper()} {lots} @ {fill:.4f}  "
                        f"{dec.reason}  trade_id={trade_id}"
                    )
                    notifier.push(
                        Msg(
                            MsgType.TRADE_OPENED,
                            f"TREND {c.base} {direction.upper()}",
                            f"{dec.reason}\n"
                            f"entry: {fill:.4f}  lots: {lots}  ({c.ticker})\n"
                            f"N={c.bb_n} k={c.bb_k}",
                        )
                    )
                    continue

                if dec.action == "close":
                    if open_t is None:
                        continue
                    fill_price = float(await broker.get_last_price(c.figi))
                    exit_dir = "sell" if open_t["direction"] == "buy" else "buy"
                    oid, _ = await _place_market(
                        broker,
                        c.figi,
                        exit_dir,
                        open_t["lots"],
                        paper=bool(settings.TREND_PAPER_MODE),
                    )
                    pnl, pnl_pct, _, comm_rub = comm.round_trip_pnl(
                        direction=open_t["direction"],
                        entry_price=open_t["entry_price"],
                        exit_price=fill_price,
                        lots=open_t["lots"],
                        lot_size=c.lot_size,
                        rub_per_point=c.rpp,
                        instrument_kind="future",
                        base_ticker=c.base,
                    )
                    await db.close_trade(
                        open_t["id"],
                        exit_price=fill_price,
                        exit_time=datetime.utcnow().isoformat(),
                        exit_reason="band_flip",
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        commission_rub=comm_rub,
                        exit_order_id=oid,
                    )
                    n_open = max(0, n_open - 1)
                    logger.info(
                        f"  {c.base}: CLOSE band_flip @ {fill_price:.4f}  "
                        f"pnl={pnl:+.2f}₽ ({pnl_pct:+.3f}%)"
                    )
                    notifier.push(
                        Msg(
                            MsgType.TRADE_CLOSED,
                            f"TREND {c.base} CLOSE",
                            f"{dec.reason}\n"
                            f"entry: {open_t['entry_price']:.4f} → exit: {fill_price:.4f}\n"
                            f"P&L: <b>{pnl:+.2f} ₽</b> ({pnl_pct:+.3f}%)",
                        )
                    )
                    continue

            # Sleep until next tick
            try:
                await asyncio.wait_for(
                    shutdown.wait(),
                    timeout=int(settings.TREND_LOOP_SECONDS),
                )
            except asyncio.TimeoutError:
                pass

    finally:
        logger.info("Shutting down trend bot…")
        try:
            await notifier.stop()
        except Exception:
            pass
        try:
            await broker.disconnect()
        except Exception:
            pass
        try:
            await db.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
