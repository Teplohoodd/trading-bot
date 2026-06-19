"""futbot.scalp.main — high-frequency scalping bot.

Run with:
    python -m futbot.scalp.main

Defaults to PAPER mode.  Subscribes to streaming order book + trades +
1-min candles for the 5 most liquid tier-1 FORTS contracts, evaluates the
combined microstructure + indicator signal every safety tick, places
small orders on strong signals, manages exits via tight ATR stop and
quick profit target.

Daily limits enforced:
  * SCALP_DAILY_LOSS_PCT_LIMIT realised drawdown → stop until UTC midnight
  * SCALP_DAILY_WIN_LOCK_PCT realised gain → stop (lock profits)
  * SCALP_MAX_TRADES_PER_HOUR — soft rate limit (default 12/h)
  * SCALP_MAX_TRADES_PER_DAY — disabled by default (set 0); enable if needed
  * Blackout hours MSK (no new entries during 9:00, 18:00, 23:00 MSK)

Existing positions are managed regardless of daily limits — we never get
"stuck" with a position past a limit without exit logic running.
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.broker import BrokerClient  # noqa: E402
from tinkoff.invest.utils import quotation_to_decimal  # noqa: E402

from futbot.scalp.config import ScalpSettings  # noqa: E402
from futbot.scalp.db import ScalpDB  # noqa: E402
from futbot.scalp.stream import StreamManager  # noqa: E402
from futbot.scalp import strategy as scalp_strategy  # noqa: E402
from futbot.universe import resolve_universe  # noqa: E402
from futbot.telegram_notifier import (  # noqa: E402
    TelegramNotifier,
    Msg,
    MsgType,
)
from futbot.telegram_commands import TelegramCommandServer  # noqa: E402
from futbot.utils import commissions as comm  # noqa: E402

MSK_OFFSET = timedelta(hours=3)


def _aggregate_5m_to_15m(bars_5m: list[dict]) -> list[dict]:
    """Group sequential 5-min bars into 15-min bars (3:1).  Last partial
    group is included only if it's complete.  Used as the source for
    'medium-timeframe ATR' that's large enough to beat commission notional.
    """
    if len(bars_5m) < 3:
        return []
    out = []
    # Walk from end backward so the most recent COMPLETE group is the last entry
    n = len(bars_5m)
    # Find the index where we have a multiple of 3 from start, working from end
    start = n - (n // 3) * 3
    for i in range(start, n, 3):
        group = bars_5m[i : i + 3]
        if len(group) < 3:
            break
        out.append(
            {
                "time": group[0]["time"],
                "open": group[0]["open"],
                "high": max(b["high"] for b in group),
                "low": min(b["low"] for b in group),
                "close": group[-1]["close"],
                "volume": sum(b["volume"] for b in group),
            }
        )
    return out


def _compute_atr_15m(state, n: int = 14) -> float:
    """Wilder ATR on '15-min' bars built by aggregating 5-min bars (3:1).
    Returns 0.0 if not enough bars — caller should fall back to ATR_1m × √15
    or skip the trade.
    """
    bars_5m = list(state.candles_5m)
    bars_15m = _aggregate_5m_to_15m(bars_5m)
    if len(bars_15m) < n + 1:
        return 0.0
    trs = []
    for i in range(1, len(bars_15m)):
        h = bars_15m[i]["high"]
        l = bars_15m[i]["low"]
        pc = bars_15m[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs[:n]) / n
    for v in trs[n:]:
        atr = (atr * (n - 1) + v) / n
    return float(atr)


def setup_logging(settings: ScalpSettings):
    settings.SCALP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.SCALP_LOG_PATH, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    for noisy in ("httpx", "telegram", "apscheduler", "grpc", "tinkoff"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Open position bookkeeping (in-memory; mirrored to DB) ───────────────────
class OpenPosition:
    __slots__ = (
        "trade_id",
        "figi",
        "ticker",
        "direction",
        "lots",
        "entry",
        "stop",
        "tp",
        "entry_time",
        "rub_per_point",
        "atr_at_entry",
        "peak",
        "instrument",
        "base",
    )

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


async def _place_entry(*, broker, figi, ticker, direction, lots, paper):
    last_p = float(await broker.get_last_price(figi))
    if paper:
        oid = f"paper-scalp-{direction}-{figi[-6:]}-{int(time.time())}"
        return oid, last_p
    from tinkoff.invest import OrderDirection

    side = (
        OrderDirection.ORDER_DIRECTION_BUY
        if direction == "buy"
        else OrderDirection.ORDER_DIRECTION_SELL
    )
    resp = await broker.post_market_order(figi, lots, side)
    return getattr(resp, "order_id", "?"), last_p


async def _place_exit(*, broker, figi, entry_direction, lots, paper):
    last_p = float(await broker.get_last_price(figi))
    if paper:
        oid = f"paper-scalp-exit-{figi[-6:]}-{int(time.time())}"
        return oid, last_p
    from tinkoff.invest import OrderDirection

    side = (
        OrderDirection.ORDER_DIRECTION_SELL
        if entry_direction == "buy"
        else OrderDirection.ORDER_DIRECTION_BUY
    )
    resp = await broker.post_market_order(figi, lots, side)
    return getattr(resp, "order_id", "?"), last_p


async def main():
    settings = ScalpSettings()
    setup_logging(settings)
    logger = logging.getLogger("futbot.scalp")

    mode = "PAPER" if settings.SCALP_PAPER_MODE else "LIVE"
    logger.info(f"scalp bot starting in {mode} mode")
    if not settings.SCALP_PAPER_MODE:
        logger.warning(
            "Live scalp mode is ON.  This places real market orders rapidly. "
            "Stop now (Ctrl-C) if unintended."
        )

    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="futbot-scalp",
    )
    await broker.connect()

    db = ScalpDB(settings.SCALP_DB_PATH)
    await db.initialize()

    notifier = TelegramNotifier(
        token=settings.TELEGRAM_BOT_TOKEN,
        chat_id=settings.TELEGRAM_CHAT_ID,
        paper=bool(settings.SCALP_PAPER_MODE),
    )
    await notifier.start()

    # Resolve universe — use scalp tier-1 list (subset of futbot's)
    # We piggyback on futbot.universe.resolve_universe but filter to scalp's bases.
    from futbot.config import FutSettings

    fs = FutSettings()
    fs.FUTBOT_TIER1_BASES = list(settings.SCALP_TIER1_BASES)
    fs.FUTBOT_TIER2_BASES = []
    fs.FUTBOT_MIN_DAYS_TO_EXPIRY = int(settings.SCALP_MIN_DAYS_TO_EXPIRY)
    universe = await resolve_universe(broker, fs)
    if not universe:
        logger.error("No tier-1 contracts resolved — exiting")
        return
    logger.info(f"Scalp universe: {[c.ticker for c in universe]}")

    # Streaming
    stream = StreamManager(token=settings.T_INVEST_TOKEN, app_name="futbot-scalp")
    for c in universe:
        await stream.add_instrument(c.figi, c.ticker)
    # Prefetch ~24h of 5-min bars so ATR(14) on aggregated 15-min bars is
    # available immediately, not after 3h of streaming warm-up.
    logger.info("Prefetching 5-min history for warm-up…")
    for c in universe:
        await stream.prefetch_history(broker, c.figi, hours_5m=24)
    await stream.start_streaming(book_depth=int(settings.SCALP_BOOK_DEPTH))

    # In-memory open-position map (mirrored to DB)
    open_pos: dict[str, OpenPosition] = {}

    # ── Reconcile open positions from DB (handles bot restart with live
    # positions still pending).  Without this, restarts would leak positions
    # — DB says "open" but nothing manages them.  Observed live (2026-05-19→20):
    # BRM6 sell stayed "open" 14 hours after bot restart, way past time_cap.
    #
    # Strategy: for each open DB row, either
    #   (a) hydrate into open_pos so the running loop manages it, OR
    #   (b) force-close it if the position is too stale (> 4 × MAX_HOLD_SECONDS)
    #       — anything older is operator-level mess we shouldn't auto-trade.
    open_rows_at_boot = await db.open_trades()
    max_hold = int(settings.SCALP_MAX_HOLD_SECONDS)
    reconciled = 0
    force_closed = 0
    for r in open_rows_at_boot:
        figi = r["figi"]
        try:
            from datetime import datetime as _dt

            entry_iso = r["entry_time"]
            entry_dt = _dt.fromisoformat(entry_iso.replace("Z", "+00:00"))
            if entry_dt.tzinfo is None:
                from datetime import timezone as _tz

                entry_dt = entry_dt.replace(tzinfo=_tz.utc)
            from datetime import timezone as _tz

            age_sec = (_dt.now(_tz.utc) - entry_dt).total_seconds()
        except Exception:
            age_sec = 0

        # Re-resolve the contract instrument from the live universe
        contract = next((c for c in universe if c.figi == figi), None)

        if age_sec > max_hold * 4 or contract is None:
            # Stale or contract no longer in universe — force-close at market.
            # rub_per_point isn't in scalp_trades schema — fetch from broker
            # metadata if we still have the contract, else fall back to 1.
            try:
                fill = float(await broker.get_last_price(figi))
            except Exception:
                fill = float(r["entry_price"])
            direction = r["direction"]
            entry = float(r["entry_price"])
            lots = int(r["lots"])
            rpp = 1.0
            if contract is not None:
                try:
                    meta = broker.extract_futures_metadata(contract.instrument)
                    rpp = float(meta.get("rub_per_point") or 1.0)
                except Exception:
                    pass
            from futbot.utils import commissions as _comm

            pnl, pnl_pct, gross, comm_rub = _comm.round_trip_pnl(
                direction=direction,
                entry_price=entry,
                exit_price=fill,
                lots=lots,
                lot_size=1,
                rub_per_point=rpp,
                instrument_kind="future",
            )
            await db.close_trade(
                r["id"],
                exit_price=fill,
                exit_reason="boot_reconcile_stale",
                pnl=pnl,
                pnl_pct=pnl_pct,
            )
            logger.warning(
                f"  RECONCILE {r['ticker']} {direction} entered {entry_iso[:19]} "
                f"({age_sec:.0f}s ago) — force-closed @ {fill:.4f} pnl={pnl:+.2f}₽"
            )
            force_closed += 1
        else:
            # Hydrate into the live tracking map so the loop manages it
            meta = broker.extract_futures_metadata(contract.instrument)
            rpp = float(meta.get("rub_per_point") or 1.0)
            # ATR at entry isn't stored — estimate from current 15-m ATR
            # (good enough for trail-update math; the actual stop level is
            # already on the DB row and respected verbatim).
            est_atr = abs(float(r["take_profit"]) - float(r["entry_price"])) / float(
                settings.SCALP_TAKE_PROFIT_ATR
            )
            open_pos[figi] = OpenPosition(
                trade_id=r["id"],
                figi=figi,
                ticker=r["ticker"],
                direction=r["direction"],
                lots=int(r["lots"]),
                entry=float(r["entry_price"]),
                stop=float(r["stop_loss"]),
                tp=float(r["take_profit"]),
                entry_time=entry_dt.timestamp(),
                rub_per_point=rpp,
                atr_at_entry=max(est_atr, 0.01),
                peak=float(r["entry_price"]),  # conservative; will update on next tick
                instrument=contract.instrument,
                base=contract.base,
            )
            logger.info(
                f"  RECONCILE {r['ticker']} {r['direction']} "
                f"entered {age_sec/60:.1f} min ago — re-hydrated to live map"
            )
            reconciled += 1
    if open_rows_at_boot:
        logger.info(
            f"Reconcile finished: {reconciled} re-hydrated, "
            f"{force_closed} force-closed, {len(open_pos)} positions now live"
        )

    # Portfolio value snapshot for daily-pnl pct
    try:
        portfolio_value = float(await broker.get_portfolio_value())
    except Exception:
        portfolio_value = 100_000.0  # paper fallback
    logger.info(f"Portfolio value (snapshot): {portfolio_value:.0f} ₽")

    # ── Telegram command handlers ─────────────────────────────────────
    # These are read by /<cmd> messages from the configured chat_id.
    # They read from db + the live open_pos dict, so they're always
    # current — no caching staleness.
    async def _cmd_status() -> str:
        today_iso = datetime.utcnow().date().isoformat()
        n_today = await db.trades_today(today_iso)
        pnl_today = await db.daily_pnl(today_iso)
        open_rows = await db.open_trades()
        kill, kill_reason = await _daily_kill_active()
        kill_line = "🛑 " + kill_reason if kill else "✅ active"
        lines = [
            f"<b>Scalp bot — {mode}</b>",
            f"Mode: {kill_line}",
            f"Universe: {len(universe)} contracts",
            f"Open positions: {len(open_pos)}",
            f"",
            f"<b>Today (UTC {today_iso}):</b>",
            f"  trades closed: {n_today}",
            f"  NET P&L: {pnl_today:+.2f} ₽",
            f"  portfolio: {portfolio_value:.0f} ₽",
        ]
        if open_pos:
            lines.append("")
            lines.append("<b>Open:</b>")
            for figi, p in open_pos.items():
                held = time.time() - p.entry_time
                lines.append(
                    f"  {p.ticker} {p.direction.upper()} @ {p.entry:.4f}  "
                    f"stop {p.stop:.4f}  tp {p.tp:.4f}  ({held:.0f}s)"
                )
        return "\n".join(lines)

    async def _cmd_pnl() -> str:
        today_iso = datetime.utcnow().date().isoformat()
        pnl_today = await db.daily_pnl(today_iso)
        # Last 7d totals
        cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
        async with db._lock:
            cur = await db._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(pnl),0) FROM scalp_trades "
                "WHERE entry_time >= ? AND pnl IS NOT NULL",
                (cutoff,),
            )
            row = await cur.fetchone()
        n7, pnl7 = (row[0], float(row[1])) if row else (0, 0.0)
        return (
            f"<b>P&amp;L</b>\n"
            f"Today: {pnl_today:+.2f} ₽\n"
            f"Last 7d: {pnl7:+.2f} ₽ over {n7} trades"
        )

    async def _cmd_open() -> str:
        if not open_pos:
            return "No open positions."
        lines = ["<b>Open positions:</b>"]
        for figi, p in open_pos.items():
            held = time.time() - p.entry_time
            # current price for unrealised
            try:
                last_p = float(await broker.get_last_price(p.figi))
            except Exception:
                last_p = p.entry
            pnl_pts = (last_p - p.entry) if p.direction == "buy" else (p.entry - last_p)
            unreal = pnl_pts * p.rub_per_point * p.lots
            lines.append(
                f"  <b>{p.ticker}</b> {p.direction.upper()}×{p.lots} @ {p.entry:.4f}\n"
                f"    last={last_p:.4f}  unreal={unreal:+.2f} ₽  held={held:.0f}s\n"
                f"    stop={p.stop:.4f}  tp={p.tp:.4f}"
            )
        return "\n".join(lines)

    async def _cmd_universe() -> str:
        lines = ["<b>Universe (live state):</b>"]
        for c in universe:
            st = await stream.get_state(c.figi)
            if st is None:
                lines.append(f"  {c.ticker}: no state")
                continue
            fresh = "✓" if st.is_fresh() else "STALE"
            sig = scalp_strategy.evaluate(
                state=st,
                instrument=c.instrument,
                settings=settings,
            )
            lines.append(
                f"  {c.ticker} {fresh}  score={sig.score:+.2f}  "
                f"book={sig.components.get('book', 0):+.2f}  "
                f"tfi={sig.components.get('tfi', 0):+.2f}  "
                f"({sig.rejection or 'OK'})"
            )
        return "\n".join(lines)

    async def _cmd_stop_bot() -> str:
        shutdown.set()
        return "🛑 Scalp bot shutdown requested."

    commands = TelegramCommandServer(
        token=settings.TELEGRAM_BOT_TOKEN,
        chat_id=settings.TELEGRAM_CHAT_ID,
        handlers={
            "status": _cmd_status,
            "pnl": _cmd_pnl,
            "open": _cmd_open,
            "universe": _cmd_universe,
            "shutdown": _cmd_stop_bot,
        },
    )
    await commands.start()

    # Shutdown plumbing
    shutdown = asyncio.Event()

    def _sig(*_):
        logger.info("Shutdown signal received")
        shutdown.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(s, _sig)
        except NotImplementedError:
            pass

    # ── Helpers ─────────────────────────────────────────────────────────
    def _now_msk_hour() -> int:
        return (datetime.now(timezone.utc) + MSK_OFFSET).hour

    async def _daily_kill_active() -> tuple[bool, str]:
        today_iso = datetime.utcnow().date().isoformat()
        pnl = await db.daily_pnl(today_iso)

        # Hard P&L floors — these are the REAL safety
        loss_cap = portfolio_value * float(settings.SCALP_DAILY_LOSS_PCT_LIMIT)
        if pnl < -loss_cap:
            return True, f"daily loss {pnl:+.0f}₽ < -{loss_cap:.0f}₽"
        win_lock = portfolio_value * float(settings.SCALP_DAILY_WIN_LOCK_PCT)
        if pnl > win_lock:
            return True, f"daily gain {pnl:+.0f}₽ > {win_lock:.0f}₽ — profits locked"

        # Optional total-day count limit (0 = disabled)
        max_total = int(settings.SCALP_MAX_TRADES_PER_DAY)
        if max_total > 0:
            trades_n = await db.trades_today(today_iso)
            if trades_n >= max_total:
                return True, f"max trades/day {trades_n} reached"

        # Per-hour rate limit (soft protection against runaway-bug days).
        # Counts trades opened in the last 60 minutes regardless of UTC hour.
        max_per_hour = int(settings.SCALP_MAX_TRADES_PER_HOUR)
        if max_per_hour > 0:
            from datetime import timedelta

            cutoff = (datetime.utcnow() - timedelta(hours=1)).isoformat()
            async with db._lock:
                cur = await db._db.execute(
                    "SELECT COUNT(*) FROM scalp_trades WHERE entry_time >= ?",
                    (cutoff,),
                )
                row = await cur.fetchone()
            n_last_hour = int(row[0]) if row else 0
            if n_last_hour >= max_per_hour:
                return True, (
                    f"rate limit: {n_last_hour} trades in last hour " f"≥ {max_per_hour}/h"
                )

        return False, ""

    async def _maybe_enter(c, st, _seen_blackout=[False]):
        """Evaluate signal for one contract; place entry if approved."""
        if c.figi in open_pos:
            return
        if len(open_pos) >= int(settings.SCALP_MAX_OPEN_POSITIONS):
            return
        if _now_msk_hour() in list(settings.SCALP_BLACKOUT_HOURS_MSK):
            if not _seen_blackout[0]:
                logger.info(f"Blackout hour MSK={_now_msk_hour()}; skipping entries")
                _seen_blackout[0] = True
            return
        _seen_blackout[0] = False

        # Stale state check
        if not st.is_fresh(30.0):
            return

        sig = scalp_strategy.evaluate(
            state=st,
            instrument=c.instrument,
            settings=settings,
        )
        if sig.direction is None:
            return

        kill, kill_reason = await _daily_kill_active()
        if kill:
            return  # silent — main tick will log it once per minute

        # ── Build stop / TP from 15-MIN ATR (not 1-min) ────────────────
        # 1-min ATR is too small vs commission notional.  We compute ATR
        # on the 15-min candle deque populated by the second streaming
        # subscription.  Falls back to ATR_1m × √15 if 15m bars haven't
        # warmed yet (rough scaling — accurate for i.i.d. returns).
        atr_1m = float(sig.components.get("atr_1m") or 0)
        atr_15m = _compute_atr_15m(st)
        if atr_15m <= 0 and atr_1m > 0:
            atr_15m = atr_1m * (15**0.5)  # √15 ≈ 3.87×
        if atr_15m <= 0:
            return
        last_p = st.last_price
        if not last_p or last_p <= 0:
            return
        last_p = float(last_p)
        if sig.direction == "buy":
            stop = last_p - atr_15m * float(settings.SCALP_INITIAL_STOP_ATR)
            tp = last_p + atr_15m * float(settings.SCALP_TAKE_PROFIT_ATR)
        else:
            stop = last_p + atr_15m * float(settings.SCALP_INITIAL_STOP_ATR)
            tp = last_p - atr_15m * float(settings.SCALP_TAKE_PROFIT_ATR)

        # rub_per_point for P&L conversion
        meta = broker.extract_futures_metadata(c.instrument)
        rub_per_point = float(meta.get("rub_per_point") or 1.0)

        # ── Edge-vs-commission sanity check ──────────────────────────────
        # TP profit must exceed (SCALP_COMMISSION_GATE_MULT × round-trip).
        # Default mult = 1.2 (small margin over breakeven).  Tune via env.
        lot_size = int(getattr(c.instrument, "lot", 1) or 1)
        tp_profit_rub = abs(tp - last_p) * rub_per_point * lot_size
        rt_commission_rub = comm.estimated_round_trip_cost(
            price=last_p,
            lots=1,
            lot_size=lot_size,
            rub_per_point=rub_per_point,
            instrument_kind="future",
            base_ticker=c.base,
        )
        gate_mult = float(settings.SCALP_COMMISSION_GATE_MULT)
        gate_threshold = gate_mult * rt_commission_rub
        if tp_profit_rub < gate_threshold:
            # Compute what ATR_15m would unblock this — helps debug
            needed_atr = gate_threshold / (
                float(settings.SCALP_TAKE_PROFIT_ATR) * rub_per_point * lot_size
            )
            logger.info(
                f"  {c.ticker}: SKIP — TP {tp_profit_rub:.2f}₽ < "
                f"{gate_mult:.1f}× comm ({gate_threshold:.2f}₽).  "
                f"ATR_15m={atr_15m:.2f} (need ≥{needed_atr:.2f})"
            )
            return

        # Place 1 lot (scalp keeps it tiny)
        oid, fill = await _place_entry(
            broker=broker,
            figi=c.figi,
            ticker=c.ticker,
            direction=sig.direction,
            lots=1,
            paper=bool(settings.SCALP_PAPER_MODE),
        )
        trade_id = await db.insert_trade(
            figi=c.figi,
            ticker=c.ticker,
            direction=sig.direction,
            lots=1,
            entry_price=fill,
            stop_loss=stop,
            take_profit=tp,
            score=sig.score,
            components_json=json.dumps(sig.components),
            paper=bool(settings.SCALP_PAPER_MODE),
        )
        open_pos[c.figi] = OpenPosition(
            trade_id=trade_id,
            figi=c.figi,
            ticker=c.ticker,
            direction=sig.direction,
            lots=1,
            entry=fill,
            stop=stop,
            tp=tp,
            entry_time=time.time(),
            rub_per_point=rub_per_point,
            atr_at_entry=atr_15m,
            peak=fill,
            instrument=c.instrument,
            base=c.base,
        )
        logger.info(
            f"  {c.ticker}: SCALP OPEN {sig.direction.upper()} @ {fill:.4f}  "
            f"score={sig.score:+.2f}  stop={stop:.4f}  tp={tp:.4f}  "
            f"comp={sig.components}"
        )
        notifier.push(
            Msg(
                MsgType.TRADE_OPENED,
                f"SCALP {c.ticker} {sig.direction.upper()}",
                f"@{fill:.4f}  score={sig.score:+.2f}\nstop={stop:.4f}  tp={tp:.4f}",
            )
        )

    async def _maybe_exit(pos: OpenPosition, st):
        """Check exits: TP, stop (using last_price as proxy for the live tape),
        signal flip, time cap."""
        # Use both last_price AND book mid to be robust against stale prices
        last_p = st.last_price or 0.0
        if last_p <= 0:
            return

        exit_reason = None

        # Update peak for trailing logic
        if pos.direction == "buy":
            pos.peak = max(pos.peak, last_p)
        else:
            pos.peak = min(pos.peak, last_p)

        # Trailing stop — REDESIGNED 2026-05-16 after observing 5/13 trades
        # exit at exactly entry-price ("break-even stops") due to over-tight
        # trail.  New rules:
        #   1. Activate ONLY after profit reaches 70 % of way to TP (not just
        #      +1 ATR).  This stops the trail from kicking in on micro-noise.
        #   2. When activated, lock in profit equal to round-trip commission
        #      + buffer (3× one-side), so trail-out is guaranteed net positive.
        cpct = comm.commission_pct("future", pos.base)
        tp_distance = abs(pos.tp - pos.entry)
        if pos.direction == "buy":
            profit = pos.peak - pos.entry
            if profit >= tp_distance * 0.70:
                commission_buffer = 3 * cpct * pos.entry
                new_stop = pos.entry + commission_buffer
                pos.stop = max(pos.stop, new_stop)
        else:
            profit = pos.entry - pos.peak
            if profit >= tp_distance * 0.70:
                commission_buffer = 3 * cpct * pos.entry
                new_stop = pos.entry - commission_buffer
                pos.stop = min(pos.stop, new_stop)

        # Stop check
        if pos.direction == "buy" and last_p <= pos.stop:
            exit_reason = "stop"
        elif pos.direction == "sell" and last_p >= pos.stop:
            exit_reason = "stop"
        # TP check
        elif pos.direction == "buy" and last_p >= pos.tp:
            exit_reason = "take_profit"
        elif pos.direction == "sell" and last_p <= pos.tp:
            exit_reason = "take_profit"
        # Time cap
        elif (time.time() - pos.entry_time) >= settings.SCALP_MAX_HOLD_SECONDS:
            exit_reason = "time_cap"
        else:
            # No hard exit triggered.  Now consider:
            #   1. Early-abandon flat positions
            #   2. Signal flip (gated by age + profit)
            age = time.time() - pos.entry_time
            if pos.direction == "buy":
                profit_atr = (last_p - pos.entry) / pos.atr_at_entry if pos.atr_at_entry > 0 else 0
            else:
                profit_atr = (pos.entry - last_p) / pos.atr_at_entry if pos.atr_at_entry > 0 else 0

            # 1) Early-abandon: held > 5 min AND not in profit by ≥ 0.15×ATR
            ea_min_age = float(settings.SCALP_EARLY_ABANDON_AFTER_SEC)
            ea_min_prof = float(settings.SCALP_EARLY_ABANDON_PROFIT_ATR)
            if age >= ea_min_age and profit_atr < ea_min_prof:
                exit_reason = "early_abandon"

            # 2) Signal flip — only if position is in profit AND aged enough
            elif age >= float(settings.SCALP_FLIP_MIN_AGE_SEC) and profit_atr >= float(
                settings.SCALP_FLIP_MIN_PROFIT_ATR
            ):
                flip = scalp_strategy.should_exit(
                    state=st,
                    instrument=pos.instrument,
                    settings=settings,
                    position_direction=pos.direction,
                )
                if flip.direction is not None:
                    exit_reason = "signal_flip"

        if exit_reason is None:
            return

        oid, fill = await _place_exit(
            broker=broker,
            figi=pos.figi,
            entry_direction=pos.direction,
            lots=pos.lots,
            paper=bool(settings.SCALP_PAPER_MODE),
        )
        # NET P&L via central commissions helper.  Same rates used in
        # paper and live so the stats during paper trading directly project
        # to live behaviour.
        lot_size = int(getattr(pos.instrument, "lot", 1) or 1)
        pnl, pnl_pct, gross, commission = comm.round_trip_pnl(
            direction=pos.direction,
            entry_price=pos.entry,
            exit_price=fill,
            lots=pos.lots,
            lot_size=lot_size,
            rub_per_point=pos.rub_per_point,
            instrument_kind="future",
            base_ticker=pos.base,
        )
        await db.close_trade(
            pos.trade_id,
            exit_price=fill,
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        held = time.time() - pos.entry_time
        logger.info(
            f"  {pos.ticker}: SCALP EXIT ({exit_reason}) @ {fill:.4f}  "
            f"pnl={pnl:+.2f}₽ ({pnl_pct:+.3f}%) held={held:.1f}s"
        )
        notifier.push(
            Msg(
                MsgType.TRADE_CLOSED,
                f"SCALP {pos.ticker} {exit_reason}",
                f"entry={pos.entry:.4f} → exit={fill:.4f}\n"
                f"P&L = {pnl:+.2f} ₽ ({pnl_pct:+.3f} %)  held={held:.0f}s",
            )
        )
        open_pos.pop(pos.figi, None)

    # ── Main loop: safety tick every SCALP_SAFETY_TICK_SECONDS ──────────
    boot_alerted = False
    last_kill_log = 0.0
    last_heartbeat = 0.0
    HEARTBEAT_SEC = 30.0  # log per-contract micro state every 30s
    last_score_log: dict[str, float] = {}  # cooldown per ticker for "near miss" logs
    try:
        while not shutdown.is_set():
            t0 = time.monotonic()
            if not boot_alerted:
                notifier.push(
                    Msg(
                        MsgType.BOOT,
                        "Scalp started",
                        f"mode: {mode}\nuniverse: {len(universe)} contracts",
                    )
                )
                boot_alerted = True

            # Manage open positions FIRST
            for c in universe:
                if c.figi in open_pos:
                    st = await stream.get_state(c.figi)
                    if st is None:
                        continue
                    await _maybe_exit(open_pos[c.figi], st)

            # Kill switch
            kill, kill_reason = await _daily_kill_active()
            if kill:
                if time.monotonic() - last_kill_log > 60:
                    logger.warning(f"Daily kill active: {kill_reason}")
                    last_kill_log = time.monotonic()
            else:
                # Evaluate entries
                for c in universe:
                    st = await stream.get_state(c.figi)
                    if st is None:
                        continue
                    try:
                        await _maybe_enter(c, st)
                    except Exception as e:
                        logger.exception(f"  {c.ticker}: maybe_enter failed: {e}")

            # ── Heartbeat: every 30s dump current signal scores ──────────
            now_mono = time.monotonic()
            if now_mono - last_heartbeat >= HEARTBEAT_SEC:
                last_heartbeat = now_mono
                lines = []
                for c in universe:
                    st = await stream.get_state(c.figi)
                    if st is None:
                        lines.append(f"  {c.ticker:<6} no state")
                        continue
                    fresh = "✓" if st.is_fresh() else "STALE"
                    n_book = len(st.bids) if st.bids else 0
                    n_trades = len(st.recent_trades)
                    n_1m = len(st.candles_1m)
                    n_5m = len(st.candles_5m)
                    atr15 = _compute_atr_15m(st)
                    sig_dbg = scalp_strategy.evaluate(
                        state=st,
                        instrument=c.instrument,
                        settings=settings,
                    )
                    score = sig_dbg.score
                    comp = sig_dbg.components
                    in_pos = "POS" if c.figi in open_pos else "   "
                    lines.append(
                        f"  {c.ticker:<6} {in_pos} {fresh:<5} book={n_book:>2} "
                        f"trades={n_trades:>3} 1m={n_1m:>3} 5m={n_5m:>3} "
                        f"atr15={atr15:>6.2f}  "
                        f"score={score:+.2f}  book={comp.get('book', 0):+.2f}  "
                        f"tfi={comp.get('tfi', 0):+.2f}(n={comp.get('tfi_n', 0)})  "
                        f"reason={sig_dbg.rejection or 'OK'}"
                    )
                logger.info("heartbeat:\n" + "\n".join(lines))

            elapsed = time.monotonic() - t0
            sleep_for = max(0.1, float(settings.SCALP_SAFETY_TICK_SECONDS) - elapsed)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=sleep_for)
            except asyncio.TimeoutError:
                pass

    finally:
        logger.info("Shutting down scalp bot...")
        try:
            await stream.stop()
        except Exception:
            pass
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
