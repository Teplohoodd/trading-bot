"""futbot entry point.

Run with:
    python -m futbot.main

Defaults to PAPER mode (no real orders).  Live trading requires:
    FUTBOT_PAPER_MODE=false   in .env

The loop:
  1. Resolve / refresh universe (every 6 hours).
  2. For each contract: fetch multi-TF candles, run pipeline, log decision.
  3. Approved decisions → run risk audit, then sizer, then paper/live order.
  4. For each OPEN position: refresh trailing stop, check stop-out / max-hold.

Logs go to stdout AND data/logs/futbot.log so this can be tailed from any
session.  Telegram notifications use the chat ID from the parent .env.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

# Add parent package (trade_claude root) to sys.path so we can import
# `core.broker`, `analysis.macro`, etc. without copy-pasting.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.broker import BrokerClient  # noqa: E402

from futbot.config import FutSettings  # noqa: E402
from futbot.db import FutbotDB  # noqa: E402
from futbot.universe import resolve_universe, Contract  # noqa: E402
from futbot.data.candles import fetch_multi_tf  # noqa: E402
from futbot.pipeline import decision as decision_module  # noqa: E402
from futbot.execution import sizer as sizer_module  # noqa: E402
from futbot.execution import stops as stops_module  # noqa: E402
from futbot.execution import orders as orders_module  # noqa: E402
from futbot.risk import audit as audit_module  # noqa: E402
from futbot.risk.circuit import CircuitBreaker  # noqa: E402
from futbot.telegram_notifier import (  # noqa: E402
    TelegramNotifier,
    fmt_trade_opened,
    fmt_trade_closed,
    fmt_boot,
    fmt_circuit,
)


def setup_logging(settings: FutSettings):
    settings.FUTBOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.FUTBOT_LOG_PATH, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    for noisy in ("httpx", "telegram", "apscheduler", "grpc", "tinkoff"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def _atr_1h(df_1h, n: int = 14) -> float:
    """Latest 14-bar Wilder ATR on 1h.  Returns 0.0 if not enough data."""
    if df_1h is None or df_1h.empty or len(df_1h) < n + 1:
        return 0.0
    high = df_1h["high"]
    low = df_1h["low"]
    close = df_1h["close"]
    import pandas as pd

    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    val = float(atr.iloc[-1])
    return val if val == val else 0.0  # NaN guard


# ─────────────────────────────────────────────────────────────────────────────
# Entry path: pipeline → audit → sizer → place order → DB
# ─────────────────────────────────────────────────────────────────────────────
async def _maybe_open(
    *,
    broker,
    db,
    contract: Contract,
    tf_data: dict,
    settings: FutSettings,
    logger: logging.Logger,
    notifier: TelegramNotifier | None = None,
):
    # Skip if we already have an open position on this contract
    existing = await db.open_trade_for_figi(contract.figi)
    if existing:
        return

    # Pipeline
    dec = await decision_module.evaluate_contract(
        figi=contract.figi,
        ticker=contract.ticker,
        tf_data=tf_data,
        settings=settings,
        base_ticker=contract.base,
    )
    dec_id = await db.insert_decision(
        figi=dec.figi,
        ticker=dec.ticker,
        proposed_direction=dec.direction,
        approved=dec.approved,
        layers=dec.layers,
        rejected_at_layer=dec.rejected_at,
        rejection_reason=dec.rejection_reason,
    )
    if not dec.approved:
        logger.info(f"  {contract.ticker}: rejected @ {dec.rejected_at} — {dec.rejection_reason}")
        return

    # Pull ГО + step_value from the resolved instrument
    meta = broker.extract_futures_metadata(contract.instrument)
    if dec.direction == "sell":
        im = meta.get("initial_margin_sell") or 0
    else:
        im = meta.get("initial_margin_buy") or 0
    if not im or im <= 0:
        # SDK doesn't always populate; fall back to dlong/dshort × price.
        d_key = "dshort" if dec.direction == "sell" else "dlong"
        d_ratio = float(meta.get(d_key) or 0)
        last_p = float(await broker.get_last_price(contract.figi))
        im = d_ratio * last_p if d_ratio > 0 else 0
    rub_per_point = float(meta.get("rub_per_point") or 1.0)

    atr_1h = await _atr_1h(tf_data["1h"])
    lot_size = int(getattr(contract.instrument, "lot", 1) or 1)

    # Risk audit
    audit_res = await audit_module.audit(
        broker=broker,
        db=db,
        contract=contract,
        direction=dec.direction,
        proposed_lots=1,  # provisional
        initial_margin=im,
        order_book=None,
        settings=settings,
    )
    if not audit_res.approved:
        logger.info(f"  {contract.ticker}: audit blocked — {audit_res.reason}")
        # Patch the decision row so the rejection is searchable
        await db.insert_decision(
            figi=contract.figi,
            ticker=contract.ticker,
            proposed_direction=dec.direction,
            approved=False,
            layers={**dec.layers, "audit": audit_res.detail},
            rejected_at_layer="audit",
            rejection_reason=audit_res.reason,
        )
        return

    # Sizer
    portfolio_value = await broker.get_portfolio_value()
    sized = sizer_module.compute_lots(
        portfolio_value=portfolio_value,
        initial_margin=im,
        rub_per_point=rub_per_point,
        atr_1h=atr_1h,
        lot_size=lot_size,
        settings=settings,
    )
    lots = int(sized.get("lots", 0))
    if lots < 1:
        logger.info(f"  {contract.ticker}: sizer says 0 lots — {sized.get('reason')}")
        return

    # Place
    oid, fill = await orders_module.place_entry(
        broker=broker,
        figi=contract.figi,
        ticker=contract.ticker,
        direction=dec.direction,
        lots=lots,
        paper=bool(settings.FUTBOT_PAPER_MODE),
    )

    # Initial trailing-stop state
    st = stops_module.initial_state(
        direction=dec.direction,
        entry=fill,
        atr_1h=atr_1h,
        settings=settings,
    )
    target_price = (fill + 2 * atr_1h) if dec.direction == "buy" else (fill - 2 * atr_1h)

    trade_id = await db.insert_trade(
        figi=contract.figi,
        ticker=contract.ticker,
        direction=dec.direction,
        lots=lots,
        entry_price=fill,
        stop_loss=round(st.current_stop, 4),
        take_profit=round(target_price, 4),
        initial_margin=im,
        rub_per_point=rub_per_point,
        paper=bool(settings.FUTBOT_PAPER_MODE),
        decision_id=dec_id,
        entry_order_id=oid,
    )
    await db.upsert_position_state(
        figi=contract.figi,
        entry_time=datetime.utcnow().isoformat(),
        direction=dec.direction,
        entry_price=fill,
        peak_price=st.peak,
        trail_active=int(st.trail_active),
        initial_stop=st.current_stop,
        current_stop=st.current_stop,
        last_updated=datetime.utcnow().isoformat(),
    )
    logger.info(
        f"  {contract.ticker}: OPENED {dec.direction.upper()} × {lots} @ {fill:.4f} "
        f"stop={st.current_stop:.4f} risk_pts={st.initial_risk:.4f} trade_id={trade_id}"
    )
    if notifier:
        notifier.push(
            fmt_trade_opened(
                ticker=contract.ticker,
                direction=dec.direction,
                lots=lots,
                entry=fill,
                stop=st.current_stop,
                reason_chain=dec.layers,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Manage open positions: refresh trailing stop, check stop-out / max-hold
# ─────────────────────────────────────────────────────────────────────────────
async def _manage_open(
    *,
    broker,
    db,
    settings: FutSettings,
    logger: logging.Logger,
    notifier: TelegramNotifier | None = None,
):
    open_rows = await db.open_trades()
    if not open_rows:
        return

    for row in open_rows:
        figi = row["figi"]
        ticker = row["ticker"]
        direction = row["direction"]
        entry = float(row["entry_price"])
        lots = int(row["lots"])
        entry_time = datetime.fromisoformat(row["entry_time"].replace("Z", "+00:00"))
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        # Fetch latest 1h ATR + latest 5m bar (for stop check)
        try:
            from futbot.data.candles import fetch_tf

            df_1h = await fetch_tf(broker, figi, "1h")
            df_5m = await fetch_tf(broker, figi, "5m")
        except Exception as e:
            logger.warning(f"  manage {ticker}: data fetch failed ({e})")
            continue

        atr_1h = await _atr_1h(df_1h)
        if df_5m.empty:
            continue
        last_bar = df_5m.iloc[-1]
        last_p = float(last_bar["close"])
        bar_high = float(last_bar["high"])
        bar_low = float(last_bar["low"])

        # Rehydrate StopState from DB
        ps = await db.get_position_state(figi)
        if ps is None:
            # Defensive: rebuild from row if state was lost (e.g. fresh DB)
            st = stops_module.initial_state(
                direction=direction,
                entry=entry,
                atr_1h=atr_1h,
                settings=settings,
            )
        else:
            st = stops_module.StopState(
                direction=ps["direction"],
                entry=entry,
                initial_risk=abs(entry - float(ps["initial_stop"])),
                peak=float(ps["peak_price"]),
                current_stop=float(ps["current_stop"]),
                trail_active=bool(ps["trail_active"]),
            )

        # Roll the stop forward
        new_st = stops_module.update(
            state=st,
            last_price=last_p,
            atr_1h=atr_1h,
            settings=settings,
        )
        if (
            new_st.current_stop != st.current_stop
            or new_st.peak != st.peak
            or new_st.trail_active != st.trail_active
        ):
            await db.upsert_position_state(
                figi=figi,
                entry_time=entry_time.isoformat(),
                direction=direction,
                entry_price=entry,
                peak_price=new_st.peak,
                trail_active=int(new_st.trail_active),
                initial_stop=st.current_stop,
                current_stop=new_st.current_stop,
                last_updated=datetime.utcnow().isoformat(),
            )

        # Check stop-out (use bar's high/low so we catch wicks)
        if stops_module.is_stopped_out(state=new_st, bar_high=bar_high, bar_low=bar_low):
            oid, fill = await orders_module.place_exit(
                broker=broker,
                figi=figi,
                ticker=ticker,
                entry_direction=direction,
                lots=lots,
                reason="trailing_stop",
                paper=bool(settings.FUTBOT_PAPER_MODE),
            )
            pnl_pts = (fill - entry) if direction == "buy" else (entry - fill)
            rpp = float(row["rub_per_point"]) if row["rub_per_point"] else 1.0
            pnl = pnl_pts * rpp * lots
            pnl_pct = pnl_pts / entry * 100 if entry else 0
            await db.close_trade(
                row["id"],
                exit_price=fill,
                exit_reason="trailing_stop",
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_order_id=oid,
            )
            await db.delete_position_state(figi)
            logger.info(
                f"  {ticker}: STOPPED OUT @ {fill:.4f} (stop was {new_st.current_stop:.4f}) "
                f"pnl={pnl:+.2f} ({pnl_pct:+.2f}%)"
            )
            if notifier:
                notifier.push(
                    fmt_trade_closed(
                        ticker=ticker,
                        direction=direction,
                        lots=lots,
                        entry=entry,
                        exit_=fill,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        reason="trailing_stop",
                    )
                )
            continue

        # Max-hold time cap
        max_hold = int(settings.FUTBOT_MAX_HOLD_HOURS)
        held_h = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600
        if held_h >= max_hold:
            oid, fill = await orders_module.place_exit(
                broker=broker,
                figi=figi,
                ticker=ticker,
                entry_direction=direction,
                lots=lots,
                reason="time_cap",
                paper=bool(settings.FUTBOT_PAPER_MODE),
            )
            pnl_pts = (fill - entry) if direction == "buy" else (entry - fill)
            rpp = float(row["rub_per_point"]) if row["rub_per_point"] else 1.0
            pnl = pnl_pts * rpp * lots
            pnl_pct = pnl_pts / entry * 100 if entry else 0
            await db.close_trade(
                row["id"],
                exit_price=fill,
                exit_reason="time_cap",
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_order_id=oid,
            )
            await db.delete_position_state(figi)
            logger.info(
                f"  {ticker}: TIME CAP at {held_h:.1f}h — closed @ {fill:.4f} "
                f"pnl={pnl:+.2f} ({pnl_pct:+.2f}%)"
            )
            if notifier:
                notifier.push(
                    fmt_trade_closed(
                        ticker=ticker,
                        direction=direction,
                        lots=lots,
                        entry=entry,
                        exit_=fill,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        reason="time_cap",
                    )
                )


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    settings = FutSettings()
    setup_logging(settings)
    logger = logging.getLogger("futbot.main")

    mode = "PAPER" if settings.FUTBOT_PAPER_MODE else "LIVE"
    # Surface env source so it's obvious if the wrong .env got picked up
    from futbot.config import _ENV_FILE  # noqa: E402

    logger.info(f"futbot starting in {mode} mode (env: {_ENV_FILE})")
    if not settings.FUTBOT_PAPER_MODE:
        logger.warning(
            "Live mode is ON.  Real orders will be placed.  "
            "Stop the bot now (Ctrl-C) if this is unintended."
        )

    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="futbot",
    )
    await broker.connect()
    logger.info("Broker connected")

    db = FutbotDB(settings.FUTBOT_DB_PATH)
    await db.initialize()

    notifier = TelegramNotifier(
        token=settings.TELEGRAM_BOT_TOKEN,
        chat_id=settings.TELEGRAM_CHAT_ID,
        paper=bool(settings.FUTBOT_PAPER_MODE),
    )
    await notifier.start()

    circuit = CircuitBreaker(settings)

    universe: list[Contract] = []
    last_universe_refresh = datetime.min.replace(tzinfo=timezone.utc)
    boot_sent = False

    shutdown = asyncio.Event()

    def _sig(*_):
        logger.info("Shutdown signal received")
        shutdown.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(s, _sig)
        except NotImplementedError:
            pass  # Windows

    try:
        while not shutdown.is_set():
            now = datetime.now(timezone.utc)
            # Refresh universe every 6h
            if (now - last_universe_refresh).total_seconds() > 6 * 3600:
                logger.info("Refreshing universe…")
                try:
                    universe = await resolve_universe(broker, settings)
                    last_universe_refresh = now
                    if not boot_sent:
                        notifier.push(fmt_boot(mode=mode, universe_count=len(universe)))
                        boot_sent = True
                except Exception as e:
                    logger.exception(f"Universe refresh failed: {e}")

            # Manage existing positions first (close before we open new ones
            # — frees ГО budget for fresh entries this tick)
            try:
                await _manage_open(
                    broker=broker,
                    db=db,
                    settings=settings,
                    logger=logger,
                    notifier=notifier,
                )
            except Exception as e:
                logger.exception(f"manage_open failed: {e}")

            # Kill-switch
            tripped, reason = await circuit.is_tripped(broker=broker, db=db)
            if tripped:
                if not getattr(circuit, "_notified_today", False):
                    notifier.push(fmt_circuit(reason))
                    circuit._notified_today = True
                logger.warning(f"Circuit breaker active — no new entries: {reason}")
            else:
                circuit._notified_today = False
                # Walk universe, evaluate each contract
                for c in universe:
                    try:
                        tf_data = await fetch_multi_tf(broker, c.figi)
                        await _maybe_open(
                            broker=broker,
                            db=db,
                            contract=c,
                            tf_data=tf_data,
                            settings=settings,
                            logger=logger,
                            notifier=notifier,
                        )
                    except Exception as e:
                        logger.exception(f"  {c.ticker}: maybe_open failed: {e}")

            # Sleep for the loop interval
            try:
                await asyncio.wait_for(
                    shutdown.wait(),
                    timeout=int(settings.FUTBOT_LOOP_SECONDS),
                )
            except asyncio.TimeoutError:
                pass

    finally:
        logger.info("Shutting down…")
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
        logger.info("futbot stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
