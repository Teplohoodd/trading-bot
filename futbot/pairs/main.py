"""futbot.pairs.main — hourly evaluation loop.

Run:   python -m futbot.pairs.main

Defaults to PAPER mode.  Hourly loop:
  1. Refresh price candles for all bases (REST, 240h lookback).
  2. Re-fit β / α / spread stats per pair (weekly) or use cached.
  3. For each pair:
     * If FLAT: check entry signal (|z| > z_entry).
     * If OPEN: check exit triggers (mean_rev / horizon / stop).
  4. Telegram alerts on every open/close.
  5. Daily kill-switch on −2 % drawdown.

Reconcile on boot: any DB-open trade older than 4× max_hold gets
force-closed; everything else gets re-hydrated.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.broker import BrokerClient  # noqa: E402
from tinkoff.invest import CandleInterval  # noqa: E402
from tinkoff.invest.utils import quotation_to_decimal  # noqa: E402

from futbot.pairs.config import PairsSettings  # noqa: E402
from futbot.pairs.db import PairsDB  # noqa: E402
from futbot.pairs import cointegration as coint  # noqa: E402
from futbot.pairs import strategy as strat  # noqa: E402
from futbot.pairs import execution as exe  # noqa: E402
from futbot.universe import resolve_universe  # noqa: E402
from futbot.config import FutSettings  # noqa: E402
from futbot.telegram_notifier import (  # noqa: E402
    TelegramNotifier,
    Msg,
    MsgType,
)


# Universe — pairs use the same 'BR/GZ/SR/LK/MX/Si' bases as scalp + futbot.
# Union of all bases in PAIRS_LIST is what we need to resolve.
def _bases_for_pairs(pair_strings: list[str]) -> list[str]:
    bases = set()
    for p in pair_strings:
        y, x = p.split("-")
        bases.add(y)
        bases.add(x)
    return sorted(bases)


def setup_logging(settings: PairsSettings):
    settings.PAIRS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.PAIRS_LOG_PATH, encoding="utf-8"),
        ],
    )
    for n in ("httpx", "telegram", "tinkoff", "grpc"):
        logging.getLogger(n).setLevel(logging.WARNING)


async def _fetch_hourly(broker, figi: str, hours_back: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    try:
        candles = await broker.get_candles(
            figi,
            now - timedelta(hours=hours_back + 10),
            now,
            interval=CandleInterval.CANDLE_INTERVAL_HOUR,
        )
    except Exception as e:
        logging.getLogger("futbot.pairs").warning(f"  fetch {figi}: {e}")
        return pd.DataFrame()
    rows = [
        {
            "time": c.time,
            "close": float(quotation_to_decimal(c.close)),
        }
        for c in candles
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return df


async def main():
    settings = PairsSettings()
    setup_logging(settings)
    logger = logging.getLogger("futbot.pairs")

    mode = "PAPER" if settings.PAIRS_PAPER_MODE else "LIVE"
    logger.info(f"pairs bot starting in {mode} mode")
    if not settings.PAIRS_PAPER_MODE:
        logger.warning(
            "LIVE mode is ON.  Pair trades involve TWO simultaneous orders.  "
            "Stop now (Ctrl-C) if this was unintended."
        )

    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="futbot-pairs",
    )
    await broker.connect()

    db = PairsDB(settings.PAIRS_DB_PATH)
    await db.initialize()

    notifier = TelegramNotifier(
        token=settings.TELEGRAM_BOT_TOKEN,
        chat_id=settings.TELEGRAM_CHAT_ID,
        paper=bool(settings.PAIRS_PAPER_MODE),
    )
    await notifier.start()

    # ── Resolve universe via futbot.universe (re-uses contract picker) ─
    bases = _bases_for_pairs(list(settings.PAIRS_LIST))
    fs = FutSettings()
    fs.FUTBOT_TIER1_BASES = bases
    fs.FUTBOT_TIER2_BASES = []
    universe = await resolve_universe(broker, fs)
    if not universe:
        logger.error("No contracts resolved — exiting")
        return
    by_base = {c.base: c for c in universe}
    logger.info(
        f"Resolved {len(universe)} contracts: "
        f"{', '.join(f'{c.base}({c.ticker})' for c in universe)}"
    )

    # ── Portfolio value snapshot ────────────────────────────────────────
    try:
        portfolio_value = float(await broker.get_portfolio_value())
    except Exception:
        portfolio_value = 100_000.0  # paper fallback
    logger.info(f"Portfolio value: {portfolio_value:.0f} ₽")

    # ── Reconcile open trades from DB ───────────────────────────────────
    open_rows = await db.open_trades()
    if open_rows:
        max_age_sec = settings.PAIRS_MAX_HOLD_HOURS * 3600 * 4
        for r in open_rows:
            entry_dt = datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00"))
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - entry_dt).total_seconds()
            if age > max_age_sec:
                logger.warning(
                    f"  RECONCILE pair {r['pair']} age {age/3600:.1f}h > "
                    f"{max_age_sec/3600:.0f}h — force-closing"
                )
                # Force-close at last price for both legs
                try:
                    exit_y = float(await broker.get_last_price(r["figi_y"]))
                    exit_x = float(await broker.get_last_price(r["figi_x"]))
                except Exception:
                    exit_y = float(r["entry_y_price"])
                    exit_x = float(r["entry_x_price"])
                # Rough P&L (paper accounting)
                pnl = exe.compute_two_leg_pnl(
                    direction=r["direction"],
                    beta=r["beta"],
                    entry_y=r["entry_y_price"],
                    entry_x=r["entry_x_price"],
                    exit_y=exit_y,
                    exit_x=exit_x,
                    lots_y=r["lots_y"],
                    lots_x=r["lots_x"],
                    rpp_y=1.0,
                    rpp_x=1.0,
                    lot_size_y=1,
                    lot_size_x=1,
                    base_y=r["base_y"],
                    base_x=r["base_x"],
                )
                await db.close_trade(
                    r["id"],
                    exit_y_price=exit_y,
                    exit_x_price=exit_x,
                    exit_time=datetime.utcnow().isoformat(),
                    exit_reason="boot_reconcile_stale",
                    pnl=pnl["pnl_pct"],
                    pnl_rub=pnl["net_rub"],
                    commission_rub=pnl["commission_rub"],
                )

    # ── Shutdown plumbing ───────────────────────────────────────────────
    shutdown = asyncio.Event()

    def _sig(*_):
        logger.info("Shutdown signal received")
        shutdown.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(s, _sig)
        except NotImplementedError:
            pass

    # ── Boot Telegram ──────────────────────────────────────────────────
    notifier.push(
        Msg(
            MsgType.BOOT,
            f"Pairs bot — {mode}",
            f"Universe: {len(universe)} contracts\n"
            f"Pairs: {', '.join(settings.PAIRS_LIST)}\n"
            f"z_entry={settings.PAIRS_Z_ENTRY} z_stop={settings.PAIRS_Z_STOP} "
            f"max_hold={settings.PAIRS_MAX_HOLD_HOURS}h",
        )
    )

    # Cache last β-refit timestamp per pair
    last_refit_ts: dict[str, datetime] = {}

    # ── Main loop ───────────────────────────────────────────────────────
    try:
        while not shutdown.is_set():
            t_start = datetime.now(timezone.utc)
            logger.info(f"── Tick at {t_start.isoformat()[:19]} ──")

            # ── Daily kill check ─────────────────────────────────────
            today_iso = t_start.date().isoformat()
            daily_pnl = await db.daily_pnl_rub(today_iso)
            loss_cap = portfolio_value * float(settings.PAIRS_DAILY_LOSS_PCT_LIMIT)
            if daily_pnl < -loss_cap:
                logger.warning(
                    f"Daily kill active: P&L {daily_pnl:+.0f} ₽ < -{loss_cap:.0f} ₽ — "
                    f"no new entries this UTC day"
                )
                kill_active = True
            else:
                kill_active = False

            # ── Fetch hourly bars for ALL bases (one shared lookback) ─
            lookback_hours = max(
                settings.PAIRS_FIT_LOOKBACK_DAYS * 24,
                settings.PAIRS_ROLLING_Z_WINDOW_HOURS + 50,
            )
            prices_by_base: dict[str, pd.Series] = {}
            for c in universe:
                df = await _fetch_hourly(broker, c.figi, lookback_hours)
                if df.empty:
                    continue
                prices_by_base[c.base] = df.set_index("time")["close"]

            # Process each pair
            for pair_name in settings.PAIRS_LIST:
                base_y, base_x = pair_name.split("-")
                if base_y not in prices_by_base or base_x not in prices_by_base:
                    logger.warning(f"  {pair_name}: missing data — skip")
                    continue
                py = prices_by_base[base_y]
                px = prices_by_base[base_x]

                # Re-fit β every PAIRS_REFIT_BETA_HOURS or if no cached state
                cached = await db.get_pair_state(pair_name)
                need_refit = cached is None
                if cached and cached["last_refit"]:
                    last_refit_dt = datetime.fromisoformat(
                        cached["last_refit"].replace("Z", "+00:00")
                    )
                    if last_refit_dt.tzinfo is None:
                        last_refit_dt = last_refit_dt.replace(tzinfo=timezone.utc)
                    if (
                        t_start - last_refit_dt
                    ).total_seconds() / 3600 >= settings.PAIRS_REFIT_BETA_HOURS:
                        need_refit = True

                if need_refit:
                    fit = coint.fit_pair(py, px, pair_name=pair_name)
                    if fit is None:
                        logger.warning(f"  {pair_name}: fit failed")
                        continue
                    await db.upsert_pair_state(
                        pair=pair_name,
                        beta=fit.beta,
                        alpha=fit.alpha,
                        adf_p=fit.adf_p,
                        spread_mean=fit.spread_mean,
                        spread_std=fit.spread_std,
                    )
                    logger.info(
                        f"  {pair_name}: refit β={fit.beta:+.4f} α={fit.alpha:+.2f} "
                        f"adf_p={fit.adf_p:.4f} (n={fit.n_bars})"
                    )
                    cached = await db.get_pair_state(pair_name)

                beta = float(cached["beta"])
                spread_mean = float(cached["spread_mean"])
                spread_std = float(cached["spread_std"])
                adf_p = float(cached["adf_p"])

                # Current z using ROLLING window (more responsive than full-sample stats)
                aligned = pd.concat([py, px], axis=1, join="inner").dropna()
                if len(aligned) < settings.PAIRS_ROLLING_Z_WINDOW_HOURS + 5:
                    logger.info(f"  {pair_name}: warming up ({len(aligned)} bars)")
                    continue
                yv = aligned.iloc[:, 0].values
                xv = aligned.iloc[:, 1].values
                spread = yv - beta * xv
                rolling = spread[-settings.PAIRS_ROLLING_Z_WINDOW_HOURS :]
                roll_mean = float(rolling.mean())
                roll_std = float(rolling.std())
                if roll_std <= 0:
                    continue
                z_now = (spread[-1] - roll_mean) / roll_std
                y_now = float(yv[-1])
                x_now = float(xv[-1])

                # Is there an open position on this pair?
                open_t = await db.open_trade_for_pair(pair_name)

                if open_t is None:
                    # ── Maybe open ───────────────────────────────────────
                    if kill_active:
                        continue
                    ent = strat.should_open(
                        z=z_now,
                        z_entry=settings.PAIRS_Z_ENTRY,
                        adf_p=adf_p,
                        max_adf_p=settings.PAIRS_MAX_ADF_PVALUE,
                    )
                    if ent.open_side is None:
                        logger.info(f"  {pair_name}: z={z_now:+.2f} hold — {ent.reason}")
                        continue

                    contract_y = by_base[base_y]
                    contract_x = by_base[base_x]
                    meta_y = broker.extract_futures_metadata(contract_y.instrument)
                    meta_x = broker.extract_futures_metadata(contract_x.instrument)
                    rpp_y = float(meta_y.get("rub_per_point") or 1.0)
                    rpp_x = float(meta_x.get("rub_per_point") or 1.0)
                    ls_y = int(getattr(contract_y.instrument, "lot", 1) or 1)
                    ls_x = int(getattr(contract_x.instrument, "lot", 1) or 1)
                    dlong_y = float(meta_y.get("dlong") or 0.0)
                    dshort_y = float(meta_y.get("dshort") or 0.0)
                    dlong_x = float(meta_x.get("dlong") or 0.0)
                    dshort_x = float(meta_x.get("dshort") or 0.0)

                    lots_y, lots_x = exe.compute_lots(
                        portfolio_value=portfolio_value,
                        beta=beta,
                        price_y=y_now,
                        price_x=x_now,
                        rpp_y=rpp_y,
                        rpp_x=rpp_x,
                        lot_size_y=ls_y,
                        lot_size_x=ls_x,
                        dlong_y=dlong_y,
                        dshort_y=dshort_y,
                        dlong_x=dlong_x,
                        dshort_x=dshort_x,
                        direction=ent.open_side,
                        capital_per_pair_pct=float(settings.PAIRS_CAPITAL_PER_PAIR_PCT),
                    )
                    if lots_y <= 0 or lots_x <= 0:
                        per_leg = portfolio_value * float(settings.PAIRS_CAPITAL_PER_PAIR_PCT) / 2
                        logger.warning(
                            f"  {pair_name}: SKIP — leg margin exceeds "
                            f"per_leg budget {per_leg:.0f} ₽ (β={beta:+.3f})"
                        )
                        continue

                    oid_y, oid_x, fill_y, fill_x = await exe.place_two_leg_entry(
                        broker=broker,
                        figi_y=contract_y.figi,
                        figi_x=contract_x.figi,
                        ticker_y=contract_y.ticker,
                        ticker_x=contract_x.ticker,
                        direction=ent.open_side,
                        lots_y=lots_y,
                        lots_x=lots_x,
                        paper=bool(settings.PAIRS_PAPER_MODE),
                    )
                    spread_entry = fill_y - beta * fill_x
                    tid = await db.insert_trade(
                        pair=pair_name,
                        base_y=base_y,
                        base_x=base_x,
                        figi_y=contract_y.figi,
                        figi_x=contract_x.figi,
                        direction=ent.open_side,
                        lots_y=lots_y,
                        lots_x=lots_x,
                        beta=beta,
                        entry_y_price=fill_y,
                        entry_x_price=fill_x,
                        entry_z=z_now,
                        entry_time=datetime.utcnow().isoformat(),
                        spread_entry=spread_entry,
                        paper=int(bool(settings.PAIRS_PAPER_MODE)),
                    )
                    logger.info(
                        f"  {pair_name}: OPENED {'LONG' if ent.open_side > 0 else 'SHORT'} "
                        f"spread @ z={z_now:+.2f}  y={fill_y:.4f} x={fill_x:.4f}  "
                        f"lots={lots_y}/{lots_x}  trade_id={tid}"
                    )
                    notifier.push(
                        Msg(
                            MsgType.TRADE_OPENED,
                            f"PAIR {pair_name} " f"{'LONG' if ent.open_side > 0 else 'SHORT'}",
                            f"z={z_now:+.2f} (entry threshold {settings.PAIRS_Z_ENTRY})\n"
                            f"y-leg: {ent.open_side>0 and 'BUY' or 'SELL'} {lots_y}×{contract_y.ticker} @ {fill_y:.4f}\n"
                            f"x-leg: {ent.open_side>0 and 'SELL' or 'BUY'} {lots_x}×{contract_x.ticker} @ {fill_x:.4f}\n"
                            f"β={beta:+.4f}",
                        )
                    )
                else:
                    # ── Maybe close ──────────────────────────────────────
                    entry_dt = datetime.fromisoformat(open_t["entry_time"].replace("Z", "+00:00"))
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    held_h = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600

                    decision = strat.should_close(
                        position_side=open_t["direction"],
                        z=z_now,
                        held_hours=held_h,
                        z_stop=settings.PAIRS_Z_STOP,
                        max_hold_hours=settings.PAIRS_MAX_HOLD_HOURS,
                    )

                    if not decision.close:
                        logger.info(
                            f"  {pair_name}: holding {open_t['direction']:+d} "
                            f"z={z_now:+.2f} ({held_h:.1f}h) — {decision.detail}"
                        )
                        continue

                    contract_y = by_base[base_y]
                    contract_x = by_base[base_x]
                    meta_y = broker.extract_futures_metadata(contract_y.instrument)
                    meta_x = broker.extract_futures_metadata(contract_x.instrument)
                    rpp_y = float(meta_y.get("rub_per_point") or 1.0)
                    rpp_x = float(meta_x.get("rub_per_point") or 1.0)
                    ls_y = int(getattr(contract_y.instrument, "lot", 1) or 1)
                    ls_x = int(getattr(contract_x.instrument, "lot", 1) or 1)

                    oid_y, oid_x, fill_y, fill_x = await exe.place_two_leg_exit(
                        broker=broker,
                        figi_y=contract_y.figi,
                        figi_x=contract_x.figi,
                        ticker_y=contract_y.ticker,
                        ticker_x=contract_x.ticker,
                        direction=open_t["direction"],
                        lots_y=open_t["lots_y"],
                        lots_x=open_t["lots_x"],
                        reason=decision.reason,
                        paper=bool(settings.PAIRS_PAPER_MODE),
                    )
                    pnl = exe.compute_two_leg_pnl(
                        direction=open_t["direction"],
                        beta=beta,
                        entry_y=open_t["entry_y_price"],
                        entry_x=open_t["entry_x_price"],
                        exit_y=fill_y,
                        exit_x=fill_x,
                        lots_y=open_t["lots_y"],
                        lots_x=open_t["lots_x"],
                        rpp_y=rpp_y,
                        rpp_x=rpp_x,
                        lot_size_y=ls_y,
                        lot_size_x=ls_x,
                        base_y=base_y,
                        base_x=base_x,
                    )
                    spread_exit = fill_y - beta * fill_x
                    await db.close_trade(
                        open_t["id"],
                        exit_y_price=fill_y,
                        exit_x_price=fill_x,
                        exit_z=z_now,
                        exit_time=datetime.utcnow().isoformat(),
                        exit_reason=decision.reason,
                        spread_exit=spread_exit,
                        pnl=pnl["pnl_pct"],
                        pnl_rub=pnl["net_rub"],
                        commission_rub=pnl["commission_rub"],
                    )
                    logger.info(
                        f"  {pair_name}: CLOSED ({decision.reason}) "
                        f"z={z_now:+.2f} held={held_h:.1f}h  "
                        f"P&L: {pnl['net_rub']:+.2f}₽ ({pnl['pnl_pct']:+.3f}%)"
                    )
                    notifier.push(
                        Msg(
                            MsgType.TRADE_CLOSED,
                            f"PAIR {pair_name} CLOSE ({decision.reason})",
                            f"z_entry={open_t['entry_z']:+.2f} → z_exit={z_now:+.2f}\n"
                            f"held: {held_h:.1f}h\n"
                            f"NET P&L: <b>{pnl['net_rub']:+.2f} ₽</b> "
                            f"({pnl['pnl_pct']:+.3f}%)\n"
                            f"commission: {pnl['commission_rub']:.2f} ₽",
                        )
                    )

            # ── Sleep until next tick ───────────────────────────────────
            try:
                await asyncio.wait_for(
                    shutdown.wait(),
                    timeout=int(settings.PAIRS_LOOP_SECONDS),
                )
            except asyncio.TimeoutError:
                pass

    finally:
        logger.info("Shutting down pairs bot…")
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
