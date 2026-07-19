"""PairsBot — class wrapper around pairs strategy for use by orchestrator.

Mirrors the logic of `futbot.pairs.main` but accepts shared broker +
notifier from outside, exposing setup/tick/shutdown/status methods.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from t_tech.invest import CandleInterval
from t_tech.invest.utils import quotation_to_decimal

from futbot.pairs.config import PairsSettings
from futbot.pairs.db import PairsDB
from futbot.pairs import cointegration as coint
from futbot.pairs import strategy as strat
from futbot.pairs import execution as exe
from futbot.universe import resolve_universe
from futbot.config import FutSettings
from futbot.telegram_notifier import Msg, MsgType

logger = logging.getLogger("orchestrator.pairs")


def _bases_for_pairs(pair_strings: list[str]) -> list[str]:
    bases = set()
    for p in pair_strings:
        y, x = p.split("-")
        bases.add(y)
        bases.add(x)
    return sorted(bases)


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
        logger.warning(f"  fetch {figi}: {e}")
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
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


class PairsBot:
    name = "pairs"

    def __init__(self):
        self.settings = PairsSettings()
        self.db: PairsDB | None = None
        self.broker = None
        self.notifier = None
        self.universe = []
        self.by_base = {}
        self.portfolio_value = 0.0
        self._initialised = False

    @property
    def mode(self) -> str:
        return "PAPER" if self.settings.PAIRS_PAPER_MODE else "LIVE"

    async def setup(self, broker, notifier):
        """One-time bootstrap: db, universe, portfolio_value, reconcile."""
        self.broker = broker
        self.notifier = notifier
        self.db = PairsDB(self.settings.PAIRS_DB_PATH)
        await self.db.initialize()

        bases = _bases_for_pairs(list(self.settings.PAIRS_LIST))
        fs = FutSettings()
        fs.FUTBOT_TIER1_BASES = bases
        fs.FUTBOT_TIER2_BASES = []
        self.universe = await resolve_universe(broker, fs)
        if not self.universe:
            logger.error("No pair contracts resolved")
            return False
        self.by_base = {c.base: c for c in self.universe}
        logger.info(
            f"[pairs] Resolved {len(self.universe)} contracts: "
            f"{', '.join(f'{c.base}({c.ticker})' for c in self.universe)}"
        )

        try:
            self.portfolio_value = float(await broker.get_portfolio_value())
        except Exception:
            self.portfolio_value = 100_000.0
        logger.info(f"[pairs] Portfolio: {self.portfolio_value:.0f} ₽")

        await self._reconcile()

        if self.notifier:
            self.notifier.push(
                Msg(
                    MsgType.BOOT,
                    f"Pairs subsystem — {self.mode}",
                    f"Universe: {len(self.universe)} contracts\n"
                    f"Pairs: {', '.join(self.settings.PAIRS_LIST)}\n"
                    f"z_entry={self.settings.PAIRS_Z_ENTRY} z_stop={self.settings.PAIRS_Z_STOP} "
                    f"max_hold={self.settings.PAIRS_MAX_HOLD_HOURS}h",
                )
            )
        self._initialised = True
        return True

    async def _reconcile(self):
        open_rows = await self.db.open_trades()
        if not open_rows:
            return
        max_age_sec = self.settings.PAIRS_MAX_HOLD_HOURS * 3600 * 4
        for r in open_rows:
            entry_dt = datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00"))
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - entry_dt).total_seconds()
            if age <= max_age_sec:
                continue
            logger.warning(
                f"[pairs] RECONCILE {r['pair']} age {age/3600:.1f}h > "
                f"{max_age_sec/3600:.0f}h — force-closing"
            )
            try:
                exit_y = float(await self.broker.get_last_price(r["figi_y"]))
                exit_x = float(await self.broker.get_last_price(r["figi_x"]))
            except Exception:
                exit_y = float(r["entry_y_price"])
                exit_x = float(r["entry_x_price"])
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
            await self.db.close_trade(
                r["id"],
                exit_y_price=exit_y,
                exit_x_price=exit_x,
                exit_time=datetime.utcnow().isoformat(),
                exit_reason="boot_reconcile_stale",
                pnl=pnl["pnl_pct"],
                pnl_rub=pnl["net_rub"],
                commission_rub=pnl["commission_rub"],
            )

    async def tick(self):
        """One full evaluation cycle — process all pairs."""
        if not self._initialised:
            return
        t_start = datetime.now(timezone.utc)
        logger.info(f"[pairs] ── Tick at {t_start.isoformat()[:19]} ──")

        # Daily kill
        today_iso = t_start.date().isoformat()
        daily_pnl = await self.db.daily_pnl_rub(today_iso)
        loss_cap = self.portfolio_value * float(self.settings.PAIRS_DAILY_LOSS_PCT_LIMIT)
        kill_active = daily_pnl < -loss_cap
        if kill_active:
            logger.warning(f"[pairs] Daily kill active: P&L {daily_pnl:+.0f}₽ < -{loss_cap:.0f}₽")

        # Fetch hourly bars
        lookback_hours = max(
            self.settings.PAIRS_FIT_LOOKBACK_DAYS * 24,
            self.settings.PAIRS_ROLLING_Z_WINDOW_HOURS + 50,
        )
        prices_by_base: dict[str, pd.Series] = {}
        for c in self.universe:
            df = await _fetch_hourly(self.broker, c.figi, lookback_hours)
            if df.empty:
                continue
            prices_by_base[c.base] = df.set_index("time")["close"]

        # Process each pair
        for pair_name in self.settings.PAIRS_LIST:
            await self._process_pair(pair_name, prices_by_base, t_start, kill_active)

    async def _process_pair(self, pair_name: str, prices_by_base, t_start, kill_active):
        base_y, base_x = pair_name.split("-")
        if base_y not in prices_by_base or base_x not in prices_by_base:
            return
        py = prices_by_base[base_y]
        px = prices_by_base[base_x]

        # Re-fit β every PAIRS_REFIT_BETA_HOURS
        cached = await self.db.get_pair_state(pair_name)
        need_refit = cached is None
        if cached and cached["last_refit"]:
            last_refit_dt = datetime.fromisoformat(cached["last_refit"].replace("Z", "+00:00"))
            if last_refit_dt.tzinfo is None:
                last_refit_dt = last_refit_dt.replace(tzinfo=timezone.utc)
            if (
                t_start - last_refit_dt
            ).total_seconds() / 3600 >= self.settings.PAIRS_REFIT_BETA_HOURS:
                need_refit = True

        if need_refit:
            fit = coint.fit_pair(py, px, pair_name=pair_name)
            if fit is None:
                return
            await self.db.upsert_pair_state(
                pair=pair_name,
                beta=fit.beta,
                alpha=fit.alpha,
                adf_p=fit.adf_p,
                spread_mean=fit.spread_mean,
                spread_std=fit.spread_std,
            )
            logger.info(
                f"[pairs]   {pair_name}: refit β={fit.beta:+.4f} "
                f"adf_p={fit.adf_p:.4f} (n={fit.n_bars})"
            )
            cached = await self.db.get_pair_state(pair_name)

        beta = float(cached["beta"])
        adf_p = float(cached["adf_p"])

        aligned = pd.concat([py, px], axis=1, join="inner").dropna()
        if len(aligned) < self.settings.PAIRS_ROLLING_Z_WINDOW_HOURS + 5:
            return
        yv = aligned.iloc[:, 0].values
        xv = aligned.iloc[:, 1].values
        spread = yv - beta * xv
        rolling = spread[-self.settings.PAIRS_ROLLING_Z_WINDOW_HOURS :]
        roll_mean = float(rolling.mean())
        roll_std = float(rolling.std())
        if roll_std <= 0:
            return
        z_now = (spread[-1] - roll_mean) / roll_std
        y_now = float(yv[-1])
        x_now = float(xv[-1])

        open_t = await self.db.open_trade_for_pair(pair_name)

        if open_t is None:
            if kill_active:
                return
            ent = strat.should_open(
                z=z_now,
                z_entry=self.settings.PAIRS_Z_ENTRY,
                adf_p=adf_p,
                max_adf_p=self.settings.PAIRS_MAX_ADF_PVALUE,
            )
            if ent.open_side is None:
                return

            contract_y = self.by_base[base_y]
            contract_x = self.by_base[base_x]
            meta_y = self.broker.extract_futures_metadata(contract_y.instrument)
            meta_x = self.broker.extract_futures_metadata(contract_x.instrument)
            rpp_y = float(meta_y.get("rub_per_point") or 1.0)
            rpp_x = float(meta_x.get("rub_per_point") or 1.0)
            ls_y = int(getattr(contract_y.instrument, "lot", 1) or 1)
            ls_x = int(getattr(contract_x.instrument, "lot", 1) or 1)
            # Margin ratios (dlong/dshort) — 0.0 if broker didn't expose them;
            # compute_lots falls back to 25% conservative estimate then.
            dlong_y = float(meta_y.get("dlong") or 0.0)
            dshort_y = float(meta_y.get("dshort") or 0.0)
            dlong_x = float(meta_x.get("dlong") or 0.0)
            dshort_x = float(meta_x.get("dshort") or 0.0)

            lots_y, lots_x = exe.compute_lots(
                portfolio_value=self.portfolio_value,
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
                capital_per_pair_pct=float(self.settings.PAIRS_CAPITAL_PER_PAIR_PCT),
            )
            if lots_y <= 0 or lots_x <= 0:
                per_leg = self.portfolio_value * float(self.settings.PAIRS_CAPITAL_PER_PAIR_PCT) / 2
                logger.warning(
                    f"[pairs]   {pair_name}: SKIP — leg margin exceeds "
                    f"per-leg budget ({per_leg:.0f} ₽); "
                    f"y_not={y_now * ls_y * rpp_y:.0f} "
                    f"(margin~{(y_now*ls_y*rpp_y)*max(dlong_y,0.25):.0f}), "
                    f"x_not={x_now * ls_x * rpp_x:.0f} "
                    f"(margin~{(x_now*ls_x*rpp_x)*max(dlong_x,0.25):.0f}), "
                    f"β={beta:+.3f}"
                )
                return
            oid_y, oid_x, fill_y, fill_x = await exe.place_two_leg_entry(
                broker=self.broker,
                figi_y=contract_y.figi,
                figi_x=contract_x.figi,
                ticker_y=contract_y.ticker,
                ticker_x=contract_x.ticker,
                direction=ent.open_side,
                lots_y=lots_y,
                lots_x=lots_x,
                paper=bool(self.settings.PAIRS_PAPER_MODE),
            )
            spread_entry = fill_y - beta * fill_x
            tid = await self.db.insert_trade(
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
                paper=int(bool(self.settings.PAIRS_PAPER_MODE)),
            )
            logger.info(
                f"[pairs]   {pair_name}: OPENED {'LONG' if ent.open_side > 0 else 'SHORT'} "
                f"spread @ z={z_now:+.2f}  lots={lots_y}/{lots_x}  trade_id={tid}"
            )
            if self.notifier:
                self.notifier.push(
                    Msg(
                        MsgType.TRADE_OPENED,
                        f"PAIR {pair_name} " f"{'LONG' if ent.open_side > 0 else 'SHORT'}",
                        f"z={z_now:+.2f} (entry threshold {self.settings.PAIRS_Z_ENTRY})\n"
                        f"y-leg: {'BUY' if ent.open_side > 0 else 'SELL'} {lots_y}×{contract_y.ticker} @ {fill_y:.4f}\n"
                        f"x-leg: {'SELL' if ent.open_side > 0 else 'BUY'} {lots_x}×{contract_x.ticker} @ {fill_x:.4f}\n"
                        f"β={beta:+.4f}",
                    )
                )
        else:
            # Existing open — check close conditions
            entry_dt = datetime.fromisoformat(open_t["entry_time"].replace("Z", "+00:00"))
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            held_h = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600

            decision = strat.should_close(
                position_side=open_t["direction"],
                z=z_now,
                held_hours=held_h,
                z_stop=self.settings.PAIRS_Z_STOP,
                max_hold_hours=self.settings.PAIRS_MAX_HOLD_HOURS,
            )

            if not decision.close:
                return

            contract_y = self.by_base[base_y]
            contract_x = self.by_base[base_x]
            meta_y = self.broker.extract_futures_metadata(contract_y.instrument)
            meta_x = self.broker.extract_futures_metadata(contract_x.instrument)
            rpp_y = float(meta_y.get("rub_per_point") or 1.0)
            rpp_x = float(meta_x.get("rub_per_point") or 1.0)
            ls_y = int(getattr(contract_y.instrument, "lot", 1) or 1)
            ls_x = int(getattr(contract_x.instrument, "lot", 1) or 1)

            oid_y, oid_x, fill_y, fill_x = await exe.place_two_leg_exit(
                broker=self.broker,
                figi_y=contract_y.figi,
                figi_x=contract_x.figi,
                ticker_y=contract_y.ticker,
                ticker_x=contract_x.ticker,
                direction=open_t["direction"],
                lots_y=open_t["lots_y"],
                lots_x=open_t["lots_x"],
                reason=decision.reason,
                paper=bool(self.settings.PAIRS_PAPER_MODE),
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
            await self.db.close_trade(
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
                f"[pairs]   {pair_name}: CLOSED ({decision.reason}) "
                f"z={z_now:+.2f} held={held_h:.1f}h  "
                f"P&L: {pnl['net_rub']:+.2f}₽ ({pnl['pnl_pct']:+.3f}%)"
            )
            if self.notifier:
                self.notifier.push(
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

    async def status(self) -> str:
        """Render /pairs status text."""
        if not self._initialised or not self.db:
            return "Pairs subsystem not initialised."

        open_t = await self.db.open_trades()
        today_iso = datetime.utcnow().date().isoformat()
        today_pnl = await self.db.daily_pnl_rub(today_iso)

        lines = [f"<b>PAIRS ({self.mode})</b>"]
        lines.append(f"Pairs configured: {len(self.settings.PAIRS_LIST)}")
        lines.append(f"Open positions: {len(open_t)}")
        for r in open_t:
            entry_dt = datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00"))
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
            side = "LONG" if r["direction"] > 0 else "SHORT"
            lines.append(f"  • {r['pair']} {side} z={r['entry_z']:+.2f} " f"held={held:.1f}h")
        lines.append(f"Today realised P&L: <b>{today_pnl:+.2f} ₽</b>")

        # 7-day stats (exclude over-leveraged historical trades)
        async with self.db._lock:
            cur = await self.db._db.execute(
                "SELECT COUNT(*) n, "
                "SUM(CASE WHEN pnl_rub > 0 THEN 1 ELSE 0 END) wins, "
                "COALESCE(SUM(pnl_rub), 0) total "
                "FROM pair_trades WHERE exit_time >= ? "
                "AND (exit_reason IS NULL "
                "     OR (exit_reason NOT LIKE '%_OVERLEV' "
                "         AND exit_reason NOT LIKE 'sizing_bug_%'))",
                ((datetime.utcnow() - timedelta(days=7)).isoformat(),),
            )
            row = await cur.fetchone()
        if row and row[0]:
            wr = row[1] / row[0] * 100
            lines.append(f"7-day: {row[0]} closed, win {wr:.0f}%, NET {row[2]:+.2f} ₽")
        return "\n".join(lines)

    async def shutdown(self):
        if self.db:
            try:
                await self.db.close()
            except Exception:
                pass
