"""CarryBot — Si (USD/RUB) calendar-spread basis mean-reversion.

Delta-neutral CIP-contango harvest.  Long 1 next-month + short 1 front-month
(or reverse) of the SAME underlying, so net USD/RUB delta ≈ 0 — P&L is the
change in basis (next − front).  We trade z-score mean-reversion of the basis
around its rolling mean.

Reuses pairs execution (it's a two-leg spread) and PairsDB (its schema fits:
base_y = next ticker, base_x = front ticker, beta = 1.0, entry_z = basis z).

Position lifecycle handles the ROLL: an open position references the specific
front/next FIGIs it was opened on (stored in DB); when the front nears expiry
we close on those FIGIs and the next tick opens on the new front/next pair.
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from tinkoff.invest import CandleInterval
from tinkoff.invest.utils import quotation_to_decimal

from futbot.carry.config import CarrySettings
from futbot.pairs.db import PairsDB
from futbot.pairs import execution as exe
from futbot.telegram_notifier import Msg, MsgType

logger = logging.getLogger("orchestrator.carry")


class CarryBot:
    name = "carry"

    def __init__(self):
        self.settings = CarrySettings()
        self.db: PairsDB | None = None
        self.broker = None
        self.notifier = None
        self.portfolio_value = 0.0
        self._gone_strikes: dict = {}  # debounce confirmed-absent reads
        self._initialised = False

    @property
    def mode(self) -> str:
        return "PAPER" if self.settings.CARRY_PAPER_MODE else "LIVE"

    # ── contract resolution ────────────────────────────────────────────
    async def _resolve_pair(self):
        """Return (front_fut, front_exp, next_fut, next_exp) for CARRY_BASE."""
        base = self.settings.CARRY_BASE
        futs = await self.broker.get_all_futures()
        now = datetime.now(timezone.utc)
        cands = []
        for f in futs:
            t = getattr(f, "ticker", "") or ""
            if not (t == base or (t.startswith(base) and len(t) == len(base) + 2)):
                continue
            exp = getattr(f, "expiration_date", None)
            if exp is None:
                continue
            if hasattr(exp, "ToDatetime"):
                exp = exp.ToDatetime()
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            cands.append((f, exp))
        cands.sort(key=lambda x: x[1])
        live = [(f, e) for f, e in cands if (e - now).days >= 1]
        if len(live) < 2:
            return None
        (ff, fe), (nf, ne) = live[0], live[1]
        return ff, fe, nf, ne

    async def _fetch_closes(self, figi: str, hours: int) -> pd.Series:
        now = datetime.now(timezone.utc)
        try:
            candles = await self.broker.get_candles(
                figi,
                now - timedelta(hours=hours + 24),
                now,
                interval=CandleInterval.CANDLE_INTERVAL_HOUR,
            )
        except Exception as e:
            logger.warning(f"  carry fetch {figi}: {e}")
            return pd.Series(dtype=float)
        rows = [{"time": c.time, "close": float(quotation_to_decimal(c.close))} for c in candles]
        if not rows:
            return pd.Series(dtype=float)
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").drop_duplicates("time", keep="last")
        return df.set_index("time")["close"]

    def _lots_for(self, price: float, rpp: float, lot: int, dlong: float) -> int:
        budget = self.portfolio_value * float(self.settings.CARRY_CAPITAL_PCT)
        margin = price * lot * rpp * (dlong if dlong > 0 else 0.10)
        if margin <= 0:
            return 0
        import math

        return max(1, min(self.settings.CARRY_MAX_LOTS, math.floor(budget / margin)))

    # ── lifecycle ───────────────────────────────────────────────────────
    async def setup(self, broker, notifier):
        self.broker = broker
        self.notifier = notifier
        self.db = PairsDB(self.settings.CARRY_DB_PATH)
        await self.db.initialize()

        pair = await self._resolve_pair()
        if pair is None:
            logger.error("[carry] could not resolve front/next contracts")
            return False
        ff, fe, nf, ne = pair
        logger.info(
            f"[carry] {self.settings.CARRY_BASE}: front {ff.ticker} "
            f"(exp {fe.date()}), next {nf.ticker} (exp {ne.date()})"
        )

        try:
            self.portfolio_value = float(await broker.get_portfolio_value())
        except Exception:
            self.portfolio_value = 100_000.0
        logger.info(f"[carry] Portfolio: {self.portfolio_value:.0f} ₽")

        # Reconcile any open carry spread against the broker (paper/live guard)
        self._initialised = True
        await self.reconcile()

        if self.notifier:
            self.notifier.push(
                Msg(
                    MsgType.BOOT,
                    f"Carry subsystem — {self.mode}",
                    f"Si calendar spread {nf.ticker}/{ff.ticker}\n"
                    f"z_entry={self.settings.CARRY_Z_ENTRY} "
                    f"z_stop={self.settings.CARRY_Z_STOP} "
                    f"max_hold={self.settings.CARRY_MAX_HOLD_HOURS}h",
                )
            )
        self._initialised = True
        return True

    async def _broker_positions(self) -> tuple[dict, bool]:
        """Return ({figi: qty}, ok).  ok=False ⇒ API failed; callers MUST NOT
        treat the empty dict as 'no positions' (would false-close real legs)."""
        try:
            pos = await self.broker.get_positions()
        except Exception as e:
            logger.warning(f"[carry] get_positions failed: {e} — skip reconcile")
            return {}, False
        out = {}
        for f in getattr(pos, "futures", []) or []:
            figi = getattr(f, "figi", None)
            try:
                q = int(getattr(f, "balance", 0))
            except Exception:
                q = 0
            if figi:
                out[figi] = q
        return out, True

    async def reconcile(self):
        """Heal paper/live desync + detect externally-closed carry spreads.

        Carry is two legs (figi_y/figi_x).  Rules mirror the trend bot:
          * paper trade while LIVE → purge in DB (no broker data needed).
          * live trade with BOTH legs absent at broker (DEBOUNCED, only on a
            successful read) → closed externally → mark closed in DB.
          * one leg present, the other gone → leg-risk → ALERT (manual).
        A failed get_positions NEVER closes anything (the 2026-06-02 bug).
        """
        rows = await self.db.open_trades()
        if not rows:
            return
        bpos, ok = await self._broker_positions()
        live = not bool(self.settings.CARRY_PAPER_MODE)
        GONE_STRIKES_NEEDED = 2
        for r in rows:
            if live and bool(r["paper"]):
                await self.db.close_trade(
                    r["id"],
                    exit_y_price=r["entry_y_price"],
                    exit_x_price=r["entry_x_price"],
                    exit_time=datetime.utcnow().isoformat(),
                    exit_reason="paper_purge_live_switch",
                    pnl=0.0,
                    pnl_rub=0.0,
                    commission_rub=0.0,
                )
                logger.warning(f"[carry] RECONCILE {r['pair']}: paper purged on live switch")
                continue
            if not (live and not bool(r["paper"])):
                continue
            if not ok:
                continue  # untrusted read — never act
            qy = bpos.get(r["figi_y"], 0)
            qx = bpos.get(r["figi_x"], 0)
            key = f"carry:{r['id']}"
            if qy == 0 and qx == 0:
                self._gone_strikes[key] = self._gone_strikes.get(key, 0) + 1
                if self._gone_strikes[key] < GONE_STRIKES_NEEDED:
                    logger.warning(
                        f"[carry] RECONCILE {r['pair']}: absent "
                        f"(strike {self._gone_strikes[key]}/{GONE_STRIKES_NEEDED})"
                    )
                    continue
                await self.db.close_trade(
                    r["id"],
                    exit_y_price=r["entry_y_price"],
                    exit_x_price=r["entry_x_price"],
                    exit_time=datetime.utcnow().isoformat(),
                    exit_reason="reconciled_gone_at_broker",
                    pnl=0.0,
                    pnl_rub=0.0,
                    commission_rub=0.0,
                )
                logger.warning(
                    f"[carry] RECONCILE {r['pair']}: both legs confirmed gone — closed in DB"
                )
                self._gone_strikes.pop(key, None)
            elif (qy == 0) != (qx == 0):
                logger.error(
                    f"[carry] ⚠ LEG RISK {r['pair']}: one leg missing "
                    f"(y={qy} x={qx}) — manual check"
                )
                if self.notifier:
                    self.notifier.push(
                        Msg(
                            MsgType.ERROR,
                            f"⚠ CARRY LEG RISK {r['pair']}",
                            f"One leg missing at broker (y={qy} x={qx}). "
                            f"Spread is no longer delta-neutral — check manually.",
                        )
                    )
            else:
                self._gone_strikes.pop(key, None)

    async def monitor(self):
        """Fast safety check (every ~5 min): reconcile against the broker.
        Carry's z-based exit stays on the hourly tick (needs the basis series);
        the urgent job here is orphan/desync detection."""
        if not self._initialised:
            return
        await self.reconcile()

    async def tick(self):
        if not self._initialised:
            return
        t_start = datetime.now(timezone.utc)
        logger.info(f"[carry] ── Tick {t_start.isoformat()[:19]} ──")

        # Daily kill
        today_iso = t_start.date().isoformat()
        daily = await self.db.daily_pnl_rub(today_iso)
        cap = self.portfolio_value * float(self.settings.CARRY_DAILY_LOSS_PCT_LIMIT)
        kill = daily < -cap
        if kill:
            logger.warning(f"[carry] daily kill: {daily:+.0f}₽ < -{cap:.0f}₽")

        win = int(self.settings.CARRY_ROLLING_Z_WINDOW_HOURS)
        open_t = await self.db.open_trades()
        open_row = open_t[0] if open_t else None

        if open_row is not None:
            await self._manage(open_row, t_start)
        elif not kill:
            await self._maybe_open(win, t_start)

    async def _basis_z(self, figi_next: str, figi_front: str, win: int):
        """Return (z_now, basis_now, next_now, front_now) or None.

        FORTS trades ~10-14h/day, so we need ~3-4× the window in CALENDAR
        hours to collect `win` trading-hour bars.  Fetch generously (capped
        under the 90-day single-request limit).
        """
        fetch_hours = min(int((win + 80) * 5), 90 * 24)  # ~60-66 days
        sn = await self._fetch_closes(figi_next, fetch_hours)
        sf = await self._fetch_closes(figi_front, fetch_hours)
        al = pd.concat([sn.rename("n"), sf.rename("f")], axis=1, join="inner").dropna()
        if len(al) < win + 5:
            return None
        basis = (al["n"] - al["f"]).values
        roll = basis[-win:]
        mean = float(roll.mean())
        sd = float(roll.std())
        if sd <= 0:
            return None
        z = (basis[-1] - mean) / sd
        return z, float(basis[-1]), float(al["n"].iloc[-1]), float(al["f"].iloc[-1])

    async def _maybe_open(self, win: int, t_start):
        pair = await self._resolve_pair()
        if pair is None:
            return
        ff, fe, nf, ne = pair
        # don't open if the front is about to expire
        if (fe - t_start).days < self.settings.CARRY_ROLL_DAYS_BEFORE_EXPIRY:
            logger.info(f"[carry] front {ff.ticker} near expiry — skip open")
            return
        bz = await self._basis_z(nf.figi, ff.figi, win)
        if bz is None:
            return
        z, basis, n_price, f_price = bz
        ze = float(self.settings.CARRY_Z_ENTRY)
        if abs(z) < ze:
            logger.info(f"[carry] z={z:+.2f} within ±{ze} — no entry")
            return
        # direction: +1 long basis (long next / short front), -1 short basis
        direction = +1 if z < 0 else -1

        meta = self.broker.extract_futures_metadata(ff)
        rpp = float(meta.get("rub_per_point") or 1.0)
        lot = int(getattr(ff, "lot", 1) or 1)
        dlong = float(meta.get("dlong") or 0.0)
        n_lots = self._lots_for(f_price, rpp, lot, dlong)
        if n_lots <= 0:
            logger.warning("[carry] sizing returned 0 — skip")
            return

        if self.settings.CARRY_USE_LIMIT_ORDERS:
            oid_y, oid_x, fill_n, fill_f = await exe.place_two_leg_limit_entry(
                broker=self.broker,
                figi_y=nf.figi,
                figi_x=ff.figi,
                ticker_y=nf.ticker,
                ticker_x=ff.ticker,
                direction=direction,
                lots_y=n_lots,
                lots_x=n_lots,
                paper=bool(self.settings.CARRY_PAPER_MODE),
                timeout=float(self.settings.CARRY_LIMIT_TIMEOUT_SEC),
            )
        else:
            oid_y, oid_x, fill_n, fill_f = await exe.place_two_leg_entry(
                broker=self.broker,
                figi_y=nf.figi,
                figi_x=ff.figi,
                ticker_y=nf.ticker,
                ticker_x=ff.ticker,
                direction=direction,
                lots_y=n_lots,
                lots_x=n_lots,
                paper=bool(self.settings.CARRY_PAPER_MODE),
            )
        tid = await self.db.insert_trade(
            pair=f"{nf.ticker}-{ff.ticker}",
            base_y=nf.ticker,
            base_x=ff.ticker,
            figi_y=nf.figi,
            figi_x=ff.figi,
            direction=direction,
            lots_y=n_lots,
            lots_x=n_lots,
            beta=1.0,
            entry_y_price=fill_n,
            entry_x_price=fill_f,
            entry_z=z,
            entry_time=datetime.utcnow().isoformat(),
            spread_entry=fill_n - fill_f,
            paper=int(bool(self.settings.CARRY_PAPER_MODE)),
        )
        logger.info(
            f"[carry] OPEN {'LONG' if direction>0 else 'SHORT'} basis "
            f"{nf.ticker}/{ff.ticker} z={z:+.2f} lots={n_lots} id={tid}"
        )
        if self.notifier:
            self.notifier.push(
                Msg(
                    MsgType.TRADE_OPENED,
                    f"CARRY {nf.ticker}/{ff.ticker} " f"{'LONG' if direction>0 else 'SHORT'} basis",
                    f"z={z:+.2f} (entry ±{ze})\n"
                    f"{'BUY' if direction>0 else 'SELL'} {n_lots}×{nf.ticker} @ {fill_n:.0f}\n"
                    f"{'SELL' if direction>0 else 'BUY'} {n_lots}×{ff.ticker} @ {fill_f:.0f}\n"
                    f"basis={fill_n-fill_f:+.0f}",
                )
            )

    async def _manage(self, row, t_start):
        win = int(self.settings.CARRY_ROLLING_Z_WINDOW_HOURS)
        bz = await self._basis_z(row["figi_y"], row["figi_x"], win)
        entry_dt = datetime.fromisoformat(row["entry_time"].replace("Z", "+00:00"))
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        held_h = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600

        # Resolve current expiry of the held front to detect roll
        roll_now = False
        try:
            ff_now = await self.broker.get_future_by_figi(row["figi_x"])
            exp = getattr(ff_now, "expiration_date", None)
            if exp is not None:
                if hasattr(exp, "ToDatetime"):
                    exp = exp.ToDatetime()
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if (exp - t_start).days < self.settings.CARRY_ROLL_DAYS_BEFORE_EXPIRY:
                    roll_now = True
        except Exception:
            pass

        reason = None
        z_now = None
        if bz is not None:
            z_now, _, _, _ = bz
            direction = row["direction"]
            crossed = (direction == +1 and z_now >= 0) or (direction == -1 and z_now <= 0)
            stopped = abs(z_now) >= float(self.settings.CARRY_Z_STOP)
            if crossed:
                reason = "mean_rev"
            elif stopped:
                reason = "stop"
        if reason is None and held_h >= self.settings.CARRY_MAX_HOLD_HOURS:
            reason = "horizon"
        if reason is None and roll_now:
            reason = "roll"
        if reason is None:
            logger.info(
                f"[carry] HOLD z={z_now if z_now is not None else float('nan'):+.2f} "
                f"held={held_h:.1f}h"
            )
            return

        # Close on the STORED figis (the contracts actually held).  Capture
        # the REAL executed fills returned by the executor (not a last-price
        # snapshot) so the P&L below reflects reality.
        if self.settings.CARRY_USE_LIMIT_ORDERS:
            _, _, fill_n, fill_f = await exe.place_two_leg_limit_exit(
                broker=self.broker,
                figi_y=row["figi_y"],
                figi_x=row["figi_x"],
                ticker_y=row["base_y"],
                ticker_x=row["base_x"],
                direction=row["direction"],
                lots_y=row["lots_y"],
                lots_x=row["lots_x"],
                reason=reason,
                paper=bool(self.settings.CARRY_PAPER_MODE),
                timeout=float(self.settings.CARRY_LIMIT_TIMEOUT_SEC),
            )
        else:
            _, _, fill_n, fill_f = await exe.place_two_leg_exit(
                broker=self.broker,
                figi_y=row["figi_y"],
                figi_x=row["figi_x"],
                ticker_y=row["base_y"],
                ticker_x=row["base_x"],
                direction=row["direction"],
                lots_y=row["lots_y"],
                lots_x=row["lots_x"],
                reason=reason,
                paper=bool(self.settings.CARRY_PAPER_MODE),
            )
        meta = self.broker.extract_futures_metadata(
            await self.broker.get_future_by_figi(row["figi_x"])
        )
        rpp = float(meta.get("rub_per_point") or 1.0)
        lot = int(meta.get("lot", 1) or 1) if isinstance(meta, dict) else 1
        pnl = exe.compute_two_leg_pnl(
            direction=row["direction"],
            beta=1.0,
            entry_y=row["entry_y_price"],
            entry_x=row["entry_x_price"],
            exit_y=fill_n,
            exit_x=fill_f,
            lots_y=row["lots_y"],
            lots_x=row["lots_x"],
            rpp_y=rpp,
            rpp_x=rpp,
            lot_size_y=1,
            lot_size_x=1,
            base_y=row["base_y"],
            base_x=row["base_x"],
        )
        await self.db.close_trade(
            row["id"],
            exit_y_price=fill_n,
            exit_x_price=fill_f,
            exit_z=z_now if z_now is not None else 0.0,
            exit_time=datetime.utcnow().isoformat(),
            exit_reason=reason,
            spread_exit=fill_n - fill_f,
            pnl=pnl["pnl_pct"],
            pnl_rub=pnl["net_rub"],
            commission_rub=pnl["commission_rub"],
        )
        logger.info(
            f"[carry] CLOSE ({reason}) held={held_h:.1f}h "
            f"P&L {pnl['net_rub']:+.2f}₽ ({pnl['pnl_pct']:+.3f}%)"
        )
        if self.notifier:
            self.notifier.push(
                Msg(
                    MsgType.TRADE_CLOSED,
                    f"CARRY {row['pair']} CLOSE ({reason})",
                    f"held: {held_h:.1f}h\n"
                    f"NET P&L: <b>{pnl['net_rub']:+.2f} ₽</b> ({pnl['pnl_pct']:+.3f}%)\n"
                    f"commission: {pnl['commission_rub']:.2f} ₽",
                )
            )

    async def status(self) -> str:
        if not self._initialised or not self.db:
            return "Carry subsystem not initialised."
        open_t = await self.db.open_trades()
        today_iso = datetime.utcnow().date().isoformat()
        today_pnl = await self.db.daily_pnl_rub(today_iso)
        lines = [
            f"<b>CARRY ({self.mode})</b>",
            f"Instrument: {self.settings.CARRY_BASE} calendar spread",
            f"Open positions: {len(open_t)}",
        ]
        for r in open_t:
            entry_dt = datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00"))
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
            side = "LONG" if r["direction"] > 0 else "SHORT"
            lines.append(f"  • {r['pair']} {side} basis z={r['entry_z']:+.2f} " f"held={held:.1f}h")
        lines.append(f"Today realised P&L: <b>{today_pnl:+.2f} ₽</b>")
        return "\n".join(lines)

    async def shutdown(self):
        if self.db:
            try:
                await self.db.close()
            except Exception:
                pass
