"""BreakdownBot — orchestrator runner for the volume-breakdown short strategy.

Signal on the STOCK (cleaner volume structure), execution on the front-month
FUTURE (stocks like IVAT can't be shorted; futures can).

Flow per tick (every BD_BAR_HOURS):
  1. Manage open trades: stop / target / timeout on 2h closes (the exchange
     stop order placed at entry is the hard backstop between ticks).
  2. Scan universe for fresh breakdown bars (last CLOSED 2h bar only).
  3. On signal: margin check → short 1 lot of the mapped future → exchange
     stop above the breakdown bar high → log + Telegram.

PAPER mode (default): full logic, no orders — builds the forward track record
this one-regime backtest still owes us.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import aiosqlite
import pandas as pd

from futbot.breakdown.config import BreakdownSettings, STOCK_TO_FUT, STOCK_FIGI

logger = logging.getLogger("orchestrator.breakdown")


def _weekday_hours(start: datetime, end: datetime) -> float:
    """Wall-clock hours between start..end EXCLUDING Sat/Sun.

    Timeout redefined 2026-07-11: it used to be pure wall-clock, so a Friday
    entry burned its whole 48h budget over a dead weekend and got dumped Monday
    before the thesis had trading time to play out.  Post-exit study on live
    trades: 6 of 13 timeout exits would have hit the FULL original target
    within the next 48 trading hours.  3yr backtest of this exact rule:
    avgR +0.144→+0.241, PF 1.27→1.49 vs the bar-based design."""
    if end <= start:
        return 0.0
    h, cur = 0.0, start
    step = timedelta(hours=1)
    while cur < end:
        if cur.weekday() < 5:
            h += 1.0
        cur += step
    return h

SCHEMA = """
CREATE TABLE IF NOT EXISTS bd_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stock TEXT, fut_base TEXT, fut_figi TEXT, fut_ticker TEXT,
    lots INTEGER, paper INTEGER,
    entry_time TEXT, entry_price REAL,        -- future fill
    stock_entry REAL, stop_price REAL, target_price REAL,
    stop_order_id TEXT,
    exit_time TEXT, exit_price REAL, exit_reason TEXT,
    pnl_rub REAL,
    rpp REAL DEFAULT 1.0, lot_size INTEGER DEFAULT 1
);
"""


class BreakdownBot:
    name = "breakdown"

    def __init__(self):
        self.settings = BreakdownSettings()
        self.broker = None
        self.notifier = None
        self.db = None
        self.fut_by_base: dict[str, dict] = {}   # base → {figi,ticker,rpp,lot_size,expiry}
        self.stock_figi: dict[str, str] = {}     # ticker → live share FIGI (resolved)
        self._last_signal_bar: dict[str, datetime] = {}
        self._last_breadth: tuple[float, int] | None = None   # (up_frac, n)
        self._regime_blocking: bool = False
        self._last_panic: float | None = None                  # market vol rank

    def mode(self) -> str:
        return "PAPER" if self.settings.BD_PAPER_MODE else "LIVE"

    # ── lifecycle ────────────────────────────────────────────────────────
    async def setup(self, broker, notifier) -> bool:
        self.broker = broker
        self.notifier = notifier
        self.db = await aiosqlite.connect(self.settings.BD_DB_PATH)
        self.db.row_factory = aiosqlite.Row
        await self.db.executescript(SCHEMA)
        await self.db.commit()
        await self._resolve_stocks()
        await self._resolve_futures()
        logger.info(f"[breakdown] setup OK ({self.mode()}): "
                    f"{len(self.stock_figi)} stocks resolved, "
                    f"{len(self.fut_by_base)} futures mapped")
        return True

    async def _resolve_stocks(self):
        """Resolve current share FIGIs by ticker (hardcoded FIGIs go stale when
        a company redomiciles, e.g. OZON → new FIGI / NOT_FOUND 50002).
        Drop any ticker that can't be resolved to a live share."""
        self.stock_figi: dict[str, str] = {}
        try:
            shares = await self.broker.get_all_shares()
        except Exception as e:
            logger.warning(f"[breakdown] get_all_shares failed ({e}); "
                           f"falling back to static FIGIs")
            self.stock_figi = dict(STOCK_FIGI)
            return
        by_ticker = {}
        for s in shares:
            t = (getattr(s, "ticker", "") or "").upper()
            if t and t not in by_ticker:
                by_ticker[t] = s.figi
        for tick in STOCK_FIGI:
            figi = by_ticker.get(tick.upper())
            if figi:
                self.stock_figi[tick] = figi
            else:
                logger.info(f"[breakdown] {tick}: no live share — signal disabled")

    async def shutdown(self):
        if self.db:
            await self.db.close()

    async def _resolve_futures(self):
        """Map futures bases to front-month contracts (≥14d to expiry)."""
        futs = await self.broker.get_all_futures()
        now = datetime.now(timezone.utc)
        for f in futs:
            ticker = getattr(f, "ticker", "") or ""
            base = getattr(f, "basic_asset", "") or ""
            exp = getattr(f, "expiration_date", None)
            if not exp or (exp - now).days < 14:
                continue
            for sb, fb in STOCK_TO_FUT.items():
                # match by basic_asset ticker OR ticker prefix
                if base.upper() == sb or ticker.upper().startswith(fb.upper()):
                    meta = self.broker.extract_futures_metadata(f)
                    cur = self.fut_by_base.get(sb)
                    if cur is None or exp < cur["expiry"]:    # nearest month
                        self.fut_by_base[sb] = {
                            "figi": f.figi, "ticker": ticker,
                            # placeholder; real value fetched below (one API
                            # call per MAPPED contract, not per candidate)
                            "rpp": 1.0,
                            "lot_size": int(getattr(f, "lot", 1) or 1),
                            "risk_rate": float(meta.get("dshort") or 0.20),
                            "expiry": exp,
                        }
        # authoritative rub-per-point (see broker.get_rub_per_point docstring)
        for sb, fut in self.fut_by_base.items():
            fut["rpp"] = await self.broker.get_rub_per_point(fut["figi"])

    # ── data ─────────────────────────────────────────────────────────────
    async def _stock_bars(self, figi: str) -> pd.DataFrame | None:
        """Hourly candles → resampled BD_BAR_HOURS bars, last CLOSED only."""
        from t_tech.invest.schemas import CandleInterval
        from t_tech.invest.utils import quotation_to_decimal
        now = datetime.now(timezone.utc)
        # Need enough 2h bars for SMA(120) AND the consolidation-filter vol
        # percentile (BD_QUIET_WIN). 50d hourly ≈ 350 2h bars — covers both.
        days = max(50, (self.settings.BD_SMA_BARS * self.settings.BD_BAR_HOURS) // 12)
        candles = await self.broker.get_candles(
            figi, now - timedelta(days=days), now,
            CandleInterval.CANDLE_INTERVAL_HOUR)
        if len(candles) < self.settings.BD_LOOKBACK_BARS * 3:
            return None
        df = pd.DataFrame(
            [{"time": c.time,
              "open": float(quotation_to_decimal(c.open)),
              "high": float(quotation_to_decimal(c.high)),
              "low": float(quotation_to_decimal(c.low)),
              "close": float(quotation_to_decimal(c.close)),
              "volume": c.volume} for c in candles])
        rule = f"{self.settings.BD_BAR_HOURS}h"
        d = (df.set_index("time").resample(rule)
               .agg({"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"})
               .dropna().reset_index())
        # drop the still-forming bar
        bar_end = d["time"].iloc[-1] + timedelta(hours=self.settings.BD_BAR_HOURS)
        if bar_end > now:
            d = d.iloc[:-1]
        return d if len(d) >= self.settings.BD_LOOKBACK_BARS + 2 else None

    def _signal_on_last_bar(self, d: pd.DataFrame) -> dict | None:
        """Apply H1 conditions to the last closed bar."""
        s = self.settings
        n = s.BD_LOOKBACK_BARS
        i = len(d) - 1
        lo_prior = d["low"].iloc[i - n:i].min()
        vmed = d["volume"].iloc[i - n:i].median()
        bar = d.iloc[i]
        bar_ret = bar["close"] / bar["open"] - 1.0
        if not (bar["close"] < lo_prior
                and bar["volume"] >= s.BD_VOL_MULT * vmed
                and bar_ret <= s.BD_SEVERITY):
            return None
        if s.BD_USE_SMA_FILTER:
            sma = d["close"].rolling(min(s.BD_SMA_BARS, len(d) - 1)).mean().iloc[i]
            if not (pd.notna(sma) and bar["close"] < sma):
                return None
        # Consolidation filter: the run-up to the break must be CALM (low prior
        # realized vol).  Rejects chase-the-bottom shorts into volatile waterfalls.
        if s.BD_QUIET_FILTER:
            ret = d["close"].pct_change()
            rv = ret.rolling(n).std()
            rv_rank = rv.shift(1).rolling(s.BD_QUIET_WIN, min_periods=60).rank(pct=True)
            rr = rv_rank.iloc[i]
            if not (pd.notna(rr) and rr <= s.BD_QUIET_MAX_RANK):
                return None
        stop = float(bar["high"])
        entry = float(bar["close"])
        risk = stop - entry
        if risk <= 0 or risk / entry > s.BD_MAX_STOP_PCT:
            return None
        target = entry - s.BD_RISK_REWARD * risk
        # Liquidity-pool check (flow-confluence study): an INTACT swing low
        # between entry and target = support in the path → that trade class
        # historically earns half per unit risk → half-size it (variant A).
        pool_obstacle = False
        if s.BD_POOL_SIZING:
            k, win = s.BD_POOL_SWING_K, s.BD_POOL_SWING_WIN
            lows = d["low"].values
            bar_low = float(bar["low"])
            a = max(k, i - win)
            swings = []
            for j in range(a, i - k):
                w = lows[j - k:j + k + 1]
                if lows[j] == w.min() and (w > lows[j]).sum() == 2 * k:
                    swings.append(lows[j])
            below = [x for x in swings if x < bar_low]
            pool_obstacle = bool(below and max(below) > target)
        return {"time": bar["time"], "entry": entry, "stop": stop,
                "target": target,
                "pool_obstacle": pool_obstacle,
                "bar_ret": bar_ret * 100,
                "vol_x": float(bar["volume"] / vmed) if vmed > 0 else 0.0}

    # ── trading ──────────────────────────────────────────────────────────
    async def _open_count(self) -> int:
        cur = await self.db.execute(
            "SELECT COUNT(*) FROM bd_trades WHERE exit_time IS NULL")
        return (await cur.fetchone())[0]

    async def _has_open(self, stock: str) -> bool:
        cur = await self.db.execute(
            "SELECT 1 FROM bd_trades WHERE stock=? AND exit_time IS NULL", (stock,))
        return (await cur.fetchone()) is not None

    async def _margin_ok(self, fut: dict, price: float, lots: int) -> bool:
        summ, ok = await self.broker.get_margin_summary()
        if not ok:
            return False
        go = price * fut["rpp"] * fut["lot_size"] * fut["risk_rate"] * lots
        return summ.get("available", 0.0) >= go * self.settings.BD_MARGIN_BUFFER

    async def _size_lots(self, fut: dict, fill: float, stop_f: float,
                         risk_factor: float = 1.0) -> int:
        """Risk-based position size, capped by ГО share and an absolute max.

        lots ≈ (equity × BD_RISK_PER_TRADE_PCT) / risk_per_lot, where
        risk_per_lot = (stop − entry) × rpp × lot_size (RUB lost if stopped).
        Then capped so one position's ГО ≤ BD_MAX_GO_PCT_PER_POS of equity and
        ≤ BD_MAX_LOTS.  Falls back to the floor (BD_LOTS_PER_TRADE) if equity is
        unavailable or the stop distance is non-positive."""
        s = self.settings
        floor = max(1, s.BD_LOTS_PER_TRADE)
        risk_per_lot = (stop_f - fill) * fut["rpp"] * fut["lot_size"]
        if risk_per_lot <= 0:
            return floor
        summ, ok = await self.broker.get_margin_summary()
        equity = summ.get("liquid", 0.0) if ok else 0.0
        if equity <= 0:
            return floor
        lots_risk = (equity * s.BD_RISK_PER_TRADE_PCT * risk_factor) / risk_per_lot
        go_per_lot = fill * fut["rpp"] * fut["lot_size"] * fut["risk_rate"]
        lots_go = ((equity * s.BD_MAX_GO_PCT_PER_POS) / go_per_lot
                   if go_per_lot > 0 else lots_risk)
        lots = max(floor, int(min(lots_risk, lots_go, s.BD_MAX_LOTS)))
        # Broker-authoritative margin cap: never request more lots than the
        # broker will actually let us short right now (replaces trusting our
        # own ГО×risk_rate estimate).  -1 = call failed → keep our estimate.
        bmax = await self.broker.get_max_lots(fut["figi"], fill, "sell")
        capped = ""
        if bmax == 0:
            logger.warning(f"[breakdown] {fut['ticker']}: broker max_lots=0 "
                           f"(no free margin) — skipping entry")
            return 0
        if bmax > 0 and bmax < lots:
            capped = f" broker-cap={bmax}"; lots = bmax
        logger.info(
            f"[breakdown] size {fut['ticker']}: equity={equity:.0f} "
            f"risk/lot={risk_per_lot:.0f} factor={risk_factor:.2f} → "
            f"risk-cap={lots_risk:.1f} ГО-cap={lots_go:.1f}{capped} → {lots} lot(s)")
        return lots

    async def _open_trade(self, stock: str, sig: dict):
        s = self.settings
        fut = self.fut_by_base.get(stock)
        if fut is None:
            await self._notify(
                f"📉 BREAKDOWN {stock}: bar {sig['bar_ret']:+.1f}% vol×{sig['vol_x']:.0f} "
                f"— сигнал без фьючерса (alert only)")
            return
        fut_px = float(await self.broker.get_last_price(fut["figi"]))
        # scale stock-based stop/target onto the future via the ratio
        ratio = fut_px / sig["entry"] if sig["entry"] > 0 else 1.0
        stop_f = sig["stop"] * ratio
        target_f = sig["target"] * ratio
        # Risk-based sizing; half risk when a liquidity pool obstructs the path.
        pool = bool(sig.get("pool_obstacle"))
        factor = s.BD_POOL_RISK_FACTOR if (s.BD_POOL_SIZING and pool) else 1.0
        lots = await self._size_lots(fut, fut_px, stop_f, risk_factor=factor)
        if lots <= 0:                       # broker margin cap → cannot size
            return
        paper = bool(s.BD_PAPER_MODE)
        oid, fill, stop_oid = "paper", fut_px, None
        if not paper:
            if not await self._margin_ok(fut, fut_px, lots):
                logger.warning(f"[breakdown] {stock}: margin block ({lots} lot(s))")
                return
            from t_tech.invest.schemas import (OrderDirection,
                                                StopOrderDirection)
            from decimal import Decimal
            res = await self.broker.post_market_order_with_fill(
                fut["figi"], lots,
                OrderDirection.ORDER_DIRECTION_SELL)
            oid, fill = res["order_id"], res["fill_price"] or fut_px
            # Re-anchor stop/target to the ACTUAL fill (2026-07-16).  They were
            # computed off the pre-order last_price; on illiquid futures (IVAT)
            # a market sell filled up to −2.2% lower, leaving the stop far and
            # the target near — RR collapsed to 1:0.3 instead of 3:1 (trades
            # #1 −351₽, #29 −1710₽).  Scaling by fill/sig.entry preserves the
            # designed stock-space geometry around the real entry.
            if fill and sig["entry"] > 0:
                slip = (fill / fut_px - 1) * 100 if fut_px else 0.0
                stop_f = sig["stop"] * (fill / sig["entry"])
                target_f = sig["target"] * (fill / sig["entry"])
                if abs(slip) > 0.5:
                    logger.info(
                        f"[breakdown] {stock}: fill slippage {slip:+.2f}% vs "
                        f"last_price — geometry re-anchored to fill "
                        f"(stop {stop_f:.2f}, target {target_f:.2f})")
            try:
                # protective stop BUYS to close the short.  Correct signature is
                # post_stop_loss(figi, lots, stop_price: Decimal, direction) —
                # the old call passed ("sell", stop_f) into (stop_price,
                # direction), which threw and left the short UNPROTECTED.
                sp = await self.broker._round_to_increment(
                    fut["figi"], Decimal(str(stop_f)))
                stop_oid = await self.broker.post_stop_loss(
                    fut["figi"], lots, sp,
                    StopOrderDirection.STOP_ORDER_DIRECTION_BUY)
                logger.info(f"[breakdown] {stock}: protective stop @ "
                            f"{float(sp):.2f} id={stop_oid}")
            except Exception as e:
                logger.warning(f"[breakdown] {stock}: stop NOT placed ({e}) — "
                               f"monitor enforces stop instead")
        await self.db.execute(
            "INSERT INTO bd_trades (stock,fut_base,fut_figi,fut_ticker,lots,paper,"
            "entry_time,entry_price,stock_entry,stop_price,target_price,"
            "stop_order_id,rpp,lot_size) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (stock, STOCK_TO_FUT[stock], fut["figi"], fut["ticker"],
             lots, int(paper),
             datetime.now(timezone.utc).isoformat(), fill, sig["entry"],
             stop_f, target_f, stop_oid, fut["rpp"], fut["lot_size"]))
        await self.db.commit()
        risk_rub = (stop_f - fill) * fut["rpp"] * fut["lot_size"] * lots
        path_tag = ("пул на пути → риск ×%.1f" % factor) if pool else "путь чист"
        # real broker commission (pre-trade estimate) for the notify + records
        comm = None
        if not paper:
            comm = await self.broker.get_order_commission_rub(
                fut["figi"], lots, "sell", fill)
        comm_tag = f" | комиссия {comm:.0f}₽" if comm is not None else ""
        await self._notify(
            f"📉 BREAKDOWN SHORT {stock} → {fut['ticker']} ×{lots} @ {fill:.2f}\n"
            f"бар {sig['bar_ret']:+.1f}% объём ×{sig['vol_x']:.0f} | {path_tag}\n"
            f"стоп {stop_f:.2f} | цель {target_f:.2f} (RR {s.BD_RISK_REWARD:.0f}:1) "
            f"| риск {risk_rub:.0f}₽{comm_tag} | таймаут {s.BD_TIMEOUT_BARS * s.BD_BAR_HOURS}ч")

    async def _real_exit_price(self, figi: str, since_iso: str) -> float | None:
        """Reconstruct the ACTUAL covering-trade price from operations history.

        When a short is found already flat at the broker (the exchange stop
        fired, or it was closed externally), the real exit happened EARLIER at
        the stop/fill price — not the current market price.  Using
        get_last_price there fabricates P&L: on 2026-06-24 high volatility
        stopped ROSN/SMLT out, but the bot read the price after it had snapped
        back and mis-recorded the stop-outs as profits.  breakdown is always
        SHORT, so the closing trade is a BUY / BUY_MARGIN on this figi.  Returns
        None if no matching operation is found (caller falls back + flags it)."""
        try:
            from t_tech.invest import OperationType
            from t_tech.invest.utils import quotation_to_decimal
            since = datetime.fromisoformat(since_iso)
            if since.tzinfo is None:
                since = since.replace(tzinfo=timezone.utc)
            # cursor endpoint with SERVER-SIDE type filter (short close = BUY)
            ops = await self.broker.get_operations(
                from_dt=since,
                operation_types=[OperationType.OPERATION_TYPE_BUY,
                                 OperationType.OPERATION_TYPE_BUY_MARGIN])
        except Exception as e:
            logger.warning(f"[breakdown] ops lookup failed: {e}")
            return None
        matches = [op for op in ops if getattr(op, "figi", None) == figi]
        if not matches:
            return None
        matches.sort(key=lambda o: getattr(
            o, "date", datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        pq = getattr(matches[0], "price", None)
        if pq is None:
            return None
        try:
            return float(quotation_to_decimal(pq))
        except Exception:
            return None

    async def _close_trade(self, row, price: float, reason: str):
        paper = bool(row["paper"])
        fill = price
        if not paper:
            from t_tech.invest.schemas import OrderDirection
            # RACE GUARD via BROKER POSITION (not the stop list — that mis-fired
            # on trend's ROSN).  breakdown is always SHORT, so "flat" = qty>=0.
            pos, ok = await self.broker.get_positions_detail()
            if ok and float(pos.get(row["fut_figi"], {}).get("qty", 0) or 0) >= 0:
                # already covered by the exchange stop / externally — don't
                # send another BUY (would open a LONG orphan).
                if row["stop_order_id"]:
                    try:
                        await self.broker.cancel_stop_order(row["stop_order_id"])
                    except Exception:
                        pass
                # Real exit = the actual covering trade, NOT the current price.
                fill = await self._real_exit_price(row["fut_figi"], row["entry_time"])
                if fill is None:
                    # No matching op — fall back to last price, but FLAG that the
                    # P&L is unverified so it isn't trusted as a clean number.
                    fill = float(await self.broker.get_last_price(row["fut_figi"]))
                    reason = f"{reason}(closed_externally~approx)"
                    logger.warning(
                        f"[breakdown] {row['stock']}: no covering op found — "
                        f"P&L approximated from last price {fill:.2f}")
                else:
                    reason = f"{reason}(closed_externally)"
                pnl = (row["entry_price"] - fill) * row["lots"] \
                    * row["lot_size"] * row["rpp"]
                await self.db.execute(
                    "UPDATE bd_trades SET exit_time=?, exit_price=?, "
                    "exit_reason=?, pnl_rub=? WHERE id=?",
                    (datetime.now(timezone.utc).isoformat(), fill, reason,
                     pnl, row["id"]))
                await self.db.commit()
                await self._notify(
                    f"🏁 BREAKDOWN CLOSE {row['stock']} ({reason})\n"
                    f"{row['entry_price']:.2f} → {fill:.2f}  P&L {pnl:+.2f}₽")
                return
            if row["stop_order_id"]:
                try:
                    await self.broker.cancel_stop_order(row["stop_order_id"])
                except Exception:
                    pass
            try:
                res = await self.broker.post_market_order_with_fill(
                    row["fut_figi"], row["lots"], OrderDirection.ORDER_DIRECTION_BUY)
            except Exception as e:
                # Market closed (30079 'instrument not available') or transient
                # broker error → DON'T crash the tick.  Leave the position open
                # and retry on the next cycle (it'll fill once the session opens).
                logger.warning(f"[breakdown] {row['stock']}: close order deferred "
                               f"({type(e).__name__}: {e}) — retry next cycle")
                return
            fill = res["fill_price"] or price
        pnl = (row["entry_price"] - fill) * row["lots"] \
            * row["lot_size"] * row["rpp"]
        await self.db.execute(
            "UPDATE bd_trades SET exit_time=?, exit_price=?, exit_reason=?, "
            "pnl_rub=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), fill, reason, pnl, row["id"]))
        await self.db.commit()
        await self._notify(
            f"🏁 BREAKDOWN CLOSE {row['stock']} ({reason})\n"
            f"{row['entry_price']:.2f} → {fill:.2f}  P&L {pnl:+.2f}₽")

    async def _manage_open(self):
        s = self.settings
        cur = await self.db.execute(
            "SELECT * FROM bd_trades WHERE exit_time IS NULL")
        rows = await cur.fetchall()
        if not rows:
            return
        # Proactive reconcile (added 2026-07-17): breakdown had NONE — a SBER
        # short closed at the broker on 07-16 but its DB row lingered 'open'
        # because price sat between stop and target and the timeout hadn't hit,
        # so _manage_open never re-checked the broker (phantom position).  Fetch
        # broker truth once and close any row whose position is already gone.
        detail, ok = await self.broker.get_positions_detail()
        now = datetime.now(timezone.utc)
        for row in rows:
            try:
                px = float(await self.broker.get_last_price(row["fut_figi"]))
                opened = datetime.fromisoformat(row["entry_time"])
                age_min = (now - opened).total_seconds() / 60
                # Position GONE at broker: breakdown is always SHORT → a live
                # position has qty < 0; qty >= 0 means flat/closed externally.
                # 10-min guard avoids a just-opened position reading flat on
                # broker-side latency.
                if ok and age_min > 10:
                    qty = float(detail.get(row["fut_figi"], {}).get("qty", 0) or 0)
                    if qty >= 0:
                        logger.info(f"[breakdown] {row['stock']}: flat at broker "
                                    f"(qty={qty}) — reconciling phantom open row")
                        await self._close_trade(row, px, "reconciled")
                        continue
                # weekend hours don't count toward the timeout (dead time)
                age_h = _weekday_hours(opened, now)
                if px >= row["stop_price"]:
                    await self._close_trade(row, px, "stop")
                elif px <= row["target_price"]:
                    await self._close_trade(row, px, "target")
                elif age_h >= s.BD_TIMEOUT_BARS * s.BD_BAR_HOURS:
                    await self._close_trade(row, px, "timeout")
            except Exception as e:
                # one position's failure must not crash the whole manage cycle
                logger.warning(f"[breakdown] manage {row['stock']}: "
                               f"{type(e).__name__}: {e}")
                continue

    def _market_panic(self, bars: dict) -> tuple[float | None, bool]:
        """(vol_rank, panicking).  Equal-weight proxy of the universe → 24h
        realized vol → percentile rank over the trailing window.  High rank =
        market-wide panic (waterfall) — new shorts get chased and bounce-stopped;
        the 3yr backtest says skip them (skipped trades avg −0.32R)."""
        s = self.settings
        series = []
        for d in bars.values():
            if d is None or len(d) < s.BD_PANIC_VOL_BARS * 3:
                continue
            c = d.set_index("time")["close"]
            series.append(c / c.iloc[0])
        if len(series) < 8:
            return None, False
        proxy = pd.concat(series, axis=1, sort=True).ffill().mean(axis=1)
        rv = proxy.pct_change().rolling(s.BD_PANIC_VOL_BARS).std()
        rank = rv.rolling(len(rv), min_periods=60).rank(pct=True)
        r = rank.iloc[-1]
        if pd.isna(r):
            return None, False
        return float(r), float(r) > s.BD_PANIC_VOL_RANK

    def _market_breadth(self, bars: dict) -> tuple[float, int]:
        """Fraction of the universe whose close ROSE over the last
        BD_REGIME_LOOKBACK_BARS bars.  Returns (up_fraction, n_valid)."""
        L = self.settings.BD_REGIME_LOOKBACK_BARS
        ups = n = 0
        for d in bars.values():
            if d is None or len(d) < L + 1:
                continue
            c_now = float(d["close"].iloc[-1])
            c_then = float(d["close"].iloc[-1 - L])
            if c_then > 0:
                n += 1
                if c_now > c_then:
                    ups += 1
        return (ups / n if n else 0.0), n

    # ── runner interface ─────────────────────────────────────────────────
    async def tick(self):
        s = self.settings
        logger.info(f"[breakdown] tick ({self.mode()})")
        await self._manage_open()
        start_open = await self._open_count()
        if start_open >= s.BD_MAX_OPEN_POSITIONS:
            return
        # Fetch all universe bars ONCE — reused for both the regime guard and
        # the signal scan (no double-fetch of candles).
        bars: dict = {}
        for stock, figi in self.stock_figi.items():
            try:
                bars[stock] = await self._stock_bars(figi)
            except Exception as e:
                logger.warning(f"[breakdown] {stock}: bars {e}")
                bars[stock] = None
        # Panic kill-switch: no NEW entries while the market itself is in a
        # volatility panic (waterfall) — entries there get bounce-stopped.
        if s.BD_PANIC_FILTER:
            vrank, panicking = self._market_panic(bars)
            self._last_panic = vrank
            if panicking:
                logger.info(f"[breakdown] PANIC filter: market vol rank "
                            f"{vrank:.2f} > {s.BD_PANIC_VOL_RANK:.2f} — "
                            f"no new entries this tick")
                return
        # Regime guard: don't INITIATE fresh shorts into a broad bounce.
        if s.BD_REGIME_GUARD:
            up_frac, n = self._market_breadth(bars)
            self._last_breadth = (up_frac, n)
            blocked = n >= s.BD_REGIME_MIN_NAMES and up_frac >= s.BD_REGIME_UP_FRAC
            self._regime_blocking = blocked
            if blocked:
                logger.info(
                    f"[breakdown] regime guard ON: {up_frac:.0%} of {n} names "
                    f"rising over {s.BD_REGIME_LOOKBACK_BARS * s.BD_BAR_HOURS}h "
                    f"(≥{s.BD_REGIME_UP_FRAC:.0%}) — skipping NEW shorts")
                return
        for stock, d in bars.items():
            try:
                if d is None or await self._has_open(stock):
                    continue
                sig = self._signal_on_last_bar(d)
                if sig is None:
                    continue
                # one trade per signal bar
                if self._last_signal_bar.get(stock) == sig["time"]:
                    continue
                self._last_signal_bar[stock] = sig["time"]
                logger.info(f"[breakdown] SIGNAL {stock}: bar {sig['bar_ret']:+.1f}% "
                            f"vol×{sig['vol_x']:.0f}")
                await self._open_trade(stock, sig)
                open_now = await self._open_count()
                # Per-bar cap: limit correlated exposure from one signal bar
                # (the 06-26 cluster fix). Counts only positions actually opened.
                if open_now - start_open >= s.BD_MAX_NEW_PER_BAR:
                    logger.info(f"[breakdown] per-bar entry cap reached "
                                f"({s.BD_MAX_NEW_PER_BAR}) — no more new shorts this tick")
                    break
                if open_now >= s.BD_MAX_OPEN_POSITIONS:
                    break
            except Exception as e:
                logger.warning(f"[breakdown] {stock}: {e}")

    async def monitor(self):
        """5-min loop: enforce stop/target/timeout between ticks."""
        await self._manage_open()

    async def status(self) -> str:
        cur = await self.db.execute(
            "SELECT * FROM bd_trades WHERE exit_time IS NULL")
        open_rows = await cur.fetchall()
        cur = await self.db.execute(
            "SELECT COUNT(*) n, SUM(CASE WHEN pnl_rub>0 THEN 1 ELSE 0 END) w, "
            "COALESCE(SUM(pnl_rub),0) p FROM bd_trades WHERE exit_time IS NOT NULL")
        n, w, p = await cur.fetchone()
        # broker-truth unrealized P&L (expected_yield, RUB) per open short
        detail, ok = await self.broker.get_positions_detail()
        lines = [f"BREAKDOWN ({self.mode()})",
                 f"Universe: {len(self.stock_figi)} stocks → futures",
                 f"Open: {len(open_rows)}"]
        if self.settings.BD_PANIC_FILTER and self._last_panic is not None:
            state = ("🛑 PANIC — no new entries"
                     if self._last_panic > self.settings.BD_PANIC_VOL_RANK
                     else "✅ calm")
            lines.append(f"Market vol: {state} (rank {self._last_panic:.2f} "
                         f"/ thr {self.settings.BD_PANIC_VOL_RANK:.2f})")
        if self.settings.BD_REGIME_GUARD and self._last_breadth is not None:
            up_frac, nb = self._last_breadth
            state = ("🛑 BLOCKING new shorts" if self._regime_blocking
                     else "✅ shorts allowed")
            lines.append(f"Regime: {state} ({up_frac:.0%} of {nb} names up "
                         f"/ thr {self.settings.BD_REGIME_UP_FRAC:.0%})")
        total_u = 0.0
        for r in open_rows:
            upnl = float(detail.get(r["fut_figi"], {}).get("unrealized", 0.0)) if ok else 0.0
            total_u += upnl
            lines.append(f"  {r['stock']} short @{r['entry_price']:.2f} "
                         f"stop {r['stop_price']:.2f} tgt {r['target_price']:.2f}  "
                         f"uP&L <b>{upnl:+.0f}₽</b>")
        if open_rows:
            lines.append(f"Open unrealized (broker): <b>{total_u:+.0f}₽</b>")
        if n:
            lines.append(f"Closed: {n}, win {100 * (w or 0) / n:.0f}%, "
                         f"P&L {p:+.2f}₽")
        return "\n".join(lines)

    async def _notify(self, text: str):
        logger.info(f"[breakdown] {text}")
        if self.notifier:
            try:
                from futbot.telegram_notifier import Msg, MsgType
                head, _, body = text.partition("\n")
                mtype = (MsgType.TRADE_CLOSED if text.startswith("🏁")
                         else MsgType.TRADE_OPENED)
                self.notifier.push(Msg(mtype, head, body))
            except Exception as e:
                logger.warning(f"[breakdown] notify failed: {e}")
