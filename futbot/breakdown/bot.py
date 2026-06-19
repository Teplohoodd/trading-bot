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
        self.fut_by_base: dict[str, dict] = {}  # base → {figi,ticker,rpp,lot_size,expiry}
        self.stock_figi: dict[str, str] = {}  # ticker → live share FIGI (resolved)
        self._last_signal_bar: dict[str, datetime] = {}

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
        logger.info(
            f"[breakdown] setup OK ({self.mode()}): "
            f"{len(self.stock_figi)} stocks resolved, "
            f"{len(self.fut_by_base)} futures mapped"
        )
        return True

    async def _resolve_stocks(self):
        """Resolve current share FIGIs by ticker (hardcoded FIGIs go stale when
        a company redomiciles, e.g. OZON → new FIGI / NOT_FOUND 50002).
        Drop any ticker that can't be resolved to a live share."""
        self.stock_figi: dict[str, str] = {}
        try:
            shares = await self.broker.get_all_shares()
        except Exception as e:
            logger.warning(
                f"[breakdown] get_all_shares failed ({e}); " f"falling back to static FIGIs"
            )
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
                    if cur is None or exp < cur["expiry"]:  # nearest month
                        self.fut_by_base[sb] = {
                            "figi": f.figi,
                            "ticker": ticker,
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
        from tinkoff.invest.schemas import CandleInterval
        from tinkoff.invest.utils import quotation_to_decimal

        now = datetime.now(timezone.utc)
        days = max(20, (self.settings.BD_SMA_BARS * self.settings.BD_BAR_HOURS) // 12)
        candles = await self.broker.get_candles(
            figi, now - timedelta(days=days), now, CandleInterval.CANDLE_INTERVAL_HOUR
        )
        if len(candles) < self.settings.BD_LOOKBACK_BARS * 3:
            return None
        df = pd.DataFrame(
            [
                {
                    "time": c.time,
                    "open": float(quotation_to_decimal(c.open)),
                    "high": float(quotation_to_decimal(c.high)),
                    "low": float(quotation_to_decimal(c.low)),
                    "close": float(quotation_to_decimal(c.close)),
                    "volume": c.volume,
                }
                for c in candles
            ]
        )
        rule = f"{self.settings.BD_BAR_HOURS}h"
        d = (
            df.set_index("time")
            .resample(rule)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )
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
        lo_prior = d["low"].iloc[i - n : i].min()
        vmed = d["volume"].iloc[i - n : i].median()
        bar = d.iloc[i]
        bar_ret = bar["close"] / bar["open"] - 1.0
        if not (
            bar["close"] < lo_prior
            and bar["volume"] >= s.BD_VOL_MULT * vmed
            and bar_ret <= s.BD_SEVERITY
        ):
            return None
        if s.BD_USE_SMA_FILTER:
            sma = d["close"].rolling(min(s.BD_SMA_BARS, len(d) - 1)).mean().iloc[i]
            if not (pd.notna(sma) and bar["close"] < sma):
                return None
        stop = float(bar["high"])
        entry = float(bar["close"])
        risk = stop - entry
        if risk <= 0 or risk / entry > s.BD_MAX_STOP_PCT:
            return None
        return {
            "time": bar["time"],
            "entry": entry,
            "stop": stop,
            "target": entry - s.BD_RISK_REWARD * risk,
            "bar_ret": bar_ret * 100,
            "vol_x": float(bar["volume"] / vmed) if vmed > 0 else 0.0,
        }

    # ── trading ──────────────────────────────────────────────────────────
    async def _open_count(self) -> int:
        cur = await self.db.execute("SELECT COUNT(*) FROM bd_trades WHERE exit_time IS NULL")
        return (await cur.fetchone())[0]

    async def _has_open(self, stock: str) -> bool:
        cur = await self.db.execute(
            "SELECT 1 FROM bd_trades WHERE stock=? AND exit_time IS NULL", (stock,)
        )
        return (await cur.fetchone()) is not None

    async def _margin_ok(self, fut: dict, price: float) -> bool:
        summ, ok = await self.broker.get_margin_summary()
        if not ok:
            return False
        go = (
            price
            * fut["rpp"]
            * fut["lot_size"]
            * fut["risk_rate"]
            * self.settings.BD_LOTS_PER_TRADE
        )
        return summ.get("available", 0.0) >= go * self.settings.BD_MARGIN_BUFFER

    async def _open_trade(self, stock: str, sig: dict):
        s = self.settings
        fut = self.fut_by_base.get(stock)
        if fut is None:
            await self._notify(
                f"📉 BREAKDOWN {stock}: bar {sig['bar_ret']:+.1f}% vol×{sig['vol_x']:.0f} "
                f"— сигнал без фьючерса (alert only)"
            )
            return
        fut_px = float(await self.broker.get_last_price(fut["figi"]))
        # scale stock-based stop/target onto the future via the ratio
        ratio = fut_px / sig["entry"] if sig["entry"] > 0 else 1.0
        stop_f = sig["stop"] * ratio
        target_f = sig["target"] * ratio
        paper = bool(s.BD_PAPER_MODE)
        oid, fill, stop_oid = "paper", fut_px, None
        if not paper:
            if not await self._margin_ok(fut, fut_px):
                logger.warning(f"[breakdown] {stock}: margin block")
                return
            from tinkoff.invest.schemas import OrderDirection, StopOrderDirection
            from decimal import Decimal

            res = await self.broker.post_market_order_with_fill(
                fut["figi"], s.BD_LOTS_PER_TRADE, OrderDirection.ORDER_DIRECTION_SELL
            )
            oid, fill = res["order_id"], res["fill_price"] or fut_px
            try:
                # protective stop BUYS to close the short.  Correct signature is
                # post_stop_loss(figi, lots, stop_price: Decimal, direction) —
                # the old call passed ("sell", stop_f) into (stop_price,
                # direction), which threw and left the short UNPROTECTED.
                sp = await self.broker._round_to_increment(fut["figi"], Decimal(str(stop_f)))
                stop_oid = await self.broker.post_stop_loss(
                    fut["figi"],
                    s.BD_LOTS_PER_TRADE,
                    sp,
                    StopOrderDirection.STOP_ORDER_DIRECTION_BUY,
                )
                logger.info(
                    f"[breakdown] {stock}: protective stop @ " f"{float(sp):.2f} id={stop_oid}"
                )
            except Exception as e:
                logger.warning(
                    f"[breakdown] {stock}: stop NOT placed ({e}) — "
                    f"monitor enforces stop instead"
                )
        await self.db.execute(
            "INSERT INTO bd_trades (stock,fut_base,fut_figi,fut_ticker,lots,paper,"
            "entry_time,entry_price,stock_entry,stop_price,target_price,"
            "stop_order_id,rpp,lot_size) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                stock,
                STOCK_TO_FUT[stock],
                fut["figi"],
                fut["ticker"],
                s.BD_LOTS_PER_TRADE,
                int(paper),
                datetime.now(timezone.utc).isoformat(),
                fill,
                sig["entry"],
                stop_f,
                target_f,
                stop_oid,
                fut["rpp"],
                fut["lot_size"],
            ),
        )
        await self.db.commit()
        await self._notify(
            f"📉 BREAKDOWN SHORT {stock} → {fut['ticker']} @ {fill:.2f}\n"
            f"бар {sig['bar_ret']:+.1f}% объём ×{sig['vol_x']:.0f}\n"
            f"стоп {stop_f:.2f} | цель {target_f:.2f} (RR {s.BD_RISK_REWARD:.0f}:1) "
            f"| таймаут {s.BD_TIMEOUT_BARS * s.BD_BAR_HOURS}ч"
        )

    async def _close_trade(self, row, price: float, reason: str):
        paper = bool(row["paper"])
        fill = price
        if not paper:
            from tinkoff.invest.schemas import OrderDirection

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
                fill = float(await self.broker.get_last_price(row["fut_figi"]))
                reason = f"{reason}(closed_externally)"
                pnl = (row["entry_price"] - fill) * row["lots"] * row["lot_size"] * row["rpp"]
                await self.db.execute(
                    "UPDATE bd_trades SET exit_time=?, exit_price=?, "
                    "exit_reason=?, pnl_rub=? WHERE id=?",
                    (datetime.now(timezone.utc).isoformat(), fill, reason, pnl, row["id"]),
                )
                await self.db.commit()
                await self._notify(
                    f"🏁 BREAKDOWN CLOSE {row['stock']} ({reason})\n"
                    f"{row['entry_price']:.2f} → {fill:.2f}  P&L {pnl:+.2f}₽"
                )
                return
            if row["stop_order_id"]:
                try:
                    await self.broker.cancel_stop_order(row["stop_order_id"])
                except Exception:
                    pass
            res = await self.broker.post_market_order_with_fill(
                row["fut_figi"], row["lots"], OrderDirection.ORDER_DIRECTION_BUY
            )
            fill = res["fill_price"] or price
        pnl = (row["entry_price"] - fill) * row["lots"] * row["lot_size"] * row["rpp"]
        await self.db.execute(
            "UPDATE bd_trades SET exit_time=?, exit_price=?, exit_reason=?, "
            "pnl_rub=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), fill, reason, pnl, row["id"]),
        )
        await self.db.commit()
        await self._notify(
            f"🏁 BREAKDOWN CLOSE {row['stock']} ({reason})\n"
            f"{row['entry_price']:.2f} → {fill:.2f}  P&L {pnl:+.2f}₽"
        )

    async def _manage_open(self):
        s = self.settings
        cur = await self.db.execute("SELECT * FROM bd_trades WHERE exit_time IS NULL")
        for row in await cur.fetchall():
            try:
                px = float(await self.broker.get_last_price(row["fut_figi"]))
            except Exception:
                continue
            opened = datetime.fromisoformat(row["entry_time"])
            age_h = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
            if px >= row["stop_price"]:
                await self._close_trade(row, px, "stop")
            elif px <= row["target_price"]:
                await self._close_trade(row, px, "target")
            elif age_h >= s.BD_TIMEOUT_BARS * s.BD_BAR_HOURS:
                await self._close_trade(row, px, "timeout")

    # ── runner interface ─────────────────────────────────────────────────
    async def tick(self):
        logger.info(f"[breakdown] tick ({self.mode()})")
        await self._manage_open()
        if await self._open_count() >= self.settings.BD_MAX_OPEN_POSITIONS:
            return
        for stock, figi in self.stock_figi.items():
            try:
                if await self._has_open(stock):
                    continue
                d = await self._stock_bars(figi)
                if d is None:
                    continue
                sig = self._signal_on_last_bar(d)
                if sig is None:
                    continue
                # one trade per signal bar
                if self._last_signal_bar.get(stock) == sig["time"]:
                    continue
                self._last_signal_bar[stock] = sig["time"]
                logger.info(
                    f"[breakdown] SIGNAL {stock}: bar {sig['bar_ret']:+.1f}% "
                    f"vol×{sig['vol_x']:.0f}"
                )
                await self._open_trade(stock, sig)
                if await self._open_count() >= self.settings.BD_MAX_OPEN_POSITIONS:
                    break
            except Exception as e:
                logger.warning(f"[breakdown] {stock}: {e}")

    async def monitor(self):
        """5-min loop: enforce stop/target/timeout between ticks."""
        await self._manage_open()

    async def status(self) -> str:
        cur = await self.db.execute("SELECT * FROM bd_trades WHERE exit_time IS NULL")
        open_rows = await cur.fetchall()
        cur = await self.db.execute(
            "SELECT COUNT(*) n, SUM(CASE WHEN pnl_rub>0 THEN 1 ELSE 0 END) w, "
            "COALESCE(SUM(pnl_rub),0) p FROM bd_trades WHERE exit_time IS NOT NULL"
        )
        n, w, p = await cur.fetchone()
        # broker-truth unrealized P&L (expected_yield, RUB) per open short
        detail, ok = await self.broker.get_positions_detail()
        lines = [
            f"BREAKDOWN ({self.mode()})",
            f"Universe: {len(self.stock_figi)} stocks → futures",
            f"Open: {len(open_rows)}",
        ]
        total_u = 0.0
        for r in open_rows:
            upnl = float(detail.get(r["fut_figi"], {}).get("unrealized", 0.0)) if ok else 0.0
            total_u += upnl
            lines.append(
                f"  {r['stock']} short @{r['entry_price']:.2f} "
                f"stop {r['stop_price']:.2f} tgt {r['target_price']:.2f}  "
                f"uP&L <b>{upnl:+.0f}₽</b>"
            )
        if open_rows:
            lines.append(f"Open unrealized (broker): <b>{total_u:+.0f}₽</b>")
        if n:
            lines.append(f"Closed: {n}, win {100 * (w or 0) / n:.0f}%, " f"P&L {p:+.2f}₽")
        return "\n".join(lines)

    async def _notify(self, text: str):
        logger.info(f"[breakdown] {text}")
        if self.notifier:
            try:
                from futbot.telegram_notifier import Msg, MsgType

                head, _, body = text.partition("\n")
                mtype = MsgType.TRADE_CLOSED if text.startswith("🏁") else MsgType.TRADE_OPENED
                self.notifier.push(Msg(mtype, head, body))
            except Exception as e:
                logger.warning(f"[breakdown] notify failed: {e}")
