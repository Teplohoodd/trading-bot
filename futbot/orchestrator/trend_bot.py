"""TrendBot — pattern-based trend strategy for orchestrator.

Replaces the failed Bollinger-band breakout (which lost 8/8 in OOS live)
with chart-pattern entries: triple_top + triple_bottom (whitelisted).
See futbot.patterns.backtest for the validation that justified this swap.

Two exit paths coexist:
  - Legacy positions (pattern_name IS NULL in DB): Bollinger band_flip,
    preserved so existing 8 open trades from the old strategy can close
    gracefully.  No NEW Bollinger entries are made.
  - New pattern positions: explicit stop_price / target_price / timeout
    stored at entry; managed bar-by-bar against last price.

Universe selection still uses the WF-validated 11 (core mode) — those
contracts already have enough liquidity / volatility for both strategies.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd

from tinkoff.invest import (
    CandleInterval,
    OrderDirection,
    StopOrderDirection,
)
from tinkoff.invest.utils import quotation_to_decimal

from futbot.trend.config import TrendSettings
from futbot.trend.db import TrendDB
from futbot.trend import strategy as strat
from futbot.trend.portfolio import (
    PORTFOLIO,
    core_portfolio,
    tradeable_portfolio,
    TrendEntry,
)
from futbot.utils import commissions as comm
from futbot.telegram_notifier import Msg, MsgType

# Pattern detector + whitelist
from futbot.patterns.primitives import find_swings
from futbot.patterns.detectors import (
    detect_triple_tops,
    detect_triple_bottoms,
)
from futbot.patterns.portfolio import (
    TUNED_PARAMS,
    is_pattern_allowed,
)

logger = logging.getLogger("orchestrator.trend")


@dataclass
class ResolvedContract:
    base: str
    ticker: str
    figi: str
    instrument: object
    expiration: datetime
    rpp: float
    lot_size: int
    bb_n: int  # kept for legacy band_flip exit path
    bb_k: float
    is_neo: bool = False  # Neo asset (USD price, RUB P&L at FX, no expiry)
    currency: str = "rub"
    risk_rate: float = 0.0  # margin fraction (dlong); leverage = 1/risk_rate

    @property
    def days_to_expiry(self) -> int:
        if self.is_neo:
            return 9999  # Neo perpetuals never expire → never roll over
        return max(0, (self.expiration - datetime.now(timezone.utc)).days)


async def _resolve_front_month(broker, base: str, min_dte: int):
    futs = await broker.get_all_futures()
    cands = []
    now = datetime.now(timezone.utc)
    for f in futs:
        t = getattr(f, "ticker", "") or ""
        if len(t) < 3:
            continue
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
    return cands[0]


async def _resolve_portfolio(
    broker, entries: list[TrendEntry], min_dte: int
) -> list[ResolvedContract]:
    out = []
    for e in entries:
        try:
            r = await _resolve_front_month(broker, e.base, min_dte)
        except Exception as ex:
            logger.warning(f"  {e.base}: resolve failed ({ex})")
            continue
        if r is None:
            logger.warning(f"  {e.base}: no front-month with >={min_dte}d to expiry")
            continue
        f, exp = r
        meta = broker.extract_futures_metadata(f)
        is_neo = (f.ticker or "").endswith("perpA")
        # REAL point value via GetFuturesMargin — get_all_futures omits
        # min_price_increment_amount, so meta rpp silently defaults to 1
        # (LTU6 is really 7.19₽/pt, GDU6/S1U6 71.9₽/pt).
        real_rpp = await broker.get_rub_per_point(f.figi)
        out.append(
            ResolvedContract(
                base=e.base,
                ticker=f.ticker,
                figi=f.figi,
                instrument=f,
                expiration=exp,
                rpp=real_rpp,
                lot_size=int(getattr(f, "lot", 1) or 1),
                bb_n=e.n,
                bb_k=e.k,
                is_neo=is_neo,
                currency=(getattr(f, "currency", "") or "rub"),
                risk_rate=float(meta.get("dlong") or 0.0),
            )
        )
    return out


async def _resolve_neo(broker, entries: list[TrendEntry]) -> list[ResolvedContract]:
    """Resolve Neo assets by EXACT ticker (no month code, no expiry filter)."""
    futs = await broker.get_all_futures()
    by_ticker = {(getattr(f, "ticker", "") or ""): f for f in futs}
    out = []
    for e in entries:
        f = by_ticker.get(e.base)
        if f is None:
            logger.warning(f"  Neo {e.base}: not found")
            continue
        meta = broker.extract_futures_metadata(f)
        exp = getattr(f, "expiration_date", None)
        if exp is not None and hasattr(exp, "ToDatetime"):
            exp = exp.ToDatetime()
        if exp is not None and exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        out.append(
            ResolvedContract(
                base=e.base,
                ticker=f.ticker,
                figi=f.figi,
                instrument=f,
                expiration=exp or (datetime.now(timezone.utc) + timedelta(days=9999)),
                rpp=float(meta.get("rub_per_point") or 1.0),
                lot_size=int(getattr(f, "lot", 1) or 1),
                bb_n=e.n,
                bb_k=e.k,
                is_neo=True,
                currency=(getattr(f, "currency", "") or "usd"),
                risk_rate=float(meta.get("dlong") or 0.20),
            )
        )
    return out


async def _fetch_ohlc(broker, figi: str, days: int) -> pd.DataFrame:
    """Fetch hourly OHLC.  Patterns need high+low; closes alone are
    insufficient because intra-bar stop/target hits must be detected."""
    now = datetime.now(timezone.utc)
    try:
        candles = await broker.get_candles(
            figi,
            now - timedelta(days=days),
            now,
            interval=CandleInterval.CANDLE_INTERVAL_HOUR,
        )
    except Exception as e:
        logger.warning(f"  {figi}: fetch failed ({e})")
        return pd.DataFrame()
    rows = [
        {
            "time": c.time,
            "open": float(quotation_to_decimal(c.open)),
            "high": float(quotation_to_decimal(c.high)),
            "low": float(quotation_to_decimal(c.low)),
            "close": float(quotation_to_decimal(c.close)),
            "volume": int(c.volume),
        }
        for c in candles
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


class QualRestricted(Exception):
    """Broker rejected order due to qualified-investor restriction (90002)."""


async def _place_market(
    broker, figi: str, direction: str, lots: int, paper: bool
) -> tuple[str, float]:
    last_p = float(await broker.get_last_price(figi))
    if paper:
        oid = f"paper-trend-{direction}-{figi[-6:]}-{int(time.time())}"
        return oid, last_p
    side = (
        OrderDirection.ORDER_DIRECTION_BUY
        if direction == "buy"
        else OrderDirection.ORDER_DIRECTION_SELL
    )
    # Use the REAL executed price for P&L, not a last-price snapshot.
    try:
        res = await broker.post_market_order_with_fill(figi, lots, side)
    except Exception as e:
        msg = str(e)
        if "90002" in msg or "qualified investor" in msg.lower():
            # Crypto Neo requires quals status — soft-skip, don't crash the
            # whole runner (a hard raise puts supervisor into 5-min backoff).
            raise QualRestricted(f"{figi}: qualified-investor required") from e
        raise
    fill = res["fill_price"] if res["fill_price"] > 0 else last_p
    if res.get("commission_rub"):
        logger.info(
            f"[trend] {figi} filled @ {fill:.4f} " f"(real comm {res['commission_rub']:.2f} ₽)"
        )
    return res["order_id"], fill


# ── Pattern-trade lifecycle helpers ─────────────────────────────────────

# How many bars after entry before a pattern position auto-times-out.
# Matches the backtester's max_bars_held.  Stored once here; can be
# overridden per-trade if patterns/portfolio.py specifies different holds.
PATTERN_TIMEOUT_BARS = 48


def _pattern_exit_reason(
    direction: str,
    last_price: float,
    stop_price: float,
    target_price: float,
    entry_time_iso: str,
    timeout_bars: int,
) -> str | None:
    """Return 'pattern_stop' / 'pattern_target' / 'pattern_timeout' / None.

    Bar-resolution timeout uses elapsed hours since entry.  Stop/target
    are checked against last_price (a conservative proxy for the bar's
    high/low — gives later exits than the optimistic backtest).
    """
    # Stop hit
    if direction == "buy":
        if last_price <= stop_price:
            return "pattern_stop"
        if last_price >= target_price:
            return "pattern_target"
    else:  # sell
        if last_price >= stop_price:
            return "pattern_stop"
        if last_price <= target_price:
            return "pattern_target"
    # Timeout
    entry_dt = datetime.fromisoformat(entry_time_iso.replace("Z", "+00:00"))
    if entry_dt.tzinfo is None:
        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
    hours_held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
    if hours_held >= timeout_bars:
        return "pattern_timeout"
    return None


class TrendBot:
    name = "trend"

    def __init__(self):
        self.settings = TrendSettings()
        self.db: TrendDB | None = None
        self.broker = None
        self.notifier = None
        self.contracts: list[ResolvedContract] = []
        self.by_base: dict = {}
        self.portfolio_value = 0.0
        self._last_universe_refresh: datetime | None = None
        self._alerted_orphans: set = set()  # dedup orphan alerts across monitor ticks
        self._gone_strikes: dict = {}  # figi → consecutive confirmed-absent reads
        self._fx_usdrub: float = 80.0  # USD/RUB for Neo P&L (refreshed live)
        self._qual_blocklist: set = set()  # bases the broker rejected for quals
        # base → bar_time of the last signal we ENTERED on.  With 15-min ticks
        # a stop-out inside the 2-bar freshness window would otherwise re-enter
        # the same pattern within minutes (enter→stop→re-enter loop).
        self._entered_sig_bar: dict = {}
        self._initialised = False

    async def _refresh_fx(self):
        """USD/RUB rate for Neo-asset P&L.  Try USDRUBF, then Si/1000."""
        try:
            futs = await self.broker.get_all_futures()
            usd = next((f for f in futs if (getattr(f, "ticker", "") or "") == "USDRUBF"), None)
            if usd:
                px = float(await self.broker.get_last_price(usd.figi))
                if px > 0:
                    self._fx_usdrub = px
                    return
            si = next(
                (
                    f
                    for f in futs
                    if (getattr(f, "ticker", "") or "").startswith("Si")
                    and len(getattr(f, "ticker", "")) == 4
                ),
                None,
            )
            if si:
                px = float(await self.broker.get_last_price(si.figi))
                if px > 0:
                    self._fx_usdrub = px / 1000.0
        except Exception as e:
            logger.warning(f"[trend] FX refresh failed ({e}); using {self._fx_usdrub}")

    @staticmethod
    def _neo_nights(entry_dt: datetime, exit_dt: datetime) -> int:
        """Number of 00:00-MSK boundaries the position is held across.

        Neo holding fee is charged per CALENDAR DAY held past midnight MSK
        (open+close same MSK day → 0).  Not proportional to hours.
        """
        msk = timezone(timedelta(hours=3))
        e = entry_dt.astimezone(msk).date()
        x = exit_dt.astimezone(msk).date()
        return max(0, (x - e).days)

    def _neo_fee_pct(self, entry_dt: datetime, exit_dt: datetime) -> float:
        """Neo holding fee as % of notional = daily_rate × nights held."""
        nights = self._neo_nights(entry_dt, exit_dt)
        return float(self.settings.TREND_NEO_DAILY_FEE_ANNUAL) / 365.0 * nights * 100.0

    async def _neo_margin_ok(self, c: "ResolvedContract", entry_price: float, lots: int) -> bool:
        """Block a Neo entry if free margin < buffer × the position's ГО.

        ГО (initial margin) = notional_RUB × risk_rate.  Conservative: if the
        margin API can't be read, DO NOT open (better to skip than risk a
        margin call)."""
        summ, ok = await self.broker.get_margin_summary()
        if not ok:
            logger.warning(f"[trend] {c.base}: margin API unavailable — skip Neo entry")
            return False
        rr = c.risk_rate if c.risk_rate > 0 else 0.20
        notional = entry_price * self._fx_usdrub * int(c.lot_size or 1) * lots
        go = notional * rr
        need = go * float(self.settings.TREND_NEO_MARGIN_BUFFER)
        avail = summ.get("available", 0.0)
        if avail < need:
            logger.warning(
                f"[trend] {c.base}: Neo entry BLOCKED — free margin {avail:,.0f}₽ "
                f"< {self.settings.TREND_NEO_MARGIN_BUFFER:.1f}× ГО ({go:,.0f}₽ "
                f"= ${entry_price:.0f}×{self._fx_usdrub:.0f}×{rr*100:.0f}%)"
            )
            return False
        logger.info(
            f"[trend] {c.base}: Neo margin OK — ГО {go:,.0f}₽ "
            f"(lev {1/rr:.1f}×), free {avail:,.0f}₽, suff {summ.get('sufficiency_pct',0):.0f}%"
        )
        return True

    def _size_lots(self, c: "ResolvedContract", entry_price: float) -> int:
        """Volatility/margin-target sizing.

        Picks N lots so ГО of one position ≈ TARGET_MARGIN_RUB:
          notional_per_lot = price × lot_size × rub_per_point
          go_per_lot       = notional_per_lot × risk_rate
          lots             = round(target / go_per_lot), clipped to [1, MAX]

        For FORTS futures: notional uses rpp and lot_size already.
        For Neo: notional = price × FX (USD→RUB), lot_size from broker.
        Falls back to fixed TREND_LOTS_PER_TRADE if vol-target disabled.
        """
        if not bool(getattr(self.settings, "TREND_VOL_TARGET_SIZING", False)):
            return int(self.settings.TREND_LOTS_PER_TRADE)
        rr = c.risk_rate if c.risk_rate > 0 else 0.20
        lot_size = int(c.lot_size or 1)
        if c.is_neo:
            notional_per_lot = entry_price * self._fx_usdrub * lot_size
        else:
            notional_per_lot = entry_price * lot_size * float(c.rpp or 1.0)
        go_per_lot = notional_per_lot * rr
        if go_per_lot <= 0:
            return int(self.settings.TREND_LOTS_PER_TRADE)
        target = float(self.settings.TREND_TARGET_MARGIN_RUB)
        max_lots = int(self.settings.TREND_LOTS_MAX_PER_TRADE)
        raw = target / go_per_lot
        lots = max(1, min(max_lots, int(round(raw))))
        logger.info(
            f"[trend] {c.base}: size = {lots} lot(s) "
            f"(ГО {go_per_lot:.0f}/lot × {lots} = {go_per_lot*lots:.0f}₽, "
            f"target {target:.0f}₽, lev {1/rr:.1f}×)"
        )
        return lots

    def _pnl_rub(
        self,
        c: "ResolvedContract",
        direction: str,
        entry: float,
        exit_p: float,
        lots: int,
        entry_dt: datetime = None,
        exit_dt: datetime = None,
    ):
        """Unified P&L → (pnl_rub, pnl_pct, commission_rub).

        Neo: USD price → RUB at the live FX rate; cost = per-night holding fee
        (charged per 00:00-MSK crossing, NOT per hour).
        Futures: standard round-trip via rub_per_point + futures commission.
        """
        if c.is_neo:
            sgn = 1.0 if direction == "buy" else -1.0
            pct = (exit_p - entry) / max(entry, 1e-9) * 100 * sgn
            gross_rub = (exit_p - entry) * sgn * lots * int(c.lot_size or 1) * self._fx_usdrub
            fee_pct = self._neo_fee_pct(entry_dt, exit_dt) if entry_dt and exit_dt else 0.0
            fee_rub = abs(entry * lots * int(c.lot_size or 1) * self._fx_usdrub) * fee_pct / 100
            return gross_rub - fee_rub, pct - fee_pct, fee_rub
        pnl, pnl_pct, _, comm_rub = comm.round_trip_pnl(
            direction=direction,
            entry_price=entry,
            exit_price=exit_p,
            lots=lots,
            lot_size=int(c.lot_size or 1),
            rub_per_point=float(c.rpp or 1.0),
            instrument_kind="future",
            base_ticker=c.base,
        )
        return pnl, pnl_pct, comm_rub

    @property
    def mode(self) -> str:
        return "PAPER" if self.settings.TREND_PAPER_MODE else "LIVE"

    async def setup(self, broker, notifier):
        self.broker = broker
        self.notifier = notifier
        self.db = TrendDB(self.settings.TREND_DB_PATH)
        await self.db.initialize()

        entries = (
            core_portfolio()
            if self.settings.TREND_UNIVERSE_MODE == "core"
            else tradeable_portfolio()
        )
        logger.info(
            f"[trend] Universe mode: {self.settings.TREND_UNIVERSE_MODE} "
            f"-> {len(entries)} contracts"
        )
        self.contracts = await _resolve_portfolio(
            broker,
            entries,
            self.settings.TREND_MIN_DAYS_TO_EXPIRY,
        )
        # Append Neo assets (US stocks / crypto) if enabled
        if bool(self.settings.TREND_TRADE_NEO):
            from futbot.trend.portfolio import neo_portfolio

            neo = await _resolve_neo(broker, neo_portfolio())
            self.contracts += neo
            await self._refresh_fx()
            logger.info(f"[trend] +{len(neo)} Neo assets (FX USD/RUB={self._fx_usdrub:.2f})")
        if not self.contracts:
            logger.error("[trend] No contracts resolved")
            return False
        self.by_base = {c.base: c for c in self.contracts}
        logger.info(
            f"[trend] Resolved {len(self.contracts)} contracts:\n  "
            + "\n  ".join(
                f"{c.base:<10} {c.ticker:<10} dte={c.days_to_expiry:>3}" for c in self.contracts
            )
        )

        try:
            self.portfolio_value = float(await broker.get_portfolio_value())
        except Exception:
            self.portfolio_value = 100_000.0
        logger.info(f"[trend] Portfolio: {self.portfolio_value:.0f} RUB")

        await self._reconcile()

        self._last_universe_refresh = datetime.now(timezone.utc)
        if self.notifier:
            self.notifier.push(
                Msg(
                    MsgType.BOOT,
                    f"Trend subsystem - {self.mode}",
                    f"Universe: {len(self.contracts)} contracts "
                    f"(mode={self.settings.TREND_UNIVERSE_MODE})\n"
                    f"Strategy: PATTERN-BASED (triple_top + triple_bottom)\n"
                    f"Freeze: {self.settings.TREND_FREEZE_NEW_ENTRIES}",
                )
            )
        self._initialised = True
        return True

    async def _figis_owned_by_other_bots(self) -> set:
        """Return FIGIs of currently-open positions owned by OTHER strategies
        (carry today; pairs if ever re-enabled).  Trend's orphan detector
        must skip these — otherwise carry's legs spam orphan alerts every
        5 minutes (the 2026-06-06 noise was exactly this).
        Read directly from the other DBs — cheap and avoids circular imports."""
        import aiosqlite

        owned = set()
        from futbot.carry.config import CarrySettings
        from futbot.pairs.config import PairsSettings
        from futbot.breakdown.config import BreakdownSettings

        for db_path, sql in [
            (
                CarrySettings().CARRY_DB_PATH,
                "SELECT figi_y, figi_x FROM pair_trades WHERE exit_time IS NULL",
            ),
            (
                PairsSettings().PAIRS_DB_PATH,
                "SELECT figi_y, figi_x FROM pair_trades WHERE exit_time IS NULL",
            ),
            # breakdown shorts single-stock futures — trend must not claim them
            (
                BreakdownSettings().BD_DB_PATH,
                "SELECT fut_figi, fut_figi FROM bd_trades WHERE exit_time IS NULL",
            ),
        ]:
            try:
                async with aiosqlite.connect(str(db_path)) as db:
                    cur = await db.execute(sql)
                    for row in await cur.fetchall():
                        owned.update(f for f in row if f)
            except Exception:
                pass
        return owned

    async def _broker_futures_positions(self) -> tuple[dict, bool]:
        """Return ({figi: signed_qty}, ok).  ok=False means the API call
        FAILED — callers MUST NOT treat the empty dict as 'no positions'
        (a transient failure previously false-closed a real position)."""
        try:
            pos = await self.broker.get_positions()
        except Exception as e:
            logger.warning(
                f"[trend] get_positions failed: {e} — skipping " f"gone/orphan detection this cycle"
            )
            return {}, False
        out = {}
        for f in getattr(pos, "futures", []) or []:
            figi = getattr(f, "figi", None)
            try:
                qty = int(getattr(f, "balance", 0))
            except Exception:
                qty = 0
            if figi:
                out[figi] = qty
        return out, True

    async def _reconcile(self):
        """Reconcile DB open trades against ACTUAL broker positions.

        Heals the paper/live desync that opened orphan live shorts:
          * paper-tagged trade while bot is LIVE  → close in DB (no order);
            it never existed at the broker.
          * live trade NOT present at broker       → closed externally/manually
            → close in DB at last price (no order).
          * broker position with NO matching DB trade → ORPHAN → alert (we do
            NOT auto-trade it; surfaced for manual/explicit handling).
        """
        open_rows = await self.db.open_trades()
        broker_pos, pos_ok = await self._broker_futures_positions()
        live_mode = not bool(self.settings.TREND_PAPER_MODE)
        matched_figis = set()
        reconciled = force_closed = purged = 0
        GONE_STRIKES_NEEDED = 2  # debounce: 2 consecutive confirmed-absent reads

        for r in open_rows:
            base = r["base"]
            figi = r["figi"]
            is_paper_trade = bool(r["paper"])

            # paper/live guard: a paper trade must never be managed with live
            # orders.  Safe regardless of API state (no broker data needed).
            if live_mode and is_paper_trade:
                await self.db.close_trade(
                    r["id"],
                    exit_price=float(r["entry_price"]),
                    exit_time=datetime.utcnow().isoformat(),
                    exit_reason="paper_purge_live_switch",
                    pnl=0.0,
                    pnl_pct=0.0,
                    commission_rub=0.0,
                )
                logger.warning(
                    f"[trend] RECONCILE {base}: paper position purged on "
                    f"live switch (was not real)"
                )
                purged += 1
                continue

            # "Gone at broker" detection — ONLY when get_positions succeeded,
            # AND debounced over GONE_STRIKES_NEEDED consecutive reads.  A
            # single transient empty read must NEVER close a real position
            # (that bug orphaned the Silver short on 2026-06-02).
            if live_mode and not is_paper_trade:
                if not pos_ok:
                    matched_figis.add(figi)  # assume still ours; don't touch
                    continue
                if broker_pos.get(figi, 0) == 0:
                    self._gone_strikes[figi] = self._gone_strikes.get(figi, 0) + 1
                    if self._gone_strikes[figi] < GONE_STRIKES_NEEDED:
                        logger.warning(
                            f"[trend] RECONCILE {base}: absent at broker "
                            f"(strike {self._gone_strikes[figi]}/{GONE_STRIKES_NEEDED}) "
                            f"— waiting for confirmation before closing"
                        )
                        matched_figis.add(figi)
                        continue
                    try:
                        fill = float(await self.broker.get_last_price(figi))
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
                    await self.db.close_trade(
                        r["id"],
                        exit_price=fill,
                        exit_time=datetime.utcnow().isoformat(),
                        exit_reason="reconciled_gone_at_broker",
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        commission_rub=comm_rub,
                    )
                    logger.warning(
                        f"[trend] RECONCILE {base}: live trade confirmed gone "
                        f"at broker ({GONE_STRIKES_NEEDED} reads) — closed @ "
                        f"{fill:.4f} (closed externally?)"
                    )
                    self._gone_strikes.pop(figi, None)
                    force_closed += 1
                    continue
                else:
                    self._gone_strikes.pop(figi, None)  # present → reset strikes

            matched_figis.add(figi)
            contract = self.by_base.get(base)
            if contract is None:
                try:
                    fill = float(await self.broker.get_last_price(r["figi"]))
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
                await self.db.close_trade(
                    r["id"],
                    exit_price=fill,
                    exit_time=datetime.utcnow().isoformat(),
                    exit_reason="boot_reconcile_unknown",
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    commission_rub=comm_rub,
                )
                logger.warning(
                    f"[trend] RECONCILE {base} (out of universe) - "
                    f"force-closed @ {fill:.4f} pnl={pnl:+.2f}RUB"
                )
                force_closed += 1
            else:
                tag = "pattern" if r["pattern_name"] else "legacy_bb"
                logger.info(
                    f"[trend] RECONCILE {base} {r['direction']} ({tag}) "
                    f"entered {r['entry_time'][:19]} - re-hydrated"
                )
                reconciled += 1

        # Orphan detection: broker futures positions with NO matching open DB
        # trade.  Only when get_positions SUCCEEDED — a failed (empty) read
        # must not be read as "no orphans".  We ALERT but never auto-trade.
        # Cross-bot guard: exclude FIGIs owned by OTHER bots' open trades
        # (carry holds both Si/GK legs; trend would otherwise spam "orphan"
        # every monitor tick — the 2026-06-06 noise was exactly this).
        other_owned = await self._figis_owned_by_other_bots()
        orphans = (
            {
                f: q
                for f, q in broker_pos.items()
                if q != 0 and f not in matched_figis and f not in other_owned
            }
            if pos_ok
            else {}
        )
        if orphans:
            msg = ", ".join(f"{f}:{q:+d}" for f, q in orphans.items())
            logger.error(f"[trend] ⚠ ORPHAN broker positions (no DB trade): {msg}")
            # Alert only NEW orphans (monitor runs every 5 min — avoid spam)
            new_orphans = {f for f in orphans if f not in self._alerted_orphans}
            if new_orphans and self.notifier:
                self.notifier.push(
                    Msg(
                        MsgType.ERROR,
                        "⚠ ORPHAN POSITION (manual check)",
                        f"Broker holds positions with no bot record:\n{msg}\n"
                        f"Not auto-traded — close manually if unintended.",
                    )
                )
            self._alerted_orphans = set(orphans.keys())
        elif pos_ok:
            self._alerted_orphans = set()  # only clear on a trusted read

        if open_rows or orphans:
            logger.info(
                f"[trend] Reconcile: {reconciled} re-hydrated, "
                f"{force_closed} closed-externally, {purged} paper-purged, "
                f"{len(orphans)} orphans"
            )

    async def tick(self):
        if not self._initialised:
            return
        t_start = datetime.now(timezone.utc)
        logger.info(f"[trend] -- Tick {t_start.isoformat()[:19]} --")

        # Refresh universe every 6h
        if (
            self._last_universe_refresh is None
            or (t_start - self._last_universe_refresh).total_seconds() > 6 * 3600
        ):
            logger.info("[trend] Refreshing portfolio resolution...")
            entries = (
                core_portfolio()
                if self.settings.TREND_UNIVERSE_MODE == "core"
                else tradeable_portfolio()
            )
            self.contracts = await _resolve_portfolio(
                self.broker,
                entries,
                self.settings.TREND_MIN_DAYS_TO_EXPIRY,
            )
            if bool(self.settings.TREND_TRADE_NEO):
                from futbot.trend.portfolio import neo_portfolio

                self.contracts += await _resolve_neo(self.broker, neo_portfolio())
                await self._refresh_fx()
            self.by_base = {c.base: c for c in self.contracts}
            self._last_universe_refresh = t_start

        # Daily kill check
        today_iso = t_start.date().isoformat()
        daily_pnl = await self.db.daily_pnl_rub(today_iso)
        loss_cap = self.portfolio_value * float(self.settings.TREND_DAILY_LOSS_PCT_LIMIT)
        kill_active = daily_pnl < -loss_cap
        if kill_active:
            logger.warning(
                f"[trend] Daily kill active: P&L {daily_pnl:+.0f}RUB < -{loss_cap:.0f}RUB"
            )

        for c in self.contracts:
            await self._process_contract(c, kill_active)

    async def monitor(self):
        """Fast position check (runs every ~5 min, between hourly signal ticks).

        Two jobs, both cheap:
          1. Reconcile against the broker — catches exchange-stop fills and
             external/manual closes, heals paper desync, alerts orphans.
          2. Enforce stop/target/timeout on open PATTERN positions at 5-min
             resolution (vs hourly), so we react far faster than the signal tick.
        Legacy Bollinger positions are left to the hourly tick (band_flip needs
        a full candle series, not just last price).
        """
        if not self._initialised:
            return
        await self._reconcile()
        for r in await self.db.open_trades():
            if not r["pattern_name"]:
                continue  # legacy bb → hourly tick
            c = self.by_base.get(r["base"])
            if c is None:
                continue
            try:
                last = float(await self.broker.get_last_price(c.figi))
            except Exception:
                continue
            reason = _pattern_exit_reason(
                direction=r["direction"],
                last_price=last,
                stop_price=float(r["stop_price"]),
                target_price=float(r["target_price"]),
                entry_time_iso=r["entry_time"],
                timeout_bars=PATTERN_TIMEOUT_BARS,
            )
            if reason:
                logger.info(f"[trend] MONITOR exit {r['base']} ({reason}) @ {last:.4f}")
                await self._close_position(c, r, reason)

    async def _process_contract(self, c: ResolvedContract, kill_active: bool):
        # 0. Qual-restricted instruments — already rejected once this session,
        # don't waste API calls or risk another backoff trying again.
        if c.base in self._qual_blocklist:
            return
        # 1. Rollover always wins
        open_t = await self.db.open_trade_for_base(c.base)
        if open_t is not None and c.days_to_expiry <= int(self.settings.TREND_ROLLOVER_DAYS):
            await self._do_rollover(c, open_t)
            return

        # 2. Manage existing position (two paths)
        if open_t is not None:
            if open_t["pattern_name"]:
                # New pattern position: use stop/target/timeout exit
                exit_reason = _pattern_exit_reason(
                    direction=open_t["direction"],
                    last_price=float(await self.broker.get_last_price(c.figi)),
                    stop_price=float(open_t["stop_price"]),
                    target_price=float(open_t["target_price"]),
                    entry_time_iso=open_t["entry_time"],
                    timeout_bars=PATTERN_TIMEOUT_BARS,
                )
                if exit_reason:
                    await self._close_position(c, open_t, exit_reason)
                else:
                    cur_p = float(await self.broker.get_last_price(c.figi))
                    logger.info(
                        f"[trend]   {c.base:<10} HOLD pattern={open_t['pattern_name']} "
                        f"dir={open_t['direction']} cur={cur_p:.4f} "
                        f"stop={open_t['stop_price']:.4f} "
                        f"target={open_t['target_price']:.4f}"
                    )
                return
            else:
                # Legacy Bollinger position: use band_flip exit
                await self._manage_legacy_bb(c, open_t)
                return

        # 3. No open position → look for pattern signal
        if kill_active:
            return
        if bool(self.settings.TREND_FREEZE_NEW_ENTRIES):
            return  # silent — not noisy logging when frozen
        open_rows = await self.db.open_trades()
        if len(open_rows) >= int(self.settings.TREND_MAX_OPEN_POSITIONS):
            return
        # Separate cap for Neo positions (higher leverage → tighter concurrency)
        if c.is_neo:
            neo_open = sum(1 for r in open_rows if (r["ticker"] or "").endswith("perpA"))
            if neo_open >= int(self.settings.TREND_NEO_MAX_OPEN):
                return

        df = await _fetch_ohlc(
            self.broker,
            c.figi,
            self.settings.TREND_CANDLE_HISTORY_DAYS,
        )
        if df.empty or len(df) < 100:
            logger.info(f"[trend]   {c.base}: insufficient bars ({len(df)})")
            return

        await self._scan_and_open(c, df)

    async def _scan_and_open(self, c: ResolvedContract, df: pd.DataFrame):
        """Run pattern detectors; open if a signal fires on the last few bars."""
        p = TUNED_PARAMS
        swings = find_swings(df, window=p.swing_window, min_prominence_pct=p.min_prominence_pct)
        if len(swings) < 5:
            return

        det_kw = dict(
            peak_tol=p.peak_tol,
            min_height=p.min_height,
            min_width=p.min_width,
            max_width=p.max_width,
            max_confirm_bars=p.max_confirm_bars,
        )
        signals = []
        if c.is_neo:
            # Neo assets: triple_top only (the validated edge — +196% on-margin,
            # 66% win; triple_bottom was weak at 49%).
            signals.extend(detect_triple_tops(df, swings, **det_kw))
        else:
            if is_pattern_allowed(c.base, "triple_top"):
                signals.extend(detect_triple_tops(df, swings, **det_kw))
            if is_pattern_allowed(c.base, "triple_bottom"):
                signals.extend(detect_triple_bottoms(df, swings, **det_kw))
        if not signals:
            return

        # We only care about signals on (or extremely close to) the last bar.
        # Live tick runs hourly, so a signal whose confirmation bar is more
        # than 2 bars in the past has likely already triggered earlier.
        last_idx = len(df) - 1
        fresh = [s for s in signals if last_idx - s.bar_idx <= 2]
        if not fresh:
            return
        # If multiple fresh signals, take the most recent
        sig = max(fresh, key=lambda s: s.bar_idx)

        # Dedupe: never enter the same signal bar twice (15-min ticks would
        # otherwise re-enter a just-stopped-out pattern while it stays fresh).
        prev = self._entered_sig_bar.get(c.base)
        if prev is not None and sig.bar_time <= prev:
            return

        direction = "buy" if sig.direction == +1 else "sell"
        lots = self._size_lots(c, sig.entry_price)

        # Free-margin guard for ALL live entries (was Neo-only; an expensive
        # FORTS leg like GDU6 at ~43k₽ ГО/lot needs the same protection).
        if not bool(self.settings.TREND_PAPER_MODE):
            if c.is_neo:
                if not await self._neo_margin_ok(c, sig.entry_price, lots):
                    return
            else:
                summ, ok = await self.broker.get_margin_summary()
                go = (
                    sig.entry_price
                    * c.rpp
                    * int(c.lot_size or 1)
                    * lots
                    * (c.risk_rate if c.risk_rate > 0 else 0.20)
                )
                buf = float(self.settings.TREND_NEO_MARGIN_BUFFER)
                if not ok or summ.get("available", 0.0) < go * buf:
                    logger.warning(
                        f"[trend] {c.base}: entry BLOCKED — free margin "
                        f"{summ.get('available', 0.0):,.0f}₽ < {buf:.1f}× ГО "
                        f"({go:,.0f}₽ for {lots} lots)"
                    )
                    return

        try:
            oid, fill = await _place_market(
                self.broker,
                c.figi,
                direction,
                lots,
                paper=bool(self.settings.TREND_PAPER_MODE),
            )
        except QualRestricted as e:
            # User isn't a qualified investor — skip this signal, don't kill
            # the runner.  Add base to a one-shot blocklist so we stop scanning
            # it this session (clears on next process start).
            self._qual_blocklist.add(c.base)
            logger.warning(
                f"[trend] {c.base}: SKIPPED — {e}.  " f"Add quals status in T-Invest app to enable."
            )
            return
        self._entered_sig_bar[c.base] = sig.bar_time
        trade_id = await self.db.insert_trade(
            base=c.base,
            ticker=c.ticker,
            figi=c.figi,
            direction=direction,
            lots=lots,
            bb_n=0,
            bb_k=0,  # legacy fields unused by pattern entries
            entry_price=fill,
            entry_time=datetime.utcnow().isoformat(),
            entry_order_id=oid,
            rub_per_point=c.rpp,
            lot_size=c.lot_size,
            paper=int(bool(self.settings.TREND_PAPER_MODE)),
            pattern_name=sig.pattern,
            stop_price=float(sig.stop_price),
            target_price=float(sig.target_price),
            pattern_height_pct=float(sig.pattern_height_pct),
        )
        logger.info(
            f"[trend]   {c.base}: OPEN {direction.upper()} {lots} @ {fill:.4f} "
            f"pattern={sig.pattern} stop={sig.stop_price:.4f} "
            f"target={sig.target_price:.4f}  trade_id={trade_id}"
        )
        # Real exchange stop-loss (hard protection if bot/monitor is down)
        await self._place_protective_stop(c, trade_id, direction, lots, float(sig.stop_price))
        if self.notifier:
            risk_pct = abs(sig.stop_price - fill) / max(fill, 1e-9) * 100
            reward_pct = abs(sig.target_price - fill) / max(fill, 1e-9) * 100
            self.notifier.push(
                Msg(
                    MsgType.TRADE_OPENED,
                    f"TREND {c.base} {direction.upper()} ({sig.pattern})",
                    f"entry: {fill:.4f}  lots: {lots}  ({c.ticker})\n"
                    f"stop: {sig.stop_price:.4f}  (-{risk_pct:.2f}%)\n"
                    f"target: {sig.target_price:.4f}  (+{reward_pct:.2f}%)\n"
                    f"R:R = 1:{reward_pct/max(risk_pct,1e-9):.2f}",
                )
            )

    async def _place_protective_stop(
        self, c: ResolvedContract, trade_id: int, direction: str, lots: int, stop_price: float
    ):
        """Place a real exchange stop-loss so the position is protected even
        if the bot/monitor is down.  Only ONE resting order (the stop) — the
        take-profit is handled by the monitor to avoid the OCO orphan trap
        (a resting TP firing after the position is already flat would open a
        new position).  Best-effort: failure doesn't block the trade."""
        if bool(self.settings.TREND_PAPER_MODE):
            return
        try:
            exit_dir = (
                StopOrderDirection.STOP_ORDER_DIRECTION_SELL
                if direction == "buy"
                else StopOrderDirection.STOP_ORDER_DIRECTION_BUY
            )
            sp = await self.broker._round_to_increment(c.figi, Decimal(str(stop_price)))
            sid = await self.broker.post_stop_loss(c.figi, lots, sp, exit_dir)
            await self.db.close_trade(trade_id, protective_stop_id=str(sid))
            logger.info(f"[trend]   {c.base}: protective stop @ {float(sp):.4f} id={sid}")
        except Exception as e:
            logger.warning(
                f"[trend]   {c.base}: protective stop NOT placed ({e}) — "
                f"monitor will enforce the stop instead"
            )

    async def _cancel_protective_stop(self, open_t):
        """Cancel the resting exchange stop before a bot-initiated close, so
        it can't fire afterwards and open a fresh (orphan) position."""
        sid = None
        try:
            sid = open_t["protective_stop_id"]
        except Exception:
            sid = None
        if not sid:
            return
        try:
            await self.broker.cancel_stop_order(sid)
            logger.info(f"[trend]   cancelled protective stop id={sid}")
        except Exception as e:
            logger.warning(f"[trend]   cancel protective stop {sid} failed: {e}")

    async def _close_position(self, c: ResolvedContract, open_t, exit_reason: str):
        # RACE GUARD — based on the BROKER POSITION, not the stop-order list.
        # The earlier "stop missing from active list ⇒ assume filled at
        # stop_price" was WRONG: on 2026-06-16 it false-closed a still-open,
        # PROFITABLE ROSN short at the stop price (reported -596 while the
        # broker showed +1871 and the position was still open).  Authoritative
        # truth = the actual position at the broker:
        #   • our side already flat  → exchange stop/external already closed it;
        #     do NOT send another order (would double-cover → orphan).
        #   • still open             → close it for real, use the broker fill.
        live = not bool(self.settings.TREND_PAPER_MODE)
        if live:
            pos, ok = await self.broker.get_positions_detail()
            if ok:
                qty = float(pos.get(c.figi, {}).get("qty", 0) or 0)
                bot_long = open_t["direction"] == "buy"
                already_flat = qty == 0 or (bot_long and qty <= 0) or ((not bot_long) and qty >= 0)
                if already_flat:
                    await self._cancel_protective_stop(open_t)
                    fill_price = float(await self.broker.get_last_price(c.figi))
                    await self._record_close(
                        c, open_t, fill_price, f"{exit_reason}(closed_externally)"
                    )
                    logger.info(
                        f"[trend]   {c.base}: already flat at broker "
                        f"(qty={qty}) — recorded close, no double-trade"
                    )
                    return

        # Cancel the resting exchange stop FIRST (avoid post-close orphan).
        await self._cancel_protective_stop(open_t)
        exit_dir = "sell" if open_t["direction"] == "buy" else "buy"
        oid, fill_price = await _place_market(
            self.broker,
            c.figi,
            exit_dir,
            open_t["lots"],
            paper=bool(self.settings.TREND_PAPER_MODE),
        )
        if not fill_price or fill_price <= 0:
            fill_price = float(await self.broker.get_last_price(c.figi))
        await self._record_close(c, open_t, fill_price, exit_reason, oid=oid)

    async def _record_close(
        self, c: ResolvedContract, open_t, fill_price: float, exit_reason: str, oid: str = ""
    ):
        """Compute P&L, write the DB close, log + notify.  Shared by the normal
        close path and the stop-already-filled race-guard path."""
        entry_dt = datetime.fromisoformat(open_t["entry_time"].replace("Z", "+00:00"))
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
        exit_dt = datetime.now(timezone.utc)
        if c.is_neo:
            await self._refresh_fx()
        pnl, pnl_pct, comm_rub = self._pnl_rub(
            c,
            open_t["direction"],
            open_t["entry_price"],
            fill_price,
            open_t["lots"],
            entry_dt=entry_dt,
            exit_dt=exit_dt,
        )
        await self.db.close_trade(
            open_t["id"],
            exit_price=fill_price,
            exit_time=datetime.utcnow().isoformat(),
            exit_reason=exit_reason,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission_rub=comm_rub,
            exit_order_id=oid,
        )
        cur = "USD" if c.is_neo else "RUB"
        logger.info(
            f"[trend]   {c.base}: CLOSE {exit_reason} @ {fill_price:.4f} {cur}  "
            f"pnl={pnl:+.2f}RUB ({pnl_pct:+.3f}%)"
            + (f"  fx={self._fx_usdrub:.2f}" if c.is_neo else "")
        )
        if self.notifier:
            tag = (open_t["pattern_name"] or "bb").upper()
            self.notifier.push(
                Msg(
                    MsgType.TRADE_CLOSED,
                    f"TREND {c.base} CLOSE ({tag})",
                    f"{exit_reason}\n"
                    f"entry: {open_t['entry_price']:.4f} -> exit: {fill_price:.4f}\n"
                    f"P&L: <b>{pnl:+.2f} RUB</b> ({pnl_pct:+.3f}%)",
                )
            )

    async def _manage_legacy_bb(self, c: ResolvedContract, open_t):
        """Manage a Bollinger-era position: exit on band_flip only.

        These are the 8 positions opened by the old strategy that are
        still in the DB.  We don't open new ones (frozen + no Bollinger
        scan path here), but we do honour their original exit rule.
        """
        df = await _fetch_ohlc(
            self.broker,
            c.figi,
            self.settings.TREND_CANDLE_HISTORY_DAYS,
        )
        if df.empty or len(df) < (c.bb_n or 20) + 5:
            return
        cur_pos = 1 if open_t["direction"] == "buy" else -1
        dec = strat.evaluate(
            close=df["close"],
            n=int(c.bb_n or open_t["bb_n"] or 20),
            k=float(c.bb_k or open_t["bb_k"] or 2.0),
            current_position=cur_pos,
        )
        if dec.action == "close":
            await self._close_position(c, open_t, "band_flip")
        else:
            cur_close = float(df["close"].iloc[-1])
            logger.info(
                f"[trend]   {c.base:<10} HOLD (legacy) close={cur_close:.4f} " f"pos={cur_pos:+d}"
            )

    async def _do_rollover(self, c: ResolvedContract, open_t):
        await self._close_position(c, open_t, "rollover")

    async def status(self) -> str:
        if not self._initialised or not self.db:
            return "Trend subsystem not initialised."

        open_t = await self.db.open_trades()
        today_iso = datetime.utcnow().date().isoformat()
        today_pnl = await self.db.daily_pnl_rub(today_iso)

        lines = [f"<b>TREND ({self.mode})</b>"]
        if bool(self.settings.TREND_FREEZE_NEW_ENTRIES):
            lines.append("⏸ <i>New entries FROZEN</i>")
        else:
            lines.append("Strategy: <b>PATTERN</b> (triple_top + triple_bottom)")
        lines.append(f"Universe: {len(self.contracts)} contracts")
        lines.append(f"Open positions: {len(open_t)}")

        # Live unrealized P&L per position — STRICTLY from the broker.
        # We report the broker's own expected_yield (= "Доход"/вармаржа shown
        # in the app), NOT a re-derived (cur-entry)×rpp number.  Re-deriving was
        # the source of past mismatches (rpp bugs, the ROSN false-close).  For
        # Neo the broker value is in USD → ×FX to RUB.
        detail, ok = await self.broker.get_positions_detail()
        if ok and detail:
            await self._refresh_fx()
        total_unreal = 0.0
        for r in open_t:
            entry_dt = datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00"))
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
            tag = r["pattern_name"] or "bb_legacy"
            d = detail.get(r["figi"], {})
            cur = d.get("current_price", 0.0) or float(r["entry_price"])
            is_neo = (r["ticker"] or "").endswith("perpA")
            # broker-truth P&L (expected_yield), RUB for FORTS, USD→RUB for Neo
            upnl = float(d.get("unrealized", 0.0)) * (self._fx_usdrub if is_neo else 1.0)
            sgn = 1.0 if r["direction"] == "buy" else -1.0
            upct = (cur - float(r["entry_price"])) / max(float(r["entry_price"]), 1e-9) * 100 * sgn
            total_unreal += upnl
            if r["pattern_name"]:
                lines.append(
                    f"  • {r['base']:<6} {r['direction']:>4} @ {r['entry_price']:.4f} "
                    f"→ {cur:.4f}  uP&L <b>{upnl:+.1f}₽</b> ({upct:+.2f}%)  "
                    f"({tag}) stop={r['stop_price']:.4f} tgt={r['target_price']:.4f} "
                    f"held={held:.1f}h"
                )
            else:
                lines.append(
                    f"  • {r['base']:<6} {r['direction']:>4} @ {r['entry_price']:.4f} "
                    f"→ {cur:.4f}  uP&L <b>{upnl:+.1f}₽</b> ({upct:+.2f}%)  "
                    f"({tag}) held={held:.1f}h"
                )
        if open_t:
            lines.append(f"Open unrealized P&L: <b>{total_unreal:+.2f} ₽</b>")
        lines.append(f"Today realised P&L: <b>{today_pnl:+.2f} ₽</b>")

        # Margin health (relevant for Neo leverage)
        if bool(self.settings.TREND_TRADE_NEO):
            summ, ok = await self.broker.get_margin_summary()
            if ok:
                suff = summ["sufficiency"]  # ratio: <1 ⇒ margin call
                flag = "🟢" if suff >= 2 else ("🟡" if suff >= 1.2 else "🔴")
                lines.append(
                    f"Margin {flag}: liquid {summ['liquid']:,.0f}₽ | "
                    f"used {summ['starting_margin']:,.0f}₽ | "
                    f"free {summ['available']:,.0f}₽ | buffer {suff:.1f}×"
                )

        # 7-day stats
        async with self.db._lock:
            cur = await self.db._db.execute(
                "SELECT COUNT(*) n, "
                "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) wins, "
                "COALESCE(SUM(pnl), 0) total "
                "FROM trend_trades WHERE exit_time >= ?",
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
