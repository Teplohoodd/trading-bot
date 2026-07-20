"""BrokerClient: async wrapper around tinkoff-investments SDK."""

import asyncio
import logging
import time as _time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from t_tech.invest import (
    AsyncClient,
    CandleInterval,
    OrderDirection,
    OrderType,
    StopOrderDirection,
    StopOrderExpirationType,
    StopOrderType,
    InstrumentStatus,
    InstrumentIdType,
    GetOrderBookResponse,
)
from t_tech.invest.utils import quotation_to_decimal, decimal_to_quotation

from utils.helpers import money_to_decimal

logger = logging.getLogger(__name__)


class _TokenBucket:
    """Simple async token bucket for rate limiting.

    Tinkoff Invest API allows 600 requests per 60 s = 10 req/s globally.
    We target 8 req/s (leaving 20% headroom for trading calls) to prevent
    RESOURCE_EXHAUSTED during screener + scan bursts.
    """

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate:     tokens added per second (= max sustained req/s)
            capacity: burst capacity (max simultaneous inflight before throttle)
        """
        self._rate = rate
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = _time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Block until a token is available."""
        while True:
            async with self._lock:
                now = _time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait)


class BrokerClient:
    def __init__(self, token: str, account_id: str, app_name: str = "trade_claude"):
        self._token = token
        self._account_id = account_id
        self._app_name = app_name
        self._client: Optional[AsyncClient] = None
        self._services = None
        self._reconnect_delay = 1.0
        # Rate limiter: 8 req/s sustained, burst up to 20.
        # The Tinkoff Invest quota is 600 req/60 s = 10 req/s.  We stay at 8
        # to leave room for order/portfolio calls that don't go through
        # get_candles but still count toward the quota.
        self._rate_limiter = _TokenBucket(rate=8.0, capacity=20)

    @property
    def account_id(self) -> str:
        return self._account_id

    # t-tech SDK defaults to invest-public-api.tbank.ru, whose cert chains to
    # the Russian Trusted Root CA (absent from grpc/certifi) → verify fails.
    # Pin to the legacy HARICA-cert host and force the SNI to match.
    _API_HOST = "invest-public-api.tinkoff.ru:443"
    _SSL_OPTS = [("grpc.ssl_target_name_override", "invest-public-api.tinkoff.ru")]

    async def connect(self):
        """Open async gRPC channel."""
        try:
            self._client = AsyncClient(
                self._token, target=self._API_HOST, options=self._SSL_OPTS,
                app_name=self._app_name)
            self._services = await self._client.__aenter__()
            self._reconnect_delay = 1.0
            logger.info("Broker connected")
        except Exception as e:
            logger.error(f"Broker connect failed: {e}")
            raise

    async def disconnect(self):
        """Close gRPC channel."""
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None
            self._services = None
            logger.info("Broker disconnected")

    async def _ensure_connected(self):
        """Reconnect if needed with exponential backoff."""
        if self._services is not None:
            return
        logger.warning(f"Reconnecting in {self._reconnect_delay}s...")
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 2, 60.0)
        await self.connect()

    async def _safe_call(self, coro_factory):
        """Execute API call with automatic reconnection + rate limiting.

        All Tinkoff Invest REST/gRPC calls go through here so the token
        bucket enforces the 600 req/min quota regardless of which method
        triggers the call.  On RESOURCE_EXHAUSTED the bucket automatically
        slows subsequent callers because the wait drains tokens.
        """
        await self._rate_limiter.acquire()
        for attempt in range(3):
            await self._ensure_connected()
            try:
                return await coro_factory()
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str:
                    # Back off and let the bucket refill before retrying.
                    # ratelimit_reset is typically 1-2 s; wait 2 s to be safe.
                    logger.debug(f"RESOURCE_EXHAUSTED on attempt {attempt+1}, waiting 2s")
                    await asyncio.sleep(2)
                    await self._rate_limiter.acquire()
                    continue
                if "UNAVAILABLE" in err_str or "INTERNAL" in err_str:
                    logger.warning(f"gRPC error (attempt {attempt+1}): {e}")
                    await self.disconnect()
                    continue
                raise
        raise ConnectionError("Failed after 3 reconnection attempts")

    # ==================== Instruments ====================

    async def find_instrument(self, query: str, kind: str = "") -> list:
        """Search instruments by ticker/name."""
        resp = await self._safe_call(
            lambda: self._services.instruments.find_instrument(query=query)
        )
        instruments = resp.instruments
        if kind:
            instruments = [i for i in instruments if i.instrument_kind.name == kind]
        return instruments

    async def get_share_by_figi(self, figi: str):
        """Get share details by FIGI."""
        resp = await self._safe_call(
            lambda: self._services.instruments.share_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi
            )
        )
        return resp.instrument

    async def get_all_shares(self):
        """Get all available shares."""
        resp = await self._safe_call(
            lambda: self._services.instruments.shares(
                instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
            )
        )
        return resp.instruments

    async def get_all_futures(self):
        """Get all available futures."""
        resp = await self._safe_call(
            lambda: self._services.instruments.futures(
                instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE
            )
        )
        return resp.instruments

    async def get_future_by_figi(self, figi: str):
        """Get future details by FIGI."""
        resp = await self._safe_call(
            lambda: self._services.instruments.future_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_FIGI, id=figi
            )
        )
        return resp.instrument

    async def get_instrument_info(self, figi: str) -> tuple:
        """Universal instrument lookup.

        Returns (instrument, kind) where kind is "share", "future", or "unknown".
        Tries share first, then future.
        """
        try:
            share = await self.get_share_by_figi(figi)
            if share is not None:
                return share, "share"
        except Exception:
            pass
        try:
            future = await self.get_future_by_figi(figi)
            if future is not None:
                return future, "future"
        except Exception:
            pass
        return None, "unknown"

    async def get_rub_per_point(self, figi: str) -> float:
        """REAL rub-per-point via GetFuturesMargin.

        get_all_futures does NOT return min_price_increment_amount, so
        extract_futures_metadata falls back to rub_per_point=1 — which is
        silently wrong for fractional-step contracts (LTU6 7.19₽/pt,
        GDU6/S1U6 71.9₽/pt: P&L was 7-72× understated and vol-target sizing
        oversized those legs).  This is the authoritative source.
        Returns 1.0 on failure (caller should log)."""
        try:
            m = await self._safe_call(
                lambda: self._services.instruments.get_futures_margin(figi=figi))
            inc = float(quotation_to_decimal(m.min_price_increment))
            amt = float(quotation_to_decimal(m.min_price_increment_amount))
            if inc > 0 and amt > 0:
                return amt / inc
        except Exception as e:
            logger.warning(f"get_rub_per_point({figi}) failed: {e}")
        return 1.0

    def extract_futures_metadata(self, instrument) -> dict:
        """Normalise a `Future` SDK object into trading-relevant metadata.

        Returns a dict with:
          - initial_margin_buy / initial_margin_sell: Decimal (RUB blocked per lot)
          - min_price_increment: Decimal (price-step, in points)
          - step_value: Decimal (RUB per 1 min_price_increment; aka "стоимость пункта")
          - rub_per_point: Decimal (RUB value of one point of price movement)
          - expiration_date: datetime | None
          - first_trade_date: datetime | None
          - asset_type: str ("TYPE_SECURITY"|"TYPE_COMMODITY"|"TYPE_CURRENCY"|"TYPE_INDEX"|"")
          - basic_asset: str (underlying ticker)

        All numeric fields default to Decimal(0) when SDK omits them.  Safe to
        call on non-futures — returns zero-filled dict in that case.
        """
        out = {
            "initial_margin_buy": Decimal(0),
            "initial_margin_sell": Decimal(0),
            "dlong": Decimal(0),   # ГО fraction for long (e.g. 0.24 = 24% of price)
            "dshort": Decimal(0),  # ГО fraction for short
            "min_price_increment": Decimal(0),
            "step_value": Decimal(0),
            "rub_per_point": Decimal(1),
            "expiration_date": None,
            "first_trade_date": None,
            "asset_type": "",
            "basic_asset": "",
        }
        if instrument is None:
            return out
        try:
            # SDK v0.2.0b59 does NOT expose initial_margin_on_buy/sell on the
            # Future object; those fields only appear in the older InstrumentShort
            # payload.  The actual margin ratios are in dlong / dshort.
            # We extract both; manager.py computes RUB ГО as dlong * price.
            mb = getattr(instrument, "initial_margin_on_buy", None)
            if mb is not None:
                out["initial_margin_buy"] = money_to_decimal(mb)
            ms = getattr(instrument, "initial_margin_on_sell", None)
            if ms is not None:
                out["initial_margin_sell"] = money_to_decimal(ms)

            # dlong / dshort — margin ratio (fraction of contract price)
            dl = getattr(instrument, "dlong", None)
            if dl is not None:
                out["dlong"] = quotation_to_decimal(dl) or Decimal(0)
            ds = getattr(instrument, "dshort", None)
            if ds is not None:
                out["dshort"] = quotation_to_decimal(ds) or Decimal(0)

            mpi = getattr(instrument, "min_price_increment", None)
            if mpi is not None:
                out["min_price_increment"] = quotation_to_decimal(mpi) or Decimal(0)

            # min_price_increment_amount absent in this SDK version — skip
            mpia = getattr(instrument, "min_price_increment_amount", None)
            if mpia is not None:
                out["step_value"] = money_to_decimal(mpia)

            # RUB value of moving 1 full point of price = step_value / min_price_increment.
            # When step_value is unavailable (SDK limitation), fall back to 1.0.
            # All FORTS contract prices returned by T-API candles are in RUB,
            # so rub_per_point=1 is a reasonable approximation for risk sizing.
            if out["step_value"] > 0 and out["min_price_increment"] > 0:
                out["rub_per_point"] = out["step_value"] / out["min_price_increment"]

            out["expiration_date"] = getattr(instrument, "expiration_date", None)
            out["first_trade_date"] = getattr(instrument, "first_trade_date", None)
            at = getattr(instrument, "asset_type", "") or ""
            out["asset_type"] = getattr(at, "name", None) or str(at)
            out["basic_asset"] = getattr(instrument, "basic_asset", "") or ""
        except Exception as e:
            logger.debug(f"extract_futures_metadata partial fail: {e}")
        return out

    async def get_fundamentals(self, asset_uid: str) -> dict:
        """Get fundamental metrics for an instrument via asset_uid.

        Returns dict with P/E, P/B, EV/EBITDA, dividend yield, market cap, etc.
        Returns empty dict if fundamentals are unavailable for this instrument.
        """
        try:
            resp = await self._safe_call(
                lambda: self._services.instruments.get_asset_fundamentals(
                    ids=[asset_uid]
                )
            )
            if not resp.fundamentals:
                return {}
            f = resp.fundamentals[0]

            def _f(val) -> float | None:
                """Extract float from a protobuf DoubleValue wrapper (or None)."""
                if val is None:
                    return None
                try:
                    v = float(val.value) if hasattr(val, "value") else float(val)
                    return round(v, 4) if v != 0 else None
                except Exception:
                    return None

            return {
                "market_cap": _f(f.market_capitalization),
                "pe": _f(f.p_to_e_ttm),
                "pb": _f(f.p_to_bv),
                "ev_ebitda": _f(f.ev_to_ebitda),
                "dividend_yield": _f(f.dividend_yield_12m),
                "revenue_ttm": _f(f.revenue_ttm),
                "net_income_ttm": _f(f.net_income_ttm),
                "total_debt": _f(getattr(f, "total_debt", None)),
                "net_margin": _f(getattr(f, "net_margin_ttm", None)),
            }
        except Exception as e:
            logger.debug(f"get_fundamentals failed for {asset_uid}: {e}")
            return {}

    # ==================== Market Data ====================

    async def get_candles(self, figi: str, from_dt: datetime, to_dt: datetime,
                          interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR) -> list:
        """Get historical candles. Handles multi-day fetching for 1min interval."""
        all_candles = []

        # Tinkoff Invest API per-request limits (empirically probed 2026-04-17
        # via scripts/test_candle_limits.py against SBER BBG004730N88):
        #   HOUR: 100d boundary — API returns INVALID_ARGUMENT 30014 at ≥110d,
        #         OK up to 100d.  90d span keeps a margin for policy tightening.
        #   1_MIN / 5_MIN / 15_MIN: documented 1-day / 1-week caps.
        #   DAY+: effectively unlimited, use 365d chunks for memory.
        # Previously HOUR was set to 365d which silently failed every ML-train
        # hourly fetch — trainer was falling back to daily candles for every
        # ticker (logs: "hourly unavailable, using N daily candles").
        max_span = {
            CandleInterval.CANDLE_INTERVAL_1_MIN: timedelta(days=1),
            CandleInterval.CANDLE_INTERVAL_5_MIN: timedelta(days=1),
            CandleInterval.CANDLE_INTERVAL_15_MIN: timedelta(days=7),
            CandleInterval.CANDLE_INTERVAL_HOUR: timedelta(days=90),
            CandleInterval.CANDLE_INTERVAL_DAY: timedelta(days=365),
            CandleInterval.CANDLE_INTERVAL_WEEK: timedelta(days=730),
            CandleInterval.CANDLE_INTERVAL_MONTH: timedelta(days=3650),
        }
        span = max_span.get(interval, timedelta(days=90))

        current_from = from_dt
        while current_from < to_dt:
            current_to = min(current_from + span, to_dt)
            try:
                resp = await self._safe_call(
                    lambda f=current_from, t=current_to: self._services.market_data.get_candles(
                        figi=figi, from_=f, to=t, interval=interval
                    )
                )
                all_candles.extend(resp.candles)
            except Exception as e:
                err_str = str(e)
                # 30014 = candle data unavailable for this instrument/interval
                if "30014" in err_str or (
                    "INVALID_ARGUMENT" in err_str and "candle" in err_str.lower()
                ):
                    logger.debug(f"Candles unavailable for {figi}: {e}")
                    break
                raise
            current_from = current_to
            if current_from < to_dt:
                await asyncio.sleep(0.1)  # Rate limit respect

        return all_candles

    async def get_last_price(self, figi: str) -> Decimal:
        """Get latest price for a single instrument."""
        resp = await self._safe_call(
            lambda: self._services.market_data.get_last_prices(figi=[figi])
        )
        if resp.last_prices:
            return quotation_to_decimal(resp.last_prices[0].price)
        return Decimal(0)

    async def get_last_prices(self, figis: list[str]) -> dict[str, Decimal]:
        """Get latest prices for multiple instruments."""
        resp = await self._safe_call(
            lambda: self._services.market_data.get_last_prices(figi=figis)
        )
        return {
            lp.figi: quotation_to_decimal(lp.price)
            for lp in resp.last_prices
        }

    async def get_order_book(self, figi: str, depth: int = 20) -> GetOrderBookResponse:
        """Get order book (bid/ask levels)."""
        return await self._safe_call(
            lambda: self._services.market_data.get_order_book(figi=figi, depth=depth)
        )

    async def get_trading_status(self, figi: str):
        """Get current trading status of instrument."""
        return await self._safe_call(
            lambda: self._services.market_data.get_trading_status(figi=figi)
        )

    # ==================== Portfolio & Positions ====================

    async def get_portfolio(self):
        """Get portfolio positions."""
        return await self._safe_call(
            lambda: self._services.operations.get_portfolio(account_id=self._account_id)
        )

    async def get_positions(self):
        """Get current positions (money + securities)."""
        return await self._safe_call(
            lambda: self._services.operations.get_positions(account_id=self._account_id)
        )

    async def get_margin_attributes(self):
        """Get margin info for leverage trading."""
        return await self._safe_call(
            lambda: self._services.users.get_margin_attributes(account_id=self._account_id)
        )

    async def get_margin_summary(self) -> tuple[dict, bool]:
        """Margin health → ({...}, ok).  ok=False if the API call failed.

        Keys: liquid (ликвидный портфель), starting_margin (начальная маржа
        used by open positions), minimal_margin, available (free margin =
        liquid − starting), sufficiency_pct (запас прочности).  A margin call
        triggers when liquid < minimal_margin.
        """
        try:
            ma = await self.get_margin_attributes()
        except Exception as e:
            logger.warning(f"get_margin_summary failed: {e}")
            return {}, False

        def _m(name):
            v = getattr(ma, name, None)
            try:
                return float(money_to_decimal(v)) if v is not None else 0.0
            except Exception:
                return 0.0
        liquid = _m("liquid_portfolio")
        starting = _m("starting_margin")
        minimal = _m("minimal_margin")
        # funds_sufficiency_level is a Quotation RATIO (liquid ÷ minimal margin):
        # >1 = above the margin-call floor, the higher the safer.  Fall back to
        # computing it ourselves if the field is absent.
        suff = getattr(ma, "funds_sufficiency_level", None)
        try:
            suff_ratio = float(quotation_to_decimal(suff)) if suff is not None else 0.0
        except Exception:
            suff_ratio = 0.0
        if suff_ratio <= 0 and minimal > 0:
            suff_ratio = liquid / minimal
        return {
            "liquid": liquid,
            "starting_margin": starting,
            "minimal_margin": minimal,
            "available": max(0.0, liquid - starting),
            "sufficiency": suff_ratio,   # ratio: <1 ⇒ margin call
        }, True

    async def get_operations(self, from_dt: datetime, to_dt: datetime | None = None,
                             operation_types=None):
        """Account operations via the modern CURSOR endpoint (paginated),
        replacing the deprecated operations.get_operations.

        Returns a list of OperationItem — note the type field is `.type`
        (OperationType enum), plus `.figi`, `.price` (Quotation), `.date`.
        `operation_types` (optional list of OperationType) is applied
        SERVER-SIDE, so callers can fetch just the ops they need."""
        from t_tech.invest import (GetOperationsByCursorRequest,
                                    OperationState)
        if to_dt is None:
            to_dt = datetime.now(timezone.utc)
        out, cursor = [], ""
        while True:
            req = GetOperationsByCursorRequest(
                account_id=self._account_id, from_=from_dt, to=to_dt,
                operation_types=operation_types or [], cursor=cursor, limit=1000,
                state=OperationState.OPERATION_STATE_EXECUTED)
            resp = await self._safe_call(
                lambda r=req: self._services.operations.get_operations_by_cursor(
                    request=r))
            out.extend(resp.items)
            if not resp.has_next or not resp.next_cursor:
                break
            cursor = resp.next_cursor
        return out

    async def get_accounts(self):
        """Get all accounts."""
        resp = await self._safe_call(
            lambda: self._services.users.get_accounts()
        )
        return resp.accounts

    async def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in RUB."""
        portfolio = await self.get_portfolio()
        return money_to_decimal(portfolio.total_amount_portfolio)

    async def get_positions_detail(self) -> tuple[dict, bool]:
        """Return ({figi: {...}}, ok) of CURRENT positions, broker-computed.

        Per figi: qty (signed lots), avg_price, current_price, unrealized
        (expected_yield, in the quote unit), currency.  Source of truth for
        live position tracking AND unrealized-P&L display — avoids the bot
        re-deriving prices/currency itself.  ok=False ⇒ API failed.
        """
        try:
            pf = await self.get_portfolio()
        except Exception as e:
            logger.warning(f"get_positions_detail failed: {e}")
            return {}, False
        out = {}
        for p in getattr(pf, "positions", []) or []:
            itype = getattr(p, "instrument_type", "") or ""
            if itype == "currency":
                continue
            figi = getattr(p, "figi", None)
            if not figi:
                continue
            cp = getattr(p, "current_price", None)
            ap = getattr(p, "average_position_price", None)
            out[figi] = {
                "qty": float(quotation_to_decimal(p.quantity)) if getattr(p, "quantity", None) else 0.0,
                "avg_price": float(money_to_decimal(ap)) if ap else 0.0,
                "current_price": float(money_to_decimal(cp)) if cp else 0.0,
                "unrealized": float(quotation_to_decimal(p.expected_yield))
                              if getattr(p, "expected_yield", None) else 0.0,
                "currency": (getattr(cp, "currency", "") or "").upper(),
                "instrument_type": itype,
            }
        return out, True

    # ==================== Orders ====================

    async def post_market_order(self, figi: str, lots: int,
                                 direction: OrderDirection) -> object:
        """Place a market order."""
        logger.info(f"Market order: {direction.name} {lots} lots of {figi}")
        return await self._safe_call(
            lambda: self._services.orders.post_order(
                figi=figi,
                quantity=lots,
                direction=direction,
                account_id=self._account_id,
                order_type=OrderType.ORDER_TYPE_MARKET,
            )
        )

    async def post_market_order_with_fill(self, figi: str, lots: int,
                                          direction: OrderDirection) -> dict:
        """Place a market order and return the REAL fill, not a last-price guess.

        The previous code recorded `get_last_price` snapshots as the "fill",
        which on thin spreads flipped P&L signs (a -73.6 ₽ carry trade was
        logged as +39.4 ₽).  This reads the actual executed price + commission
        from the broker's PostOrderResponse (or a one-shot order-state poll).

        Returns {order_id, fill_price (instrument points), commission_rub,
        lots_executed}.  fill_price falls back to last price only if the venue
        returns no executed price at all (logged as a warning).
        """
        resp = await self.post_market_order(figi, lots, direction)
        order_id = getattr(resp, "order_id", None) or "?"

        def _m(v) -> float:
            try:
                return float(money_to_decimal(v)) if v is not None else 0.0
            except Exception:
                return 0.0

        # Anchor: the per-lot PRICE must be near the last traded price.  Some
        # contracts return executed_order_price as the position VALUE in rubles
        # (e.g. Silver: 5318 ₽ = 74.88 pt × 71 ₽/pt) instead of the price 74.88,
        # which silently corrupted entry prices.  We sanity-check every
        # candidate against last_price and reject anything off by >50%.
        last_p = float(await self.get_last_price(figi))

        def _price_ok(p: float) -> bool:
            return p > 0 and last_p > 0 and 0.5 <= (p / last_p) <= 2.0

        candidates = [
            _m(getattr(resp, "executed_order_price", None)),
            _m(getattr(resp, "average_position_price", None)),
        ]
        commission = _m(getattr(resp, "executed_commission", None))
        lots_exec = int(getattr(resp, "lots_executed", 0) or 0)

        if not any(_price_ok(c) for c in candidates) and order_id != "?":
            # Poll order state once — executed price may settle slightly later.
            try:
                st = await self.get_order_state(order_id)
                candidates += [
                    _m(getattr(st, "average_position_price", None)),
                    _m(getattr(st, "executed_order_price", None)),
                ]
                if commission <= 0:
                    commission = _m(getattr(st, "executed_commission", None))
                if lots_exec <= 0:
                    lots_exec = int(getattr(st, "lots_executed", 0) or 0)
            except Exception:
                pass

        fill = next((c for c in candidates if _price_ok(c)), 0.0)
        if fill <= 0:
            fill = last_p
            logger.warning(
                f"order {order_id} ({figi}): no sane executed price "
                f"(candidates={[round(c, 2) for c in candidates]}, last={last_p:.4f}) "
                f"— using last price (P&L estimate)"
            )
        return {
            "order_id": order_id,
            "fill_price": fill,
            "commission_rub": commission,
            "lots_executed": lots_exec or lots,
        }

    async def post_limit_order(self, figi: str, lots: int, price: Decimal,
                                direction: OrderDirection) -> object:
        """Place a limit order."""
        logger.info(f"Limit order: {direction.name} {lots} lots of {figi} @ {price}")
        return await self._safe_call(
            lambda: self._services.orders.post_order(
                figi=figi,
                quantity=lots,
                price=decimal_to_quotation(price),
                direction=direction,
                account_id=self._account_id,
                order_type=OrderType.ORDER_TYPE_LIMIT,
            )
        )

    async def cancel_order(self, order_id: str):
        """Cancel an active order."""
        logger.info(f"Cancelling order {order_id}")
        return await self._safe_call(
            lambda: self._services.orders.cancel_order(
                account_id=self._account_id, order_id=order_id
            )
        )

    async def get_orders(self) -> list:
        """Get all active orders."""
        resp = await self._safe_call(
            lambda: self._services.orders.get_orders(account_id=self._account_id)
        )
        return resp.orders

    async def get_order_state(self, order_id: str):
        """Get state of a specific order."""
        return await self._safe_call(
            lambda: self._services.orders.get_order_state(
                account_id=self._account_id, order_id=order_id
            )
        )

    # ==================== Stop Orders ====================

    async def post_stop_loss(self, figi: str, lots: int, stop_price: Decimal,
                              direction: StopOrderDirection) -> str:
        """Place a stop-loss order. Returns stop_order_id."""
        resp = await self._safe_call(
            lambda: self._services.stop_orders.post_stop_order(
                figi=figi,
                quantity=lots,
                stop_price=decimal_to_quotation(stop_price),
                direction=direction,
                account_id=self._account_id,
                stop_order_type=StopOrderType.STOP_ORDER_TYPE_STOP_LOSS,
                expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
            )
        )
        logger.info(f"Stop-loss placed: {figi} @ {stop_price}, id={resp.stop_order_id}")
        return resp.stop_order_id

    async def post_take_profit(self, figi: str, lots: int, stop_price: Decimal,
                                direction: StopOrderDirection) -> str:
        """Place a take-profit order. Returns stop_order_id."""
        resp = await self._safe_call(
            lambda: self._services.stop_orders.post_stop_order(
                figi=figi,
                quantity=lots,
                stop_price=decimal_to_quotation(stop_price),
                direction=direction,
                account_id=self._account_id,
                stop_order_type=StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT,
                expiration_type=StopOrderExpirationType.STOP_ORDER_EXPIRATION_TYPE_GOOD_TILL_CANCEL,
            )
        )
        logger.info(f"Take-profit placed: {figi} @ {stop_price}, id={resp.stop_order_id}")
        return resp.stop_order_id

    async def cancel_stop_order(self, stop_order_id: str):
        """Cancel a stop order."""
        return await self._safe_call(
            lambda: self._services.stop_orders.cancel_stop_order(
                account_id=self._account_id, stop_order_id=stop_order_id
            )
        )

    async def get_stop_orders(self) -> list:
        """Get all active stop orders."""
        resp = await self._safe_call(
            lambda: self._services.stop_orders.get_stop_orders(
                account_id=self._account_id
            )
        )
        return resp.stop_orders

    # ==================== Smart Order Execution ====================

    async def get_fair_price(self, figi: str, direction: str,
                              mode: str = "limit_aggressive") -> Optional[Decimal]:
        """Calculate fair limit price from order book.

        Args:
            figi: Instrument FIGI.
            direction: "buy" or "sell".
            mode: "limit_aggressive" (cross spread) or "limit_passive" (join queue).

        Returns:
            Fair price as Decimal, or None if order book unavailable.
        """
        try:
            ob = await self.get_order_book(figi, depth=5)
            if not ob.bids or not ob.asks:
                return None

            best_bid = quotation_to_decimal(ob.bids[0].price)
            best_ask = quotation_to_decimal(ob.asks[0].price)
            mid = (best_bid + best_ask) / 2

            if mode == "limit_aggressive":
                # Cross the spread — buy at ask, sell at bid
                # Guaranteed fill once order hits exchange
                price = best_ask if direction == "buy" else best_bid
            elif mode == "limit_passive":
                # Join the queue — buy at bid+1tick, sell at ask-1tick
                # Saves spread but may not fill immediately
                price = best_bid if direction == "buy" else best_ask
            else:
                return None

            # Round to instrument's min_price_increment
            price = await self._round_to_increment(figi, price)
            logger.debug(
                f"Fair price {figi} {direction}: {price} "
                f"(bid={best_bid}, ask={best_ask}, mid={mid:.4f}, mode={mode})"
            )
            return price

        except Exception as e:
            logger.warning(f"Could not compute fair price for {figi}: {e}")
            return None

    async def _round_to_increment(self, figi: str, price: Decimal) -> Decimal:
        """Round price to instrument's min_price_increment (share or future)."""
        try:
            instrument, _kind = await self.get_instrument_info(figi)
            if instrument is not None:
                inc = quotation_to_decimal(instrument.min_price_increment)
                if inc and inc > 0:
                    rounded = (price / inc).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * inc
                    return rounded
        except Exception:
            pass
        return price

    async def wait_for_fill(self, order_id: str, timeout: float = 45.0) -> object:
        """Poll order state until fully filled or timeout/cancel.

        Returns the final OrderState.
        """
        deadline = _time.monotonic() + timeout
        poll_interval = 1.0  # seconds between polls

        while _time.monotonic() < deadline:
            try:
                state = await self.get_order_state(order_id)
                status = state.execution_report_status.name

                if status in (
                    "EXECUTION_REPORT_STATUS_FILL",       # fully filled
                    "EXECUTION_REPORT_STATUS_CANCELLED",
                    "EXECUTION_REPORT_STATUS_REJECTED",
                ):
                    return state

                # Partially filled — continue waiting
                if state.lots_executed >= state.lots_requested:
                    return state

            except Exception as e:
                logger.warning(f"Order state poll error ({order_id}): {e}")

            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.2, 3.0)  # back-off up to 3s

        return await self.get_order_state(order_id)

    async def post_limit_with_fallback(
        self,
        figi: str,
        lots: int,
        direction: OrderDirection,
        mode: str = "limit_aggressive",
        timeout: float = 45.0,
        fallback_market: bool = True,
    ) -> tuple[object, float, str, int]:
        """Place smart limit order; fallback to market if not filled in time.

        Returns:
            (order_response, avg_fill_price, order_type_used, filled_lots)

        `filled_lots` is the ACTUAL total filled quantity.  Callers MUST use
        this (not the requested `lots`) when placing stop-loss / take-profit
        orders — otherwise a partial fill leaves orphan uncovered lots or a
        stop that's bigger than our real position (broker rejects it).
        """
        direction_str = "buy" if direction == OrderDirection.ORDER_DIRECTION_BUY else "sell"

        fair_price = await self.get_fair_price(figi, direction_str, mode)

        if fair_price is None:
            # No order book — go straight to market
            resp = await self.post_market_order(figi, lots, direction)
            fill_price = float(await self.get_last_price(figi))
            filled_lots = getattr(resp, "lots_executed", lots) or lots
            return resp, fill_price, "market_fallback", filled_lots

        # Place limit order
        resp = await self.post_limit_order(figi, lots, fair_price, direction)
        order_id = resp.order_id

        # Wait for fill
        state = await self.wait_for_fill(order_id, timeout)
        filled_lots = getattr(state, "lots_executed", 0)
        total_lots = getattr(state, "lots_requested", lots)

        # Calculate weighted average fill price
        avg_price_q = getattr(state, "average_position_price", None)
        if avg_price_q and filled_lots > 0:
            avg_fill_price = float(quotation_to_decimal(avg_price_q))
        else:
            avg_fill_price = float(fair_price)

        if filled_lots >= total_lots:
            logger.info(
                f"Limit order filled: {direction_str} {filled_lots} lots {figi} "
                f"@ {avg_fill_price:.4f} (placed {fair_price})"
            )
            return state, avg_fill_price, "limit", filled_lots

        # Partial or no fill — cancel, then RE-CHECK state before falling back.
        # There is a natural race here: between wait_for_fill's last poll and
        # cancel_order landing at the broker, the remaining lots can fill.  If
        # we don't re-check we end up posting a market order for "unfilled"
        # lots that the limit order just filled → duplicate position (the
        # NVTK 14:02 / 14:05 pattern).
        try:
            await self.cancel_order(order_id)
            logger.info(f"Limit order cancelled (filled {filled_lots}/{total_lots}), falling back")
        except Exception as e:
            logger.warning(f"Cancel order error: {e}")

        # Re-poll state after the cancel attempt — the broker may have
        # filled additional lots while our cancel was in flight.
        try:
            post_cancel_state = await self.get_order_state(order_id)
            post_filled = getattr(post_cancel_state, "lots_executed", filled_lots) or filled_lots
            if post_filled > filled_lots:
                logger.info(
                    f"Limit filled additional {post_filled - filled_lots} lots "
                    f"during cancel race ({filled_lots} → {post_filled}/{total_lots})"
                )
                post_avg_q = getattr(post_cancel_state, "average_position_price", None)
                if post_avg_q and post_filled > 0:
                    avg_fill_price = float(quotation_to_decimal(post_avg_q))
                filled_lots = post_filled
                state = post_cancel_state
                # Fully covered now?  Skip the fallback entirely — otherwise
                # we'd post a duplicate market order on top.
                if filled_lots >= total_lots:
                    logger.info(
                        f"Limit fully filled after cancel race: "
                        f"{direction_str} {filled_lots}/{total_lots} {figi} "
                        f"@ {avg_fill_price:.4f}"
                    )
                    return state, avg_fill_price, "limit", filled_lots
        except Exception as e:
            logger.debug(f"Post-cancel state re-check failed: {e}")

        unfilled_lots = total_lots - filled_lots
        if fallback_market and unfilled_lots > 0:
            market_resp = await self.post_market_order(figi, unfilled_lots, direction)
            market_price = float(await self.get_last_price(figi))
            # Trust market_resp.lots_executed if present; market orders
            # usually fully fill on liquid names but rejects/partial-fills
            # do happen (circuit breaker, halted instrument, etc.)
            market_filled = getattr(market_resp, "lots_executed", unfilled_lots) or unfilled_lots
            total_filled = filled_lots + market_filled

            # Blend fill price over ACTUAL filled lots (not requested)
            if total_filled > 0:
                blended = (
                    (avg_fill_price * filled_lots + market_price * market_filled)
                    / total_filled
                )
            else:
                blended = market_price

            logger.info(
                f"Market fallback: {market_filled}/{unfilled_lots} lots @ {market_price:.4f} "
                f"(total_filled={total_filled}/{total_lots}, blended={blended:.4f})"
            )
            return market_resp, blended, "limit+market_fallback", total_filled

        return state, avg_fill_price, "limit_partial", filled_lots
