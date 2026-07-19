"""Streaming subscription manager.

Subscribes to ORDER BOOK + TRADES + 1-min CANDLES for a set of instruments
and maintains a live `InstrumentState` per FIGI.  Consumers can call
`get_state(figi)` from anywhere for a thread-safe snapshot.

Why we don't use polling:
  * Polling `get_order_book` at 1Hz = 50 RPS for 50 instruments — burns
    the 50 req/sec budget for nothing.
  * Streams push events as they happen (book throttled to 100ms, trades
    unthrottled).  Lower latency, lower API load.

Architecture:
  * One asyncio task per subscription type runs the stream consumer.
  * Each event updates the per-FIGI state via the state aggregator.
  * State is a plain dataclass guarded by an asyncio.Lock.

Failure handling:
  * The stream task auto-reconnects on disconnection with exponential backoff.
  * If a stream is dead for > 30s, the safety tick in scalp.main detects
    stale state and refuses to take new positions.
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from t_tech.invest import (
    AsyncClient,
    MarketDataRequest,
    SubscribeOrderBookRequest,
    SubscribeTradesRequest,
    SubscribeCandlesRequest,
    SubscriptionAction,
    SubscriptionInterval,
    OrderBookInstrument,
    TradeInstrument,
    CandleInstrument,
    TradeDirection,
)
from t_tech.invest.utils import quotation_to_decimal

logger = logging.getLogger("futbot.scalp.stream")


# ── Per-instrument live state ───────────────────────────────────────────────
@dataclass
class BookLevel:
    price: float
    volume: int  # in LOTS as reported by exchange


@dataclass
class InstrumentState:
    figi: str
    ticker: str
    # Order book snapshot
    bids: list[BookLevel] = field(default_factory=list)
    asks: list[BookLevel] = field(default_factory=list)
    last_book_update: float = 0.0  # monotonic
    # Recent trades (rolling window for TFI)
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=500))
    last_trade_update: float = 0.0
    # 1-min candles (the most recent ~240 closed bars).  Used for the
    # microstructure-side signals (RSI, EMA, VWAP).
    candles_1m: deque = field(default_factory=lambda: deque(maxlen=240))
    last_candle_update: float = 0.0
    # 5-min candles (the SDK's largest streaming interval; we use these
    # plus an aggregation factor of 3 for "15-min ATR" used in TP/SL sizing).
    candles_5m: deque = field(default_factory=lambda: deque(maxlen=200))
    last_candle_5m_update: float = 0.0
    # Latest last-trade price (denormalized for fast access)
    last_price: float = 0.0

    def is_fresh(self, max_age_seconds: float = 30.0) -> bool:
        """Have we seen any update recently?  Used by the safety tick."""
        now = time.monotonic()
        latest = max(self.last_book_update, self.last_trade_update, self.last_candle_update)
        return latest > 0 and (now - latest) < max_age_seconds


class StreamManager:
    def __init__(self, *, token: str, app_name: str = "futbot-scalp"):
        self.token = token
        self.app_name = app_name
        self._instruments: dict[str, InstrumentState] = {}
        self._lock = asyncio.Lock()
        self._tasks: list[asyncio.Task] = []
        self._stop = False
        self._book_depth = 10

    async def add_instrument(self, figi: str, ticker: str):
        """Register a figi so its state is initialised.  Subscriptions are
        applied at start_streaming() time — call this BEFORE that."""
        async with self._lock:
            if figi not in self._instruments:
                self._instruments[figi] = InstrumentState(figi=figi, ticker=ticker)

    async def prefetch_history(self, broker, figi: str, hours_5m: int = 24, hours_1m: int = 3):
        """Backfill BOTH 1-min and 5-min candle deques via REST get_candles.
          * 5-min × 24h  → ~280 bars → ATR(14) on 15-min aggregate ready instantly
          * 1-min × 3h   → ~180 bars → RSI/EMA/VWAP signals ready instantly

        Without this the bot needs 3+ hours of streaming before it can
        score signals or size TP/SL.  Silent no-op on failure (stream
        catches up over time).
        """
        from datetime import datetime, timedelta, timezone
        from t_tech.invest import CandleInterval

        async def _fetch(interval, hours, target_deque, label):
            try:
                now = datetime.now(timezone.utc)
                candles = await broker.get_candles(
                    figi,
                    now - timedelta(hours=hours),
                    now,
                    interval=interval,
                )
            except Exception as e:
                logger.warning(f"  {figi} prefetch {label} failed ({e})")
                return 0
            for c in candles:
                bar = {
                    "time": c.time,
                    "open": float(quotation_to_decimal(c.open)),
                    "high": float(quotation_to_decimal(c.high)),
                    "low": float(quotation_to_decimal(c.low)),
                    "close": float(quotation_to_decimal(c.close)),
                    "volume": int(c.volume),
                }
                target_deque.append(bar)
            return len(candles)

        async with self._lock:
            st = self._instruments.get(figi)
            if st is None:
                return
            n5 = await _fetch(
                CandleInterval.CANDLE_INTERVAL_5_MIN,
                hours_5m,
                st.candles_5m,
                "5m",
            )
            n1 = await _fetch(
                CandleInterval.CANDLE_INTERVAL_1_MIN,
                hours_1m,
                st.candles_1m,
                "1m",
            )
            now_mono = time.monotonic()
            if n5:
                st.last_candle_5m_update = now_mono
            if n1:
                st.last_candle_update = now_mono
        logger.info(
            f"  {figi}: prefetched {n1} × 1m + {n5} × 5m bars "
            f"(history: {hours_1m}h + {hours_5m}h)"
        )

    async def get_state(self, figi: str) -> InstrumentState | None:
        async with self._lock:
            return self._instruments.get(figi)

    async def all_states(self) -> list[InstrumentState]:
        async with self._lock:
            return list(self._instruments.values())

    async def start_streaming(self, *, book_depth: int = 10):
        """Spawn three background tasks: order book, trades, 1-min candles.
        Each runs its own stream connection with auto-reconnect."""
        self._book_depth = book_depth
        figis = [s.figi for s in self._instruments.values()]
        if not figis:
            raise RuntimeError("StreamManager: no instruments registered")
        logger.info(
            f"Starting 3 stream tasks for {len(figis)} instruments " f"(book_depth={book_depth})"
        )
        # SDK exposes only ONE_MINUTE and FIVE_MINUTES for streaming
        # subscriptions.  We subscribe to both: 1m drives the signal
        # indicators, 5m is the source for "15-min" ATR (aggregated 3:1).
        self._tasks = [
            asyncio.create_task(self._run_book_stream(figis), name="scalp-stream-book"),
            asyncio.create_task(self._run_trades_stream(figis), name="scalp-stream-trades"),
            asyncio.create_task(
                self._run_candles_stream(
                    figis, SubscriptionInterval.SUBSCRIPTION_INTERVAL_ONE_MINUTE, "1m"
                ),
                name="scalp-stream-c1m",
            ),
            asyncio.create_task(
                self._run_candles_stream(
                    figis, SubscriptionInterval.SUBSCRIPTION_INTERVAL_FIVE_MINUTES, "5m"
                ),
                name="scalp-stream-c5m",
            ),
        ]

    async def stop(self):
        self._stop = True
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ─── Stream consumers ─────────────────────────────────────────────────
    async def _reconnect_loop(self, name: str, run_once):
        """Generic reconnect-with-backoff wrapper used by all three streams."""
        backoff = 1.0
        while not self._stop:
            try:
                await run_once()
                # Graceful return = re-subscribe immediately
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"[{name}] stream error: {type(e).__name__}: {e} — "
                    f"reconnecting in {backoff:.0f}s"
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2, 60.0)

    async def _run_book_stream(self, figis: list[str]):
        async def once():
            async with AsyncClient(self.token, app_name=self.app_name) as client:

                async def gen():
                    yield MarketDataRequest(
                        subscribe_order_book_request=SubscribeOrderBookRequest(
                            subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                            instruments=[
                                OrderBookInstrument(figi=f, depth=self._book_depth) for f in figis
                            ],
                        )
                    )
                    while not self._stop:
                        await asyncio.sleep(60)

                stream = client.market_data_stream.market_data_stream(gen())
                async for msg in stream:
                    if self._stop:
                        return
                    ob = getattr(msg, "orderbook", None)
                    if ob is None:
                        continue
                    await self._on_orderbook(ob)

        await self._reconnect_loop("book", once)

    async def _run_trades_stream(self, figis: list[str]):
        async def once():
            async with AsyncClient(self.token, app_name=self.app_name) as client:

                async def gen():
                    yield MarketDataRequest(
                        subscribe_trades_request=SubscribeTradesRequest(
                            subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                            instruments=[TradeInstrument(figi=f) for f in figis],
                        )
                    )
                    while not self._stop:
                        await asyncio.sleep(60)

                stream = client.market_data_stream.market_data_stream(gen())
                async for msg in stream:
                    if self._stop:
                        return
                    t = getattr(msg, "trade", None)
                    if t is None:
                        continue
                    await self._on_trade(t)

        await self._reconnect_loop("trades", once)

    async def _run_candles_stream(
        self, figis: list[str], interval: SubscriptionInterval, label: str
    ):
        """Spawn a candle stream for ONE timeframe.  Multiple TFs run in
        parallel as separate tasks; events route to the right deque via
        `interval` matching in _on_candle."""

        async def once():
            async with AsyncClient(self.token, app_name=self.app_name) as client:

                async def gen():
                    yield MarketDataRequest(
                        subscribe_candles_request=SubscribeCandlesRequest(
                            subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                            instruments=[
                                CandleInstrument(figi=f, interval=interval) for f in figis
                            ],
                        )
                    )
                    while not self._stop:
                        await asyncio.sleep(60)

                stream = client.market_data_stream.market_data_stream(gen())
                async for msg in stream:
                    if self._stop:
                        return
                    c = getattr(msg, "candle", None)
                    if c is None:
                        continue
                    await self._on_candle(c, interval, label)

        await self._reconnect_loop(f"candles-{label}", once)

    # ─── Event handlers (update state) ─────────────────────────────────
    async def _on_orderbook(self, ob):
        figi = ob.figi
        async with self._lock:
            st = self._instruments.get(figi)
            if st is None:
                return
            st.bids = [
                BookLevel(float(quotation_to_decimal(b.price)), int(b.quantity))
                for b in (ob.bids or [])
            ]
            st.asks = [
                BookLevel(float(quotation_to_decimal(a.price)), int(a.quantity))
                for a in (ob.asks or [])
            ]
            st.last_book_update = time.monotonic()

    async def _on_trade(self, t):
        figi = t.figi
        async with self._lock:
            st = self._instruments.get(figi)
            if st is None:
                return
            price = float(quotation_to_decimal(t.price))
            qty = int(t.quantity)
            direction = +1 if t.direction == TradeDirection.TRADE_DIRECTION_BUY else -1
            ts = t.time.timestamp() if hasattr(t.time, "timestamp") else time.time()
            st.recent_trades.append(
                {
                    "ts": ts,
                    "price": price,
                    "qty": qty,
                    "dir": direction,
                }
            )
            st.last_price = price
            st.last_trade_update = time.monotonic()

    async def _on_candle(self, c, interval: SubscriptionInterval, label: str):
        figi = c.figi
        async with self._lock:
            st = self._instruments.get(figi)
            if st is None:
                return
            bar = {
                "time": c.time,
                "open": float(quotation_to_decimal(c.open)),
                "high": float(quotation_to_decimal(c.high)),
                "low": float(quotation_to_decimal(c.low)),
                "close": float(quotation_to_decimal(c.close)),
                "volume": int(c.volume),
            }
            # Pick the right deque for this TF.  Streamed candles repeat
            # the same bar multiple times as it forms — we replace the
            # last entry if same timestamp, else append a new bar.
            if label == "1m":
                buf = st.candles_1m
                if buf and buf[-1]["time"] == bar["time"]:
                    buf[-1] = bar
                else:
                    buf.append(bar)
                st.last_candle_update = time.monotonic()
            elif label == "5m":
                buf = st.candles_5m
                if buf and buf[-1]["time"] == bar["time"]:
                    buf[-1] = bar
                else:
                    buf.append(bar)
                st.last_candle_5m_update = time.monotonic()
