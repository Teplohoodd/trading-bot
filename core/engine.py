"""TradingEngine: main orchestrator for autonomous and interactive trading."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from t_tech.invest import CandleInterval, OrderDirection, StopOrderDirection
from t_tech.invest.utils import quotation_to_decimal

from core.broker import BrokerClient
from risk.manager import RiskManager
from analysis.screener import Screener, _candles_to_df
from database.db import Repository, Trade
from config.settings import Settings
from telegram_bot.notifications import Notification, NotificationType

logger = logging.getLogger(__name__)


class TradingEngine:
    """Main trading orchestrator. Runs multiple async loops."""

    def __init__(self, broker: BrokerClient, risk_manager: RiskManager,
                 strategies: list, screener: Screener,
                 db: Repository, notification_queue: asyncio.Queue,
                 settings: Settings):
        self.broker = broker
        self.risk_manager = risk_manager
        self.strategies = {s.name: s for s in strategies}
        self.screener = screener
        self.db = db
        self.notification_queue = notification_queue
        self.settings = settings

        self._mode: str = settings.MODE
        self._is_running: bool = False
        self._watchlist: list[dict] = []
        self._regime_detector = None
        self._trainer = None
        # Advisory mode: pending signals awaiting user approval  {signal_id: {...}}
        self._pending_signals: dict = {}
        # Anti-whipsaw: last close time per FIGI.  Used to enforce
        # settings.SAME_TICKER_COOLDOWN_MINUTES so we don't re-enter a
        # position right after closing one on commission-eating noise.
        self._recent_closes: dict[str, datetime] = {}
        # Trailing-stop state: per trade_id → {"peak_price", "activated"}.
        # In-memory only — on restart it resets and trailing re-starts
        # from current price as "peak".  No harm in that (just misses a
        # bit of potential lock-in).
        self._position_peaks: dict[int, dict] = {}
        # Serialises the entry critical section "check open_figis → place
        # order → record trade" across all entry paths (autonomous scan,
        # advisory approval callback, manual /buy).  Without it two paths
        # can both observe "no open NVTK", both call post_limit_with_fallback,
        # and we end up holding 2× the intended size (observed live:
        # NVTK 14:02 and 14:05 on the same day).
        self._entry_lock: asyncio.Lock = asyncio.Lock()

    def set_trainer(self, trainer):
        self._trainer = trainer

    def set_regime_detector(self, detector):
        self._regime_detector = detector

    # ==================== Main Loop ====================

    async def run(self):
        """Start all engine loops concurrently."""
        self._is_running = True
        logger.info(f"Trading engine started in {self._mode} mode")

        # Hydrate persistent engine state from DB before starting any loop.
        # Both _position_peaks and _recent_closes were previously in-memory
        # only — postmortem 2026-04-30: a bot restart after lunch wiped the
        # trailing-stop peaks (so trail re-anchored at the post-restart
        # current price) and dropped cooldowns (so a freshly-stopped name
        # could be re-entered seconds later).  Hydrating from DB makes
        # restart safe.
        try:
            peaks = await self.db.get_position_peaks()
            self._position_peaks = {
                tid: dict(info) for tid, info in peaks.items()
            }
            logger.info(f"Hydrated {len(self._position_peaks)} position peaks from DB")
        except Exception as e:
            logger.warning(f"Could not hydrate position_peaks: {e}")
        try:
            cd = await self.db.get_cooldowns()
            for figi, ts in cd.items():
                try:
                    self._recent_closes[figi] = datetime.fromisoformat(
                        ts.replace("Z", "+00:00")
                    )
                except Exception:
                    pass
            logger.info(f"Hydrated {len(self._recent_closes)} cooldowns from DB")
        except Exception as e:
            logger.warning(f"Could not hydrate cooldowns: {e}")

        # Seed the watchlist once on startup regardless of mode.  Gives the
        # user immediate feedback that the screener is alive, and makes
        # /watchlist respond instantly (no wait for the first scan tick).
        # In interactive mode the autonomous loop never runs the screener,
        # so without this seeding /watchlist would sit empty until the user
        # switched to autonomous.
        asyncio.create_task(self._initial_watchlist_scan())

        await asyncio.gather(
            self._autonomous_loop(),
            self._position_monitor_loop(),
            self._portfolio_sync_loop(),
            self._retrain_scheduler(),
            return_exceptions=True,
        )

    async def _initial_watchlist_scan(self):
        """Fire-and-forget initial screener pass on startup.

        Runs once at boot in every mode (autonomous / advisory / interactive)
        REGARDLESS of market hours — weekend/holiday scans are still useful
        for Mon-open prep + weekend-trading-eligible shares.  Trade approval
        still enforces market hours per-instrument (regular vs weekend
        session), so no stray orders can leak out.
        """
        try:
            from utils.helpers import is_moex_open, is_weekend_session
            market_state = (
                "open" if is_moex_open()
                else "weekend-session" if is_weekend_session()
                else "closed"
            )
            custom = await self.db.get_custom_tickers()
            custom_figis = [c["figi"] for c in custom if c.get("figi")]
            self._watchlist = await self._build_watchlist(custom_figis)
            n = len(self._watchlist)
            long_n = sum(1 for c in self._watchlist if c.get("direction") == "long")
            short_n = sum(1 for c in self._watchlist if c.get("direction") == "short")
            logger.info(
                f"Startup scan: {n} candidates ({long_n} long + {short_n} short), "
                f"market={market_state}"
            )
            top = ", ".join(c.get("ticker", "?") for c in self._watchlist[:5])
            await self._notify(Notification(
                type=NotificationType.INFO,
                title=f"Screener ready: {n} tickers",
                data={
                    "top5": top,
                    "mode": self._mode,
                    "market": market_state,
                    "total": n,
                },
            ))
        except Exception as e:
            logger.error(f"Startup watchlist scan failed: {e}", exc_info=True)

    async def _autonomous_loop(self):
        """Periodic scan + signal + trade loop (autonomous and advisory modes)."""
        while self._is_running:
            try:
                if self._mode in ("autonomous", "advisory"):
                    await self._run_scan_cycle()
            except Exception as e:
                logger.error(f"Autonomous loop error: {e}", exc_info=True)
                await self._notify(Notification(
                    type=NotificationType.ERROR,
                    title=f"Autonomous loop error: {e}"
                ))

            interval = self.settings.SCAN_INTERVAL_MINUTES * 60
            await asyncio.sleep(interval)

    async def _run_scan_cycle(self):
        """One full scan: screen -> signal -> risk check -> execute."""
        from utils.helpers import is_moex_open, now_msk
        if not is_moex_open():
            logger.debug("Market closed, skipping scan")
            return

        logger.info("Running scan cycle...")

        # --- Cycle-level entry filters (hoisted out of the per-ticker loop) ---
        # Evaluating these per candidate logged 30 identical lines per cycle
        # AND made root-cause analysis painful when no signals fired (silent
        # debug skips).  Now we log ONCE per cycle at INFO level and short-
        # circuit the whole scan when the cycle is blocked.
        skip_days = getattr(self.settings, "SKIP_ENTRY_WEEKDAYS", [])
        if skip_days:
            today_dow = now_msk().weekday()  # 0=Mon, 4=Fri
            if today_dow in skip_days:
                logger.info(
                    f"Scan cycle skipped: weekday filter "
                    f"(dow={today_dow}, skip_days={skip_days})"
                )
                return
        skip_hours = getattr(self.settings, "SKIP_ENTRY_HOURS_MSK", [])
        if skip_hours:
            current_hour_msk = now_msk().hour
            if current_hour_msk in skip_hours:
                logger.info(
                    f"Scan cycle skipped: hour filter "
                    f"(hour_msk={current_hour_msk}, skip_hours={skip_hours})"
                )
                return

        # Rebuild watchlist every scan cycle.  Earlier code cached it forever
        # (`if not self._watchlist:`) which meant: (a) momentum leaders from
        # hours-old scan stayed, (b) enabling shorts mid-run did nothing
        # because the cached list had no short candidates.  Fresh scan per
        # cycle costs ~5-10s (fits in 30min SCAN_INTERVAL budget easily).
        custom = await self.db.get_custom_tickers()
        custom_figis = [c["figi"] for c in custom if c.get("figi")]
        self._watchlist = await self._build_watchlist(custom_figis)

        # Get current open positions to avoid duplicate entries
        open_trades = await self.db.get_open_trades()
        open_figis = {t["figi"] for t in open_trades}

        # Per-cycle skip counters for the post-loop summary log.  Without this,
        # zero-signal cycles produced no diagnostic trail and the bot's silence
        # was indistinguishable between "no edge today" vs "filter bug".
        skip_counts = {
            "open_position": 0, "cooldown": 0, "hold": 0, "low_vol": 0,
            "dir_mismatch": 0, "short_low_conf": 0, "short_cutoff": 0,
            "short_gap_open": 0, "rejected_risk": 0, "executed": 0,
            "max_positions": 0, "low_short_carry": 0, "errors": 0,
        }

        for candidate in self._watchlist:
            figi = candidate["figi"]
            ticker = candidate["ticker"]
            preferred_direction = candidate.get("direction", "long")  # "long" or "short"

            if figi in open_figis:
                skip_counts["open_position"] += 1
                continue
            if self._is_in_cooldown(figi):
                last = self._recent_closes.get(figi)
                remaining = (
                    self.settings.SAME_TICKER_COOLDOWN_MINUTES
                    - int((datetime.now(timezone.utc) - last).total_seconds() // 60)
                ) if last else 0
                logger.debug(
                    f"{ticker}: in cooldown after recent close "
                    f"(~{remaining}m remaining), skipping"
                )
                skip_counts["cooldown"] += 1
                continue
            if len(open_trades) >= self.settings.MAX_POSITIONS:
                skip_counts["max_positions"] += 1
                break

            # Gap-open protection: don't open SHORT entries in the first 30 min
            # of the MOEX session.  At open, market makers widen spreads and
            # prices can gap through stop levels, making SL orders physically
            # unable to protect the position.
            # Real example (22-Apr-2026): SMLT short entered at 07:58 MSK,
            # price gapped from ~617 through SL=626.2 to 639 — lost 22 RUB/lot
            # with no SL execution possible.
            # Futures are excluded: different open mechanics, no overnight carry.
            if preferred_direction == "short" and candidate.get("kind", "share") != "future":
                _t = now_msk()
                _minutes_since_open = (_t.hour - 7) * 60 + _t.minute
                if 0 <= _minutes_since_open < 30:
                    logger.info(
                        f"{ticker}: skipping short — gap-open window "
                        f"({_minutes_since_open}m into session, shorts blocked until 07:30 MSK)"
                    )
                    skip_counts["short_gap_open"] += 1
                    continue

            try:
                candidate_kind = candidate.get("kind", "share")
                signal = await self._evaluate_instrument(figi, ticker,
                                                         instrument_kind=candidate_kind)
                if signal.direction == "hold":
                    skip_counts["hold"] += 1
                    continue

                # Propagate instrument kind (share/future) from screener candidate
                # into the signal so downstream gates (short cutoff, carry cap)
                # can behave asymmetrically.
                signal.instrument_kind = candidate_kind

                # ATR volatility filter (#1 feature by permutation importance,
                # 0.158 — 3× more important than #2).  In low-vol regimes the
                # model's directional signal is noise.  Skip entries when
                # current ATR% is below ATR_FILTER_MEDIAN_MULT × median over
                # the last 60 bars.  Candidate ATR% is pre-scored by screener.
                atr_filter_mult = getattr(
                    self.settings, "ATR_FILTER_MEDIAN_MULT", 0.8
                )
                if atr_filter_mult > 0:
                    cand_atr_pct = candidate.get("atr_pct", 0.0)
                    cand_atr_median = candidate.get("atr_pct_median60", 0.0)
                    if cand_atr_median > 0 and cand_atr_pct < atr_filter_mult * cand_atr_median:
                        logger.debug(
                            f"{ticker}: low-vol filter — "
                            f"atr_pct={cand_atr_pct:.3f} < "
                            f"{atr_filter_mult}×median({cand_atr_median:.3f}), skipping"
                        )
                        skip_counts["low_vol"] += 1
                        continue

                # Direction-gating: strategy signal must match candidate thesis.
                # A "long" candidate (picked for +momentum) shouldn't be shorted
                # just because strategies briefly flipped bearish — screener's
                # direction filters the universe for each thesis separately.
                expected_signal = "buy" if preferred_direction == "long" else "sell"
                if signal.direction != expected_signal:
                    logger.debug(
                        f"{ticker}: signal={signal.direction} does not match "
                        f"candidate thesis ({preferred_direction}→{expected_signal}), skipping"
                    )
                    skip_counts["dir_mismatch"] += 1
                    continue

                # Short-specific tighter confidence gate (shorts have more
                # tail risk than longs — forced closes, borrow recall, etc.)
                if signal.direction == "sell" and signal.confidence < self.settings.SHORT_MIN_CONFIDENCE:
                    logger.info(
                        f"{ticker}: short signal below SHORT_MIN_CONFIDENCE "
                        f"({signal.confidence:.2f} < {self.settings.SHORT_MIN_CONFIDENCE}), skipping"
                    )
                    skip_counts["short_low_conf"] += 1
                    continue

                # Short entry cutoff: don't open new shorts late in the
                # session.  Overnight carry hits at end-of-day clearing, and
                # a short opened at 22:30 pays the full day's carry on a
                # position held only a few minutes.  Existing shorts are
                # unaffected — this only blocks NEW short entries.
                # Futures have NO overnight short carry (they're symmetric),
                # so skip the cutoff for them.
                signal_kind = getattr(signal, "instrument_kind", None) or "share"
                if signal.direction == "sell" and signal_kind != "future":
                    msk_hour = now_msk().hour
                    cutoff = self.settings.SHORT_ENTRY_CUTOFF_HOUR_MSK
                    if msk_hour >= cutoff:
                        logger.info(
                            f"{ticker}: short entry cutoff ({msk_hour}:XX MSK "
                            f">= {cutoff}:00), skipping"
                        )
                        skip_counts["short_cutoff"] += 1
                        continue

                # Log signal — capture row id so we can update approved/reason
                # post-hoc once the risk manager has decided.  Without this
                # update path, every signal stayed approved=0 in DB even when
                # actually executed (data-quality bug observed pre-2026-04-27).
                signal_row_id = await self.db.insert_signal(
                    figi=figi, ticker=ticker, direction=signal.direction,
                    confidence=signal.confidence, strategy=signal.strategy_name,
                    features=signal.features, approved=False, rejection_reason=""
                )

                # Risk check
                approval = await self.risk_manager.approve_trade(
                    figi=figi, ticker=ticker, direction=signal.direction,
                    confidence=signal.confidence,
                    suggested_stop_pct=signal.suggested_stop_pct,
                    suggested_target_pct=signal.suggested_target_pct,
                )

                if not approval.approved:
                    logger.info(f"Trade rejected for {ticker}: {approval.reason}")
                    # Persist the rejection_reason on the signal row so the
                    # forensic / postmortem queries have a non-empty trail.
                    try:
                        await self.db.update_signal_approval(
                            signal_row_id, approved=False,
                            rejection_reason=str(approval.reason)[:200],
                        )
                    except Exception as e:
                        logger.debug(f"Could not update signal approval: {e}")
                    skip_counts["rejected_risk"] += 1
                    continue

                # Short-specific position scaling.  Shorts sized at
                # SHORT_POSITION_SCALE × Kelly-recommended to keep tail-risk
                # exposure under control even when signal confidence is high.
                if signal.direction == "sell":
                    scaled_lots = max(1, int(approval.lots * self.settings.SHORT_POSITION_SCALE))
                    if scaled_lots != approval.lots:
                        logger.info(
                            f"{ticker}: scaling short size {approval.lots} → {scaled_lots} "
                            f"(×{self.settings.SHORT_POSITION_SCALE})"
                        )
                        approval.lots = scaled_lots

                    # Carry-free tier cap: keep short notional under Tinkoff's
                    # overnight carry threshold so we pay 0 carry.  Set
                    # SHORT_CARRY_FREE_THRESHOLD_RUB = 0 to disable.
                    # Futures don't pay overnight short carry — skip the cap.
                    approval_kind = approval.risk_metrics.get("instrument_kind", "share") \
                        if approval.risk_metrics else "share"
                    threshold = self.settings.SHORT_CARRY_FREE_THRESHOLD_RUB
                    if threshold > 0 and approval_kind != "future":
                        price_f = approval.risk_metrics.get("price", 0.0)
                        lot_size = approval.risk_metrics.get("lot_size", 1)
                        lot_value = price_f * lot_size
                        if lot_value > 0:
                            max_lots_carry = int(threshold // lot_value)
                            if max_lots_carry < 1:
                                # A single lot already exceeds the tier — we'd
                                # be paying carry regardless.  Skip to avoid
                                # surprise overnight cost.
                                logger.info(
                                    f"{ticker}: 1 lot notional "
                                    f"({lot_value:.0f} RUB) > carry-free "
                                    f"threshold ({threshold:.0f}), skipping short"
                                )
                                skip_counts["low_short_carry"] += 1
                                continue
                            if max_lots_carry < approval.lots:
                                logger.info(
                                    f"{ticker}: capping short {approval.lots} "
                                    f"→ {max_lots_carry} lots to stay under "
                                    f"{threshold:.0f} RUB carry-free tier "
                                    f"(lot notional={lot_value:.0f})"
                                )
                                approval.lots = max_lots_carry

                if self._mode == "advisory":
                    # Advisory: send notification and wait for user approval
                    await self._queue_advisory_signal(signal, approval)
                    continue

                # Hard MAX_POSITIONS re-check from DB immediately before execute.
                # The outer `open_trades` variable can be stale when an earlier
                # iteration threw an exception and `continue`d without refreshing
                # (observed: 12 positions with MAX_POSITIONS=5, 22-Apr-2026).
                _fresh_open = await self.db.get_open_trades()
                if len(_fresh_open) >= self.settings.MAX_POSITIONS:
                    logger.info(
                        f"{ticker}: aborting — MAX_POSITIONS reached at pre-trade "
                        f"check ({len(_fresh_open)}/{self.settings.MAX_POSITIONS})"
                    )
                    # Also update outer cache so sibling loop iterations see it.
                    open_trades = _fresh_open
                    open_figis = {t["figi"] for t in open_trades}
                    break

                # Autonomous: Execute immediately
                result = await self._execute_trade(figi, ticker, signal.direction,
                                                    approval.lots, signal.strategy_name,
                                                    signal.confidence, approval)
                if result.get("success"):
                    skip_counts["executed"] += 1
                    # Mark the signal row as approved+executed so DB reflects
                    # ground truth (was always approved=0 even for executed
                    # trades — broke confidence-quartile postmortems).
                    try:
                        await self.db.update_signal_approval(
                            signal_row_id, approved=True, rejection_reason=""
                        )
                    except Exception as e:
                        logger.debug(f"Could not update signal approval: {e}")
                    open_trades = await self.db.get_open_trades()
                    open_figis = {t["figi"] for t in open_trades}

            except Exception as e:
                logger.error(f"Error evaluating {ticker}: {e}", exc_info=True)
                skip_counts["errors"] += 1
                # Refresh open_trades after any error so the next iteration's
                # MAX_POSITIONS guard uses a fresh count (not the stale cache
                # that caused the 12-position overflow on 22-Apr-2026).
                try:
                    open_trades = await self.db.get_open_trades()
                    open_figis = {t["figi"] for t in open_trades}
                except Exception:
                    pass  # keep stale cache if DB itself is unavailable
                continue

        # --- Cycle summary ---
        # Compact one-liner so a `tail bot.log | grep "Scan summary"` answers
        # "why didn't the bot trade?" without code-diving into per-ticker
        # debug lines.  Added 2026-04-27 after a 3-day silent gap caused by
        # over-aggressive weekday + hour filters going un-logged.
        nz = {k: v for k, v in skip_counts.items() if v > 0}
        logger.info(
            f"Scan summary: candidates={len(self._watchlist)} {nz}"
        )

    async def _evaluate_instrument(self, figi: str, ticker: str,
                                    instrument_kind: str = "share"):
        """Run strategy pipeline for an instrument, return aggregated signal."""
        from strategy.base import Signal
        now = datetime.now(timezone.utc)
        from_dt = now - timedelta(days=30)

        candles = await self.broker.get_candles(
            figi, from_dt, now, CandleInterval.CANDLE_INTERVAL_HOUR
        )
        if len(candles) < 50:
            return Signal(figi=figi, ticker=ticker, direction="hold",
                         confidence=0.0, strategy_name="none", timestamp=now)

        import pandas as pd
        df = _candles_to_df(candles)

        # Get order book.  We always populate a context dict so strategies
        # can read instrument_kind even when the order book itself fails.
        order_book = {"instrument_kind": instrument_kind}
        # For futures, surface expiration_date so build_features can compute
        # days_to_expiry / in_roll_window.  Silently skipped for shares.
        if instrument_kind == "future":
            try:
                instr, _k = await self.broker.get_instrument_info(figi)
                meta = self.broker.extract_futures_metadata(instr)
                order_book["expiration_date"] = meta.get("expiration_date")
            except Exception as e:
                logger.debug(f"{ticker}: could not fetch futures metadata for features: {e}")
        try:
            ob = await self.broker.get_order_book(figi, depth=10)
            if ob.bids and ob.asks:
                bid = float(quotation_to_decimal(ob.bids[0].price))
                ask = float(quotation_to_decimal(ob.asks[0].price))
                mid = (bid + ask) / 2
                total_bid = sum(b.quantity for b in ob.bids)
                total_ask = sum(a.quantity for a in ob.asks)
                total = total_bid + total_ask
                order_book.update({
                    "spread_bps": (ask - bid) / mid * 10000 if mid > 0 else 0,
                    "imbalance": (total_bid - total_ask) / total if total > 0 else 0,
                })
        except Exception:
            pass

        # Detect regime
        regime_weights = {"ml_lightgbm": 0.6, "technical": 0.4}
        if self._regime_detector:
            from strategy.regime import RegimeDetector
            regime = self._regime_detector.detect(df)
            regime_weights = self._regime_detector.get_strategy_weights(regime)
            regime_scale = self._regime_detector.get_position_scale(regime)
        else:
            regime_scale = 1.0

        # Collect signals from all strategies
        signals = []
        for name, strategy in self.strategies.items():
            try:
                sig = await strategy.generate_signal(figi, ticker, df, order_book)
                weight = regime_weights.get(name, 0.5)
                signals.append((sig, weight))
            except Exception as e:
                logger.debug(f"Strategy {name} error for {ticker}: {e}")

        if not signals:
            return Signal(figi=figi, ticker=ticker, direction="hold",
                         confidence=0.0, strategy_name="none", timestamp=now)

        # Weighted voting
        buy_score = sum(w for s, w in signals if s.direction == "buy") * \
                    (sum(s.confidence * w for s, w in signals if s.direction == "buy") /
                     max(sum(w for s, w in signals if s.direction == "buy"), 1e-9))
        sell_score = sum(w for s, w in signals if s.direction == "sell") * \
                     (sum(s.confidence * w for s, w in signals if s.direction == "sell") /
                      max(sum(w for s, w in signals if s.direction == "sell"), 1e-9))

        # Only count strategies that took a directional stance in the denominator.
        # Abstaining strategies (hold + ~zero confidence) should not dilute
        # the signal from strategies that do have a clear directional view.
        active_weight = sum(w for s, w in signals if s.direction in ("buy", "sell"))
        total_weight = active_weight if active_weight > 1e-9 else sum(w for _, w in signals)
        buy_score /= max(total_weight, 1e-9)
        sell_score /= max(total_weight, 1e-9)

        # Futures get a separate (lower) threshold because the ML model has no
        # futures training data yet — TechnicalStrategy drives futures signals alone.
        # SIGNAL_THRESHOLD_FUTURES is applied here so instrument_kind is respected
        # at the combined-voting level, not just inside MLStrategy.
        entry_threshold = (
            getattr(self.settings, "SIGNAL_THRESHOLD_FUTURES", self.settings.SIGNAL_THRESHOLD)
            if instrument_kind == "future"
            else self.settings.SIGNAL_THRESHOLD
        )
        if buy_score > sell_score and buy_score > entry_threshold:
            direction = "buy"
            confidence = buy_score
        elif sell_score > buy_score and sell_score > entry_threshold:
            direction = "sell"
            confidence = sell_score
        else:
            direction = "hold"
            confidence = max(buy_score, sell_score)

        # Use best signal's stop/target suggestions
        best_signal = max(signals, key=lambda x: x[0].confidence * x[1])[0]

        from strategy.base import Signal
        return Signal(
            figi=figi, ticker=ticker, direction=direction,
            confidence=round(confidence, 3),
            strategy_name="+".join(self.strategies.keys()),
            timestamp=now,
            suggested_stop_pct=best_signal.suggested_stop_pct,
            suggested_target_pct=best_signal.suggested_target_pct,
            features=best_signal.features,
        )

    async def _execute_trade(self, figi: str, ticker: str, direction: str,
                              lots: int, strategy: str, confidence: float,
                              approval) -> dict:
        """Place order with smart execution and record in database.

        Uses limit orders (aggressive/passive) or market depending on
        settings.ORDER_EXECUTION_MODE. Falls back to market if limit
        doesn't fill within LIMIT_ORDER_TIMEOUT.
        """
        from t_tech.invest import OrderDirection, StopOrderDirection
        order_dir = (OrderDirection.ORDER_DIRECTION_BUY
                     if direction == "buy"
                     else OrderDirection.ORDER_DIRECTION_SELL)

        # Serialise entries across all paths (scan / advisory / manual) and
        # re-check the open-positions snapshot inside the lock.  The scan
        # loop already checks `open_figis` before calling us, but the snapshot
        # it holds is stale by the time we reach here — another path may
        # have opened the same figi in between.  This is the backstop.
        async with self._entry_lock:
            try:
                open_trades_snapshot = await self.db.get_open_trades()
                if any(t["figi"] == figi for t in open_trades_snapshot):
                    logger.warning(
                        f"{ticker}: entry aborted — figi already has an open "
                        f"trade (dedup guard in _execute_trade)"
                    )
                    return {"success": False, "reason": "Already open (dedup)"}
            except Exception as e:
                logger.debug(f"Dedup snapshot check skipped: {e}")

            return await self._execute_trade_locked(
                figi, ticker, direction, lots, strategy, confidence,
                approval, order_dir,
            )

    async def _execute_trade_locked(
        self, figi: str, ticker: str, direction: str, lots: int,
        strategy: str, confidence: float, approval, order_dir,
    ) -> dict:
        """Inner body of _execute_trade — runs inside self._entry_lock."""
        from t_tech.invest import StopOrderDirection

        try:
            exec_mode = self.settings.ORDER_EXECUTION_MODE
            timeout = self.settings.LIMIT_ORDER_TIMEOUT
            fallback = self.settings.LIMIT_ORDER_RETRY_MARKET

            if approval.use_twap:
                order_id, price, order_type, filled_lots = await self._execute_twap(
                    figi, lots, order_dir, exec_mode, timeout
                )
            elif exec_mode == "market":
                resp = await self.broker.post_market_order(figi, lots, order_dir)
                order_id = resp.order_id
                price = float(await self.broker.get_last_price(figi))
                order_type = "market"
                # Market orders usually fully fill on liquid names, but trust
                # broker response when it reports a partial (halted / circuit-breaker)
                filled_lots = getattr(resp, "lots_executed", lots) or lots
            else:
                resp, price, order_type, filled_lots = await self.broker.post_limit_with_fallback(
                    figi=figi,
                    lots=lots,
                    direction=order_dir,
                    mode=exec_mode,
                    timeout=float(timeout),
                    fallback_market=fallback,
                )
                order_id = resp.order_id if hasattr(resp, "order_id") else "limit"

            # Abort if nothing actually filled — no trade to record, no
            # stops to place (posting a stop on zero lots is invalid).
            if filled_lots <= 0:
                logger.error(
                    f"{ticker}: order executed but 0 lots filled "
                    f"(requested {lots}, type {order_type}) — skipping stop/TP and trade record"
                )
                return {"success": False, "reason": "No lots filled"}

            # Partial-fill warning: stops and DB record MUST track the real
            # position size.  Prior bug: used requested `lots` for
            # post_stop_loss / post_take_profit, leaving orphan uncovered
            # shares when market fallback fell short or TWAP slices failed.
            if filled_lots < lots:
                logger.warning(
                    f"{ticker}: partial fill {filled_lots}/{lots} lots "
                    f"(type={order_type}) — stops/record will reflect actual fill"
                )

            stop = approval.stop_loss_price
            target = approval.take_profit_price

            # Recalculate stops from actual fill price (approval computed from pre-trade price)
            atr_pct = approval.risk_metrics.get("atr_pct", None)
            if atr_pct and price > 0:
                stop_dist = approval.risk_metrics.get("stop_distance",
                                                       abs(stop - float(approval.risk_metrics.get("price", price))))
                target_dist = approval.risk_metrics.get("target_distance", stop_dist * 1.5)
                if direction == "buy":
                    stop = price - stop_dist
                    target = price + target_dist
                else:
                    stop = price + stop_dist
                    target = price - target_dist

            # Get instrument lot size + kind (share/future)
            lot_size = 1
            instrument_kind = approval.risk_metrics.get("instrument_kind", "share") \
                if hasattr(approval, "risk_metrics") and approval.risk_metrics else "share"
            try:
                instrument, kind = await self.broker.get_instrument_info(figi)
                if instrument is not None:
                    lot_size = getattr(instrument, "lot", 1) or 1
                if kind != "unknown":
                    instrument_kind = kind
            except Exception:
                pass

            # Place stop-loss and take-profit
            stop_dir = (StopOrderDirection.STOP_ORDER_DIRECTION_SELL
                        if direction == "buy"
                        else StopOrderDirection.STOP_ORDER_DIRECTION_BUY)
            # Round to the instrument's min_price_increment so the broker accepts the stop prices
            try:
                stop_rounded = await self.broker._round_to_increment(figi, Decimal(str(stop)))
            except Exception:
                stop_rounded = Decimal(str(round(stop, 4)))
            try:
                target_rounded = await self.broker._round_to_increment(figi, Decimal(str(target)))
            except Exception:
                target_rounded = Decimal(str(round(target, 4)))
            # Place SL + TP as INDEPENDENT calls.  Previously both were in the
            # same try/except so a SL success followed by TP failure would
            # abort silently → observed live: TRMK, MTLR opened without TP
            # and rode the losing leg fully to SL.  We also retry once on
            # failure (price rounding / rate-limit transient) and notify the
            # user on final failure so they can intervene.
            sl_ok = False
            tp_ok = False
            for attempt in range(2):
                try:
                    await self.broker.post_stop_loss(figi, filled_lots, stop_rounded, stop_dir)
                    sl_ok = True
                    break
                except Exception as e:
                    logger.warning(f"{ticker}: stop-loss attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(1)
            for attempt in range(2):
                try:
                    await self.broker.post_take_profit(figi, filled_lots, target_rounded, stop_dir)
                    tp_ok = True
                    break
                except Exception as e:
                    logger.warning(f"{ticker}: take-profit attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(1)

            stop = float(stop_rounded)
            target = float(target_rounded)

            if not sl_ok or not tp_ok:
                # Loud notification — user needs to know a position is
                # running without full broker-side protection.  The engine's
                # position_monitor still enforces exits via trailing stop /
                # signal reversal, but a hung gRPC call could leave the
                # position un-stopped between monitor ticks.
                logger.error(
                    f"{ticker}: protection INCOMPLETE — "
                    f"SL={'OK' if sl_ok else 'MISSING'}, "
                    f"TP={'OK' if tp_ok else 'MISSING'} (filled {filled_lots} lots @ {price:.4f})"
                )
                try:
                    await self._notify(Notification(
                        type=NotificationType.RISK_ALERT,
                        title=f"⚠ {ticker}: stop orders incomplete",
                        data={
                            "ticker": ticker,
                            "lots": filled_lots,
                            "price": round(price, 4),
                            "stop_loss_placed": sl_ok,
                            "take_profit_placed": tp_ok,
                            "stop_price": round(float(stop_rounded), 4),
                            "target_price": round(float(target_rounded), 4),
                            "hint": (
                                "Engine-side monitor still applies trailing + reversal "
                                "exits every 5 min, but broker-side protection is "
                                "incomplete. Consider closing manually."
                            ),
                        },
                    ))
                except Exception:
                    pass

            # Record trade with ACTUAL filled lots (not requested).  Keeps
            # P&L / sizing / position-sync consistent with broker state.
            trade = Trade(
                figi=figi, ticker=ticker, direction=direction,
                lots=filled_lots, entry_price=price, strategy=strategy,
                signal_confidence=confidence,
                entry_time=datetime.now(timezone.utc).isoformat(),
                entry_order_id=order_id,
                stop_loss=round(stop, 4), take_profit=round(target, 4),
                lot_size=lot_size,
                instrument_kind=instrument_kind,
            )
            trade_id = await self.db.insert_trade(trade)

            trade_data = {
                "id": trade_id,
                "figi": figi, "ticker": ticker, "direction": direction,
                "lots": filled_lots, "entry_price": price,
                "stop_loss": round(stop, 4), "take_profit": round(target, 4),
                "strategy": strategy, "signal_confidence": confidence,
                "lot_size": lot_size,
                "instrument_kind": instrument_kind,
                "order_type": order_type,
            }

            await self._notify(Notification(
                type=NotificationType.TRADE_OPENED,
                title=f"Opened {direction.upper()} {ticker}",
                data=trade_data, priority="high"
            ))

            logger.info(
                f"Trade opened: {direction.upper()} {filled_lots} lots {ticker} "
                f"@ {price:.4f} [{order_type}]"
            )
            return {"success": True, **trade_data}

        except Exception as e:
            logger.error(f"Execute trade failed for {ticker}: {e}", exc_info=True)
            await self._notify(Notification(
                type=NotificationType.ERROR,
                title=f"Order failed: {ticker} - {e}"
            ))
            return {"success": False, "reason": str(e)}

    async def _execute_twap(self, figi: str, total_lots: int,
                             direction, exec_mode: str,
                             timeout: float) -> tuple[str, float, str, int]:
        """TWAP execution using limit or market slices.

        Returns (order_id, avg_price, order_type, total_filled_lots).
        total_filled_lots sums the actual fill of each slice — used by the
        caller to place correctly-sized stop/take-profit orders.
        """
        results = await self.risk_manager.execution.execute_twap(
            figi, total_lots, direction, exec_mode=exec_mode, timeout=timeout
        )
        order_id = results[0][0].order_id if results else "twap"
        # Price-weighted average over FILLED lots across slices
        total_filled = sum(r[3] for r in results) if results else 0
        if total_filled > 0:
            avg_price = sum(r[1] * r[3] for r in results) / total_filled
        else:
            avg_price = float(await self.broker.get_last_price(figi))
        return order_id, avg_price, "twap", total_filled

    # ==================== Advisory Mode ====================

    async def _queue_advisory_signal(self, signal, approval):
        """Store a pending signal and notify user to approve/reject."""
        import time as _time
        signal_id = f"{signal.ticker}_{signal.direction}_{int(_time.time())}"
        # Try to fetch lot_size so the notification can show shares/RUB volume
        lot_size = 1
        try:
            instrument, _kind = await self.broker.get_instrument_info(signal.figi)
            if instrument is not None:
                lot_size = getattr(instrument, "lot", 1) or 1
        except Exception:
            pass
        self._pending_signals[signal_id] = {
            "signal_id": signal_id,
            "figi": signal.figi,
            "ticker": signal.ticker,
            "direction": signal.direction,
            "confidence": signal.confidence,
            "strategy": signal.strategy_name,
            "lots": approval.lots,
            "lot_size": lot_size,
            "stop_loss": approval.stop_loss_price,
            "take_profit": approval.take_profit_price,
            "risk_metrics": approval.risk_metrics,
        }
        await self._notify(Notification(
            type=NotificationType.ADVISORY_SIGNAL,
            title=f"Signal: {signal.direction.upper()} {signal.ticker}",
            data=self._pending_signals[signal_id],
            priority="high",
        ))
        logger.info(f"Advisory signal queued: {signal_id}")

    async def execute_advisory_signal(self, signal_id: str) -> dict:
        """Execute a previously queued advisory signal."""
        pending = self._pending_signals.pop(signal_id, None)
        if not pending:
            return {"success": False, "reason": "Signal expired or not found"}

        from risk.manager import TradeApproval
        approval = TradeApproval(
            approved=True,
            lots=pending["lots"],
            stop_loss_price=pending["stop_loss"],
            take_profit_price=pending["take_profit"],
            use_twap=False,
            reason="User approved",
            risk_metrics=pending["risk_metrics"],
        )
        return await self._execute_trade(
            pending["figi"], pending["ticker"], pending["direction"],
            pending["lots"], pending["strategy"], pending["confidence"], approval,
        )

    def get_pending_signals(self) -> list[dict]:
        """Return list of pending advisory signals."""
        return list(self._pending_signals.values())

    # ==================== Position Monitor ====================

    async def _position_monitor_loop(self):
        """Check open positions for exit conditions."""
        while self._is_running:
            try:
                await self._check_exit_conditions()
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
            await asyncio.sleep(self.settings.PORTFOLIO_CHECK_MINUTES * 60)

    async def _check_exit_conditions(self):
        """Evaluate all open positions for exit signals."""
        open_trades = await self.db.get_open_trades()
        if not open_trades:
            return

        # GC trailing-stop state for trades that no longer exist
        live_ids = {t["id"] for t in open_trades}
        self._position_peaks = {
            tid: info for tid, info in self._position_peaks.items()
            if tid in live_ids
        }

        now = datetime.now(timezone.utc)
        for trade in open_trades:
            trade_id = trade["id"]
            figi = trade["figi"]
            ticker = trade["ticker"]
            direction = trade["direction"]
            entry_price = trade["entry_price"]
            entry_time_str = trade["entry_time"]

            try:
                # Fetch recent candles
                candles = await self.broker.get_candles(
                    figi,
                    now - timedelta(days=7),
                    now,
                    CandleInterval.CANDLE_INTERVAL_HOUR,
                )
                if len(candles) < 10:
                    continue

                df = _candles_to_df(candles)
                # Real-time price for SL/TP gating.  candles[-1].close is the
                # last COMPLETED hour's close — using it as "current price"
                # means the hard-SL check can lag up to 60 min while a 5-min
                # bar pierces the level.  postmortem 2026-04-30: 8/57 trades
                # had stop touched intraday but exit_reason ≠ stop_loss
                # because of this stale-price bug.  Fall back to candle close
                # only if the live tick fetch fails.
                try:
                    current_price = float(await self.broker.get_last_price(figi))
                    if current_price <= 0:
                        current_price = float(quotation_to_decimal(candles[-1].close))
                except Exception:
                    current_price = float(quotation_to_decimal(candles[-1].close))

                # Current unrealised P&L (gross, before commission)
                if direction == "buy":
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - current_price) / entry_price * 100

                # Round-trip commission as % — everything below this is noise
                # because even a "profitable" exit at that level is net-zero.
                # Use the trade's actual instrument_kind (shares vs futures have
                # different commission tiers).
                trade_kind = trade.get("instrument_kind") or "share"
                round_trip_comm_pct = self._commission_pct(trade_kind) * 2 * 100
                # Minimum |P&L%| required before signal_reversal may fire
                min_reversal_pnl_pct = (
                    round_trip_comm_pct * self.settings.MIN_EXIT_PROFIT_MULT_COMMISSION
                )

                # Progress toward take-profit target (0 .. 1+).  Used by both
                # the reversal gate and the trailing-stop activation.
                tp = trade.get("take_profit") or 0
                target_dist = abs(tp - entry_price) if tp else 0
                if target_dist > 0:
                    if direction == "buy":
                        progress = (current_price - entry_price) / target_dist
                    else:
                        progress = (entry_price - current_price) / target_dist
                else:
                    progress = 0.0  # no target info → disables progress gate

                # ---- Trailing stop (engine-side) -----------------------------
                # Activates once we've covered ACTIVATION_FRAC of the distance
                # to target, then closes the position if price retraces by
                # TRAIL_ATR_MULT × ATR from the best price seen.
                if self.settings.TRAILING_STOP_ENABLED:
                    peak_info = self._position_peaks.setdefault(trade_id, {
                        "peak_price": entry_price,
                        "activated": False,
                        "partial_taken": False,
                        "partial_price": None,
                    })
                    prev_peak = peak_info["peak_price"]
                    prev_activated = peak_info["activated"]
                    if direction == "buy":
                        peak_info["peak_price"] = max(
                            peak_info["peak_price"], current_price
                        )
                    else:
                        peak_info["peak_price"] = min(
                            peak_info["peak_price"] or current_price, current_price
                        )
                    if (not peak_info["activated"]
                            and progress >= self.settings.TRAILING_STOP_ACTIVATION_FRAC):
                        peak_info["activated"] = True
                        logger.info(
                            f"{ticker}: trailing stop ACTIVATED at "
                            f"progress={progress*100:.0f}% "
                            f"(peak={peak_info['peak_price']:.4f})"
                        )
                    # Persist when peak meaningfully moved or activation flipped.
                    # Avoid hammering DB every tick — only on real changes.
                    if (peak_info["peak_price"] != prev_peak
                            or peak_info["activated"] != prev_activated):
                        try:
                            await self.db.upsert_position_peak(
                                trade_id=trade_id,
                                peak_price=peak_info["peak_price"],
                                activated=peak_info["activated"],
                                partial_taken=peak_info.get("partial_taken", False),
                                partial_price=peak_info.get("partial_price"),
                            )
                        except Exception as e:
                            logger.debug(f"position_peak persist failed: {e}")

                    if peak_info["activated"]:
                        # Compute ATR for trail distance
                        from analysis.indicators import compute_indicators
                        try:
                            df_ind = compute_indicators(df)
                            atr = float(df_ind.iloc[-1].get("atr_14") or 0)
                        except Exception:
                            atr = 0.0
                        trail_dist = atr * self.settings.TRAILING_STOP_ATR_MULT
                        if trail_dist > 0:
                            peak = peak_info["peak_price"]
                            if direction == "buy":
                                trail_trigger = peak - trail_dist
                                if current_price <= trail_trigger:
                                    logger.info(
                                        f"{ticker}: trailing stop HIT "
                                        f"(peak={peak:.4f}, trail={trail_trigger:.4f}, "
                                        f"cur={current_price:.4f})"
                                    )
                                    await self.close_position(figi, "trailing_stop")
                                    self._position_peaks.pop(trade_id, None)
                                    try:
                                        await self.db.delete_position_peak(trade_id)
                                    except Exception as e:
                                        logger.debug(f"peak delete failed: {e}")
                                    continue
                            else:
                                trail_trigger = peak + trail_dist
                                if current_price >= trail_trigger:
                                    logger.info(
                                        f"{ticker}: trailing stop HIT "
                                        f"(peak={peak:.4f}, trail={trail_trigger:.4f}, "
                                        f"cur={current_price:.4f})"
                                    )
                                    await self.close_position(figi, "trailing_stop")
                                    self._position_peaks.pop(trade_id, None)
                                    try:
                                        await self.db.delete_position_peak(trade_id)
                                    except Exception as e:
                                        logger.debug(f"peak delete failed: {e}")
                                    continue

                # ---- Partial take-profit (MFE-aware) ------------------------
                # postmortem 2026-04-30: 47 % of losers (14/30) had touched
                # ≥+0.5 % MFE before turning back.  Scaling 50 % of the lots
                # off at MFE ≥ PARTIAL_TP_TRIGGER_ATR × ATR converts those
                # round-trip losers into break-even-or-better partial-winners
                # while leaving the runner half on for the trailing stop.
                if (getattr(self.settings, "PARTIAL_TP_ENABLED", False)
                        and not (self._position_peaks.get(trade_id, {}).get("partial_taken", False))
                        and trade["lots"] >= getattr(self.settings, "PARTIAL_TP_MIN_LOTS", 2)):
                    try:
                        from analysis.indicators import compute_indicators as _ci
                        df_ind2 = _ci(df)
                        atr_now = float(df_ind2.iloc[-1].get("atr_14") or 0)
                    except Exception:
                        atr_now = 0.0
                    trig_atr = float(getattr(self.settings, "PARTIAL_TP_TRIGGER_ATR", 1.5))
                    if atr_now > 0:
                        if direction == "buy":
                            mfe_now = current_price - entry_price
                        else:
                            mfe_now = entry_price - current_price
                        if mfe_now >= trig_atr * atr_now:
                            frac = float(getattr(self.settings, "PARTIAL_TP_FRAC", 0.5))
                            scale_lots = max(1, int(round(trade["lots"] * frac)))
                            scale_lots = min(scale_lots, trade["lots"] - 1)
                            if scale_lots >= 1:
                                logger.info(
                                    f"{ticker}: partial TP triggered — closing "
                                    f"{scale_lots}/{trade['lots']} lots at "
                                    f"MFE={mfe_now:+.4f} (ATR={atr_now:.4f}, "
                                    f"trigger={trig_atr}×)"
                                )
                                ok = await self._scale_out_position(
                                    trade, scale_lots, current_price
                                )
                                if ok:
                                    pi = self._position_peaks.setdefault(trade_id, {
                                        "peak_price": entry_price,
                                        "activated": False,
                                    })
                                    pi["partial_taken"] = True
                                    pi["partial_price"] = current_price
                                    try:
                                        await self.db.upsert_position_peak(
                                            trade_id=trade_id,
                                            peak_price=pi["peak_price"],
                                            activated=pi["activated"],
                                            partial_taken=True,
                                            partial_price=current_price,
                                        )
                                    except Exception as e:
                                        logger.debug(f"partial-TP persist failed: {e}")

                # ---- Hard stop-loss / take-profit (software fallback) ------
                # Broker-side stop orders may fail or be lost on bot restart.
                # This is the authoritative software guard.  Runs on every
                # PORTFOLIO_CHECK_MINUTES tick using the latest candle close.
                # postmortem 2026-04-25: 10/23 external_close trades exited
                # PAST their stop level because this check was absent.
                sl = trade.get("stop_loss")
                tp = trade.get("take_profit")
                if sl and sl > 0:
                    if direction == "buy" and current_price <= sl:
                        logger.info(
                            f"{ticker}: hard STOP-LOSS hit "
                            f"(cur={current_price:.4f} <= sl={sl:.4f})"
                        )
                        await self.close_position(figi, "stop_loss")
                        continue
                    elif direction == "sell" and current_price >= sl:
                        logger.info(
                            f"{ticker}: hard STOP-LOSS hit "
                            f"(cur={current_price:.4f} >= sl={sl:.4f})"
                        )
                        await self.close_position(figi, "stop_loss")
                        continue
                if tp and tp > 0:
                    if direction == "buy" and current_price >= tp:
                        logger.info(
                            f"{ticker}: hard TAKE-PROFIT hit "
                            f"(cur={current_price:.4f} >= tp={tp:.4f})"
                        )
                        await self.close_position(figi, "take_profit")
                        continue
                    elif direction == "sell" and current_price <= tp:
                        logger.info(
                            f"{ticker}: hard TAKE-PROFIT hit "
                            f"(cur={current_price:.4f} <= tp={tp:.4f})"
                        )
                        await self.close_position(figi, "take_profit")
                        continue

                # ---- Time-based exit ----------------------------------------
                # Open > 5 days and |P&L| still tiny → position has stalled,
                # free the slot for better trades.
                try:
                    entry_time = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                    days_open = (now - entry_time).days
                    if days_open > 5 and abs(pnl_pct) < self.settings.TIME_EXIT_MAX_PNL_PCT:
                        await self.close_position(figi, "time_exit")
                        continue
                except Exception:
                    pass

                # ---- Strategy exit signals ----------------------------------
                for strategy in self.strategies.values():
                    try:
                        exit_signal = await strategy.should_exit(
                            figi, ticker, entry_price, direction, df
                        )
                        if not exit_signal:
                            continue

                        # Reversal signals are gated asymmetrically:
                        #   Winners (P&L > 0):  require progress ≥ 40 % of
                        #       target so MACD wiggles don't close the WUSH
                        #       trade at +0.7 % with target +3.2 %.
                        #   Losers  (P&L < 0):  progress is negative by
                        #       definition — bypass the progress gate so the
                        #       bot can cut losses when the model flips
                        #       instead of riding every trade to full SL.
                        # Commission gate (|P&L| ≥ 2× round-trip) applies to
                        # BOTH cases — we never close on pure noise.
                        if exit_signal.reason == "signal_reversal":
                            # Gate 0: minimum hold time.  Without this, the
                            # 5-min portfolio_check loop re-scored the bar
                            # that JUST triggered entry, causing whipsaw exits
                            # (TTM6: long opened 12:07, reversed 12:12,
                            # −318 RUB futures loss).  Hold-time analysis
                            # showed the 5–30 min bucket lost −300 RUB across
                            # 14 trades while the 2–4 h bucket made +87 RUB
                            # at 91% win-rate.  We block any reversal that
                            # happens before MIN_HOLD_MINUTES_BEFORE_REVERSAL.
                            try:
                                _et = datetime.fromisoformat(entry_time_str.replace("Z", "+00:00"))
                                _hold_min = (now - _et).total_seconds() / 60.0
                                _min_hold = getattr(
                                    self.settings,
                                    "MIN_HOLD_MINUTES_BEFORE_REVERSAL",
                                    60,
                                )
                                if _hold_min < _min_hold:
                                    logger.info(
                                        f"{ticker}: ignoring signal_reversal "
                                        f"(held only {_hold_min:.1f} min, "
                                        f"need ≥ {_min_hold} min to avoid whipsaw)"
                                    )
                                    continue
                            except Exception:
                                pass

                            # Gate A: P&L must clear 2× round-trip commission
                            if abs(pnl_pct) < min_reversal_pnl_pct:
                                logger.info(
                                    f"{ticker}: ignoring signal_reversal "
                                    f"(P&L={pnl_pct:+.2f}%, need |P&L|≥"
                                    f"{min_reversal_pnl_pct:.2f}% to cover commission)"
                                )
                                continue
                            # Gate B (winners only): must have covered
                            # MIN_TARGET_PROGRESS_FRAC of the way to target.
                            # For losers this gate is skipped — we want to
                            # cut losses on strong reversal signals, not
                            # ride every mistake to the stop-loss.
                            if (pnl_pct > 0
                                    and target_dist > 0
                                    and progress < self.settings.MIN_TARGET_PROGRESS_FRAC):
                                logger.info(
                                    f"{ticker}: ignoring signal_reversal "
                                    f"(winner, progress {progress*100:.0f}% < "
                                    f"{self.settings.MIN_TARGET_PROGRESS_FRAC*100:.0f}% of target)"
                                )
                                continue
                            if pnl_pct < 0:
                                logger.info(
                                    f"{ticker}: accepting signal_reversal on LOSER "
                                    f"(P&L={pnl_pct:+.2f}%) — cutting losses early"
                                )

                        logger.info(f"Exit signal for {ticker}: {exit_signal.reason}")
                        await self.close_position(figi, exit_signal.reason)
                        break
                    except Exception as e:
                        logger.debug(f"Strategy exit check error for {ticker}: {e}")

            except Exception as e:
                logger.error(f"Exit check error for {ticker}: {e}")

    # ==================== Portfolio Sync ====================

    async def _portfolio_sync_loop(self):
        """Sync local state with broker portfolio."""
        while self._is_running:
            await asyncio.sleep(self.settings.PORTFOLIO_CHECK_MINUTES * 60)
            try:
                await self._sync_portfolio()
            except Exception as e:
                logger.error(f"Portfolio sync error: {e}")

    async def _sync_portfolio(self):
        """Update P&L for all open trades and detect externally closed positions."""
        open_trades = await self.db.get_open_trades()
        if not open_trades:
            return

        figis = list({t["figi"] for t in open_trades})
        try:
            prices = await self.broker.get_last_prices(figis)
        except Exception:
            return

        # Detect positions closed externally (via terminal / another app)
        try:
            broker_positions = await self.broker.get_positions()
            # Build a set of FIGIs with non-zero balance at the broker.
            # IMPORTANT: Tinkoff returns SHARES in `positions.securities` and
            # FUTURES in `positions.futures` (separate fields).  Previously only
            # securities were read → external-close detection silently treated
            # every open futures position as "still alive at broker", so
            # manually-closed futures would stay open forever in our DB.
            broker_figis: set[str] = set()
            for sec in getattr(broker_positions, "securities", []):
                balance = getattr(sec, "balance", 0)
                if balance and balance != 0:
                    broker_figis.add(sec.figi)
            for fut in getattr(broker_positions, "futures", []):
                balance = getattr(fut, "balance", 0)
                if balance and balance != 0:
                    broker_figis.add(fut.figi)
        except Exception as e:
            logger.warning(f"Could not fetch broker positions for sync: {e}")
            broker_figis = None  # skip external-close check this cycle

        if broker_figis is not None:
            for trade in open_trades:
                if trade["figi"] not in broker_figis:
                    # Position is gone at the broker but still open in our DB.
                    # Try to get the actual execution price from operations history
                    # (covers manual closes, broker SL/TP fills, etc.).
                    # Fall back to last market price only if no matching op found.
                    exit_price = 0.0
                    try:
                        from t_tech.invest import OperationType
                        from_dt = datetime.now(timezone.utc) - timedelta(minutes=90)
                        # Long close = SELL; short close = BUY
                        if trade.get("direction") == "sell":
                            close_types = [
                                OperationType.OPERATION_TYPE_BUY,
                                OperationType.OPERATION_TYPE_BUY_MARGIN,
                            ]
                        else:
                            close_types = [
                                OperationType.OPERATION_TYPE_SELL,
                                OperationType.OPERATION_TYPE_SELL_MARGIN,
                            ]
                        ops = await self.broker.get_operations(
                            from_dt=from_dt, operation_types=close_types)
                        figi_ops = [
                            op for op in ops
                            if getattr(op, "figi", None) == trade["figi"]
                        ]
                        if figi_ops:
                            # Most recent matching operation
                            figi_ops.sort(
                                key=lambda o: getattr(o, "date", datetime.min.replace(tzinfo=timezone.utc)),
                                reverse=True,
                            )
                            op = figi_ops[0]
                            price_q = getattr(op, "price", None)
                            if price_q is not None:
                                from t_tech.invest.utils import quotation_to_decimal
                                exit_price = float(quotation_to_decimal(price_q))
                                logger.info(
                                    f"_sync_portfolio: exit price for {trade['ticker']} "
                                    f"taken from operations history: {exit_price:.4f}"
                                )
                    except Exception as e:
                        logger.debug(f"Could not fetch operations for exit price: {e}")

                    if exit_price <= 0:
                        exit_price = float(prices.get(trade["figi"], Decimal("0")))
                    if exit_price <= 0:
                        try:
                            exit_price = float(await self.broker.get_last_price(trade["figi"]))
                        except Exception:
                            exit_price = trade["entry_price"]

                    entry = trade["entry_price"]
                    lots = trade["lots"]
                    lot_size = trade.get("lot_size", 1)
                    pnl, pnl_pct, commission = self._compute_pnl(
                        trade["direction"], entry, exit_price, lots, lot_size
                    )

                    now_iso = datetime.now(timezone.utc).isoformat()

                    # Infer WHY the position disappeared from the broker:
                    #   • If exit_price crossed the stored stop_loss → broker
                    #     stop order fired (not a manual close).
                    #   • If exit_price crossed take_profit → broker TP fired.
                    #   • Otherwise → genuine external close (user terminal etc.)
                    # postmortem 2026-04-25: 23/61 trades were labelled
                    # "external_close" even though the user never manually closed
                    # them — all were broker stop/TP executions.
                    sl = trade.get("stop_loss") or 0
                    tp = trade.get("take_profit") or 0
                    direction_t = trade["direction"]
                    inferred_reason = "external_close"
                    # Tightened tolerance: 0.05% instead of 0.2%.  postmortem
                    # 2026-04-30: the 0.2% band was bucketing genuine manual
                    # closes as "stop_loss" when the exit happened to fall
                    # within a wide ATR-based stop's 0.2% halo.  Forensic
                    # pipelines downstream then under-counted the manual-
                    # close incidence and over-reported the bot's own SL
                    # accuracy.  0.05% (~5 bps) approximates broker-side
                    # stop slippage on liquid MOEX shares.
                    SL_TOL = 0.0005
                    if sl and sl > 0:
                        if direction_t == "buy" and exit_price <= sl * (1.0 + SL_TOL):
                            inferred_reason = "stop_loss"
                        elif direction_t == "sell" and exit_price >= sl * (1.0 - SL_TOL):
                            inferred_reason = "stop_loss"
                    if inferred_reason == "external_close" and tp and tp > 0:
                        if direction_t == "buy" and exit_price >= tp * (1.0 - SL_TOL):
                            inferred_reason = "take_profit"
                        elif direction_t == "sell" and exit_price <= tp * (1.0 + SL_TOL):
                            inferred_reason = "take_profit"

                    logger.info(
                        f"Broker-closed position detected: {trade['ticker']} "
                        f"exit={exit_price:.4f} sl={sl:.4f} tp={tp:.4f} "
                        f"→ reason inferred as '{inferred_reason}'"
                    )

                    await self.db.update_trade(
                        trade["id"],
                        status="closed",
                        exit_price=exit_price,
                        exit_order_id="external",
                        exit_time=now_iso,
                        exit_reason=inferred_reason,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                    )
                    self.risk_manager.update_daily_pnl(pnl)
                    today = datetime.now(timezone.utc).date().isoformat()
                    await self.db.upsert_daily_pnl(today, realized_pnl=pnl)

                    # Cooldown applies to all broker-closes (stop, TP, external).
                    self._record_close(trade["figi"])

                    # Clean up orphaned partner stop/TP order at broker.
                    await self._cancel_stops_for_figi(trade["figi"])
                    self._position_peaks.pop(trade["id"], None)
                    try:
                        await self.db.delete_position_peak(trade["id"])
                    except Exception as e:
                        logger.debug(f"peak delete failed: {e}")

                    await self._notify(Notification(
                        type=NotificationType.TRADE_CLOSED,
                        title=f"{inferred_reason.replace('_',' ').title()}: {trade['ticker']} {pnl:+.0f} ₽",
                        data={**dict(trade), "exit_price": exit_price,
                              "pnl": pnl, "pnl_pct": pnl_pct,
                              "commission": commission,
                              "exit_reason": inferred_reason},
                        priority="high",
                    ))
                    logger.info(
                        f"Broker-closed: {trade['ticker']} reason={inferred_reason} "
                        f"P&L={pnl:+.2f} ₽ ({pnl_pct:+.2f}%) "
                        f"commission={commission:.2f} ₽"
                    )

        # Reload open trades (some may have just been closed above)
        open_trades = await self.db.get_open_trades()

        total_unrealized = 0.0
        for trade in open_trades:
            price = float(prices.get(trade["figi"], Decimal("0")))
            if price <= 0:
                continue

            entry = trade["entry_price"]
            lots = trade["lots"]
            lot_size = trade.get("lot_size", 1)

            # Unrealised P&L must subtract the round-trip commission we'd
            # pay if we closed now — otherwise the dashboard shows wins
            # that disappear the moment you exit.
            pnl, _pnl_pct, _comm = self._compute_pnl(
                trade["direction"], entry, price, lots, lot_size
            )

            total_unrealized += pnl

        today = datetime.now(timezone.utc).date().isoformat()
        await self.db.upsert_daily_pnl(today, unrealized_pnl=round(total_unrealized, 2))

    # ==================== Retrain Scheduler ====================

    async def _retrain_scheduler(self):
        """Periodic ML model retraining.

        First iteration sleeps for ML_RETRAIN_HOURS (default 24h) so the
        scheduled retrain happens once a day — not immediately on startup.
        On startup the API quota is often partially exhausted by the screener
        scan, which means the training candle fetches fail and the model is
        trained on a tiny dataset (observed: 2402 vs 36975 samples →
        acc 0.39 vs 0.45 → rollback gate rejects it, wasting ~5min of CPU).
        Sleeping first gives the token-bucket rate-limiter time to refill.
        """
        while self._is_running:
            await asyncio.sleep(self.settings.ML_RETRAIN_HOURS * 3600)
            if self._trainer and self._watchlist:
                await self.retrain_models()

    async def retrain_models(self):
        """Retrain ML models for watchlist instruments."""
        if not self._trainer:
            logger.warning("No trainer configured")
            return

        tickers_figis = [
            (c["ticker"], c["figi"], c.get("kind", "share"))
            for c in self._watchlist[:15]
        ]
        try:
            model = await self._trainer.train_universal_model(tickers_figis)
            if model and "ml_lightgbm" in self.strategies:
                self.strategies["ml_lightgbm"].set_model(model)
                await self._notify(Notification(
                    type=NotificationType.MODEL_TRAINED,
                    title="ML model retrained",
                    data={
                        "ticker": "universal",
                        "accuracy": model.metadata.accuracy if model.metadata else 0,
                        "f1": model.metadata.f1 if model.metadata else 0,
                        "samples": model.metadata.train_samples if model.metadata else 0,
                    }
                ))
        except Exception as e:
            logger.error(f"Retrain error: {e}")
            await self._notify(Notification(
                type=NotificationType.ERROR,
                title=f"Retrain failed: {e}"
            ))

    # ==================== Manual Trading ====================

    async def manual_trade(self, figi: str, ticker: str, lots: int,
                            direction: str) -> dict:
        """Execute a manual trade from Telegram."""
        # Pass manual_lots so risk manager uses user-specified count (bypasses Kelly)
        approval = await self.risk_manager.approve_trade(
            figi=figi, ticker=ticker, direction=direction,
            confidence=1.0,
            suggested_stop_pct=2.0, suggested_target_pct=4.0,
            manual_lots=lots,
        )

        if not approval.approved:
            return {"success": False, "reason": approval.reason}

        return await self._execute_trade(
            figi, ticker, direction, approval.lots,
            "manual", 1.0, approval
        )

    async def close_position(self, figi: str, reason: str = "manual") -> dict:
        """Close an open position."""
        open_trades = await self.db.get_open_trades()
        trade = next((t for t in open_trades if t["figi"] == figi), None)
        if not trade:
            return {"success": False, "reason": "Position not found"}

        direction = trade["direction"]
        lots = trade["lots"]

        # Opposite direction to close
        close_dir = OrderDirection.ORDER_DIRECTION_SELL if direction == "buy" \
                    else OrderDirection.ORDER_DIRECTION_BUY
        stop_cancel_dir = StopOrderDirection.STOP_ORDER_DIRECTION_SELL if direction == "sell" \
                          else StopOrderDirection.STOP_ORDER_DIRECTION_BUY

        try:
            from utils.helpers import is_moex_open
            if not is_moex_open():
                logger.warning(
                    f"close_position({trade['ticker']}): MOEX is closed right now "
                    f"— skipping market order, will retry when market opens"
                )
                return {"success": False, "reason": "market_closed", "retry_later": True}

            response = await self.broker.post_market_order(figi, lots, close_dir)
            exit_price = float(await self.broker.get_last_price(figi))

            # Cancel the SL/TP stop orders we placed at entry.  Without this,
            # orphan stops sit at the broker and can fire on a later unrelated
            # re-entry of the same ticker (or against a zero balance).
            await self._cancel_stops_for_figi(figi)

            # Forget any trailing-stop peak we tracked
            self._position_peaks.pop(trade["id"], None)
            try:
                await self.db.delete_position_peak(trade["id"])
            except Exception as e:
                logger.debug(f"peak delete failed: {e}")

            entry_price = trade["entry_price"]
            lot_size = trade.get("lot_size", 1)
            pnl, pnl_pct, commission = self._compute_pnl(
                direction, entry_price, exit_price, lots, lot_size
            )

            now = datetime.now(timezone.utc).isoformat()
            await self.db.update_trade(
                trade["id"],
                status="closed",
                exit_price=exit_price,
                exit_order_id=response.order_id,
                exit_time=now,
                exit_reason=reason,
                pnl=pnl,
                pnl_pct=pnl_pct,
            )

            self.risk_manager.update_daily_pnl(pnl)

            # Update daily P&L
            today = datetime.now(timezone.utc).date().isoformat()
            await self.db.upsert_daily_pnl(today, realized_pnl=pnl)

            # Cooldown: no re-entry into this ticker for a while
            self._record_close(figi)

            trade_data = {**dict(trade), "exit_price": exit_price,
                         "pnl": pnl, "pnl_pct": pnl_pct,
                         "commission": commission,
                         "exit_reason": reason}

            await self._notify(Notification(
                type=NotificationType.TRADE_CLOSED,
                title=f"Closed {trade['ticker']}: {pnl:+.0f} RUB",
                data=trade_data, priority="high"
            ))

            logger.info(
                f"Position closed: {trade['ticker']} "
                f"P&L={pnl:+.2f} RUB ({pnl_pct:+.2f}%) "
                f"commission={commission:.2f} RUB"
            )
            return {"success": True, "pnl": pnl, "pnl_pct": pnl_pct,
                    "commission": commission}

        except Exception as e:
            err_str = str(e)
            # 30049 = "Замороженная цена не соответствует типу заявки"
            # Occurs when MOEX is in pre-open auction or on holidays.
            # Not a real error — just can't execute a market order right now.
            if "30049" in err_str:
                logger.warning(
                    f"close_position({trade.get('ticker', figi)}): market order "
                    f"rejected (30049 frozen price) — MOEX likely in auction/holiday. "
                    f"Will retry when market reopens."
                )
                return {"success": False, "reason": "market_closed_30049", "retry_later": True}
            logger.error(f"Close position error for {figi}: {e}")
            return {"success": False, "reason": str(e)}

    async def _scale_out_position(self, trade: dict, lots: int,
                                   ref_price: float) -> bool:
        """Close ``lots`` of an open trade (partial scale-out) without ending
        the trade row in the DB.  Updates trade.lots and credits a notional
        partial-realised P&L on the daily counter.  Returns True on success.

        Note: SL/TP stop orders at the broker are NOT resized here — they
        still cover the original lot count.  When the residual position is
        finally closed, _cancel_stops_for_figi cleans up.  Worst case: the
        broker stop is bigger than the remaining position, and the broker
        rejects with "insufficient balance" — handled gracefully by their
        risk system.  The benefit (locked-in partial profit) outweighs the
        residual stop-sizing mismatch.
        """
        if lots <= 0 or lots >= trade["lots"]:
            return False
        figi = trade["figi"]
        ticker = trade["ticker"]
        direction = trade["direction"]
        close_dir = OrderDirection.ORDER_DIRECTION_SELL if direction == "buy" \
                    else OrderDirection.ORDER_DIRECTION_BUY
        try:
            await self.broker.post_market_order(figi, lots, close_dir)
            try:
                fill_price = float(await self.broker.get_last_price(figi))
            except Exception:
                fill_price = ref_price
            entry_price = trade["entry_price"]
            lot_size = trade.get("lot_size", 1)
            partial_pnl, partial_pnl_pct, partial_comm = self._compute_pnl(
                direction, entry_price, fill_price, lots, lot_size
            )
            # Reduce the open trade's lot count.  We don't close the row —
            # the runner is still live.
            new_lots = trade["lots"] - lots
            await self.db.update_trade(trade["id"], lots=new_lots)
            self.risk_manager.update_daily_pnl(partial_pnl)
            today = datetime.now(timezone.utc).date().isoformat()
            await self.db.upsert_daily_pnl(today, realized_pnl=partial_pnl)
            await self._notify(Notification(
                type=NotificationType.INFO,
                title=f"Partial TP {ticker}: {partial_pnl:+.0f} ₽ ({lots} lots)",
                data={
                    "ticker": ticker,
                    "lots_closed": lots,
                    "lots_remaining": new_lots,
                    "fill_price": fill_price,
                    "partial_pnl": partial_pnl,
                    "partial_pnl_pct": partial_pnl_pct,
                },
            ))
            logger.info(
                f"Partial TP filled: {ticker} closed {lots} lots @ {fill_price:.4f} "
                f"P&L={partial_pnl:+.2f} ₽ ({partial_pnl_pct:+.2f}%); "
                f"runner = {new_lots} lots"
            )
            return True
        except Exception as e:
            logger.error(f"Partial TP for {ticker} failed: {e}")
            return False

    async def emergency_stop(self):
        """Cancel all orders, pause autonomous mode."""
        logger.warning("EMERGENCY STOP triggered")
        self._mode = "interactive"
        self.settings.MODE = "interactive"

        try:
            orders = await self.broker.get_orders()
            for order in orders:
                try:
                    await self.broker.cancel_order(order.order_id)
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Cancel orders error: {e}")

        try:
            stop_orders = await self.broker.get_stop_orders()
            for so in stop_orders:
                try:
                    await self.broker.cancel_stop_order(so.stop_order_id)
                except Exception:
                    pass
        except Exception:
            pass

        await self._notify(Notification(
            type=NotificationType.INFO,
            title="Emergency stop executed. Mode: INTERACTIVE",
            priority="critical"
        ))

    # ==================== Status & Display ====================

    async def get_status(self) -> dict:
        """Get engine status for Telegram display."""
        open_trades = await self.db.get_open_trades()
        risk = await self.risk_manager.get_risk_status()
        return {
            "mode": self._mode,
            "is_running": self._is_running,
            "open_positions": len(open_trades),
            "daily_pnl": risk["daily_pnl"],
            "drawdown": risk["drawdown"],
        }

    async def get_portfolio_display(self) -> list[dict]:
        """Get portfolio for display with current prices."""
        open_trades = await self.db.get_open_trades()
        if not open_trades:
            return []

        figis = list({t["figi"] for t in open_trades})
        try:
            prices = await self.broker.get_last_prices(figis)
        except Exception:
            prices = {}

        result = []
        for t in open_trades:
            current = float(prices.get(t["figi"], Decimal(str(t["entry_price"]))))
            result.append({**t, "current_price": current})
        return result

    # ==================== Helpers ====================

    def _commission_pct(self, instrument_kind: str = "share") -> float:
        """Round-trip commission fraction per Tinkoff 'Trader' tariff.

        One side of the trade; the round-trip (entry + exit) is 2x this.
        Defaults to shares tariff — the only instrument kind we currently
        trade via the screener.
        """
        kind = (instrument_kind or "share").lower()
        if kind in ("future", "futures"):
            return self.settings.COMMISSION_FUTURES_PCT
        if kind in ("currency", "fx"):
            return self.settings.COMMISSION_CURRENCY_PCT
        if kind in ("metal", "metals", "precious_metal"):
            return self.settings.COMMISSION_METALS_PCT
        # share / bond / etf / fund
        return self.settings.COMMISSION_SHARES_PCT

    def _compute_pnl(self, direction: str, entry_price: float, exit_price: float,
                      lots: int, lot_size: int,
                      instrument_kind: str = "share") -> tuple[float, float, float]:
        """P&L net of round-trip broker commission.

        Returns (pnl_rub, pnl_pct, commission_rub).
        pnl_pct is computed on the gross move (so it matches what the
        user sees on their chart), but pnl_rub is net — commission is
        already subtracted.
        """
        if entry_price <= 0 or lots <= 0 or lot_size <= 0:
            return 0.0, 0.0, 0.0

        comm_pct = self._commission_pct(instrument_kind)
        notional_entry = entry_price * lots * lot_size
        notional_exit = exit_price * lots * lot_size
        commission = (notional_entry + notional_exit) * comm_pct

        if direction == "buy":
            gross_pnl = (exit_price - entry_price) * lots * lot_size
            pnl_pct = (exit_price - entry_price) / entry_price * 100
        else:
            gross_pnl = (entry_price - exit_price) * lots * lot_size
            pnl_pct = (entry_price - exit_price) / entry_price * 100

        net_pnl = gross_pnl - commission
        return round(net_pnl, 2), round(pnl_pct, 2), round(commission, 2)

    def _is_in_cooldown(self, figi: str) -> bool:
        """True iff figi was closed within the last SAME_TICKER_COOLDOWN_MINUTES."""
        last = self._recent_closes.get(figi)
        if not last:
            return False
        window = timedelta(minutes=self.settings.SAME_TICKER_COOLDOWN_MINUTES)
        return (datetime.now(timezone.utc) - last) < window

    def _record_close(self, figi: str):
        """Stamp a close time for cooldown tracking (in-memory + persistent)."""
        now = datetime.now(timezone.utc)
        self._recent_closes[figi] = now
        # Persist asynchronously so a bot restart preserves the cooldown.
        # Fire-and-forget — failure here mustn't block the close path.
        try:
            asyncio.get_event_loop().create_task(
                self.db.upsert_cooldown(figi, now.isoformat())
            )
        except Exception as e:
            logger.debug(f"Cooldown persist task failed: {e}")
        # Garbage-collect old entries so the dict doesn't grow forever
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=self.settings.SAME_TICKER_COOLDOWN_MINUTES * 2
        )
        self._recent_closes = {
            k: v for k, v in self._recent_closes.items() if v >= cutoff
        }

    async def _build_watchlist(self, custom_figis: list[str]) -> list[dict]:
        """Build combined long+short watchlist.

        Long pass: ranks by +momentum.  Short pass (only if ALLOW_SHORTS):
        ranks by −momentum AND filters on short_enabled_flag.  Results are
        merged and deduplicated by FIGI — if the same ticker somehow ranks
        in both, the higher-scoring side wins (the candidate's "direction"
        field then dictates which trade side we attempt).

        Output is sorted by score desc and length-capped by the original
        screener.top_n — so turning shorts on does NOT halve long throughput.
        """
        long_wl = await self.screener.scan_universe(
            custom_figis=custom_figis, direction="long"
        )
        if not self.settings.ALLOW_SHORTS:
            return long_wl

        short_wl = await self.screener.scan_universe(
            custom_figis=custom_figis, direction="short"
        )

        # Dedupe: keep whichever side scored higher for each FIGI
        merged: dict[str, dict] = {}
        for cand in long_wl + short_wl:
            existing = merged.get(cand["figi"])
            if existing is None or cand.get("score", 0) > existing.get("score", 0):
                merged[cand["figi"]] = cand

        combined = sorted(merged.values(), key=lambda c: c.get("score", 0), reverse=True)
        # Cap to screener.top_n so the loop size stays the same whether
        # shorts are on or off
        capped = combined[: self.screener.top_n]

        long_n = sum(1 for c in capped if c.get("direction") == "long")
        short_n = sum(1 for c in capped if c.get("direction") == "short")
        logger.info(
            f"Watchlist built: {long_n} long + {short_n} short = {len(capped)} total"
        )
        return capped

    async def _cancel_stops_for_figi(self, figi: str) -> int:
        """Cancel every active stop order (SL + TP) matching this FIGI.

        Fixes two orphan-order bugs at once:
        1. When an SL fires, the partner TP remains active at the broker
           and can trigger on a later unrelated re-entry.
        2. When close_position() runs, neither SL nor TP is cancelled —
           same orphan problem.

        Called from close_position() and from the external-close branch
        of _sync_portfolio().

        Returns the number of stops cancelled.
        """
        cancelled = 0
        try:
            stops = await self.broker.get_stop_orders()
        except Exception as e:
            logger.warning(f"Could not fetch stop orders for cleanup: {e}")
            return 0
        for so in stops:
            so_figi = getattr(so, "figi", None)
            so_id = getattr(so, "stop_order_id", None)
            if so_figi == figi and so_id:
                try:
                    await self.broker.cancel_stop_order(so_id)
                    cancelled += 1
                except Exception as e:
                    logger.warning(f"Could not cancel stop {so_id} for {figi}: {e}")
        if cancelled:
            logger.info(f"Cancelled {cancelled} stop order(s) for {figi}")
        return cancelled

    async def _notify(self, notification: Notification):
        """Push notification to queue (non-blocking)."""
        try:
            self.notification_queue.put_nowait(notification)
        except asyncio.QueueFull:
            logger.warning("Notification queue full, dropping notification")
