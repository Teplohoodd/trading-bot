"""Central risk manager: approves/rejects trades based on multiple checks."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Optional

from t_tech.invest import OrderDirection, StopOrderDirection
from t_tech.invest.utils import quotation_to_decimal

from core.broker import BrokerClient
from risk.spread_monitor import SpreadMonitor
from risk.position_sizer import PositionSizer
from risk.execution import ExecutionScheduler
from database.db import Repository
from config.settings import Settings
from utils.helpers import (
    is_moex_open,
    is_market_tradeable,
    is_weekend_session,
    now_msk,
    money_to_decimal,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeApproval:
    approved: bool
    lots: int = 0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    use_twap: bool = False
    reason: str = ""
    risk_metrics: dict = field(default_factory=dict)


class RiskManager:
    """Central risk gate. Every trade must pass approve_trade()."""

    def __init__(
        self,
        broker: BrokerClient,
        spread_monitor: SpreadMonitor,
        position_sizer: PositionSizer,
        execution_scheduler: ExecutionScheduler,
        db: Repository,
        settings: Settings,
    ):
        self.broker = broker
        self.spread_monitor = spread_monitor
        self.position_sizer = position_sizer
        self.execution = execution_scheduler
        self.db = db
        self.settings = settings

        # Daily tracking
        self._daily_pnl: float = 0.0
        self._peak_portfolio: float = 0.0
        self._current_portfolio: float = 0.0
        self._trade_count_today: int = 0
        self._last_reset_date: Optional[date] = None

    async def approve_trade(
        self,
        figi: str,
        ticker: str,
        direction: str,
        confidence: float,
        suggested_stop_pct: float,
        suggested_target_pct: float,
        manual_lots: int = 0,
    ) -> TradeApproval:
        """Run all risk checks and return approval with sizing.

        Args:
            figi: Instrument FIGI.
            ticker: Instrument ticker.
            direction: "buy" or "sell".
            confidence: Signal confidence (0-1).
            suggested_stop_pct: ATR-based stop as % of price.
            suggested_target_pct: Target as % of price.

        Returns:
            TradeApproval with lots, stops, and approval status.
        """
        metrics = {}

        # 0. Daily reset check
        await self._check_daily_reset()

        # 1. Resolve instrument up-front — needed for the market-hours check
        # (weekend_flag) and reused later for sizing so we don't pay a second
        # round-trip to the broker.
        try:
            instrument, instrument_kind = await self.broker.get_instrument_info(figi)
            if instrument is None:
                return TradeApproval(approved=False, reason=f"Unknown instrument (figi={figi})")
            weekend_flag = bool(getattr(instrument, "weekend_flag", False))
            lot_size = getattr(instrument, "lot", 1) or 1
        except Exception as e:
            return TradeApproval(approved=False, reason=f"Cannot get instrument info: {e}")

        # 2. Market hours — honour weekend trading for weekend-enabled shares.
        if not is_market_tradeable(weekend_flag=weekend_flag):
            if is_weekend_session() and not weekend_flag:
                return TradeApproval(
                    approved=False, reason=f"{ticker} not available in weekend session"
                )
            return TradeApproval(approved=False, reason="Market closed")

        # 3. Check trading status
        try:
            status = await self.broker.get_trading_status(figi)
            market_ok = getattr(status, "market_order_available_flag", False)
            limit_ok = getattr(status, "limit_order_available_flag", False)
            if not market_ok and not limit_ok:
                return TradeApproval(approved=False, reason=f"Trading not available for {ticker}")
        except Exception as e:
            logger.warning(f"Could not check trading status for {ticker}: {e}")

        # 3. Portfolio value
        try:
            portfolio_value = await self.broker.get_portfolio_value()
            self._current_portfolio = float(portfolio_value)
            if self._peak_portfolio == 0:
                self._peak_portfolio = self._current_portfolio
            else:
                self._peak_portfolio = max(self._peak_portfolio, self._current_portfolio)
        except Exception as e:
            return TradeApproval(approved=False, reason=f"Cannot get portfolio value: {e}")

        # 4. Daily loss circuit breaker
        if self._current_portfolio > 0:
            daily_loss_pct = abs(self._daily_pnl) / self._current_portfolio
            if self._daily_pnl < 0 and daily_loss_pct > self.settings.MAX_DAILY_LOSS_PCT:
                return TradeApproval(
                    approved=False,
                    reason=f"Daily loss limit reached: {daily_loss_pct:.1%} > {self.settings.MAX_DAILY_LOSS_PCT:.1%}",
                )
            metrics["daily_loss_pct"] = round(daily_loss_pct * 100, 2)

        # 5. Max drawdown circuit breaker
        if self._peak_portfolio > 0:
            drawdown = (self._peak_portfolio - self._current_portfolio) / self._peak_portfolio
            if drawdown > self.settings.MAX_DRAWDOWN_PCT:
                return TradeApproval(
                    approved=False,
                    reason=f"Max drawdown reached: {drawdown:.1%} > {self.settings.MAX_DRAWDOWN_PCT:.1%}",
                )
            metrics["drawdown_pct"] = round(drawdown * 100, 2)

        # 6. Position count limit
        open_trades = await self.db.get_open_trades()
        if len(open_trades) >= self.settings.MAX_POSITIONS:
            return TradeApproval(
                approved=False,
                reason=f"Max positions reached: {len(open_trades)}/{self.settings.MAX_POSITIONS}",
            )
        metrics["open_positions"] = len(open_trades)

        # 7. Check if already in this instrument
        for t in open_trades:
            if t["figi"] == figi:
                return TradeApproval(approved=False, reason=f"Already in position for {ticker}")

        # 8. Spread check (Glosten-Milgrom)
        try:
            ob = await self.broker.get_order_book(figi, depth=10)
            if ob.bids and ob.asks:
                bid = quotation_to_decimal(ob.bids[0].price)
                ask = quotation_to_decimal(ob.asks[0].price)
                self.spread_monitor.record_spread(figi, bid, ask)

                spread_normal, spread_ratio = self.spread_monitor.is_spread_normal(figi)
                metrics["spread_ratio"] = spread_ratio
                if not spread_normal:
                    return TradeApproval(
                        approved=False,
                        reason=f"Spread anomaly: {spread_ratio:.1f}x normal (Glosten-Milgrom filter)",
                    )
        except Exception as e:
            logger.warning(f"Order book check failed for {ticker}: {e}")

        # 9. Last price (instrument + lot_size already resolved at step 1)
        try:
            price = await self.broker.get_last_price(figi)
        except Exception as e:
            return TradeApproval(approved=False, reason=f"Cannot get last price: {e}")

        metrics["instrument_kind"] = instrument_kind
        metrics["lot_size"] = lot_size
        metrics["weekend_flag"] = weekend_flag
        if is_weekend_session():
            metrics["weekend_session"] = True

        # 9b. Short-selling check
        if direction == "sell":
            # Check if we already hold a long position (selling to close is allowed)
            already_long = any(t["figi"] == figi and t["direction"] == "buy" for t in open_trades)
            if not already_long:
                # Opening a short
                if instrument_kind == "future":
                    # Futures: shorting is free and symmetric, gated by ALLOW_FUTURES_SHORT
                    if not getattr(self.settings, "ALLOW_FUTURES_SHORT", True):
                        return TradeApproval(
                            approved=False,
                            reason="Futures shorts disabled (set ALLOW_FUTURES_SHORT=true to enable)",
                        )
                else:
                    # Shares: require margin short enablement + ALLOW_SHORTS
                    if not self.settings.ALLOW_SHORTS:
                        return TradeApproval(
                            approved=False,
                            reason="Short selling disabled (set ALLOW_SHORTS=true to enable)",
                        )
                    short_ok = getattr(instrument, "short_enabled_flag", False)
                    if not short_ok:
                        return TradeApproval(
                            approved=False, reason=f"{ticker} does not support short selling"
                        )

        # 10. Estimate daily volume and volatility
        try:
            from datetime import timedelta, timezone

            now = datetime.now(timezone.utc)
            candles = await self.broker.get_candles(figi, now - timedelta(days=20), now)
            if candles:
                volumes = [c.volume for c in candles]
                avg_daily_volume = sum(volumes) / len(volumes) * lot_size if volumes else 100000
                closes = [float(quotation_to_decimal(c.close)) for c in candles]
                returns = [
                    (closes[i] - closes[i - 1]) / closes[i - 1]
                    for i in range(1, len(closes))
                    if closes[i - 1] > 0
                ]
                volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5 if returns else 0.02
                # ATR
                atr_values = []
                for i in range(1, len(candles)):
                    h = float(quotation_to_decimal(candles[i].high))
                    l = float(quotation_to_decimal(candles[i].low))
                    pc = float(quotation_to_decimal(candles[i - 1].close))
                    tr = max(h - l, abs(h - pc), abs(l - pc))
                    atr_values.append(tr)
                atr = (
                    Decimal(str(sum(atr_values[-14:]) / min(len(atr_values), 14)))
                    if atr_values
                    else price * Decimal("0.02")
                )
            else:
                avg_daily_volume = 100000
                volatility = 0.02
                atr = price * Decimal("0.02")
        except Exception:
            avg_daily_volume = 100000
            volatility = 0.02
            atr = price * Decimal("0.02")

        # 11. Position sizing (Kyle-Obizhaeva + Kelly)
        # Per-direction stats (postmortem 2026-04-30): pooled stats made the
        # bot under-size profitable shorts because losing longs dragged the
        # win-rate down.  When same-direction history is thin (<10 trades)
        # we fall back to pooled stats so cold-start sizing still works.
        stats = await self.db.get_trade_stats(direction=direction)
        if stats.get("total", 0) < 10:
            stats = await self.db.get_trade_stats()
        metrics["stats_kelly_f"] = round(stats.get("kelly_f", 0.0), 3)
        metrics["stats_n"] = stats.get("total", 0)

        # 11a. StoplossGuard (per-side).  freqtrade-style: after N losing
        # exits on the same side within a lookback window, pause that side
        # for a cool-off period.  Manual trades bypass.
        if manual_lots == 0 and getattr(self.settings, "STOP_GUARD_ENABLED", False):
            try:
                lookback_h = int(getattr(self.settings, "STOP_GUARD_LOOKBACK_HOURS", 4))
                pause_h = int(getattr(self.settings, "STOP_GUARD_PAUSE_HOURS", 4))
                count_thr = int(getattr(self.settings, "STOP_GUARD_COUNT", 3))
                cutoff = datetime.utcnow() - timedelta(hours=lookback_h)
                cur = await self.db._db.execute(
                    "SELECT COUNT(*) FROM trades WHERE status='closed' "
                    "AND direction = ? AND pnl < 0 "
                    "AND exit_time >= ?",
                    (direction, cutoff.isoformat()),
                )
                row = await cur.fetchone()
                losses_recent = int(row[0] or 0) if row else 0
                if losses_recent >= count_thr:
                    return TradeApproval(
                        approved=False,
                        reason=(
                            f"StoplossGuard: {losses_recent} losing {direction} trades in "
                            f"last {lookback_h}h ≥ threshold {count_thr}; pausing "
                            f"side for {pause_h}h"
                        ),
                    )
            except Exception as e:
                logger.warning(f"StoplossGuard check failed (allowing trade): {e}")

        # 11b. Long-side auto-pause.  When realised long-Kelly is negative and
        # we have enough history to trust that, refuse new long entries unless
        # confidence is well above the regular threshold.  Manual trades
        # (manual_lots>0) bypass this — operator override always wins.
        if (
            manual_lots == 0
            and direction == "buy"
            and getattr(self.settings, "LONG_AUTO_PAUSE", False)
        ):
            long_stats = await self.db.get_trade_stats(direction="buy")
            if (
                long_stats.get("total", 0) >= getattr(self.settings, "LONG_PAUSE_MIN_HISTORY", 20)
                and long_stats.get("kelly_f", 0.0) < 0
            ):
                hi_conf_floor = getattr(self.settings, "LONG_MIN_CONFIDENCE_WHEN_BAD_EDGE", 0.85)
                if confidence < hi_conf_floor:
                    return TradeApproval(
                        approved=False,
                        reason=(
                            f"Long-side auto-pause: realised Kelly f*="
                            f"{long_stats['kelly_f']:+.2f} on n={long_stats['total']} buys, "
                            f"need confidence ≥ {hi_conf_floor:.2f} (got {confidence:.2f})"
                        ),
                    )

        if manual_lots > 0:
            # Manual trade: use user-requested lots, but cap by portfolio risk limit
            atr_f = float(atr) if atr else float(price) * 0.02
            stop_mult = float(getattr(self.settings, "STOP_ATR_MULT", 2.0))
            tgt_mult = float(getattr(self.settings, "TARGET_ATR_MULT", 3.0))
            stop_dist = atr_f * stop_mult
            price_f = float(price)
            max_risk = float(portfolio_value) * self.settings.MAX_PORTFOLIO_RISK_PCT
            max_lots_risk = int(max_risk / max(stop_dist, 0.01) / lot_size)
            lots = min(manual_lots, max_lots_risk) if max_lots_risk > 0 else manual_lots
            lots = max(lots, 1)  # at least 1 lot for manual
            sizing = {
                "lots": lots,
                "kelly_pct": 0,
                "position_pct": round(lots * lot_size * price_f / float(portfolio_value) * 100, 2),
                "impact": 0,
                "stop_distance": stop_dist,
                "target_distance": atr_f * tgt_mult,
                "position_value": round(lots * lot_size * price_f, 2),
            }
        else:
            # Extract futures margin / step_value metadata from the instrument
            # object resolved at step 1.  For shares these values are ignored;
            # for futures they drive capital-per-lot sizing (initial_margin)
            # and risk-per-lot conversion (rub_per_point = step_value / step).
            initial_margin = 0.0
            rub_per_point = 1.0
            if instrument_kind == "future":
                try:
                    meta = self.broker.extract_futures_metadata(instrument)
                    if direction == "sell":
                        im = meta.get("initial_margin_sell") or 0
                    else:
                        im = meta.get("initial_margin_buy") or 0
                    initial_margin = float(im)

                    # SDK v0.2.0b59 does not populate initial_margin_on_buy/sell
                    # on the Future object — they are always 0.  Fall back to
                    # dlong/dshort × current price (standard FORTS ГО formula).
                    if initial_margin <= 0:
                        d_key = "dshort" if direction == "sell" else "dlong"
                        d_ratio = float(meta.get(d_key) or 0)
                        if d_ratio > 0:
                            initial_margin = d_ratio * float(price)
                            logger.debug(
                                f"{ticker}: ГО computed from {d_key}={d_ratio:.4f} "
                                f"× price={price:.2f} = {initial_margin:.2f} RUB"
                            )

                    rpp = meta.get("rub_per_point") or 1
                    rub_per_point = float(rpp) if float(rpp) > 0 else 1.0
                    metrics["initial_margin"] = round(initial_margin, 2)
                    metrics["rub_per_point"] = round(rub_per_point, 4)
                except Exception as e:
                    logger.warning(f"Cannot extract futures metadata for {ticker}: {e}")

            sizing = self.position_sizer.compute_lots(
                portfolio_value=portfolio_value,
                price=price,
                lot_size=lot_size,
                daily_volume=avg_daily_volume,
                volatility=volatility,
                atr=atr,
                confidence=confidence,
                win_rate=stats["win_rate"],
                avg_win=stats["avg_win"],
                avg_loss=stats["avg_loss"],
                instrument_kind=instrument_kind,
                initial_margin=initial_margin,
                rub_per_point=rub_per_point,
                direction=direction,
            )

        lots = sizing["lots"]
        if lots <= 0:
            return TradeApproval(approved=False, reason="Position too small after risk adjustments")

        metrics.update(sizing)

        # 12. Leverage check
        if self.settings.USE_LEVERAGE:
            try:
                margin = await self.broker.get_margin_attributes()
                funds_level = float(quotation_to_decimal(margin.funds_sufficiency_level))
                if funds_level < 1.5:
                    return TradeApproval(
                        approved=False,
                        reason=f"Insufficient margin buffer: {funds_level:.2f} < 1.5",
                    )
                metrics["margin_level"] = round(funds_level, 2)
            except Exception as e:
                logger.warning(f"Margin check failed: {e}")

        # Compute stop/target prices from pre-trade price
        # (engine will recalculate from actual fill price)
        price_f = float(price)
        stop_dist = sizing["stop_distance"]
        target_dist = sizing["target_distance"]
        if direction == "buy":
            stop_price = price_f - stop_dist
            target_price = price_f + target_dist
        else:
            # Short: stop is above entry, target is below entry
            stop_price = price_f + stop_dist
            target_price = price_f - target_dist

        # Pass sizing metadata to engine for stop recalculation after fill
        metrics["stop_distance"] = stop_dist
        metrics["target_distance"] = target_dist
        metrics["price"] = price_f
        metrics["lot_size"] = lot_size

        # TWAP check
        use_twap = self.execution.should_use_twap(lots, lot_size, avg_daily_volume)
        metrics["use_twap"] = use_twap

        return TradeApproval(
            approved=True,
            lots=lots,
            stop_loss_price=round(stop_price, 4),
            take_profit_price=round(target_price, 4),
            use_twap=use_twap,
            reason="Approved",
            risk_metrics=metrics,
        )

    def update_daily_pnl(self, pnl_change: float):
        """Update daily P&L tracking."""
        self._daily_pnl += pnl_change
        self._trade_count_today += 1

    async def _check_daily_reset(self):
        """Reset daily counters at start of new trading day."""
        today = now_msk().date()
        if self._last_reset_date != today:
            self._daily_pnl = 0.0
            self._trade_count_today = 0
            self._last_reset_date = today
            logger.info("Daily risk counters reset")

    async def get_risk_status(self) -> dict:
        """Get current risk metrics for display.

        Pulls live portfolio value from the broker and today's realised +
        unrealised P&L from the DB, so the dashboard stays populated even in
        interactive mode (where ``approve_trade`` is never called) and
        survives a bot restart (in-memory counters reset to 0 on boot).
        """
        # Refresh portfolio from broker on every call — get_risk_status is
        # hit only when the user opens the risk screen, so the extra API
        # round-trip is fine.
        try:
            pv = await self.broker.get_portfolio_value()
            self._current_portfolio = float(pv)
            if self._peak_portfolio == 0 or self._current_portfolio > self._peak_portfolio:
                self._peak_portfolio = self._current_portfolio
        except Exception as e:
            logger.warning(f"get_risk_status: portfolio fetch failed: {e}")

        # Restore today's P&L from DB — the in-memory _daily_pnl counter
        # resets to 0 on every bot restart, but upsert_daily_pnl has the
        # persisted realised+unrealised breakdown.
        today_iso = now_msk().date().isoformat()
        realized = 0.0
        unrealized = 0.0
        trades_today = self._trade_count_today
        try:
            row = await self.db.get_daily_pnl_for_date(today_iso)
            if row:
                realized = float(row.get("realized_pnl") or 0.0)
                unrealized = float(row.get("unrealized_pnl") or 0.0)
                trades_today = int(row.get("total_trades") or trades_today)
        except Exception as e:
            logger.debug(f"get_risk_status: daily_pnl read failed: {e}")

        total_pnl = realized + unrealized
        # Keep in-memory counter in sync so approve_trade()'s daily-loss
        # circuit breaker uses the persisted value rather than 0 on restart.
        self._daily_pnl = realized

        drawdown = 0.0
        if self._peak_portfolio > 0:
            drawdown = (self._peak_portfolio - self._current_portfolio) / self._peak_portfolio

        return {
            "daily_pnl": round(total_pnl, 2),
            "daily_pnl_realized": round(realized, 2),
            "daily_pnl_unrealized": round(unrealized, 2),
            "daily_loss_limit": f"{self.settings.MAX_DAILY_LOSS_PCT:.0%}",
            "drawdown": f"{drawdown:.1%}",
            "drawdown_limit": f"{self.settings.MAX_DRAWDOWN_PCT:.0%}",
            "peak_portfolio": round(self._peak_portfolio, 2),
            "current_portfolio": round(self._current_portfolio, 2),
            "trades_today": trades_today,
        }
