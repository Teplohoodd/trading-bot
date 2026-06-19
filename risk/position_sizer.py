"""Position sizing: Kyle-Obizhaeva market impact + fractional Kelly criterion."""

import logging
import math
from decimal import Decimal

from config.settings import Settings

logger = logging.getLogger(__name__)


class PositionSizer:
    """Calculates optimal position size considering impact and Kelly."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Compute fractional Kelly criterion.

        f* = (p * b - q) / b, where p=win_rate, q=1-p, b=avg_win/avg_loss.
        Apply KELLY_FRACTION (default 1/4).
        """
        fallback = getattr(self.settings, "KELLY_FALLBACK_PCT", 0.03)

        if avg_loss <= 0 or win_rate <= 0:
            return fallback  # Cold-start: no trade history

        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate
        q = 1 - p

        kelly = (p * b - q) / b
        kelly = max(kelly, 0)  # Never negative

        # Apply fraction
        adjusted = kelly * self.settings.KELLY_FRACTION
        # Clamp to reasonable range — never go below the fallback floor even
        # when raw Kelly is 0 (that's the exact condition the fallback exists
        # for: default win_rate=0.5, avg_win=avg_loss=1.0 → Kelly=0).
        return min(max(adjusted, fallback), self.settings.MAX_POSITION_PCT)

    def kyle_obizhaeva_impact(
        self, order_lots: int, lot_size: int, daily_volume: float, volatility: float
    ) -> float:
        """Estimate market impact using square-root model.

        impact = sigma * C * sqrt(Q / V)

        Args:
            order_lots: Number of lots to trade.
            lot_size: Shares per lot.
            daily_volume: Average daily volume in shares.
            volatility: Daily volatility (as fraction, e.g., 0.02 for 2%).

        Returns:
            Estimated price impact as fraction (e.g., 0.005 for 0.5%).
        """
        if daily_volume <= 0:
            return 1.0  # Max impact if no volume data

        order_shares = order_lots * lot_size
        participation_rate = order_shares / daily_volume
        C = 1.0  # Calibration constant (conservative)
        impact = volatility * C * math.sqrt(participation_rate)
        return impact

    def compute_lots(
        self,
        portfolio_value: Decimal,
        price: Decimal,
        lot_size: int,
        daily_volume: float,
        volatility: float,
        atr: Decimal,
        confidence: float,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        regime_scale: float = 1.0,
        instrument_kind: str = "share",
        initial_margin: float = 0.0,
        rub_per_point: float = 1.0,
        direction: str = "buy",
    ) -> dict:
        """Compute optimal position size in lots.

        For shares (default path, unchanged):
          - capital-per-lot = price × lot_size (you block the full notional)
          - stop_distance is measured in RUB/share (ATR × 2)

        For futures (instrument_kind="future"):
          - capital-per-lot = initial_margin (ГО) — the RUB blocked on the
            account per contract.  Using price × lot_size would conflate the
            contract NOTIONAL (50-100k RUB on Si) with the capital REQUIREMENT
            (5-10k RUB ГО), zeroing out the sizer 100 % of the time.  This was
            the root cause of "no futures trades ever executed" (22-Apr-2026
            audit).
          - stop_distance is in price POINTS, so risk-per-lot = stop_distance
            × rub_per_point × lot_size.  For Si (1 point = 1 RUB) this equals
            the legacy formula; for BR/GD (where min_price_increment < 1 or
            step_value ≠ 1) it diverges — the futures path uses the right
            conversion.
          - concentration cap uses ``FUTURES_MAX_POSITION_PCT`` (default 10%)
            instead of ``MAX_POSITION_PCT`` (20%).  Lower cap reflects higher
            leverage — 10 % of portfolio in ГО ≈ 60-100 % in notional.

        Args:
            instrument_kind: "share" or "future".
            initial_margin: ГО in RUB per contract (futures only; ignored for shares).
            rub_per_point: step_value / min_price_increment (futures only; ignored for shares).
            direction: "buy" or "sell".  Futures have separate ГО for long/short;
                caller should pass the right one via initial_margin.

        Returns dict with lots, stop_distance (in price units, unchanged semantic
        for the engine's ATR-based stop math), impact, kelly_pct, etc.
        """
        price_f = float(price)
        portfolio_f = float(portfolio_value)
        atr_f = float(atr) if atr else price_f * 0.02
        is_future = instrument_kind == "future"

        if price_f <= 0 or portfolio_f <= 0 or lot_size <= 0:
            return {"lots": 0, "reason": "invalid inputs", "instrument_kind": instrument_kind}

        # --- Capital per lot & concentration cap ------------------------------
        if is_future:
            if initial_margin <= 0:
                # Margin metadata missing — refuse to size rather than guess.
                # Risk manager should log this and fall back to a share-style
                # approximation only if the operator explicitly opts in.
                return {
                    "lots": 0,
                    "reason": "futures initial_margin unknown — cannot size safely",
                    "instrument_kind": instrument_kind,
                }
            capital_per_lot = float(initial_margin)
            max_position_pct = getattr(
                self.settings,
                "FUTURES_MAX_POSITION_PCT",
                self.settings.MAX_POSITION_PCT,
            )
            rub_per_unit = float(rub_per_point) if rub_per_point and rub_per_point > 0 else 1.0
        else:
            capital_per_lot = lot_size * price_f  # shares: block the notional
            max_position_pct = self.settings.MAX_POSITION_PCT
            rub_per_unit = 1.0  # price already in RUB/share

        # 1. Kelly fraction.  Kelly returns a fraction of portfolio; cap it to
        # the instrument-specific concentration so futures don't blow past 10 %.
        kelly_pct = self.kelly_fraction(win_rate, avg_win, avg_loss)
        kelly_pct = min(kelly_pct, max_position_pct)

        # Scale by confidence and regime
        position_pct = kelly_pct * confidence * regime_scale

        # 2. Convert desired capital to lots via capital-per-lot
        position_value = portfolio_f * position_pct
        lots = int(position_value / capital_per_lot) if capital_per_lot > 0 else 0

        # 3. Risk-per-trade check.  stop_distance is in *price units* (pts for
        # futures, RUB for shares); convert to RUB via rub_per_unit × lot_size.
        stop_mult = float(getattr(self.settings, "STOP_ATR_MULT", 2.0))
        stop_distance = atr_f * stop_mult
        max_lots_by_risk = lots  # fallback: unconstrained
        if stop_distance > 0:
            risk_pct = (
                float(getattr(self.settings, "FUTURES_MAX_PORTFOLIO_RISK_PCT", 0.02))
                if is_future
                else float(self.settings.MAX_PORTFOLIO_RISK_PCT)
            )
            max_risk = portfolio_f * risk_pct
            risk_per_lot = stop_distance * rub_per_unit * lot_size
            if risk_per_lot > 0:
                max_lots_by_risk = int(max_risk / risk_per_lot)
                lots = min(lots, max_lots_by_risk)

        # 3b. Volatility-targeting cap.  Frazzini-Pedersen / Moskowitz: scale
        # so each position contributes a fixed % daily P&L volatility,
        # regardless of instrument vol regime.  Quiet names get larger
        # positions, vol-spike names get smaller — the opposite of pure
        # Kelly when win/loss stats are pooled.
        vt_enabled = bool(getattr(self.settings, "VOL_TARGET_ENABLED", False))
        if vt_enabled and not is_future:
            target_pct = float(getattr(self.settings, "VOL_TARGET_DAILY_PCT", 0.5)) / 100.0
            min_vol = float(getattr(self.settings, "VOL_TARGET_MIN_DAILY_VOL", 0.005))
            inst_vol = max(float(volatility) if volatility else 0.0, min_vol)
            # Lot's daily P&L vol in RUB ≈ price × lot_size × inst_vol
            lot_pnl_vol_rub = price_f * lot_size * inst_vol
            if lot_pnl_vol_rub > 0:
                target_pnl_vol_rub = portfolio_f * target_pct
                vol_target_lots = int(target_pnl_vol_rub / lot_pnl_vol_rub)
                if vol_target_lots >= 0 and vol_target_lots < lots:
                    logger.debug(
                        f"vol-target cap: {lots} → {vol_target_lots} lots "
                        f"(inst_vol={inst_vol:.4f}, target_pct={target_pct*100:.2f}%)"
                    )
                    lots = vol_target_lots

        # 1-lot floor: when Kelly gives 0 lots but the risk budget allows at
        # least 1 lot, use 1 lot.  This prevents "Position too small" from
        # blocking every trade when no trade history exists yet (Kelly → 0)
        # or the portfolio is small relative to lot price.  The floor is only
        # applied when the single lot also fits within the concentration limit.
        if lots == 0 and max_lots_by_risk >= 1:
            if capital_per_lot / portfolio_f <= max_position_pct:
                lots = 1
                logger.debug(
                    f"Kelly 0-lot → 1-lot floor applied "
                    f"(kelly_pct={kelly_pct:.3%}, max_by_risk={max_lots_by_risk}, "
                    f"kind={instrument_kind})"
                )

        # 4. Impact check (Kyle-Obizhaeva): reduce if impact > 0.5%
        max_impact = 0.005
        while lots > 0:
            impact = self.kyle_obizhaeva_impact(lots, lot_size, daily_volume, volatility)
            if impact <= max_impact:
                break
            lots = max(int(lots * 0.8), lots - 1)

        # 5. Concentration check — measured in blocked capital, not notional.
        capital_used = lots * capital_per_lot
        if portfolio_f > 0 and capital_used / portfolio_f > max_position_pct:
            lots = int(portfolio_f * max_position_pct / capital_per_lot)

        # Ensure at least 0
        lots = max(lots, 0)

        # Compute actual impact
        actual_impact = (
            self.kyle_obizhaeva_impact(lots, lot_size, daily_volume, volatility) if lots > 0 else 0
        )

        # Stop and target prices (always in price units — the engine multiplies
        # by price so the unit stays consistent).  Multipliers from settings
        # (defaults 4.0 / 2.0; legacy was hardcoded 2.0 / 3.0).
        target_mult = float(getattr(self.settings, "TARGET_ATR_MULT", 3.0))
        stop_distance_price = Decimal(str(atr_f * stop_mult))
        target_distance_price = Decimal(str(atr_f * target_mult))

        return {
            "lots": lots,
            "kelly_pct": round(kelly_pct * 100, 2),
            "position_pct": round(position_pct * 100, 2),
            "impact": round(actual_impact * 100, 3),
            "stop_distance": float(stop_distance_price),
            "target_distance": float(target_distance_price),
            "position_value": round(lots * capital_per_lot, 2),
            "capital_per_lot": round(capital_per_lot, 2),
            "rub_per_point": round(rub_per_unit, 4),
            "instrument_kind": instrument_kind,
        }
