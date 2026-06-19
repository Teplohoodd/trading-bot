"""ГО-aware position sizer.

Two caps applied in order, take the minimum:
  1. ГО budget: lots = floor(portfolio × MAX_GO_PCT / initial_margin_per_lot)
     — never blocks more than FUTBOT_MAX_GO_PCT of the account in margin.
  2. Vol target: lots = floor(portfolio × VOL_TARGET_DAILY_PCT /
                              expected_daily_pnl_vol_per_lot)
     — each position contributes a fixed share of portfolio daily vol.

Fallback when ГО is unknown (rare): FUTBOT_FALLBACK_LOTS (default 1).

Returns dict with `lots`, `initial_margin`, `rub_per_point`, `reason`.
"""

import logging
import math
from decimal import Decimal

logger = logging.getLogger("futbot.sizer")


def compute_lots(
    *,
    portfolio_value: Decimal,
    initial_margin: float,
    rub_per_point: float,
    atr_1h: float,
    lot_size: int,
    settings,
) -> dict:
    portfolio_f = float(portfolio_value)
    pct_go = float(settings.FUTBOT_MAX_GO_PCT)
    fallback = int(settings.FUTBOT_FALLBACK_LOTS)

    # Sanity
    if portfolio_f <= 0:
        return {"lots": 0, "reason": "non-positive portfolio"}
    if initial_margin is None or initial_margin <= 0:
        return {
            "lots": fallback,
            "reason": f"ГО unknown — using fallback {fallback} lot(s)",
            "initial_margin": None,
            "rub_per_point": rub_per_point,
        }

    # Cap 1 — ГО budget
    lots_by_go = math.floor(portfolio_f * pct_go / initial_margin)
    if lots_by_go < 1:
        return {
            "lots": 0,
            "reason": (
                f"ГО {initial_margin:.0f}₽/lot > budget "
                f"({portfolio_f*pct_go:.0f}₽); cannot afford 1 lot"
            ),
            "initial_margin": initial_margin,
            "rub_per_point": rub_per_point,
        }

    # Cap 2 — vol target.  atr_1h is in price points; expected daily P&L 1-σ
    # per lot ≈ ATR_1h × rub_per_point × √24 (24 hourly bars/day) × lot_size.
    # The lot_size term is included because rub_per_point is per-point
    # per-contract, and one futures lot covers `lot_size` underlying units
    # (e.g. Si lot_size=1000 — 1000 USD).  For most FORTS contracts lot_size
    # multiplies straight through rub_per_point so this is conservative.
    if atr_1h and atr_1h > 0 and rub_per_point and rub_per_point > 0:
        daily_pnl_vol_rub = atr_1h * rub_per_point * math.sqrt(24)
        target_pnl_vol_rub = portfolio_f * float(settings.FUTBOT_VOL_TARGET_DAILY_PCT)
        if daily_pnl_vol_rub > 0:
            lots_by_vol = math.floor(target_pnl_vol_rub / daily_pnl_vol_rub)
            if lots_by_vol < 1:
                # Vol target says less than 1 lot — go with 1 (minimum
                # tradeable) but log the warning.  This is the "high-vol
                # contract on small portfolio" case.
                lots_by_vol = 1
        else:
            lots_by_vol = lots_by_go
    else:
        lots_by_vol = lots_by_go  # no ATR → don't apply vol cap

    lots = min(lots_by_go, lots_by_vol)
    return {
        "lots": int(lots),
        "lots_by_go": int(lots_by_go),
        "lots_by_vol": int(lots_by_vol),
        "initial_margin": float(initial_margin),
        "rub_per_point": float(rub_per_point),
        "reason": (
            f"min(ГО-cap={lots_by_go}, vol-cap={lots_by_vol}) = {lots}; "
            f"ГО {initial_margin:.0f}₽/lot, ATR_1h={atr_1h:.2f}"
        ),
    }
