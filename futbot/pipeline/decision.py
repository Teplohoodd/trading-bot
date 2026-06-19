"""Pipeline orchestrator.

Walks layers in order.  Each layer can either:
  * approve with a direction (continue chain),
  * pass through (keep prior direction),
  * VETO (chain stops, no trade).

Returns a `Decision` dict ready for risk audit + sizer.  Every layer's
result is included verbatim so the DB / Telegram can show *why* a trade
was rejected or approved.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from futbot.pipeline import trend, regime, setup as setup_layer, trigger, ml_gate
from futbot.pipeline.base import LayerResult

logger = logging.getLogger("futbot.decision")


@dataclass
class Decision:
    figi: str
    ticker: str
    direction: Optional[str] = None  # "buy" / "sell" / None
    approved: bool = False
    rejected_at: Optional[str] = None
    rejection_reason: Optional[str] = None
    layers: dict = field(default_factory=dict)  # layer_name → LayerResult.to_dict()

    @property
    def vote(self) -> int:
        if self.direction == "buy":
            return 1
        if self.direction == "sell":
            return -1
        return 0


async def evaluate_contract(
    *, figi: str, ticker: str, tf_data: dict, settings, base_ticker: str | None = None
) -> Decision:
    """Run the full pipeline on one contract.  tf_data must contain keys
    '1h', '15m', '5m' mapping to candle DataFrames.

    `base_ticker` (e.g. "BR" for BRM6) lets the ML gate look up the
    concept→model mapping.  When None, ML gate falls back to pass-through.
    """
    dec = Decision(figi=figi, ticker=ticker)

    # 1 — Trend on 1h
    r = trend.evaluate(tf_data.get("1h"), settings)
    dec.layers["trend"] = r.to_dict()
    if r.vetoes:
        dec.rejected_at = "trend"
        dec.rejection_reason = r.reason
        return dec
    vote = r.vote

    # 2 — Regime on 1h
    r = regime.evaluate(tf_data.get("1h"), trend_vote=vote, settings=settings)
    dec.layers["regime"] = r.to_dict()
    if r.vetoes:
        dec.rejected_at = "regime"
        dec.rejection_reason = r.reason
        return dec
    vote = r.vote  # regime may flip the direction in choppy mode

    # 3 — Setup on 15m
    r = setup_layer.evaluate(tf_data.get("15m"), prior_vote=vote, settings=settings)
    dec.layers["setup"] = r.to_dict()
    if r.vetoes:
        dec.rejected_at = "setup"
        dec.rejection_reason = r.reason
        return dec
    vote = r.vote

    # 4 — Trigger on 5m
    r = trigger.evaluate(tf_data.get("5m"), prior_vote=vote, settings=settings)
    dec.layers["trigger"] = r.to_dict()
    if r.vetoes:
        dec.rejected_at = "trigger"
        dec.rejection_reason = r.reason
        return dec
    vote = r.vote

    # 5 — ML gate (looks up the per-concept LGBM model if one's trained)
    r = ml_gate.evaluate(
        prior_vote=vote,
        settings=settings,
        df_1h=tf_data.get("1h"),
        base_ticker=base_ticker,
    )
    dec.layers["ml"] = r.to_dict()
    if r.vetoes:
        dec.rejected_at = "ml_gate"
        dec.rejection_reason = r.reason
        return dec
    vote = r.vote

    # All layers approved.
    dec.direction = "buy" if vote > 0 else "sell" if vote < 0 else None
    if dec.direction is None:
        dec.rejected_at = "ml_gate"  # nominally
        dec.rejection_reason = "vote collapsed to 0 after final layer"
        return dec
    dec.approved = True
    return dec
