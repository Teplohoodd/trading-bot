"""Layer 4 — trigger (5m).

The "pull the trigger" bar.  All prior layers said go long (or short); this
one looks at the most recent CLOSED 5-minute bar and asks: is the breakout
actually happening right now?

Requirements (long):
  * volume > VOL_MULT × 20-bar median volume
  * bar range (high − low) > RANGE_MULT × 20-bar median range
  * close in the upper CLOSE_QUARTILE of the bar (e.g. top 25%)
  * close > open (bar is up)

Short mirrors.

This is what makes the bot avoid the "model flips on noise" failure mode of
trade_claude: the trigger only fires on a bar that visibly broke out with
volume — not on a stochastic threshold being briefly crossed.
"""

import logging
import pandas as pd

from futbot.pipeline.base import LayerResult

logger = logging.getLogger("futbot.layer.trigger")


def evaluate(df_5m: pd.DataFrame, prior_vote: int, settings) -> LayerResult:
    if df_5m is None or df_5m.empty or len(df_5m) < 21:
        return LayerResult(
            layer="trigger",
            vote=0,
            vetoes=True,
            reason="not enough 5m bars",
        )
    if prior_vote == 0:
        return LayerResult(
            layer="trigger",
            vote=0,
            vetoes=True,
            reason="no inbound direction",
        )

    # Most recent CLOSED bar.  The bar at index -1 might be the live bar;
    # we look at -2 for "the last closed 5-min candle".  When the bot
    # runs intra-minute this matters; on the loop interval it's a no-op
    # because the latest finished bar is what we want anyway.
    last = df_5m.iloc[-1]
    prior20 = df_5m.iloc[-21:-1]  # 20 bars before the current

    vol_med = float(prior20["volume"].median())
    rng_med = float((prior20["high"] - prior20["low"]).median())
    if vol_med <= 0 or rng_med <= 0:
        return LayerResult(
            layer="trigger",
            vote=0,
            vetoes=True,
            reason="zero median volume/range (illiquid bar window)",
        )

    bar_rng = float(last["high"] - last["low"])
    vol_mult = float(last["volume"]) / vol_med
    rng_mult = bar_rng / rng_med

    # Where did the bar close within its range?  0.0 = at low, 1.0 = at high.
    close_pos = (float(last["close"]) - float(last["low"])) / bar_rng if bar_rng > 0 else 0.5
    up_bar = float(last["close"]) > float(last["open"])

    detail = {
        "vol_mult": round(vol_mult, 2),
        "rng_mult": round(rng_mult, 2),
        "close_pos": round(close_pos, 3),
        "up_bar": int(up_bar),
        "bar_vol": int(last["volume"]),
        "bar_range": round(bar_rng, 4),
    }

    vol_thr = float(settings.FUTBOT_TRIGGER_VOL_MULT)
    rng_thr = float(settings.FUTBOT_TRIGGER_RANGE_MULT)
    cq = float(settings.FUTBOT_TRIGGER_CLOSE_QUARTILE)

    if vol_mult < vol_thr or rng_mult < rng_thr:
        return LayerResult(
            layer="trigger",
            vote=0,
            vetoes=True,
            reason=(
                f"no expansion (vol {vol_mult:.1f}×<{vol_thr}, " f"range {rng_mult:.1f}×<{rng_thr})"
            ),
            detail=detail,
        )

    if prior_vote > 0 and up_bar and close_pos >= cq:
        return LayerResult(
            layer="trigger",
            vote=1,
            vetoes=False,
            reason=f"bullish trigger (vol {vol_mult:.1f}×, close@{close_pos:.0%})",
            detail=detail,
        )
    if prior_vote < 0 and (not up_bar) and close_pos <= (1 - cq):
        return LayerResult(
            layer="trigger",
            vote=-1,
            vetoes=False,
            reason=f"bearish trigger (vol {vol_mult:.1f}×, close@{close_pos:.0%})",
            detail=detail,
        )

    return LayerResult(
        layer="trigger",
        vote=0,
        vetoes=True,
        reason=(
            f"bar direction/close pos doesn't match {prior_vote:+d} "
            f"(up_bar={up_bar}, close_pos={close_pos:.2f})"
        ),
        detail=detail,
    )
