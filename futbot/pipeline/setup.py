"""Layer 3 — setup (15m).

KDJ stochastic oscillator + Bollinger %B.  Confirms the regime-adjusted
direction by requiring price to be in an oversold zone for a long (or
overbought for a short).  This is the "wait for pullback" gate — even when
the trend is up and the regime says follow it, we don't chase price; we wait
for it to dip into a buy zone.

A long setup requires:
  * KDJ %K < BUY_BELOW  (default 25) — stochastic oversold
  * AND Bollinger %B < BUY_BELOW  (default 0.2) — near or below lower band

Short setup mirrors:
  * KDJ %K > SELL_ABOVE  (default 75)
  * AND %B > SELL_ABOVE  (default 0.8)

If neither, no setup → VETOES.
"""

import logging
import numpy as np
import pandas as pd

from futbot.pipeline.base import LayerResult

logger = logging.getLogger("futbot.layer.setup")


def _stoch_kdj(df: pd.DataFrame, n: int = 9, k_smooth: int = 3) -> pd.Series:
    """Classic %K of KDJ.  We use just %K for simplicity (%D and J are
    smoothings on top); zone-based decisions don't need the full set."""
    low_n = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    rsv = 100 * (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)
    return rsv.ewm(alpha=1 / k_smooth, adjust=False).mean()


def _bb_percent_b(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.Series:
    ma = df["close"].rolling(n).mean()
    sd = df["close"].rolling(n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    return (df["close"] - lower) / (upper - lower).replace(0, np.nan)


def evaluate(df_15m: pd.DataFrame, prior_vote: int, settings) -> LayerResult:
    if df_15m is None or df_15m.empty or len(df_15m) < 30:
        return LayerResult(
            layer="setup",
            vote=0,
            vetoes=True,
            reason="not enough 15m bars",
        )

    if prior_vote == 0:
        # Earlier layer already said hold; we don't synthesise a new direction.
        return LayerResult(
            layer="setup",
            vote=0,
            vetoes=True,
            reason="no inbound direction to confirm",
        )

    k_pct = float(_stoch_kdj(df_15m).iloc[-1])
    bb_b = float(_bb_percent_b(df_15m).iloc[-1])
    if pd.isna(k_pct) or pd.isna(bb_b):
        return LayerResult(
            layer="setup",
            vote=0,
            vetoes=True,
            reason="indicators NaN (warm-up)",
        )

    detail = {"kdj_k": round(k_pct, 2), "bb_b": round(bb_b, 3)}

    buy_kdj = float(settings.FUTBOT_SETUP_KDJ_BUY_BELOW)
    sell_kdj = float(settings.FUTBOT_SETUP_KDJ_SELL_ABOVE)
    buy_bb = float(settings.FUTBOT_SETUP_BB_BUY_BELOW)
    sell_bb = float(settings.FUTBOT_SETUP_BB_SELL_ABOVE)

    long_setup = (k_pct < buy_kdj) and (bb_b < buy_bb)
    short_setup = (k_pct > sell_kdj) and (bb_b > sell_bb)

    if prior_vote > 0 and long_setup:
        return LayerResult(
            layer="setup",
            vote=1,
            vetoes=False,
            reason=f"long setup (KDJ {k_pct:.1f}<{buy_kdj}, %B {bb_b:.2f}<{buy_bb})",
            detail=detail,
        )
    if prior_vote < 0 and short_setup:
        return LayerResult(
            layer="setup",
            vote=-1,
            vetoes=False,
            reason=f"short setup (KDJ {k_pct:.1f}>{sell_kdj}, %B {bb_b:.2f}>{sell_bb})",
            detail=detail,
        )

    return LayerResult(
        layer="setup",
        vote=0,
        vetoes=True,
        reason=(
            f"no setup zone (KDJ={k_pct:.1f}, %B={bb_b:.2f}, "
            f"need {'buy' if prior_vote > 0 else 'sell'} zone)"
        ),
        detail=detail,
    )
