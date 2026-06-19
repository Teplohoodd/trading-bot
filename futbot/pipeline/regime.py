"""Layer 2 — regime classifier (1h).

Classifies the current market into one of:
  * trending     — rolling-50 linear regression R² > 0.55
  * choppy       — R² ≤ 0.55 and ATR ≤ 2× 30-day median
  * vol_spike    — ATR > 2× 30-day median (regardless of R²)

Behaviour:
  * vol_spike VETOES — too dangerous to trade through (gap risk, stop-runs).
  * trending  passes through and HINTS the trend direction (carries Layer-1
    vote).
  * choppy    passes through but FLIPS the direction (mean-reversion mode).

The flip is what makes the pipeline regime-adaptive — same setup signal
can produce a long or short depending on whether we're trend-following
or fading.
"""

import logging
import numpy as np
import pandas as pd

from futbot.pipeline.base import LayerResult

logger = logging.getLogger("futbot.layer.regime")


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def _rolling_r2(close: pd.Series, n: int = 50) -> float:
    """R² of a linear regression on the last n closes.  Returns 0 on failure."""
    y = close.tail(n).values
    if len(y) < n:
        return 0.0
    x = np.arange(len(y))
    if np.std(y) == 0:
        return 0.0
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def evaluate(df_1h: pd.DataFrame, trend_vote: int, settings) -> LayerResult:
    if df_1h is None or df_1h.empty or len(df_1h) < 60:
        return LayerResult(
            layer="regime",
            vote=0,
            vetoes=True,
            reason="not enough 1h bars for regime classification",
        )

    atr = _atr(df_1h)
    atr_now = float(atr.iloc[-1])
    lookback = int(settings.FUTBOT_REGIME_ATR_LOOKBACK)
    atr_median = float(atr.tail(min(lookback, len(atr))).median())
    if atr_median <= 0:
        return LayerResult(
            layer="regime",
            vote=0,
            vetoes=True,
            reason="median ATR == 0 (stale data?)",
        )
    atr_ratio = atr_now / atr_median

    r2 = _rolling_r2(df_1h["close"], n=50)

    detail = {
        "atr_now": round(atr_now, 4),
        "atr_median": round(atr_median, 4),
        "atr_ratio": round(atr_ratio, 3),
        "r2_50": round(r2, 3),
    }

    spike_mult = float(settings.FUTBOT_REGIME_VOL_SPIKE_MULT)
    if atr_ratio > spike_mult:
        return LayerResult(
            layer="regime",
            vote=0,
            vetoes=True,
            reason=f"vol spike (ATR {atr_ratio:.1f}× median > {spike_mult})",
            detail={**detail, "classification": "vol_spike"},
        )

    trending_thr = float(settings.FUTBOT_REGIME_TRENDING_R2_MIN)
    if r2 >= trending_thr:
        # Trending — keep the trend-layer direction.
        return LayerResult(
            layer="regime",
            vote=trend_vote,
            vetoes=False,
            reason=f"trending (R²={r2:.2f}); follow trend",
            detail={**detail, "classification": "trending"},
        )

    # Choppy — flip direction: fade strength, buy weakness.
    flipped = -trend_vote if trend_vote != 0 else 0
    return LayerResult(
        layer="regime",
        vote=flipped,
        vetoes=False,
        reason=f"choppy (R²={r2:.2f}); mean-reversion (vote flipped to {flipped})",
        detail={**detail, "classification": "choppy"},
    )
