"""Layer 1 — trend (1h).

Aligned EMA cross + ADX strength.  Trades only when:
  * EMA_fast > EMA_slow AND ADX >= min  → vote +1 (long bias)
  * EMA_fast < EMA_slow AND ADX >= min  → vote −1 (short bias)
  * otherwise → vote 0, VETOES the trade

ADX gate avoids the textbook EMA-cross trap in choppy markets where every
2-week MA cross is followed by a reversion.  ADX < 18 means "no trend";
the chain stops here in that case.
"""

import logging
import numpy as np
import pandas as pd

from futbot.pipeline.base import LayerResult

logger = logging.getLogger("futbot.layer.trend")


def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Manual ADX (pandas-ta isn't on Python 3.11).  Wilder smoothing."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def evaluate(df_1h: pd.DataFrame, settings) -> LayerResult:
    n_fast = int(settings.FUTBOT_TREND_EMA_FAST)
    n_slow = int(settings.FUTBOT_TREND_EMA_SLOW)
    adx_min = float(settings.FUTBOT_TREND_ADX_MIN)

    if df_1h is None or df_1h.empty or len(df_1h) < n_slow + 14:
        return LayerResult(
            layer="trend",
            vote=0,
            vetoes=True,
            reason=f"not enough 1h bars ({0 if df_1h is None else len(df_1h)} < {n_slow + 14})",
        )

    close = df_1h["close"]
    ema_fast = _ema(close, n_fast).iloc[-1]
    ema_slow = _ema(close, n_slow).iloc[-1]
    adx_series = _adx(df_1h)
    adx = float(adx_series.iloc[-1]) if not pd.isna(adx_series.iloc[-1]) else 0.0

    detail = {
        "ema_fast": round(float(ema_fast), 4),
        "ema_slow": round(float(ema_slow), 4),
        "adx": round(adx, 2),
        "close": round(float(close.iloc[-1]), 4),
    }

    if adx < adx_min:
        return LayerResult(
            layer="trend",
            vote=0,
            vetoes=True,
            reason=f"weak trend (ADX={adx:.1f} < {adx_min})",
            detail=detail,
        )

    if ema_fast > ema_slow:
        return LayerResult(
            layer="trend",
            vote=1,
            vetoes=False,
            reason=f"uptrend (EMA{n_fast}>{n_slow}, ADX={adx:.1f})",
            detail=detail,
        )
    if ema_fast < ema_slow:
        return LayerResult(
            layer="trend",
            vote=-1,
            vetoes=False,
            reason=f"downtrend (EMA{n_fast}<{n_slow}, ADX={adx:.1f})",
            detail=detail,
        )
    return LayerResult(
        layer="trend",
        vote=0,
        vetoes=True,
        reason="EMAs flat (no edge)",
        detail=detail,
    )
