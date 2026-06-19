"""Feature engineering for the ML gate.

Inputs:  daily OHLCV DataFrame (time, open, high, low, close, volume).
Outputs: same DataFrame with engineered feature columns appended.

Feature set (25 features grouped by what they capture).  The goal is to
give the model enough signal without bloating with redundant indicators;
LightGBM handles correlated features fine, but on ~5-7k daily rows more
features = more overfit risk.  Each group below has 3-5 features that
each measure a *different* aspect of the same concept:

1. Multi-horizon returns                   ret_{1,2,3,5,10,20}      (6)
2. Volatility regime                       atr_pct, atr_pct_change_5, bb_width_pct, range_z_20  (4)
3. Trend / mean-reversion oscillators      rsi_14, rsi_14_change_5, bb_pct_b, adx_14            (4)
4. Trend deviation                         ema_fast_minus_slow_pct, close_vs_ema20_pct          (2)
5. Structure / position-in-range           hl_range_pct, dist_from_high_20_pct, dist_from_low_20_pct, r2_50  (4)
6. Higher-order distributional stats       ret_skew_20, ret_kurt_20                             (2)
7. Streaks                                 consec_up_bars                                       (1)
8. Volume                                  vol_z_20                                             (1)
9. Calendar                                dow, month, week_of_year                             (3)

NaN rows are dropped at the end so the trainer can use them directly.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("futbot.ml.features")


FEATURE_NAMES = [
    # 1. multi-horizon log returns
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    # 2. vol regime
    "atr_pct",
    "atr_pct_change_5",
    "bb_width_pct",
    "range_z_20",
    # 3. oscillators
    "rsi_14",
    "rsi_14_change_5",
    "bb_pct_b",
    "adx_14",
    # 4. trend deviation
    "ema_fast_minus_slow_pct",
    "close_vs_ema20_pct",
    # 5. structure
    "hl_range_pct",
    "dist_from_high_20_pct",
    "dist_from_low_20_pct",
    "r2_50",
    # 6. higher moments
    "ret_skew_20",
    "ret_kurt_20",
    # 7. streaks
    "consec_up_bars",
    # 8. volume
    "vol_z_20",
    # 9. calendar
    "dow",
    "month",
    "week_of_year",
]


# ── Indicator primitives ─────────────────────────────────────────────────────
def _wilder_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = _wilder_ema(up, n)
    avg_dn = _wilder_ema(down, n).replace(0, np.nan)
    rs = avg_up / avg_dn
    return 100 - 100 / (1 + rs)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return _wilder_ema(tr, n)


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    down = -l.diff()
    plus_dm = ((up > down) & (up > 0)).astype(float) * up.clip(lower=0)
    minus_dm = ((down > up) & (down > 0)).astype(float) * down.clip(lower=0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder_ema(tr, n).replace(0, np.nan)
    plus_di = 100 * _wilder_ema(plus_dm, n) / atr
    minus_di = 100 * _wilder_ema(minus_dm, n) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return _wilder_ema(dx.fillna(0), n)


def _rolling_r2(close: pd.Series, n: int = 50) -> pd.Series:
    """Rolling R² of a linear regression — how trendy is the last n bars?"""

    def _r2(window):
        if len(window) < n or np.std(window) == 0:
            return np.nan
        x = np.arange(len(window))
        slope, intercept = np.polyfit(x, window, 1)
        yhat = slope * x + intercept
        ss_res = np.sum((window - yhat) ** 2)
        ss_tot = np.sum((window - window.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return close.rolling(n).apply(_r2, raw=True)


def _consec_up_bars(close: pd.Series, max_n: int = 10) -> pd.Series:
    """Number of consecutive UP closes ending at bar t.  Negative if last
    bar(s) were DOWN.  Capped at ±max_n to keep distribution bounded."""
    up = (close.diff() > 0).astype(int).values
    n = len(up)
    out = np.zeros(n, dtype=int)
    streak = 0
    for i in range(n):
        if up[i] == 1:
            streak = max(streak, 0) + 1
        else:
            streak = min(streak, 0) - 1
        out[i] = max(min(streak, max_n), -max_n)
    return pd.Series(out, index=close.index)


# ── Public API ───────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, *, dropna: bool = True) -> pd.DataFrame:
    """Append FEATURE_NAMES columns.  Caller passes sorted daily OHLCV
    DataFrame.  Returns a NEW DataFrame (does not mutate input)."""
    if df.empty:
        return df.copy()
    out = df.copy().reset_index(drop=True)
    close = out["close"]
    log_close = np.log(close.replace(0, np.nan))

    # ── 1. Multi-horizon log returns (MOMENTUM signal at different scales) ─
    out["ret_1"] = log_close.diff(1)
    out["ret_2"] = log_close.diff(2)
    out["ret_3"] = log_close.diff(3)
    out["ret_5"] = log_close.diff(5)
    out["ret_10"] = log_close.diff(10)
    out["ret_20"] = log_close.diff(20)

    # ── 2. Volatility regime ───────────────────────────────────────────────
    atr = _atr(out, 14)
    out["atr_pct"] = atr / close.replace(0, np.nan) * 100
    # Acceleration of vol: change in ATR% over last 5 bars
    out["atr_pct_change_5"] = out["atr_pct"].diff(5)
    # Bollinger band width (a different vol measure — sd-based, not range-based)
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    out["bb_width_pct"] = (4 * sd20) / ma20.replace(0, np.nan) * 100
    # Today's range z-score against last 20 days
    rng = out["high"] - out["low"]
    rng_mean = rng.rolling(20).mean()
    rng_std = rng.rolling(20).std().replace(0, np.nan)
    out["range_z_20"] = (rng - rng_mean) / rng_std

    # ── 3. Trend / mean-reversion oscillators ─────────────────────────────
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_14_change_5"] = out["rsi_14"].diff(5)
    out["bb_pct_b"] = (close - (ma20 - 2 * sd20)) / (4 * sd20).replace(0, np.nan)
    out["adx_14"] = _adx(out, 14)

    # ── 4. Trend deviation ─────────────────────────────────────────────────
    ema_f = close.ewm(span=20, adjust=False).mean()
    ema_s = close.ewm(span=50, adjust=False).mean()
    out["ema_fast_minus_slow_pct"] = (ema_f - ema_s) / close.replace(0, np.nan) * 100
    out["close_vs_ema20_pct"] = (close - ema_f) / ema_f.replace(0, np.nan) * 100

    # ── 5. Structure / position in recent range ────────────────────────────
    out["hl_range_pct"] = rng / close.replace(0, np.nan) * 100
    rolling_high20 = out["high"].rolling(20).max()
    rolling_low20 = out["low"].rolling(20).min()
    out["dist_from_high_20_pct"] = (
        (close - rolling_high20) / rolling_high20.replace(0, np.nan) * 100
    )
    out["dist_from_low_20_pct"] = (close - rolling_low20) / rolling_low20.replace(0, np.nan) * 100
    out["r2_50"] = _rolling_r2(close, 50)

    # ── 6. Higher-order distributional stats ───────────────────────────────
    out["ret_skew_20"] = out["ret_1"].rolling(20).skew()
    out["ret_kurt_20"] = out["ret_1"].rolling(20).kurt()

    # ── 7. Streaks ─────────────────────────────────────────────────────────
    out["consec_up_bars"] = _consec_up_bars(close, max_n=10)

    # ── 8. Volume ──────────────────────────────────────────────────────────
    vol_mean = out["volume"].rolling(20).mean()
    vol_std = out["volume"].rolling(20).std().replace(0, np.nan)
    out["vol_z_20"] = (out["volume"] - vol_mean) / vol_std

    # ── 9. Calendar ────────────────────────────────────────────────────────
    out["dow"] = out["time"].dt.dayofweek
    out["month"] = out["time"].dt.month
    out["week_of_year"] = out["time"].dt.isocalendar().week.astype(int)

    if dropna:
        out = out.dropna(subset=FEATURE_NAMES).reset_index(drop=True)
    return out
