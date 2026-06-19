"""Fractional differentiation (López de Prado, AFML ch.5).

Why we need this:
  Price series are non-stationary (random-walk-like).  Fitting a model on
  raw prices is brittle — train/test distribution drift kills generalisation.
  The classical fix is first differences (d=1, i.e. returns), which IS
  stationary but destroys ALL long-memory information about the price level
  (no memory of where we are in the trend).

  Fractional differentiation with d ∈ (0, 1) hits the sweet spot: makes the
  series stationary while preserving long memory.  d ≈ 0.4-0.5 is a common
  value for daily equity prices (Sephton 1991, LdP 2018).  Below the threshold
  d_opt the series is non-stationary (ADF p > 0.05); above it the series is
  over-differenced (memory wiped out, correlation with raw price drops < 0.5).

Implementation: Fixed-Width FractDiff (FFD), LdP snippet 5.3.  Weights drop
geometrically; we truncate when |w_k| < thres.  Result has constant lag
(unlike expanding-window which gives variable lookback).

Usage:
    from analysis.frac_diff import frac_diff_ffd
    df["close_fd"] = frac_diff_ffd(df["close"], d=0.4)

The first ``len(w) - 1`` rows are NaN by construction — the rolling window
needs that many lookback bars.  Drop or fillna(0) before feeding to a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_weights_ffd(d: float, thres: float = 1e-4, max_size: int = 10000) -> np.ndarray:
    """Compute fractional-differentiation weights up to ``thres`` cutoff.

    Args:
        d: differentiation order, typically 0.0 < d < 1.0.
        thres: stop adding weights once |w_k| < thres.  Default 1e-4 gives
            ~30 bars for d=0.4 — small enough to avoid eating too many rows
            but large enough to capture meaningful memory.
        max_size: hard cap to prevent runaway loops on degenerate inputs.

    Returns:
        np.ndarray of weights, oldest-first (so np.dot(w, slice) is the
        right-aligned convolution at the latest bar).
    """
    if d < 0:
        raise ValueError("d must be non-negative (use d=1 for first diff, d=0 for identity)")
    w = [1.0]
    for k in range(1, max_size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
    return np.array(w[::-1], dtype=float)


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 1e-4) -> pd.Series:
    """Apply fixed-width fractional differentiation to a series.

    The first ``len(w) - 1`` outputs are NaN — the FFD kernel needs
    that much lookback.

    Args:
        series: 1-D pandas Series of price/level data.  Use log-prices for
            equity time series — that's the canonical LdP approach.
        d: differentiation order in (0, 1).  Use ``find_min_ffd_d`` to
            calibrate per-instrument; 0.4 is a reasonable global default.
        thres: weight-cutoff threshold.

    Returns:
        Same-index Series with fractional differences.
    """
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    n = len(series)
    if n <= width:
        return pd.Series(np.nan, index=series.index, dtype=float)
    arr = series.to_numpy(dtype=float)
    out = np.full(n, np.nan, dtype=float)
    # Vectorised over the trailing window — LdP loop is the simplest correct
    # form and runs in ~ms for 1k-bar series so we keep it.
    for i in range(width, n):
        out[i] = float(np.dot(w, arr[i - width : i + 1]))
    return pd.Series(out, index=series.index, dtype=float)


def find_min_ffd_d(
    series: pd.Series, ds: list[float] | None = None, thres: float = 1e-4, p_value: float = 0.05
) -> float:
    """Search for the smallest d that makes the FFD-transformed series
    stationary at the given ADF p-value.  Returns d_opt or 1.0 if none found.

    Useful one-off for choosing d per-instrument before training.  The model
    pipeline doesn't need to call this every train cycle — pick one d once
    and stick with it.
    """
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore
    except Exception:
        # statsmodels is heavyweight; if missing, just use the conventional default
        return 0.4
    if ds is None:
        ds = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
    for d in ds:
        try:
            fd = frac_diff_ffd(series, d, thres).dropna()
            if len(fd) < 30:
                continue
            stat = adfuller(fd, maxlag=1, regression="c", autolag=None)
            if stat[1] < p_value:
                return float(d)
        except Exception:
            continue
    return 1.0
