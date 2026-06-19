"""Swing-point detection — foundation for all pattern detectors.

A swing HIGH at bar i means high[i] is the highest in window [i-w, i+w].
Similarly for swing LOW.  We require BOTH:
  - geometric isolation (no higher high in window)
  - minimum prominence vs the surrounding troughs (filters noise)

This is intentionally simpler than scipy.find_peaks: we work on OHLC and
need the actual high/low extremes, not just close.  Also keeps the
dependency surface small (only numpy/pandas).

References:
- Bulkowski, "Encyclopedia of Chart Patterns" (2nd ed.) — swing definition
- Murphy, "Technical Analysis of the Financial Markets" — peak/trough rules
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

SwingKind = Literal["H", "L"]


@dataclass(frozen=True)
class Swing:
    """One pivot point on a price series."""

    idx: int  # bar index in source DataFrame
    kind: SwingKind  # "H" = swing high, "L" = swing low
    price: float  # actual high (for H) or low (for L)
    bar_time: pd.Timestamp


def find_swings(
    df: pd.DataFrame,
    *,
    window: int = 5,
    min_prominence_pct: float = 0.005,
) -> list[Swing]:
    """Return swing highs + lows on `df` (must have columns: high, low, time).

    Parameters
    ----------
    window
        Half-window for the "highest in N bars on each side" test.
        window=5 → swing confirmed if it's the highest of 11 bars.
    min_prominence_pct
        Reject swings that are within this fraction of the immediately
        preceding swing of the SAME kind.  Filters micro-noise.

    Returns swings in chronological order, strictly alternating H/L.
    If two consecutive swings of the same kind appear (rare with
    well-tuned window), the lower-prominence one is dropped.
    """
    if "high" not in df.columns or "low" not in df.columns:
        raise ValueError("df must have 'high' and 'low' columns")
    n = len(df)
    if n < 2 * window + 1:
        return []

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    times = df["time"].to_numpy() if "time" in df.columns else np.arange(n)

    swings: list[Swing] = []
    for i in range(window, n - window):
        lo_h = max(0, i - window)
        hi_h = min(n, i + window + 1)
        # Swing high if high[i] is strict max in window (ties on left only)
        win_h = highs[lo_h:hi_h]
        if (
            highs[i] == win_h.max() and highs[i] > win_h[: i - lo_h].max(initial=-np.inf)
            if i > lo_h
            else True
        ):
            # Also require it dominates the right side (no equal-or-higher)
            if i + 1 < hi_h and highs[i] <= highs[i + 1 : hi_h].max(initial=-np.inf):
                pass
            else:
                swings.append(
                    Swing(
                        idx=i,
                        kind="H",
                        price=float(highs[i]),
                        bar_time=pd.Timestamp(times[i]),
                    )
                )
                continue
        # Swing low
        win_l = lows[lo_h:hi_h]
        if (
            lows[i] == win_l.min() and lows[i] < win_l[: i - lo_h].min(initial=np.inf)
            if i > lo_h
            else True
        ):
            if i + 1 < hi_h and lows[i] >= lows[i + 1 : hi_h].min(initial=np.inf):
                pass
            else:
                swings.append(
                    Swing(
                        idx=i,
                        kind="L",
                        price=float(lows[i]),
                        bar_time=pd.Timestamp(times[i]),
                    )
                )

    if not swings:
        return []

    # Enforce alternation: if two consecutive same-kind swings, keep the
    # more extreme one (higher H or lower L).
    cleaned: list[Swing] = [swings[0]]
    for s in swings[1:]:
        prev = cleaned[-1]
        if s.kind == prev.kind:
            if s.kind == "H" and s.price > prev.price:
                cleaned[-1] = s
            elif s.kind == "L" and s.price < prev.price:
                cleaned[-1] = s
            # else keep prev (already more extreme)
        else:
            cleaned.append(s)

    # Prominence filter: drop swings within min_prominence_pct of the
    # IMMEDIATELY surrounding opposite-kind swings.  A real H must rise
    # significantly above its bracketing Ls (and vice-versa).
    if min_prominence_pct <= 0:
        return cleaned

    filtered: list[Swing] = []
    for i, s in enumerate(cleaned):
        if 0 < i < len(cleaned) - 1:
            prev_opp = cleaned[i - 1]
            next_opp = cleaned[i + 1]
            if s.kind == "H":
                base = max(prev_opp.price, next_opp.price)
                if (s.price - base) / max(base, 1e-9) < min_prominence_pct:
                    continue
            else:  # "L"
                base = min(prev_opp.price, next_opp.price)
                if (base - s.price) / max(base, 1e-9) < min_prominence_pct:
                    continue
        filtered.append(s)

    # Re-enforce alternation after filtering
    final: list[Swing] = []
    for s in filtered:
        if final and final[-1].kind == s.kind:
            # Keep more extreme
            if s.kind == "H" and s.price > final[-1].price:
                final[-1] = s
            elif s.kind == "L" and s.price < final[-1].price:
                final[-1] = s
        else:
            final.append(s)
    return final


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's ATR — used for stop sizing in pattern trades."""
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat(
        [
            (h - l),
            (h - pc).abs(),
            (l - pc).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()
