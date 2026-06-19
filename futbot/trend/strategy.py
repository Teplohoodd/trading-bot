"""Bollinger breakout strategy logic (stateless functions).

Entry rule:
    Long  when close > MA(N) + k·σ(N)
    Short when close < MA(N) − k·σ(N)
    All on CLOSED hourly bars (not intra-bar).

Exit rule:
    Long  exits on close < MA(N) − k·σ(N)
    Short exits on close > MA(N) + k·σ(N)
    No SL.  No TP.  Pure mechanical band-flip.

Forced exit:
    Position must close N days before contract expiration (rollover guard).
    Handled in main.py, not here.

`evaluate()` returns one of three actions:
    OPEN_LONG / OPEN_SHORT — when flat and a band breakout printed
    CLOSE                  — when in a position and the opposite band hit
    HOLD                   — otherwise

The fact that bands are computed on the SAME close that's being tested is
slightly future-leaking in a backtest sense, but in LIVE the last closed
bar's close is exactly what we have when we evaluate.  In backtest this
is conservative because we use bar's close vs bands derived from that
same close.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Bands:
    upper: float
    middle: float
    lower: float
    sigma: float
    n_bars: int


def compute_bands(close: pd.Series, n: int, k: float) -> Bands | None:
    """Return latest Bollinger bands.  None if not enough data."""
    if len(close) < n + 1:
        return None
    window = close.iloc[-n:]
    mid = float(window.mean())
    sd = float(window.std())
    if sd <= 0:
        return None
    return Bands(
        upper=mid + k * sd,
        middle=mid,
        lower=mid - k * sd,
        sigma=sd,
        n_bars=int(n),
    )


@dataclass
class Decision:
    action: str  # "open_long" | "open_short" | "close" | "hold"
    reason: str
    bands: Bands | None = None


def evaluate(*, close: pd.Series, n: int, k: float, current_position: int = 0) -> Decision:
    """current_position: 0 = flat, +1 = long, -1 = short.

    Returns the decision based on the latest close vs Bollinger bands.
    """
    bands = compute_bands(close, n, k)
    if bands is None:
        return Decision("hold", f"warming up ({len(close)} < {n + 1} bars)", None)

    c = float(close.iloc[-1])
    if current_position == 0:
        if c > bands.upper:
            return Decision(
                "open_long",
                f"close {c:.4f} > upper {bands.upper:.4f} (k={k}σ)",
                bands,
            )
        if c < bands.lower:
            return Decision(
                "open_short",
                f"close {c:.4f} < lower {bands.lower:.4f} (k={k}σ)",
                bands,
            )
        return Decision(
            "hold",
            f"close {c:.4f} inside [{bands.lower:.4f}, {bands.upper:.4f}]",
            bands,
        )

    # In a position — check for opposite band cross
    if current_position == +1 and c < bands.lower:
        return Decision(
            "close",
            f"long exit: close {c:.4f} < lower {bands.lower:.4f}",
            bands,
        )
    if current_position == -1 and c > bands.upper:
        return Decision(
            "close",
            f"short exit: close {c:.4f} > upper {bands.upper:.4f}",
            bands,
        )
    return Decision(
        "hold",
        f"pos {current_position:+d} holding ({c:.4f} inside bands)",
        bands,
    )
