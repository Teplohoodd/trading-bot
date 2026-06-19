"""Pattern detectors — strict programmatic definitions.

Each detector scans a swing sequence and yields `Signal` objects.  A signal
is emitted on the bar that CONFIRMS the pattern (neckline break for
reversals, breakout bar for rectangles).  Backtester treats the signal's
`bar_idx` as the entry bar (we use that bar's close as the fill — fast and
deterministic; live trading would use the next-bar open).

Naming convention for direction:
    +1 = LONG  (pattern is bullish; we expect price to rise)
    -1 = SHORT (pattern is bearish; we expect price to fall)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from .primitives import Swing, find_swings

PatternName = Literal[
    "double_top",
    "double_bottom",
    "triple_top",
    "triple_bottom",
    "head_shoulders",
    "inv_head_shoulders",
    "rectangle_up",
    "rectangle_down",
    "ascending_triangle",
    "descending_triangle",
    "symmetric_triangle_up",
    "symmetric_triangle_down",
]


@dataclass(frozen=True)
class Signal:
    pattern: PatternName
    direction: int  # +1 long, -1 short
    bar_idx: int  # index of confirmation bar (entry bar)
    bar_time: pd.Timestamp
    entry_price: float  # close of confirmation bar
    stop_price: float  # pattern invalidation level
    target_price: float  # measured-move target
    # Bookkeeping
    pattern_height_pct: float  # pattern's full range / entry price


# ── helpers ────────────────────────────────────────────────────────────


def _close_at(df: pd.DataFrame, idx: int) -> float:
    return float(df["close"].iat[idx])


def _bar_time(df: pd.DataFrame, idx: int) -> pd.Timestamp:
    return pd.Timestamp(df["time"].iat[idx])


# ── Double Top / Double Bottom ─────────────────────────────────────────
#
# Double Top:
#   peak1 (H) — trough (L) — peak2 (H), where:
#     |peak1 - peak2| / max(peak1, peak2) <= peak_tol
#     (peak_avg - trough) / peak_avg >= min_height
#     bars between peak1..peak2 in [min_width, max_width]
#   Confirmed when subsequent bar closes BELOW trough.
#
# Stop: above max(peak1, peak2) by 0.3 * pattern_height
# Target: trough - pattern_height (measured move)


def _scan_double(
    df: pd.DataFrame,
    swings: list[Swing],
    *,
    peak_kind: Literal["H", "L"],
    peak_tol: float,
    min_height: float,
    min_width: int,
    max_width: int,
    max_confirm_bars: int,
) -> Iterator[Signal]:
    """Generic double-top (peak_kind=H) or double-bottom (peak_kind=L)."""
    trough_kind = "L" if peak_kind == "H" else "H"
    direction = -1 if peak_kind == "H" else +1
    pattern_name: PatternName = "double_top" if peak_kind == "H" else "double_bottom"

    # Walk pairs of peaks separated by a single trough
    for i in range(len(swings) - 2):
        p1, t, p2 = swings[i], swings[i + 1], swings[i + 2]
        if p1.kind != peak_kind or t.kind != trough_kind or p2.kind != peak_kind:
            continue
        width = p2.idx - p1.idx
        if not (min_width <= width <= max_width):
            continue
        peak_max = max(p1.price, p2.price) if peak_kind == "H" else min(p1.price, p2.price)
        peak_min = min(p1.price, p2.price) if peak_kind == "H" else max(p1.price, p2.price)
        peak_avg = (p1.price + p2.price) / 2
        # Peak similarity
        if abs(p1.price - p2.price) / max(peak_max, 1e-9) > peak_tol:
            continue
        # Height (depth of trough vs peaks)
        if peak_kind == "H":
            height = peak_avg - t.price
        else:
            height = t.price - peak_avg
        if height <= 0 or height / max(peak_avg, 1e-9) < min_height:
            continue

        # Look forward up to max_confirm_bars for confirmation close beyond trough
        neckline = t.price
        confirm_end = min(len(df) - 1, p2.idx + max_confirm_bars)
        for j in range(p2.idx + 1, confirm_end + 1):
            cj = _close_at(df, j)
            confirmed = (cj < neckline) if peak_kind == "H" else (cj > neckline)
            if confirmed:
                # Build signal
                pattern_height = height
                entry = cj
                if peak_kind == "H":
                    stop = peak_max + 0.3 * pattern_height
                    target = neckline - pattern_height
                else:
                    stop = peak_max - 0.3 * pattern_height
                    target = neckline + pattern_height
                yield Signal(
                    pattern=pattern_name,
                    direction=direction,
                    bar_idx=j,
                    bar_time=_bar_time(df, j),
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    pattern_height_pct=pattern_height / max(entry, 1e-9),
                )
                break  # only one confirmation per pattern


def detect_double_tops(df, swings, **kw) -> Iterator[Signal]:
    return _scan_double(df, swings, peak_kind="H", **kw)


def detect_double_bottoms(df, swings, **kw) -> Iterator[Signal]:
    return _scan_double(df, swings, peak_kind="L", **kw)


# ── Triple Top / Bottom ────────────────────────────────────────────────
#
# Same as double but with three peaks at similar height separated by
# two troughs of similar depth.  More reliable but rarer.


def _scan_triple(
    df: pd.DataFrame,
    swings: list[Swing],
    *,
    peak_kind: Literal["H", "L"],
    peak_tol: float,
    min_height: float,
    min_width: int,
    max_width: int,
    max_confirm_bars: int,
) -> Iterator[Signal]:
    trough_kind = "L" if peak_kind == "H" else "H"
    direction = -1 if peak_kind == "H" else +1
    pattern_name: PatternName = "triple_top" if peak_kind == "H" else "triple_bottom"

    for i in range(len(swings) - 4):
        p1, t1, p2, t2, p3 = swings[i : i + 5]
        if (
            p1.kind != peak_kind
            or t1.kind != trough_kind
            or p2.kind != peak_kind
            or t2.kind != trough_kind
            or p3.kind != peak_kind
        ):
            continue
        full_width = p3.idx - p1.idx
        if not (min_width <= full_width <= max_width):
            continue
        peaks = [p1.price, p2.price, p3.price]
        peak_max = max(peaks) if peak_kind == "H" else min(peaks)
        peak_avg = sum(peaks) / 3
        # All three peaks within tol
        if max(abs(p - peak_avg) for p in peaks) / max(peak_avg, 1e-9) > peak_tol:
            continue
        # Use lower trough (more conservative neckline)
        if peak_kind == "H":
            neckline = max(t1.price, t2.price)  # higher of two troughs = pattern broken sooner
            height = peak_avg - min(t1.price, t2.price)
        else:
            neckline = min(t1.price, t2.price)
            height = max(t1.price, t2.price) - peak_avg
        if height <= 0 or height / max(peak_avg, 1e-9) < min_height:
            continue

        confirm_end = min(len(df) - 1, p3.idx + max_confirm_bars)
        for j in range(p3.idx + 1, confirm_end + 1):
            cj = _close_at(df, j)
            confirmed = (cj < neckline) if peak_kind == "H" else (cj > neckline)
            if confirmed:
                entry = cj
                if peak_kind == "H":
                    stop = peak_max + 0.3 * height
                    target = neckline - height
                else:
                    stop = peak_max - 0.3 * height
                    target = neckline + height
                yield Signal(
                    pattern=pattern_name,
                    direction=direction,
                    bar_idx=j,
                    bar_time=_bar_time(df, j),
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    pattern_height_pct=height / max(entry, 1e-9),
                )
                break


def detect_triple_tops(df, swings, **kw) -> Iterator[Signal]:
    return _scan_triple(df, swings, peak_kind="H", **kw)


def detect_triple_bottoms(df, swings, **kw) -> Iterator[Signal]:
    return _scan_triple(df, swings, peak_kind="L", **kw)


# ── Head & Shoulders / Inverse H&S ─────────────────────────────────────
#
# H&S (bearish):  L0 - H1(left shoulder) - L1 - H2(head, HIGHEST) -
#                 L2 - H3(right shoulder) - L3
#   |H1 - H3| / H_avg <= shoulder_tol
#   H2 > max(H1, H3) by at least head_premium
#   L1, L2 within trough_tol of each other (neckline must be ~horizontal
#     for simplest version; sloped neckline left as TODO)
# Confirmation: close < min(L1, L2)


def _scan_hs(
    df: pd.DataFrame,
    swings: list[Swing],
    *,
    peak_kind: Literal["H", "L"],
    shoulder_tol: float,
    head_premium: float,
    trough_tol: float,
    min_width: int,
    max_width: int,
    max_confirm_bars: int,
) -> Iterator[Signal]:
    trough_kind = "L" if peak_kind == "H" else "H"
    direction = -1 if peak_kind == "H" else +1
    pattern_name: PatternName = "head_shoulders" if peak_kind == "H" else "inv_head_shoulders"

    # Pattern: P - T - P(head) - T - P  (5 swings of alternating kind)
    for i in range(len(swings) - 4):
        s1, t1, s2, t2, s3 = swings[i : i + 5]
        if (
            s1.kind != peak_kind
            or t1.kind != trough_kind
            or s2.kind != peak_kind
            or t2.kind != trough_kind
            or s3.kind != peak_kind
        ):
            continue
        full_width = s3.idx - s1.idx
        if not (min_width <= full_width <= max_width):
            continue
        # Shoulders similar
        sh_avg = (s1.price + s3.price) / 2
        if abs(s1.price - s3.price) / max(sh_avg, 1e-9) > shoulder_tol:
            continue
        # Head premium
        if peak_kind == "H":
            if s2.price < max(s1.price, s3.price) * (1 + head_premium):
                continue
        else:
            if s2.price > min(s1.price, s3.price) * (1 - head_premium):
                continue
        # Troughs similar (horizontal neckline)
        tr_avg = (t1.price + t2.price) / 2
        if abs(t1.price - t2.price) / max(tr_avg, 1e-9) > trough_tol:
            continue

        if peak_kind == "H":
            neckline = min(t1.price, t2.price)
            height = s2.price - tr_avg
        else:
            neckline = max(t1.price, t2.price)
            height = tr_avg - s2.price
        if height <= 0:
            continue

        confirm_end = min(len(df) - 1, s3.idx + max_confirm_bars)
        for j in range(s3.idx + 1, confirm_end + 1):
            cj = _close_at(df, j)
            confirmed = (cj < neckline) if peak_kind == "H" else (cj > neckline)
            if confirmed:
                entry = cj
                if peak_kind == "H":
                    stop = s2.price  # above head
                    target = neckline - height
                else:
                    stop = s2.price  # below head
                    target = neckline + height
                yield Signal(
                    pattern=pattern_name,
                    direction=direction,
                    bar_idx=j,
                    bar_time=_bar_time(df, j),
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    pattern_height_pct=height / max(entry, 1e-9),
                )
                break


def detect_head_shoulders(df, swings, **kw) -> Iterator[Signal]:
    return _scan_hs(df, swings, peak_kind="H", **kw)


def detect_inv_head_shoulders(df, swings, **kw) -> Iterator[Signal]:
    return _scan_hs(df, swings, peak_kind="L", **kw)


# ── Rectangle breakout ─────────────────────────────────────────────────
#
# A "rectangle" / consolidation: at least 4 alternating swings (H-L-H-L
# or L-H-L-H) where all Hs are within `band_tol` of each other and same
# for Ls.  Range height = avg(H) - avg(L) must exceed min_height.
# Confirmation: close beyond range by at least `break_pct` of height.


def detect_rectangles(
    df: pd.DataFrame,
    swings: list[Swing],
    *,
    min_swings: int = 4,
    band_tol: float = 0.012,
    min_height_pct: float = 0.02,
    break_pct: float = 0.2,
    min_width: int = 15,
    max_width: int = 80,
    max_confirm_bars: int = 5,
) -> Iterator[Signal]:
    """Yield rectangle_up (bullish break) and rectangle_down (bearish break)."""
    if len(swings) < min_swings:
        return

    # Sliding window over consecutive runs of strictly-alternating swings
    for start in range(len(swings) - min_swings + 1):
        # Take the longest alternating run from `start`
        run: list[Swing] = [swings[start]]
        for k in range(start + 1, len(swings)):
            if swings[k].kind != run[-1].kind:
                run.append(swings[k])
            else:
                break
        if len(run) < min_swings:
            continue

        # Trim to last min_swings + try expanding
        for window_size in range(min_swings, len(run) + 1):
            window = run[:window_size]
            highs = [s.price for s in window if s.kind == "H"]
            lows = [s.price for s in window if s.kind == "L"]
            if len(highs) < 2 or len(lows) < 2:
                continue
            h_avg = float(np.mean(highs))
            l_avg = float(np.mean(lows))
            # Bands tight enough
            if (max(highs) - min(highs)) / max(h_avg, 1e-9) > band_tol:
                continue
            if (max(lows) - min(lows)) / max(l_avg, 1e-9) > band_tol:
                continue
            height = h_avg - l_avg
            if height <= 0 or height / max(h_avg, 1e-9) < min_height_pct:
                continue
            full_width = window[-1].idx - window[0].idx
            if not (min_width <= full_width <= max_width):
                continue

            # Wait for breakout AFTER the last swing
            confirm_end = min(len(df) - 1, window[-1].idx + max_confirm_bars)
            broken = False
            for j in range(window[-1].idx + 1, confirm_end + 1):
                cj = _close_at(df, j)
                if cj > h_avg + break_pct * height:
                    # Bullish breakout
                    entry = cj
                    stop = l_avg
                    target = entry + height  # measured move from break
                    yield Signal(
                        pattern="rectangle_up",
                        direction=+1,
                        bar_idx=j,
                        bar_time=_bar_time(df, j),
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        pattern_height_pct=height / max(entry, 1e-9),
                    )
                    broken = True
                    break
                if cj < l_avg - break_pct * height:
                    entry = cj
                    stop = h_avg
                    target = entry - height
                    yield Signal(
                        pattern="rectangle_down",
                        direction=-1,
                        bar_idx=j,
                        bar_time=_bar_time(df, j),
                        entry_price=entry,
                        stop_price=stop,
                        target_price=target,
                        pattern_height_pct=height / max(entry, 1e-9),
                    )
                    broken = True
                    break
            if broken:
                break  # don't re-scan same window in larger sizes


# ── Triangles (ascending / descending / symmetric) ─────────────────────
#
# A triangle is a converging consolidation traced by ≥2 highs and ≥2 lows.
# We fit a least-squares trendline to the swing-highs and another to the
# swing-lows over a window of alternating swings, then classify by slope:
#
#   Ascending  : highs ~flat (resistance),  lows rising      → bullish bias
#   Descending : lows  ~flat (support),     highs falling     → bearish bias
#   Symmetric  : highs falling AND lows rising (converging)   → breakout dir
#
# "Flat" = |slope| / price-per-bar < flat_tol.  Convergence requires the
# high-line slope < low-line slope (apex ahead, range narrowing).
# Confirmation = close beyond the relevant trendline by break_pct·height.
# Target = measured move (widest part of the triangle) from breakout.


def _fit_line(idxs: list[int], prices: list[float]) -> tuple[float, float]:
    """OLS slope, intercept for prices ~ a·idx + b.  Returns (slope, intercept)."""
    x = np.asarray(idxs, dtype=float)
    y = np.asarray(prices, dtype=float)
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def detect_triangles(
    df: pd.DataFrame,
    swings: list[Swing],
    *,
    min_swings: int = 5,
    flat_tol: float = 0.0006,  # |slope|/price per bar below this ⇒ "flat"
    min_height_pct: float = 0.015,
    break_pct: float = 0.15,
    min_width: int = 15,
    max_width: int = 90,
    max_confirm_bars: int = 8,
) -> Iterator[Signal]:
    """Yield ascending/descending/symmetric triangle breakout signals.

    Scans windows of alternating swings; fits high-line and low-line; if the
    lines converge (narrowing range) and at least one breaks, emit a signal.
    """
    n = len(swings)
    if n < min_swings:
        return

    for start in range(n - min_swings + 1):
        # Build a maximal alternating run from `start`
        run: list[Swing] = [swings[start]]
        for k in range(start + 1, n):
            if swings[k].kind != run[-1].kind:
                run.append(swings[k])
            else:
                break
        if len(run) < min_swings:
            continue

        # Try the largest valid window first (more swings = more reliable)
        for window_size in range(len(run), min_swings - 1, -1):
            window = run[:window_size]
            highs = [(s.idx, s.price) for s in window if s.kind == "H"]
            lows = [(s.idx, s.price) for s in window if s.kind == "L"]
            if len(highs) < 2 or len(lows) < 2:
                continue

            full_width = window[-1].idx - window[0].idx
            if not (min_width <= full_width <= max_width):
                continue

            h_slope, h_int = _fit_line([h[0] for h in highs], [h[1] for h in highs])
            l_slope, l_int = _fit_line([l[0] for l in lows], [l[1] for l in lows])

            ref_price = float(np.mean([p for _, p in highs] + [p for _, p in lows]))
            if ref_price <= 0:
                continue
            # Height = widest vertical gap (at the left edge of the window)
            x0 = window[0].idx
            top0 = h_slope * x0 + h_int
            bot0 = l_slope * x0 + l_int
            height = top0 - bot0
            if height <= 0 or height / ref_price < min_height_pct:
                continue

            # Normalised slopes (fraction of price per bar)
            h_norm = h_slope / ref_price
            l_norm = l_slope / ref_price
            h_flat = abs(h_norm) < flat_tol
            l_flat = abs(l_norm) < flat_tol

            # Convergence: top line must descend relative to bottom line
            converging = h_slope < l_slope
            if not converging:
                continue

            # Classify
            if h_flat and l_slope > 0:
                kind = "ascending_triangle"
                direction = +1
            elif l_flat and h_slope < 0:
                kind = "descending_triangle"
                direction = -1
            elif h_slope < 0 and l_slope > 0:
                kind = "symmetric"
                direction = 0  # decided by break dir
            else:
                continue

            # Wait for breakout after last swing
            confirm_end = min(len(df) - 1, window[-1].idx + max_confirm_bars)
            emitted = False
            for j in range(window[-1].idx + 1, confirm_end + 1):
                cj = _close_at(df, j)
                up_line = h_slope * j + h_int
                dn_line = l_slope * j + l_int
                broke_up = cj > up_line + break_pct * height
                broke_dn = cj < dn_line - break_pct * height

                if direction == +1 and broke_up:
                    yield Signal(
                        pattern="ascending_triangle",
                        direction=+1,
                        bar_idx=j,
                        bar_time=_bar_time(df, j),
                        entry_price=cj,
                        stop_price=dn_line,
                        target_price=cj + height,
                        pattern_height_pct=height / max(cj, 1e-9),
                    )
                    emitted = True
                    break
                if direction == -1 and broke_dn:
                    yield Signal(
                        pattern="descending_triangle",
                        direction=-1,
                        bar_idx=j,
                        bar_time=_bar_time(df, j),
                        entry_price=cj,
                        stop_price=up_line,
                        target_price=cj - height,
                        pattern_height_pct=height / max(cj, 1e-9),
                    )
                    emitted = True
                    break
                if direction == 0:
                    if broke_up:
                        yield Signal(
                            pattern="symmetric_triangle_up",
                            direction=+1,
                            bar_idx=j,
                            bar_time=_bar_time(df, j),
                            entry_price=cj,
                            stop_price=dn_line,
                            target_price=cj + height,
                            pattern_height_pct=height / max(cj, 1e-9),
                        )
                        emitted = True
                        break
                    if broke_dn:
                        yield Signal(
                            pattern="symmetric_triangle_down",
                            direction=-1,
                            bar_idx=j,
                            bar_time=_bar_time(df, j),
                            entry_price=cj,
                            stop_price=up_line,
                            target_price=cj - height,
                            pattern_height_pct=height / max(cj, 1e-9),
                        )
                        emitted = True
                        break
            if emitted:
                break  # don't re-scan smaller windows from same start


# ── Public API ─────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "double": dict(
        peak_tol=0.025,  # peaks within 2.5%
        min_height=0.015,  # trough at least 1.5% from peaks
        min_width=8,
        max_width=60,
        max_confirm_bars=10,
    ),
    "triple": dict(
        peak_tol=0.030,
        min_height=0.015,
        min_width=15,
        max_width=80,
        max_confirm_bars=10,
    ),
    "hs": dict(
        shoulder_tol=0.030,
        head_premium=0.010,  # head must exceed shoulders by 1%
        trough_tol=0.020,
        min_width=12,
        max_width=70,
        max_confirm_bars=10,
    ),
    "rect": dict(
        band_tol=0.015,
        min_height_pct=0.020,
        break_pct=0.20,
        min_width=15,
        max_width=80,
        max_confirm_bars=5,
    ),
    "triangle": dict(
        flat_tol=0.0006,
        min_height_pct=0.015,
        break_pct=0.15,
        min_width=15,
        max_width=90,
        max_confirm_bars=8,
    ),
}


def detect_all(
    df: pd.DataFrame,
    *,
    swing_window: int = 5,
    min_prominence_pct: float = 0.005,
    params: dict | None = None,
) -> list[Signal]:
    """Run every detector on `df`; return all signals sorted by bar_idx."""
    if params is None:
        params = DEFAULT_PARAMS
    swings = find_swings(df, window=swing_window, min_prominence_pct=min_prominence_pct)
    signals: list[Signal] = []
    signals.extend(detect_double_tops(df, swings, **params["double"]))
    signals.extend(detect_double_bottoms(df, swings, **params["double"]))
    signals.extend(detect_triple_tops(df, swings, **params["triple"]))
    signals.extend(detect_triple_bottoms(df, swings, **params["triple"]))
    signals.extend(detect_head_shoulders(df, swings, **params["hs"]))
    signals.extend(detect_inv_head_shoulders(df, swings, **params["hs"]))
    signals.extend(detect_rectangles(df, swings, **params["rect"]))
    signals.extend(detect_triangles(df, swings, **params["triangle"]))
    signals.sort(key=lambda s: s.bar_idx)
    return signals
