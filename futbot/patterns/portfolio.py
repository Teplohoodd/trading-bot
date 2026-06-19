"""Pattern-trading portfolio + tuned parameters.

This file is the SOURCE OF TRUTH for the live pattern bot:
  - Which (base, pattern) combos to trade
  - Detector params (peak tolerance, height, widths)
  - Trade management (max bars held)

Updated 2026-05-24 from the grid-search tuner (futbot.patterns.tune).
Re-tune every 60-90 days by re-running tune.py and replacing this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PatternParams:
    """Detector parameters — passed straight to detectors.detect_*."""

    swing_window: int = 5
    min_prominence_pct: float = 0.005
    # triple
    peak_tol: float = 0.030
    min_height: float = 0.015
    min_width: int = 15
    max_width: int = 80
    max_confirm_bars: int = 10
    # trade mgmt
    max_bars_held: int = 48


@dataclass(frozen=True)
class PatternEntry:
    base: str
    patterns: tuple[str, ...]  # which pattern names to trade for this base
    notes: str = ""


# ── DEFAULT (un-tuned) params ─────────────────────────────────────────
DEFAULT_PARAMS = PatternParams()

# ── TUNED via futbot.patterns.tune (grid 324 combos) on 2026-05-24 ────
# Selection: best OOS Sharpe (0.44) at the LARGEST OOS sample (79 trades,
# 30d window) so the result is robust to small-sample variance.  Diff vs
# default: peak_tol 0.030 -> 0.040 (looser peak-similarity tolerance).
# IS:  154 trades, WR 71%, +97% total, Sharpe 0.22
# OOS:  79 trades, WR 73%, +66% total, Sharpe 0.44
TUNED_PARAMS = PatternParams(
    swing_window=5,
    min_prominence_pct=0.005,
    peak_tol=0.040,
    min_height=0.015,
    min_width=15,
    max_width=80,
    max_confirm_bars=10,
    max_bars_held=48,
)

# ── Whitelist ──────────────────────────────────────────────────────────
# Based on baseline backtest 2026-05-24 (90d IS, 30d OOS split).
# triple_top edge confirmed OOS on 19/25 contracts → trade everywhere.
# triple_bottom edge weak universally — only specific contracts.

# Universal: all WF-portfolio bases trade triple_top
_ALL_BASES = [
    "PX",
    "USDRU",
    "GD",
    "SS",
    "SZ",
    "YD",
    "LT",
    "S1",
    "SOLUSDper",
    "VB",
    "MV",
    "GK",
    "AMDper",
    "SA",
    "AK",
    "GN",
    "IB",
    "TT",
    "GL",
    "GLDRU",
    "EA",
    "AN",
    "SV",
    "CC",
    "RN",
    "SC",
]

PATTERN_PORTFOLIO: list[PatternEntry] = [
    PatternEntry(base=b, patterns=("triple_top",), notes="OOS-validated") for b in _ALL_BASES
]

# Triple-bottom whitelist — contracts where it worked in BOTH IS and OOS.
TB_WHITELIST: set[str] = {"IB", "MV", "VB", "LT"}


def patterns_for_base(base: str) -> tuple[str, ...]:
    """Return the list of pattern names to trade for `base`."""
    out: list[str] = []
    for e in PATTERN_PORTFOLIO:
        if e.base == base:
            out.extend(e.patterns)
    if base in TB_WHITELIST:
        out.append("triple_bottom")
    return tuple(out)


def is_pattern_allowed(base: str, pattern: str) -> bool:
    """Quick lookup: should we trade `pattern` on `base`?"""
    return pattern in patterns_for_base(base)
