"""Trend-pattern walk-forward across 9 years / many regimes (daily data).

Uses the user-supplied MOEX daily dataset (2015-2024, TQBR spot) to test
whether the triple_top / triple_bottom pattern edge holds OUT-OF-SAMPLE
across very different market regimes — something the ~180-day futures
history could never show.

Caveats (honest):
  • Daily bars (not the bot's hourly) — tests STRATEGY LOGIC, not exact
    hourly execution.
  • Stock SPOT (TQBR), not futures — no leverage/basis, but futures track
    spot, so pattern co-behaviour transfers.
  • Excel has CLOSE only → high=low=close (lose intrabar wicks; pattern
    structure from closes is standard and conservative for stop/target).

Per calendar-year breakdown reveals regime dependence: a strategy that only
works in one regime is fragile.

Usage:
    python -u -m futbot.scripts.trend_daily_walkforward
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from futbot.patterns.detectors import detect_all
from futbot.patterns.backtest import simulate_trades

# Daily-tuned params: patterns form over weeks-months.  Widths in DAYS.
DAILY_PARAMS = {
    "double": dict(peak_tol=0.04, min_height=0.03, min_width=5, max_width=40, max_confirm_bars=8),
    "triple": dict(peak_tol=0.05, min_height=0.03, min_width=8, max_width=60, max_confirm_bars=8),
    "hs": dict(
        shoulder_tol=0.04,
        head_premium=0.02,
        trough_tol=0.03,
        min_width=8,
        max_width=50,
        max_confirm_bars=8,
    ),
    "rect": dict(
        band_tol=0.03,
        min_height_pct=0.04,
        break_pct=0.20,
        min_width=8,
        max_width=60,
        max_confirm_bars=5,
    ),
    "triangle": dict(
        flat_tol=0.0015,
        min_height_pct=0.03,
        break_pct=0.15,
        min_width=8,
        max_width=60,
        max_confirm_bars=6,
    ),
}
MAX_HOLD_DAYS = 15
COMMISSION_RT = 0.0008
CORE = [
    "GAZP",
    "SBER",
    "LKOH",
    "GMKN",
    "YNDX",
    "MOEX",
    "NVTK",
    "PLZL",
    "MGNT",
    "VTBR",
    "FIVE",
    "TCSG",
]


def main():
    path = glob.glob("C:/Users/Teplohood/Downloads/*market*.xlsx")[0]
    raw = pd.read_excel(path, sheet_name="Sheet1")
    raw["TRADEDATE"] = pd.to_datetime(raw["TRADEDATE"])
    raw = raw[raw["BOARDID"] == "TQBR"]

    all_trades = []
    for sec in CORE:
        d = raw[raw["SECID"] == sec].sort_values("TRADEDATE")
        if len(d) < 300:
            continue
        df = pd.DataFrame(
            {
                "time": d["TRADEDATE"].values,
                "close": d["CLOSE"].astype(float).values,
            }
        )
        df["high"] = df["close"]
        df["low"] = df["close"]
        sigs = [
            s
            for s in detect_all(df, swing_window=5, min_prominence_pct=0.01, params=DAILY_PARAMS)
            if s.pattern in ("triple_top", "triple_bottom")
        ]
        trades = simulate_trades(
            df, sigs, base=sec, max_bars_held=MAX_HOLD_DAYS, commission_rt_pct=COMMISSION_RT
        )
        all_trades.extend(trades)

    if not all_trades:
        print("No trades generated")
        return
    td = pd.DataFrame(
        [
            {
                "base": t.base,
                "pattern": t.pattern,
                "direction": t.direction,
                "entry_time": pd.Timestamp(t.entry_time),
                "net": t.net_pnl_pct,
                "exit_reason": t.exit_reason,
            }
            for t in all_trades
        ]
    )
    # Drop NaN/inf trades (2022 MOEX closure left data gaps → bad fills)
    n_before = len(td)
    td = td[np.isfinite(td["net"])].copy()
    dropped = n_before - len(td)
    if dropped:
        print(f"(dropped {dropped} NaN/inf trades — likely 2022 MOEX closure gaps)")
    td["year"] = td["entry_time"].dt.year

    print("=" * 84)
    print("TREND triple_top/bottom — WALK-FORWARD across 9 years (daily, MOEX spot)")
    print("=" * 84)
    print(
        f"Total trades: {len(td)}  across {td.base.nunique()} names, "
        f"{td.year.min()}-{td.year.max()}"
    )

    # Per-year breakdown (each year is effectively out-of-sample for the next)
    print(f"\n{'year':>6}{'n':>5}{'win%':>7}{'avg%':>8}{'total%':>9}{'sharpe':>8}  regime note")
    print("-" * 84)
    regimes = {
        2015: "oil crash / RUB devaluation",
        2016: "recovery",
        2017: "low-vol bull",
        2018: "sanctions selloff",
        2019: "bull",
        2020: "COVID crash+rebound",
        2021: "bull top",
        2022: "war/sanctions shock",
        2023: "recovery rally",
        2024: "partial year",
    }
    pos_years = 0
    yr_stats = []
    for yr, g in td.groupby("year"):
        a = g["net"].values
        sh = a.mean() / a.std() * np.sqrt(len(a)) if a.std() > 0 else 0.0
        if a.sum() > 0:
            pos_years += 1
        yr_stats.append((yr, len(a), (a > 0).mean(), a.mean(), a.sum(), sh))
        print(
            f"{yr:>6}{len(a):>5}{(a>0).mean()*100:>6.0f}%{a.mean():>+8.2f}"
            f"{a.sum():>+9.1f}{sh:>+8.2f}  {regimes.get(yr,'')}"
        )
    n_years = td.year.nunique()
    print("-" * 84)
    tot = td["net"].values
    overall_sh = tot.mean() / tot.std() * np.sqrt(len(tot)) if tot.std() > 0 else 0.0
    print(
        f"Positive years: {pos_years}/{n_years}   "
        f"overall: win {(tot>0).mean()*100:.0f}%  avg {tot.mean():+.2f}%  "
        f"total {tot.sum():+.1f}%  per-trade Sharpe {overall_sh:+.2f}"
    )

    # Direction split
    print(f"\nBy pattern:")
    for pat, g in td.groupby("pattern"):
        a = g["net"].values
        print(
            f"  {pat:<14} n={len(a):>4} win={(a>0).mean()*100:>3.0f}% "
            f"total={a.sum():>+7.1f}% avg={a.mean():+.2f}%"
        )

    verdict = (
        "✅ ROBUST across regimes"
        if pos_years >= n_years - 2
        else (
            "⚠ REGIME-DEPENDENT" if pos_years >= n_years * 0.5 else "❌ FRAGILE / mostly one regime"
        )
    )
    print(f"\nVERDICT: {verdict}  ({pos_years}/{n_years} years positive)")

    out = Path(__file__).resolve().parents[2] / "data" / "trend_daily_wf.csv"
    td.to_csv(out, index=False)
    print(f"Saved trades → {out}")


if __name__ == "__main__":
    main()
