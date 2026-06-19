"""Systematic candlestick-pattern backtest on hourly FORTS via TA-Lib (61 CDL*).

Discipline (same as every other strategy this project): CAUSAL entry, pooled
across instruments for sample size, judged by PER-YEAR consistency — NOT a
single headline number.  Candlestick patterns are famously curve-fit-prone, so
the bar is high: a pattern must be positive in ≥3 of 4 years AND beat costs.

Mechanics:
  • TA-Lib CDL* functions return +100 (bullish) / -100 (bearish) / 0 at each
    bar, using only that bar and a few preceding ones → causal by construction.
  • Signal at bar t (computed on closes ≤ t) → ENTER at t+1 OPEN (no look-ahead).
  • Hold H bars, exit at close.  Direction = sign of the pattern value.
  • Net of round-trip cost.  Pooled over all 7 instruments × 2022-2025.

Usage:  python -u -m futbot.scripts.candlestick_backtest --hold 6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import talib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path(__file__).resolve().parents[2] / "data"
BASES = ["Si", "LK", "GZ", "SR", "RN", "GK", "MM"]
COST_RT = 0.0008  # round-trip cost fraction
HOURS_PER_YEAR = 24 * 365


PREFIX = "hist_"  # overridden by --oos to "histOOS_"


def _load(base: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"{PREFIX}{base}.parquet").sort_values("time")
    df["time"] = pd.to_datetime(df["time"])
    return df.reset_index(drop=True)


def _pattern_funcs():
    """All TA-Lib candlestick functions: name → callable."""
    names = talib.get_function_groups()["Pattern Recognition"]
    return {n: getattr(talib, n) for n in names}


def backtest_pattern(frames: dict, func, hold: int) -> dict:
    """Pool one pattern across all instruments.  Returns per-year + aggregate."""
    rows = []  # (year, net_return, direction)
    for base, df in frames.items():
        o, h, l, c = (df[x].values.astype(float) for x in ("open", "high", "low", "close"))
        sig = func(o, h, l, c)  # +100/-100/0, causal
        years = df["time"].dt.year.values
        n = len(c)
        for t in range(1, n - hold - 1):
            s = sig[t]
            if s == 0:
                continue
            d = 1 if s > 0 else -1
            entry = o[t + 1]  # enter next bar open (no look-ahead)
            exit_ = c[t + 1 + hold]
            if entry <= 0:
                continue
            ret = d * (exit_ / entry - 1.0) - COST_RT
            rows.append((years[t + 1], ret))
    if len(rows) < 30:
        return {"n": len(rows)}
    arr = pd.DataFrame(rows, columns=["year", "ret"])
    by_year = arr.groupby("year")["ret"].agg(["sum", "size", "mean"])
    pos_years = int((by_year["sum"] > 0).sum())
    n_years = int(len(by_year))
    total = float(arr["ret"].sum())
    win = float((arr["ret"] > 0).mean())
    sharpe = (
        float(arr["ret"].mean() / arr["ret"].std() * np.sqrt(252)) if arr["ret"].std() > 0 else 0.0
    )
    return {
        "n": len(rows),
        "win": win * 100,
        "total": total * 100,
        "avg_bps": float(arr["ret"].mean() * 1e4),
        "sharpe": sharpe,
        "pos_years": pos_years,
        "n_years": n_years,
        "by_year": {int(y): float(r) * 100 for y, r in by_year["sum"].items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=6, help="bars to hold")
    ap.add_argument(
        "--oos", action="store_true", help="run on histOOS_* new tickers (true out-of-sample)"
    )
    ap.add_argument("--only", default="", help="comma-sep pattern names to test")
    args = ap.parse_args()

    global PREFIX
    bases = BASES
    if args.oos:
        PREFIX = "histOOS_"
        bases = ["BR", "MX", "RI", "GD"]
    frames = {b: _load(b) for b in bases if (DATA / f"{PREFIX}{b}.parquet").exists()}
    funcs = _pattern_funcs()
    if args.only:
        want = {("CDL" + x).upper().replace("CDLCDL", "CDL") for x in args.only.split(",")}
        funcs = {n: f for n, f in funcs.items() if n.upper() in want}
    print("=" * 104)
    print(
        f"CANDLESTICK PATTERNS (TA-Lib {talib.__version__}) — {len(funcs)} patterns, "
        f"hold={args.hold}h, {'OOS ' if args.oos else ''}pooled {list(frames)} 2022-2025, "
        f"causal, cost {COST_RT*1e4:.0f}bps RT"
    )
    print("=" * 104)
    results = []
    for name, fn in funcs.items():
        try:
            r = backtest_pattern(frames, fn, args.hold)
        except Exception:
            continue
        if r.get("n", 0) >= 100 and "sharpe" in r:
            r["name"] = name.replace("CDL", "")
            results.append(r)
    # Sort by per-year robustness then Sharpe
    results.sort(key=lambda x: (x["pos_years"], x["sharpe"]), reverse=True)
    print(
        f"\n{'pattern':<20}{'n':>6}{'win%':>7}{'avgbps':>8}{'sharpe':>8}" f"{'+yrs':>6}  per-year %"
    )
    print("-" * 104)
    survivors = []
    for r in results:
        py = " ".join(f"{y}:{v:+.1f}" for y, v in r["by_year"].items())
        keep = r["pos_years"] >= r["n_years"] - 0 and r["sharpe"] >= 0.5 and r["avg_bps"] > 0
        flag = "  <<KEEP" if keep else ""
        if keep:
            survivors.append(r)
        # only print patterns with some edge to keep output readable
        if r["sharpe"] >= 0.3 or r["pos_years"] >= r["n_years"] - 1:
            print(
                f"{r['name']:<20}{r['n']:>6}{r['win']:>7.1f}{r['avg_bps']:>8.1f}"
                f"{r['sharpe']:>8.2f}{r['pos_years']:>4}/{r['n_years']}  {py}{flag}"
            )
    print("-" * 104)
    print(f"\nGate: ALL years positive AND Sharpe ≥ 0.5 AND avg > 0.")
    print(f"Survivors: {len(survivors)}/{len(results)} patterns with ≥100 trades")
    for s in survivors:
        print(
            f"  ✓ {s['name']:<18} n={s['n']:>5} win={s['win']:.0f}% "
            f"sharpe={s['sharpe']:.2f} +{s['pos_years']}/{s['n_years']} "
            f"total={s['total']:+.0f}%"
        )
    if not survivors:
        print("  (none — candlestick patterns do not survive per-year consistency)")


if __name__ == "__main__":
    main()
