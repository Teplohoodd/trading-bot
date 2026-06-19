"""carry_lab.py — Multi-instrument calendar-spread carry lab.

For each base in {Si, GZ, LK, SR, RN, GK}:
  - Build continuous front/next pair series. front data comes from cached
    quarterly parquets (data/hist_{BASE}.parquet). next data is fetched live
    from MOEX ISS for each front's active window (the parquets only contain
    each contract during its own front window).
  - Compute structural annualised basis.
  - Run delta-neutral basis mean-reversion backtest (reusing si_calendar_carry
    logic verbatim): z_entry=1.5, z_stop=3.5, roll=240h, max_hold=72h.
  - Report PER CALENDAR YEAR: trades, win%, total%, Sharpe.
  - Verdict: robust if >= (n_years - 1) positive years.

Usage:
    python -u futbot/scripts/carry_lab.py 2>&1 | tee data/carry_lab.log
"""

import sys
import math
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futbot.scripts.moex_iss_history import _candles

# ── Backtest parameters (identical to si_calendar_carry) ──────────────────────
COMMISSION_RT = 0.0008
Z_ENTRY = 1.5
Z_STOP = 3.5
ROLL_WIN = 240  # hours
MAX_HOLD = 72  # hours
QUARTERLY = ["H", "M", "U", "Z"]
QMONTH = {"H": 3, "M": 6, "U": 9, "Z": 12}
DT_YEARS = 3 / 12.0  # ~quarterly spacing

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ── Core helpers ──────────────────────────────────────────────────────────────


def backtest_basis_reversion(front: np.ndarray, nxt: np.ndarray, return_trades: bool = False):
    """Verbatim copy from si_calendar_carry — z-score mean-reversion of basis."""
    basis = nxt - front
    n = len(basis)
    if n < ROLL_WIN + 50:
        return ({"n": 0}, []) if return_trades else {"n": 0}
    z = np.full(n, np.nan)
    for t in range(ROLL_WIN, n):
        w = basis[t - ROLL_WIN : t]
        sd = w.std()
        z[t] = (basis[t] - w.mean()) / sd if sd > 0 else 0.0
    pos = 0
    entry = None
    pnls = []
    trades = []
    for t in range(ROLL_WIN, n):
        if np.isnan(z[t]):
            continue
        if pos == 0:
            if z[t] > Z_ENTRY:
                pos, entry = -1, t
            elif z[t] < -Z_ENTRY:
                pos, entry = +1, t
            continue
        crossed = (pos == +1 and z[t] >= 0) or (pos == -1 and z[t] <= 0)
        stopped = abs(z[t]) >= Z_STOP
        timed = (t - entry) >= MAX_HOLD
        if not (crossed or stopped or timed):
            continue
        d_basis = basis[t] - basis[entry]
        ref = front[entry]
        gross = pos * d_basis / ref if ref > 0 else 0
        pnl = gross - COMMISSION_RT
        pnls.append(pnl)
        trades.append((t, pnl))
        pos = 0
    if not pnls:
        return ({"n": 0}, []) if return_trades else {"n": 0}
    a = np.array(pnls)
    stats = {
        "n": len(a),
        "win": float((a > 0).mean()),
        "total_pct": float(a.sum() * 100),
        "avg_pct": float(a.mean() * 100),
        "sharpe": float(a.mean() / a.std() * math.sqrt(len(a))) if a.std() > 0 else 0.0,
    }
    return (stats, trades) if return_trades else stats


def _ordered_contracts(base, start_year, end_year):
    """Return ordered list of contract secids and their roll boundaries."""
    result = []
    prev_boundary = None
    for yr in range(start_year - 1, end_year + 2):
        for q in QUARTERLY:
            secid = base + q + str(yr % 10)
            exp_month = QMONTH[q]
            boundary = pd.Timestamp(yr, exp_month, 15)
            if prev_boundary is not None:
                result.append((secid, prev_boundary, boundary))
            prev_boundary = boundary
    return result


def build_front_next(base: str, start_year: int = 2022, end_year: int = 2025):
    """
    Build aligned front/next DataFrame by:
    1. Loading front-contract data from parquet (already stitched front-month).
    2. For each consecutive contract pair (front_i, next_i+1):
       fetching next_i+1's candles from MOEX ISS during front_i's active window
       (the parquet only has next during its OWN front window, not during the
        prior contract's window).
    3. Inner-joining on timestamp.
    Returns DataFrame with index=time, columns=[front, next].
    """
    path = DATA_DIR / ("hist_" + base + ".parquet")
    if not path.exists():
        print("  [SKIP] no parquet: " + str(path))
        return pd.DataFrame()

    raw = pd.read_parquet(path)
    raw["time"] = pd.to_datetime(raw["time"])

    # Build per-contract front series from parquet
    by_contract = {}
    for secid, grp in raw.groupby("contract"):
        s = grp.set_index("time")["close"].sort_index()
        by_contract[secid] = s

    # Filter to 2022-2025 only
    t_min = pd.Timestamp(str(start_year) + "-01-01")
    t_max = pd.Timestamp(str(end_year + 1) + "-01-01")

    # Ordered contract pairs with their active windows
    contracts = _ordered_contracts(base, start_year, end_year)

    frames = []
    for secid, lo, hi in contracts:
        # Clip to our study window
        lo_clip = max(lo, t_min)
        hi_clip = min(hi, t_max)
        if lo_clip >= hi_clip:
            continue
        if secid not in by_contract:
            continue

        # Figure out which contract comes NEXT after this one
        # Find secid's position in the ordered list and get the next one
        idx_in_q = None
        for i, (s, _, _) in enumerate(contracts):
            if s == secid:
                idx_in_q = i
                break
        if idx_in_q is None or idx_in_q + 1 >= len(contracts):
            continue
        next_secid = contracts[idx_in_q + 1][0]

        # Front series from parquet
        f_series = by_contract[secid]
        f_series = f_series[(f_series.index >= lo_clip) & (f_series.index < hi_clip)]
        if f_series.empty:
            continue

        # Fetch next-contract data from MOEX ISS for this window
        frm = lo_clip.strftime("%Y-%m-%d")
        til = hi_clip.strftime("%Y-%m-%d")
        print(
            "  Fetching next " + next_secid + " for front " + secid + " [" + frm + ".." + til + "]",
            flush=True,
        )
        n_df = _candles("futures", "forts", next_secid, 60, frm, til)
        _time.sleep(0.1)

        if n_df.empty:
            print("    -> no data for " + next_secid)
            continue

        n_series = n_df.set_index("time")["close"].sort_index()
        # Clip to same window
        n_series = n_series[
            (n_series.index >= lo_clip.replace(tzinfo=None))
            & (n_series.index < hi_clip.replace(tzinfo=None))
        ]

        # Align (inner join)
        aligned = pd.DataFrame({"front": f_series, "next": n_series}).dropna()
        if aligned.empty:
            print("    -> alignment empty (no common timestamps)")
            continue

        frames.append(aligned)
        print("    -> " + str(len(aligned)) + " aligned bars")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames).sort_index()
    out.index.name = "time"
    # Remove any remaining NaN or duplicates
    out = out.dropna().loc[~out.index.duplicated(keep="first")]
    return out


def analyse_instrument(base: str):
    print()
    print("=" * 80)
    print("INSTRUMENT: " + base)
    print("=" * 80)

    pairs = build_front_next(base, 2022, 2025)
    if pairs.empty or len(pairs) < ROLL_WIN + 100:
        n = len(pairs) if not pairs.empty else 0
        print("  INSUFFICIENT DATA (" + str(n) + " bars, need " + str(ROLL_WIN + 100) + ") — skip")
        return None

    # ── Structural carry ──────────────────────────────────────────────────────
    ratio = pairs["next"] / pairs["front"] - 1.0
    ann_basis_pct = ratio.mean() / DT_YEARS * 100
    basis_std_pct = ratio.std() / DT_YEARS * 100

    print()
    print("Aligned bars (front+next): " + str(len(pairs)))
    print("Date range: " + str(pairs.index.min().date()) + " .. " + str(pairs.index.max().date()))
    print(
        "Structural annualised basis: {:+.2f}%/yr  (std {:+.2f}%/yr)".format(
            ann_basis_pct, basis_std_pct
        )
    )

    front_vals = pairs["front"].values.astype(float)
    next_vals = pairs["next"].values.astype(float)
    times = pairs.index

    # ── Full-sample backtest ──────────────────────────────────────────────────
    r, trades = backtest_basis_reversion(front_vals, next_vals, return_trades=True)
    if r.get("n", 0) == 0:
        print("Backtest: too few trades / no signals")
        return None

    print()
    print(
        "FULL-SAMPLE basis mean-reversion backtest "
        "(z_entry={}, z_stop={}, roll={}h, max_hold={}h, comm={}):".format(
            Z_ENTRY, Z_STOP, ROLL_WIN, MAX_HOLD, COMMISSION_RT
        )
    )
    print(
        "  trades={n}  win={win:.0%}  total={total_pct:+.2f}%  "
        "avg={avg_pct:+.4f}%  Sharpe={sharpe:+.2f}".format(**r)
    )

    # ── Per-calendar-year breakdown ───────────────────────────────────────────
    print()
    print("PER-YEAR breakdown:")
    print(
        "  {:<6} {:>7} {:>6} {:>8} {:>8}  {}".format(
            "Year", "Trades", "Win%", "Total%", "Sharpe", "Verdict"
        )
    )
    print("  " + "-" * 52)

    # Collect all years that have ANY trade index
    trade_years = set()
    for idx, _ in trades:
        if idx < len(times):
            trade_years.add(times[idx].year)

    # Also include years with data even if no trades
    data_years = set(times.year.unique())
    years = sorted(data_years)

    year_results = {}
    for yr in years:
        yr_trades = [(idx, p) for (idx, p) in trades if idx < len(times) and times[idx].year == yr]
        if not yr_trades:
            verdict = "no trades"
            year_results[yr] = None
            print("  {:<6} {:>7} {:>6} {:>8} {:>8}  {}".format(yr, 0, "-", "-", "-", verdict))
            continue
        pnls = np.array([p for _, p in yr_trades])
        n_yr = len(pnls)
        win_yr = float((pnls > 0).mean())
        tot_yr = float(pnls.sum() * 100)
        sh_yr = float(pnls.mean() / pnls.std() * math.sqrt(n_yr)) if pnls.std() > 0 else 0.0
        flag = "OK" if tot_yr > 0 else "NEG"
        year_results[yr] = tot_yr
        print(
            "  {:<6} {:>7} {:>5.0f}% {:>+8.2f}% {:>+8.2f}  {}".format(
                yr, n_yr, win_yr * 100, tot_yr, sh_yr, flag
            )
        )

    # ── Consistency verdict ───────────────────────────────────────────────────
    valid_years = [yr for yr, v in year_results.items() if v is not None]
    pos_years = [yr for yr, v in year_results.items() if v is not None and v > 0]
    n_valid = len(valid_years)
    n_pos = len(pos_years)
    threshold = max(n_valid - 1, 1)
    robust = n_pos >= threshold

    print()
    print(
        "CONSISTENCY: {}/{} positive years (threshold >= {})  ->  {}".format(
            n_pos, n_valid, threshold, "ROBUST" if robust else "FRAGILE"
        )
    )

    return {
        "base": base,
        "n_bars": len(pairs),
        "ann_basis_pct": ann_basis_pct,
        "basis_std_pct": basis_std_pct,
        "n_trades": r["n"],
        "win_pct": r["win"] * 100,
        "total_pct": r["total_pct"],
        "sharpe": r["sharpe"],
        "n_pos_years": n_pos,
        "n_valid_years": n_valid,
        "robust": robust,
        "year_results": year_results,
    }


def main():
    print("carry_lab.py  — Multi-instrument calendar carry lab")
    print(
        "Params: z_entry={}, z_stop={}, roll={}h, max_hold={}h, comm={}".format(
            Z_ENTRY, Z_STOP, ROLL_WIN, MAX_HOLD, COMMISSION_RT
        )
    )
    print("Fetching next-contract data from MOEX ISS (public API)...")

    bases = ["Si", "GZ", "LK", "SR", "RN", "GK"]
    results = []

    for base in bases:
        res = analyse_instrument(base)
        if res is not None:
            results.append(res)

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print()
    print("=" * 80)
    print("SUMMARY — ranked by structural carry (annualised basis)")
    print("=" * 80)
    print(
        "{:<6} {:>12} {:>12} {:>8} {:>6} {:>9} {:>8}  {:>14}".format(
            "Base", "AnnBasis%/yr", "BasisStd", "Trades", "Win%", "Total%", "Sharpe", "PosYrs/Valid"
        )
    )
    print("-" * 80)

    results_sorted = sorted(results, key=lambda x: x["ann_basis_pct"], reverse=True)
    for r in results_sorted:
        print(
            "{:<6} {:>+12.2f} {:>+12.2f} {:>8} {:>5.0f}% {:>+9.2f}% {:>+8.2f}  {}/{} {}".format(
                r["base"],
                r["ann_basis_pct"],
                r["basis_std_pct"],
                r["n_trades"],
                r["win_pct"],
                r["total_pct"],
                r["sharpe"],
                r["n_pos_years"],
                r["n_valid_years"],
                "ROBUST" if r["robust"] else "FRAGILE",
            )
        )

    print()
    print("VERDICT SUMMARY:")
    robust_list = [r for r in results_sorted if r["robust"]]
    fragile_list = [r for r in results_sorted if not r["robust"]]
    if robust_list:
        print("  ROBUST (consistent calendar carry edge):")
        for r in robust_list:
            print(
                "    {:5s} — {:+.1f}%/yr basis | {}/{} pos years | total {:.2f}%".format(
                    r["base"],
                    r["ann_basis_pct"],
                    r["n_pos_years"],
                    r["n_valid_years"],
                    r["total_pct"],
                )
            )
    else:
        print("  ROBUST: none")
    if fragile_list:
        print("  FRAGILE (inconsistent — do not deploy):")
        for r in fragile_list:
            print(
                "    {:5s} — {:+.1f}%/yr basis | {}/{} pos years | total {:.2f}%".format(
                    r["base"],
                    r["ann_basis_pct"],
                    r["n_pos_years"],
                    r["n_valid_years"],
                    r["total_pct"],
                )
            )
    else:
        print("  FRAGILE: none")


if __name__ == "__main__":
    main()
