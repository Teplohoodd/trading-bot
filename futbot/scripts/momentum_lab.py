"""Momentum strategy backtest on 4 years of hourly FORTS futures.

Tests two momentum families:
  1. Time-Series Momentum (TSMOM) — Moskowitz-Ooi-Pedersen (2012)
     Vol-scaled: pos = sign(ret_L) * (target_vol / realised_vol), clipped ±1
     Lookbacks L in {120, 240, 480, 720} hours.
     Per-instrument + equal-weight diversified portfolio.

  2. Cross-Sectional Momentum (XS) — AQR "Value and Momentum Everywhere"
     Rank all instruments by trailing-L return; long top 1/3, short bottom 1/3.
     Lookbacks L in {240, 480}; rebalance every R in {24, 120} bars.

Consistency verdict: robust if >= (n_years-1) positive years, else fragile.

Run:
    python -u futbot/scripts/momentum_lab.py
"""

import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Constants ───────────────────────────────────────────────────────────────
HOURS_PER_YEAR = 24 * 365
COMMISSION_RT = 0.0008  # round-trip per unit of position change
VOL_WINDOW = 240  # bars for realised-vol estimate
TARGET_VOL_ANN = 0.15  # 15% annual per-instrument vol target
MARGIN_FRAC = 0.15  # conservative avg FORTS margin (10-25%)
TSMOM_LOOKBACKS = [120, 240, 480, 720]
XS_LOOKBACKS = [240, 480]
XS_REBALANCES = [24, 120]
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BASES = ["Si", "LK", "GZ", "SR", "RN", "GK", "MM"]


# ── Helper: annualised Sharpe ────────────────────────────────────────────────
def sharpe(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float)
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(HOURS_PER_YEAR))


def max_drawdown(r: np.ndarray) -> float:
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


# ── Load cached parquets ─────────────────────────────────────────────────────
def load_series() -> dict[str, pd.Series]:
    """Return {base: pd.Series(close, index=DatetimeIndex)} from parquet cache."""
    out = {}
    for base in BASES:
        fp = DATA_DIR / f"hist_{base}.parquet"
        if not fp.exists():
            print(f"  SKIP {base}: parquet not found")
            continue
        df = pd.read_parquet(fp)
        if "time" not in df.columns or "close" not in df.columns:
            print(f"  SKIP {base}: unexpected columns {df.columns.tolist()}")
            continue
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").drop_duplicates("time", keep="last")
        srs = df.set_index("time")["close"].astype(float)
        out[base] = srs
        print(
            f"  {base:4s}: {len(srs)} bars  "
            f"[{srs.index.min().date()} .. {srs.index.max().date()}]"
        )
    return out


# ── Strategy 1: TSMOM (per-instrument) ──────────────────────────────────────
def tsmom_returns(close: np.ndarray, lookback: int) -> np.ndarray:
    """Per-bar notional P&L of vol-scaled TSMOM. Causal only."""
    n = len(close)
    needed = lookback + VOL_WINDOW + 50
    pnl = np.zeros(n)
    if n < needed:
        return pnl
    ret = np.zeros(n)
    ret[1:] = close[1:] / close[:-1] - 1.0
    tgt = TARGET_VOL_ANN / math.sqrt(HOURS_PER_YEAR)

    pos = np.zeros(n)
    start = lookback + VOL_WINDOW
    for t in range(start, n):
        raw = np.sign(close[t] - close[t - lookback])
        rv = ret[t - VOL_WINDOW : t].std()
        pos[t] = float(np.clip(raw * (tgt / rv if rv > 0 else 0.0), -1.0, 1.0))

    for t in range(start, n - 1):
        turn = abs(pos[t] - pos[t - 1])
        pnl[t + 1] = pos[t] * ret[t + 1] - COMMISSION_RT * turn

    return pnl


def per_year_stats(pnl: np.ndarray, times: pd.DatetimeIndex):
    """Returns {year: (annual_return_pct, sharpe)} dict."""
    out = {}
    for yr in sorted(set(times.year)):
        mask = times.year == yr
        r = pnl[mask]
        ann_ret = float(r.sum() * 100)
        sh = sharpe(r)
        out[yr] = (ann_ret, sh)
    return out


# ── Strategy 2: Cross-Sectional Momentum ────────────────────────────────────
def xs_momentum_returns(
    price_matrix: pd.DataFrame,  # columns = bases, aligned hourly index
    lookback: int,
    rebalance: int,
) -> np.ndarray:
    """
    At each rebalance bar: rank instruments by trailing-lookback return.
    Long top 1/3, short bottom 1/3, equal weight, hold until next rebalance.
    Commission on position changes.
    """
    mat = price_matrix.values.astype(float)  # (n_bars, n_inst)
    n, m = mat.shape
    if n < lookback + rebalance + 10 or m < 3:
        return np.zeros(n)

    pnl = np.zeros(n)
    pos = np.zeros(m)

    for t in range(lookback, n - 1):
        # Only rebalance at multiples of rebalance period
        if (t - lookback) % rebalance != 0:
            # Hold position, earn bar return
            bar_ret = np.zeros(m)
            for i in range(m):
                if mat[t - 1, i] > 0 and mat[t, i] > 0:
                    bar_ret[i] = mat[t, i] / mat[t - 1, i] - 1.0
            turn = 0.0
            pnl[t + 1] = float(np.dot(pos, bar_ret)) - COMMISSION_RT * turn
            continue

        # Compute trailing returns
        trailing = np.full(m, np.nan)
        for i in range(m):
            v0 = mat[t - lookback, i]
            v1 = mat[t, i]
            if v0 > 0 and v1 > 0:
                trailing[i] = v1 / v0 - 1.0

        valid = ~np.isnan(trailing)
        if valid.sum() < 3:
            # Not enough instruments — carry position, no rebalance
            bar_ret = np.zeros(m)
            for i in range(m):
                if mat[t - 1, i] > 0 and mat[t, i] > 0:
                    bar_ret[i] = mat[t, i] / mat[t - 1, i] - 1.0
            pnl[t + 1] = float(np.dot(pos, bar_ret))
            continue

        idx_valid = np.where(valid)[0]
        tr_valid = trailing[idx_valid]
        n_valid = len(tr_valid)
        cutoff = n_valid // 3

        ranks = np.argsort(tr_valid)  # ascending
        longs = idx_valid[ranks[-cutoff:]] if cutoff > 0 else np.array([], dtype=int)
        shorts = idx_valid[ranks[:cutoff]] if cutoff > 0 else np.array([], dtype=int)

        n_leg = max(len(longs), len(shorts), 1)
        new_pos = np.zeros(m)
        for i in longs:
            new_pos[i] = 1.0 / n_leg
        for i in shorts:
            new_pos[i] = -1.0 / n_leg

        # Bar return using close[t] → close[t+1]
        bar_ret = np.zeros(m)
        for i in range(m):
            if mat[t, i] > 0 and mat[t + 1, i] > 0:
                bar_ret[i] = mat[t + 1, i] / mat[t, i] - 1.0

        turn = float(np.sum(np.abs(new_pos - pos)))
        pnl[t + 1] = float(np.dot(new_pos, bar_ret)) - COMMISSION_RT * turn
        pos = new_pos

    return pnl


# ── Reporting helpers ────────────────────────────────────────────────────────
def verdict(positive_years: int, n_years: int) -> str:
    return "ROBUST" if positive_years >= n_years - 1 else "FRAGILE"


def print_yearly(stats: dict, label: str):
    """Print per-year table and return (n_positive, n_years)."""
    years = sorted(stats)
    n_pos = sum(1 for y in years if stats[y][0] > 0)
    print(f"\n  {label}")
    print(f"  {'Year':>6}  {'Ret%':>8}  {'Sharpe':>8}  {'Pos?':>6}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*6}")
    for y in years:
        ret, sh = stats[y]
        flag = "YES" if ret > 0 else " no"
        print(f"  {y:>6}  {ret:>8.2f}  {sh:>8.3f}  {flag:>6}")
    v = verdict(n_pos, len(years))
    print(f"  Positive years: {n_pos}/{len(years)}  -> {v}")
    return n_pos, len(years)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("MOMENTUM BACKTEST — FORTS futures 2022-2025, hourly")
    print("Commission: 0.08% round-trip per position change")
    print("=" * 80)

    print("\nLoading data…")
    series = load_series()
    if not series:
        print("ERROR: no data loaded. Check data/*.parquet files.")
        return

    # ── Build aligned matrix for XS momentum ───────────────────────────────
    all_series = {b: s for b, s in series.items()}
    price_mat = pd.DataFrame(all_series).sort_index()
    price_mat.index = pd.to_datetime(price_mat.index)
    # Forward-fill short gaps (≤3 bars) but don't fill across roll gaps
    price_mat = price_mat.ffill(limit=3)

    # ════════════════════════════════════════════════════════════════════════
    # STRATEGY 1 — TSMOM
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("STRATEGY 1: TIME-SERIES MOMENTUM (TSMOM, vol-scaled)")
    print("=" * 80)

    tsmom_summary = []  # (base, lb, n_pos, n_years, overall_sharpe, total_ret)

    for lb in TSMOM_LOOKBACKS:
        print(f"\n--- Lookback L={lb}h ---")
        for base, srs in sorted(series.items()):
            close = srs.values
            times = srs.index
            pnl = tsmom_returns(close, lb)
            active = pnl[lb + VOL_WINDOW + 1 :]
            active_t = times[lb + VOL_WINDOW + 1 :]
            if len(active) < 100:
                continue
            total_ret = float(active.sum() * 100)
            sh = sharpe(active)
            ann_ret = float(active.mean() * HOURS_PER_YEAR * 100)
            stats = per_year_stats(pnl, times)
            # filter to years with meaningful data (>500 bars)
            year_counts = times.year.value_counts()
            valid_stats = {y: v for y, v in stats.items() if year_counts.get(y, 0) > 500}
            n_pos, n_yr = print_yearly(
                valid_stats,
                f"{base} L={lb}h  total={total_ret:+.1f}%  annRet={ann_ret:+.1f}%  "
                f"Sharpe={sh:.3f}",
            )
            tsmom_summary.append(
                dict(
                    base=base,
                    lb=lb,
                    n_pos=n_pos,
                    n_years=n_yr,
                    sharpe=sh,
                    total_ret=total_ret,
                    ann_ret=ann_ret,
                )
            )

    # Diversified TSMOM portfolio per lookback
    print("\n" + "=" * 80)
    print("STRATEGY 1b: TSMOM DIVERSIFIED EQUAL-WEIGHT PORTFOLIO")
    print("=" * 80)

    port_summary = []
    for lb in TSMOM_LOOKBACKS:
        print(f"\n--- Portfolio L={lb}h ---")
        # Align all instrument return paths on common index
        aligned = {}
        for base, srs in series.items():
            pnl = tsmom_returns(srs.values, lb)
            aligned[base] = pd.Series(pnl, index=srs.index)
        mat = pd.DataFrame(aligned).sort_index()
        port = mat.mean(axis=1).fillna(0.0)
        port_arr = port.values
        port_t = port.index

        # Overall stats
        total_ret = float(port_arr.sum() * 100)
        sh = sharpe(port_arr)
        mdd = max_drawdown(port_arr) * 100
        ann_ret = float(port_arr.mean() * HOURS_PER_YEAR * 100)
        on_margin = total_ret / MARGIN_FRAC
        mdd_margin = mdd / MARGIN_FRAC

        print(
            f"  Overall: total={total_ret:+.2f}%  annRet={ann_ret:+.2f}%  "
            f"Sharpe={sh:.3f}  maxDD={mdd:.2f}%"
        )
        print(f"  On-margin (15% margin): total={on_margin:+.1f}%  " f"maxDD={mdd_margin:.1f}%")

        year_counts = port_t.year.value_counts()
        stats = per_year_stats(port_arr, port_t)
        valid_stats = {y: v for y, v in stats.items() if year_counts.get(y, 0) > 500}
        n_pos, n_yr = print_yearly(valid_stats, f"Portfolio L={lb}h")
        v = verdict(n_pos, n_yr)

        port_summary.append(
            dict(
                lb=lb,
                n_pos=n_pos,
                n_years=n_yr,
                sharpe=sh,
                total_ret=total_ret,
                ann_ret=ann_ret,
                mdd=mdd,
                mdd_margin=mdd_margin,
                verdict=v,
            )
        )

    # ════════════════════════════════════════════════════════════════════════
    # STRATEGY 2 — CROSS-SECTIONAL MOMENTUM
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("STRATEGY 2: CROSS-SECTIONAL MOMENTUM (long top-1/3, short bottom-1/3)")
    print("=" * 80)

    xs_summary = []
    for lb in XS_LOOKBACKS:
        for rb in XS_REBALANCES:
            print(f"\n--- XS-MOM L={lb}h R={rb}h ---")
            pnl = xs_momentum_returns(price_mat, lb, rb)
            times = price_mat.index

            total_ret = float(pnl.sum() * 100)
            sh = sharpe(pnl)
            mdd = max_drawdown(pnl) * 100
            ann_ret = float(pnl.mean() * HOURS_PER_YEAR * 100)
            on_margin = total_ret / MARGIN_FRAC
            mdd_margin = mdd / MARGIN_FRAC

            print(
                f"  Overall: total={total_ret:+.2f}%  annRet={ann_ret:+.2f}%  "
                f"Sharpe={sh:.3f}  maxDD={mdd:.2f}%"
            )
            print(f"  On-margin (15% margin): total={on_margin:+.1f}%  " f"maxDD={mdd_margin:.1f}%")

            year_counts = times.year.value_counts()
            stats = per_year_stats(pnl, times)
            valid_stats = {y: v for y, v in stats.items() if year_counts.get(y, 0) > 500}
            n_pos, n_yr = print_yearly(valid_stats, f"XS-MOM L={lb}h R={rb}h")
            v = verdict(n_pos, n_yr)

            xs_summary.append(
                dict(
                    lb=lb,
                    rb=rb,
                    n_pos=n_pos,
                    n_years=n_yr,
                    sharpe=sh,
                    total_ret=total_ret,
                    ann_ret=ann_ret,
                    mdd=mdd,
                    mdd_margin=mdd_margin,
                    verdict=v,
                )
            )

    # ════════════════════════════════════════════════════════════════════════
    # FINAL VERDICT TABLE
    # ════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL VERDICT SUMMARY")
    print("=" * 80)

    print("\n-- TSMOM Per-Instrument (best lookback per instrument by Sharpe) --")
    df_ts = pd.DataFrame(tsmom_summary)
    if not df_ts.empty:
        best_per = df_ts.sort_values("sharpe", ascending=False).groupby("base").head(1)
        print(
            f"  {'Base':>5}  {'LB':>5}  {'Sharpe':>8}  {'AnnRet%':>8}  "
            f"{'PosYrs':>7}  {'Verdict':>10}"
        )
        for _, r in best_per.sort_values("sharpe", ascending=False).iterrows():
            v = verdict(int(r["n_pos"]), int(r["n_years"]))
            print(
                f"  {r['base']:>5}  {int(r['lb']):>5}  {r['sharpe']:>8.3f}  "
                f"{r['ann_ret']:>8.2f}  {int(r['n_pos'])}/{int(r['n_years']):>2}  "
                f"{v:>10}"
            )

    print("\n-- TSMOM Diversified Portfolio --")
    print(
        f"  {'LB':>5}  {'Sharpe':>8}  {'AnnRet%':>8}  {'maxDD%':>8}  "
        f"{'mddMarg%':>9}  {'PosYrs':>7}  {'Verdict':>10}"
    )
    for r in port_summary:
        print(
            f"  {r['lb']:>5}  {r['sharpe']:>8.3f}  {r['ann_ret']:>8.2f}  "
            f"{r['mdd']:>8.2f}  {r['mdd_margin']:>9.1f}  "
            f"{r['n_pos']}/{r['n_years']:>2}  {r['verdict']:>10}"
        )

    print("\n-- Cross-Sectional Momentum --")
    print(
        f"  {'LB':>5}  {'RB':>5}  {'Sharpe':>8}  {'AnnRet%':>8}  {'maxDD%':>8}  "
        f"{'mddMarg%':>9}  {'PosYrs':>7}  {'Verdict':>10}"
    )
    for r in xs_summary:
        print(
            f"  {r['lb']:>5}  {r['rb']:>5}  {r['sharpe']:>8.3f}  "
            f"{r['ann_ret']:>8.2f}  {r['mdd']:>8.2f}  {r['mdd_margin']:>9.1f}  "
            f"{r['n_pos']}/{r['n_years']:>2}  {r['verdict']:>10}"
        )

    print("\n-- MIRAGE FLAGS --")
    all_verdicts = [
        (f"TSMOM-port L={r['lb']}h", r["verdict"], r["sharpe"]) for r in port_summary
    ] + [(f"XS L={r['lb']}h R={r['rb']}h", r["verdict"], r["sharpe"]) for r in xs_summary]
    robust = [(n, sh) for n, v, sh in all_verdicts if v == "ROBUST"]
    fragile = [(n, sh) for n, v, sh in all_verdicts if v == "FRAGILE"]
    if robust:
        print(f"  Variants with ROBUST verdict: {[n for n, _ in robust]}")
        print(f"  NOTE: 'robust' here means >= (n_years-1) positive years on")
        print(f"        notional return, no leverage, 7 instruments only.")
        print(f"        Positional Sharpes are small — see if any exceed 0.5.")
    if fragile:
        print(
            f"  Variants with FRAGILE verdict (in-sample mirages?): " f"{[n for n, _ in fragile]}"
        )
    if not robust:
        print("  ALL variants are FRAGILE — no consistent momentum edge found.")

    print("\n" + "=" * 80)
    print("END")
    print("=" * 80)


if __name__ == "__main__":
    main()
