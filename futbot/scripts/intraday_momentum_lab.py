"""
Market Intraday Momentum Backtest
==================================
Based on: Gao, Han, Li, Zhou (2018) "Market intraday momentum", J. Financial Economics

Strategy: the return of the FIRST bar of the trading session predicts the return of the
LAST bar of the session (same sign).

Session structure decision:
- FORTS trades a main session + evening session.
- In 2022-2025, a full day has 15-17 hourly bars (7:00 or 8:00 to 23:00 Moscow time).
- During the Mar-2022 market halt period some days have only 9 bars (10:00-18:00 main session only).
- Decision: treat the ENTIRE calendar day (all bars from first to last) as one session.
  This captures both main and evening sessions — the evening session is part of FORTS and
  shares the same contract, so intraday momentum should apply across the full day.
- Minimum 4 bars per day required to form a first/last signal.

Variants tested:
  V1 (base): signal = first 1 bar return, trade = last 1 bar return
  V2: signal = first 2 bars return, trade = last 2 bars return
  V3: signal = first 1 bar return, trade = rest-of-day return (enter after bar 1, exit at close)
"""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("F:/trade_claude/data")
BASES = ["Si", "LK", "GZ", "SR", "RN", "GK", "MM"]
COMMISSION = 0.0008  # round-trip per trade
MIN_BARS_PER_DAY = 4  # require at least this many bars to form a valid signal
BACKTEST_START = "2022-01-01"


def load_and_prep(base: str) -> pd.DataFrame:
    """Load parquet, filter to backtest period, add date/hour columns."""
    path = DATA_DIR / f"hist_{base}.parquet"
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"] >= BACKTEST_START].copy()
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df = df.sort_values("time").reset_index(drop=True)
    return df


def build_daily_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each day build:
      - r_first1: return of first bar (close/open - 1)
      - r_last1:  return of last bar (close/open - 1)
      - r_first2: return of first 2 bars (close of bar2 / open of bar1 - 1)
      - r_last2:  return of last 2 bars (close of last bar / open of second-to-last bar - 1)
      - r_rest:   return from end of first bar to session close
                  (close of last bar / close of first bar - 1)
    """
    records = []
    for date, grp in df.groupby("date"):
        grp = grp.sort_values("time")
        n = len(grp)
        if n < MIN_BARS_PER_DAY:
            continue
        bars = grp[["time", "open", "close"]].reset_index(drop=True)

        # V1: first bar signal, last bar trade
        r_first1 = bars.loc[0, "close"] / bars.loc[0, "open"] - 1
        r_last1 = bars.loc[n - 1, "close"] / bars.loc[n - 1, "open"] - 1

        # V2: first 2 bars signal, last 2 bars trade
        if n >= 5:
            r_first2 = bars.loc[1, "close"] / bars.loc[0, "open"] - 1
            r_last2 = bars.loc[n - 1, "close"] / bars.loc[n - 2, "open"] - 1
        else:
            r_first2 = np.nan
            r_last2 = np.nan

        # V3: first bar signal, rest-of-day trade (enter at open of bar 2, exit at close of last bar)
        # r_rest = close_last / close_first - 1 (approximation: first close = entry price)
        r_rest = bars.loc[n - 1, "close"] / bars.loc[0, "close"] - 1

        year = pd.Timestamp(date).year
        records.append(
            {
                "date": date,
                "year": year,
                "r_first1": r_first1,
                "r_last1": r_last1,
                "r_first2": r_first2,
                "r_last2": r_last2,
                "r_rest": r_rest,
                "n_bars": n,
            }
        )
    return pd.DataFrame(records)


def compute_pnl(signals: pd.DataFrame, sig_col: str, ret_col: str, label: str) -> pd.DataFrame:
    """
    Given daily signal and return columns, compute P&L:
      position = sign(signal)  (0 if signal == 0)
      pnl = position * ret_col - COMMISSION * |position|
    Returns per-day pnl DataFrame with a 'pnl' column.
    """
    valid = signals[[sig_col, ret_col, "date", "year"]].dropna().copy()
    valid["position"] = np.sign(valid[sig_col])
    valid["pnl"] = valid["position"] * valid[ret_col] - COMMISSION * valid["position"].abs()
    valid["label"] = label
    return valid


def yearly_stats(pnl_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-year stats: n_trades, win%, total_pct, Sharpe."""
    rows = []
    for year, grp in pnl_df.groupby("year"):
        n = len(grp)
        wins = (grp["pnl"] > 0).sum()
        total = grp["pnl"].sum() * 100  # in percent
        if grp["pnl"].std() > 0 and n > 1:
            sharpe = grp["pnl"].mean() / grp["pnl"].std() * np.sqrt(252)
        else:
            sharpe = np.nan
        rows.append(
            {
                "year": year,
                "n_trades": n,
                "win_pct": 100 * wins / n if n > 0 else np.nan,
                "total_pct": total,
                "sharpe": sharpe,
                "positive": total > 0,
            }
        )
    return pd.DataFrame(rows)


def print_separator(char="-", width=80):
    print(char * width)


def run_instrument(base: str):
    df = load_and_prep(base)
    signals = build_daily_signals(df)

    years = sorted(signals["year"].unique())
    n_years = len(years)

    print()
    print_separator("=")
    print(
        f'INSTRUMENT: {base}   ({signals["date"].min()} to {signals["date"].max()}, '
        f'{signals["date"].nunique()} trading days, {n_years} years)'
    )
    print_separator("=")

    # --- Session structure note ---
    bars_dist = signals["n_bars"].value_counts().sort_index()
    modal = signals["n_bars"].mode()[0]
    print(
        f'  Bars/day: modal={modal}, min={signals["n_bars"].min()}, max={signals["n_bars"].max()}'
    )

    variants = [
        ("V1 first1 -> last1", "r_first1", "r_last1"),
        ("V2 first2 -> last2", "r_first2", "r_last2"),
        ("V3 first1 -> rest  ", "r_first1", "r_rest"),
    ]

    results_summary = []

    for label, sig_col, ret_col in variants:
        pnl_df = compute_pnl(signals, sig_col, ret_col, label)
        if len(pnl_df) == 0:
            print(f"\n  {label}: no data")
            continue

        ys = yearly_stats(pnl_df)
        positive_years = int(ys["positive"].sum())
        verdict = (
            "ROBUST"
            if positive_years >= (n_years - 1)
            else ("MARGINAL" if positive_years >= n_years // 2 else "NOISE")
        )

        print()
        print(
            f"  Variant: {label}   --> VERDICT: {verdict} ({positive_years}/{n_years} positive years)"
        )
        print(f'  {"Year":<6} {"N_trades":>8} {"Win%":>7} {"Total%":>8} {"Sharpe":>8} {"Pos?":>5}')
        print(f'  {"-"*6} {"-"*8} {"-"*7} {"-"*8} {"-"*8} {"-"*5}')
        for _, row in ys.iterrows():
            pos_mark = "YES" if row["positive"] else "NO"
            sharpe_str = f"{row['sharpe']:.2f}" if not np.isnan(row["sharpe"]) else "  N/A"
            print(
                f'  {int(row["year"]):<6} {int(row["n_trades"]):>8} {row["win_pct"]:>7.1f} '
                f'{row["total_pct"]:>8.2f} {sharpe_str:>8} {pos_mark:>5}'
            )

        # Overall stats
        all_pnl = pnl_df["pnl"]
        overall_total = all_pnl.sum() * 100
        overall_sharpe = (
            all_pnl.mean() / all_pnl.std() * np.sqrt(252) if all_pnl.std() > 0 else np.nan
        )
        overall_win = 100 * (all_pnl > 0).mean()
        sharpe_str = f"{overall_sharpe:.2f}" if not np.isnan(overall_sharpe) else "N/A"
        print(
            f'  {"TOTAL":<6} {len(pnl_df):>8} {overall_win:>7.1f} {overall_total:>8.2f} {sharpe_str:>8}'
        )

        results_summary.append(
            {
                "instrument": base,
                "variant": label,
                "positive_years": positive_years,
                "n_years": n_years,
                "verdict": verdict,
                "total_pct": overall_total,
                "sharpe": overall_sharpe,
            }
        )

    return results_summary


def autocorr_check(base: str):
    """Compute Spearman correlation between first-bar return and last-bar return per year."""
    from scipy.stats import spearmanr

    df = load_and_prep(base)
    signals = build_daily_signals(df)
    valid = signals[["year", "r_first1", "r_last1"]].dropna()
    print(f"\n  Autocorr check {base} (Spearman r_first1 vs r_last1):")
    print(f'  {"Year":<6} {"N":>5} {"Spearman_r":>12} {"p-val":>10}')
    for year, grp in valid.groupby("year"):
        if len(grp) < 10:
            continue
        r, p = spearmanr(grp["r_first1"], grp["r_last1"])
        sig_mark = "*" if p < 0.10 else ""
        print(f"  {year:<6} {len(grp):>5} {r:>12.4f} {p:>10.4f} {sig_mark}")
    # Overall
    if len(valid) > 10:
        r, p = spearmanr(valid["r_first1"], valid["r_last1"])
        sig_mark = "*" if p < 0.10 else ""
        print(f'  {"ALL":<6} {len(valid):>5} {r:>12.4f} {p:>10.4f} {sig_mark}')


def main():
    print("=" * 80)
    print("MARKET INTRADAY MOMENTUM BACKTEST")
    print("Gao, Han, Li, Zhou (2018) on FORTS hourly futures")
    print("Backtest period: 2022-01-01 onwards  |  Commission: 0.08% round-trip")
    print()
    print("SESSION STRUCTURE DECISION:")
    print("  FORTS has a main session (~10:00-18:50 MSK) + evening session (~19:05-23:50 MSK).")
    print("  The data includes both sessions as a continuous bar sequence per calendar day.")
    print("  Decision: treat ENTIRE calendar day (first bar to last bar) as one session.")
    print("  The evening session is liquid and part of the same contract; treating them")
    print("  together maximises sample size and matches how continuous traders operate.")
    print("  Mar-2022 halt period (main-session-only, 9 bars/day) is included but filtered")
    print("  if fewer than 4 bars — all such days pass the 4-bar threshold.")
    print("=" * 80)

    all_results = []
    for base in BASES:
        try:
            results = run_instrument(base)
            all_results.extend(results)
        except Exception as e:
            print(f"\nERROR on {base}: {e}")
            import traceback

            traceback.print_exc()

    # --- Cross-instrument autocorrelation check ---
    print()
    print_separator("=")
    print("STATISTICAL SIGNAL CHECK: Spearman rank correlation (first-bar vs last-bar return)")
    print("  * = p < 0.10")
    print_separator("=")
    try:
        from scipy import stats as _stats  # noqa: just testing availability

        for base in BASES:
            try:
                autocorr_check(base)
            except Exception as e:
                print(f"  autocorr error on {base}: {e}")
    except ImportError:
        print("  scipy not available, skipping correlation check")

    # --- Summary table ---
    print()
    print_separator("=")
    print("SUMMARY TABLE (V1: first bar -> last bar)")
    print_separator("=")
    v1 = [r for r in all_results if "first1 -> last1" in r["variant"]]
    if v1:
        print(f'  {"Instr":<6} {"PosYrs":>7} {"Verdict":<10} {"Total%":>8} {"Sharpe":>8}')
        print(f'  {"-"*6} {"-"*7} {"-"*10} {"-"*8} {"-"*8}')
        for r in sorted(v1, key=lambda x: x["total_pct"], reverse=True):
            sharpe_str = f"{r['sharpe']:.2f}" if not np.isnan(r["sharpe"]) else "  N/A"
            print(
                f'  {r["instrument"]:<6} {r["positive_years"]}/{r["n_years"]}     '
                f'{r["verdict"]:<10} {r["total_pct"]:>8.2f} {sharpe_str:>8}'
            )

    print()
    print_separator("=")
    print("FINAL VERDICT")
    print_separator("=")
    robust = [r for r in v1 if r["verdict"] == "ROBUST"]
    marginal = [r for r in v1 if r["verdict"] == "MARGINAL"]
    noise = [r for r in v1 if r["verdict"] == "NOISE"]
    print(
        f"  ROBUST  ({len(robust)}): "
        + (", ".join(r["instrument"] for r in robust) if robust else "none")
    )
    print(
        f"  MARGINAL ({len(marginal)}): "
        + (", ".join(r["instrument"] for r in marginal) if marginal else "none")
    )
    print(
        f"  NOISE   ({len(noise)}): "
        + (", ".join(r["instrument"] for r in noise) if noise else "none")
    )
    print()
    if not robust:
        print("  Market intraday momentum does NOT hold robustly on these FORTS futures.")
        print("  The effect is either absent or too inconsistent across years to trade.")
    elif len(robust) <= 2:
        print("  Market intraday momentum holds for a minority of instruments.")
        print("  Effect is instrument-specific, not a broad market-wide phenomenon on FORTS.")
    else:
        print("  Market intraday momentum shows broad support across FORTS futures.")
    print_separator("=")


if __name__ == "__main__":
    main()
