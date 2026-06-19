"""Trend-following strategy lab — per-year consistency across 4 years of FORTS futures.

Strategies tested (causal, no lookahead):
  1. Donchian channel breakout (Turtle-style): N in {20, 55, 100}
  2. EMA crossover: (fast, slow) in {(20,100),(50,200)}
  3. KAMA(10,2,30) adaptive trend

Judged by PER-YEAR consistency (2022-2025), not headline return.
Commission: 0.0008 round-trip per position change.
VERDICT: robust if >= (n_years - 1) positive years, else fragile.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from futbot.scripts.moex_iss_history import continuous_hourly

BASES = ["Si", "LK", "GZ", "SR", "RN", "GK", "MM"]
COMMISSION_RT = 0.0008
Y0, Y1 = 2022, 2025
MARGIN_FRAC = 0.15  # typical FORTS initial margin ~10-25%; use 15% midpoint
# Only count these as the "target" years for consistency; 2021 is partial warmup
TARGET_YEARS = {2022, 2023, 2024, 2025}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load(base):
    cache = DATA_DIR / f"hist_{base}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = continuous_hourly(base, Y0, Y1)
        if not df.empty:
            df.to_parquet(cache)
    if df.empty:
        return None
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")
    return df["close"].astype(float)


# ---------------------------------------------------------------------------
# Indicator helpers (all causal)
# ---------------------------------------------------------------------------


def donchian_signals(close: np.ndarray, n: int):
    """
    Enter long when close breaks above n-bar high (of prior n bars, not including
    current bar to avoid lookahead).
    Enter short when close breaks below n-bar low.
    Exit long when close drops below n/2-bar low; exit short above n/2-bar high.
    Returns pos array (+1 long, -1 short, 0 flat).
    """
    m = len(close)
    n2 = max(1, n // 2)
    pos = np.zeros(m, dtype=float)
    cur = 0.0
    for t in range(n, m):
        hi_n = close[t - n : t].max()  # highest of prior n bars
        lo_n = close[t - n : t].min()
        hi_n2 = close[t - n2 : t].max()  # highest of prior n/2 bars
        lo_n2 = close[t - n2 : t].min()
        if cur == 0:
            if close[t] > hi_n:
                cur = 1.0
            elif close[t] < lo_n:
                cur = -1.0
        elif cur == 1:
            if close[t] < lo_n2:
                cur = 0.0
            # re-entry on opposite side checked next bar
        elif cur == -1:
            if close[t] > hi_n2:
                cur = 0.0
        pos[t] = cur
    return pos


def ema_crossover_signals(close: np.ndarray, fast: int, slow: int):
    """Long when fast EMA > slow EMA, short when fast < slow, flat otherwise."""
    m = len(close)
    pos = np.zeros(m, dtype=float)
    af = 2.0 / (fast + 1)
    as_ = 2.0 / (slow + 1)
    ema_f = close[0]
    ema_s = close[0]
    for t in range(1, m):
        ema_f = ema_f + af * (close[t] - ema_f)
        ema_s = ema_s + as_ * (close[t] - ema_s)
        if t >= slow:  # need enough bars for EMA to stabilize
            if ema_f > ema_s:
                pos[t] = 1.0
            elif ema_f < ema_s:
                pos[t] = -1.0
    return pos


def kama_signals(close: np.ndarray, n: int = 10, fast: int = 2, slow: int = 30):
    """
    KAMA(10,2,30): efficiency ratio over n bars, adaptive smoothing constant.
    Long when close > KAMA and KAMA rising, short when close < KAMA and KAMA falling.
    """
    m = len(close)
    pos = np.zeros(m, dtype=float)
    kama = close[0]
    kama_prev = close[0]
    sc_fast = 2.0 / (fast + 1)
    sc_slow = 2.0 / (slow + 1)
    for t in range(1, m):
        if t >= n:
            direction = abs(close[t] - close[t - n])
            volatility = np.sum(np.abs(np.diff(close[t - n : t + 1])))
            er = direction / volatility if volatility > 0 else 0.0
            sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2
        else:
            sc = sc_slow**2
        kama_prev = kama
        kama = kama + sc * (close[t] - kama)
        if t >= n + 5:  # allow stabilization
            rising = kama > kama_prev
            if close[t] > kama and rising:
                pos[t] = 1.0
            elif close[t] < kama and not rising:
                pos[t] = -1.0
    return pos


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


def backtest(close: np.ndarray, times: pd.DatetimeIndex, pos_signal: np.ndarray, label: str):
    """
    Given a position signal (causal, set at bar t, trade executes at t+1 open
    — approximated as same bar close), compute per-trade stats and per-year PnL.

    Returns: dict with per-year breakdown and aggregate stats.
    """
    n = len(close)
    assert len(pos_signal) == n
    # Compute bar returns
    ret = np.zeros(n)
    ret[1:] = close[1:] / close[:-1] - 1.0

    # Vectorized PnL: position held from bar t earns ret[t+1], commission on changes
    pnl = np.zeros(n)
    pos_prev = 0.0
    for t in range(n - 1):
        p = pos_signal[t]
        comm = COMMISSION_RT * abs(p - pos_prev)
        pnl[t + 1] = p * ret[t + 1] - comm
        pos_prev = p

    # Trade-level stats: identify entry/exit events
    # A "trade" = a continuous non-zero position block
    trades_pnl = []
    trade_acc = 0.0
    in_trade = False
    for t in range(1, n):
        p_prev = pos_signal[t - 1]
        p_curr = pos_signal[t]
        if p_prev != 0:
            trade_acc += pnl[t]
        if p_prev != 0 and (p_curr == 0 or p_curr != p_prev):
            # position closed or reversed — record trade
            trades_pnl.append(trade_acc)
            trade_acc = 0.0
        if p_curr != 0 and p_prev == 0:
            trade_acc = 0.0  # start fresh accumulator

    trades_arr = np.array(trades_pnl) if trades_pnl else np.array([0.0])

    # Per-year breakdown
    years_data = {}
    years = pd.DatetimeIndex(times).year
    for yr in sorted(set(years)):
        mask = years == yr
        yr_pnl = pnl[mask]
        yr_total = yr_pnl.sum() * 100
        # Trades that start in this year
        yr_trades = []
        in_t = False
        acc = 0.0
        for t in range(n):
            if years[t] != yr:
                if in_t:
                    yr_trades.append(acc)
                    acc = 0.0
                    in_t = False
                continue
            if pos_signal[t] != 0:
                if not in_t:
                    in_t = True
                    acc = 0.0
                acc += pnl[t]
            else:
                if in_t:
                    yr_trades.append(acc)
                    acc = 0.0
                    in_t = False
        if in_t:
            yr_trades.append(acc)

        yr_arr = np.array(yr_trades) if yr_trades else np.array([0.0])
        n_t = len(yr_arr)
        win_pct = float((yr_arr > 0).mean() * 100) if n_t > 0 else 0.0
        # per-trade Sharpe (annualized by trade count relative to 1y)
        if n_t > 1 and yr_arr.std() > 0:
            pt_sharpe = yr_arr.mean() / yr_arr.std() * np.sqrt(n_t)
        else:
            pt_sharpe = 0.0
        years_data[yr] = dict(
            n_trades=n_t, win_pct=win_pct, total_pct=yr_total, pt_sharpe=pt_sharpe
        )

    return years_data


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _verdict(years_data: dict) -> tuple:
    """Judge only TARGET_YEARS (2022-2025); 2021 is partial warmup, excluded."""
    target = sorted(y for y in years_data.keys() if y in TARGET_YEARS)
    n_pos = sum(1 for y in target if years_data[y]["total_pct"] > 0)
    n_years = len(target)
    label = "ROBUST" if (n_years > 0 and n_pos >= (n_years - 1)) else "FRAGILE"
    return label, n_pos, n_years


def print_result(base: str, strategy: str, config: str, years_data: dict):
    label, n_pos, n_years = _verdict(years_data)
    total_all = sum(v["total_pct"] for v in years_data.values())
    margin_note = f"on-margin={total_all / MARGIN_FRAC:+.0f}%"
    flag = "ROBUST" if label == "ROBUST" else "fragile"
    print(
        f"\n  [{base}] {strategy} {config}  => {flag} ({n_pos}/{n_years} pos yrs)  "
        f"notional={total_all:+.1f}%  {margin_note}"
    )
    print(f"  {'year':>4}  {'n_tr':>5}  {'win%':>5}  {'ret%':>7}  {'pt_sharpe':>10}")
    for yr in sorted(years_data.keys()):
        d = years_data[yr]
        mark = "+" if d["total_pct"] > 0 else "-"
        print(
            f"  {yr:>4}  {d['n_trades']:>5}  {d['win_pct']:>4.0f}%  "
            f"  {d['total_pct']:>+6.1f}%  {d['pt_sharpe']:>+9.2f}  {mark}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 90)
    print("TREND-FOLLOWING LAB — per-year consistency on FORTS hourly futures 2022-2025")
    print(f"Commission: {COMMISSION_RT} round-trip | Margin fraction: {MARGIN_FRAC:.0%}")
    print("=" * 90)

    # Collect best configs per (base, strategy) for summary
    summary = []

    for base in BASES:
        srs = _load(base)
        if srs is None or len(srs) < 300:
            print(f"\n{base}: SKIP (no data)")
            continue
        close = srs.values
        times = srs.index

        print(f"\n{'=' * 60}")
        print(
            f"  {base}  ({len(close)} bars, "
            f"{str(times.min().date())} .. {str(times.max().date())})"
        )
        print(f"{'=' * 60}")

        # --- Strategy 1: Donchian ---
        print("\n  -- DONCHIAN CHANNEL BREAKOUT --")
        best_dc = None
        best_dc_npos = -1
        best_dc_nyears = len(TARGET_YEARS)
        for N in [20, 55, 100]:
            pos = donchian_signals(close, N)
            yd = backtest(close, times, pos, f"Donchian(N={N})")
            print_result(base, "Donchian", f"N={N}", yd)
            lbl, n_pos, n_years = _verdict(yd)
            if n_pos > best_dc_npos:
                best_dc_npos = n_pos
                best_dc = f"N={N}"
                best_dc_nyears = n_years
        summary.append((base, "Donchian", best_dc, best_dc_npos, best_dc_nyears))

        # --- Strategy 2: EMA crossover ---
        print("\n  -- EMA CROSSOVER --")
        best_ema = None
        best_ema_npos = -1
        best_ema_nyears = 4
        for fast, slow in [(20, 100), (50, 200)]:
            pos = ema_crossover_signals(close, fast, slow)
            yd = backtest(close, times, pos, f"EMA({fast},{slow})")
            print_result(base, "EMA", f"({fast},{slow})", yd)
            lbl, n_pos, n_years = _verdict(yd)
            if n_pos > best_ema_npos:
                best_ema_npos = n_pos
                best_ema = f"({fast},{slow})"
                best_ema_nyears = n_years
        summary.append((base, "EMA", best_ema, best_ema_npos, best_ema_nyears))

        # --- Strategy 3: KAMA ---
        print("\n  -- KAMA ADAPTIVE TREND --")
        pos = kama_signals(close, n=10, fast=2, slow=30)
        yd = backtest(close, times, pos, "KAMA(10,2,30)")
        print_result(base, "KAMA", "(10,2,30)", yd)
        lbl, n_pos, n_years = _verdict(yd)
        summary.append((base, "KAMA", "(10,2,30)", n_pos, n_years))

    # --- Summary table ---
    n_target = len(TARGET_YEARS)
    print("\n\n" + "=" * 90)
    print("SUMMARY TABLE — best config per (instrument x strategy)")
    print(
        f"VERDICT: ROBUST = >= {n_target - 1} of {n_target} target years (2022-2025) positive; else FRAGILE"
    )
    print("=" * 90)
    print(f"  {'base':<5}  {'strategy':<10}  {'best_cfg':<12}  {'pos_yrs':>10}  {'verdict':<10}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*10}")
    for base, strat, cfg, npos, nyrs in summary:
        verdict = "ROBUST" if npos >= (nyrs - 1) and nyrs > 0 else "FRAGILE"
        print(f"  {base:<5}  {strat:<10}  {cfg:<12}  {npos:>4}/{nyrs:<4}   {verdict}")

    print("\n" + "=" * 90)
    print("NOTES:")
    print(
        f"  * Notional returns. On-margin (FORTS ~{MARGIN_FRAC:.0%} init margin) multiply by "
        f"~{1/MARGIN_FRAC:.0f}x."
    )
    print(
        "  * All signals causal (bar-t uses only bars <= t). No parameter search = "
        "no fitting bias."
    )
    print(
        "  * 'ROBUST' is a low bar — one bad year forgiven. Look at per-year column "
        "for real consistency."
    )
    print(
        "  * KAMA and EMA crossovers may have very few trades in slow years — "
        "check n_trades vs noise."
    )


if __name__ == "__main__":
    main()
