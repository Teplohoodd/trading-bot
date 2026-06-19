"""Cross-sectional rotation: momentum + skew on FORTS hourly.

Two strategies sharing one rotation harness:

  XSMOM: at every REBAL bars, rank the universe by trailing R-bar return,
         long the top-K, short the bottom-K, hold REBAL bars.
         Different from per-instrument TSMOM (we already disproved that):
         this is RELATIVE strength inside the basket.

  XSSKEW: same harness, rank by rolling SKEW of returns (Carver).  Long
          negative-skew (crash-prone, paid for risk), short positive-skew.

Both equal-weighted across the K longs+K shorts.  Causal, vol-scaled by
inverse realised vol so each leg contributes roughly equal risk.

Kill criterion (per spec): SR ≥ 0.5 AND positive years ≥ 3/4, else drop.
"""

import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASES = ["Si", "LK", "GZ", "SR", "RN", "GK", "MM"]  # 7 instruments with data
COMMISSION_RT = 0.0008
HOURS_PER_YEAR = 24 * 365


def _load_aligned() -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parents[2] / "data"
    series = {}
    for b in BASES:
        p = data_dir / f"hist_{b}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_values("time")
        df["time"] = pd.to_datetime(df["time"])
        series[b] = df.set_index("time")["close"].astype(float)
    return pd.concat(series, axis=1, join="inner").dropna()


def _stats(pnl: np.ndarray, times: pd.Index, costs: float):
    pnl = pnl[~np.isnan(pnl)]
    if len(pnl) == 0 or pnl.std() == 0:
        return None
    sh = pnl.mean() / pnl.std() * np.sqrt(HOURS_PER_YEAR)
    cum = np.cumsum(pnl)
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    df = pd.DataFrame({"r": pnl}, index=times[-len(pnl) :])
    df["y"] = df.index.year
    by_y = df.groupby("y")["r"].sum() * 100
    return {
        "sharpe": float(sh),
        "total": float(pnl.sum() * 100),
        "mdd": mdd * 100,
        "by_year": {int(y): float(v) for y, v in by_y.items()},
        "pos_years": int((by_y > 0).sum()),
        "n_years": int(by_y.size),
    }


def backtest_rotation(prices: pd.DataFrame, *, signal: str, lookback: int, rebal: int, k: int):
    """signal in {'mom','skew'}.  Long top-k by signal, short bottom-k."""
    n, m = prices.shape
    p = prices.values
    ret = np.zeros_like(p)
    ret[1:] = p[1:] / p[:-1] - 1.0
    # vol scaling (per-instrument)
    vol = pd.DataFrame(ret).rolling(240).std().values
    pos = np.zeros_like(p)
    turn = np.zeros(n)
    for t in range(lookback + 240, n, rebal):
        # signal
        if signal == "mom":
            sig = ret[t - lookback : t].sum(axis=0)
        else:  # skew
            r = ret[t - lookback : t]
            mu = r.mean(axis=0)
            sd = r.std(axis=0)
            sk = ((r - mu) ** 3).mean(axis=0) / (sd**3 + 1e-12)
            sig = -sk  # long negative skew → invert
        order = np.argsort(sig)  # ascending
        shorts = order[:k]
        longs = order[-k:]
        v = vol[t]
        # inverse-vol weights inside each leg
        wl = np.zeros(m)
        ws = np.zeros(m)
        wl_raw = 1.0 / np.maximum(v[longs], 1e-6)
        wl_raw /= wl_raw.sum()
        ws_raw = 1.0 / np.maximum(v[shorts], 1e-6)
        ws_raw /= ws_raw.sum()
        new = np.zeros(m)
        new[longs] = wl_raw
        new[shorts] = -ws_raw
        # apply for next REBAL bars
        end = min(n, t + rebal)
        pos[t:end] = new
        turn[t] = np.abs(new - pos[t - 1]).sum() if t > 0 else np.abs(new).sum()
    # P&L from t to t+1 using pos at t
    pnl = (pos * ret).sum(axis=1)
    # leg cost = COMMISSION_RT per unit gross turnover (sum of |delta weights|)
    pnl -= turn * COMMISSION_RT
    return _stats(pnl[lookback + 240 + 1 :], prices.index[lookback + 240 + 1 :], COMMISSION_RT)


def main():
    prices = _load_aligned()
    n, m = prices.shape
    print("=" * 96)
    print(
        f"CROSS-SECTIONAL ROTATION — universe {list(prices.columns)} " f"| {n} aligned hourly bars"
    )
    print("=" * 96)
    # Reasonable grids (not over-mined: 3 lookbacks × 2 rebals × 2 K)
    grid = list(product(["mom", "skew"], [120, 240, 480], [24, 72], [2, 3]))
    out = []
    for sig, lb, rb, k in grid:
        r = backtest_rotation(prices, signal=sig, lookback=lb, rebal=rb, k=k)
        if r:
            r["cfg"] = f"{sig}/lb{lb}/rb{rb}/k{k}"
            out.append(r)
    # rank by Sharpe
    out.sort(key=lambda x: -x["sharpe"])
    print(f"\n{'config':<24}{'Sharpe':>8}{'total%':>9}{'mdd%':>8}" f"{'+years':>10}  per-year %")
    print("-" * 96)
    survivors = []
    for r in out:
        py = " ".join(f"{y}:{v:+.0f}" for y, v in r["by_year"].items())
        keep = r["sharpe"] >= 0.5 and r["pos_years"] >= 3
        flag = "  KEEP" if keep else ""
        if keep:
            survivors.append(r)
        print(
            f"{r['cfg']:<24}{r['sharpe']:>+8.2f}{r['total']:>+9.1f}"
            f"{r['mdd']:>+8.1f}{r['pos_years']:>5}/{r['n_years']}  {py}{flag}"
        )
    print("-" * 96)
    print(f"\nKill gate: Sharpe ≥ 0.5 AND positive years ≥ 3 of 4.")
    print(f"Survivors: {len(survivors)}/{len(out)}")
    if not survivors:
        print(
            "→ No cross-sectional rotation passes our consistency gate. "
            "Honest answer: stick with triple_top + carry."
        )
    else:
        for s in survivors:
            print(
                f"  ✓ {s['cfg']}  Sharpe {s['sharpe']:+.2f}  "
                f"total {s['total']:+.0f}%  +{s['pos_years']}/{s['n_years']}"
            )


if __name__ == "__main__":
    main()
