"""HOURLY pairs walk-forward across 4 years — the FAIR test of the live strategy.

The daily test used slow params and didn't represent the hourly bot.  Using
MOEX ISS stitched continuous hourly futures (2022-2025), we now run the EXACT
live strategy params (240h rolling window, 48h max hold, z_entry 2) causally
across 4 years / many regimes — including the actual live pairs (LK-Si, GZ-Si,
SR-Si, LK-RN) and GK-MM.

Causal: β refit each bar on trailing 240h, z from same window, no lookahead.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from futbot.scripts.moex_iss_history import continuous_hourly

PAIRS = ["LK-Si", "GZ-Si", "SR-Si", "LK-RN", "GK-MM", "GZ-LK", "SR-GZ"]
WIN, Z_ENTRY, Z_STOP, MAX_HOLD = 240, 2.0, 4.0, 48
COMMISSION_RT = 0.0016
Y0, Y1 = 2022, 2025


def _load(base):
    cache = Path(__file__).resolve().parents[2] / "data" / f"hist_{base}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        df = continuous_hourly(base, Y0, Y1)
        if not df.empty:
            df.to_parquet(cache)
    if df.empty:
        return None
    return df.set_index("time")["close"].astype(float)


def causal_backtest(y, x, times):
    n = len(y)
    trades = []
    pos, entry, beta_e = 0, None, None
    for t in range(WIN, n):
        wy, wx = y[t - WIN : t], x[t - WIN : t]
        vx = np.var(wx, ddof=1)
        if vx <= 0:
            continue
        beta = np.cov(wy, wx, ddof=1)[0, 1] / vx
        sp = wy - beta * wx
        m, sd = sp.mean(), sp.std()
        if sd <= 0:
            continue
        z = (y[t] - beta * x[t] - m) / sd
        if pos == 0:
            if z > Z_ENTRY:
                pos, entry, beta_e = -1, t, beta
            elif z < -Z_ENTRY:
                pos, entry, beta_e = +1, t, beta
            continue
        crossed = (pos == +1 and z >= 0) or (pos == -1 and z <= 0)
        if not (crossed or abs(z) >= Z_STOP or (t - entry) >= MAX_HOLD):
            continue
        sp_e = y[entry] - beta_e * x[entry]
        sp_x = y[t] - beta_e * x[t]
        comb = abs(y[entry]) + abs(beta_e) * abs(x[entry])
        g = pos * (sp_x - sp_e) / comb if comb > 0 else 0
        trades.append((times[entry], (g - COMMISSION_RT) * 100))
        pos = 0
    return trades


def main():
    bases = sorted({b for p in PAIRS for b in p.split("-")})
    print(f"Loading continuous hourly for: {bases}")
    series = {}
    for base in bases:
        s = _load(base)
        if s is not None and len(s) > WIN + 500:
            series[base] = s
            print(f"  {base}: {len(s)} bars [{s.index.min().date()}..{s.index.max().date()}]")
        else:
            print(f"  {base}: insufficient/none")

    print("\n" + "=" * 96)
    print(
        f"HOURLY PAIRS WALK-FORWARD (live params: win={WIN}h hold={MAX_HOLD}h z={Z_ENTRY}) — {Y0}-{Y1}"
    )
    print("=" * 96)
    print(
        f"{'pair':<8}{'n':>5}{'win%':>6}{'avg%':>7}{'total%':>9}{'sharpe':>8}"
        f"{'+yrs':>7}  per-year total%"
    )
    print("-" * 96)
    for p in PAIRS:
        a, c = p.split("-")
        if a not in series or c not in series:
            print(f"{p:<8} (missing {a if a not in series else c})")
            continue
        al = pd.concat(
            [series[a].rename("y"), series[c].rename("x")], axis=1, join="inner"
        ).dropna()
        if len(al) < WIN + 500:
            print(f"{p:<8} (insufficient overlap {len(al)})")
            continue
        tr = causal_backtest(al["y"].values, al["x"].values, al.index.to_numpy())
        if len(tr) < 10:
            print(f"{p:<8} (only {len(tr)} trades)")
            continue
        td = pd.DataFrame(tr, columns=["t", "net"])
        td["year"] = pd.to_datetime(td["t"]).dt.year
        yr = td.groupby("year")["net"].sum()
        a_ = td["net"].values
        sh = a_.mean() / a_.std() * np.sqrt(len(a_)) if a_.std() > 0 else 0
        pos_y = int((yr > 0).sum())
        peryr = " ".join(f"{y}:{v:+.0f}" for y, v in yr.items())
        print(
            f"{p:<8}{len(a_):>5}{(a_>0).mean()*100:>5.0f}%{a_.mean():>+7.2f}"
            f"{a_.sum():>+9.1f}{sh:>+8.2f}{pos_y:>4}/{yr.size}  {peryr}"
        )
    print("-" * 96)
    print("Causal rolling-β, no lookahead. Stitched continuous front-month futures.")
    print("This is the FAIR multi-year test of the actual hourly pairs strategy.")


if __name__ == "__main__":
    main()
