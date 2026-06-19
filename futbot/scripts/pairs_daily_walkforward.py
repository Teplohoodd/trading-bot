"""Pairs walk-forward across 9 years (daily MOEX spot) — the real truth test.

The 180-day futures backtest said pairs barely hold OOS, and GK-MM / YD-GK
FAILED.  This tests the pairs APPROACH — and those exact relationships — over
9 years and every regime, fully causally (no lookahead):

  • β refit each bar on a trailing window (rolling OLS) — never uses future.
  • z-score from rolling mean/std of the spread on the same trailing window.
  • Entry |z|>2, exit z→0 / |z|>4 stop / max-hold timeout.

The user's data contains GMKN+MOEX (= our GK-MM) and YNDX+GMKN (= YD-GK), so
the two pairs that failed the short OOS can finally be judged on a decade.

Caveats: daily (not hourly), stock SPOT (not futures), so this validates the
co-integration EDGE & regime-robustness, not exact hourly execution.

Usage:
    python -u -m futbot.scripts.pairs_daily_walkforward
"""

import glob
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORE = ["GAZP", "SBER", "LKOH", "GMKN", "YNDX", "MOEX", "NVTK", "PLZL", "MGNT", "VTBR"]
# Map to our futures pair notation for the ones we trade/considered
ALIAS = {
    "GMKN": "GK",
    "MOEX": "MM",
    "YNDX": "YD",
    "LKOH": "LK",
    "GAZP": "GZ",
    "SBER": "SR",
    "PLZL": "PX",
    "NVTK": "NG_stock",
    "MGNT": "MN",
    "VTBR": "VB",
}
WIN = 120  # trailing window (days) for β + z
Z_ENTRY, Z_STOP, MAX_HOLD = 2.0, 4.0, 20
COMMISSION_RT = 0.0016
ADF_P_MAX = 0.10


def causal_backtest(y, x, times):
    """Fully causal rolling-β z-score mean reversion.  Returns per-trade
    list of (entry_time, pnl_pct)."""
    n = len(y)
    trades = []
    pos, entry = 0, None
    beta_e = None
    for t in range(WIN, n):
        wy = y[t - WIN : t]
        wx = x[t - WIN : t]
        vx = np.var(wx, ddof=1)
        if vx <= 0:
            continue
        beta = np.cov(wy, wx, ddof=1)[0, 1] / vx
        sp_win = wy - beta * wx
        m, sd = sp_win.mean(), sp_win.std()
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
        gross = pos * (sp_x - sp_e) / comb if comb > 0 else 0
        trades.append((times[entry], (gross - COMMISSION_RT) * 100))
        pos = 0
    return trades


def main():
    path = glob.glob("C:/Users/Teplohood/Downloads/*market*.xlsx")[0]
    raw = pd.read_excel(path, sheet_name="Sheet1")
    raw = raw[raw["BOARDID"] == "TQBR"]
    raw["TRADEDATE"] = pd.to_datetime(raw["TRADEDATE"])
    series = {}
    for sec in CORE:
        d = raw[raw["SECID"] == sec].sort_values("TRADEDATE")
        if len(d) >= 1000:
            series[sec] = d.set_index("TRADEDATE")["CLOSE"].astype(float)

    from statsmodels.tsa.stattools import adfuller

    results = []
    for a, b in combinations(series.keys(), 2):
        al = pd.concat(
            [series[a].rename("y"), series[b].rename("x")], axis=1, join="inner"
        ).dropna()
        if len(al) < WIN + 200:
            continue
        y = al["y"].values
        x = al["x"].values
        # full-sample ADF (diagnostic only)
        beta_full = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
        try:
            adf_p = float(adfuller(y - beta_full * x, maxlag=5, autolag=None)[1])
        except Exception:
            adf_p = 1.0
        trades = causal_backtest(y, x, al.index.to_numpy())
        if len(trades) < 20:
            continue
        td = pd.DataFrame(trades, columns=["t", "net"])
        td["year"] = pd.to_datetime(td["t"]).dt.year
        yrs = td.groupby("year")["net"].sum()
        pos_years = int((yrs > 0).sum())
        n_years = int(yrs.size)
        arr = td["net"].values
        results.append(
            {
                "pair": f"{ALIAS.get(a,a)}-{ALIAS.get(b,b)}",
                "raw": f"{a}-{b}",
                "adf_p": adf_p,
                "n": len(arr),
                "win": (arr > 0).mean(),
                "total": arr.sum(),
                "avg": arr.mean(),
                "sharpe": arr.mean() / arr.std() * np.sqrt(len(arr)) if arr.std() > 0 else 0,
                "pos_years": pos_years,
                "n_years": n_years,
            }
        )

    res = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    print("=" * 100)
    print("PAIRS WALK-FORWARD — 9 years daily (causal rolling-β), MOEX spot")
    print("=" * 100)
    print(
        f"{'pair':<10}{'raw':<14}{'adf_p':>7}{'n':>5}{'win%':>6}"
        f"{'avg%':>7}{'total%':>9}{'sharpe':>8}{'+yrs':>7}"
    )
    print("-" * 100)
    for _, r in res.iterrows():
        print(
            f"{r['pair']:<10}{r['raw']:<14}{r['adf_p']:>7.3f}{r['n']:>5}"
            f"{r['win']*100:>5.0f}%{r['avg']:>+7.2f}{r['total']:>+9.1f}"
            f"{r['sharpe']:>+8.2f}{r['pos_years']:>4}/{r['n_years']}"
        )
    print("-" * 100)

    # Spotlight the pairs we actually care about
    print("\nSPOTLIGHT — pairs from our research:")
    for want in ["GK-MM", "YD-GK", "LK-GZ", "SR-VB", "GK-PX", "LK-GK"]:
        row = res[res["pair"] == want]
        if not row.empty:
            r = row.iloc[0]
            verdict = (
                "✅ robust"
                if r["pos_years"] >= r["n_years"] - 2
                else "⚠ mixed" if r["pos_years"] >= r["n_years"] * 0.5 else "❌ fragile"
            )
            print(
                f"  {want:<8} ({r['raw']:<12}): {r['pos_years']}/{r['n_years']} yrs +, "
                f"win {r['win']*100:.0f}%, total {r['total']:+.0f}%, "
                f"Sharpe {r['sharpe']:+.2f}  {verdict}"
            )

    robust = res[res["pos_years"] >= res["n_years"] - 2]
    print(f"\nPairs robust across regimes (≥ n-2 positive years): " f"{len(robust)}/{len(res)}")
    print("Reminder: daily/spot — validates the cointegration edge & regime")
    print("robustness, not exact hourly futures execution.")
    out = Path(__file__).resolve().parents[2] / "data" / "pairs_daily_wf.csv"
    res.to_csv(out, index=False)


if __name__ == "__main__":
    main()
