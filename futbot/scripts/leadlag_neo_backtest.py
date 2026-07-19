"""Lead/lag cross-correlation pairs trade on Neo perps vs BTC.

Source: MetaTrader article 21811 (lead/lag CCF pair trading).  Academic basis:
Hou (2007) — large-cap leads small-cap.  Tested here on Neo perpetuals
(ETH/SOL/NBIS/CVNA/...) as FOLLOWERS, BTC as LEADER.

Method (CAUSAL, per pair):
  1. On a trailing CCF_WIN window, compute cross-correlation of returns
     between follower (Y) and leader (X) at lags 0..MAX_LAG hours.  The
     argmax is the empirical lead time L (X leads Y by L bars).
  2. Fit hedge ratio beta via OLS on returns: ret_Y[t] = beta * ret_X[t-L].
  3. Residual at bar t:  e[t] = price_Y[t] - beta * price_X[t-L].
  4. Rolling z-score of e over Z_WIN.
  5. Entry  |z| > Z_ENTRY:
        z > +Z_ENTRY  → Y rich vs X(lagged)  → SHORT Y
        z < -Z_ENTRY  → Y cheap              → LONG Y
     We trade ONLY the Y leg (single-instrument Neo trade), exploiting the
     mean-reversion of the residual that the lead/lag predicts.
  6. Exit z crosses 0 / |z| > Z_STOP / max-hold timeout.

Judged by per-year/quarter consistency.  Neo history is ~90 days, so we use
QUARTERLY blocks instead of years.
"""

import sys
from pathlib import Path
import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import Settings
from core.broker import BrokerClient
from t_tech.invest import CandleInterval
from t_tech.invest.utils import quotation_to_decimal

LEADER = "BTCUSDperpA"
FOLLOWERS = [
    "ETHUSDperpA",
    "SOLUSDperpA",
    "XRPUSDperpA",
    "TRXUSDperpA",
    "NBISperpA",
    "CVNAperpA",
    "APPperpA",
    "HOODperpA",
]

CCF_WIN = 240  # ~10 days of trading hours, refit window
MAX_LAG = 8  # search lags 0..8 hours
Z_WIN = 168  # 1 week rolling z
Z_ENTRY = 2.0
Z_STOP = 4.0
MAX_HOLD = 48
COMMISSION_RT = 0.0008
REFIT = 48  # refit lead/beta every 2 days


async def _fetch(broker, figi, days=120):
    now = datetime.now(timezone.utc)
    out, end, left = [], now, days
    while left > 0:
        cd = min(left, 89)
        start = end - timedelta(days=cd)
        try:
            c = await broker.get_candles(
                figi, start, end, interval=CandleInterval.CANDLE_INTERVAL_HOUR
            )
            for x in c:
                out.append({"time": x.time, "close": float(quotation_to_decimal(x.close))})
        except Exception:
            pass
        end = start
        left -= cd
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _ccf_lead(rx, ry, max_lag):
    """Return lag L (>=0) where corr(ry[t], rx[t-L]) is largest."""
    best_L, best_c = 0, -2.0
    n = len(rx)
    for L in range(0, max_lag + 1):
        if n - L < 50:
            continue
        a = rx[: n - L] if L else rx
        b = ry[L:] if L else ry
        sa, sb = a.std(), b.std()
        if sa == 0 or sb == 0:
            continue
        c = np.corrcoef(a, b)[0, 1]
        if c > best_c:
            best_c, best_L = c, L
    return best_L, best_c


def backtest_pair(df_y, df_x, name_y):
    al = (
        pd.merge(
            df_y[["time", "close"]].rename(columns={"close": "y"}),
            df_x[["time", "close"]].rename(columns={"close": "x"}),
            on="time",
            how="inner",
        )
        .sort_values("time")
        .reset_index(drop=True)
    )
    if len(al) < CCF_WIN + Z_WIN + 50:
        return {}
    y = al["y"].values
    x = al["x"].values
    n = len(y)
    ry = np.zeros(n)
    ry[1:] = y[1:] / y[:-1] - 1
    rx = np.zeros(n)
    rx[1:] = x[1:] / x[:-1] - 1

    pos = 0
    entry_idx = None
    pnls, hold_h, exits = [], [], []
    times = pd.to_datetime(al["time"].values)
    last_fit = -(10**9)
    beta, lead = 1.0, 1
    spread = np.full(n, np.nan)
    z = np.full(n, np.nan)
    leads_used = []

    for t in range(CCF_WIN + Z_WIN, n):
        if t - last_fit >= REFIT:
            lo = t - CCF_WIN
            L, _ = _ccf_lead(rx[lo:t], ry[lo:t], MAX_LAG)
            lead = max(1, L)
            # beta from contemporaneous lag-L return regression
            a = rx[lo : t - lead]
            b = ry[lo + lead : t]
            if a.std() > 0:
                beta = float(np.cov(a, b, ddof=1)[0, 1] / np.var(a, ddof=1))
            last_fit = t
            leads_used.append(lead)
        if t - lead < 0:
            continue
        s = y[t] - beta * x[t - lead]
        spread[t] = s
        wlo = t - Z_WIN
        w = spread[wlo:t]
        w = w[~np.isnan(w)]
        if len(w) < Z_WIN // 2:
            continue
        mu, sd = w.mean(), w.std()
        if sd <= 0:
            continue
        z[t] = (s - mu) / sd

        if pos == 0:
            if z[t] > Z_ENTRY:
                pos, entry_idx = -1, t
            elif z[t] < -Z_ENTRY:
                pos, entry_idx = +1, t
            continue
        crossed = (pos == +1 and z[t] >= 0) or (pos == -1 and z[t] <= 0)
        stopped = abs(z[t]) >= Z_STOP
        timed = (t - entry_idx) >= MAX_HOLD
        if not (crossed or stopped or timed):
            continue
        # P&L from Y leg only (single-instrument Neo trade)
        gross = pos * (y[t] - y[entry_idx]) / y[entry_idx]
        pnls.append(gross - COMMISSION_RT)
        hold_h.append(t - entry_idx)
        exits.append(pd.Timestamp(times[entry_idx]))
        pos = 0
    if not pnls:
        return {}
    arr = np.array(pnls)
    et = pd.to_datetime(pd.Index(exits))
    res = pd.DataFrame({"t": et, "r": arr})
    res["q"] = res["t"].dt.to_period("M")
    by_month = res.groupby("q")["r"].sum()
    pos_months = int((by_month > 0).sum())
    return {
        "name": name_y,
        "n": len(arr),
        "win": float((arr > 0).mean()),
        "total_pct": float(arr.sum() * 100),
        "avg_pct": float(arr.mean() * 100),
        "sharpe": float(arr.mean() / arr.std() * np.sqrt(len(arr))) if arr.std() > 0 else 0,
        "median_lead": int(np.median(leads_used)) if leads_used else 0,
        "by_month": {str(k): float(v * 100) for k, v in by_month.items()},
        "pos_months": pos_months,
        "n_months": int(len(by_month)),
        "avg_hold_h": float(np.mean(hold_h)),
    }


async def main():
    s = Settings()
    b = BrokerClient(token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="leadlag")
    await b.connect()
    futs = await b.get_all_futures()
    by_t = {(getattr(f, "ticker", "") or ""): f for f in futs}

    leader_df = await _fetch(b, by_t[LEADER].figi)
    results = []
    for tk in FOLLOWERS:
        f = by_t.get(tk)
        if f is None:
            continue
        df = await _fetch(b, f.figi)
        if len(df) < CCF_WIN + Z_WIN + 50:
            continue
        r = backtest_pair(df, leader_df, tk)
        if r:
            results.append(r)
    await b.disconnect()

    print("=" * 100)
    print(
        f"LEAD/LAG PAIRS — Neo follower vs {LEADER} (causal CCF, residual z; " f"Z_entry={Z_ENTRY})"
    )
    print("=" * 100)
    print(
        f"{'follower':<14}{'lead':>5}{'n':>5}{'win%':>6}{'avg%':>8}"
        f"{'total%':>9}{'sharpe':>8}{'+months':>9}"
    )
    print("-" * 100)
    for r in sorted(results, key=lambda x: -x["sharpe"]):
        print(
            f"{r['name']:<14}{r['median_lead']:>4}h"
            f"{r['n']:>5}{r['win']*100:>5.0f}%{r['avg_pct']:>+8.2f}"
            f"{r['total_pct']:>+9.1f}{r['sharpe']:>+8.2f}"
            f"{r['pos_months']:>5}/{r['n_months']}"
        )
    print("-" * 100)
    if results:
        robust = [
            r
            for r in results
            if r["pos_months"] >= r["n_months"] - 1 and r["sharpe"] > 0 and r["n"] >= 5
        ]
        print(
            f"\nRobust (≥n-1 months positive AND Sharpe>0 AND n≥5): "
            f"{len(robust)}/{len(results)}"
        )
        for r in robust:
            print(
                f"  ✓ {r['name']} (lead {r['median_lead']}h, win "
                f"{r['win']*100:.0f}%, total {r['total_pct']:+.1f}%)"
            )
    print("\nCaveat: Neo history ~90d → monthly (not yearly) consistency check.")
    print("Single Neo leg only (no BTC trade) — Neo P&L would convert USD×FX in live.")


if __name__ == "__main__":
    asyncio.run(main())
