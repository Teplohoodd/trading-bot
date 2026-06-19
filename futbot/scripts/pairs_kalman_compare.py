"""Kalman dynamic-hedge vs static-beta — pairs scanner extension.

Vanilla pairs trading fits one β by OLS and refits weekly.  But the true
hedge ratio drifts (changing vol regimes, rate moves, liquidity).  A Kalman
filter models β as a hidden state evolving as a random walk and updates it
every bar — Chan, "Algorithmic Trading" (2013), ch. on mean reversion.

Chan's formulation (state = [slope, intercept]):
    state RW:   β_t = β_{t-1} + w,   Cov(w) = Vw = (δ/(1-δ))·I
    observe:    y_t = [x_t, 1]·β_t + e_t,   Var(e) = Ve
    predict:    R = P + Vw
    innov:      e_t = y_t - [x_t,1]·β_pred
                Q_t = [x_t,1]·R·[x_t,1]' + Ve     (forecast-error variance)
    gain:       K = R·[x_t,1]' / Q_t
    update:     β = β_pred + K·e_t ;  P = R - K·[x_t,1]·R
    signal:     z_t = e_t / sqrt(Q_t)             (online spread z-score)

We backtest the SAME entry/exit rules with (a) static OLS β + rolling-z and
(b) Kalman β + online-z, on the same data, and compare.

Usage:
    python -u -m futbot.scripts.pairs_kalman_compare
    python -u -m futbot.scripts.pairs_kalman_compare --pairs LK-Si,GK-MM,YD-GK
"""

import argparse
import asyncio
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import Settings
from core.broker import BrokerClient
from tinkoff.invest import CandleInterval
from tinkoff.invest.utils import quotation_to_decimal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("tinkoff").setLevel(logging.WARNING)
logger = logging.getLogger("kalman")

DEFAULT_PAIRS = ["LK-Si", "GZ-Si", "SR-Si", "LK-RN", "GK-MM", "YD-GK", "PX-YD", "SR-RT", "RT-GK"]

COMMISSION_RT = 0.0016
Z_ENTRY = 2.0
Z_STOP = 4.0
MAX_HOLD = 48
ROLL_WIN = 240
DELTA = 1e-4  # Kalman state-evolution; smaller = β adapts slower
VE = 1e-3  # measurement noise (relative; scaled by series var below)


async def _resolve_front(broker, base):
    futs = await broker.get_all_futures()
    now = datetime.now(timezone.utc)
    cands = []
    for f in futs:
        t = getattr(f, "ticker", "") or ""
        if not (t == base or (t.startswith(base) and len(t) == len(base) + 2)):
            continue
        exp = getattr(f, "expiration_date", None)
        if exp is None:
            continue
        if hasattr(exp, "ToDatetime"):
            exp = exp.ToDatetime()
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        cands.append((f, exp))
    if not cands:
        return None
    cands.sort(key=lambda x: x[1])
    for f, exp in cands:
        if (exp - now).days >= 14:
            return f
    return cands[0][0]


def _candles_df(c):
    rows = [{"time": x.time, "close": float(quotation_to_decimal(x.close))} for x in c]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


async def _fetch(broker, figi, days):
    now = datetime.now(timezone.utc)
    chunks = []
    end = now
    left = days
    while left > 0:
        cd = min(left, 89)
        start = end - timedelta(days=cd)
        try:
            c = await broker.get_candles(
                figi, start, end, interval=CandleInterval.CANDLE_INTERVAL_HOUR
            )
            df = _candles_df(c)
            if not df.empty:
                chunks.append(df)
        except Exception:
            pass
        end = start
        left -= cd
    if not chunks:
        return pd.DataFrame()
    return (
        pd.concat(chunks)
        .drop_duplicates("time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )


def _trade_from_z(z: np.ndarray, y: np.ndarray, x: np.ndarray, beta_series: np.ndarray) -> dict:
    """Common entry/exit engine given a z-score series + (time-varying) beta."""
    n = len(z)
    pos = 0
    entry_idx = None
    pnls, holds = [], []
    for t in range(n):
        if np.isnan(z[t]):
            continue
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
        b = beta_series[entry_idx]
        sp_e = y[entry_idx] - b * x[entry_idx]
        sp_x = y[t] - b * x[t]
        combined = abs(y[entry_idx]) + abs(b) * abs(x[entry_idx])
        gross = pos * (sp_x - sp_e) / combined if combined > 0 else 0
        pnls.append(gross - COMMISSION_RT)
        holds.append(t - entry_idx)
        pos = 0
    if not pnls:
        return {"n_trades": 0}
    arr = np.array(pnls)
    return {
        "n_trades": len(arr),
        "win_rate": float((arr > 0).mean()),
        "total_pct": float(arr.sum() * 100),
        "sharpe": float(arr.mean() / arr.std() * math.sqrt(len(arr))) if arr.std() > 0 else 0.0,
        "median_hold": float(np.median(holds)),
    }


def static_backtest(y, x):
    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    spread = y - beta * x
    n = len(spread)
    z = np.full(n, np.nan)
    for t in range(ROLL_WIN, n):
        w = spread[t - ROLL_WIN : t]
        sd = w.std()
        z[t] = (spread[t] - w.mean()) / sd if sd > 0 else 0.0
    beta_series = np.full(n, beta)
    r = _trade_from_z(z, y, x, beta_series)
    r["beta"] = beta
    return r


def kalman_backtest(y, x):
    """Chan's Kalman dynamic hedge; z = innovation / sqrt(forecast var)."""
    n = len(y)
    delta = DELTA
    Vw = delta / (1 - delta) * np.eye(2)
    Ve = VE * np.var(y)  # scale measurement noise to series
    beta = np.zeros((n, 2))  # [slope, intercept]
    P = np.zeros((2, 2))
    b = np.zeros(2)
    z = np.full(n, np.nan)
    for t in range(n):
        F = np.array([x[t], 1.0])  # observation matrix row
        if t > 0:
            R = P + Vw
        else:
            R = np.eye(2)  # diffuse prior
        yhat = F @ b
        e = y[t] - yhat  # innovation (the "spread")
        Q = F @ R @ F + Ve  # innovation variance
        K = (R @ F) / Q  # Kalman gain (2-vector)
        b = b + K * e
        P = R - np.outer(K, F @ R)
        beta[t] = b
        if Q > 0 and t > ROLL_WIN:  # warm-up to let β settle
            z[t] = e / math.sqrt(Q)
    r = _trade_from_z(z, y, x, beta[:, 0])
    r["beta"] = float(beta[-1, 0])  # last slope
    return r


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS))
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    bases = sorted({b for p in pairs for b in p.split("-")})

    s = Settings()
    broker = BrokerClient(
        token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="kalman"
    )
    await broker.connect()
    by_base = {}
    for base in bases:
        f = await _resolve_front(broker, base)
        if f is None:
            logger.warning(f"  {base}: no contract")
            continue
        df = await _fetch(broker, f.figi, args.days)
        if len(df) < 600:
            logger.warning(f"  {base}: {len(df)} bars — skip")
            continue
        by_base[base] = df.set_index("time")["close"].rename(base)
        logger.info(f"  {base:6} {f.ticker:8} {len(df)} bars")
    await broker.disconnect()

    print("\n" + "=" * 110)
    print("STATIC β (rolling-z)  vs  KALMAN dynamic-β (online-z)")
    print("=" * 110)
    print(
        f"{'pair':<9}│{'  STATIC: n   win   total%  sharpe':<36}│"
        f"{'  KALMAN: n   win   total%  sharpe':<36}│ winner"
    )
    print("-" * 110)
    wins = {"static": 0, "kalman": 0}
    for p in pairs:
        a, c = p.split("-")
        if a not in by_base or c not in by_base:
            continue
        al = pd.concat(
            [by_base[a].rename("y"), by_base[c].rename("x")], axis=1, join="inner"
        ).dropna()
        if len(al) < 600:
            continue
        y = al["y"].values
        x = al["x"].values
        st = static_backtest(y, x)
        kf = kalman_backtest(y, x)
        s_sh = st.get("sharpe", 0)
        k_sh = kf.get("sharpe", 0)
        win = "KALMAN" if k_sh > s_sh else "static"
        wins["kalman" if k_sh > s_sh else "static"] += 1
        print(
            f"{p:<9}│  {st.get('n_trades',0):>3} "
            f"{st.get('win_rate',0)*100:>4.0f}% {st.get('total_pct',0):>+7.1f} "
            f"{s_sh:>+6.2f}            │  "
            f"{kf.get('n_trades',0):>3} {kf.get('win_rate',0)*100:>4.0f}% "
            f"{kf.get('total_pct',0):>+7.1f} {k_sh:>+6.2f}            │ {win}"
        )
    print("-" * 110)
    print(f"Pairs where Kalman beat static: {wins['kalman']}/{wins['static']+wins['kalman']}")


if __name__ == "__main__":
    asyncio.run(main())
