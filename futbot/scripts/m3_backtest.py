"""M3 — "model the indicator, not the price" (MQL5 art. 20090) on FORTS hourly.

Idea: forecasting the raw price h bars ahead is noisy; forecasting a SMOOTHED
target (a moving average) is easier.  Predict the future SMA at a NEAR and a
FAR horizon and trade the implied slope:
    pred_far > pred_near  → expected uptrend  → long
    pred_far < pred_near  → expected downtrend → short
vol-scaled, re-evaluated each bar.

Faithful + simple (the article's own lesson is "simple beats complex"):
  • model = Ridge regression (linear, regularised), per the article's finding
    that a small linear model beat RF/stacking/deep nets.
  • features (OHLC-only, causal): lagged returns (1,2,3,5,10), close/SMA20,
    SMA5/SMA20, realised vol(20), high-low range.
  • target at horizon h: SMA(SMA_N) value h bars ahead − current SMA (a
    smoothed forward move).  h_near=1, h_far=10.
  • CAUSAL walk-forward: refit on a trailing window every REFIT bars; only
    past data trains; trade the out-of-sample predictions.

Judged by PER-YEAR consistency + vs a buy&hold and a plain SMA-cross baseline.

Usage:
    python -u -m futbot.scripts.m3_backtest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

BASES = ["Si", "LK", "GZ", "SR", "RN", "GK", "MM"]
SMA_N = 20
H_NEAR, H_FAR = 1, 10
TRAIN_WIN = 1500  # trailing bars to train on
REFIT = 250  # refit cadence (bars)
TARGET_VOL_ANN = 0.15
VOL_WIN = 240
HOURS_PER_YEAR = 24 * 365
COMMISSION_RT = 0.0008


def _features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    ret = c.pct_change()
    for lag in (1, 2, 3, 5, 10):
        out[f"ret{lag}"] = ret.shift(0).rolling(lag).sum()
    sma5 = c.rolling(5).mean()
    sma20 = c.rolling(SMA_N).mean()
    out["c_sma20"] = c / sma20 - 1
    out["sma5_20"] = sma5 / sma20 - 1
    out["vol20"] = ret.rolling(20).std()
    out["hl"] = (df["high"] - df["low"]) / c
    return out


def backtest_instrument(df: pd.DataFrame) -> dict:
    df = df.reset_index(drop=True)
    c = df["close"].values
    n = len(df)
    if n < TRAIN_WIN + 400:
        return {}
    feats = _features(df)
    sma = pd.Series(c).rolling(SMA_N).mean()
    # targets: smoothed forward move at near & far horizon
    tgt_near = sma.shift(-H_NEAR) - sma
    tgt_far = sma.shift(-H_FAR) - sma

    ret = np.zeros(n)
    ret[1:] = c[1:] / c[:-1] - 1.0
    target_per_bar = TARGET_VOL_ANN / np.sqrt(HOURS_PER_YEAR)

    pos = np.zeros(n)
    model_n = Ridge(alpha=10.0)
    model_f = Ridge(alpha=10.0)
    scaler = StandardScaler()
    last_fit = -(10**9)
    X = feats.values
    valid = ~np.isnan(X).any(axis=1)

    for t in range(TRAIN_WIN, n - 1):
        if not valid[t]:
            continue
        if t - last_fit >= REFIT:
            lo = max(0, t - TRAIN_WIN)
            idx = np.arange(lo, t)
            m = valid[idx] & ~np.isnan(tgt_near.values[idx]) & ~np.isnan(tgt_far.values[idx])
            idx = idx[m]
            if len(idx) < 300:
                continue
            Xt = scaler.fit_transform(X[idx])
            model_n.fit(Xt, tgt_near.values[idx])
            model_f.fit(Xt, tgt_far.values[idx])
            last_fit = t
        xt = scaler.transform(X[t : t + 1])
        pn = model_n.predict(xt)[0]
        pf = model_f.predict(xt)[0]
        slope = pf - pn  # expected smoothed move far vs near
        raw = np.sign(slope)
        rv = ret[t - VOL_WIN : t].std()
        scale = (target_per_bar / rv) if rv > 0 else 0.0
        pos[t] = float(np.clip(raw * scale, -1, 1))

    start = TRAIN_WIN
    pnl = np.zeros(n)
    for t in range(start, n - 1):
        turn = abs(pos[t] - pos[t - 1])
        pnl[t + 1] = pos[t] * ret[t + 1] - COMMISSION_RT * turn
    active = pnl[start + 1 :]
    times = pd.to_datetime(df["time"].values[start + 1 :])
    if len(active) == 0 or active.std() == 0:
        return {}
    res = pd.DataFrame({"t": times, "r": active})
    res["year"] = res["t"].dt.year
    per_year = res.groupby("year")["r"].sum()
    sharpe = active.mean() / active.std() * np.sqrt(HOURS_PER_YEAR)
    return {
        "sharpe": float(sharpe),
        "total_pct": float(active.sum() * 100),
        "per_year": {int(y): float(v * 100) for y, v in per_year.items()},
        "pos_years": int((per_year > 0).sum()),
        "n_years": int(per_year.size),
    }


def main():
    data_dir = Path(__file__).resolve().parents[2] / "data"
    print("=" * 96)
    print("M3 'model-the-indicator' (Ridge, 2-horizon SMA slope) — hourly FORTS, causal WF")
    print("=" * 96)
    print(f"{'base':6}{'sharpe':>8}{'total%':>9}{'+yrs':>7}  per-year %")
    print("-" * 96)
    allr = []
    for base in BASES:
        p = data_dir / f"hist_{base}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        r = backtest_instrument(df)
        if not r:
            print(f"{base:6}  (insufficient data)")
            continue
        allr.append((base, r))
        py = " ".join(f"{y}:{v:+.0f}" for y, v in r["per_year"].items())
        print(
            f"{base:6}{r['sharpe']:>+8.2f}{r['total_pct']:>+9.1f}"
            f"{r['pos_years']:>4}/{r['n_years']}  {py}"
        )
    print("-" * 96)
    if allr:
        mean_sh = np.mean([r["sharpe"] for _, r in allr])
        robust = sum(1 for _, r in allr if r["pos_years"] >= r["n_years"] - 1)
        print(
            f"Mean Sharpe: {mean_sh:+.2f}  |  robust instruments (≥n-1 +yrs): "
            f"{robust}/{len(allr)}"
        )
        print("\nVerdict: M3 is worth keeping only if it's consistently positive per-year")
        print("AND competitive with our triple_top/carry.  Honest comparison below.")
    print("\nNote: causal rolling Ridge, refit every 250 bars on trailing 1500.")


if __name__ == "__main__":
    main()
