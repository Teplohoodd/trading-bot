"""Walk-forward validation for pairs — the capital-trust gate.

In-sample (IS) backtests overfit: β is fit on the same data we score on.
Walk-forward fixes this: fit β on a trailing IS window, trade the NEXT OOS
window with that frozen β + IS-derived z-stats, then roll forward.  Only the
OOS trades count.  If OOS stats collapse vs IS, the pair was a mirage.

Folds over `days` history:
    IS = is_days (β fit + spread mean/std)
    OOS = oos_days (trade with frozen params)
    step = oos_days (non-overlapping OOS)

Reports per-pair IS-vs-OOS so degradation is obvious.

Usage:
    python -u -m futbot.scripts.pairs_walkforward
    python -u -m futbot.scripts.pairs_walkforward --pairs GK-MM,YD-GK,LK-Si --days 360
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
from t_tech.invest import CandleInterval
from t_tech.invest.utils import quotation_to_decimal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("t_tech").setLevel(logging.WARNING)
logger = logging.getLogger("wf")

DEFAULT_PAIRS = ["GK-MM", "YD-GK", "LK-Si", "GZ-Si", "SR-Si", "LK-RN"]
COMMISSION_RT = 0.0016
Z_ENTRY = 2.0
Z_STOP = 4.0
MAX_HOLD = 48
ROLL_WIN = 240


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


def _simulate(y, x, beta, spread_mean, spread_std):
    """Trade with FROZEN beta + frozen z-normalisation (from IS)."""
    spread = y - beta * x
    if spread_std <= 0:
        return []
    z = (spread - spread_mean) / spread_std
    n = len(z)
    pos = 0
    entry_idx = None
    pnls = []
    for t in range(n):
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
        sp_e = y[entry_idx] - beta * x[entry_idx]
        sp_x = y[t] - beta * x[t]
        combined = abs(y[entry_idx]) + abs(beta) * abs(x[entry_idx])
        gross = pos * (sp_x - sp_e) / combined if combined > 0 else 0
        pnls.append(gross - COMMISSION_RT)
        pos = 0
    return pnls


def _stats(pnls):
    if not pnls:
        return {"n": 0, "win": 0.0, "total": 0.0, "sharpe": 0.0}
    a = np.array(pnls)
    return {
        "n": len(a),
        "win": float((a > 0).mean()),
        "total": float(a.sum() * 100),
        "sharpe": float(a.mean() / a.std() * math.sqrt(len(a))) if a.std() > 0 else 0.0,
    }


def _rolling_z(spread):
    z = np.full(len(spread), np.nan)
    for t in range(ROLL_WIN, len(spread)):
        w = spread[t - ROLL_WIN : t]
        sd = w.std()
        z[t] = (spread[t] - w.mean()) / sd if sd > 0 else 0.0
    return z


def walk_forward(y_full, x_full, *, is_bars, oos_bars):
    """Roll IS→OOS.  Returns (is_pnls, oos_pnls) aggregated across folds.

    OOS replicates LIVE behaviour: β frozen from IS (live refits weekly), but
    the z-score uses the SAME rolling 240-bar window the live bot uses — so we
    warm the rolling-z with the tail of IS and trade only the OOS bars.
    """
    n = len(y_full)
    is_pnls, oos_pnls = [], []
    start = 0
    while start + is_bars + oos_bars <= n:
        y_is = y_full[start : start + is_bars]
        x_is = x_full[start : start + is_bars]
        beta = np.cov(y_is, x_is, ddof=1)[0, 1] / np.var(x_is, ddof=1)
        sp_is = y_is - beta * x_is
        is_pnls += _trade_z(_rolling_z(sp_is), y_is, x_is, beta)

        # OOS with rolling-z warmed by IS tail (live-equivalent)
        warm = ROLL_WIN
        lo = max(0, start + is_bars - warm)
        hi = start + is_bars + oos_bars
        y_seg = y_full[lo:hi]
        x_seg = x_full[lo:hi]
        sp_seg = y_seg - beta * x_seg
        z_seg = _rolling_z(sp_seg)
        # Only allow entries in the OOS portion (index >= warmup offset)
        oos_offset = (start + is_bars) - lo
        z_masked = z_seg.copy()
        z_masked[:oos_offset] = np.nan
        oos_pnls += _trade_z(z_masked, y_seg, x_seg, beta)
        start += oos_bars
    return is_pnls, oos_pnls


def _trade_z(z, y, x, beta):
    n = len(z)
    pos = 0
    entry = None
    out = []
    for t in range(n):
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
        sp_e = y[entry] - beta * x[entry]
        sp_x = y[t] - beta * x[t]
        comb = abs(y[entry]) + abs(beta) * abs(x[entry])
        out.append((pos * (sp_x - sp_e) / comb if comb > 0 else 0) - COMMISSION_RT)
        pos = 0
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS))
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--is_days", type=int, default=60)
    ap.add_argument("--oos_days", type=int, default=20)
    args = ap.parse_args()
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    bases = sorted({b for p in pairs for b in p.split("-")})

    s = Settings()
    broker = BrokerClient(token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="wf")
    await broker.connect()
    by_base = {}
    for base in bases:
        f = await _resolve_front(broker, base)
        if f is None:
            continue
        df = await _fetch(broker, f.figi, args.days)
        if len(df) < 600:
            continue
        by_base[base] = df.set_index("time")["close"].rename(base)
        logger.info(f"  {base:6} {f.ticker:8} {len(df)} bars")
    await broker.disconnect()

    # FORTS trades ~10-14h/day on weekdays, so calendar-days × 24 vastly
    # over-counts bars.  Derive the real bars/day from the data span.
    spans = []
    for srs in by_base.values():
        days_span = max((srs.index.max() - srs.index.min()).total_seconds() / 86400, 1)
        spans.append(len(srs) / days_span)
    bars_per_day = float(np.median(spans)) if spans else 10.0
    is_bars = int(args.is_days * bars_per_day)
    oos_bars = int(args.oos_days * bars_per_day)
    logger.info(f"bars/day≈{bars_per_day:.1f} → IS={is_bars} bars, OOS={oos_bars} bars")
    print("\n" + "=" * 100)
    print(
        f"WALK-FORWARD  IS={args.is_days}d  OOS={args.oos_days}d  "
        f"(only OOS trades count as validation)"
    )
    print("=" * 100)
    print(
        f"{'pair':<9}│{'   IN-SAMPLE: n  win  total% sharpe':<35}│"
        f"{'  OUT-SAMPLE: n  win  total% sharpe':<35}│ verdict"
    )
    print("-" * 100)
    for p in pairs:
        a, c = p.split("-")
        if a not in by_base or c not in by_base:
            print(f"{p:<9}│ (missing data)")
            continue
        al = pd.concat(
            [by_base[a].rename("y"), by_base[c].rename("x")], axis=1, join="inner"
        ).dropna()
        if len(al) < is_bars + oos_bars:
            print(f"{p:<9}│ (insufficient: {len(al)} bars)")
            continue
        y = al["y"].values
        x = al["x"].values
        is_p, oos_p = walk_forward(y, x, is_bars=is_bars, oos_bars=oos_bars)
        s_is = _stats(is_p)
        s_oos = _stats(oos_p)
        # Verdict: OOS positive + win>50 + enough trades
        if s_oos["n"] >= 4 and s_oos["total"] > 0 and s_oos["win"] >= 0.5:
            verdict = "✅ HOLDS"
        elif s_oos["n"] < 4:
            verdict = "⚠ thin"
        else:
            verdict = "❌ FAILS"
        print(
            f"{p:<9}│  {s_is['n']:>3} {s_is['win']*100:>4.0f}% "
            f"{s_is['total']:>+6.1f} {s_is['sharpe']:>+5.2f}        │  "
            f"{s_oos['n']:>3} {s_oos['win']*100:>4.0f}% {s_oos['total']:>+6.1f} "
            f"{s_oos['sharpe']:>+5.2f}        │ {verdict}"
        )
    print("-" * 100)
    print("HOLDS = OOS positive, win≥50%, ≥4 trades.  This is the live-capital gate.")


if __name__ == "__main__":
    asyncio.run(main())
