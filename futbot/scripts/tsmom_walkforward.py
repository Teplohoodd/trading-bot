"""TSMOM out-of-sample stability — is the edge consistent or one lucky window?

The lookback (480h) is FIXED, so there's no parameter fitting — the whole
backtest is effectively out-of-sample.  The real question for a trend
strategy is *consistency*: a 0.47 Sharpe is worthless if it comes entirely
from one 3-week window and is flat/negative the rest of the time.

We build the diversified equal-weight TSMOM(480h) portfolio return path
(each instrument vol-scaled to 15% ann, equal weight), then split the
timeline into consecutive sub-periods and report per-period Sharpe + the
portfolio equity curve / max drawdown.  Leverage-adjusted at the end.

Usage:
    python -u -m futbot.scripts.tsmom_walkforward --days 180 --segments 4
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
logger = logging.getLogger("tsmom_wf")

UNIVERSE = [
    "BR",
    "GZ",
    "SR",
    "LK",
    "MX",
    "Si",
    "GD",
    "RT",
    "USDRU",
    "GLDRU",
    "EURRU",
    "CR",
    "PX",
    "YD",
    "VB",
    "RN",
    "GK",
    "MM",
    "PT",
    "S1",
    "MV",
    "SV",
]
LOOKBACK = 480
VOL_WINDOW = 240
TARGET_VOL_ANN = 0.15
HOURS_PER_YEAR = 24 * 365
COMMISSION_RT = 0.0008


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


def tsmom_return_path(close: np.ndarray) -> np.ndarray:
    """Per-bar net return series of vol-scaled TSMOM on one instrument."""
    n = len(close)
    out = np.zeros(n)
    if n < LOOKBACK + VOL_WINDOW + 50:
        return out
    ret = np.zeros(n)
    ret[1:] = close[1:] / close[:-1] - 1.0
    tgt = TARGET_VOL_ANN / math.sqrt(HOURS_PER_YEAR)
    pos = np.zeros(n)
    for t in range(LOOKBACK + VOL_WINDOW, n):
        raw = np.sign(close[t] - close[t - LOOKBACK])
        rv = ret[t - VOL_WINDOW : t].std()
        pos[t] = float(np.clip(raw * (tgt / rv if rv > 0 else 0), -1, 1))
    for t in range(LOOKBACK + VOL_WINDOW, n - 1):
        turn = abs(pos[t] - pos[t - 1])
        out[t + 1] = pos[t] * ret[t + 1] - COMMISSION_RT * turn
    return out


def _sharpe(r):
    r = r[r != 0] if (r != 0).any() else r
    return (r.mean() / r.std() * math.sqrt(HOURS_PER_YEAR)) if r.std() > 0 else 0.0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--segments", type=int, default=4)
    args = ap.parse_args()

    s = Settings()
    broker = BrokerClient(
        token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="tsmom-wf"
    )
    await broker.connect()
    paths = {}
    margins = {}
    for base in UNIVERSE:
        f = await _resolve_front(broker, base)
        if f is None:
            continue
        df = await _fetch(broker, f.figi, args.days)
        if len(df) < LOOKBACK + VOL_WINDOW + 100:
            continue
        srs = df.set_index("time")["close"]
        paths[base] = srs
        meta = broker.extract_futures_metadata(f)
        dl = float(meta.get("dlong") or 0.0)
        margins[base] = dl if dl > 0 else 0.25
    await broker.disconnect()

    # Align all instruments on a common index, build per-bar TSMOM returns
    common = None
    rets = {}
    for base, srs in paths.items():
        r = tsmom_return_path(srs.values)
        rets[base] = pd.Series(r, index=srs.index)
    mat = pd.DataFrame(rets).dropna(how="all")
    # Equal-weight portfolio: mean across instruments active that bar
    port = mat.mean(axis=1).fillna(0.0).values
    idx = mat.index

    print("\n" + "=" * 92)
    print(
        f"TSMOM(480h) DIVERSIFIED PORTFOLIO — out-of-sample stability " f"({len(paths)} contracts)"
    )
    print("=" * 92)
    overall = _sharpe(port)
    cum = np.cumsum(port)
    peak = np.maximum.accumulate(cum)
    mdd = float((cum - peak).min())
    print(
        f"Overall: ann.Sharpe={overall:.2f}  total notional={cum[-1]*100:+.1f}%  "
        f"maxDD notional={mdd*100:.1f}%"
    )

    # Per-segment consistency
    seg = len(port) // args.segments
    print(f"\nPer-segment Sharpe (consistency check, {args.segments} blocks):")
    pos_segs = 0
    for i in range(args.segments):
        a = i * seg
        b = (i + 1) * seg if i < args.segments - 1 else len(port)
        sh = _sharpe(port[a:b])
        tot = port[a:b].sum() * 100
        t0 = str(idx[a].date())
        t1 = str(idx[min(b, len(idx)) - 1].date())
        flag = "✅" if sh > 0 else "❌"
        if sh > 0:
            pos_segs += 1
        print(f"   seg{i+1} [{t0}..{t1}]: Sharpe={sh:>+5.2f}  total={tot:>+6.2f}% {flag}")
    print(
        f"\nPositive segments: {pos_segs}/{args.segments}  "
        f"→ {'consistent' if pos_segs >= args.segments-1 else 'CONCENTRATED / fragile'}"
    )

    # Leverage view (avg portfolio leverage from instrument margins)
    avg_lev = np.mean([1.0 / m for m in margins.values()])
    print(
        f"\nAvg instrument leverage ≈ {avg_lev:.1f}×  → on-margin total ≈ "
        f"{cum[-1]*avg_lev*100:+.0f}%, on-margin maxDD ≈ {mdd*avg_lev*100:.0f}%"
    )
    print("(leverage multiplies BOTH — the drawdown is what kills accounts)")


if __name__ == "__main__":
    asyncio.run(main())
