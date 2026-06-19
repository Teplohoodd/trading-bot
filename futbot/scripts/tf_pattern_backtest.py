"""Timeframe tune for triple_top/bottom: 15m vs 30m vs 1h bars.

User observation: the live bot entered LT "a bit late" — pattern confirmation
+ hourly tick adds latency.  Finer bars confirm patterns sooner; but they also
produce smaller patterns (height filter binds) and more noise.  Empirical
question — same detectors, same exits, three bar sizes, which nets more?

Data: 15-min candles for the last ~90 days of the JUNE (M6) FORTS contracts
(they carry the full window; the September U6s only became front this week),
resampled to 15m/30m/60m.  Same universe as the live core portfolio.

Usage:
    python -u -m futbot.scripts.tf_pattern_backtest --fetch   # once
    python -u -m futbot.scripts.tf_pattern_backtest
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from futbot.patterns.detectors import find_swings, detect_triple_tops, detect_triple_bottoms
from futbot.patterns.portfolio import TUNED_PARAMS as P

# Same detector kwargs as the LIVE bot (trend_bot._scan_and_open)
DET_KW = dict(
    peak_tol=P.peak_tol,
    min_height=P.min_height,
    min_width=P.min_width,
    max_width=P.max_width,
    max_confirm_bars=P.max_confirm_bars,
)
SWING_KW = dict(window=P.swing_window, min_prominence_pct=P.min_prominence_pct)

DATA = Path(__file__).resolve().parents[2] / "data" / "tf15"
BASES = ["PX", "GD", "SS", "YD", "LT", "S1", "VB", "MV"]
COST_PCT = 0.0008  # round-trip
TIMEOUT_BARS_1H = 48  # live timeout is 48 hourly bars — scale per TF


async def fetch_all():
    from config.settings import Settings
    from core.broker import BrokerClient
    from tinkoff.invest.schemas import CandleInterval
    from tinkoff.invest.utils import quotation_to_decimal

    DATA.mkdir(parents=True, exist_ok=True)
    s = Settings()
    b = BrokerClient(token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="tf-fetch")
    await b.connect()
    futs = await b.get_all_futures()
    now = datetime.now(timezone.utc)
    for base in BASES:
        # June contract = expiry within 5..40 days (carries the 90d window)
        cands = []
        for f in futs:
            t = getattr(f, "ticker", "") or ""
            exp = getattr(f, "expiration_date", None)
            if not t.startswith(base) or len(t) != len(base) + 2 or exp is None:
                continue
            dte = (exp - now).days
            if 0 <= dte <= 45:
                cands.append((dte, f))
        if not cands:
            print(f"  {base}: no June contract")
            continue
        f = min(cands)[1]
        rows = []
        # 15-min candles: fetch day-by-day (API range limit)
        for d in range(92, -1, -1):
            t0 = now - timedelta(days=d + 1)
            t1 = now - timedelta(days=d)
            try:
                cs = await b.get_candles(f.figi, t0, t1, CandleInterval.CANDLE_INTERVAL_15_MIN)
            except Exception:
                continue
            for c in cs:
                rows.append(
                    {
                        "time": c.time,
                        "open": float(quotation_to_decimal(c.open)),
                        "high": float(quotation_to_decimal(c.high)),
                        "low": float(quotation_to_decimal(c.low)),
                        "close": float(quotation_to_decimal(c.close)),
                        "volume": c.volume,
                    }
                )
        df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time").reset_index(drop=True)
        df.to_parquet(DATA / f"{base}.parquet")
        print(f"  {base} ({f.ticker}): {len(df)} 15m bars")
    await b.disconnect()


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 15:
        return df
    d = (
        df.set_index("time")
        .resample(f"{minutes}min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return d


def simulate(df: pd.DataFrame, timeout_bars: int) -> list[dict]:
    """Detect patterns and simulate entry→stop/target/timeout, % returns."""
    swings = find_swings(df, **SWING_KW)
    signals = list(detect_triple_tops(df, swings, **DET_KW)) + list(
        detect_triple_bottoms(df, swings, **DET_KW)
    )
    signals.sort(key=lambda s: s.bar_idx)
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    out, busy_until = [], -1
    for s in signals:
        i = s.bar_idx
        if i <= busy_until:  # one position at a time per instrument
            continue
        entry = s.entry_price
        ret = None
        for j in range(i + 1, min(i + 1 + timeout_bars, len(c))):
            if s.direction == +1:  # long
                if l[j] <= s.stop_price:
                    ret = s.stop_price / entry - 1
                    break
                if h[j] >= s.target_price:
                    ret = s.target_price / entry - 1
                    break
            else:  # short
                if h[j] >= s.stop_price:
                    ret = -(s.stop_price / entry - 1)
                    break
                if l[j] <= s.target_price:
                    ret = -(s.target_price / entry - 1)
                    break
        else:
            j = min(i + timeout_bars, len(c) - 1)
            ret = (c[j] / entry - 1) * s.direction
        if ret is None:
            j = min(i + timeout_bars, len(c) - 1)
            ret = (c[j] / entry - 1) * s.direction
        busy_until = j
        out.append(
            {
                "time": df["time"].iloc[i],
                "dir": s.direction,
                "pattern": s.pattern,
                "ret": ret - COST_PCT,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        asyncio.run(fetch_all())
        return

    frames = {}
    for p in DATA.glob("*.parquet"):
        df = pd.read_parquet(p)
        df["time"] = pd.to_datetime(df["time"])
        frames[p.stem] = df
    print(f"universe: {sorted(frames)} (15m base data)")
    print("=" * 100)
    print(f"{'TF':>5} {'base':>5} | {'trades':>6} {'win%':>6} {'avg%':>7} " f"{'total%':>8}")
    print("-" * 100)
    for minutes in (15, 30, 60):
        timeout = TIMEOUT_BARS_1H * 60 // minutes
        agg = []
        per_base = {}
        for base, raw in sorted(frames.items()):
            d = _resample(raw, minutes)
            tr = simulate(d, timeout)
            per_base[base] = tr
            agg.extend(t["ret"] for t in tr)
        a = np.array(agg)
        for base, tr in sorted(per_base.items()):
            r = np.array([t["ret"] for t in tr])
            if len(r) == 0:
                continue
            print(
                f"{minutes:>4}m {base:>5} | {len(r):>6} "
                f"{(r > 0).mean() * 100:>5.0f}% {r.mean() * 100:>+6.2f} "
                f"{r.sum() * 100:>+7.1f}"
            )
        print(
            f"{minutes:>4}m {'ALL':>5} | {len(a):>6} "
            f"{(a > 0).mean() * 100:>5.0f}% {a.mean() * 100:>+6.2f} "
            f"{a.sum() * 100:>+7.1f}   <= POOLED"
        )
        print("-" * 100)
    print("\nSame detectors/exits; only bar size changes (timeout scaled to " "constant 48 hours).")


if __name__ == "__main__":
    main()
