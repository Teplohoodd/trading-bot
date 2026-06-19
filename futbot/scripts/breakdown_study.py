"""Event study: volume-confirmed breakdown after quiet consolidation (H1).

Inspired by the IVAT crashes (2026-05-25..29 and 2026-06-08): both were
preceded by LOW-VOLUME consolidation, then a break below the range low on
ABNORMAL volume, after which the move CONTINUED for hours/days.

Hypothesis H1: bar t is a "breakdown event" if
    1. close[t] < min(low[t-N..t-1])                  (range breakdown)
    2. volume[t] >= VMULT × median(volume[t-N..t-1])  (volume confirmation)
    3. realised vol of prior N bars is in the lower QPCT of its 90d range
       (quiet before the storm — optional filter, tested with/without)

Measured: forward return close[t] → close[t+h] for h in 1..48 bars.
If H1 holds, forward returns after events are significantly NEGATIVE
(continuation), not mean-reverting.

Universe: ~30 liquid MOEX stocks, 90 days, intervals 15min/1h/4h (resampled).
Usage:  python -u -m futbot.scripts.breakdown_study [--fetch]
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

DATA = Path(__file__).resolve().parents[2] / "data" / "stocks90"

# Liquid MOEX stocks (ticker → FIGI), spanning sectors + a few hot small-caps
UNIVERSE = {
    "SBER": "BBG004730N88",
    "GAZP": "BBG004730RP0",
    "LKOH": "BBG004731032",
    "ROSN": "BBG004731354",
    "GMKN": "BBG004731489",
    "NVTK": "BBG00475KKY8",
    "TATN": "BBG004RVFFC0",
    "MGNT": "BBG004RVFCY3",
    "YDEX": "TCS00A107T19",
    "VTBR": "BBG004730ZJ9",
    "ALRS": "BBG004S68B31",
    "CHMF": "BBG00475K6C3",
    "MTLR": "BBG004S68598",
    "AFLT": "BBG004S683W7",
    "MOEX": "BBG004730JJ5",
    "OZON": "TCS00A10CW95",
    "POSI": "TCS00A103X66",
    "SOFL": "TCS00A0ZZBC2",
    "WUSH": "TCS00A105EX7",
    "SMLT": "BBG00F6NKQX3",
    "SGZH": "BBG0100R9963",
    "IVAT": "TCS00A108GD8",
    "ASTR": "TCS00A106T36",
    "DIAS": "TCS00A107ER5",
    "DELI": "TCS00A107J11",
    "UGLD": "TCS10A0JPP37",
    "HEAD": "TCS20A107662",
    "MTSS": "BBG004S681W1",
    "AFKS": "BBG004S68614",
    "PIKK": "BBG004S68BH6",
}


async def fetch_all():
    from config.settings import Settings
    from core.broker import BrokerClient
    from tinkoff.invest.schemas import CandleInterval
    from tinkoff.invest.utils import quotation_to_decimal

    DATA.mkdir(parents=True, exist_ok=True)
    s = Settings()
    b = BrokerClient(token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="bd-study")
    await b.connect()
    now = datetime.now(timezone.utc)
    ok = 0
    for tick, figi in UNIVERSE.items():
        try:
            candles = await b.get_candles(
                figi, now - timedelta(days=92), now, CandleInterval.CANDLE_INTERVAL_HOUR
            )
            rows = [
                {
                    "time": c.time,
                    "open": float(quotation_to_decimal(c.open)),
                    "high": float(quotation_to_decimal(c.high)),
                    "low": float(quotation_to_decimal(c.low)),
                    "close": float(quotation_to_decimal(c.close)),
                    "volume": c.volume,
                }
                for c in candles
            ]
            if len(rows) < 200:
                print(f"  {tick}: only {len(rows)} bars — skip")
                continue
            pd.DataFrame(rows).to_parquet(DATA / f"{tick}.parquet")
            ok += 1
            print(f"  {tick}: {len(rows)} bars")
        except Exception as e:
            print(f"  {tick}: FAIL {e}")
    await b.disconnect()
    print(f"fetched {ok}/{len(UNIVERSE)}")


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = (
        df.set_index("time")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    return d


def find_events(df: pd.DataFrame, n: int, vmult: float, quiet: bool) -> list[int]:
    """Indices t that satisfy the breakdown conditions."""
    lo = df["low"].rolling(n).min().shift(1)  # prior N-bar low
    vmed = df["volume"].rolling(n).median().shift(1)
    ret = df["close"].pct_change()
    rv = ret.rolling(n).std().shift(1)
    rv_rank = rv.rolling(500, min_periods=60).rank(pct=True)
    cond = (df["close"] < lo) & (df["volume"] >= vmult * vmed)
    if quiet:
        cond &= rv_rank <= 0.5
    idx = list(np.where(cond.values)[0])
    # de-cluster: keep first event, skip events within n bars after it
    out, last = [], -(10**9)
    for i in idx:
        if i - last >= n:
            out.append(i)
            last = i
    return out


def event_study(
    frames: dict, n: int, vmult: float, quiet: bool, horizons=(1, 3, 6, 12, 24, 48)
) -> dict:
    fwd = {h: [] for h in horizons}
    n_events = 0
    for tick, df in frames.items():
        ev = find_events(df, n, vmult, quiet)
        c = df["close"].values
        for t in ev:
            for h in horizons:
                if t + h < len(c):
                    fwd[h].append(c[t + h] / c[t] - 1.0)
            n_events += 1
    res = {"n_events": n_events}
    for h in horizons:
        a = np.array(fwd[h])
        if len(a) >= 10:
            res[h] = {
                "mean": float(a.mean() * 100),
                "median": float(np.median(a) * 100),
                "neg_share": float((a < 0).mean() * 100),
                "t": float(a.mean() / (a.std() / np.sqrt(len(a)))) if a.std() > 0 else 0.0,
                "n": len(a),
            }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    if args.fetch:
        asyncio.run(fetch_all())
        return

    raw = {}
    for p in DATA.glob("*.parquet"):
        df = pd.read_parquet(p)
        df["time"] = pd.to_datetime(df["time"])
        raw[p.stem] = df.sort_values("time").reset_index(drop=True)
    print(f"universe: {len(raw)} stocks, 90d hourly")

    for tf, rule in [("15min", None), ("1h", None), ("4h", "4h")]:
        # 15min would need separate fetch; resample only down (1h→4h)
        if tf == "15min":
            continue
        frames = {t: _resample(d, rule) for t, d in raw.items()} if rule else raw
        print("\n" + "=" * 92)
        print(f"TIMEFRAME {tf}")
        print("=" * 92)
        for n, vmult, quiet in [
            (24, 2.0, False),
            (24, 3.0, False),
            (48, 2.0, False),
            (24, 2.0, True),
            (48, 3.0, True),
            (12, 2.0, False),
        ]:
            r = event_study(frames, n, vmult, quiet)
            if r["n_events"] < 15:
                print(f"N={n} V≥{vmult}x quiet={quiet}: {r['n_events']} events (too few)")
                continue
            line = f"N={n} V≥{vmult}x quiet={int(quiet)}: {r['n_events']:>4} ev | "
            for h in (3, 6, 12, 24, 48):
                if h in r:
                    s = r[h]
                    line += f"+{h}h:{s['mean']:+.2f}%(t{s['t']:+.1f}) "
            print(line)
    print("\nIf mean forward returns are NEGATIVE with |t|>2 → continuation edge (H1 holds).")
    print("If positive → mean-reversion; if ~0 → no edge.")


if __name__ == "__main__":
    main()
