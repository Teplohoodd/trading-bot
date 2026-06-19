"""MOEX ISS multi-year HOURLY history — incl. expired futures (stitched).

Tinkoff caps hourly history at ~180 days, which is why the pairs walk-forward
could only use daily spot.  MOEX ISS (free, public, no auth) retains hourly
candles for EXPIRED futures contracts, so we can stitch consecutive quarterly
contracts (H/M/U/Z) into a continuous front-month hourly series spanning years
— enabling a faithful test of the HOURLY pairs strategy across regimes.

Continuous construction: each quarterly contract is the "front" from the prior
contract's roll boundary to its own; we take hourly bars only within that
window and concatenate.  Roll boundary ≈ mid contract-month (Si et al. expire
mid-month); good enough for backtests.

Usage (library):
    from futbot.scripts.moex_iss_history import continuous_hourly
    df = continuous_hourly("Si", 2022, 2025)   # columns: time, open/high/low/close
"""

import sys
import time as _time
import urllib.request
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ISS = "https://iss.moex.com/iss"
QUARTERLY = ["H", "M", "U", "Z"]  # Mar, Jun, Sep, Dec
QMONTH = {"H": 3, "M": 6, "U": 9, "Z": 12}


def _get(url: str, retries: int = 3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.loads(urllib.request.urlopen(req, timeout=40).read())
        except Exception:
            _time.sleep(1.5 * (i + 1))
    return None


def _candles(engine, market, secid, interval, frm, till):
    """Paginated hourly candles for one security."""
    out = []
    start = 0
    while True:
        url = (
            f"{ISS}/engines/{engine}/markets/{market}/securities/{secid}"
            f"/candles.json?interval={interval}&from={frm}&till={till}&start={start}"
        )
        d = _get(url)
        if not d or "candles" not in d:
            break
        cols = d["candles"]["columns"]
        rows = d["candles"]["data"]
        if not rows:
            break
        ci = {c: i for i, c in enumerate(cols)}
        for r in rows:
            out.append(
                {
                    "time": r[ci["begin"]],
                    "open": r[ci["open"]],
                    "close": r[ci["close"]],
                    "high": r[ci["high"]],
                    "low": r[ci["low"]],
                    "volume": r[ci["volume"]],
                }
            )
        if len(rows) < 500:
            break
        start += len(rows)
        _time.sleep(0.15)
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["time"] = pd.to_datetime(df["time"])
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def continuous_hourly(
    base: str, start_year: int, end_year: int, engine="futures", market="forts"
) -> pd.DataFrame:
    """Stitch quarterly contracts into a continuous front-month hourly series."""
    # Build ordered contract list with (secid, roll_start, roll_end)
    contracts = []
    prev_boundary = datetime(start_year - 1, 12, 15, tzinfo=timezone.utc)
    for yr in range(start_year, end_year + 1):
        for q in QUARTERLY:
            secid = f"{base}{q}{yr % 10}"
            exp_month = QMONTH[q]
            # roll boundary ≈ 15th of expiry month
            boundary = datetime(yr, exp_month, 15, tzinfo=timezone.utc)
            contracts.append((secid, prev_boundary, boundary))
            prev_boundary = boundary

    frames = []
    for secid, lo, hi in contracts:
        if hi < datetime(start_year, 1, 1, tzinfo=timezone.utc):
            continue
        df = _candles(engine, market, secid, 60, lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"))
        if df.empty:
            continue
        df = df[(df["time"] >= lo.replace(tzinfo=None)) & (df["time"] < hi.replace(tzinfo=None))]
        if not df.empty:
            df["contract"] = secid
            frames.append(df)
            print(
                f"  {base}: {secid} {len(df)} bars "
                f"[{df['time'].min().date()}..{df['time'].max().date()}]"
            )
    if not frames:
        return pd.DataFrame()
    full = (
        pd.concat(frames)
        .drop_duplicates("time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )
    return full


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Si")
    ap.add_argument("--y0", type=int, default=2022)
    ap.add_argument("--y1", type=int, default=2025)
    args = ap.parse_args()
    print(f"Building continuous hourly {args.base} {args.y0}-{args.y1}…")
    df = continuous_hourly(args.base, args.y0, args.y1)
    if df.empty:
        print("NO DATA")
    else:
        print(
            f"\nTOTAL {args.base}: {len(df)} hourly bars  "
            f"[{df['time'].min()}..{df['time'].max()}]  "
            f"contracts={df['contract'].nunique()}"
        )
        out = Path(__file__).resolve().parents[2] / "data" / f"hist_{args.base}.parquet"
        df.to_parquet(out)
        print(f"saved → {out}")
