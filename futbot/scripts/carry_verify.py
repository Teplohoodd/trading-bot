"""carry_verify.py — ROLL-SAFE re-check of the multi-instrument carry results.

carry_lab.py concatenates per-contract front/next segments and backtests the
joined array.  That lets a position span a contract-roll boundary (basis jumps
discontinuously) and lets the rolling-z window straddle rolls — both can
manufacture spurious P&L and inflate Sharpe.

This re-runs the SAME basis mean-reversion, but PER SEGMENT (each contract
pair independently): z is computed only within a segment, and no trade can
cross a roll.  If an instrument's edge survives this, it's real; if it
collapses, carry_lab was contaminated by roll discontinuities.
"""

import sys
import math
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futbot.scripts.moex_iss_history import _candles

COMMISSION_RT = 0.0008
Z_ENTRY, Z_STOP, ROLL_WIN, MAX_HOLD = 1.5, 3.5, 240, 72
QUARTERLY = ["H", "M", "U", "Z"]
QMONTH = {"H": 3, "M": 6, "U": 9, "Z": 12}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BASES = ["Si", "GK", "GZ", "RN", "LK", "SR"]


def _ordered_contracts(base, y0, y1):
    res, prev = [], None
    for yr in range(y0 - 1, y1 + 2):
        for q in QUARTERLY:
            secid = f"{base}{q}{yr % 10}"
            boundary = pd.Timestamp(yr, QMONTH[q], 15)
            if prev is not None:
                res.append((secid, prev, boundary))
            prev = boundary
    return res


def segment_trades(front, nxt, times):
    """Causal basis mean-reversion WITHIN one segment. Returns [(year,pnl)]."""
    basis = nxt - front
    n = len(basis)
    if n < ROLL_WIN + 30:
        return []
    out = []
    pos, entry = 0, None
    for t in range(ROLL_WIN, n):
        w = basis[t - ROLL_WIN : t]
        sd = w.std()
        if sd <= 0:
            continue
        z = (basis[t] - w.mean()) / sd
        if pos == 0:
            if z > Z_ENTRY:
                pos, entry = -1, t
            elif z < -Z_ENTRY:
                pos, entry = +1, t
            continue
        crossed = (pos == +1 and z >= 0) or (pos == -1 and z <= 0)
        if not (crossed or abs(z) >= Z_STOP or (t - entry) >= MAX_HOLD):
            continue
        d = basis[t] - basis[entry]
        ref = front[entry]
        pnl = (pos * d / ref if ref > 0 else 0) - COMMISSION_RT
        out.append((pd.Timestamp(times[entry]).year, pnl))
        pos = 0
    return out


def analyse(base, y0=2022, y1=2025):
    path = DATA_DIR / f"hist_{base}.parquet"
    if not path.exists():
        return None
    raw = pd.read_parquet(path)
    raw["time"] = pd.to_datetime(raw["time"])
    by_contract = {s: g.set_index("time")["close"].sort_index() for s, g in raw.groupby("contract")}
    contracts = _ordered_contracts(base, y0, y1)
    t_min, t_max = pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y1+1}-01-01")

    all_trades = []
    n_segments = 0
    for i, (secid, lo, hi) in enumerate(contracts):
        lo_c, hi_c = max(lo, t_min), min(hi, t_max)
        if lo_c >= hi_c or secid not in by_contract or i + 1 >= len(contracts):
            continue
        next_secid = contracts[i + 1][0]
        f = by_contract[secid]
        f = f[(f.index >= lo_c) & (f.index < hi_c)]
        if f.empty:
            continue
        nd = _candles(
            "futures", "forts", next_secid, 60, lo_c.strftime("%Y-%m-%d"), hi_c.strftime("%Y-%m-%d")
        )
        _time.sleep(0.1)
        if nd.empty:
            continue
        ns = nd.set_index("time")["close"].sort_index()
        ns = ns[(ns.index >= lo_c) & (ns.index < hi_c)]
        al = pd.DataFrame({"front": f, "next": ns}).dropna()
        if len(al) < ROLL_WIN + 30:
            continue
        n_segments += 1
        all_trades += segment_trades(
            al["front"].values.astype(float), al["next"].values.astype(float), al.index.to_numpy()
        )
    if not all_trades:
        return None
    df = pd.DataFrame(all_trades, columns=["year", "pnl"])
    per_year = df.groupby("year")["pnl"].agg(["count", "sum", lambda s: (s > 0).mean()])
    per_year.columns = ["n", "total", "win"]
    pnls = df["pnl"].values
    pos_years = int((per_year["total"] > 0).sum())
    n_years = len(per_year)
    return {
        "base": base,
        "segments": n_segments,
        "trades": len(pnls),
        "win": (pnls > 0).mean(),
        "total": pnls.sum() * 100,
        "sharpe": pnls.mean() / pnls.std() * math.sqrt(len(pnls)) if pnls.std() > 0 else 0,
        "pos_years": pos_years,
        "n_years": n_years,
        "per_year": per_year,
    }


def main():
    print("ROLL-SAFE carry verification (per-segment, no cross-roll trades)")
    print("=" * 78)
    rows = []
    for base in BASES:
        r = analyse(base)
        if r is None:
            print(f"{base}: no data")
            continue
        rows.append(r)
        py = "  ".join(f"{y}:{v:+.1f}%" for y, v in r["per_year"]["total"].items())
        verdict = "ROBUST" if r["pos_years"] >= r["n_years"] - 1 else "FRAGILE"
        print(
            f"\n{base}: segs={r['segments']} trades={r['trades']} "
            f"win={r['win']*100:.0f}% total={r['total']:+.1f}% "
            f"Sharpe(t)={r['sharpe']:+.2f}  {r['pos_years']}/{r['n_years']}yr  {verdict}"
        )
        print(f"   per-year: {py}")
    print("\n" + "=" * 78)
    print("Compare to carry_lab (concatenated): if numbers shrink a lot here,")
    print("carry_lab was inflated by roll-boundary contamination.")


if __name__ == "__main__":
    main()
