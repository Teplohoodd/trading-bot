"""Si (USD/RUB) calendar-spread carry — direction-neutral CIP harvest.

Pure carry-and-hold has ruble-devaluation tail risk.  A CALENDAR SPREAD is
direction-neutral: long 1 next-month + short 1 front-month of the SAME
underlying.  Both legs move ~1:1 with spot, so net USD/RUB delta ≈ 0 — the
P&L is purely the change in the basis (next − front).  We exploit two things:

  1. STRUCTURAL CARRY — the basis is positive (CIP contango, RUB rate > USD
     rate).  We report its annualised level from the live data.
  2. BASIS MEAN-REVERSION — the basis oscillates around its CIP-implied level;
     trade z-score reversion (long spread when basis unusually tight, short
     when unusually wide), with a z-stop.

Risk is tiny vs an outright (same underlying both legs → no fundamental
divergence, only basis/liquidity), so exchanges grant margin offsets and the
effective leverage is high.  We size conservatively (single front-leg margin).

Usage:
    python -u -m futbot.scripts.si_calendar_carry --base Si --days 180
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
logger = logging.getLogger("carry")

COMMISSION_RT = 0.0008
Z_ENTRY = 1.5
Z_STOP = 3.5
ROLL_WIN = 240
MAX_HOLD = 72
HOURS_PER_YEAR = 24 * 365


async def _all_expiries(broker, base):
    futs = await broker.get_all_futures()
    now = datetime.now(timezone.utc)
    out = []
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
        out.append((f, exp))
    out.sort(key=lambda x: x[1])
    return out, now


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


def backtest_basis_reversion(front: np.ndarray, nxt: np.ndarray, return_trades: bool = False):
    """Trade z-score mean-reversion of basis = next - front (delta-neutral).

    P&L per round trip is the change in basis as % of FRONT price (one front-
    contract notional is the capital reference; the calendar is ~delta-flat).
    If return_trades, also return a list of (exit_idx, pnl) for segmentation.
    """
    basis = nxt - front
    n = len(basis)
    if n < ROLL_WIN + 50:
        return ({"n": 0}, []) if return_trades else {"n": 0}
    z = np.full(n, np.nan)
    for t in range(ROLL_WIN, n):
        w = basis[t - ROLL_WIN : t]
        sd = w.std()
        z[t] = (basis[t] - w.mean()) / sd if sd > 0 else 0.0
    pos = 0
    entry = None
    pnls = []
    trades = []
    for t in range(ROLL_WIN, n):
        if np.isnan(z[t]):
            continue
        if pos == 0:
            if z[t] > Z_ENTRY:
                pos, entry = -1, t  # short basis (short next / long front)
            elif z[t] < -Z_ENTRY:
                pos, entry = +1, t  # long basis
            continue
        crossed = (pos == +1 and z[t] >= 0) or (pos == -1 and z[t] <= 0)
        stopped = abs(z[t]) >= Z_STOP
        timed = (t - entry) >= MAX_HOLD
        if not (crossed or stopped or timed):
            continue
        d_basis = basis[t] - basis[entry]
        ref = front[entry]
        gross = pos * d_basis / ref if ref > 0 else 0
        pnl = gross - COMMISSION_RT
        pnls.append(pnl)
        trades.append((t, pnl))
        pos = 0
    if not pnls:
        return ({"n": 0}, []) if return_trades else {"n": 0}
    a = np.array(pnls)
    stats = {
        "n": len(a),
        "win": float((a > 0).mean()),
        "total_pct": float(a.sum() * 100),
        "avg_pct": float(a.mean() * 100),
        "sharpe": float(a.mean() / a.std() * math.sqrt(len(a))) if a.std() > 0 else 0.0,
    }
    return (stats, trades) if return_trades else stats


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="Si")
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()

    s = Settings()
    broker = BrokerClient(
        token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="carry"
    )
    await broker.connect()
    cands, now = await _all_expiries(broker, args.base)
    cands = [(f, e) for f, e in cands if (e - now).days >= 5]
    if len(cands) < 2:
        logger.error("Need ≥2 live expiries")
        await broker.disconnect()
        return
    (f_front, e_front), (f_next, e_next) = cands[0], cands[1]
    logger.info(
        f"Front {f_front.ticker} exp {e_front.date()}  " f"Next {f_next.ticker} exp {e_next.date()}"
    )
    df_f = await _fetch(broker, f_front.figi, args.days)
    df_n = await _fetch(broker, f_next.figi, args.days)
    meta = broker.extract_futures_metadata(f_front)
    dlong = float(meta.get("dlong") or 0.1)
    await broker.disconnect()

    al = pd.concat(
        [
            df_f.set_index("time")["close"].rename("front"),
            df_n.set_index("time")["close"].rename("next"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    logger.info(f"Aligned {len(al)} bars")
    front = al["front"].values
    nxt = al["next"].values

    dt_years = max((e_next - e_front).days, 1) / 365.0
    ann_basis = (nxt / front - 1.0).mean() / dt_years * 100

    print("\n" + "=" * 90)
    print(f"Si CALENDAR-SPREAD CARRY  ({f_front.ticker}/{f_next.ticker})")
    print("=" * 90)
    print(f"Structural carry (mean annualised basis over {len(al)} bars): " f"{ann_basis:+.1f}%/yr")
    print(f"  → this is the CIP contango you earn structurally if spot is flat")
    print(
        f"  → leg margin ≈ {dlong*100:.1f}% → calendar leverage is high "
        f"(exchange grants spread margin offset)"
    )
    print()
    r, trades = backtest_basis_reversion(front, nxt, return_trades=True)
    if r.get("n", 0):
        lev = 1.0 / dlong
        print(f"BASIS MEAN-REVERSION backtest (delta-neutral, z_entry={Z_ENTRY}):")
        print(
            f"  trades={r['n']}  win={r['win']*100:.0f}%  "
            f"avg={r['avg_pct']:+.3f}%  total={r['total_pct']:+.2f}% (notional)"
        )
        print(f"  per-trade Sharpe={r['sharpe']:+.2f}")
        print(
            f"  total on-margin ≈ {r['total_pct']*lev:+.1f}% "
            f"(lev {lev:.1f}×, conservative single-leg)"
        )

        # ── Walk-forward consistency: bucket trades into time segments ──
        # The strategy is parameter-light (rolling-z, fixed z_entry), so the
        # OOS test is consistency: is the edge spread across the timeline or
        # concentrated in one lucky window?
        n_seg = 4
        seg_len = len(al) // n_seg
        print(f"\n  WALK-FORWARD consistency ({n_seg} time blocks):")
        pos_segs = 0
        for i in range(n_seg):
            a0 = i * seg_len
            a1 = (i + 1) * seg_len if i < n_seg - 1 else len(al)
            seg_pnls = [p for (idx, p) in trades if a0 <= idx < a1]
            if seg_pnls:
                arr = np.array(seg_pnls)
                sh = arr.mean() / arr.std() * math.sqrt(len(arr)) if arr.std() > 0 else 0.0
                tot = arr.sum() * 100
                wr = (arr > 0).mean() * 100
                flag = "✅" if tot > 0 else "❌"
                if tot > 0:
                    pos_segs += 1
            else:
                sh, tot, wr, flag = 0.0, 0.0, 0.0, "·"
            t0 = str(al.index[a0].date())
            t1 = str(al.index[min(a1, len(al)) - 1].date())
            print(
                f"    seg{i+1} [{t0}..{t1}]: n={len(seg_pnls):>2} "
                f"win={wr:>3.0f}% total={tot:>+6.2f}% Sharpe={sh:>+5.2f} {flag}"
            )
        active = sum(
            1
            for i in range(n_seg)
            if any(
                i * seg_len <= idx < (i + 1) * seg_len if i < n_seg - 1 else idx >= i * seg_len
                for idx, _ in trades
            )
        )
        print(
            f"  → positive blocks: {pos_segs}/{n_seg}  "
            f"verdict: {'✅ CONSISTENT (build bot)' if pos_segs >= 3 else '⚠ concentrated/fragile'}"
        )
    else:
        print("BASIS MEAN-REVERSION: too few trades / insufficient data")


if __name__ == "__main__":
    asyncio.run(main())
