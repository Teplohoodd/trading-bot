"""Combined orchestrator projection — what would ALL strategies have earned?

Honest, backtest-derived ₽/month at 50k/300k/600k for the three validated
orchestrator strategies, using each bot's ACTUAL sizing logic:

  • carry  — Si calendar spread.  10% budget, MAX_LOTS=10.  Scales w/ capital.
  • pairs  — validated LK-Si, GZ-Si, LK-RN.  compute_lots (30%/pair, max 3
             open).  Scales w/ capital.
  • trend  — triple_top/triple_bottom.  FIXED 1 lot/signal (max 10 concurrent),
             so its ₽ is INDEPENDENT of portfolio size; %/portfolio shrinks.

All figures are IN-SAMPLE / short-history (≈180d) and PAPER — treat as an
optimistic upper bound, not a forward guarantee.
"""

import asyncio
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.disable(logging.CRITICAL)

from config.settings import Settings
from core.broker import BrokerClient
from t_tech.invest import CandleInterval
from t_tech.invest.utils import quotation_to_decimal
from futbot.pairs.execution import compute_lots, compute_two_leg_pnl

PORTFOLIOS = [50_000, 300_000, 600_000]
VALIDATED_PAIRS = ["LK-Si", "GZ-Si", "LK-RN"]
Z_ENTRY, Z_STOP, ROLL_WIN, MAX_HOLD = 2.0, 4.0, 240, 48
COMMISSION_RT = 0.0016


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


async def _fetch(broker, figi, days):
    now = datetime.now(timezone.utc)
    chunks, end, left = [], now, days
    while left > 0:
        cd = min(left, 89)
        start = end - timedelta(days=cd)
        try:
            c = await broker.get_candles(
                figi, start, end, interval=CandleInterval.CANDLE_INTERVAL_HOUR
            )
            rows = [{"time": x.time, "close": float(quotation_to_decimal(x.close))} for x in c]
            if rows:
                chunks.append(pd.DataFrame(rows))
        except Exception:
            pass
        end = start
        left -= cd
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def _pair_trades_pct(y, x, beta):
    """Return list of per-trade pnl as % of COMBINED notional (rolling-z)."""
    spread = y - beta * x
    n = len(spread)
    z = np.full(n, np.nan)
    for t in range(ROLL_WIN, n):
        w = spread[t - ROLL_WIN : t]
        sd = w.std()
        z[t] = (spread[t] - w.mean()) / sd if sd > 0 else 0
    pos, entry, out = 0, None, []
    for t in range(ROLL_WIN, n):
        if np.isnan(z[t]):
            continue
        if pos == 0:
            if z[t] > Z_ENTRY:
                pos, entry = -1, t
            elif z[t] < -Z_ENTRY:
                pos, entry = +1, t
            continue
        crossed = (pos == +1 and z[t] >= 0) or (pos == -1 and z[t] <= 0)
        if not (crossed or abs(z[t]) >= Z_STOP or (t - entry) >= MAX_HOLD):
            continue
        out.append((entry, t))
        pos = 0
    return out


async def main():
    s = Settings()
    b = BrokerClient(
        token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="combined-proj"
    )
    await b.connect()

    # ── CARRY (per-lot ₽/180d) ─────────────────────────────────────────
    cands = []
    futs = await b.get_all_futures()
    now = datetime.now(timezone.utc)
    for f in futs:
        t = getattr(f, "ticker", "") or ""
        if t.startswith("Si") and len(t) == 4:
            exp = getattr(f, "expiration_date", None)
            if exp is None:
                continue
            if hasattr(exp, "ToDatetime"):
                exp = exp.ToDatetime()
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if (exp - now).days >= 5:
                cands.append((f, exp))
    cands.sort(key=lambda x: x[1])
    (ff, fe), (nf, ne) = cands[0], cands[1]
    df_f = await _fetch(b, ff.figi, 180)
    df_n = await _fetch(b, nf.figi, 180)
    meta = b.extract_futures_metadata(ff)
    si_dlong = float(meta.get("dlong") or 0.1)
    si_rpp = float(meta.get("rub_per_point") or 1.0)
    si_lot = int(getattr(ff, "lot", 1) or 1)
    al = pd.concat(
        [df_f.set_index("time")["close"].rename("f"), df_n.set_index("time")["close"].rename("n")],
        axis=1,
        join="inner",
    ).dropna()
    span_days = (al.index.max() - al.index.min()).total_seconds() / 86400
    months = span_days / 30.0
    front = al["f"].values
    nxt = al["n"].values
    basis = nxt - front
    z = np.full(len(basis), np.nan)
    for t in range(ROLL_WIN, len(basis)):
        w = basis[t - ROLL_WIN : t]
        sd = w.std()
        z[t] = (basis[t] - w.mean()) / sd if sd > 0 else 0
    pos, entry, carry_pnls = 0, None, []
    for t in range(ROLL_WIN, len(basis)):
        if np.isnan(z[t]):
            continue
        if pos == 0:
            if z[t] > 1.5:
                pos, entry = -1, t
            elif z[t] < -1.5:
                pos, entry = +1, t
            continue
        crossed = (pos == +1 and z[t] >= 0) or (pos == -1 and z[t] <= 0)
        if not (crossed or abs(z[t]) >= 3.5 or (t - entry) >= 72):
            continue
        d = basis[t] - basis[entry]
        # carry RT commission is 0.0008 (single-instrument round trip), applied
        # as a fraction of front notional — matches si_calendar_carry.py.
        net_frac = pos * d / front[entry] - 0.0008
        carry_pnls.append(net_frac * front[entry])  # ₽ per 1-lot spread
        pos = 0
    si_notional = front[-1] * si_lot * si_rpp
    si_margin = si_notional * si_dlong
    carry_rub_per_lot = float(np.sum(carry_pnls))  # ₽/180d for 1-lot spread
    carry_trades = len(carry_pnls)

    # ── PAIRS (validated) ───────────────────────────────────────────────
    pair_data = {}
    for pr in VALIDATED_PAIRS:
        a, c = pr.split("-")
        for base in (a, c):
            if base not in pair_data:
                fr = await _resolve_front(b, base)
                if fr is None:
                    continue
                df = await _fetch(b, fr.figi, 180)
                m = b.extract_futures_metadata(fr)
                pair_data[base] = {
                    "srs": df.set_index("time")["close"],
                    "price": float(df["close"].iloc[-1]),
                    "rpp": float(m.get("rub_per_point") or 1.0),
                    "lot": int(getattr(fr, "lot", 1) or 1),
                    "dlong": float(m.get("dlong") or 0.0),
                    "dshort": float(m.get("dshort") or 0.0),
                }
    await b.disconnect()

    # ── TREND from pattern CSV ──────────────────────────────────────────
    trend_rub = 0.0
    trend_trades = 0
    csv = Path("data/patterns_backtest_v2.csv")
    if csv.exists():
        td = pd.read_csv(csv)
        td = td[td["pattern"].isin(["triple_top", "triple_bottom"])]
        # ₽ per trade at 1 lot = net% × notional(contract). Approx notional
        # using entry_price (FORTS rpp≈1, lot≈1 for most; conservative).
        # We don't have per-contract lot/rpp here, so approximate notional as
        # entry_price (most FORTS single-name futures: 1 lot ≈ price ₽).
        trend_trades = len(td)
        trend_rub = float((td["net_pnl_pct"] / 100.0 * td["entry_price"]).sum())

    # ════════════════ REPORT ════════════════
    print("=" * 84)
    print(f"COMBINED ORCHESTRATOR PROJECTION  (backtest ≈ {months:.1f} months, PAPER)")
    print("=" * 84)
    print(
        f"\nCarry: {carry_trades} trades/{months:.0f}mo, {carry_rub_per_lot:,.0f} ₽/180d per lot "
        f"(margin {si_margin:,.0f}/lot)"
    )
    print(
        f"Trend: {trend_trades} triple trades/{months:.0f}mo, fixed 1 lot/signal → "
        f"{trend_rub:,.0f} ₽/180d TOTAL (portfolio-independent)"
    )

    print(
        f"\n{'Portfolio':>10} │ {'carry ₽/mo':>11} {'pairs ₽/mo':>11} "
        f"{'trend ₽/mo':>11} │ {'TOTAL ₽/mo':>11} {'%/mo':>7} {'%/yr':>7}"
    )
    print("-" * 84)
    for P in PORTFOLIOS:
        # carry lots
        c_lots = max(1, min(10, math.floor((P * 0.10) / si_margin)))
        carry_mo = carry_rub_per_lot / months * c_lots
        # pairs ₽
        pairs_rub_total = 0.0
        pairs_tr = 0
        for pr in VALIDATED_PAIRS:
            ay, ax = pr.split("-")
            if ay not in pair_data or ax not in pair_data:
                continue
            dy, dx = pair_data[ay], pair_data[ax]
            al2 = pd.concat(
                [dy["srs"].rename("y"), dx["srs"].rename("x")], axis=1, join="inner"
            ).dropna()
            if len(al2) < ROLL_WIN + 50:
                continue
            y = al2["y"].values
            x = al2["x"].values
            beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
            ly, lx = compute_lots(
                portfolio_value=P,
                beta=beta,
                price_y=dy["price"],
                price_x=dx["price"],
                rpp_y=dy["rpp"],
                rpp_x=dx["rpp"],
                lot_size_y=dy["lot"],
                lot_size_x=dx["lot"],
                dlong_y=dy["dlong"],
                dshort_y=dy["dshort"],
                dlong_x=dx["dlong"],
                dshort_x=dx["dshort"],
                direction=+1,
                capital_per_pair_pct=0.30,
            )
            if ly <= 0 or lx <= 0:
                continue
            for e, t in _pair_trades_pct(y, x, beta):
                # direction sign of trade
                sp = y - beta * x
                d = +1 if sp[e] < np.mean(sp[max(0, e - ROLL_WIN) : e]) else -1
                pnl = compute_two_leg_pnl(
                    direction=d,
                    beta=beta,
                    entry_y=y[e],
                    entry_x=x[e],
                    exit_y=y[t],
                    exit_x=x[t],
                    lots_y=ly,
                    lots_x=lx,
                    rpp_y=dy["rpp"],
                    rpp_x=dx["rpp"],
                    lot_size_y=dy["lot"],
                    lot_size_x=dx["lot"],
                    base_y=ay,
                    base_x=ax,
                )
                pairs_rub_total += pnl["net_rub"]
                pairs_tr += 1
        pairs_mo = pairs_rub_total / months
        trend_mo = trend_rub / months  # fixed, portfolio-independent
        total_mo = carry_mo + pairs_mo + trend_mo
        print(
            f"{P:>10,} │ {carry_mo:>11,.0f} {pairs_mo:>11,.0f} {trend_mo:>11,.0f} │ "
            f"{total_mo:>11,.0f} {total_mo/P*100:>6.2f}% {total_mo/P*1200:>6.1f}%"
        )
    print("-" * 84)
    print("Caveats: in-sample, ~6mo history, PAPER, no slippage beyond commission.")
    print("Trend ₽ is fixed (1 lot/signal) → %/portfolio shrinks as capital grows.")
    print("Pairs share the Si leg (LK-Si, GZ-Si) → capped by max-3-open + 30%/pair.")


if __name__ == "__main__":
    asyncio.run(main())
