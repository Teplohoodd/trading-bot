"""Trend pattern portfolio simulator — what do the position limits cost/save?

Reconstructs a realistic portfolio equity curve from the triple_top/bottom
backtest trades (data/patterns_backtest_v2.csv), enforcing a configurable
concurrency cap and lot size, then reports for each config:

    total ₽ / %       — return over the ~6-month backtest
    max drawdown      — realized-equity peak-to-trough (the pain)
    peak concurrent   — most positions open at once (correlation risk)
    peak margin util  — max margin / portfolio (can you even hold it?)

This makes the limit trade-off explicit: removing the cap raises return but
also raises peak concurrent exposure, margin usage, and drawdown.  Per-trade
₽ uses each contract's real notional (price × lot × rub_per_point) and margin
(dlong).  IN-SAMPLE, ~6mo, PAPER — trend patterns are NOT walk-forward
validated, so scaling them up scales an unproven edge.

Usage:
    python -u -m futbot.scripts.trend_portfolio_sim --portfolio 300000
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.disable(logging.CRITICAL)

from config.settings import Settings
from core.broker import BrokerClient


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


def simulate(
    trades: pd.DataFrame,
    *,
    notional: dict,
    margin: dict,
    portfolio: float,
    max_open: int,
    lots: int,
    per_trade_pct: float | None,
):
    """Walk trades chronologically with a concurrency cap.

    per_trade_pct: if set, size each trade to that % of portfolio margin
    (lots derived per-contract); else use fixed `lots`.
    """
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    # Build event list; we accept/reject at entry based on current open count.
    open_trades = []  # list of dicts with exit_time, pnl_rub, margin_rub
    realized = []  # (time, pnl_rub) at exits
    peak_concurrent = 0
    peak_margin = 0.0
    accepted = 0
    import math

    # We process entries in order; to know "currently open" we must release
    # trades whose exit_time <= this entry_time.
    def _release(now_t):
        nonlocal open_trades
        still = []
        for ot in open_trades:
            if ot["exit_time"] <= now_t:
                realized.append((ot["exit_time"], ot["pnl_rub"]))
            else:
                still.append(ot)
        open_trades = still

    for _, r in trades.iterrows():
        et = r["entry_time"]
        _release(et)
        if len(open_trades) >= max_open:
            continue  # cap hit — skip signal (like live)
        base = r["base"]
        notl = notional.get(base, 0.0)
        marg = margin.get(base, 0.0)
        if notl <= 0:
            continue
        if per_trade_pct is not None and marg > 0:
            n_lots = max(1, math.floor((portfolio * per_trade_pct) / marg))
        else:
            n_lots = lots
        pnl_rub = r["net_pnl_pct"] / 100.0 * notl * n_lots
        margin_rub = marg * n_lots
        open_trades.append(
            {"exit_time": r["exit_time"], "pnl_rub": pnl_rub, "margin_rub": margin_rub}
        )
        accepted += 1
        peak_concurrent = max(peak_concurrent, len(open_trades))
        cur_margin = sum(o["margin_rub"] for o in open_trades)
        peak_margin = max(peak_margin, cur_margin)

    # release the rest
    for ot in open_trades:
        realized.append((ot["exit_time"], ot["pnl_rub"]))

    realized.sort(key=lambda x: x[0])
    pnls = np.array([p for _, p in realized])
    total = float(pnls.sum())
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    mdd = float((cum - peak).min()) if len(cum) else 0.0
    return {
        "accepted": accepted,
        "total_rub": total,
        "mdd_rub": mdd,
        "peak_concurrent": peak_concurrent,
        "peak_margin": peak_margin,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", type=float, default=300_000)
    args = ap.parse_args()
    P = args.portfolio

    df = pd.read_csv("data/patterns_backtest_v2.csv")
    df = df[df["pattern"].isin(["triple_top", "triple_bottom"])].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    bases = sorted(df["base"].unique())
    span_days = (df["exit_time"].max() - df["entry_time"].min()).total_seconds() / 86400
    months = span_days / 30.0

    s = Settings()
    b = BrokerClient(token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="trend-sim")
    await b.connect()
    notional, margin = {}, {}
    for base in bases:
        f = await _resolve_front(b, base)
        if f is None:
            continue
        m = b.extract_futures_metadata(f)
        price = float(await b.get_last_price(f.figi))
        rpp = float(m.get("rub_per_point") or 1.0)
        lot = int(getattr(f, "lot", 1) or 1)
        dlong = float(m.get("dlong") or 0.0)
        notl = price * lot * rpp
        notional[base] = notl
        margin[base] = notl * (dlong if dlong > 0 else 0.25)
    await b.disconnect()

    print("=" * 92)
    print(f"TREND PATTERN PORTFOLIO SIM  (portfolio {P:,.0f} ₽, ~{months:.1f} months, IN-SAMPLE)")
    print("=" * 92)
    print(f"{len(df)} triple trades across {len(bases)} contracts")
    print(
        f"\n{'config':<34}{'₽/mo':>9}{'%/mo':>7}{'maxDD ₽':>11}"
        f"{'peakPos':>8}{'peakMargin%':>12}"
    )
    print("-" * 92)

    configs = [
        ("CURRENT: max10, 1 lot", dict(max_open=10, lots=1, per_trade_pct=None)),
        ("no cap (max26), 1 lot", dict(max_open=26, lots=1, per_trade_pct=None)),
        ("max10, 2 lots", dict(max_open=10, lots=2, per_trade_pct=None)),
        ("no cap, size 3%/trade", dict(max_open=26, lots=1, per_trade_pct=0.03)),
        ("no cap, size 5%/trade", dict(max_open=26, lots=1, per_trade_pct=0.05)),
    ]
    for name, cfg in configs:
        r = simulate(df, notional=notional, margin=margin, portfolio=P, **cfg)
        pm_pct = r["peak_margin"] / P * 100
        flag = "  ⚠ OVER-MARGIN" if pm_pct > 100 else ""
        print(
            f"{name:<34}{r['total_rub']/months:>9,.0f}"
            f"{r['total_rub']/months/P*100:>6.2f}%{r['mdd_rub']:>11,.0f}"
            f"{r['peak_concurrent']:>8}{pm_pct:>11.0f}%{flag}"
        )
    print("-" * 92)
    print("maxDD = realized-equity peak-to-trough (intratrade MTM worse).")
    print("peakMargin% > 100 → position physically un-holdable on this portfolio.")
    print("Reminder: trend patterns are IN-SAMPLE, not walk-forward validated.")


if __name__ == "__main__":
    asyncio.run(main())
