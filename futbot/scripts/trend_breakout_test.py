"""Trend-breakout backtest — mimics OsEngine's FuturesTrend strategies.

Tests two classic trend-following approaches that have been published
in trading literature for decades and are battle-tested in OsEngine
(the largest Russian retail algo platform):

  1. Bollinger Breakout (FuturesTrendBollinger.cs pattern)
       Long  on close > BB_upper(N, k_dev)
       Short on close < BB_lower(N, k_dev)
       Exit  on close crossing opposite band
       Default N=50 bars, k=1.9 σ
       NO STOP-LOSS — pure mechanical band-flip exit

  2. Donchian Channel Breakout (FuturesTrendPriceChannel.cs pattern)
       Long  on close > rolling-N high (prior bars)
       Short on close < rolling-N low
       Exit  on close crossing opposite extreme
       Default N=50 bars
       NO STOP-LOSS

Tested on:
  * Hourly candles, 180 days
  * 6 FORTS contracts (BR, GZ, SR, LK, MX, Si)
  * Commission 0.04 % per side
  * Position sizing 1 lot (paper)
  * Grid: N ∈ {20, 30, 50, 80}, k ∈ {1.5, 2.0, 2.5} (Bollinger only)

This is the FIRST scalp-free test we run.  If trend breakout shows
positive NET on hourly bars, we have a path to "swing-style scalp"
matching OsEngine philosophy and the user's intuition that bigger
moves = lower commission ratio.

Usage:
    python -m futbot.scripts.trend_breakout_test           # default 180d
    python -m futbot.scripts.trend_breakout_test 365       # 365d
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

from config.settings import Settings
from core.broker import BrokerClient
from tinkoff.invest import CandleInterval
from tinkoff.invest.utils import quotation_to_decimal

from futbot.utils import commissions as comm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("tinkoff").setLevel(logging.WARNING)
logger = logging.getLogger("trend_test")


BASES = ["BR", "GZ", "SR", "LK", "MX", "Si"]


# ─────────────────────────────────────────────────────────────────────────────
# Data fetch
# ─────────────────────────────────────────────────────────────────────────────
async def _resolve_front_month(broker, base: str):
    futs = await broker.get_all_futures()
    cands = []
    for f in futs:
        t = getattr(f, "ticker", "") or ""
        if t == base or (t.startswith(base) and len(t) == len(base) + 2):
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
    now = datetime.now(timezone.utc)
    for f, exp in cands:
        if (exp - now).days >= 14:
            return f
    return cands[0][0]


def _candles_to_df(candles) -> pd.DataFrame:
    rows = [
        {
            "time": c.time,
            "open": float(quotation_to_decimal(c.open)),
            "high": float(quotation_to_decimal(c.high)),
            "low": float(quotation_to_decimal(c.low)),
            "close": float(quotation_to_decimal(c.close)),
            "volume": int(c.volume),
        }
        for c in candles
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Bollinger Breakout
# ─────────────────────────────────────────────────────────────────────────────
def backtest_bollinger(
    df: pd.DataFrame, *, base: str, n: int, k: float, rpp: float, lot_size: int
) -> dict:
    close = df["close"].values
    ma = pd.Series(close).rolling(n).mean()
    sd = pd.Series(close).rolling(n).std()
    upper = (ma + k * sd).values
    lower = (ma - k * sd).values

    pos = 0
    entry_p = 0.0
    entry_idx = None
    pnls = []
    holds = []
    for t in range(n + 5, len(df)):
        if np.isnan(upper[t]) or np.isnan(lower[t]):
            continue
        c = close[t]
        if pos == 0:
            if c > upper[t]:
                pos, entry_p, entry_idx = +1, c, t
            elif c < lower[t]:
                pos, entry_p, entry_idx = -1, c, t
            continue
        # Exit on opposite band cross
        exit_p = None
        if pos == +1 and c < lower[t]:
            exit_p = c
        elif pos == -1 and c > upper[t]:
            exit_p = c
        if exit_p is None:
            continue
        net, _, _, _ = comm.round_trip_pnl(
            direction="buy" if pos > 0 else "sell",
            entry_price=entry_p,
            exit_price=exit_p,
            lots=1,
            lot_size=lot_size,
            rub_per_point=rpp,
            instrument_kind="future",
            base_ticker=base,
        )
        pnls.append(net)
        holds.append(t - entry_idx)
        pos = 0
        entry_idx = None

    if not pnls:
        return {"n_trades": 0}
    arr = np.array(pnls)
    sharpe = arr.mean() / arr.std() * math.sqrt(len(arr)) if arr.std() > 0 else 0.0
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = float((cum - peak).min())
    return {
        "n_trades": len(arr),
        "win_rate": round(float((arr > 0).mean()), 3),
        "total_rub": round(float(arr.sum()), 2),
        "avg_rub": round(float(arr.mean()), 3),
        "median_hold_h": round(float(np.median(holds)), 1),
        "sharpe": round(float(sharpe), 2),
        "dd_rub": round(dd, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Donchian Channel Breakout
# ─────────────────────────────────────────────────────────────────────────────
def backtest_donchian(df: pd.DataFrame, *, base: str, n: int, rpp: float, lot_size: int) -> dict:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    # Channel: rolling N-bar high/low EXCLUDING current bar
    hh = pd.Series(high).rolling(n).max().shift(1).values
    ll = pd.Series(low).rolling(n).min().shift(1).values

    pos = 0
    entry_p = 0.0
    entry_idx = None
    pnls = []
    holds = []
    for t in range(n + 5, len(df)):
        if np.isnan(hh[t]) or np.isnan(ll[t]):
            continue
        c = close[t]
        if pos == 0:
            if c > hh[t]:
                pos, entry_p, entry_idx = +1, c, t
            elif c < ll[t]:
                pos, entry_p, entry_idx = -1, c, t
            continue
        exit_p = None
        if pos == +1 and c < ll[t]:
            exit_p = c
        elif pos == -1 and c > hh[t]:
            exit_p = c
        if exit_p is None:
            continue
        net, _, _, _ = comm.round_trip_pnl(
            direction="buy" if pos > 0 else "sell",
            entry_price=entry_p,
            exit_price=exit_p,
            lots=1,
            lot_size=lot_size,
            rub_per_point=rpp,
            instrument_kind="future",
            base_ticker=base,
        )
        pnls.append(net)
        holds.append(t - entry_idx)
        pos = 0
        entry_idx = None

    if not pnls:
        return {"n_trades": 0}
    arr = np.array(pnls)
    sharpe = arr.mean() / arr.std() * math.sqrt(len(arr)) if arr.std() > 0 else 0.0
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = float((cum - peak).min())
    return {
        "n_trades": len(arr),
        "win_rate": round(float((arr > 0).mean()), 3),
        "total_rub": round(float(arr.sum()), 2),
        "avg_rub": round(float(arr.mean()), 3),
        "median_hold_h": round(float(np.median(holds)), 1),
        "sharpe": round(float(sharpe), 2),
        "dd_rub": round(dd, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    settings = Settings()
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="trend-test",
    )
    await broker.connect()
    logger.info(f"Fetching {days}d × hourly for {BASES}")

    contracts = {}
    now = datetime.now(timezone.utc)
    for base in BASES:
        f = await _resolve_front_month(broker, base)
        if f is None:
            continue
        try:
            candles = await broker.get_candles(
                f.figi,
                now - timedelta(days=days),
                now,
                interval=CandleInterval.CANDLE_INTERVAL_HOUR,
            )
            df = _candles_to_df(candles)
            if len(df) < 200:
                continue
            meta = broker.extract_futures_metadata(f)
            rpp = float(meta.get("rub_per_point") or 1.0)
            ls = int(getattr(f, "lot", 1) or 1)
            contracts[base] = {
                "df": df,
                "ticker": f.ticker,
                "rpp": rpp,
                "lot": ls,
                "last": float(df["close"].iloc[-1]),
            }
            logger.info(f"  {base} ({f.ticker}): {len(df)} bars, last={df['close'].iloc[-1]:.4f}")
        except Exception as e:
            logger.warning(f"  {base}: {e}")
    await broker.disconnect()
    if not contracts:
        return

    # ── Bollinger grid ──
    N_LIST = [20, 30, 50, 80]
    K_LIST = [1.5, 2.0, 2.5]
    print()
    print("=" * 130)
    print(f"BOLLINGER BREAKOUT — {days}d × hourly × {len(contracts)} contracts")
    print(f"Pattern: long on close > MA(N) + k×σ, exit on close < MA(N) − k×σ.  NO stop-loss.")
    print("=" * 130)
    boll_summary = []
    for base, info in contracts.items():
        print(f"\n── {base} ({info['ticker']}) ──")
        print(f"  {'N':>4} | " + " | ".join(f"k={k}".rjust(20) for k in K_LIST))
        best_for_base = None
        for n in N_LIST:
            row = []
            for k in K_LIST:
                r = backtest_bollinger(
                    info["df"], base=base, n=n, k=k, rpp=info["rpp"], lot_size=info["lot"]
                )
                if r["n_trades"] == 0:
                    row.append("        no trades")
                else:
                    row.append(
                        f"{r['total_rub']:+6.1f}₽ n={r['n_trades']} wr={r['win_rate']*100:.0f}%"
                    )
                    if r["n_trades"] >= 3:
                        c = {"base": base, "n": n, "k": k, **r}
                        if best_for_base is None or c["total_rub"] > best_for_base["total_rub"]:
                            best_for_base = c
            print(f"  {n:>4} | " + " | ".join(s.rjust(20) for s in row))
        if best_for_base:
            boll_summary.append(best_for_base)

    # ── Donchian grid ──
    print()
    print("=" * 130)
    print(f"DONCHIAN BREAKOUT — {days}d × hourly")
    print(f"Pattern: long on close > prior-N high, exit on close < prior-N low.  NO stop-loss.")
    print("=" * 130)
    donch_summary = []
    DONCH_N = [10, 20, 30, 50, 80, 120]
    for base, info in contracts.items():
        print(f"\n── {base} ({info['ticker']}) ──")
        print(
            f"  {'N':>4} | {'n_trades':>8} | {'win%':>6} | {'NET ₽':>10} | "
            f"{'avg ₽':>8} | {'medHold':>8} | {'sharpe':>7} | {'DD ₽':>8}"
        )
        for n in DONCH_N:
            r = backtest_donchian(info["df"], base=base, n=n, rpp=info["rpp"], lot_size=info["lot"])
            if r["n_trades"] == 0:
                print(f"  {n:>4} |    (no trades)")
                continue
            print(
                f"  {n:>4} | {r['n_trades']:>8} | {r['win_rate']*100:>5.1f}% | "
                f"{r['total_rub']:>+10.2f} | {r['avg_rub']:>+8.3f} | "
                f"{r['median_hold_h']:>7.0f}h | {r['sharpe']:>+7.2f} | "
                f"{r['dd_rub']:>+8.2f}"
            )
            if r["n_trades"] >= 3:
                c = {"base": base, "n": n, **r}
                donch_summary.append(c)

    # ── Top cells aggregate ──
    print()
    print("=" * 130)
    print("BEST BOLLINGER (per contract)")
    print("=" * 130)
    print(
        f"  {'base':<5} {'N':>4} {'k':>4} {'n':>4} {'win%':>6} "
        f"{'NET ₽':>10} {'sharpe':>7} {'medHold':>8} {'DD ₽':>8}"
    )
    boll_summary.sort(key=lambda c: -c["total_rub"])
    for c in boll_summary:
        print(
            f"  {c['base']:<5} {c['n']:>4} {c['k']:>4} {c['n_trades']:>4} "
            f"{c['win_rate']*100:>5.1f}% {c['total_rub']:>+10.2f} "
            f"{c['sharpe']:>+7.2f} {c['median_hold_h']:>7.0f}h {c['dd_rub']:>+8.2f}"
        )

    print()
    print("=" * 130)
    print("BEST DONCHIAN (per contract)")
    print("=" * 130)
    print(
        f"  {'base':<5} {'N':>4} {'n':>4} {'win%':>6} "
        f"{'NET ₽':>10} {'sharpe':>7} {'medHold':>8} {'DD ₽':>8}"
    )
    best_donch_by_base = {}
    for c in donch_summary:
        if (
            c["base"] not in best_donch_by_base
            or c["total_rub"] > best_donch_by_base[c["base"]]["total_rub"]
        ):
            best_donch_by_base[c["base"]] = c
    for base, c in sorted(best_donch_by_base.items(), key=lambda x: -x[1]["total_rub"]):
        print(
            f"  {c['base']:<5} {c['n']:>4} {c['n_trades']:>4} "
            f"{c['win_rate']*100:>5.1f}% {c['total_rub']:>+10.2f} "
            f"{c['sharpe']:>+7.2f} {c['median_hold_h']:>7.0f}h {c['dd_rub']:>+8.2f}"
        )

    print()
    print("Reading:")
    print("  • Positive NET ₽ → strategy works after Tinkoff Trader 0.04% commission")
    print("  • Sharpe > 0.5 → decent edge worth deploying")
    print("  • medHold in hours — these are SWING trades (days), not scalp")


if __name__ == "__main__":
    asyncio.run(main())
