"""Expand pairs universe: scan cointegration on a wider base set.

Combines original pairs bases + trend WF survivors → ~25 bases =
~300 candidate pairs.  Tests each via Engle-Granger ADF, keeps
adf_p < 0.10 AND meaningful trade activity in spread mean-reversion
backtest (n_trades ≥ 5, total_pnl > 0 after commission).

Output: ranked list of (a, b, β, params) ready for paste into
`futbot/pairs/config.py::PAIRS_LIST`.

Usage:  python -m futbot.scripts.pairs_universe_expand
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("tinkoff").setLevel(logging.WARNING)
logger = logging.getLogger("pairs_exp")


# Original + trend WF survivors (skip perps — different mechanics)
BASES = [
    "BR",
    "GZ",
    "SR",
    "LK",
    "MX",
    "Si",  # original
    "NG",
    "GD",
    "RT",
    "USDRU",
    "GLDRU",
    "EURRU",
    "CR",  # commodities + currencies
    "PX",
    "YD",
    "VB",
    "TT",
    "RN",
    "GK",
    "MM",  # stocks + index
    "PT",
    "LT",
    "S1",
    "MV",
    "SV",  # smaller liquid stocks
]
ADF_P_MAX = 0.10
DAYS = 180
MIN_TRADES = 5
COMMISSION_RT = 0.0016  # 0.04% × 2 sides × 2 legs
Z_ENTRY = 2.0
Z_STOP = 4.0
ROLLING_Z_WINDOW = 240


async def _resolve_front_month(broker, base: str):
    futs = await broker.get_all_futures()
    cands = []
    now = datetime.now(timezone.utc)
    for f in futs:
        t = getattr(f, "ticker", "") or ""
        is_match = t == base or (t.startswith(base) and len(t) == len(base) + 2)
        if not is_match:
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


def _candles_to_df(candles) -> pd.DataFrame:
    rows = [{"time": c.time, "close": float(quotation_to_decimal(c.close))} for c in candles]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


async def _fetch_chunked(broker, figi: str, days: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    chunks = []
    end = now
    days_left = days
    while days_left > 0:
        chunk_days = min(days_left, 89)
        start = end - timedelta(days=chunk_days)
        try:
            candles = await broker.get_candles(
                figi,
                start,
                end,
                interval=CandleInterval.CANDLE_INTERVAL_HOUR,
            )
            df = _candles_to_df(candles)
            if not df.empty:
                chunks.append(df)
        except Exception:
            pass
        end = start
        days_left -= chunk_days
    if not chunks:
        return pd.DataFrame()
    return (
        pd.concat(chunks)
        .drop_duplicates("time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )


def _backtest_pair(*, prices: pd.DataFrame, a: str, b: str, beta: float) -> dict:
    y = prices[a].values
    x = prices[b].values
    spread = y - beta * x
    n = len(spread)
    if n < ROLLING_Z_WINDOW + 50:
        return {"n_trades": 0}

    z = np.zeros(n)
    z[:ROLLING_Z_WINDOW] = np.nan
    for t in range(ROLLING_Z_WINDOW, n):
        w = spread[t - ROLLING_Z_WINDOW : t]
        m = w.mean()
        s = w.std()
        z[t] = (spread[t] - m) / s if s > 0 else 0.0

    pos = 0
    entry_idx = None
    pnls = []
    holds = []
    for t in range(ROLLING_Z_WINDOW, n):
        if np.isnan(z[t]):
            continue
        if pos == 0:
            if z[t] > Z_ENTRY:
                pos, entry_idx = -1, t
            elif z[t] < -Z_ENTRY:
                pos, entry_idx = +1, t
            continue
        # check exits
        crossed_zero = (pos == +1 and z[t] >= 0) or (pos == -1 and z[t] <= 0)
        stopped = abs(z[t]) >= Z_STOP
        timed_out = (t - entry_idx) >= 48  # 48h cap (matches live)
        if not (crossed_zero or stopped or timed_out):
            continue
        sp_e = y[entry_idx] - beta * x[entry_idx]
        sp_x = y[t] - beta * x[t]
        combined = abs(y[entry_idx]) + abs(beta) * abs(x[entry_idx])
        gross = pos * (sp_x - sp_e) / combined if combined > 0 else 0
        pnls.append(gross - COMMISSION_RT)
        holds.append(t - entry_idx)
        pos = 0
    if not pnls:
        return {"n_trades": 0}
    arr = np.array(pnls)
    sharpe = arr.mean() / arr.std() * math.sqrt(len(arr)) if arr.std() > 0 else 0
    return {
        "n_trades": len(arr),
        "win_rate": float((arr > 0).mean()),
        "total_pct": float(arr.sum() * 100),
        "avg_pct": float(arr.mean() * 100),
        "sharpe": float(sharpe),
        "median_hold_h": float(np.median(holds)),
    }


async def main():
    settings = Settings()
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="pairs-exp",
    )
    await broker.connect()
    logger.info(f"Fetching {DAYS}d hourly for {len(BASES)} bases")

    series_list = []
    for base in BASES:
        f = await _resolve_front_month(broker, base)
        if f is None:
            logger.warning(f"  {base}: no contract")
            continue
        df = await _fetch_chunked(broker, f.figi, DAYS)
        if len(df) < 500:
            logger.warning(f"  {base}: only {len(df)} bars, skipping")
            continue
        s = df.set_index("time")["close"].rename(base)
        series_list.append(s)
        logger.info(f"  {base} ({f.ticker}): {len(df)} bars")
    await broker.disconnect()

    if len(series_list) < 2:
        logger.error("Not enough data")
        return

    prices = pd.concat(series_list, axis=1, join="inner").dropna()
    logger.info(f"Aligned bars: {len(prices)}  span: {prices.index.min()}..{prices.index.max()}")
    bases = list(prices.columns)
    logger.info(f"Active bases: {bases}")

    # ── Cointegration scan ────────────────────────────────────────
    from statsmodels.tsa.stattools import adfuller

    candidates = []
    pair_count = 0
    for i, a in enumerate(bases):
        for b in bases[i + 1 :]:
            pair_count += 1
            y = prices[a].values
            x = prices[b].values
            beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
            alpha = y.mean() - beta * x.mean()
            resid = y - (beta * x + alpha)
            try:
                pval = float(adfuller(resid, maxlag=5, autolag=None)[1])
            except Exception:
                continue
            if pval > ADF_P_MAX:
                continue
            candidates.append({"a": a, "b": b, "beta": beta, "adf_p": pval})

    candidates.sort(key=lambda c: c["adf_p"])
    logger.info(f"Tested {pair_count} pairs, {len(candidates)} cointegrated (adf_p ≤ {ADF_P_MAX})")

    # ── Backtest each cointegrated pair ───────────────────────────
    print()
    print("=" * 130)
    print(
        f"COINTEGRATED PAIRS BACKTEST (z_entry={Z_ENTRY}, hold=48h, comm RT={COMMISSION_RT*100:.2f}%)"
    )
    print("=" * 130)
    profitable = []
    for c in candidates:
        r = _backtest_pair(prices=prices, a=c["a"], b=c["b"], beta=c["beta"])
        if r.get("n_trades", 0) < MIN_TRADES:
            continue
        if r["total_pct"] <= 0:
            continue
        c.update(r)
        profitable.append(c)

    profitable.sort(key=lambda c: -c["sharpe"])
    print(
        f"\n{'pair':<14} {'beta':>9} {'adf_p':>7} {'n':>4} {'win%':>6} {'avg%':>8} {'total%':>8} {'sharpe':>7} {'medHold':>8}"
    )
    for c in profitable:
        pair = f"{c['a']}-{c['b']}"
        print(
            f"{pair:<14} {c['beta']:>+8.4f} {c['adf_p']:>7.4f} {c['n_trades']:>4} "
            f"{c['win_rate']*100:>5.1f}% {c['avg_pct']:>+8.3f} {c['total_pct']:>+8.3f} "
            f"{c['sharpe']:>+7.2f} {c['median_hold_h']:>6.0f}h"
        )

    print()
    print("=" * 130)
    print("PROPOSED PAIRS_LIST for futbot/pairs/config.py:")
    print("=" * 130)
    print("    PAIRS_LIST: list = [")
    for c in profitable:
        print(f'        "{c["a"]}-{c["b"]}",')
    print("    ]")


if __name__ == "__main__":
    asyncio.run(main())
