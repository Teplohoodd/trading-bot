"""Walk-forward validation of the Bollinger breakout universe scan.

Problem with `trend_universe_scan.py`:
  We tested 171 contracts × 12 cells = ~2000 parameter combinations.
  Even with NO real edge, statistical noise would produce dozens of
  positive cells (multiple-comparison bias / "selection effect").
  98 positive contracts out of 140 looks impressive but is close to
  what pure noise would give.

Walk-forward separates real edge from overfit:

  1. Split each contract's 180d history into IN-SAMPLE (first 90d) +
     OUT-OF-SAMPLE (last 90d).
  2. Find best (N, k) cell on IN-sample only — that's the "model fit".
  3. Apply those EXACT params to OUT-of-sample data — no re-optimisation.
  4. If OOS still positive → real edge.  If OOS negative or near zero →
     the IS result was overfit (we hill-climbed to noise).

Output: ranked list of contracts where BOTH halves are positive
(genuine edge), with their walk-forward stats.

Usage:
    python -m futbot.scripts.trend_walk_forward [days]   # default 180
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
logger = logging.getLogger("trend_wf")


N_LIST = [20, 30, 50, 80]
K_LIST = [1.5, 2.0, 2.5]
MIN_BARS = 500
MIN_TRADES_IS = 5  # IS sample size threshold to consider a cell
MIN_TRADES_OOS = 3  # OOS sample size threshold to trust the result
DAYS_TO_EXPIRY_MIN = 14


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


async def _resolve_all_front_months(broker) -> list[dict]:
    futs = await broker.get_all_futures()
    by_base: dict[str, list] = {}
    now = datetime.now(timezone.utc)
    for f in futs:
        t = getattr(f, "ticker", "") or ""
        if len(t) < 3:
            continue
        base = t[:-2]
        exp = getattr(f, "expiration_date", None)
        if exp is None:
            continue
        if hasattr(exp, "ToDatetime"):
            exp = exp.ToDatetime()
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        dte = (exp - now).days
        if dte < DAYS_TO_EXPIRY_MIN:
            continue
        by_base.setdefault(base, []).append((f, exp))

    front = []
    for base, lst in by_base.items():
        lst.sort(key=lambda x: x[1])
        f, exp = lst[0]
        meta = broker.extract_futures_metadata(f)
        front.append(
            {
                "base": base,
                "ticker": f.ticker,
                "figi": f.figi,
                "expiration": exp,
                "rpp": float(meta.get("rub_per_point") or 1.0),
                "lot": int(getattr(f, "lot", 1) or 1),
            }
        )
    front.sort(key=lambda x: x["base"])
    return front


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
    pnls = []
    for t in range(n + 5, len(df)):
        if np.isnan(upper[t]) or np.isnan(lower[t]):
            continue
        c = close[t]
        if pos == 0:
            if c > upper[t]:
                pos, entry_p = +1, c
            elif c < lower[t]:
                pos, entry_p = -1, c
            continue
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
        pos = 0
    if not pnls:
        return {"n_trades": 0, "total_rub": 0.0, "sharpe": 0.0}
    arr = np.array(pnls)
    sharpe = arr.mean() / arr.std() * math.sqrt(len(arr)) if arr.std() > 0 else 0.0
    return {
        "n_trades": len(arr),
        "win_rate": round(float((arr > 0).mean()), 3),
        "total_rub": round(float(arr.sum()), 2),
        "avg_rub": round(float(arr.mean()), 3),
        "sharpe": round(float(sharpe), 2),
    }


async def _fetch_candles_chunked(
    broker, figi: str, days: int, interval: CandleInterval, max_chunk_days: int
) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    chunks = []
    end = now
    days_left = days
    while days_left > 0:
        chunk_days = min(days_left, max_chunk_days)
        start = end - timedelta(days=chunk_days)
        try:
            candles = await broker.get_candles(figi, start, end, interval=interval)
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


async def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    settings = Settings()
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="trend-wf",
    )
    await broker.connect()
    logger.info("Resolving front-month contracts…")
    universe = await _resolve_all_front_months(broker)
    logger.info(f"Found {len(universe)} contracts")

    results = []
    for i, c in enumerate(universe, 1):
        try:
            df = await _fetch_candles_chunked(
                broker,
                c["figi"],
                days,
                CandleInterval.CANDLE_INTERVAL_HOUR,
                max_chunk_days=89,
            )
        except Exception:
            continue
        if len(df) < MIN_BARS:
            continue

        # Split 50/50: in-sample is first half, out-of-sample is second half
        midpoint = len(df) // 2
        df_is = df.iloc[:midpoint].reset_index(drop=True)
        df_oos = df.iloc[midpoint:].reset_index(drop=True)

        # Find best (n, k) on IN-sample
        best_is = None
        for n in N_LIST:
            for k in K_LIST:
                r = backtest_bollinger(
                    df_is, base=c["base"], n=n, k=k, rpp=c["rpp"], lot_size=c["lot"]
                )
                if r.get("n_trades", 0) < MIN_TRADES_IS:
                    continue
                if r["total_rub"] <= 0:
                    continue
                if best_is is None or r["total_rub"] > best_is["total_rub"]:
                    best_is = {**r, "n": n, "k": k}

        if best_is is None:
            continue

        # Apply best params to OUT-OF-sample (NO re-optimisation)
        oos = backtest_bollinger(
            df_oos,
            base=c["base"],
            n=best_is["n"],
            k=best_is["k"],
            rpp=c["rpp"],
            lot_size=c["lot"],
        )

        results.append(
            {
                "base": c["base"],
                "ticker": c["ticker"],
                "figi": c["figi"],
                "rpp": c["rpp"],
                "lot": c["lot"],
                "n": best_is["n"],
                "k": best_is["k"],
                "is_trades": best_is["n_trades"],
                "is_winrate": best_is["win_rate"],
                "is_pnl": best_is["total_rub"],
                "is_sharpe": best_is["sharpe"],
                "oos_trades": oos.get("n_trades", 0),
                "oos_winrate": oos.get("win_rate", 0),
                "oos_pnl": oos.get("total_rub", 0),
                "oos_sharpe": oos.get("sharpe", 0),
            }
        )
        if i % 20 == 0:
            logger.info(f"  [{i}/{len(universe)}] processed")

    await broker.disconnect()

    # ── Filter: contracts where BOTH halves are positive ──
    print()
    print("=" * 145)
    print(f"WALK-FORWARD VALIDATION — {days}d (90d IS + 90d OOS)")
    print(f"Goal: filter out overfit cells.  Real edge = positive in BOTH halves.")
    print("=" * 145)

    confirmed = [r for r in results if r["oos_pnl"] > 0 and r["oos_trades"] >= MIN_TRADES_OOS]
    confirmed.sort(key=lambda r: -r["oos_sharpe"])

    print()
    print(f"Contracts where IS edge survived OOS: {len(confirmed)} / {len(results)}")
    print()
    print(
        f"{'rank':>4} {'base':<6} {'ticker':<8} {'N':>3} {'k':>4}   "
        f"{'IS_trd':>6} {'IS_win':>7} {'IS_NET':>10} {'IS_Sh':>6}  "
        f"{'OOS_trd':>7} {'OOS_win':>8} {'OOS_NET':>10} {'OOS_Sh':>7}"
    )
    for i, r in enumerate(confirmed[:30], 1):
        print(
            f"{i:>4} {r['base']:<6} {r['ticker']:<8} {r['n']:>3} {r['k']:>4}   "
            f"{r['is_trades']:>6} {r['is_winrate']*100:>6.1f}% "
            f"{r['is_pnl']:>+10.2f} {r['is_sharpe']:>+6.2f}  "
            f"{r['oos_trades']:>7} {r['oos_winrate']*100:>7.1f}% "
            f"{r['oos_pnl']:>+10.2f} {r['oos_sharpe']:>+7.2f}"
        )

    # Stats
    if confirmed:
        is_total = sum(r["is_pnl"] for r in confirmed)
        oos_total = sum(r["oos_pnl"] for r in confirmed)
        is_avg_sharpe = sum(r["is_sharpe"] for r in confirmed) / len(confirmed)
        oos_avg_sharpe = sum(r["oos_sharpe"] for r in confirmed) / len(confirmed)
        survival_rate = len(confirmed) / len(results) * 100 if results else 0
        print()
        print(f"Summary:")
        print(f"  Survival rate: {len(confirmed)}/{len(results)} = {survival_rate:.0f}%")
        print(f"  Avg IS Sharpe:  {is_avg_sharpe:+.2f}")
        print(
            f"  Avg OOS Sharpe: {oos_avg_sharpe:+.2f}   ← if much lower than IS, overfit suspected"
        )
        print(f"  Total IS NET:   {is_total:+.2f} ₽")
        print(f"  Total OOS NET:  {oos_total:+.2f} ₽")
        print()
        if oos_avg_sharpe < 0.5 * is_avg_sharpe:
            print("  ⚠  OOS Sharpe < half of IS Sharpe — heavy overfit, distrust IS numbers")
        elif oos_avg_sharpe > 0.7 * is_avg_sharpe:
            print("  ✓ OOS Sharpe close to IS — edge appears genuine in this subset")
        else:
            print("  ~ Moderate overfit — expect live results closer to OOS than IS")

    # Negative half — for reference
    failed = [r for r in results if r["oos_pnl"] <= 0 or r["oos_trades"] < MIN_TRADES_OOS]
    if failed:
        print()
        print(f"Failed OOS (overfit candidates): {len(failed)}")
        # Stats on failed ones
        all_oos = sum(r["oos_pnl"] for r in failed)
        print(f"  Total OOS NET on failed: {all_oos:+.2f} ₽")


if __name__ == "__main__":
    asyncio.run(main())
