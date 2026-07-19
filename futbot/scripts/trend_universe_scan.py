"""Trend-breakout universe scan + timeframe analysis.

Step 1: scan EVERY FORTS futures base with active front-month contract,
        run Bollinger breakout backtest on 1h bars over 180d.
        Filter to those with positive NET after commission AND ≥5 trades.

Step 2: for top-10 contracts from step 1, compare their behaviour on
        DIFFERENT timeframes (1h / 30m / 15m) to answer the user's
        question: does shorter timeframe help (more trades) or hurt
        (worse commission ratio)?

Goal: build a SHORT-LIST of contracts × timeframes × params where
trend-following actually has measurable edge, then deploy bot on those.

Usage:
    python -m futbot.scripts.trend_universe_scan
    python -m futbot.scripts.trend_universe_scan 180        # custom days
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
from t_tech.invest import CandleInterval
from t_tech.invest.utils import quotation_to_decimal

from futbot.utils import commissions as comm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("t_tech").setLevel(logging.WARNING)
logger = logging.getLogger("uni_scan")


# Bollinger parameter grid — same as before
N_LIST = [20, 30, 50, 80]
K_LIST = [1.5, 2.0, 2.5]
MIN_BARS = 500  # need this many candles for grid search
MIN_TRADES = 5  # minimum sample size for confidence
DAYS_TO_EXPIRY_MIN = 14  # skip about-to-roll contracts


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
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
    """Return one front-month per base, with at least DAYS_TO_EXPIRY_MIN days."""
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
                "instrument": f,
            }
        )
    front.sort(key=lambda x: x["base"])
    return front


def backtest_bollinger(
    df: pd.DataFrame, *, base: str, n: int, k: float, rpp: float, lot_size: int
) -> dict:
    """Same algo as trend_breakout_test.py.  Pure Bollinger band-flip strategy."""
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
        "median_hold_bars": round(float(np.median(holds)), 1),
        "sharpe": round(float(sharpe), 2),
        "dd_rub": round(dd, 2),
    }


async def _fetch_candles_chunked(
    broker, figi: str, days: int, interval: CandleInterval, max_chunk_days: int = 90
) -> pd.DataFrame:
    """Fetch candles with chunking to bypass per-request limits."""
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
        except Exception as e:
            logger.debug(f"  fetch chunk {start.date()}..{end.date()} failed: {e}")
        end = start
        days_left -= chunk_days
    if not chunks:
        return pd.DataFrame()
    combined = (
        pd.concat(chunks)
        .drop_duplicates("time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    settings = Settings()
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="uni-scan",
    )
    await broker.connect()
    logger.info("Resolving all FORTS bases with active front-month…")
    universe = await _resolve_all_front_months(broker)
    logger.info(f"Found {len(universe)} active front-month contracts")

    # ── Step 1: scan all on 1h ────────────────────────────────────────────
    print()
    print("=" * 140)
    print(
        f"STEP 1 — UNIVERSE SCAN, 1h × {days}d × {len(universe)} contracts × "
        f"{len(N_LIST)*len(K_LIST)} param cells"
    )
    print("=" * 140)

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
        except Exception as e:
            logger.debug(f"  {c['base']:>6}: fetch failed ({e})")
            continue
        if len(df) < MIN_BARS:
            logger.debug(f"  {c['base']:>6} ({c['ticker']}): only {len(df)} bars, skipping")
            continue
        best = None
        for n in N_LIST:
            for k in K_LIST:
                r = backtest_bollinger(
                    df, base=c["base"], n=n, k=k, rpp=c["rpp"], lot_size=c["lot"]
                )
                if r.get("n_trades", 0) < MIN_TRADES:
                    continue
                if best is None or r["total_rub"] > best["total_rub"]:
                    best = {**r, "n": n, "k": k}
        if best is None:
            logger.debug(f"  {c['base']:>6}: no cells with ≥{MIN_TRADES} trades")
            continue
        results.append(
            {
                "base": c["base"],
                "ticker": c["ticker"],
                "figi": c["figi"],
                "rpp": c["rpp"],
                "lot": c["lot"],
                **best,
            }
        )
        if i % 10 == 0:
            logger.info(
                f"  [{i}/{len(universe)}] processed; positive so far: "
                f"{sum(1 for r in results if r['total_rub'] > 0)}"
            )

    # Filter to positive cells, rank by Sharpe
    positive = [r for r in results if r["total_rub"] > 0 and r["n_trades"] >= MIN_TRADES]
    positive.sort(key=lambda r: -r["sharpe"])

    print()
    print(
        f"Scanned {len(universe)} contracts.  Got results for {len(results)}.  "
        f"Positive NET after commission: {len(positive)}."
    )
    print()
    print(
        f"{'rank':>4} {'base':<6} {'ticker':<8} {'N':>3} {'k':>4} {'n':>4} "
        f"{'win%':>6} {'NET ₽':>11} {'avg ₽':>9} {'sharpe':>7} "
        f"{'medHold':>8} {'DD ₽':>10}"
    )
    for i, r in enumerate(positive[:30], 1):
        print(
            f"{i:>4} {r['base']:<6} {r['ticker']:<8} "
            f"{r['n']:>3} {r['k']:>4} {r['n_trades']:>4} "
            f"{r['win_rate']*100:>5.1f}% "
            f"{r['total_rub']:>+11.2f} {r['avg_rub']:>+9.3f} "
            f"{r['sharpe']:>+7.2f} {r['median_hold_bars']:>6.0f}h "
            f"{r['dd_rub']:>+10.2f}"
        )

    # ── Step 2: timeframe comparison on top-5 ───────────────────────────
    top5 = positive[:5]
    if not top5:
        print("\nNo positive contracts — nothing to compare across timeframes.")
        return

    print()
    print("=" * 140)
    print(f"STEP 2 — TIMEFRAME COMPARISON on top-{len(top5)} contracts × {days}d")
    print(f"Same Bollinger logic, lookback N scaled per timeframe.")
    print("=" * 140)
    print()
    print(
        f"{'base':<6} {'ticker':<8} {'TF':>4} {'N':>3} {'k':>4} {'n':>5} "
        f"{'win%':>6} {'NET ₽':>11} {'sharpe':>7} {'med_hold_bars':>14} "
        f"{'trades/mo':>10}"
    )

    # Timeframes to compare (with smaller max-chunk because limits differ)
    tf_specs = [
        ("1h", CandleInterval.CANDLE_INTERVAL_HOUR, 89),
        ("30m", CandleInterval.CANDLE_INTERVAL_30_MIN, 20),
        ("15m", CandleInterval.CANDLE_INTERVAL_15_MIN, 20),
    ]

    for r in top5:
        for tf_label, tf, chunk_days in tf_specs:
            try:
                df_tf = await _fetch_candles_chunked(
                    broker,
                    r["figi"],
                    days,
                    tf,
                    max_chunk_days=chunk_days,
                )
            except Exception as e:
                logger.warning(f"  {r['base']} {tf_label}: {e}")
                continue
            if len(df_tf) < MIN_BARS:
                print(
                    f"{r['base']:<6} {r['ticker']:<8} {tf_label:>4}  "
                    f"only {len(df_tf)} bars (need ≥{MIN_BARS})"
                )
                continue

            # Try the same N×k grid; on shorter TF the optimal N may differ
            best_cell = None
            for n in N_LIST:
                for k in K_LIST:
                    res = backtest_bollinger(
                        df_tf, base=r["base"], n=n, k=k, rpp=r["rpp"], lot_size=r["lot"]
                    )
                    if res.get("n_trades", 0) < MIN_TRADES:
                        continue
                    if best_cell is None or res["total_rub"] > best_cell["total_rub"]:
                        best_cell = {**res, "n": n, "k": k}
            if best_cell is None:
                print(
                    f"{r['base']:<6} {r['ticker']:<8} {tf_label:>4}  "
                    f"no cells with ≥{MIN_TRADES} trades"
                )
                continue
            trades_per_month = best_cell["n_trades"] / (days / 30.0)
            print(
                f"{r['base']:<6} {r['ticker']:<8} {tf_label:>4} "
                f"{best_cell['n']:>3} {best_cell['k']:>4} {best_cell['n_trades']:>5} "
                f"{best_cell['win_rate']*100:>5.1f}% "
                f"{best_cell['total_rub']:>+11.2f} "
                f"{best_cell['sharpe']:>+7.2f} "
                f"{best_cell['median_hold_bars']:>12.0f}b "
                f"{trades_per_month:>9.1f}"
            )

    print()
    print("Reading:")
    print("  • Higher trades/month = more activity (the user's goal)")
    print("  • Sharpe ↓ on shorter TF = commission eating more")
    print("  • Sharpe ↑ on shorter TF = signal works at higher frequency")
    print("  • Pick a (contract, TF, N, k) cell with: trades/month ≥ 4 AND Sharpe ≥ 0.5")

    await broker.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
