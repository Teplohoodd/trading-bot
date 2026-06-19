"""Scalp horizon + parameter grid search on historical 1-min candles.

⚠️ HONEST CAVEAT:
The live scalp uses order-book imbalance + tick-level trade-flow imbalance
(TFI), neither of which is available in historical bar data.  This backtest
runs ONLY the INDICATOR part of the scalp signal (RSI / EMA / MACD / ATR)
against 1-min candles.

What this can tell us:
  * Which contracts have favourable price-action structure for short holds
  * What holding period suits the INDICATOR signal
  * Where commission economics work (signal magnitude × ATR vs round-trip cost)

What it CAN'T tell us:
  * Whether live book_imb / TFI signals work at those horizons (they're
    microstructure, must be tested live in paper mode)
  * Exact P&L the live bot would have made

For pair-trading we had a precise model.  For scalp this is closer to
"directional indicator scan" — useful but not predictive of live P&L.

────────────────────────────────────────────────────────────────────────────
Indicator signal computed per bar:
    rsi_score = (buy_thr - rsi) / 20 if rsi < buy_thr
                (rsi - sell_thr) / 20 × -1 if rsi > sell_thr   else 0
    ema_score = clip(ema_fast - ema_slow / close × 200, -1, +1)
    macd_score = clip(macd_hist / close × 400, -1, +1)
    score = (rsi_score + ema_score + macd_score) / 3

Trade rules:
    Enter when |score| ≥ score_min  AND ATR_5m × rpp × lot > 2 × commission
    Exit on:
        TP   — price moves +TP_atr × ATR_5m   (favourable)
        SL   — price moves -SL_atr × ATR_5m   (unfavourable)
        Time — held > max_hold_min minutes
        Flip — score sign reverses to >EXIT_min after MIN_AGE_SEC and in profit
              (mirrors the live scalp gate)

Grid:
    max_hold_min ∈ {3, 5, 10, 15, 30, 60, 120}
    score_min    ∈ {0.30, 0.45, 0.60}
    TP_atr / SL_atr fixed at 2.0 / 1.2 (current live config)

Usage:
    python -m futbot.scripts.scalp_horizon_search           # default 60d
    python -m futbot.scripts.scalp_horizon_search 30        # 30 days
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
logger = logging.getLogger("scalp_horizon")


# Universe — all viable scalp candidates.  Si excluded (market orders
# disabled).  MX/SR included so we can verify they're as bad as predicted.
BASES = ["BR", "GZ", "LK", "NG", "GD", "MX", "SR"]

# Grid
HOLD_MIN = [3, 5, 10, 15, 30, 60, 120]
SCORE_MIN = [0.30, 0.45, 0.60]
TP_ATR_5M = 2.0
SL_ATR_5M = 1.2
RSI_BUY = 40.0
RSI_SELL = 60.0
EMA_FAST_N = 9
EMA_SLOW_N = 21
RSI_N = 7


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
# Indicators (vectorised — much faster than per-bar loops)
# ─────────────────────────────────────────────────────────────────────────────
def _wilder_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = _wilder_ema(up, n)
    avg_dn = _wilder_ema(down, n).replace(0, np.nan)
    rs = avg_up / avg_dn
    return 100 - 100 / (1 + rs)


def _atr_5m_proxy(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Approximate 5m ATR from 1m bars: roll 5-bar high/low, then ATR on that."""
    high5 = df["high"].rolling(5).max()
    low5 = df["low"].rolling(5).min()
    close5 = df["close"]
    tr = pd.concat(
        [
            high5 - low5,
            (high5 - close5.shift()).abs(),
            (low5 - close5.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _wilder_ema(tr, n)


def compute_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'score' and 'atr_5m' columns to df.

    score = (rsi_score + ema_score + macd_score) / 3, in [-1, +1].
    """
    close = df["close"]
    rsi = _rsi(close, RSI_N)
    ema_f = close.ewm(span=EMA_FAST_N, adjust=False).mean()
    ema_s = close.ewm(span=EMA_SLOW_N, adjust=False).mean()
    # Short-period MACD (5, 13, 5)
    macd_f = close.ewm(span=5, adjust=False).mean()
    macd_s = close.ewm(span=13, adjust=False).mean()
    macd_line = macd_f - macd_s
    macd_sig = macd_line.ewm(span=5, adjust=False).mean()
    macd_hist = macd_line - macd_sig

    rsi_score = pd.Series(0.0, index=df.index)
    rsi_score = rsi_score.where(rsi.isna() | (rsi >= RSI_BUY), (RSI_BUY - rsi) / 20.0)
    rsi_score = rsi_score.where(rsi.isna() | (rsi <= RSI_SELL), -(rsi - RSI_SELL) / 20.0)
    rsi_score = rsi_score.clip(-1.0, 1.0)
    ema_score = ((ema_f - ema_s) / close.replace(0, np.nan) * 200).clip(-1, 1)
    macd_score = (macd_hist / close.replace(0, np.nan) * 400).clip(-1, 1)
    score = (rsi_score + ema_score + macd_score) / 3.0
    df = df.copy()
    df["score"] = score
    df["atr_5m"] = _atr_5m_proxy(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Backtest one (contract, hold_min, score_min) cell
# ─────────────────────────────────────────────────────────────────────────────
def backtest_cell(
    df: pd.DataFrame,
    *,
    base: str,
    hold_min: int,
    score_min: float,
    rpp: float,
    lot_size: int,
    tp_atr: float = TP_ATR_5M,
    sl_atr: float = SL_ATR_5M,
) -> dict:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    score = df["score"].values
    atr = df["atr_5m"].values
    n = len(df)

    pos = 0
    entry_idx = None
    entry_p = 0.0
    stop = 0.0
    tp = 0.0
    pnls_rub = []
    pnls_pct = []
    holds = []
    exit_reasons = {"tp": 0, "sl": 0, "time": 0}

    for t in range(50, n):  # skip warm-up
        s = score[t]
        c = close[t]
        a = atr[t]
        if np.isnan(s) or np.isnan(a) or a <= 0:
            continue

        if pos == 0:
            if abs(s) < score_min:
                continue
            # Commission gate: TP profit must beat 2× round-trip
            tp_profit_rub = abs(tp_atr * a) * rpp * lot_size
            rt_comm = comm.estimated_round_trip_cost(
                price=c,
                lots=1,
                lot_size=lot_size,
                rub_per_point=rpp,
                instrument_kind="future",
                base_ticker=base,
            )
            if tp_profit_rub < 2 * rt_comm:
                continue
            pos = +1 if s > 0 else -1
            entry_idx = t
            entry_p = c
            if pos == +1:
                stop = c - sl_atr * a
                tp = c + tp_atr * a
            else:
                stop = c + sl_atr * a
                tp = c - tp_atr * a
            continue

        # Holding — check exits (TP/SL using bar high/low intra-bar)
        held_min = t - entry_idx
        exit_p = None
        exit_reason = None
        if pos == +1:
            if low[t] <= stop:
                exit_p = stop
                exit_reason = "sl"
            elif high[t] >= tp:
                exit_p = tp
                exit_reason = "tp"
        else:
            if high[t] >= stop:
                exit_p = stop
                exit_reason = "sl"
            elif low[t] <= tp:
                exit_p = tp
                exit_reason = "tp"
        if exit_p is None and held_min >= hold_min:
            exit_p = c
            exit_reason = "time"

        if exit_p is None:
            continue

        # P&L net of commission
        net_rub, pct, _, _ = comm.round_trip_pnl(
            direction="buy" if pos > 0 else "sell",
            entry_price=entry_p,
            exit_price=exit_p,
            lots=1,
            lot_size=lot_size,
            rub_per_point=rpp,
            instrument_kind="future",
            base_ticker=base,
        )
        pnls_rub.append(net_rub)
        pnls_pct.append(pct)
        holds.append(held_min)
        exit_reasons[exit_reason] += 1
        pos = 0
        entry_idx = None

    if not pnls_rub:
        return {"n_trades": 0}

    arr = np.array(pnls_rub)
    arr_pct = np.array(pnls_pct)
    sharpe = arr_pct.mean() / arr_pct.std() * math.sqrt(len(arr_pct)) if arr_pct.std() > 0 else 0.0
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = float((cum - peak).min())
    return {
        "n_trades": len(arr),
        "win_rate": round(float((arr > 0).mean()), 3),
        "total_rub": round(float(arr.sum()), 2),
        "avg_rub": round(float(arr.mean()), 3),
        "avg_pct": round(float(arr_pct.mean()), 4),
        "best_rub": round(float(arr.max()), 2),
        "worst_rub": round(float(arr.min()), 2),
        "median_hold_min": round(float(np.median(holds)), 1),
        "sharpe": round(float(sharpe), 2),
        "dd_rub": round(dd, 2),
        "exits": exit_reasons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    settings = Settings()
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="scalp-horizon",
    )
    await broker.connect()
    logger.info(f"Fetching {days}d of 1-min candles for {BASES}")

    contracts = {}
    now = datetime.now(timezone.utc)
    for base in BASES:
        f = await _resolve_front_month(broker, base)
        if f is None:
            logger.warning(f"  {base}: no front-month contract")
            continue
        try:
            candles = await broker.get_candles(
                f.figi,
                now - timedelta(days=days),
                now,
                interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
            )
            df = _candles_to_df(candles)
            if len(df) < 500:
                logger.warning(f"  {base}: only {len(df)} bars — skip")
                continue
            # Extract metadata for sizing/commission calc
            meta = broker.extract_futures_metadata(f)
            rpp = float(meta.get("rub_per_point") or 1.0)
            lot_size = int(getattr(f, "lot", 1) or 1)
            df = compute_signal(df)
            contracts[base] = {
                "df": df,
                "ticker": f.ticker,
                "rpp": rpp,
                "lot": lot_size,
                "last_price": float(df["close"].iloc[-1]),
            }
            logger.info(
                f"  {base} ({f.ticker}): {len(df)} bars  rpp={rpp:.2f}  lot={lot_size}  "
                f"last={df['close'].iloc[-1]:.4f}"
            )
        except Exception as e:
            logger.warning(f"  {base}: fetch failed ({e})")
    await broker.disconnect()
    if not contracts:
        logger.error("No data")
        return

    print()
    print("=" * 130)
    print(f"SCALP HORIZON SEARCH — {days}d × 1-min × {len(contracts)} contracts")
    print(f"TP/SL = {TP_ATR_5M}/{SL_ATR_5M} × ATR_5m, commission = 0.04 % per side")
    print("=" * 130)

    summary = []
    for base, info in contracts.items():
        print(f"\n── {base} ({info['ticker']}, last {info['last_price']:.4f}) ──")
        print(f"  {'score_min':>10} |   " + "   |   ".join(f"{h:>3}m" for h in HOLD_MIN))
        cells_for_base = []
        for sm in SCORE_MIN:
            row = []
            for hm in HOLD_MIN:
                r = backtest_cell(
                    info["df"],
                    base=base,
                    hold_min=hm,
                    score_min=sm,
                    rpp=info["rpp"],
                    lot_size=info["lot"],
                )
                cells_for_base.append({"score_min": sm, "hold_min": hm, **r})
                if r.get("n_trades", 0) == 0:
                    row.append("  no trades")
                else:
                    row.append(f"{r['total_rub']:>+7.1f}₽/{r['n_trades']:>3}")
            print(f"  {sm:>10.2f} | " + " | ".join(row))

        # Pick best by sharpe with min 5 trades
        scored = [c for c in cells_for_base if c.get("n_trades", 0) >= 5]
        if not scored:
            scored = [c for c in cells_for_base if c.get("n_trades", 0) > 0]
        if scored:
            best_sharpe = max(scored, key=lambda c: c.get("sharpe", -999))
            best_total = max(scored, key=lambda c: c.get("total_rub", -999_999))
            print(
                f"  → Best Sharpe:  score≥{best_sharpe['score_min']}, "
                f"hold={best_sharpe['hold_min']}m, n={best_sharpe['n_trades']}, "
                f"win={best_sharpe['win_rate']*100:.0f}%, "
                f"total={best_sharpe['total_rub']:+.1f}₽, "
                f"sharpe={best_sharpe['sharpe']:+.2f}, "
                f"medHold={best_sharpe['median_hold_min']:.0f}m, "
                f"exits={best_sharpe['exits']}"
            )
            print(
                f"  → Best Total:   score≥{best_total['score_min']}, "
                f"hold={best_total['hold_min']}m, n={best_total['n_trades']}, "
                f"win={best_total['win_rate']*100:.0f}%, "
                f"total={best_total['total_rub']:+.1f}₽"
            )
            summary.append({"base": base, "sharpe_best": best_sharpe, "total_best": best_total})

    # Top-line summary
    print()
    print("=" * 130)
    print("BEST PARAMETERS PER CONTRACT (sorted by total ₽)")
    print("=" * 130)
    summary.sort(key=lambda s: -s["sharpe_best"].get("total_rub", -1e9))
    print(
        f"{'base':<5} {'score_min':>10} {'hold_min':>9} {'n':>5} {'win%':>6} "
        f"{'total_₽':>10} {'avg_₽':>9} {'sharpe':>7} {'medHold':>8} "
        f"{'tp/sl/tm':>10}"
    )
    for s in summary:
        b = s["sharpe_best"]
        e = b["exits"]
        et = f"{e['tp']}/{e['sl']}/{e['time']}"
        print(
            f"{s['base']:<5} {b['score_min']:>10.2f} {b['hold_min']:>8}m "
            f"{b['n_trades']:>5} {b['win_rate']*100:>5.1f}% "
            f"{b['total_rub']:>+10.2f} {b['avg_rub']:>+9.2f} "
            f"{b['sharpe']:>+7.2f} {b['median_hold_min']:>7.0f}m {et:>10}"
        )

    print()
    print("Reading the table:")
    print("  • total_₽ is the NET P&L after 0.04% per-side commission (Tinkoff Trader)")
    print("  • tp/sl/tm = TP exits / SL exits / time-cap exits")
    print("  • This tests INDICATOR signal only — live scalp also uses book_imb + TFI")
    print("    (microstructure, not available in candles).  Live edge MAY be higher.")
    print("  • n < 5 trades = LOW CONFIDENCE")


if __name__ == "__main__":
    asyncio.run(main())
