"""Scalp v2 backtest — microstructure-only signal, approximated from 1m candles.

⚠️ APPROXIMATION DISCLOSURE
The live scalp uses real order-book imbalance + tick-level trade-flow
imbalance.  T-Invest API doesn't store historical L2 or trade-by-trade
data beyond the last hour, so this backtest uses two DOCUMENTED proxies
to test whether the signal STRUCTURE has any historical edge:

  proxy_tfi  — Lee-Ready (1991) heuristic: each 1-min bar is classified
               by where its close sits in the (low, high) range.
               close_pos = (close - low) / (high - low)
               signed_vol = volume * (2 * close_pos - 1)
               proxy_tfi[t] = sum(signed_vol[t-W:t]) / sum(volume[t-W:t])

  proxy_book — Volume burst direction.  Big volume + close-near-high
               typically reflects book imbalance toward the bid (buyers
               pushed price up).
               vol_z = (volume - vol_mean_20) / vol_std_20
               proxy_book[t] = sign(close_pos - 0.5) × min(vol_z / 2, 1)

These are approximations of approximations.  They CORRELATE with the
real microstructure signals at coarse resolution but the live tick
versions are far more responsive (and noisier).

What we can learn:
  * Does the proxy_book + proxy_tfi composite have ANY directional edge
    at 1-10 minute holds?
  * Which contracts respond best to these signals?
  * What thresholds / hold times are in the sweet spot?

What we CAN'T learn:
  * The exact P&L of scalp v2 in live (signals differ).  But if the
    proxy edge is meaningfully positive, the live edge is plausibly
    higher (real signals are stronger).  If the proxy edge is zero or
    negative, that's STRONG evidence against scalp v2.

scalp v2 signal formula (mirrors live design but pure microstructure):
    score = 0.60 × proxy_book + 0.40 × proxy_tfi

Trade rules:
    Enter when |score| ≥ score_min   AND TP > 2 × round-trip commission
    Exit on:
      TP   — +TP_atr × ATR_1m       (very tight)
      SL   — -SL_atr × ATR_1m
      Time — held > max_hold_min minutes (tested 1, 2, 3, 5, 10)
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
logger = logging.getLogger("scalp_v2")


# Universe — viable scalp candidates.  Si excluded (limit-only).
BASES = ["BR", "GZ", "LK", "NG", "GD", "MX", "SR"]

# Tight grid — microstructure should work on short horizons or not at all
HOLD_MIN = [1, 2, 3, 5, 10]
SCORE_MIN = [0.30, 0.45, 0.60, 0.75]
TP_SL = [(0.8, 0.6), (1.0, 0.8), (1.5, 1.0), (2.0, 1.2)]

# TFI rolling window in 1-min bars
TFI_WINDOW = 3
# Vol-burst lookback for proxy_book
VOL_WINDOW = 20


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
# Microstructure proxies + ATR
# ─────────────────────────────────────────────────────────────────────────────
def _wilder_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _atr_1m(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _wilder_ema(tr, n)


def compute_proxy_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    close_pos = ((out["close"] - out["low"]) / rng).fillna(0.5).clip(0, 1)

    # ── proxy_tfi: Lee-Ready bar classification + rolling normalisation
    signed_vol = out["volume"] * (2 * close_pos - 1)
    rolling_signed = signed_vol.rolling(TFI_WINDOW).sum()
    rolling_total = out["volume"].rolling(TFI_WINDOW).sum().replace(0, np.nan)
    out["proxy_tfi"] = (rolling_signed / rolling_total).fillna(0).clip(-1, 1)

    # ── proxy_book: volume burst with directional info
    vol_mean = out["volume"].rolling(VOL_WINDOW).mean()
    vol_std = out["volume"].rolling(VOL_WINDOW).std().replace(0, np.nan)
    vol_z = ((out["volume"] - vol_mean) / vol_std).fillna(0)
    # Direction = sign of close_pos minus 0.5  (closes near high → +1, near low → −1)
    direction = (close_pos - 0.5) * 2  # in [-1, +1]
    # Magnitude capped at 1, requires meaningful volume z-score (>0)
    magnitude = np.clip(vol_z / 2, 0, 1)
    out["proxy_book"] = (direction * magnitude).clip(-1, 1)

    # ── composite score (mirrors v2 design: 0.60 book + 0.40 tfi)
    out["score"] = 0.60 * out["proxy_book"] + 0.40 * out["proxy_tfi"]

    # ── ATR (used for stops/TP)
    out["atr_1m"] = _atr_1m(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Backtest one cell
# ─────────────────────────────────────────────────────────────────────────────
def backtest_cell(
    df: pd.DataFrame,
    *,
    base: str,
    hold_min: int,
    score_min: float,
    tp_atr: float,
    sl_atr: float,
    rpp: float,
    lot_size: int,
) -> dict:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    score = df["score"].values
    atr = df["atr_1m"].values
    n = len(df)

    pos = 0
    entry_idx = None
    entry_p = 0.0
    stop = 0.0
    tp = 0.0
    pnls = []
    holds = []
    exits = {"tp": 0, "sl": 0, "time": 0}

    for t in range(max(VOL_WINDOW, TFI_WINDOW) + 5, n):
        s = score[t]
        c = close[t]
        a = atr[t]
        if np.isnan(s) or np.isnan(a) or a <= 0:
            continue

        if pos == 0:
            if abs(s) < score_min:
                continue
            # Commission gate — TP must beat 2× RT commission
            tp_profit_rub = tp_atr * a * rpp * lot_size
            rt = comm.estimated_round_trip_cost(
                price=c,
                lots=1,
                lot_size=lot_size,
                rub_per_point=rpp,
                instrument_kind="future",
                base_ticker=base,
            )
            if tp_profit_rub < 2 * rt:
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

        held_min = t - entry_idx
        exit_p = None
        reason = None
        # Intra-bar high/low against the levels
        if pos == +1:
            if low[t] <= stop:
                exit_p = stop
                reason = "sl"
            elif high[t] >= tp:
                exit_p = tp
                reason = "tp"
        else:
            if high[t] >= stop:
                exit_p = stop
                reason = "sl"
            elif low[t] <= tp:
                exit_p = tp
                reason = "tp"
        if exit_p is None and held_min >= hold_min:
            exit_p = c
            reason = "time"
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
        holds.append(held_min)
        exits[reason] += 1
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
        "best_rub": round(float(arr.max()), 2),
        "worst_rub": round(float(arr.min()), 2),
        "median_hold_min": round(float(np.median(holds)), 1),
        "sharpe": round(float(sharpe), 2),
        "dd_rub": round(dd, 2),
        "exits": exits,
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
        app_name="scalp-v2",
    )
    await broker.connect()
    logger.info(f"Fetching {days}d × 1-min for {BASES}")

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
                interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
            )
            df = _candles_to_df(candles)
            if len(df) < 500:
                continue
            meta = broker.extract_futures_metadata(f)
            rpp = float(meta.get("rub_per_point") or 1.0)
            lot_size = int(getattr(f, "lot", 1) or 1)
            df = compute_proxy_signals(df)
            contracts[base] = {
                "df": df,
                "ticker": f.ticker,
                "rpp": rpp,
                "lot": lot_size,
                "last": float(df["close"].iloc[-1]),
            }
            logger.info(f"  {base} ({f.ticker}): {len(df)} bars  last={df['close'].iloc[-1]:.4f}")
        except Exception as e:
            logger.warning(f"  {base}: {e}")
    await broker.disconnect()
    if not contracts:
        return

    # ─── Grid: per contract, find best (hold, score_min, tp/sl) ──────
    print()
    print("=" * 130)
    print(
        f"SCALP V2 BACKTEST — proxy signals (book + tfi), {days}d × 1-min × {len(contracts)} contracts"
    )
    print(f"Composite: 0.60 × proxy_book + 0.40 × proxy_tfi   |   Commission 0.04 % per side")
    print("=" * 130)

    summary = []
    for base, info in contracts.items():
        print(f"\n── {base} ({info['ticker']}) ──")
        best_by_sharpe = None
        best_by_total = None
        cells = []
        for tp_atr, sl_atr in TP_SL:
            for sm in SCORE_MIN:
                for hm in HOLD_MIN:
                    r = backtest_cell(
                        info["df"],
                        base=base,
                        hold_min=hm,
                        score_min=sm,
                        tp_atr=tp_atr,
                        sl_atr=sl_atr,
                        rpp=info["rpp"],
                        lot_size=info["lot"],
                    )
                    if r.get("n_trades", 0) >= 5:
                        c = {"tp": tp_atr, "sl": sl_atr, "score_min": sm, "hold_min": hm, **r}
                        cells.append(c)
                        if best_by_sharpe is None or c["sharpe"] > best_by_sharpe["sharpe"]:
                            best_by_sharpe = c
                        if best_by_total is None or c["total_rub"] > best_by_total["total_rub"]:
                            best_by_total = c

        if not cells:
            print(f"  (no cell with ≥5 trades — signal too rare for this contract)")
            continue

        bs = best_by_sharpe
        bt = best_by_total
        print(
            f"  best Sharpe:  TP/SL={bs['tp']}/{bs['sl']}  "
            f"score≥{bs['score_min']}  hold={bs['hold_min']}m  "
            f"n={bs['n_trades']}  win={bs['win_rate']*100:.0f}%  "
            f"NET={bs['total_rub']:+.1f}₽  Sharpe={bs['sharpe']:+.2f}  "
            f"DD={bs['dd_rub']:+.1f}₽  exits={bs['exits']}"
        )
        print(
            f"  best Total:   TP/SL={bt['tp']}/{bt['sl']}  "
            f"score≥{bt['score_min']}  hold={bt['hold_min']}m  "
            f"n={bt['n_trades']}  win={bt['win_rate']*100:.0f}%  "
            f"NET={bt['total_rub']:+.1f}₽  Sharpe={bt['sharpe']:+.2f}"
        )
        summary.append({"base": base, "sharpe": bs, "total": bt})

    print()
    print("=" * 130)
    print("BEST PARAMETERS — sorted by total NET ₽ (Sharpe-best per contract)")
    print("=" * 130)
    summary.sort(key=lambda s: -s["sharpe"]["total_rub"])
    print(
        f"{'base':<5} {'TP/SL':>9} {'score≥':>7} {'hold':>5} {'n':>5} "
        f"{'win%':>6} {'NET_₽':>10} {'avg_₽':>9} {'sharpe':>7} {'DD_₽':>9} {'tp/sl/tm':>10}"
    )
    for s in summary:
        b = s["sharpe"]
        et = f"{b['exits']['tp']}/{b['exits']['sl']}/{b['exits']['time']}"
        tp_sl = f"{b['tp']}/{b['sl']}"
        print(
            f"{s['base']:<5} {tp_sl:>9} {b['score_min']:>7.2f} "
            f"{b['hold_min']:>4}m {b['n_trades']:>5} {b['win_rate']*100:>5.1f}% "
            f"{b['total_rub']:>+10.2f} {b['avg_rub']:>+9.2f} "
            f"{b['sharpe']:>+7.2f} {b['dd_rub']:>+9.2f} {et:>10}"
        )
    print()
    print("Reading:")
    print("  • signals are APPROXIMATIONS of live book/tfi (Lee-Ready + vol-burst).")
    print("  • NET ₽ is after 0.04% per-side commission.")
    print("  • Positive NET here = the proxy signal has SOME directional edge.")
    print("  • Live edge from real microstructure may differ (typically stronger).")


if __name__ == "__main__":
    asyncio.run(main())
