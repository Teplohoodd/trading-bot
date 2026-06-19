"""Test the 'bigger TP / longer hold' hypothesis on historical 1-min candles.

Hypothesis (user 2026-05-20):
  "If our signal direction is right and only commission eats us, maybe
  we should aim for BIGGER moves so commission becomes a smaller % of P&L."

Math check:
  Commission round-trip on BR (price ≈ 100 ₽) = 100 × 0.04% × 2 = 0.08 ₽
  TP = 1.0 × ATR_15m (≈ 0.5 ₽) → commission is 16% of TP
  TP = 5.0 × ATR_15m (≈ 2.5 ₽) → commission is only 3% of TP

  If signal IC stays positive AT LONGER HORIZONS, bigger TP wins.
  If signal decays (typical for microstructure), bigger TP becomes
  random — same commission drag, less hits.

This script runs the SAME microstructure-proxy signal (book + tfi)
from scalp_v2_backtest.py on a MUCH WIDER grid:

  TP / SL ATR-mult: {1, 2, 3, 5, 8}  ×  {0.8, 1.5, 2.5}
  Max hold minutes: {3, 10, 30, 60, 120, 240}
  Score threshold: {0.45, 0.60, 0.75}

Goal: see if there's ANY (TP, hold, threshold) cell with positive NET
after Tinkoff Trader 0.04% commission.

Usage:  python -m futbot.scripts.extended_horizon_test [days]
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
logger = logging.getLogger("ext_horizon")


BASES = ["BR", "GZ", "LK", "GD"]  # focused — drop SR/MX/NG (commission math fails)

# WIDE grid — that's the point
TP_ATR_MULTS = [1.0, 2.0, 3.0, 5.0, 8.0]
SL_ATR_MULTS = [0.8, 1.5, 2.5]
HOLD_MIN = [3, 10, 30, 60, 120, 240]
SCORE_MINS = [0.45, 0.60, 0.75]

# Microstructure proxies (same as scalp_v2_backtest.py)
TFI_WINDOW = 3
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
# Signals (same proxies as scalp_v2_backtest)
# ─────────────────────────────────────────────────────────────────────────────
def _wilder_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _atr_15m(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR on 15-min aggregate from 1-min bars."""
    df15 = df.copy()
    df15["t15"] = df15.index // 15
    g = df15.groupby("t15")
    bars = pd.DataFrame(
        {
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
        }
    ).reset_index(drop=True)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - bars["close"].shift()).abs(),
            (bars["low"] - bars["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr15 = _wilder_ema(tr, n)
    # Re-index back to 1-min bars (each 15-min ATR applies to next 15 min)
    expanded = atr15.reindex(df15["t15"].values).reset_index(drop=True)
    return expanded


def compute_proxy_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    close_pos = ((out["close"] - out["low"]) / rng).fillna(0.5).clip(0, 1)
    signed_vol = out["volume"] * (2 * close_pos - 1)
    rolling_signed = signed_vol.rolling(TFI_WINDOW).sum()
    rolling_total = out["volume"].rolling(TFI_WINDOW).sum().replace(0, np.nan)
    out["proxy_tfi"] = (rolling_signed / rolling_total).fillna(0).clip(-1, 1)
    vol_mean = out["volume"].rolling(VOL_WINDOW).mean()
    vol_std = out["volume"].rolling(VOL_WINDOW).std().replace(0, np.nan)
    vol_z = ((out["volume"] - vol_mean) / vol_std).fillna(0)
    direction = (close_pos - 0.5) * 2
    magnitude = np.clip(vol_z / 2, 0, 1)
    out["proxy_book"] = (direction * magnitude).clip(-1, 1)
    out["score"] = 0.60 * out["proxy_book"] + 0.40 * out["proxy_tfi"]
    out["atr_15m"] = _atr_15m(out)
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
    atr = df["atr_15m"].values
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
            # Commission gate
            tp_rub = tp_atr * a * rpp * lot_size
            rt = comm.estimated_round_trip_cost(
                price=c,
                lots=1,
                lot_size=lot_size,
                rub_per_point=rpp,
                instrument_kind="future",
                base_ticker=base,
            )
            if tp_rub < 2 * rt:
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
        held = t - entry_idx
        exit_p = None
        reason = None
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
        if exit_p is None and held >= hold_min:
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
        holds.append(held)
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
        app_name="ext-horizon",
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
            ls = int(getattr(f, "lot", 1) or 1)
            df = compute_proxy_signals(df)
            contracts[base] = {"df": df, "ticker": f.ticker, "rpp": rpp, "lot": ls}
            logger.info(f"  {base} ({f.ticker}): {len(df)} bars")
        except Exception as e:
            logger.warning(f"  {base}: {e}")
    await broker.disconnect()

    print()
    print("=" * 145)
    print(f"EXTENDED HORIZON TEST — {days}d × 1-min × {len(contracts)} contracts")
    print("Question: does BIGGER TP help when commission is ~constant?")
    print("=" * 145)

    # Per-contract: find all positive cells across the wide grid
    for base, info in contracts.items():
        print(f"\n── {base} ({info['ticker']}) ──")
        positive = []
        all_cells = []
        for tp_m in TP_ATR_MULTS:
            for sl_m in SL_ATR_MULTS:
                for hm in HOLD_MIN:
                    for sm in SCORE_MINS:
                        r = backtest_cell(
                            info["df"],
                            base=base,
                            hold_min=hm,
                            score_min=sm,
                            tp_atr=tp_m,
                            sl_atr=sl_m,
                            rpp=info["rpp"],
                            lot_size=info["lot"],
                        )
                        if r.get("n_trades", 0) >= 5:
                            c = {"tp": tp_m, "sl": sl_m, "hold": hm, "score_min": sm, **r}
                            all_cells.append(c)
                            if c["total_rub"] > 0:
                                positive.append(c)

        if not all_cells:
            print("  no cells with ≥5 trades")
            continue

        # Top 5 by total_rub (good or bad — to show full landscape)
        all_cells.sort(key=lambda x: -x["total_rub"])
        print(f"  Total cells tested: {len(all_cells)}  positive (NET>0): {len(positive)}")
        print(
            f"  {'TP/SL':>9} {'hold':>6} {'sc≥':>5} {'n':>5} {'win%':>6} "
            f"{'NET_₽':>10} {'avg_₽':>9} {'sharpe':>7} {'tp/sl/tm':>10}"
        )
        print(f"  ─── TOP 5 by NET ───")
        for c in all_cells[:5]:
            e = c["exits"]
            et = f"{e['tp']}/{e['sl']}/{e['time']}"
            tp_sl = f"{c['tp']}/{c['sl']}"
            print(
                f"  {tp_sl:>9} {c['hold']:>5}m {c['score_min']:>5.2f} "
                f"{c['n_trades']:>5} {c['win_rate']*100:>5.1f}% "
                f"{c['total_rub']:>+10.2f} {c['avg_rub']:>+9.3f} "
                f"{c['sharpe']:>+7.2f} {et:>10}"
            )
        if positive:
            print(f"  ─── BOTTOM 5 by NET ───")
            for c in all_cells[-5:]:
                e = c["exits"]
                et = f"{e['tp']}/{e['sl']}/{e['time']}"
                tp_sl = f"{c['tp']}/{c['sl']}"
                print(
                    f"  {tp_sl:>9} {c['hold']:>5}m {c['score_min']:>5.2f} "
                    f"{c['n_trades']:>5} {c['win_rate']*100:>5.1f}% "
                    f"{c['total_rub']:>+10.2f} {c['avg_rub']:>+9.3f} "
                    f"{c['sharpe']:>+7.2f} {et:>10}"
                )

    # Aggregate: how does NET evolve with TP size?
    print()
    print("=" * 145)
    print(
        "AGGREGATE: avg NET ₽ per cell, by TP multiplier (averaged across contracts/holds/scores)"
    )
    print("=" * 145)
    print(f"  {'TP_atr':<8}", end="")
    for sl_m in SL_ATR_MULTS:
        print(f"  SL={sl_m}".rjust(15), end="")
    print()
    for tp_m in TP_ATR_MULTS:
        print(f"  {tp_m:<8.1f}", end="")
        for sl_m in SL_ATR_MULTS:
            vals = []
            for base, info in contracts.items():
                for hm in HOLD_MIN:
                    for sm in SCORE_MINS:
                        r = backtest_cell(
                            info["df"],
                            base=base,
                            hold_min=hm,
                            score_min=sm,
                            tp_atr=tp_m,
                            sl_atr=sl_m,
                            rpp=info["rpp"],
                            lot_size=info["lot"],
                        )
                        if r.get("n_trades", 0) >= 5:
                            vals.append(r["avg_rub"])
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {avg:>+12.3f}₽  ".rjust(15), end="")
            else:
                print(f"  {'—':>13}  ".rjust(15), end="")
        print()

    print()
    print("Reading:")
    print("  • If NET grows with TP_atr → user's hypothesis confirmed (bigger TP helps)")
    print("  • If NET stays negative or flat → signal decays before bigger TPs hit")
    print("  • Per-cell NET should be positive after Tinkoff Trader 0.04% commission")


if __name__ == "__main__":
    asyncio.run(main())
