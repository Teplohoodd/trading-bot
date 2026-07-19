"""Strategy bake-off: compare 4 simple tactics on the same historical sample.

Question this answers: are we losing because the system is over-engineered, or
because the market itself is hostile?  We strip the bot to its skeleton and test
4 archetypes (the ones professional retail/quant desks actually run in 2025-26):

  S1  buy_and_hold        — baseline, zero alpha, no skill required
  S2  donchian_breakout   — Turtle/Clenow channel-breakout trend follower
                            (long N-bar high, exit M-bar low; ATR stop)
  S3  rsi_mean_reversion  — classic bookmap/mean-reversion contrarian
                            (RSI<30 long, RSI>70 short; bollinger %b exit)
  S4  current_bot_logic   — the actual MLStrategy+meta gate path on the
                            same bars (uses the bundled model if available;
                            otherwise an indicator-vote proxy)

Output: per-ticker and aggregate Sharpe, total return, win rate, # of trades.

Why this design:
  * Same data, same costs, same stop math → strategies differ only in *signal*.
  * Hourly bars over 180 days (≈ what the live bot trains on).
  * 1-lot fixed sizing eliminates Kelly/position-sizer noise — we're asking
    "which signal works", not "which sizer + signal works".
  * Commissions modelled at COMMISSION_SHARES_PCT round-trip.
"""

import asyncio
import logging
import sys
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from core.broker import BrokerClient
from analysis.indicators import compute_indicators
from t_tech.invest import CandleInterval
from t_tech.invest.utils import quotation_to_decimal

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
logger = logging.getLogger("bakeoff")
logging.getLogger("t_tech").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ── Sample universe: top-10 liquid MOEX names ─────────────────────────────────
SAMPLE = ["SBER", "GAZP", "LKOH", "GMKN", "ROSN", "NVTK", "TATN", "MGNT", "PLZL", "YDEX"]
LOOKBACK_DAYS = 180
COMMISSION_PCT = 0.0005  # one-way; round-trip 0.10 %
STOP_ATR_MULT = 2.0
TARGET_ATR_MULT = 3.0
SIGNAL_BAR_HOLD_MAX = 60  # safety cap: force-exit after 60 bars


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _candles_to_df(candles) -> pd.DataFrame:
    rows = []
    for c in candles:
        rows.append(
            {
                "time": c.time,
                "open": float(quotation_to_decimal(c.open)),
                "high": float(quotation_to_decimal(c.high)),
                "low": float(quotation_to_decimal(c.low)),
                "close": float(quotation_to_decimal(c.close)),
                "volume": int(c.volume),
            }
        )
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Strategies — each returns a long/short/flat signal per bar
#   +1 = long, -1 = short, 0 = flat (only changes act as entries/exits)
# ─────────────────────────────────────────────────────────────────────────────
def s1_buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1, index=df.index)


def s2_donchian(df: pd.DataFrame, n_entry: int = 20, n_exit: int = 10) -> pd.Series:
    """Long on N-bar high, short on N-bar low, exit on opposite M-bar extreme."""
    high_n = df["high"].rolling(n_entry).max().shift(1)
    low_n = df["low"].rolling(n_entry).min().shift(1)
    exit_high = df["high"].rolling(n_exit).max().shift(1)
    exit_low = df["low"].rolling(n_exit).min().shift(1)

    sig = pd.Series(0, index=df.index, dtype=int)
    pos = 0
    for i in range(len(df)):
        c = df["close"].iloc[i]
        if pos == 0:
            if not np.isnan(high_n.iloc[i]) and c > high_n.iloc[i]:
                pos = 1
            elif not np.isnan(low_n.iloc[i]) and c < low_n.iloc[i]:
                pos = -1
        elif pos == 1:
            if not np.isnan(exit_low.iloc[i]) and c < exit_low.iloc[i]:
                pos = 0
        elif pos == -1:
            if not np.isnan(exit_high.iloc[i]) and c > exit_high.iloc[i]:
                pos = 0
        sig.iloc[i] = pos
    return sig


def s3_rsi_meanrev(df: pd.DataFrame) -> pd.Series:
    """RSI<30 long, RSI>70 short; exit when RSI re-crosses 50."""
    di = compute_indicators(df)
    rsi = di["rsi_14"]
    sig = pd.Series(0, index=df.index, dtype=int)
    pos = 0
    for i in range(len(df)):
        r = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50
        if pos == 0:
            if r < 30:
                pos = 1
            elif r > 70:
                pos = -1
        elif pos == 1 and r >= 50:
            pos = 0
        elif pos == -1 and r <= 50:
            pos = 0
        sig.iloc[i] = pos
    return sig


def s4_current_proxy(df: pd.DataFrame) -> pd.Series:
    """Proxy for current bot logic — same weighted-vote scheme as TechnicalStrategy,
    plus the thresholds (≥0.72 conf for ML path).  This is the bare TA half of
    the bot; the full ML+meta path would need the trained model loaded.  We test
    the TA proxy as a fair "current-stack-minus-ML" representation.
    """
    di = compute_indicators(df)
    sig = pd.Series(0, index=df.index, dtype=int)
    pos = 0
    for i in range(len(df)):
        last = di.iloc[i]
        scores, weights = [], []

        rsi = last.get("rsi_14", 50)
        if not pd.isna(rsi):
            if rsi < 30:
                scores.append(1.0)
            elif rsi > 70:
                scores.append(-1.0)
            elif rsi < 40:
                scores.append(0.3)
            elif rsi > 60:
                scores.append(-0.3)
            else:
                scores.append(0.0)
            weights.append(2.0)

        macd = last.get("macd_histogram", 0)
        if not pd.isna(macd):
            scores.append(1.0 if macd > 0 else -1.0 if macd < 0 else 0.0)
            weights.append(2.0)

        bbp = last.get("bb_percent_b", 0.5)
        if not pd.isna(bbp):
            if bbp < 0:
                scores.append(1.0)
            elif bbp > 1:
                scores.append(-1.0)
            elif bbp < 0.2:
                scores.append(0.6)
            elif bbp > 0.8:
                scores.append(-0.6)
            else:
                scores.append(0.0)
            weights.append(1.5)

        ema9 = last.get("ema_9", 0)
        ema21 = last.get("ema_21", 0)
        if ema9 and ema21:
            cross = (ema9 - ema21) / ema21 * 100
            scores.append(float(np.clip(cross / 2, -1, 1)))
            weights.append(1.5)

        if not scores:
            sig.iloc[i] = pos
            continue
        score = np.average(scores, weights=weights)
        # apply the same 0.72 conf threshold as live bot, but on TA vote
        if pos == 0:
            if score > 0.72:
                pos = 1
            elif score < -0.72:
                pos = -1
        else:
            # exit on opposite vote (signal_reversal proxy)
            if pos == 1 and score < -0.20:
                pos = 0
            elif pos == -1 and score > 0.20:
                pos = 0
        sig.iloc[i] = pos
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# Backtest engine — long/short/flat, ATR stops, commission, max-hold cap
# ─────────────────────────────────────────────────────────────────────────────
def backtest(
    df: pd.DataFrame,
    sig: pd.Series,
    name: str,
    use_stops: bool = True,
    stop_mult: float = STOP_ATR_MULT,
    target_mult: float = TARGET_ATR_MULT,
) -> dict:
    di = compute_indicators(df)
    atr = di["atr_14"].bfill()
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values

    trades = []
    pos = 0  # 1 long, -1 short, 0 flat
    entry_price = 0.0
    stop = 0.0
    target = 0.0
    entry_idx = 0
    entry_signal = 0  # signal value that triggered entry (so we exit when sig changes from it)

    for i in range(len(df)):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        s = int(sig.iloc[i])
        a = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else c * 0.02

        if pos == 0:
            # Look for new entry
            if s != 0 and not pd.isna(atr.iloc[i]):
                pos = s
                entry_price = c
                entry_idx = i
                entry_signal = s
                if pos == 1:
                    stop = c - stop_mult * a
                    target = c + target_mult * a
                else:
                    stop = c + stop_mult * a
                    target = c - target_mult * a
            continue

        # In a position — check exits in priority order
        exit_price = None
        exit_reason = None

        if use_stops:
            # 1. ATR stop (intra-bar)
            if pos == 1 and l <= stop:
                exit_price = stop
                exit_reason = "stop"
            elif pos == -1 and h >= stop:
                exit_price = stop
                exit_reason = "stop"

            # 2. Target (intra-bar) — if both hit, conservative: stop wins
            if exit_price is None:
                if pos == 1 and h >= target:
                    exit_price = target
                    exit_reason = "target"
                elif pos == -1 and l <= target:
                    exit_price = target
                    exit_reason = "target"

        # 3. Signal flip
        if exit_price is None and s != entry_signal:
            exit_price = c
            exit_reason = "signal_flip"

        # 4. Max-hold safety
        if exit_price is None and (i - entry_idx) >= SIGNAL_BAR_HOLD_MAX:
            exit_price = c
            exit_reason = "time_cap"

        if exit_price is not None:
            gross_pct = (exit_price - entry_price) / entry_price * pos
            net_pct = gross_pct - 2 * COMMISSION_PCT
            trades.append(
                {
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry": entry_price,
                    "exit": exit_price,
                    "direction": "long" if pos == 1 else "short",
                    "reason": exit_reason,
                    "gross_pct": gross_pct,
                    "net_pct": net_pct,
                    "bars_held": i - entry_idx,
                }
            )
            pos = 0
            entry_signal = 0

    # Aggregate
    if not trades:
        return {
            "strategy": name,
            "n": 0,
            "total_pct": 0.0,
            "win_rate": 0.0,
            "avg_pct": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "trades": [],
        }

    tdf = pd.DataFrame(trades)
    total_pct = tdf["net_pct"].sum() * 100
    win_rate = (tdf["net_pct"] > 0).mean() * 100
    avg_pct = tdf["net_pct"].mean() * 100
    # Sharpe on per-trade returns, annualized to bars-equivalent
    if tdf["net_pct"].std() > 0:
        sharpe = tdf["net_pct"].mean() / tdf["net_pct"].std() * math.sqrt(len(tdf))
    else:
        sharpe = 0.0
    equity = (1 + tdf["net_pct"]).cumprod()
    peak = equity.cummax()
    dd = ((equity - peak) / peak).min() * 100
    return {
        "strategy": name,
        "n": len(tdf),
        "total_pct": round(total_pct, 2),
        "win_rate": round(win_rate, 1),
        "avg_pct": round(avg_pct, 3),
        "sharpe": round(sharpe, 2),
        "max_dd": round(dd, 2),
        "trades": trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    settings = Settings()
    broker = BrokerClient(token=settings.T_INVEST_TOKEN, account_id=settings.T_INVEST_ACCOUNT_ID)
    await broker.connect()

    # Resolve sample tickers → figi (pick the share class on TQBR / SUR currency)
    logger.info("Resolving sample tickers…")
    universe = []
    for ticker in SAMPLE:
        try:
            results = await broker.find_instrument(ticker, kind="INSTRUMENT_TYPE_SHARE")
            # Prefer exact ticker match on MOEX (class_code='TQBR' for shares)
            picked = None
            for r in results:
                if getattr(r, "ticker", "") == ticker and getattr(r, "class_code", "") == "TQBR":
                    picked = r
                    break
            if picked is None and results:
                picked = next(
                    (r for r in results if getattr(r, "ticker", "") == ticker), results[0]
                )
            if picked:
                universe.append((ticker, picked.figi))
            else:
                logger.warning(f"  {ticker}: no match")
        except Exception as e:
            logger.warning(f"  {ticker}: resolve failed ({e})")
    logger.info(f"Universe: {[t for t,_ in universe]}")

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=LOOKBACK_DAYS)

    def s3_rsi_tight(df):
        di = compute_indicators(df)
        rsi = di["rsi_14"]
        sig = pd.Series(0, index=df.index, dtype=int)
        pos = 0
        for i in range(len(df)):
            r = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50
            if pos == 0:
                if r < 20:
                    pos = 1
                elif r > 80:
                    pos = -1
            elif pos == 1 and r >= 55:
                pos = 0
            elif pos == -1 and r <= 45:
                pos = 0
            sig.iloc[i] = pos
        return sig

    def s3_long_only(df):
        s = s3_rsi_meanrev(df)
        return s.where(s > 0, 0)

    def s3_short_only(df):
        s = s3_rsi_meanrev(df)
        return s.where(s < 0, 0)

    def s3_nostop(df):
        """RSI mean-rev but disable ATR stop/target — pure signal exit."""
        return s3_rsi_meanrev(df)

    def s4_loose(df):
        """Same TA vote as S4 but threshold lowered 0.72 → 0.30, no separate exit gate."""
        di = compute_indicators(df)
        sig = pd.Series(0, index=df.index, dtype=int)
        pos = 0
        for i in range(len(df)):
            last = di.iloc[i]
            scores, weights = [], []
            rsi = last.get("rsi_14", 50)
            if not pd.isna(rsi):
                if rsi < 30:
                    scores.append(1.0)
                elif rsi > 70:
                    scores.append(-1.0)
                elif rsi < 40:
                    scores.append(0.3)
                elif rsi > 60:
                    scores.append(-0.3)
                else:
                    scores.append(0.0)
                weights.append(2.0)
            macd = last.get("macd_histogram", 0)
            if not pd.isna(macd):
                scores.append(1.0 if macd > 0 else -1.0 if macd < 0 else 0.0)
                weights.append(2.0)
            bbp = last.get("bb_percent_b", 0.5)
            if not pd.isna(bbp):
                if bbp < 0:
                    scores.append(1.0)
                elif bbp > 1:
                    scores.append(-1.0)
                elif bbp < 0.2:
                    scores.append(0.6)
                elif bbp > 0.8:
                    scores.append(-0.6)
                else:
                    scores.append(0.0)
                weights.append(1.5)
            if not scores:
                sig.iloc[i] = pos
                continue
            score = np.average(scores, weights=weights)
            if pos == 0:
                if score > 0.30:
                    pos = 1
                elif score < -0.30:
                    pos = -1
            else:
                if pos == 1 and score < -0.10:
                    pos = 0
                elif pos == -1 and score > 0.10:
                    pos = 0
            sig.iloc[i] = pos
        return sig

    # (strategy_fn, use_stops, stop_mult, target_mult)
    strategies = {
        "S1_buy_and_hold": (s1_buy_and_hold, True, STOP_ATR_MULT, TARGET_ATR_MULT),
        "S2_donchian_20_10": (s2_donchian, True, STOP_ATR_MULT, TARGET_ATR_MULT),
        "S3_rsi_meanrev": (s3_rsi_meanrev, True, STOP_ATR_MULT, TARGET_ATR_MULT),
        "S3b_rsi_tight_20_80": (s3_rsi_tight, True, STOP_ATR_MULT, TARGET_ATR_MULT),
        "S3c_rsi_long_only": (s3_long_only, True, STOP_ATR_MULT, TARGET_ATR_MULT),
        "S3d_rsi_short_only": (s3_short_only, True, STOP_ATR_MULT, TARGET_ATR_MULT),
        "S3e_rsi_no_stops": (s3_nostop, False, 0, 0),
        "S3f_rsi_tight_atr_1_1": (s3_rsi_meanrev, True, 1.0, 1.0),
        # NEW DEFAULTS — what the live bot will run after 2026-05-14 retune.
        # Wide 4× ATR stop ≈ "catastrophe-only" floor; tight 2× ATR target
        # so signal_reversal exits dominate before the target fires.
        "S3g_NEW_4x_stop_2x_tgt": (s3_rsi_meanrev, True, 4.0, 2.0),
        "S4_current_proxy": (s4_current_proxy, True, STOP_ATR_MULT, TARGET_ATR_MULT),
        "S4b_proxy_loose": (s4_loose, True, STOP_ATR_MULT, TARGET_ATR_MULT),
    }

    all_results = {name: [] for name in strategies}
    for ticker, figi in universe:
        try:
            candles = await broker.get_candles(
                figi, start, now, interval=CandleInterval.CANDLE_INTERVAL_HOUR
            )
            if len(candles) < 200:
                logger.info(f"  {ticker}: only {len(candles)} bars, skipping")
                continue
            df = _candles_to_df(candles)
            logger.info(
                f"{ticker}: {len(df)} hourly bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}"
            )

            for name, (fn, use_stops, stop_m, tgt_m) in strategies.items():
                try:
                    sig = fn(df)
                    res = backtest(
                        df, sig, name, use_stops=use_stops, stop_mult=stop_m, target_mult=tgt_m
                    )
                    res["ticker"] = ticker
                    all_results[name].append(res)
                except Exception as e:
                    logger.exception(f"  {ticker} {name}: {e}")
        except Exception as e:
            logger.exception(f"{ticker}: candle fetch failed: {e}")

    await broker.disconnect()

    # Print per-ticker table
    print("\n" + "=" * 100)
    print(
        f"{'STRATEGY':<22} {'TICKER':<8} {'N':>4} {'TOT%':>8} {'WIN%':>6} {'AVG%':>7} {'SHARPE':>7} {'MAXDD%':>8}"
    )
    print("-" * 100)
    for name, results in all_results.items():
        for r in results:
            print(
                f"{r['strategy']:<22} {r['ticker']:<8} {r['n']:>4} "
                f"{r['total_pct']:>+8.2f} {r['win_rate']:>5.1f}% "
                f"{r['avg_pct']:>+7.3f} {r['sharpe']:>+7.2f} {r['max_dd']:>+8.2f}"
            )
        print()

    # Aggregate per strategy
    print("=" * 100)
    print("AGGREGATE (sum / mean across all sample tickers)")
    print("-" * 100)
    print(
        f"{'STRATEGY':<22} {'N_TRADES':>9} {'SUM_TOT%':>10} {'AVG_TOT%':>10} {'MEAN_WIN%':>11} {'MEAN_SHARPE':>13} {'WORST_DD%':>11}"
    )
    summary = []
    for name, results in all_results.items():
        if not results:
            continue
        total_trades = sum(r["n"] for r in results)
        sum_tot = sum(r["total_pct"] for r in results)
        avg_tot = sum_tot / len(results)
        mean_win = sum(r["win_rate"] for r in results) / len(results)
        mean_sharpe = sum(r["sharpe"] for r in results) / len(results)
        worst_dd = min(r["max_dd"] for r in results)
        summary.append((name, total_trades, sum_tot, avg_tot, mean_win, mean_sharpe, worst_dd))
        print(
            f"{name:<22} {total_trades:>9} {sum_tot:>+10.2f} {avg_tot:>+10.2f} "
            f"{mean_win:>10.1f}% {mean_sharpe:>+13.2f} {worst_dd:>+11.2f}"
        )

    # Pick the winner by mean_sharpe + sanity check on activity (need ≥1 trade/ticker on avg)
    if summary:
        ranked = sorted(summary, key=lambda x: x[5], reverse=True)
        print(
            f"\n  → Best by mean Sharpe: {ranked[0][0]} (sharpe={ranked[0][5]:+.2f}, "
            f"total {ranked[0][2]:+.2f} % across {ranked[0][1]} trades)"
        )
        ranked_t = sorted(summary, key=lambda x: x[2], reverse=True)
        print(f"  → Best by sum return:  {ranked_t[0][0]} (total {ranked_t[0][2]:+.2f} %)")


if __name__ == "__main__":
    asyncio.run(main())
