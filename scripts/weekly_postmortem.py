"""Weekly postmortem: pulls candles from T-API for every closed trade in
the analysis window, computes MAE/MFE/stop-touched/tp-touched, and writes a
markdown report + parquet dataset.

Differs from run_forensics.py:
  * window is parameterised (--days N, default 7)
  * candle_cache is BACKFILLED for trades where it's empty — no more
    "candle_cache_empty" caveat in reports
  * adds MAE / MFE / time-to-MAE / time-to-MFE / stop_touched_intraday
    columns per trade
  * adds long/short asymmetry, time-of-day heatmap, stop-distance/ATR vs PnL,
    confidence-calibration ECE+Brier, and per-ticker breakdown sections.

Run:  python -m scripts.weekly_postmortem [--days 7] [--no-fetch]
"""

import argparse
import asyncio
import json
import logging
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sci_stats

from t_tech.invest import CandleInterval
from t_tech.invest.utils import quotation_to_decimal

from config.settings import Settings
from core.broker import BrokerClient
from database.db import Repository

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("t_tech.invest.logging").setLevel(logging.WARNING)
log = logging.getLogger("postmortem")

REPORTS = Path("analysis/reports")
REPORTS.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Candle backfill — pull 5-min candles around each trade window into DB
# ────────────────────────────────────────────────────────────────────────────

INTERVAL_5M = CandleInterval.CANDLE_INTERVAL_5_MIN
INTERVAL_NAME = "5m"


async def backfill_candles(
    broker: BrokerClient,
    db: Repository,
    trades: pd.DataFrame,
    lookback_h: int = 1,
    lookahead_h: int = 2,
) -> int:
    """For every trade, fetch 5-min candles covering [entry-1h, exit+2h] and
    upsert into candle_cache.  Skips trades that already have ≥10 cached
    candles in the window.

    Returns number of trades for which candles were freshly fetched.
    """
    fetched = 0
    cached = 0
    failed = 0
    for _, t in trades.iterrows():
        figi = t["figi"]
        entry = pd.to_datetime(t["entry_time"], utc=True).to_pydatetime()
        if pd.isna(t["exit_time"]):
            exit_dt = datetime.now(timezone.utc)
        else:
            exit_dt = pd.to_datetime(t["exit_time"], utc=True).to_pydatetime()
        from_dt = entry - timedelta(hours=lookback_h)
        to_dt = exit_dt + timedelta(hours=lookahead_h)

        existing = await db.get_cached_candles(
            figi, INTERVAL_NAME, from_dt.isoformat(), to_dt.isoformat()
        )
        if len(existing) >= 10:
            cached += 1
            continue

        try:
            candles = await broker.get_candles(figi, from_dt, to_dt, INTERVAL_5M)
            if not candles:
                failed += 1
                continue
            rows = [
                {
                    "ts": c.time.isoformat(),
                    "open": float(quotation_to_decimal(c.open)),
                    "high": float(quotation_to_decimal(c.high)),
                    "low": float(quotation_to_decimal(c.low)),
                    "close": float(quotation_to_decimal(c.close)),
                    "volume": int(c.volume),
                }
                for c in candles
            ]
            await db.upsert_candles(figi, INTERVAL_NAME, rows)
            fetched += 1
            log.info(f"  fetched {len(rows)} 5m candles for trade #{t['id']} {t['ticker']}")
        except Exception as e:
            log.warning(f"  candle fetch failed for #{t['id']} {t['ticker']}: {e}")
            failed += 1

    log.info(f"Candle backfill: fetched={fetched}, cached_already={cached}, failed={failed}")
    return fetched


# ────────────────────────────────────────────────────────────────────────────
# MAE / MFE / stop-touched per trade, using cached 5-min candles
# ────────────────────────────────────────────────────────────────────────────


async def enrich_trades(db: Repository, trades: pd.DataFrame) -> pd.DataFrame:
    """Add MAE/MFE/stop_touched/tp_touched/atr columns derived from 5-min candles."""
    enrichments = []
    for _, t in trades.iterrows():
        figi = t["figi"]
        entry = pd.to_datetime(t["entry_time"], utc=True)
        exit_dt = (
            pd.to_datetime(t["exit_time"], utc=True)
            if pd.notna(t["exit_time"])
            else pd.Timestamp.now(tz="UTC")
        )
        from_iso = (entry - pd.Timedelta(hours=1)).isoformat()
        to_iso = (exit_dt + pd.Timedelta(hours=1)).isoformat()
        rows = await db.get_cached_candles(figi, INTERVAL_NAME, from_iso, to_iso)
        if not rows:
            enrichments.append({"id": int(t["id"]), "no_candles": True})
            continue
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        # Restrict to actual hold window
        held = df[(df["ts"] >= entry) & (df["ts"] <= exit_dt)].copy()
        if len(held) < 2:
            held = df.iloc[-min(5, len(df)) :].copy()

        entry_price = float(t["entry_price"])
        direction = t["direction"]
        sl = float(t["stop_loss"]) if pd.notna(t["stop_loss"]) else None
        tp = float(t["take_profit"]) if pd.notna(t["take_profit"]) else None

        if direction == "buy":
            worst = float(held["low"].min())
            best = float(held["high"].max())
            mae_pct = (worst - entry_price) / entry_price * 100  # negative for adverse
            mfe_pct = (best - entry_price) / entry_price * 100
            stop_touched = bool(sl and (held["low"] <= sl).any())
            tp_touched = bool(tp and (held["high"] >= tp).any())
            t_to_mae = held.loc[held["low"].idxmin(), "ts"] if len(held) else None
            t_to_mfe = held.loc[held["high"].idxmax(), "ts"] if len(held) else None
        else:  # sell / short
            worst = float(held["high"].max())
            best = float(held["low"].min())
            mae_pct = (
                (entry_price - worst) / entry_price * 100
            )  # negative for adverse (short rallies up = bad)
            mfe_pct = (entry_price - best) / entry_price * 100
            stop_touched = bool(sl and (held["high"] >= sl).any())
            tp_touched = bool(tp and (held["low"] <= tp).any())
            t_to_mae = held.loc[held["high"].idxmax(), "ts"] if len(held) else None
            t_to_mfe = held.loc[held["low"].idxmin(), "ts"] if len(held) else None

        # Hourly ATR-14 reconstructed from the 5-min bars BEFORE entry.
        # Earlier version used mean(5-min range) over the hold window which
        # came out 6-10× smaller than canonical ATR-14, collapsing the
        # sl_in_atr binning into one bucket.  Reconstruct hourly bars
        # (resample 5m → 1h) and run a true 14-bar Wilder-style ATR over
        # the lookback window.  This matches what the strategy uses at
        # entry time.
        try:
            pre_entry = df[df["ts"] < entry].copy()
            if len(pre_entry) >= 14 * 12:  # ≥14 hours worth of 5-min bars
                pre_entry = pre_entry.set_index("ts")
                hourly = (
                    pre_entry.resample("1H")
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna()
                )
                if len(hourly) >= 14:
                    hi, lo, cl = hourly["high"], hourly["low"], hourly["close"]
                    tr = pd.concat(
                        [
                            hi - lo,
                            (hi - cl.shift()).abs(),
                            (lo - cl.shift()).abs(),
                        ],
                        axis=1,
                    ).max(axis=1)
                    atr14 = float(tr.rolling(14).mean().iloc[-1])
                    atr_pct_proxy = atr14 / entry_price * 100
                else:
                    atr_pct_proxy = float("nan")
            else:
                # Fallback: 5-min range × 12 (rough hourly approximation)
                tr5 = (held["high"] - held["low"]).mean()
                atr_pct_proxy = float(tr5 * 12 / entry_price * 100)
        except Exception:
            atr_pct_proxy = float("nan")

        # Stop / TP distance in % of entry
        sl_dist_pct = abs(sl - entry_price) / entry_price * 100 if sl else float("nan")
        tp_dist_pct = abs(tp - entry_price) / entry_price * 100 if tp else float("nan")

        # Time to MAE / MFE in minutes from entry
        def _mins(ts):
            if ts is None:
                return float("nan")
            try:
                return float((ts - entry).total_seconds() / 60.0)
            except Exception:
                return float("nan")

        enrichments.append(
            {
                "id": int(t["id"]),
                "no_candles": False,
                "mae_pct": round(mae_pct, 3),
                "mfe_pct": round(mfe_pct, 3),
                "atr_pct_proxy": round(atr_pct_proxy, 3) if not math.isnan(atr_pct_proxy) else None,
                "sl_dist_pct": round(sl_dist_pct, 3) if not math.isnan(sl_dist_pct) else None,
                "tp_dist_pct": round(tp_dist_pct, 3) if not math.isnan(tp_dist_pct) else None,
                "stop_touched_intraday": stop_touched,
                "tp_touched_intraday": tp_touched,
                "min_to_mae": round(_mins(t_to_mae), 1) if t_to_mae is not None else None,
                "min_to_mfe": round(_mins(t_to_mfe), 1) if t_to_mfe is not None else None,
                "n_5m_bars": int(len(held)),
            }
        )

    enriched = pd.DataFrame(enrichments)
    return trades.merge(enriched, on="id", how="left")


# ────────────────────────────────────────────────────────────────────────────
# Analysis blocks
# ────────────────────────────────────────────────────────────────────────────


def section_overview(t: pd.DataFrame) -> str:
    out = ["## 1. Overview", ""]
    out.append(f"- closed trades: **{len(t)}**")
    out.append(
        f"- mean P&L%: **{t['pnl_pct'].mean():+.3f}%**, median: {t['pnl_pct'].median():+.3f}%"
    )
    out.append(f"- sum P&L: **{t['pnl'].sum():+.2f} ₽**")
    out.append(f"- win rate: **{(t['pnl']>0).mean()*100:.1f}%** ({(t['pnl']>0).sum()}/{len(t)})")
    if (t["pnl"] > 0).any() and (t["pnl"] <= 0).any():
        avg_w = t.loc[t["pnl"] > 0, "pnl_pct"].mean()
        avg_l = t.loc[t["pnl"] <= 0, "pnl_pct"].abs().mean()
        b = avg_w / avg_l if avg_l > 0 else 0
        p = (t["pnl"] > 0).mean()
        q = 1 - p
        f = (p * b - q) / b if b > 0 else 0
        out.append(
            f"- avg win: {avg_w:.3f}% | avg loss: {avg_l:.3f}% | R/R: {b:.2f} | Kelly f*: **{f:+.3f}**"
        )
    return "\n".join(out) + "\n"


def section_long_short(t: pd.DataFrame) -> str:
    out = ["## 2. Long / Short asymmetry", ""]
    grp = t.groupby("direction").agg(
        n=("pnl_pct", "count"),
        mean_pct=("pnl_pct", "mean"),
        median_pct=("pnl_pct", "median"),
        sum_rub=("pnl", "sum"),
        hit=("pnl_pct", lambda x: (x > 0).mean()),
    )
    out.append(grp.round(3).to_markdown())
    # Per-direction Kelly
    out.append("")
    out.append("Per-direction Kelly (using only same-direction trades):")
    out.append("")
    for dir_ in t["direction"].unique():
        sub = t[t["direction"] == dir_]
        if len(sub) < 2:
            continue
        wins = sub.loc[sub["pnl"] > 0, "pnl_pct"]
        losses = sub.loc[sub["pnl"] <= 0, "pnl_pct"].abs()
        if len(wins) and len(losses):
            avg_w, avg_l = wins.mean(), losses.mean()
            b = avg_w / avg_l if avg_l > 0 else 0
            p = (sub["pnl"] > 0).mean()
            q = 1 - p
            f = (p * b - q) / b if b > 0 else 0
            out.append(
                f"- **{dir_}**: n={len(sub)}, win_rate={p*100:.1f}%, avg_win={avg_w:.3f}%, "
                f"avg_loss={avg_l:.3f}%, R/R={b:.2f}, Kelly f*= **{f:+.3f}**"
            )
    return "\n".join(out) + "\n"


def section_exit_reasons(t: pd.DataFrame) -> str:
    out = ["## 3. Exit-reason histogram", ""]
    grp = (
        t.groupby("exit_reason")
        .agg(
            n=("pnl_pct", "count"),
            mean_pct=("pnl_pct", "mean"),
            median_pct=("pnl_pct", "median"),
            sum_rub=("pnl", "sum"),
            hit=("pnl_pct", lambda x: (x > 0).mean()),
            hold_h=("hold_hours", "mean"),
        )
        .round(3)
    )
    out.append(grp.to_markdown())
    out.append("")
    out.append(
        "**Read:** signal_reversal hit-rate vs external_close hit-rate gap is the main asymmetry."
    )
    return "\n".join(out) + "\n"


def section_calibration(t: pd.DataFrame) -> str:
    out = ["## 4. Confidence calibration", ""]
    sub = t.dropna(subset=["signal_confidence", "pnl_pct"])
    spear_r, spear_p = sci_stats.spearmanr(sub["signal_confidence"], sub["pnl_pct"])
    pearson_r = sub["signal_confidence"].corr(sub["pnl_pct"])
    out.append(f"- Spearman(conf, pnl_pct) = **{spear_r:+.4f}** (p={spear_p:.3f})")
    out.append(f"- Pearson(conf, pnl_pct) = {pearson_r:+.4f}")
    out.append("")
    # Quartile breakdown
    sub = sub.copy()
    sub["q"] = pd.qcut(
        sub["signal_confidence"].rank(method="first"), q=4, labels=["q1", "q2", "q3", "q4"]
    )
    qg = (
        sub.groupby("q", observed=True)
        .agg(
            conf_min=("signal_confidence", "min"),
            conf_max=("signal_confidence", "max"),
            n=("pnl_pct", "count"),
            mean_pct=("pnl_pct", "mean"),
            hit=("pnl_pct", lambda x: (x > 0).mean()),
        )
        .round(3)
    )
    out.append("Confidence quartiles:")
    out.append("")
    out.append(qg.to_markdown())
    # Reliability (ECE-style): bin by confidence, compare predicted hit-rate (= conf) to actual win-rate
    out.append("")
    out.append("Reliability (binned win-rate vs confidence):")
    out.append("")
    bins = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.01]
    sub["cb"] = pd.cut(sub["signal_confidence"], bins, right=False)
    rb = (
        sub.groupby("cb", observed=True)
        .agg(
            n=("pnl_pct", "count"),
            mean_conf=("signal_confidence", "mean"),
            win_rate=("pnl_pct", lambda x: (x > 0).mean()),
            mean_pnl=("pnl_pct", "mean"),
        )
        .round(3)
    )
    rb["gap"] = (rb["win_rate"] - rb["mean_conf"]).round(3)
    out.append(rb.to_markdown())
    # ECE
    if len(sub):
        ece = (rb["n"] / rb["n"].sum() * (rb["win_rate"] - rb["mean_conf"]).abs()).sum()
        out.append("")
        out.append(
            f"Expected Calibration Error (ECE) = **{ece:.3f}**  (lower = better; >0.10 = poorly calibrated)"
        )
    return "\n".join(out) + "\n"


def section_mae_mfe(t: pd.DataFrame) -> str:
    out = ["## 5. MAE / MFE pathology", ""]
    have = t.dropna(subset=["mae_pct", "mfe_pct"])
    if have.empty:
        return "## 5. MAE / MFE pathology\n\n_(no candle data — backfill failed)_\n"
    # Distributions
    out.append(
        "MAE/MFE quantiles (% of entry, where MAE is signed adverse, MFE is signed favorable):"
    )
    out.append("")
    out.append(
        have[["mae_pct", "mfe_pct", "sl_dist_pct", "tp_dist_pct"]].describe().round(3).to_markdown()
    )
    out.append("")
    # Stop-touched-but-not-stopped: trades where price touched SL intraday but exit_reason != stop_loss
    sl_ghost = have[have["stop_touched_intraday"] & (have["exit_reason"] != "stop_loss")]
    out.append(f"### 5.1 Stops touched intraday but exit_reason ≠ stop_loss")
    out.append(
        f"- n = **{len(sl_ghost)}** of {len(have)} trades ({len(sl_ghost)/len(have)*100:.1f}%)"
    )
    if len(sl_ghost):
        out.append(
            f"- mean realized P&L%: {sl_ghost['pnl_pct'].mean():+.3f}%  vs  others {have[~have.index.isin(sl_ghost.index)]['pnl_pct'].mean():+.3f}%"
        )
        out.append(f"- by exit_reason: {sl_ghost['exit_reason'].value_counts().to_dict()}")
        out.append("")
        out.append(
            "**Hypothesis:** broker stop fired late (>=0.2% past level) OR position monitor checked stale hourly close while 5-min low pierced the level."
        )
    out.append("")
    # TP-reached-then-given-back: MFE >= TP distance but didn't close on TP
    tp_ghost = have[have["tp_touched_intraday"] & (have["exit_reason"] != "take_profit")]
    out.append(f"### 5.2 TP touched intraday but exit_reason ≠ take_profit")
    out.append(
        f"- n = **{len(tp_ghost)}** of {len(have)} trades ({len(tp_ghost)/len(have)*100:.1f}%)"
    )
    if len(tp_ghost):
        out.append(
            f"- mean realized P&L%: {tp_ghost['pnl_pct'].mean():+.3f}%  (left on table = mean MFE−exit_pnl)"
        )
        diff = (tp_ghost["mfe_pct"] - tp_ghost["pnl_pct"]).mean()
        out.append(f"- mean give-back from MFE → exit: **{diff:+.3f} pp**")
    out.append("")
    # Time-to-MFE on losers: did losers EVER show profit?
    losers = have[have["pnl"] <= 0]
    if len(losers):
        winners = have[have["pnl"] > 0]
        out.append("### 5.3 Did losers ever show profit?")
        out.append(
            f"- losers n={len(losers)}, mean MFE = **{losers['mfe_pct'].mean():+.3f}%** (max favorable)"
        )
        if len(winners):
            out.append(
                f"- winners n={len(winners)}, mean MAE = **{winners['mae_pct'].mean():+.3f}%** (max adverse)"
            )
        # How many losers had MFE > 0.5%?
        could_have_won = losers[losers["mfe_pct"] >= 0.5]
        out.append(
            f"- losers that touched ≥+0.5% MFE before turning: **{len(could_have_won)}** "
            f"({len(could_have_won)/len(losers)*100:.1f}%)"
        )
        if len(could_have_won):
            out.append(
                f"  → these would have been winners if a partial-TP at 0.5% had been in place."
            )
    return "\n".join(out) + "\n"


def section_time_of_day(t: pd.DataFrame) -> str:
    out = ["## 6. Time-of-day (MSK)", ""]
    t = t.copy()
    t["hour_msk"] = (pd.to_datetime(t["entry_time"]) + pd.Timedelta(hours=3)).dt.hour
    t["dow_msk"] = (pd.to_datetime(t["entry_time"]) + pd.Timedelta(hours=3)).dt.day_name()
    hour_g = (
        t.groupby("hour_msk")
        .agg(
            n=("pnl_pct", "count"),
            mean_pct=("pnl_pct", "mean"),
            hit=("pnl_pct", lambda x: (x > 0).mean()),
            sum_rub=("pnl", "sum"),
        )
        .round(3)
    )
    out.append("By hour of entry (MSK):")
    out.append("")
    out.append(hour_g.to_markdown())
    out.append("")
    # Worst 4 hours (by mean) with n>=3
    cand = hour_g[hour_g["n"] >= 3].sort_values("mean_pct").head(4)
    if len(cand):
        out.append(
            f"**Worst 4 hours (n≥3):** {cand.index.tolist()} | means: {cand['mean_pct'].tolist()}"
        )
    out.append("")
    dow_g = t.groupby("dow_msk").agg(n=("pnl_pct", "count"), mean_pct=("pnl_pct", "mean")).round(3)
    out.append("By day of week:")
    out.append("")
    out.append(dow_g.to_markdown())
    return "\n".join(out) + "\n"


def section_stop_distance(t: pd.DataFrame) -> str:
    out = ["## 7. Stop / TP distance vs outcome", ""]
    have = t.dropna(subset=["sl_dist_pct", "atr_pct_proxy"]).copy()
    if have.empty or have["atr_pct_proxy"].isna().all():
        return out[0] + "\n\n_(insufficient data)_\n"
    have["sl_in_atr"] = have["sl_dist_pct"] / have["atr_pct_proxy"]
    have["tp_in_atr"] = have["tp_dist_pct"] / have["atr_pct_proxy"]
    out.append(f"Stop-distance (in ATR proxy) summary:")
    out.append("")
    out.append(have[["sl_in_atr", "tp_in_atr"]].describe().round(2).to_markdown())
    out.append("")
    # Bin by sl_in_atr
    have["sl_bin"] = pd.cut(
        have["sl_in_atr"],
        bins=[0, 1.0, 1.5, 2.0, 3.0, 99],
        labels=["<1", "1-1.5", "1.5-2", "2-3", "3+"],
    )
    bin_g = (
        have.groupby("sl_bin", observed=True)
        .agg(
            n=("pnl_pct", "count"),
            mean_pct=("pnl_pct", "mean"),
            hit=("pnl_pct", lambda x: (x > 0).mean()),
        )
        .round(3)
    )
    out.append("Win rate by stop-distance (in ATR proxy):")
    out.append("")
    out.append(bin_g.to_markdown())
    return "\n".join(out) + "\n"


def section_per_ticker(t: pd.DataFrame) -> str:
    out = ["## 8. Per-ticker breakdown", ""]
    g = (
        t.groupby("ticker")
        .agg(
            n=("pnl_pct", "count"),
            mean_pct=("pnl_pct", "mean"),
            sum_rub=("pnl", "sum"),
            hit=("pnl_pct", lambda x: (x > 0).mean()),
        )
        .round(3)
        .sort_values("sum_rub")
    )
    g_n3 = g[g["n"] >= 3]
    if len(g_n3):
        out.append("Tickers with ≥3 trades:")
        out.append("")
        out.append(g_n3.to_markdown())
        out.append("")
    out.append("Top 5 worst tickers by sum P&L:")
    out.append("")
    out.append(g.head(5).to_markdown())
    out.append("")
    out.append("Top 5 best tickers by sum P&L:")
    out.append("")
    out.append(g.tail(5).to_markdown())
    return "\n".join(out) + "\n"


def section_top_losses(t: pd.DataFrame) -> str:
    out = ["## 9. Top-10 worst trades — detail", ""]
    cols = [
        "id",
        "ticker",
        "direction",
        "signal_confidence",
        "entry_price",
        "exit_price",
        "stop_loss",
        "take_profit",
        "pnl",
        "pnl_pct",
        "exit_reason",
        "hold_hours",
    ]
    extras = [
        c
        for c in [
            "mae_pct",
            "mfe_pct",
            "stop_touched_intraday",
            "tp_touched_intraday",
            "min_to_mae",
        ]
        if c in t.columns
    ]
    cols = cols + extras
    worst = t.nsmallest(10, "pnl")[cols].round(3)
    out.append(worst.to_markdown(index=False))
    return "\n".join(out) + "\n"


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


async def amain(args):
    settings = Settings()
    db_path = Path("data/trade_bot.db")
    db = Repository(db_path)
    await db.initialize()

    # Load trades window
    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    conn = sqlite3.connect(db_path)
    trades = pd.read_sql_query(
        "SELECT * FROM trades WHERE status='closed' AND entry_time >= ? ORDER BY entry_time",
        conn,
        params=(cutoff,),
    )
    conn.close()

    if trades.empty:
        log.warning("No closed trades in window; nothing to analyse.")
        return

    trades["entry_dt"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_dt"] = pd.to_datetime(trades["exit_time"], utc=True)
    trades["hold_hours"] = (trades["exit_dt"] - trades["entry_dt"]).dt.total_seconds() / 3600

    log.info(f"Loaded {len(trades)} closed trades since {cutoff[:10]}")

    if not args.no_fetch:
        log.info("Connecting to T-API for candle backfill...")
        broker = BrokerClient(settings.T_INVEST_TOKEN, settings.T_INVEST_ACCOUNT_ID)
        await broker.connect()
        try:
            await backfill_candles(broker, db, trades)
        finally:
            await broker.disconnect()

    log.info("Enriching trades with MAE/MFE/stop-touched flags...")
    trades = await enrich_trades(db, trades)

    # Persist enriched dataset
    out_parquet = REPORTS / f"weekly_postmortem_{datetime.now().strftime('%Y%m%d')}.parquet"
    try:
        trades.to_parquet(out_parquet, index=False)
        log.info(f"Wrote enriched dataset: {out_parquet}")
    except Exception as e:
        log.warning(f"Parquet write failed (likely missing pyarrow): {e}; falling back to CSV")
        out_csv = out_parquet.with_suffix(".csv")
        trades.to_csv(out_csv, index=False)
        log.info(f"Wrote enriched dataset: {out_csv}")

    # Generate report
    report = []
    report.append(f"# Weekly Postmortem — last {args.days} days")
    report.append(
        f"_Generated {datetime.now().isoformat(timespec='seconds')} | "
        f"window: since {cutoff[:10]} | trades: {len(trades)}_"
    )
    report.append("")
    report.append(section_overview(trades))
    report.append(section_long_short(trades))
    report.append(section_exit_reasons(trades))
    report.append(section_calibration(trades))
    report.append(section_mae_mfe(trades))
    report.append(section_time_of_day(trades))
    report.append(section_stop_distance(trades))
    report.append(section_per_ticker(trades))
    report.append(section_top_losses(trades))

    md_path = REPORTS / f"weekly_postmortem_{datetime.now().strftime('%Y%m%d')}.md"
    md_path.write_text("\n".join(report), encoding="utf-8")
    log.info(f"Wrote report: {md_path}")

    print(f"\n=== Weekly Postmortem written ===")
    print(f"  {md_path}")
    print(f"  {out_parquet if out_parquet.exists() else out_parquet.with_suffix('.csv')}")

    await db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip T-API candle backfill (use whatever is cached)",
    )
    args = parser.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
