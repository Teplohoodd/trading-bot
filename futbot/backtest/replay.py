"""Walk-forward replay of the 4-layer pipeline on historical FORTS bars.

For each "evaluation point" t in the historical window:
  1. Reconstruct what the pipeline WOULD have seen at time t (slice each TF
     DataFrame to bars ≤ t).
  2. Run the same `decision.evaluate_contract()` used live.
  3. If approved, simulate a fixed-1-lot position with the same Chandelier
     trailing stop logic, exiting on stop / max-hold / signal flip.
  4. Record the trade and continue.

Output: a DataFrame of simulated trades + aggregate metrics
(Sharpe, profit factor, max-DD, win rate, n trades).

This is conservative: it reuses the SAME code paths as live.  No separate
"backtest engine" that could diverge from production behaviour — the only
difference is that orders are simulated and TF slices are explicit.
"""

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd

from futbot.pipeline.decision import evaluate_contract
from futbot.execution import stops as stops_module
from futbot.config import FutSettings
from futbot.utils import commissions as comm

logger = logging.getLogger("futbot.backtest")


@dataclass
class SimTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    ticker: str
    direction: str
    entry: float
    exit: float
    bars_held: int
    pnl_pct: float
    reason: str


def _slice_to(df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """All rows with time ≤ ts.  Returns empty df if nothing matches."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df["time"] <= ts].reset_index(drop=True)


def _atr_1h_at(df_1h: pd.DataFrame, n: int = 14) -> float:
    if df_1h is None or df_1h.empty or len(df_1h) < n + 1:
        return 0.0
    h, l, c = df_1h["high"], df_1h["low"], df_1h["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    return float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0


async def backtest_contract(
    *,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    ticker: str,
    base_ticker: str | None,
    settings: FutSettings,
    figi: str = "BACKTEST",
    eval_every_bars: int = 1,
    instrument_kind: str = "future",
) -> tuple[list[SimTrade], dict]:
    """Run the pipeline on every (or every N) 5m bar.  Returns trades + metrics.

    eval_every_bars=1 → score every 5m bar (slow but most accurate).
    For long historical windows, 6 = once an hour is a reasonable compromise.
    """
    trades: list[SimTrade] = []
    open_pos = None  # dict with direction, entry, entry_idx, stop_state

    # We walk the 5m index (most granular) and at each step build the
    # snapshot for each TF.
    if df_5m is None or df_5m.empty:
        return [], {}

    df_5m = df_5m.copy().sort_values("time").reset_index(drop=True)
    df_15m = (
        df_15m.copy().sort_values("time").reset_index(drop=True)
        if df_15m is not None
        else pd.DataFrame()
    )
    df_1h = (
        df_1h.copy().sort_values("time").reset_index(drop=True)
        if df_1h is not None
        else pd.DataFrame()
    )

    for i in range(0, len(df_5m), eval_every_bars):
        ts = df_5m.iloc[i]["time"]
        bar = df_5m.iloc[i]
        last_p = float(bar["close"])

        # 1) Manage open position first
        if open_pos is not None:
            atr_1h = _atr_1h_at(_slice_to(df_1h, ts))
            new_st = stops_module.update(
                state=open_pos["stop_state"],
                last_price=last_p,
                atr_1h=atr_1h,
                settings=settings,
            )
            open_pos["stop_state"] = new_st

            stopped = stops_module.is_stopped_out(
                state=new_st,
                bar_high=float(bar["high"]),
                bar_low=float(bar["low"]),
            )
            held_bars = i - open_pos["entry_idx"]
            held_hours = held_bars * 5 / 60  # 5-min bars
            timed_out = held_hours >= settings.FUTBOT_MAX_HOLD_HOURS

            if stopped or timed_out:
                # Exit at the stop price (if stopped) or current close (if timed-out)
                if stopped:
                    exit_price = new_st.current_stop
                    reason = "trailing_stop"
                else:
                    exit_price = last_p
                    reason = "time_cap"
                gross = (
                    (exit_price - open_pos["entry"]) / open_pos["entry"]
                    if open_pos["direction"] == "buy"
                    else (open_pos["entry"] - exit_price) / open_pos["entry"]
                )
                net = gross - 2 * comm.commission_pct(instrument_kind)
                trades.append(
                    SimTrade(
                        entry_time=open_pos["entry_time"],
                        exit_time=ts,
                        ticker=ticker,
                        direction=open_pos["direction"],
                        entry=open_pos["entry"],
                        exit=exit_price,
                        bars_held=held_bars,
                        pnl_pct=net * 100,
                        reason=reason,
                    )
                )
                open_pos = None

        # 2) Look for a new entry if we're flat
        if open_pos is None:
            # Build TF snapshots up to `ts` (inclusive).
            snap_1h = _slice_to(df_1h, ts)
            snap_15m = _slice_to(df_15m, ts)
            snap_5m = df_5m.iloc[: i + 1].reset_index(drop=True)
            if len(snap_1h) < 60 or len(snap_15m) < 30 or len(snap_5m) < 21:
                continue
            dec = await evaluate_contract(
                figi=figi,
                ticker=ticker,
                tf_data={"1h": snap_1h, "15m": snap_15m, "5m": snap_5m},
                settings=settings,
                base_ticker=base_ticker,
            )
            if not dec.approved:
                continue
            atr_1h = _atr_1h_at(snap_1h)
            if atr_1h <= 0:
                continue
            st = stops_module.initial_state(
                direction=dec.direction,
                entry=last_p,
                atr_1h=atr_1h,
                settings=settings,
            )
            open_pos = {
                "direction": dec.direction,
                "entry": last_p,
                "entry_time": ts,
                "entry_idx": i,
                "stop_state": st,
            }

    # Close any still-open position at the last bar
    if open_pos is not None:
        bar = df_5m.iloc[-1]
        gross = (
            (float(bar["close"]) - open_pos["entry"]) / open_pos["entry"]
            if open_pos["direction"] == "buy"
            else (open_pos["entry"] - float(bar["close"])) / open_pos["entry"]
        )
        net = gross - 2 * comm.commission_pct(instrument_kind)
        trades.append(
            SimTrade(
                entry_time=open_pos["entry_time"],
                exit_time=bar["time"],
                ticker=ticker,
                direction=open_pos["direction"],
                entry=open_pos["entry"],
                exit=float(bar["close"]),
                bars_held=len(df_5m) - 1 - open_pos["entry_idx"],
                pnl_pct=net * 100,
                reason="backtest_end",
            )
        )

    metrics = _aggregate(trades)
    return trades, metrics


def _aggregate(trades: list[SimTrade]) -> dict:
    if not trades:
        return {"n_trades": 0}
    rets = pd.Series([t.pnl_pct for t in trades])
    eq = (1 + rets / 100).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    profit_factor = (
        (wins.sum() / -losses.sum()) if not losses.empty and losses.sum() < 0 else float("inf")
    )
    sharpe = rets.mean() / rets.std() * math.sqrt(252) if rets.std() > 0 else 0.0
    return {
        "n_trades": len(trades),
        "win_rate": round((rets > 0).mean() * 100, 1),
        "total_pct": round(float(rets.sum()), 2),
        "avg_pct": round(float(rets.mean()), 3),
        "profit_factor": round(float(profit_factor), 2),
        "sharpe": round(float(sharpe), 2),
        "max_dd_pct": round(float(dd), 2),
        "avg_bars_held": round(float(pd.Series([t.bars_held for t in trades]).mean()), 1),
    }
