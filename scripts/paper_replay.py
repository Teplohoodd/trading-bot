"""Paper-replay: estimate the P&L impact of each fix on last week's trades.

Reads the enriched dataset written by scripts.weekly_postmortem (parquet),
applies a single counterfactual at a time, and prints a comparison table.

Counterfactuals:
  * FIX 1  — replace exit with stop_loss for trades whose 5m-low/high
             touched the SL while exit_reason != stop_loss.
  * FIX 3  — drop long entries with confidence < LONG_MIN_CONFIDENCE_WHEN_BAD_EDGE
             (effectively the long-pause filter).
  * FIX 4  — partial-TP at MFE ≥ 1.5 × atr_pct_proxy on 50% of position;
             remainder uses original exit price.
  * FIX 6  — drop entries in skipped hours [10, 12, 17, 18].
  * FIX 9  — drop entries that were the 4th+ same-side loss in 4h window.

These are *post-hoc* estimates — slippage, partial-fill execution risk and
intra-bar dynamics are simplifying assumptions.  Treat as a directional
sanity check, not a backtest.
"""

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
from tabulate import tabulate

REPORTS = Path("analysis/reports")


def load_dataset(date_tag: str | None = None) -> pd.DataFrame:
    if date_tag:
        candidates = list(REPORTS.glob(f"weekly_postmortem_{date_tag}.*"))
    else:
        candidates = sorted(REPORTS.glob("weekly_postmortem_*.parquet"), reverse=True)
        if not candidates:
            candidates = sorted(REPORTS.glob("weekly_postmortem_*.csv"), reverse=True)
    if not candidates:
        print("No postmortem dataset found. Run scripts.weekly_postmortem first.")
        sys.exit(1)
    p = candidates[0]
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    print(f"Loaded {len(df)} trades from {p}")
    return df


def baseline(t: pd.DataFrame) -> dict:
    return {
        "label": "baseline",
        "n": len(t),
        "win_rate": float((t["pnl"] > 0).mean()),
        "mean_pnl_pct": float(t["pnl_pct"].mean()),
        "sum_pnl_rub": float(t["pnl"].sum()),
    }


def fix1_realtime_stop(t: pd.DataFrame) -> pd.DataFrame:
    """Trades whose 5m-low/high touched SL but exit_reason != stop_loss
    are recomputed as if they exited at the stop level."""
    t = t.copy()
    if "stop_touched_intraday" not in t.columns:
        return t
    mask = t["stop_touched_intraday"] & (t["exit_reason"] != "stop_loss")
    for idx in t.index[mask]:
        row = t.loc[idx]
        sl = row["stop_loss"]
        if pd.isna(sl) or sl <= 0:
            continue
        entry = row["entry_price"]
        lots = row["lots"]
        lot_size = row.get("lot_size", 1) or 1
        if row["direction"] == "buy":
            new_pnl_pct = (sl - entry) / entry * 100
            new_pnl = (sl - entry) * lots * lot_size
        else:
            new_pnl_pct = (entry - sl) / entry * 100
            new_pnl = (entry - sl) * lots * lot_size
        # subtract a reasonable round-trip commission (5bps each side ≈ 0.1%)
        new_pnl -= abs(entry * lots * lot_size) * 0.0010
        t.at[idx, "pnl"] = new_pnl
        t.at[idx, "pnl_pct"] = new_pnl_pct
        t.at[idx, "exit_reason"] = "stop_loss"
    return t


def fix3_long_pause(t: pd.DataFrame, min_conf: float = 0.85) -> pd.DataFrame:
    """Drop long entries below the long-pause confidence floor."""
    keep = ~((t["direction"] == "buy") & (t["signal_confidence"] < min_conf))
    return t[keep].copy()


def fix4_partial_tp(t: pd.DataFrame, trigger_mult: float = 1.5, frac: float = 0.5) -> pd.DataFrame:
    """Simulate 50% scale-out when MFE % >= trigger_mult * atr_pct_proxy.
    The remaining half exits at the original exit_price."""
    t = t.copy()
    if "mfe_pct" not in t.columns or "atr_pct_proxy" not in t.columns:
        return t
    for idx in t.index:
        row = t.loc[idx]
        if pd.isna(row["mfe_pct"]) or pd.isna(row["atr_pct_proxy"]):
            continue
        if row["lots"] < 2:
            continue
        trigger_pct = trigger_mult * row["atr_pct_proxy"]
        if row["mfe_pct"] >= trigger_pct:
            entry = row["entry_price"]
            lots = row["lots"]
            lot_size = row.get("lot_size", 1) or 1
            scale_lots = max(1, int(round(lots * frac)))
            scale_lots = min(scale_lots, lots - 1)
            runner_lots = lots - scale_lots
            # Partial filled at trigger price (entry + trigger_pct*entry/100)
            if row["direction"] == "buy":
                partial_price = entry * (1 + trigger_pct / 100)
                partial_pnl = (partial_price - entry) * scale_lots * lot_size
            else:
                partial_price = entry * (1 - trigger_pct / 100)
                partial_pnl = (entry - partial_price) * scale_lots * lot_size
            # Runner uses original exit
            exit_price = row["exit_price"]
            if row["direction"] == "buy":
                runner_pnl = (exit_price - entry) * runner_lots * lot_size
            else:
                runner_pnl = (entry - exit_price) * runner_lots * lot_size
            new_pnl = partial_pnl + runner_pnl
            new_pnl -= abs(entry * lots * lot_size) * 0.0010
            new_pnl_pct = (new_pnl / (entry * lots * lot_size)) * 100
            t.at[idx, "pnl"] = new_pnl
            t.at[idx, "pnl_pct"] = new_pnl_pct
    return t


def fix6_hour_filter(t: pd.DataFrame, skip_hours=(10, 12, 17, 18)) -> pd.DataFrame:
    t = t.copy()
    t["hour_msk"] = (pd.to_datetime(t["entry_time"]) + pd.Timedelta(hours=3)).dt.hour
    return t[~t["hour_msk"].isin(skip_hours)].copy()


def fix9_stoploss_guard(t: pd.DataFrame, count_thr: int = 3, lookback_h: int = 4) -> pd.DataFrame:
    """Drop entries that would have been blocked by StoplossGuard:
    when ≥count_thr same-side losing exits occurred in the prior lookback window."""
    t = t.copy().sort_values("entry_time").reset_index(drop=True)
    t["entry_dt"] = pd.to_datetime(t["entry_time"])
    t["exit_dt"] = pd.to_datetime(t["exit_time"])
    keep = []
    for i, row in t.iterrows():
        side = row["direction"]
        cutoff = row["entry_dt"] - pd.Timedelta(hours=lookback_h)
        prior_losses = t[
            (t["direction"] == side)
            & (t["pnl"] < 0)
            & (t["exit_dt"] < row["entry_dt"])
            & (t["exit_dt"] >= cutoff)
        ]
        keep.append(len(prior_losses) < count_thr)
    return t[pd.Series(keep)].copy()


def summarise(rows: list[dict]) -> str:
    return tabulate(rows, headers="keys", floatfmt=".3f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Postmortem date tag (default: latest)")
    args = ap.parse_args()

    t0 = load_dataset(args.date)
    rows = [baseline(t0)]
    rows[0]["delta_rub"] = 0.0

    # FIX 1
    t1 = fix1_realtime_stop(t0)
    r = baseline(t1)
    r["label"] = "+ FIX1 realtime SL"
    r["delta_rub"] = r["sum_pnl_rub"] - rows[0]["sum_pnl_rub"]
    rows.append(r)

    # FIX 3 (drop low-conf longs)
    t3 = fix3_long_pause(t0, min_conf=0.85)
    r = baseline(t3)
    r["label"] = "+ FIX3 long-pause @0.85"
    r["delta_rub"] = r["sum_pnl_rub"] - rows[0]["sum_pnl_rub"]
    rows.append(r)

    # FIX 4 (partial TP)
    t4 = fix4_partial_tp(t0, 1.5, 0.5)
    r = baseline(t4)
    r["label"] = "+ FIX4 partial TP@1.5atr"
    r["delta_rub"] = r["sum_pnl_rub"] - rows[0]["sum_pnl_rub"]
    rows.append(r)

    # FIX 6 (hour filter)
    t6 = fix6_hour_filter(t0, (10, 12, 17, 18))
    r = baseline(t6)
    r["label"] = "+ FIX6 hour filter"
    r["delta_rub"] = r["sum_pnl_rub"] - rows[0]["sum_pnl_rub"]
    rows.append(r)

    # FIX 9 (stoploss guard)
    t9 = fix9_stoploss_guard(t0, count_thr=3, lookback_h=4)
    r = baseline(t9)
    r["label"] = "+ FIX9 stop-guard"
    r["delta_rub"] = r["sum_pnl_rub"] - rows[0]["sum_pnl_rub"]
    rows.append(r)

    # Combined: FIX 1+3+6+9 (entry filters + realtime SL).  FIX 4 (partial TP)
    # operates on remaining trades inside the kept set.
    t_combo = t0.copy()
    t_combo = fix3_long_pause(t_combo, 0.85)
    t_combo = fix6_hour_filter(t_combo, (10, 12, 17, 18))
    t_combo = fix9_stoploss_guard(t_combo, 3, 4)
    t_combo = fix1_realtime_stop(t_combo)
    t_combo = fix4_partial_tp(t_combo, 1.5, 0.5)
    r = baseline(t_combo)
    r["label"] = "ALL fixes combined"
    r["delta_rub"] = r["sum_pnl_rub"] - rows[0]["sum_pnl_rub"]
    rows.append(r)

    print()
    print(summarise(rows))
    print()
    print("(delta_rub = sum_pnl_rub - baseline sum_pnl_rub; positive = improvement)")
    print()
    print("Caveats: post-hoc estimate; assumes 0.10% round-trip commission for")
    print("re-priced exits, no slippage on partial fills, no intra-bar dynamics.")


if __name__ == "__main__":
    main()
