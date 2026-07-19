"""Pair-trading grid search across holding horizons and entry thresholds.

For every cointegrated FORTS pair, runs a 2D grid:
    max_hold_hours  ∈ {3, 5, 8, 12, 24, 48, 96, 168, None=dynamic-only}
    z_entry         ∈ {1.5, 2.0, 2.5}

Strategy per cell:
    1. Compute spread = price_y − β × price_x  on hourly bars
    2. z = (spread − mean) / std  on a rolling window
    3. Enter when |z| > z_entry  (long-y / short-βx if z<0, mirror if z>0)
    4. Exit when:
        a. z crosses 0 (mean reversion done)             ← classic exit
        b. holding ≥ max_hold_hours                       ← horizon cap
        c. |z| > 4.0 (structural break, give up)          ← stop-loss
    Whichever happens first.

Reports per pair:
    * grid of total_pnl_pct
    * best (max_hold, z_entry) combo by total_pnl_pct AND by per-trade Sharpe
    * annualised return at the best combo
    * sample size warning when n_trades < 5

Costs:
    Round-trip commission 0.16 % (2 legs × 2 sides × 0.04 %) subtracted
    from each trade's gross P&L.  Slippage NOT modelled — at most-liquid
    futures with limit-aggressive execution it's ~0.5 tick per leg ≈ tiny
    on 0.x% spread moves.

Usage:
    python -m futbot.scripts.pairs_grid_search                # default 180d
    python -m futbot.scripts.pairs_grid_search 365            # 365d if available
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("t_tech").setLevel(logging.WARNING)
logger = logging.getLogger("pairs_grid")


BASES = ["BR", "GZ", "SR", "LK", "MX", "Si"]

# Grid dimensions
HORIZONS_H = [3, 5, 8, 12, 24, 48, 96, 168, None]  # None = dynamic-only exit
Z_ENTRIES = [1.5, 2.0, 2.5]
Z_STOP = 4.0  # structural-break SL
COMMISSION_RT = 0.0016  # 0.16% per pair round-trip

# Cointegration prefilter
ADF_P_MAX = 0.10  # include pairs with p ≤ this

# Spread normalisation window (rolling) — for z-score stability over long periods
ROLLING_Z_WINDOW = 240  # ~10 days of hourly bars


# ─────────────────────────────────────────────────────────────────────────────
# Data fetch (same as edge_backtest)
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
            "close": float(quotation_to_decimal(c.close)),
        }
        for c in candles
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


async def _fetch_all(broker, days: int) -> pd.DataFrame:
    series_list = []
    now = datetime.now(timezone.utc)
    for base in BASES:
        f = await _resolve_front_month(broker, base)
        if f is None:
            logger.warning(f"  {base}: not found")
            continue
        try:
            candles = await broker.get_candles(
                f.figi,
                now - timedelta(days=days),
                now,
                interval=CandleInterval.CANDLE_INTERVAL_HOUR,
            )
            df = _candles_to_df(candles)
            if len(df) > 100:
                s = df.set_index("time")["close"].rename(base)
                # Tinkoff candle history occasionally returns duplicate bars
                # near session boundaries — keep last to avoid reindex error.
                s = s[~s.index.duplicated(keep="last")]
                series_list.append(s)
                logger.info(f"  {base} ({f.ticker}): {len(df)} bars (uniq {len(s)})")
            else:
                logger.warning(f"  {base}: only {len(df)} bars")
        except Exception as e:
            logger.warning(f"  {base}: {e}")
    if not series_list:
        return pd.DataFrame()
    return pd.concat(series_list, axis=1, join="inner").dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Cointegration prefilter
# ─────────────────────────────────────────────────────────────────────────────
def find_cointegrated_pairs(prices: pd.DataFrame) -> list[dict]:
    from statsmodels.tsa.stattools import adfuller

    bases = list(prices.columns)
    out = []
    for i, a in enumerate(bases):
        for b in bases[i + 1 :]:
            y = prices[a].values
            x = prices[b].values
            beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
            alpha = y.mean() - beta * x.mean()
            resid = y - (beta * x + alpha)
            try:
                pval = float(adfuller(resid, maxlag=5, autolag=None)[1])
            except Exception:
                continue
            if pval <= ADF_P_MAX:
                out.append({"pair": (a, b), "beta": beta, "alpha": alpha, "adf_p": pval})
    out.sort(key=lambda r: r["adf_p"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Backtest one (pair, horizon, z_entry) cell
# ─────────────────────────────────────────────────────────────────────────────
def backtest_pair(
    prices: pd.DataFrame,
    *,
    a: str,
    b: str,
    beta: float,
    z_entry: float,
    max_hold_hours: int | None,
    z_stop: float = Z_STOP,
    commission_rt: float = COMMISSION_RT,
) -> dict:
    y = prices[a].values
    x = prices[b].values
    spread = y - beta * x
    n = len(spread)
    if n < ROLLING_Z_WINDOW + 50:
        return {"n_trades": 0, "error": "too few bars"}

    # Rolling z-score: at each t, use mean/std of spread[t-W:t]
    z = np.zeros(n)
    z[:ROLLING_Z_WINDOW] = np.nan
    for t in range(ROLLING_Z_WINDOW, n):
        w = spread[t - ROLLING_Z_WINDOW : t]
        m = w.mean()
        s = w.std()
        z[t] = (spread[t] - m) / s if s > 0 else 0.0

    # Trade loop
    pos = 0  # +1 = long-y short-βx (z<0 entry); -1 = mirror
    entry_idx = None
    pnls = []
    holding = []
    stops = 0
    horizon_exits = 0
    mean_rev_exits = 0
    for t in range(ROLLING_Z_WINDOW, n):
        if np.isnan(z[t]):
            continue
        if pos == 0:
            if z[t] > z_entry:
                pos, entry_idx = -1, t
            elif z[t] < -z_entry:
                pos, entry_idx = +1, t
            continue

        # In a position — check exits
        held_h = t - entry_idx
        exit_reason = None
        # 1. Structural break
        if abs(z[t]) >= z_stop:
            exit_reason = "stop"
            stops += 1
        # 2. Time cap
        elif max_hold_hours is not None and held_h >= max_hold_hours:
            exit_reason = "horizon"
            horizon_exits += 1
        # 3. Mean reversion (z crossed 0)
        elif (pos == +1 and z[t] >= 0) or (pos == -1 and z[t] <= 0):
            exit_reason = "mean_rev"
            mean_rev_exits += 1

        if exit_reason is None:
            continue

        sp_entry = y[entry_idx] - beta * x[entry_idx]
        sp_exit = y[t] - beta * x[t]
        combined_notional = abs(y[entry_idx]) + abs(beta) * abs(x[entry_idx])
        gross = pos * (sp_exit - sp_entry) / combined_notional if combined_notional > 0 else 0
        net = gross - commission_rt
        pnls.append(net)
        holding.append(held_h)
        pos, entry_idx = 0, None

    if not pnls:
        return {"n_trades": 0}

    pnls_arr = np.array(pnls)
    holding_arr = np.array(holding)
    sharpe = (
        pnls_arr.mean() / pnls_arr.std() * math.sqrt(len(pnls_arr)) if pnls_arr.std() > 0 else 0.0
    )
    # Drawdown: cum returns
    cum = np.cumsum(pnls_arr)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak).min()
    return {
        "n_trades": int(len(pnls_arr)),
        "win_rate": round(float((pnls_arr > 0).mean()), 3),
        "avg_pct": round(float(pnls_arr.mean()) * 100, 4),
        "total_pct": round(float(pnls_arr.sum()) * 100, 3),
        "best_pct": round(float(pnls_arr.max()) * 100, 3),
        "worst_pct": round(float(pnls_arr.min()) * 100, 3),
        "avg_hold_h": round(float(holding_arr.mean()), 1),
        "median_hold_h": round(float(np.median(holding_arr)), 1),
        "sharpe": round(float(sharpe), 2),
        "max_dd_pct": round(float(dd) * 100, 3),
        "exits": {
            "mean_rev": mean_rev_exits,
            "horizon": horizon_exits,
            "stop": stops,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    settings = Settings()
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="pairs-grid",
    )
    await broker.connect()
    logger.info(f"Fetching {days}d of hourly candles for {BASES}…")
    prices = await _fetch_all(broker, days)
    await broker.disconnect()

    if prices.empty:
        logger.error("no data")
        return

    logger.info(
        f"Aligned bars: {len(prices)}  span: " f"{prices.index.min()} → {prices.index.max()}"
    )

    coint = find_cointegrated_pairs(prices)
    if not coint:
        print("No cointegrated pairs in this period.  Try a longer window.")
        return
    print()
    print(f"Found {len(coint)} cointegrated pairs (adf_p ≤ {ADF_P_MAX}):")
    for c in coint:
        a, b = c["pair"]
        print(f"  {a:>3}–{b:<3}  β={c['beta']:+.4f}  adf_p={c['adf_p']:.4f}")

    # ── Grid search ──
    print()
    print("=" * 120)
    print(
        f"GRID SEARCH — {days}d hourly bars, commission {COMMISSION_RT*100:.2f}% RT, "
        f"z_stop={Z_STOP}, rolling-z window={ROLLING_Z_WINDOW}h"
    )
    print("=" * 120)
    summary = []  # (pair, best_horizon, best_z, metrics)
    for c in coint:
        a, b = c["pair"]
        beta = c["beta"]
        print(f"\n── Pair {a}–{b}  (β={beta:+.4f}, adf_p={c['adf_p']:.4f}) ──")
        header_hz = [f"{h}h" if h is not None else "dyn" for h in HORIZONS_H]
        # Print total_pct grid
        print(f"  total_pct × n_trades grid:")
        print(f"  {'z_entry':>8} | " + " | ".join(f"{h:>11}" for h in header_hz))
        rows = []
        per_pair_cells = []
        for z_ent in Z_ENTRIES:
            cells = []
            for hz in HORIZONS_H:
                r = backtest_pair(prices, a=a, b=b, beta=beta, z_entry=z_ent, max_hold_hours=hz)
                cells.append(r)
                per_pair_cells.append(
                    {
                        "z_entry": z_ent,
                        "horizon_h": hz,
                        **r,
                    }
                )
            tot_strs = []
            for cell in cells:
                if cell.get("n_trades", 0) == 0:
                    tot_strs.append("    .       ")
                else:
                    tot_strs.append(f"{cell['total_pct']:+6.2f}%/{cell['n_trades']:>2}".rjust(11))
            rows.append((z_ent, tot_strs))
            print(f"  {z_ent:>8.1f} | " + " | ".join(tot_strs))

        # Pick best cell by sharpe with min sample size 4
        scored = [c for c in per_pair_cells if c.get("n_trades", 0) >= 4]
        if not scored:
            # Fallback: any cell with > 0 trades
            scored = [c for c in per_pair_cells if c.get("n_trades", 0) > 0]
        if scored:
            best = max(scored, key=lambda c: c.get("sharpe", -999))
            hz_label = f"{best['horizon_h']}h" if best["horizon_h"] is not None else "dynamic"
            ann = best["total_pct"] * 365 / days
            print(
                f"  → Best by Sharpe: z={best['z_entry']:.1f}, horizon={hz_label}, "
                f"n={best['n_trades']}, win={best['win_rate']*100:.0f}%, "
                f"total={best['total_pct']:+.2f}%, sharpe={best['sharpe']:+.2f}, "
                f"hold-median={best['median_hold_h']:.0f}h, DD={best['max_dd_pct']:+.2f}%, "
                f"ann≈{ann:+.1f}%"
            )
            print(
                f"  → Exits: mean-rev={best['exits']['mean_rev']}  "
                f"horizon={best['exits']['horizon']}  stop={best['exits']['stop']}"
            )
            summary.append(
                {
                    "pair": f"{a}-{b}",
                    "best": best,
                    "ann_pct": ann,
                }
            )

    # ── Top-line summary ──
    if summary:
        print()
        print("=" * 120)
        print("BEST COMBO PER PAIR (sorted by annualised return)")
        print("=" * 120)
        summary.sort(key=lambda s: -s["ann_pct"])
        print(
            f"{'pair':<8} {'z_entry':>7} {'horizon':>9} {'n':>4} {'win%':>6} "
            f"{'total%':>8} {'sharpe':>7} {'medHold':>8} {'DD%':>7} {'ann%':>7}"
        )
        for s in summary:
            b = s["best"]
            hz = f"{b['horizon_h']}h" if b["horizon_h"] is not None else "dynamic"
            print(
                f"{s['pair']:<8} {b['z_entry']:>7.1f} {hz:>9} {b['n_trades']:>4} "
                f"{b['win_rate']*100:>5.1f}% {b['total_pct']:>+8.2f} "
                f"{b['sharpe']:>+7.2f} {b['median_hold_h']:>7.0f}h "
                f"{b['max_dd_pct']:>+7.2f} {s['ann_pct']:>+7.1f}"
            )

        print()
        print("Reading the table:")
        print("  • total% = net return after 0.16% round-trip × n_trades")
        print("  • ann% = total% × 365/days (rough annualised projection)")
        print("  • sharpe = mean/std × √n (per-trade scaled, not annualised)")
        print("  • DD% = worst peak-to-trough on the equity curve")
        print("  • n < 5 trades = LOW CONFIDENCE — don't over-interpret")


if __name__ == "__main__":
    asyncio.run(main())
