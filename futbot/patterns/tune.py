"""Grid-search tuner for pattern detector params.

Approach:
  1. Cache OHLC for all WF contracts to parquet (once).
  2. Run a grid over detector params on the cached data — fast (in-memory).
  3. Score each combo on IS (first 60d) and OOS (last 30d) separately.
  4. Reject combos with bad OOS — overfit gate.
  5. Print the top combos by OOS Sharpe.

Focus: triple_top only (the validated winner from baseline backtest).
Other patterns had too few samples to tune meaningfully.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tinkoff.invest import CandleInterval  # noqa: E402
from tinkoff.invest.utils import quotation_to_decimal  # noqa: E402

from core.broker import BrokerClient  # noqa: E402
from futbot.config import FutSettings  # noqa: E402
from futbot.trend.portfolio import PORTFOLIO  # noqa: E402
from futbot.patterns.detectors import (  # noqa: E402
    detect_triple_tops,
    detect_triple_bottoms,
)
from futbot.patterns.primitives import find_swings  # noqa: E402
from futbot.patterns.backtest import (  # noqa: E402
    simulate_trades,
    fetch_ohlc,
    _resolve_front,
)

logger = logging.getLogger("patterns.tune")

CACHE_DIR = Path("data/cache/ohlc")


async def cache_data(days: int = 90):
    """Fetch + cache OHLC for all WF contracts."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s = FutSettings()
    broker = BrokerClient(
        token=s.T_INVEST_TOKEN,
        account_id=s.T_INVEST_ACCOUNT_ID,
        app_name="patterns-tune",
    )
    await broker.connect()
    for e in PORTFOLIO:
        out_path = CACHE_DIR / f"{e.base}_1h_{days}d.parquet"
        if out_path.exists():
            continue
        try:
            r = await _resolve_front(broker, e.base)
            if r is None:
                continue
            f, _ = r
            df = await fetch_ohlc(broker, f.figi, days=days)
            if df.empty:
                continue
            df.to_parquet(out_path)
            print(f"  cached {e.base}: {len(df)} bars")
        except Exception as ex:
            logger.exception(f"  {e.base}: {ex}")
    await broker.disconnect()


def load_cached(days: int = 90) -> dict[str, pd.DataFrame]:
    out = {}
    for e in PORTFOLIO:
        p = CACHE_DIR / f"{e.base}_1h_{days}d.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            out[e.base] = df.sort_values("time").reset_index(drop=True)
    return out


def run_grid(
    data: dict[str, pd.DataFrame],
    grid: list[dict],
    is_oos_split_pct: float = 2 / 3,
    max_bars_held: int = 48,
) -> pd.DataFrame:
    """For each param set, run detector+sim across all contracts, score IS/OOS."""
    results = []
    for params in grid:
        all_trades = []
        for base, df in data.items():
            swings = find_swings(
                df, window=params["swing_window"], min_prominence_pct=params["min_prom"]
            )
            tt = list(
                detect_triple_tops(
                    df,
                    swings,
                    peak_tol=params["peak_tol"],
                    min_height=params["min_height"],
                    min_width=params["min_width"],
                    max_width=params["max_width"],
                    max_confirm_bars=params["max_confirm"],
                )
            )
            tb = list(
                detect_triple_bottoms(
                    df,
                    swings,
                    peak_tol=params["peak_tol"],
                    min_height=params["min_height"],
                    min_width=params["min_width"],
                    max_width=params["max_width"],
                    max_confirm_bars=params["max_confirm"],
                )
            )
            sigs = sorted(tt + tb, key=lambda s: s.bar_idx)
            trades = simulate_trades(
                df, sigs, base=base, max_bars_held=params.get("hold", max_bars_held)
            )
            all_trades.extend(trades)

        if not all_trades:
            continue
        tdf = pd.DataFrame(
            [
                {
                    "base": t.base,
                    "pattern": t.pattern,
                    "entry_time": t.entry_time,
                    "net_pct": t.net_pnl_pct,
                    "exit_reason": t.exit_reason,
                }
                for t in all_trades
            ]
        )

        # IS/OOS split
        t_min, t_max = tdf.entry_time.min(), tdf.entry_time.max()
        split = t_min + (t_max - t_min) * is_oos_split_pct
        is_d = tdf[tdf.entry_time < split]
        oos_d = tdf[tdf.entry_time >= split]

        def stats(d, pat):
            sub = d[d.pattern == pat] if pat != "all" else d
            if len(sub) < 1:
                return dict(n=0, wr=0.0, avg=0.0, total=0.0, sh=0.0)
            wr = (sub.net_pct > 0).mean()
            avg = sub.net_pct.mean()
            std = sub.net_pct.std()
            return dict(
                n=len(sub),
                wr=wr,
                avg=avg,
                total=sub.net_pct.sum(),
                sh=avg / std if std and std > 0 else 0.0,
            )

        row = {**params}
        for half, d in [("is", is_d), ("oos", oos_d)]:
            for pat in ("triple_top", "triple_bottom", "all"):
                s = stats(d, pat)
                row[f"{half}_{pat[:3] if pat=='all' else pat[7:10]}_n"] = s["n"]
                row[f"{half}_{pat[:3] if pat=='all' else pat[7:10]}_wr"] = s["wr"]
                row[f"{half}_{pat[:3] if pat=='all' else pat[7:10]}_avg"] = s["avg"]
                row[f"{half}_{pat[:3] if pat=='all' else pat[7:10]}_tot"] = s["total"]
                row[f"{half}_{pat[:3] if pat=='all' else pat[7:10]}_sh"] = s["sh"]
        results.append(row)

    return pd.DataFrame(results)


def main():
    days = 90
    print("Loading cached OHLC…")
    data = load_cached(days=days)
    if len(data) < len(PORTFOLIO):
        print(f"Only {len(data)}/{len(PORTFOLIO)} cached, fetching the rest…")
        asyncio.run(cache_data(days=days))
        data = load_cached(days=days)
    print(f"Loaded {len(data)} contracts.")

    # ── Grid (slim — only most impactful axes) ────────────────────────
    grid_axes = {
        "swing_window": [3, 5, 7],
        "min_prom": [0.005],
        "peak_tol": [0.020, 0.030, 0.040],
        "min_height": [0.010, 0.015, 0.025],
        "min_width": [10, 15],
        "max_width": [60, 80],
        "max_confirm": [10],
        "hold": [24, 48, 72],
    }
    grid = [dict(zip(grid_axes, vs)) for vs in itertools.product(*grid_axes.values())]
    print(f"Grid size: {len(grid)} combos")

    df = run_grid(data, grid)
    if df.empty:
        print("No results!")
        return

    # Save full grid
    out_csv = Path("data/patterns_grid.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved full grid: {out_csv}")

    # Filter: oos triple_top n>=20 (enough sample) AND total > 0 in both IS/OOS
    f = df[(df["oos_top_n"] >= 20) & (df["oos_top_tot"] > 0) & (df["is_top_tot"] > 0)].copy()
    print(f"\nFiltered combos (oos n>=20, both halves positive): {len(f)}")
    if f.empty:
        print("No combos pass the gate; showing top OOS Sharpe overall:")
        f = df[df["oos_top_n"] >= 10].copy()
    f = f.sort_values("oos_top_sh", ascending=False)

    print("\nTOP 20 by OOS Sharpe (triple_top focus):")
    cols = [
        "swing_window",
        "min_prom",
        "peak_tol",
        "min_height",
        "min_width",
        "max_width",
        "max_confirm",
        "hold",
        "is_top_n",
        "is_top_wr",
        "is_top_tot",
        "is_top_sh",
        "oos_top_n",
        "oos_top_wr",
        "oos_top_tot",
        "oos_top_sh",
    ]
    with pd.option_context(
        "display.float_format", "{:.3f}".format, "display.width", 200, "display.max_columns", None
    ):
        print(f[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
    main()
