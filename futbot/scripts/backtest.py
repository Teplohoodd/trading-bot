"""Run the futbot pipeline as a backtest on historical Tinkoff candles.

Fetches the last N days of 1h / 15m / 5m bars for each tier-1 contract
and replays the 4-layer decision pipeline at every 5m bar (or every Nth
bar if eval_every_bars > 1).  Reports per-contract and aggregate metrics.

Usage:
    python -m futbot.scripts.backtest              # last 90 days, every 6 bars (= 1h cadence)
    python -m futbot.scripts.backtest 180 1        # 180 days, every bar
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.broker import BrokerClient  # noqa: E402
from futbot.config import FutSettings  # noqa: E402
from futbot.universe import resolve_universe  # noqa: E402
from futbot.data.candles import fetch_tf  # noqa: E402
from futbot.backtest.replay import backtest_contract  # noqa: E402


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("tinkoff").setLevel(logging.WARNING)
    log = logging.getLogger("futbot.backtest_cli")

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    eval_every = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    settings = FutSettings()
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="futbot-backtest",
    )
    await broker.connect()

    log.info(f"Resolving universe…")
    universe = await resolve_universe(broker, settings)
    # Tier-1 only for backtest (faster).
    universe = [c for c in universe if c.tier == 1]
    log.info(f"Tier-1 contracts: {[c.ticker for c in universe]}")

    all_metrics: dict[str, dict] = {}
    all_trades: list[dict] = []

    for c in universe:
        log.info(f"--- {c.ticker} (last {days}d) ---")
        try:
            df_1h = await fetch_tf(broker, c.figi, "1h")
            df_15m = await fetch_tf(broker, c.figi, "15m")
            df_5m = await fetch_tf(broker, c.figi, "5m")
            # Trim to the requested window
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            df_1h = df_1h[df_1h["time"] >= cutoff].reset_index(drop=True)
            df_15m = df_15m[df_15m["time"] >= cutoff].reset_index(drop=True)
            df_5m = df_5m[df_5m["time"] >= cutoff].reset_index(drop=True)
            log.info(f"  bars 1h={len(df_1h)} 15m={len(df_15m)} 5m={len(df_5m)}")
            if len(df_5m) < 100:
                log.info(f"  {c.ticker}: too few 5m bars, skipping")
                continue
            trades, metrics = await backtest_contract(
                df_1h=df_1h,
                df_15m=df_15m,
                df_5m=df_5m,
                ticker=c.ticker,
                base_ticker=c.base,
                settings=settings,
                figi=c.figi,
                eval_every_bars=eval_every,
            )
            all_metrics[c.ticker] = metrics
            for t in trades:
                all_trades.append(
                    {
                        "ticker": c.ticker,
                        "direction": t.direction,
                        "entry": t.entry,
                        "exit": t.exit,
                        "bars_held": t.bars_held,
                        "pnl_pct": t.pnl_pct,
                        "reason": t.reason,
                        "entry_time": t.entry_time,
                        "exit_time": t.exit_time,
                    }
                )
            log.info(f"  {c.ticker}: {metrics}")
        except Exception as e:
            log.exception(f"  {c.ticker}: backtest failed: {e}")

    await broker.disconnect()

    # Aggregate report
    print()
    print("=" * 100)
    print(
        f"{'ticker':<8} {'n':>4} {'win%':>5} {'sum%':>7} {'avg%':>7} "
        f"{'PF':>5} {'sharpe':>7} {'maxDD%':>7} {'bars':>5}"
    )
    print("-" * 100)
    for tk, m in all_metrics.items():
        if not m or m.get("n_trades", 0) == 0:
            print(f"{tk:<8} {0:>4}  (no trades)")
            continue
        print(
            f"{tk:<8} {m['n_trades']:>4} {m['win_rate']:>5.1f} "
            f"{m['total_pct']:>+7.2f} {m['avg_pct']:>+7.3f} "
            f"{m['profit_factor']:>5.2f} {m['sharpe']:>+7.2f} "
            f"{m['max_dd_pct']:>+7.2f} {m['avg_bars_held']:>5.1f}"
        )

    # Persist trades for inspection
    if all_trades:
        import pandas as pd

        out = Path("data/futbot_backtest_trades.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_trades).to_csv(out, index=False)
        print(f"\nTrade-level detail → {out}  (n={len(all_trades)})")


if __name__ == "__main__":
    asyncio.run(main())
