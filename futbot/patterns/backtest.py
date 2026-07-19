"""Pattern backtester — simulate trades from detector signals.

Trade lifecycle:
    1. Signal fires on bar `i` (confirmation close).  Open at signal.entry_price.
    2. Walk forward bar-by-bar.  Exit on whichever fires first:
         a) stop hit: bar's high/low crosses signal.stop_price
         b) target hit: bar's high/low crosses signal.target_price
         c) timeout: max_bars_held reached (exit at close)
         d) opposite-side signal fires for SAME pattern family
    3. Commission applied via futbot.utils.commissions.round_trip_pnl

Concurrency: only ONE open position per contract at a time.  If a signal
fires while a position is open, it's skipped (same as live trading would
do under TREND_MAX_OPEN_POSITIONS).

Output: per-trade DataFrame + aggregated stats per (contract, pattern).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from t_tech.invest import CandleInterval  # noqa: E402
from t_tech.invest.utils import quotation_to_decimal  # noqa: E402

from core.broker import BrokerClient  # noqa: E402
from futbot.config import FutSettings as Settings  # noqa: E402
from futbot.trend.portfolio import PORTFOLIO  # noqa: E402
from futbot.patterns.detectors import detect_all, Signal  # noqa: E402
from futbot.utils import commissions as comm  # noqa: E402

logger = logging.getLogger("patterns.bt")


@dataclass
class Trade:
    base: str
    pattern: str
    direction: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str  # "stop", "target", "timeout", "opposite"
    bars_held: int
    gross_pnl_pct: float
    net_pnl_pct: float  # after one round-trip commission as % of entry
    pattern_height_pct: float


def simulate_trades(
    df: pd.DataFrame,
    signals: list[Signal],
    *,
    base: str,
    max_bars_held: int = 48,
    commission_rt_pct: float = 0.0008,  # 0.08% RT for futures
) -> list[Trade]:
    """Walk through signals chronologically, simulate each."""
    trades: list[Trade] = []
    next_free_bar = -1  # we're "free" to open when current bar > this

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    times = df["time"].to_numpy()
    n = len(df)

    for sig in signals:
        if sig.bar_idx <= next_free_bar:
            continue  # already in a trade

        # Walk forward from bar after entry
        exit_idx = None
        exit_price = None
        exit_reason = None
        for j in range(sig.bar_idx + 1, min(n, sig.bar_idx + 1 + max_bars_held)):
            hi, lo = highs[j], lows[j]
            if sig.direction == +1:  # long
                # Conservative: check stop FIRST (worst case if both hit on same bar)
                if lo <= sig.stop_price:
                    exit_idx, exit_price, exit_reason = j, sig.stop_price, "stop"
                    break
                if hi >= sig.target_price:
                    exit_idx, exit_price, exit_reason = j, sig.target_price, "target"
                    break
            else:  # short
                if hi >= sig.stop_price:
                    exit_idx, exit_price, exit_reason = j, sig.stop_price, "stop"
                    break
                if lo <= sig.target_price:
                    exit_idx, exit_price, exit_reason = j, sig.target_price, "target"
                    break

        if exit_idx is None:
            # Timeout at max_bars_held
            exit_idx = min(n - 1, sig.bar_idx + max_bars_held)
            exit_price = float(closes[exit_idx])
            exit_reason = "timeout"

        gross_pct = (
            (exit_price - sig.entry_price) / sig.entry_price
            if sig.direction == +1
            else (sig.entry_price - exit_price) / sig.entry_price
        ) * 100
        net_pct = gross_pct - commission_rt_pct * 100

        trades.append(
            Trade(
                base=base,
                pattern=sig.pattern,
                direction=sig.direction,
                entry_time=pd.Timestamp(times[sig.bar_idx]),
                entry_price=sig.entry_price,
                exit_time=pd.Timestamp(times[exit_idx]),
                exit_price=float(exit_price),
                exit_reason=exit_reason,
                bars_held=exit_idx - sig.bar_idx,
                gross_pnl_pct=round(gross_pct, 4),
                net_pnl_pct=round(net_pct, 4),
                pattern_height_pct=round(sig.pattern_height_pct, 4),
            )
        )
        next_free_bar = exit_idx

    return trades


# ── Data fetching helpers ─────────────────────────────────────────────


async def _resolve_front(broker: BrokerClient, base: str, min_dte: int = 14):
    futs = await broker.get_all_futures()
    now = datetime.now(timezone.utc)
    cands = []
    for f in futs:
        t = getattr(f, "ticker", "") or ""
        if len(t) < 3:
            continue
        if not (t == base or (t.startswith(base) and len(t) == len(base) + 2)):
            continue
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
    for f, exp in cands:
        if (exp - now).days >= min_dte:
            return f, exp
    return cands[0]


async def fetch_ohlc(broker: BrokerClient, figi: str, days: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    candles = await broker.get_candles(
        figi,
        now - timedelta(days=days),
        now,
        interval=CandleInterval.CANDLE_INTERVAL_HOUR,
    )
    if not candles:
        return pd.DataFrame()
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
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


# ── Aggregation ───────────────────────────────────────────────────────


def aggregate_stats(trades: list[Trade]) -> pd.DataFrame:
    """Group trades by (base, pattern) and compute stats."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(t) for t in trades])
    g = df.groupby(["base", "pattern"])
    out = g.agg(
        n_trades=("net_pnl_pct", "size"),
        wins=("net_pnl_pct", lambda x: (x > 0).sum()),
        avg_net_pct=("net_pnl_pct", "mean"),
        median_net_pct=("net_pnl_pct", "median"),
        std_net_pct=("net_pnl_pct", "std"),
        total_net_pct=("net_pnl_pct", "sum"),
        avg_bars=("bars_held", "mean"),
    ).reset_index()
    out["win_rate"] = out["wins"] / out["n_trades"]
    # Sharpe-equivalent: mean/std on PER-TRADE returns; not annualised, but
    # useful for ranking.
    out["per_trade_sharpe"] = out["avg_net_pct"] / out["std_net_pct"].replace(0, np.nan)
    return out.sort_values("per_trade_sharpe", ascending=False)


def aggregate_pattern_only(trades: list[Trade]) -> pd.DataFrame:
    """Group by pattern only — does any pattern work UNIVERSALLY?"""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([asdict(t) for t in trades])
    g = df.groupby("pattern")
    out = g.agg(
        n_trades=("net_pnl_pct", "size"),
        wins=("net_pnl_pct", lambda x: (x > 0).sum()),
        avg_net_pct=("net_pnl_pct", "mean"),
        std_net_pct=("net_pnl_pct", "std"),
        total_net_pct=("net_pnl_pct", "sum"),
    ).reset_index()
    out["win_rate"] = out["wins"] / out["n_trades"]
    out["per_trade_sharpe"] = out["avg_net_pct"] / out["std_net_pct"].replace(0, np.nan)
    return out.sort_values("per_trade_sharpe", ascending=False)


# ── Main runner ───────────────────────────────────────────────────────


async def run(
    *,
    days: int = 90,
    max_bars_held: int = 48,
    bases: list[str] | None = None,
    out_csv: Path | None = None,
) -> dict:
    settings = Settings()
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="patterns-bt",
    )
    await broker.connect()

    target_bases = bases or [e.base for e in PORTFOLIO]
    all_trades: list[Trade] = []
    per_contract_meta: list[dict] = []

    for base in target_bases:
        try:
            r = await _resolve_front(broker, base)
            if r is None:
                logger.warning(f"  {base}: no front-month found")
                continue
            f, exp = r
            df = await fetch_ohlc(broker, f.figi, days=days)
            if df.empty or len(df) < 50:
                logger.warning(f"  {base}: insufficient bars ({len(df)})")
                continue
            sigs = detect_all(df)
            trades = simulate_trades(df, sigs, base=base, max_bars_held=max_bars_held)
            all_trades.extend(trades)
            per_contract_meta.append(
                {
                    "base": base,
                    "ticker": f.ticker,
                    "bars": len(df),
                    "signals": len(sigs),
                    "trades": len(trades),
                }
            )
            print(
                f"  {base:<10} bars={len(df):>5}  signals={len(sigs):>3}  "
                f"trades={len(trades):>3}"
            )
        except Exception as e:
            logger.exception(f"  {base}: failed ({e})")

    await broker.disconnect()

    by_pair = aggregate_stats(all_trades)
    by_pat = aggregate_pattern_only(all_trades)

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_trades = pd.DataFrame([asdict(t) for t in all_trades])
        df_trades.to_csv(out_csv, index=False)
        print(f"\nWrote {len(df_trades)} trades → {out_csv}")

    return {
        "trades": all_trades,
        "by_pair": by_pair,
        "by_pattern": by_pat,
        "meta": per_contract_meta,
    }


if __name__ == "__main__":
    import argparse

    # Windows console defaults to cp1251 → ₽/→ chars crash on print.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=90, help="Hourly history days")
    p.add_argument("--hold", type=int, default=48, help="Max bars held")
    p.add_argument(
        "--bases", type=str, default="", help="Comma-sep bases (default: all WF portfolio)"
    )
    p.add_argument("--out", type=str, default="data/patterns_backtest.csv", help="CSV output path")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    bases = [b.strip() for b in args.bases.split(",") if b.strip()] or None
    out_path = Path(args.out) if args.out else None

    res = asyncio.run(run(days=args.days, max_bars_held=args.hold, bases=bases, out_csv=out_path))

    print("\n" + "=" * 70)
    print("PER-PATTERN (across all contracts):")
    print("=" * 70)
    by_pat = res["by_pattern"]
    if not by_pat.empty:
        with pd.option_context(
            "display.float_format",
            "{:.3f}".format,
            "display.width",
            200,
            "display.max_columns",
            None,
        ):
            print(by_pat.to_string(index=False))

    print("\n" + "=" * 70)
    print("TOP 20 (contract, pattern) combos by per-trade Sharpe (min 5 trades):")
    print("=" * 70)
    by_pair = res["by_pair"]
    if not by_pair.empty:
        good = by_pair[by_pair["n_trades"] >= 5].copy()
        with pd.option_context(
            "display.float_format",
            "{:.3f}".format,
            "display.width",
            200,
            "display.max_columns",
            None,
        ):
            print(good.head(20).to_string(index=False))

    print("\nTotal trades:", len(res["trades"]))
