"""Neo-asset pattern backtest — do our trend patterns work on Neo assets?

Neo assets ("неоактивы") track US stocks / crypto.  Mechanics differ from
normal futures:
  • price in USD; P&L credited in RUB at the close-day FX rate (the % move is
    currency-independent, so pattern returns transfer directly);
  • NO buy/sell commission, but a DAILY holding fee charged if held past
    00:00 MSK: ~4.5%/yr (срочные) or 4.5% + CB-rate (~16-21%) for perpetuals
    → ~20.5%/yr ≈ 0.056%/day;
  • high leverage: risk rate (margin) 10-27% → 4-10× → return-on-margin =
    price_return / margin_fraction.

So Neo are attractive for patterns: tiny per-trade friction on short holds +
big leverage.  This tests triple_top/triple_bottom (our live edge) on all Neo
assets, with the Neo cost model, leverage-adjusted, plus an IS/OOS split.

Usage:
    python -u -m futbot.scripts.neo_backtest --days 90
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.disable(logging.CRITICAL)

from config.settings import Settings
from core.broker import BrokerClient
from tinkoff.invest import CandleInterval
from tinkoff.invest.utils import quotation_to_decimal
from futbot.patterns.primitives import find_swings
from futbot.patterns.detectors import detect_triple_tops, detect_triple_bottoms
from futbot.patterns.portfolio import TUNED_PARAMS

# Neo cost model
CB_RATE = 0.16  # approx Bank of Russia key rate
NEO_ANNUAL_HOLD = 0.045 + CB_RATE  # perpetual Neo: 4.5% + CB
NEO_DAILY = NEO_ANNUAL_HOLD / 365.0  # ≈ 0.056 %/day
NEO_TRADING_HRS_PER_DAY = 17.0  # ~07:00-00:00 MSK


async def _fetch(broker, figi, days):
    now = datetime.now(timezone.utc)
    out = []
    end = now
    left = days
    while left > 0:
        cd = min(left, 89)
        start = end - timedelta(days=cd)
        try:
            c = await broker.get_candles(
                figi, start, end, interval=CandleInterval.CANDLE_INTERVAL_HOUR
            )
            for x in c:
                out.append(
                    {
                        "time": x.time,
                        "open": float(quotation_to_decimal(x.open)),
                        "high": float(quotation_to_decimal(x.high)),
                        "low": float(quotation_to_decimal(x.low)),
                        "close": float(quotation_to_decimal(x.close)),
                    }
                )
        except Exception:
            pass
        end = start
        left -= cd
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def simulate_neo(df, signals, max_bars_held=48):
    """Stop/target/timeout sim with NEO cost (daily holding, no RT comm)."""
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    times = df["time"].to_numpy()
    n = len(df)
    trades = []
    free = -1
    for s in signals:
        if s.bar_idx <= free:
            continue
        ex_idx = ex_px = reason = None
        for j in range(s.bar_idx + 1, min(n, s.bar_idx + 1 + max_bars_held)):
            hi, lo = highs[j], lows[j]
            if s.direction == +1:
                if lo <= s.stop_price:
                    ex_idx, ex_px, reason = j, s.stop_price, "stop"
                    break
                if hi >= s.target_price:
                    ex_idx, ex_px, reason = j, s.target_price, "target"
                    break
            else:
                if hi >= s.stop_price:
                    ex_idx, ex_px, reason = j, s.stop_price, "stop"
                    break
                if lo <= s.target_price:
                    ex_idx, ex_px, reason = j, s.target_price, "target"
                    break
        if ex_idx is None:
            ex_idx = min(n - 1, s.bar_idx + max_bars_held)
            ex_px = float(closes[ex_idx])
            reason = "timeout"
        gross = (
            (ex_px - s.entry_price) / s.entry_price
            if s.direction == +1
            else (s.entry_price - ex_px) / s.entry_price
        ) * 100
        bars_held = ex_idx - s.bar_idx
        # Holding fee = per 00:00-MSK crossing (nights held), NOT per hour.
        msk = timezone(timedelta(hours=3))
        e_d = pd.Timestamp(times[s.bar_idx]).tz_convert(msk).date()
        x_d = pd.Timestamp(times[ex_idx]).tz_convert(msk).date()
        nights = max(0, (x_d - e_d).days)
        hold_cost = NEO_DAILY * 100 * nights  # % holding fee
        net = gross - hold_cost
        trades.append(
            {
                "pattern": s.pattern,
                "direction": s.direction,
                "entry_time": pd.Timestamp(times[s.bar_idx]),
                "bars_held": bars_held,
                "gross_pct": gross,
                "hold_cost_pct": hold_cost,
                "net_pct": net,
                "reason": reason,
            }
        )
        free = ex_idx
    return trades


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    args = ap.parse_args()
    s = Settings()
    b = BrokerClient(token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="neo-bt")
    await b.connect()
    futs = await b.get_all_futures()
    neo = [f for f in futs if (getattr(f, "ticker", "") or "").endswith("perpA")]

    p = TUNED_PARAMS
    det_kw = dict(
        peak_tol=p.peak_tol,
        min_height=p.min_height,
        min_width=p.min_width,
        max_width=p.max_width,
        max_confirm_bars=p.max_confirm_bars,
    )
    all_trades = []
    margins = {}
    for f in neo:
        df = await _fetch(b, f.figi, args.days)
        if len(df) < 200:
            continue
        meta = b.extract_futures_metadata(f)
        dl = float(meta.get("dlong") or 0.0)
        margins[f.ticker] = dl if dl > 0 else 0.20
        swings = find_swings(df, window=p.swing_window, min_prominence_pct=p.min_prominence_pct)
        sigs = list(detect_triple_tops(df, swings, **det_kw)) + list(
            detect_triple_bottoms(df, swings, **det_kw)
        )
        sigs.sort(key=lambda x: x.bar_idx)
        for t in simulate_neo(df, sigs, max_bars_held=p.max_bars_held):
            t["base"] = f.ticker
            t["margin"] = margins[f.ticker]
            all_trades.append(t)
    await b.disconnect()

    if not all_trades:
        print("No Neo trades generated")
        return
    td = pd.DataFrame(all_trades)
    td["lev"] = 1.0 / td["margin"]
    td["net_on_margin"] = td["net_pct"] * td["lev"]

    print("=" * 92)
    print(
        f"NEO-ASSET PATTERN BACKTEST (triple_top/bottom, ~{args.days}d, "
        f"daily hold {NEO_DAILY*100:.3f}%/day)"
    )
    print("=" * 92)
    n = len(td)
    print(f"Total trades: {n} across {td.base.nunique()} Neo assets")
    print(
        f"Win rate: {(td.net_pct>0).mean()*100:.0f}%  "
        f"avg net (price): {td.net_pct.mean():+.2f}%  "
        f"total net (price): {td.net_pct.sum():+.1f}%"
    )
    print(
        f"avg gross: {td.gross_pct.mean():+.2f}%  "
        f"avg hold cost: {td.hold_cost_pct.mean():.3f}%  "
        f"avg hold: {td.bars_held.mean():.0f}h"
    )
    print(
        f"LEVERAGE-ADJUSTED (return on margin): "
        f"avg {td.net_on_margin.mean():+.2f}%  total {td.net_on_margin.sum():+.0f}%"
    )

    # IS/OOS split (first half / second half by entry time)
    td = td.sort_values("entry_time")
    mid = td["entry_time"].iloc[len(td) // 2]
    is_ = td[td.entry_time < mid]
    oos = td[td.entry_time >= mid]

    def _st(x):
        return (
            f"n={len(x)} win={ (x.net_pct>0).mean()*100:.0f}% "
            f"net={x.net_pct.sum():+.1f}% onMargin={x.net_on_margin.sum():+.0f}%"
        )

    print(f"\nIS  (<{str(mid.date())}): {_st(is_)}")
    print(f"OOS (>={str(mid.date())}): {_st(oos)}")

    print("\nPer pattern:")
    for pat, g in td.groupby("pattern"):
        print(
            f"  {pat:<14} n={len(g):>3} win={(g.net_pct>0).mean()*100:>3.0f}% "
            f"net={g.net_pct.sum():>+7.1f}% onMargin={g.net_on_margin.sum():>+6.0f}%"
        )

    print("\nTop assets by net (price) %:")
    by = (
        td.groupby("base")
        .agg(
            n=("net_pct", "size"),
            win=("net_pct", lambda x: (x > 0).mean() * 100),
            net=("net_pct", "sum"),
            onM=("net_on_margin", "sum"),
        )
        .sort_values("net", ascending=False)
    )
    for base, r in by.head(12).iterrows():
        print(
            f"  {base:<14} n={int(r['n']):>2} win={r['win']:>3.0f}% "
            f"net={r['net']:>+7.1f}% onMargin={r['onM']:>+6.0f}%"
        )
    print("\nCaveat: ~90d single period (Neo assets are new). IS/OOS split is the")
    print("only OOS signal available.  Daily-hold + leverage modelled; funding ignored.")
    out = Path(__file__).resolve().parents[2] / "data" / "neo_backtest.csv"
    td.to_csv(out, index=False)


if __name__ == "__main__":
    asyncio.run(main())
