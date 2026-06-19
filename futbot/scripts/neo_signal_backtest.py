"""Backtest following the Telegram "📈" channel's crypto calls on Neo assets.

The channel posts discretionary BTC/ETH/SOL directional calls.  We extract the
CLEAR directional signals (instrument + long/short + timestamp), then test a
mechanical "follow every call" strategy on the corresponding Neo asset:
  • enter at the first hourly bar AT/AFTER the message time;
  • hold until the next OPPOSITE-direction call on the same instrument, or a
    max holding horizon, whichever comes first;
  • P&L in % of entry, then leverage-adjusted (Neo margin → return-on-margin).

This removes the channel's hindsight ("Готово ✅" is marked after the fact):
we only ever act at message time and exit by rule.

Honest caveats: signals are interpreted from free text; vague/ambiguous posts
are skipped; Neo price history starts ~2026-03, so only late-Apr→Jun signals
are testable.

Usage:
    python -u -m futbot.scripts.neo_signal_backtest --hold-days 4
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import Settings
from core.broker import BrokerClient
from tinkoff.invest import CandleInterval
from tinkoff.invest.utils import quotation_to_decimal

# ── Extracted signals: (instrument, direction +1 long / -1 short, UTC time) ──
# Parsed from channel "📈" message history (msg id in comment).
SIGNALS = [
    ("BTC", -1, "2026-04-27T15:21"),  # 703: "Пробуйте Short'ить ... под 75k"
    ("BTC", +1, "2026-05-01T12:14"),  # 705: "Пробуйте открыть небольшой Long"
    ("ETH", +1, "2026-05-05T20:16"),  # 710: "Long'анул ETH $2400-2500"
    ("ETH", -1, "2026-05-07T10:24"),  # 712: "Short ETH под $2300"
    ("BTC", -1, "2026-05-08T01:14"),  # 715: "пробовать Short'ы BTC $79-78k"
    ("ETH", +1, "2026-05-09T16:41"),  # 719: "От $2330 Long лимитка"
    ("BTC", -1, "2026-05-12T01:55"),  # 722: "BTC ниже $80 000"
    ("BTC", -1, "2026-05-13T02:31"),  # 724: "BTC под $79-80k"
    ("ETH", -1, "2026-05-16T08:54"),  # 726: "по'Short'ить ETH/SOL"
    ("SOL", -1, "2026-05-16T08:54"),  # 726
    ("SOL", -1, "2026-05-24T16:46"),  # 741: "SOL Short $81.5-82"
    # CORRECTED: he was bearish for June (weekly review 752) and SHORTED —
    # the 06-01 image proves a SOL short @80.303 (not a BTC long).
    ("SOL", -1, "2026-06-01T16:55"),  # image: SOL Short 15x @80.303
    ("ETH", -1, "2026-06-05T14:30"),  # 759 + image: ETH Short 15x @~2000
]

NEO = {"BTC": "BTCUSDperpA", "ETH": "ETHUSDperpA", "SOL": "SOLUSDperpA"}


async def _fetch(broker, figi, days=120):
    now = datetime.now(timezone.utc)
    out, end, left = [], now, days
    while left > 0:
        cd = min(left, 89)
        start = end - timedelta(days=cd)
        try:
            c = await broker.get_candles(
                figi, start, end, interval=CandleInterval.CANDLE_INTERVAL_HOUR
            )
            for x in c:
                out.append({"time": x.time, "close": float(quotation_to_decimal(x.close))})
        except Exception:
            pass
        end = start
        left -= cd
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _price_at(df, ts):
    """First close at/after ts."""
    sub = df[df["time"] >= ts]
    if sub.empty:
        return None, None
    return float(sub.iloc[0]["close"]), sub.iloc[0]["time"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-days", type=float, default=4.0)
    args = ap.parse_args()
    s = Settings()
    b = BrokerClient(token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="neo-sig")
    await b.connect()
    futs = await b.get_all_futures()
    data, margins = {}, {}
    for sym, tk in NEO.items():
        f = next((x for x in futs if (getattr(x, "ticker", "") or "") == tk), None)
        if f is None:
            continue
        data[sym] = await _fetch(b, f.figi)
        meta = b.extract_futures_metadata(f)
        margins[sym] = float(meta.get("dlong") or 0.15) or 0.15
    await b.disconnect()

    # Group signals by instrument, sort by time
    by_sym = {}
    for sym, d, ts in SIGNALS:
        by_sym.setdefault(sym, []).append((pd.Timestamp(ts, tz="UTC"), d))
    for sym in by_sym:
        by_sym[sym].sort()

    trades = []
    for sym, sigs in by_sym.items():
        df = data.get(sym)
        if df is None or df.empty:
            continue
        for i, (ts, d) in enumerate(sigs):
            entry_px, entry_t = _price_at(df, ts)
            if entry_px is None:
                continue
            # exit: next opposite signal OR hold horizon
            exit_deadline = ts + timedelta(days=args.hold_days)
            nxt_opp = next((t for (t, dd) in sigs[i + 1 :] if dd != d), None)
            exit_ts = min([x for x in [nxt_opp, exit_deadline] if x is not None])
            exit_px, exit_t = _price_at(df, exit_ts)
            if exit_px is None:
                exit_px = float(df.iloc[-1]["close"])
                exit_t = df.iloc[-1]["time"]
            pct = (exit_px - entry_px) / entry_px * 100 * d
            trades.append(
                {
                    "sym": sym,
                    "dir": "LONG" if d > 0 else "SHORT",
                    "entry_t": entry_t,
                    "entry": entry_px,
                    "exit_t": exit_t,
                    "exit": exit_px,
                    "pct": pct,
                    "lev": 1.0 / margins[sym],
                    "on_margin": pct / margins[sym],
                }
            )

    td = pd.DataFrame(trades)
    print("=" * 96)
    print(f"FOLLOWING CHANNEL '📈' ON NEO ASSETS  (hold ≤{args.hold_days}d or until opposite call)")
    print("=" * 96)
    if td.empty:
        print("No testable signals (Neo history too short).")
        return
    print(
        f"{'sym':<5}{'dir':<6}{'entry_time':<17}{'entry':>10}{'exit':>10}"
        f"{'pct%':>8}{'onMargin%':>11}"
    )
    print("-" * 96)
    for _, r in td.iterrows():
        print(
            f"{r['sym']:<5}{r['dir']:<6}{str(r['entry_t'])[:16]:<17}"
            f"{r['entry']:>10.2f}{r['exit']:>10.2f}{r['pct']:>+8.2f}{r['on_margin']:>+11.1f}"
        )
    print("-" * 96)
    n = len(td)
    wins = (td.pct > 0).sum()
    print(
        f"Signals: {n}  win-rate: {wins/n*100:.0f}%  "
        f"avg: {td.pct.mean():+.2f}%  total (price): {td.pct.sum():+.1f}%"
    )
    print(
        f"LEVERAGE-ADJUSTED (return on Neo margin): "
        f"total {td.on_margin.sum():+.0f}%  avg {td.on_margin.mean():+.1f}%"
    )
    print(f"\nPer instrument:")
    for sym, g in td.groupby("sym"):
        print(
            f"  {sym}: n={len(g)} win={ (g.pct>0).mean()*100:.0f}% "
            f"total={g.pct.sum():+.1f}% onMargin={g.on_margin.sum():+.0f}%"
        )
    # Buy&hold benchmark for comparison (BTC over the window)
    if "BTC" in data and not data["BTC"].empty:
        d0, d1 = data["BTC"].iloc[0]["close"], data["BTC"].iloc[-1]["close"]
        print(f"\nBenchmark BTC buy&hold over window: {(d1-d0)/d0*100:+.1f}%")
    print("\nCaveat: signals interpreted from free text; entries at MESSAGE time")
    print("(no hindsight); vague posts skipped; ~90d Neo history limits sample.")


if __name__ == "__main__":
    asyncio.run(main())
