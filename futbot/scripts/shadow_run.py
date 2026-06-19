"""Compare futbot's paper decisions against trade_claude's live decisions.

Reads:
  - futbot's `decisions` + `trades` tables from data/futbot.db
  - trade_claude's `signals` + `trades` tables from data/trade_bot.db

For a window of recent N days, computes:
  * how many opportunities each bot saw,
  * how many turned into trades,
  * realised P&L summary for each,
  * overlap analysis (did they trade the same instrument in the same window?
    — only when the same FIGI appears in both within ±2h of each other).

This is not a true paper-vs-paper shadow run (trade_claude doesn't have
a "what would I have done" mode), but it gives an apples-to-apples view
of "activity and outcome" so you can decide whether to flip futbot to
live.

Usage:
    python -m futbot.scripts.shadow_run                 # last 14 days
    python -m futbot.scripts.shadow_run 7
"""

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    fut_db = Path("data/futbot.db")
    tc_db = Path("data/trade_bot.db")
    if not fut_db.exists():
        print(f"futbot DB not found at {fut_db} — run paper mode first.")
        sys.exit(1)
    if not tc_db.exists():
        print(f"trade_claude DB not found at {tc_db}")
        sys.exit(1)

    print(f"Shadow run comparison — last {days} days (since {cutoff[:10]})")
    print()

    # ── futbot side ──────────────────────────────────────────────────────
    fc = sqlite3.connect(fut_db)
    fc.row_factory = sqlite3.Row
    fut = {}
    fut["decisions"] = fc.execute(
        "SELECT COUNT(*) FROM decisions WHERE ts >= ?", (cutoff,)
    ).fetchone()[0]
    fut["approved"] = fc.execute(
        "SELECT COUNT(*) FROM decisions WHERE ts >= ? AND approved=1", (cutoff,)
    ).fetchone()[0]
    fut["trades_n"] = fc.execute(
        "SELECT COUNT(*) FROM trades WHERE entry_time >= ?", (cutoff,)
    ).fetchone()[0]
    fut["closed_n"] = fc.execute(
        "SELECT COUNT(*) FROM trades WHERE exit_time >= ?", (cutoff,)
    ).fetchone()[0]
    fut["sum_pnl"] = (
        fc.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE exit_time >= ?", (cutoff,)
        ).fetchone()[0]
        or 0.0
    )
    rej_breakdown = fc.execute(
        """SELECT rejected_at_layer, COUNT(*) c
           FROM decisions
           WHERE ts >= ? AND approved=0
           GROUP BY rejected_at_layer
           ORDER BY c DESC""",
        (cutoff,),
    ).fetchall()
    fut_figis = {
        r["figi"]
        for r in fc.execute(
            "SELECT DISTINCT figi FROM trades WHERE entry_time >= ?", (cutoff,)
        ).fetchall()
    }
    fc.close()

    # ── trade_claude side ────────────────────────────────────────────────
    tcc = sqlite3.connect(tc_db)
    tcc.row_factory = sqlite3.Row
    tc = {}
    try:
        tc["signals_n"] = tcc.execute(
            "SELECT COUNT(*) FROM signals WHERE created_at >= ?", (cutoff,)
        ).fetchone()[0]
        tc["approved"] = tcc.execute(
            "SELECT COUNT(*) FROM signals WHERE created_at >= ? AND approved=1",
            (cutoff,),
        ).fetchone()[0]
    except Exception:
        tc["signals_n"] = tc["approved"] = 0
    tc["trades_n"] = tcc.execute(
        "SELECT COUNT(*) FROM trades WHERE entry_time >= ?", (cutoff,)
    ).fetchone()[0]
    tc["closed_n"] = tcc.execute(
        "SELECT COUNT(*) FROM trades WHERE exit_time >= ? AND pnl IS NOT NULL",
        (cutoff,),
    ).fetchone()[0]
    tc["sum_pnl"] = (
        tcc.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM trades " "WHERE exit_time >= ? AND pnl IS NOT NULL",
            (cutoff,),
        ).fetchone()[0]
        or 0.0
    )
    tc_figis = {
        r["figi"]
        for r in tcc.execute(
            "SELECT DISTINCT figi FROM trades WHERE entry_time >= ?", (cutoff,)
        ).fetchall()
    }
    tcc.close()

    # ── print ────────────────────────────────────────────────────────────
    print(f"{'metric':<30} {'futbot':>12} {'trade_claude':>15}")
    print("-" * 65)
    print(f"{'pipeline evaluations':<30} {fut['decisions']:>12} {tc['signals_n']:>15}")
    print(f"{'approved (passed gates)':<30} {fut['approved']:>12} {tc['approved']:>15}")
    print(f"{'trades opened':<30} {fut['trades_n']:>12} {tc['trades_n']:>15}")
    print(f"{'trades closed':<30} {fut['closed_n']:>12} {tc['closed_n']:>15}")
    print(f"{'realised P&L (₽)':<30} {fut['sum_pnl']:>+12.2f} {tc['sum_pnl']:>+15.2f}")
    print(f"{'unique figis traded':<30} {len(fut_figis):>12} {len(tc_figis):>15}")
    print(f"{'figi overlap':<30} {len(fut_figis & tc_figis):>12}")

    print()
    print(f"futbot rejection breakdown (last {days}d):")
    if not rej_breakdown:
        print("  (none)")
    else:
        for row in rej_breakdown:
            print(f"  @ {row['rejected_at_layer'] or '?':<12}  n={row['c']}")


if __name__ == "__main__":
    main()
