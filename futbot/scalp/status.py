"""Live status / results dashboard for scalp bot.

Prints to stdout:
  * Open positions (in-flight trades with realised + unrealised P&L)
  * Today's closed trades summary
  * Last 20 closed trades
  * Aggregate stats over the last N days (win rate, profit factor, avg)
  * Per-ticker breakdown

Usage:
    python -m futbot.scalp.status              # today + 7d summary
    python -m futbot.scalp.status 30           # last 30 days
"""

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futbot.scalp.config import ScalpSettings
from futbot.utils import commissions as comm


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    settings = ScalpSettings()
    db_path = Path(settings.SCALP_DB_PATH)
    if not db_path.exists():
        print(f"DB не найдена: {db_path}")
        print("Запусти бота хотя бы раз: python -m futbot.scalp.main")
        sys.exit(1)

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    today_iso = datetime.now(timezone.utc).date().isoformat()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # ── 1. Open positions ────────────────────────────────────────────────
    cur.execute("SELECT * FROM scalp_trades WHERE exit_time IS NULL ORDER BY entry_time")
    open_rows = cur.fetchall()
    print("=" * 90)
    print(f"СКАЛЬП-БОТ — статус {datetime.now(timezone.utc).isoformat()[:19]} UTC")
    print("=" * 90)
    if not open_rows:
        print("Открытых позиций: 0")
    else:
        print(f"Открытых позиций: {len(open_rows)}")
        print(
            f"  {'ticker':<8} {'dir':>4} {'lots':>4} {'entry':>10} {'stop':>10} {'tp':>10} {'opened':>20}"
        )
        for r in open_rows:
            print(
                f"  {r['ticker']:<8} {r['direction']:>4} {r['lots']:>4} "
                f"{r['entry_price']:>10.4f} {r['stop_loss']:>10.4f} {r['take_profit']:>10.4f} "
                f"{r['entry_time'][:19]:>20}"
            )

    # ── 2. Today's closed trades ─────────────────────────────────────────
    cur.execute(
        "SELECT * FROM scalp_trades "
        "WHERE date(entry_time) = ? AND exit_time IS NOT NULL "
        "ORDER BY exit_time DESC",
        (today_iso,),
    )
    today_rows = cur.fetchall()
    today_pnl = sum((r["pnl"] or 0) for r in today_rows)
    today_wins = sum(1 for r in today_rows if (r["pnl"] or 0) > 0)
    print()
    print(f"=== СЕГОДНЯ ({today_iso} UTC) — {len(today_rows)} закрытых сделок ===")
    if today_rows:
        wr = today_wins / len(today_rows) * 100 if today_rows else 0
        print(f"  Win rate: {wr:.0f}%   Сумма P&L: {today_pnl:+.2f} ₽")

    # ── 3. Recent N closed trades ─────────────────────────────────────────
    cur.execute(
        "SELECT * FROM scalp_trades WHERE exit_time IS NOT NULL " "ORDER BY exit_time DESC LIMIT 20"
    )
    recent = cur.fetchall()
    print()
    print(f"=== ПОСЛЕДНИЕ {len(recent)} ЗАКРЫТЫХ СДЕЛОК ===")
    if not recent:
        print("  (пока нет закрытых сделок)")
    else:
        print(
            f"  {'time':<20} {'ticker':<7} {'dir':>4} {'entry':>9} {'exit':>9} "
            f"{'pnl_₽':>9} {'pnl_%':>7} {'reason':>14} {'score':>7}"
        )
        for r in recent:
            pnl = r["pnl"] or 0
            pnl_pct = r["pnl_pct"] or 0
            print(
                f"  {r['exit_time'][:19]:<20} {r['ticker']:<7} "
                f"{r['direction']:>4} {r['entry_price']:>9.4f} {r['exit_price']:>9.4f} "
                f"{pnl:>+9.2f} {pnl_pct:>+7.3f} "
                f"{(r['exit_reason'] or '')[:14]:>14} "
                f"{(r['score'] or 0):>+7.2f}"
            )

    # ── 4. Aggregate stats over `days` ────────────────────────────────────
    cur.execute(
        "SELECT * FROM scalp_trades " "WHERE entry_time >= ? AND exit_time IS NOT NULL",
        (cutoff,),
    )
    period = cur.fetchall()
    print()
    print(f"=== АГРЕГАТ ЗА ПОСЛЕДНИЕ {days} ДНЕЙ ===")
    if not period:
        print("  Нет данных")
    else:
        n = len(period)
        # pnl in DB after Fix 2 includes commission (NET).  We can recover
        # GROSS by adding back 2 × 0.0004 × entry × rub_per_point × lots.
        # Note: old trades (pre-fix) have GROSS in `pnl`.  No clean way to
        # distinguish; the gross/net columns will agree for those (showing
        # historical artefact).
        cpct = comm.commission_pct("future")
        nets = []
        grosses = []
        comms_paid = []
        for r in period:
            net = r["pnl"] or 0
            # Approximation: 2 × commission × notional.  rub_per_point not
            # stored on the row; we assume 1.0 (true for share-style futures
            # like SR/GZ/LK/MX).  For currency/oil futures the absolute ₽
            # figure is approximate but the ratio is unchanged.
            rt_comm = 2 * cpct * (r["entry_price"] or 0) * (r["lots"] or 1)
            nets.append(net)
            grosses.append(net + rt_comm)
            comms_paid.append(rt_comm)
        wins = [p for p in nets if p > 0]
        losses = [p for p in nets if p < 0]
        total_net = sum(nets)
        total_gross = sum(grosses)
        total_comm = sum(comms_paid)
        wr = len(wins) / n * 100
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf")
        print(f"  Сделок: {n}    Win-rate: {wr:.1f}%")
        print(f"  GROSS P&L:      {total_gross:+.2f} ₽   (без комиссии)")
        print(f"  Комиссия:      {-total_comm:+.2f} ₽")
        print(f"  NET P&L:        {total_net:+.2f} ₽   ← фактический результат")
        print(f"  avg NET/trade:  {total_net/n:+.2f} ₽")
        print(
            f"  Avg win: {avg_win:+.2f} ₽    Avg loss: {avg_loss:+.2f} ₽    "
            f"Profit factor (NET): {pf:.2f}"
        )

    # ── 5. Per-ticker breakdown ──────────────────────────────────────────
    cur.execute(
        """SELECT ticker, COUNT(*) n,
                  SUM(pnl) total_pnl,
                  AVG(pnl) avg_pnl,
                  SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) wins
           FROM scalp_trades
           WHERE entry_time >= ? AND exit_time IS NOT NULL
           GROUP BY ticker
           ORDER BY total_pnl DESC""",
        (cutoff,),
    )
    by_ticker = cur.fetchall()
    if by_ticker:
        print()
        print(f"=== ПО ТИКЕРАМ ===")
        print(f"  {'ticker':<8} {'n':>4} {'win%':>5} {'total_₽':>10} {'avg_₽':>9}")
        for r in by_ticker:
            wr = r["wins"] / r["n"] * 100 if r["n"] else 0
            print(
                f"  {r['ticker']:<8} {r['n']:>4} {wr:>4.0f}% "
                f"{(r['total_pnl'] or 0):>+10.2f} "
                f"{(r['avg_pnl'] or 0):>+9.2f}"
            )

    # ── 6. Exit reason breakdown ─────────────────────────────────────────
    cur.execute(
        """SELECT exit_reason, COUNT(*) n,
                  SUM(pnl) total_pnl,
                  SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) wins
           FROM scalp_trades
           WHERE entry_time >= ? AND exit_time IS NOT NULL
           GROUP BY exit_reason
           ORDER BY total_pnl DESC""",
        (cutoff,),
    )
    by_reason = cur.fetchall()
    if by_reason:
        print()
        print(f"=== ПО ПРИЧИНЕ ВЫХОДА ===")
        print(f"  {'reason':<14} {'n':>4} {'win%':>5} {'total_₽':>10}")
        for r in by_reason:
            wr = r["wins"] / r["n"] * 100 if r["n"] else 0
            print(
                f"  {(r['exit_reason'] or '?'):<14} {r['n']:>4} {wr:>4.0f}% "
                f"{(r['total_pnl'] or 0):>+10.2f}"
            )

    print()
    con.close()


if __name__ == "__main__":
    main()
