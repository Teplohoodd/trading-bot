"""Telegram command handlers — aggregate across pairs + trend.

Commands available:
    /status   — short summary of all enabled strategies
    /pairs    — detailed pairs status
    /trend    — detailed trend status
    /open     — every open position across all subsystems
    /pnl      — today + 7-day NET per strategy
    /help     — list of commands
"""

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("orchestrator.commands")


def build_handlers(*, pairs_bot=None, trend_bot=None, carry_bot=None) -> dict:
    """Return {cmd_name: async_callable} ready for TelegramCommandServer."""

    async def cmd_status():
        parts = []
        if pairs_bot is not None and pairs_bot._initialised:
            parts.append(await pairs_bot.status())
        if trend_bot is not None and trend_bot._initialised:
            parts.append(await trend_bot.status())
        if carry_bot is not None and carry_bot._initialised:
            parts.append(await carry_bot.status())
        if not parts:
            return "No subsystems initialised yet."
        # Add overall combined P&L
        total_today = 0.0
        if pairs_bot and pairs_bot.db:
            total_today += await pairs_bot.db.daily_pnl_rub(datetime.utcnow().date().isoformat())
        if trend_bot and trend_bot.db:
            total_today += await trend_bot.db.daily_pnl_rub(datetime.utcnow().date().isoformat())
        if carry_bot and carry_bot.db:
            total_today += await carry_bot.db.daily_pnl_rub(datetime.utcnow().date().isoformat())
        parts.append(f"\n<b>COMBINED today: {total_today:+.2f} ₽</b>")
        return "\n\n".join(parts)

    async def cmd_pairs():
        if pairs_bot is None:
            return "Pairs subsystem disabled."
        return await pairs_bot.status()

    async def cmd_trend():
        if trend_bot is None:
            return "Trend subsystem disabled."
        return await trend_bot.status()

    async def cmd_carry():
        if carry_bot is None:
            return "Carry subsystem disabled."
        return await carry_bot.status()

    async def cmd_open():
        lines = ["<b>OPEN POSITIONS</b>"]
        any_open = False
        if pairs_bot and pairs_bot.db:
            open_p = await pairs_bot.db.open_trades()
            if open_p:
                any_open = True
                lines.append(f"\n<b>Pairs ({len(open_p)}):</b>")
                for r in open_p:
                    entry_dt = datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00"))
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
                    side = "LONG" if r["direction"] > 0 else "SHORT"
                    lines.append(
                        f"  {r['pair']:<10} {side}  z={r['entry_z']:+.2f}  " f"held={held:.1f}h"
                    )
        if trend_bot and trend_bot.db:
            open_t = await trend_bot.db.open_trades()
            if open_t:
                any_open = True
                lines.append(f"\n<b>Trend ({len(open_t)}):</b>")
                for r in open_t:
                    entry_dt = datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00"))
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
                    lines.append(
                        f"  {r['base']:<6} {r['direction']:>4}  "
                        f"@ {r['entry_price']:>9.4f}  held={held:.1f}h"
                    )
        if carry_bot and carry_bot.db:
            open_c = await carry_bot.db.open_trades()
            if open_c:
                any_open = True
                lines.append(f"\n<b>Carry ({len(open_c)}):</b>")
                for r in open_c:
                    entry_dt = datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00"))
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    held = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 3600
                    side = "LONG" if r["direction"] > 0 else "SHORT"
                    lines.append(
                        f"  {r['pair']:<12} {side} basis z={r['entry_z']:+.2f}  "
                        f"held={held:.1f}h"
                    )
        if not any_open:
            lines.append("\n(none)")
        return "\n".join(lines)

    async def cmd_pnl():
        lines = ["<b>P&L SUMMARY</b>"]
        today_iso = datetime.utcnow().date().isoformat()
        week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()

        def _fmt_strategy(name: str, today: float, week: float, n: int, wins: int) -> str:
            wr = wins / n * 100 if n else 0
            return (
                f"\n<b>{name}:</b>\n"
                f"  Today: {today:+.2f} ₽\n"
                f"  7-day: {week:+.2f} ₽ ({n} trades, win {wr:.0f}%)"
            )

        total_today = 0.0
        total_week = 0.0

        if pairs_bot and pairs_bot.db:
            today_p = await pairs_bot.db.daily_pnl_rub(today_iso)
            async with pairs_bot.db._lock:
                # Exclude over-leveraged historical trades (kept in DB
                # for audit but their P&L isn't reflective of edge).
                cur = await pairs_bot.db._db.execute(
                    "SELECT COUNT(*) n, "
                    "SUM(CASE WHEN pnl_rub > 0 THEN 1 ELSE 0 END) wins, "
                    "COALESCE(SUM(pnl_rub), 0) total "
                    "FROM pair_trades WHERE exit_time >= ? "
                    "AND (exit_reason IS NULL "
                    "     OR (exit_reason NOT LIKE '%_OVERLEV' "
                    "         AND exit_reason NOT LIKE 'sizing_bug_%'))",
                    (week_ago,),
                )
                row = await cur.fetchone()
            week_p = row[2] if row else 0.0
            n_p = row[0] if row else 0
            wins_p = row[1] if row else 0
            lines.append(_fmt_strategy("Pairs", today_p, week_p, n_p, wins_p))
            total_today += today_p
            total_week += week_p

        if trend_bot and trend_bot.db:
            today_t = await trend_bot.db.daily_pnl_rub(today_iso)
            async with trend_bot.db._lock:
                cur = await trend_bot.db._db.execute(
                    "SELECT COUNT(*) n, "
                    "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) wins, "
                    "COALESCE(SUM(pnl), 0) total "
                    "FROM trend_trades WHERE exit_time >= ?",
                    (week_ago,),
                )
                row = await cur.fetchone()
            week_t = row[2] if row else 0.0
            n_t = row[0] if row else 0
            wins_t = row[1] if row else 0
            lines.append(_fmt_strategy("Trend", today_t, week_t, n_t, wins_t))
            total_today += today_t
            total_week += week_t

        if carry_bot and carry_bot.db:
            today_c = await carry_bot.db.daily_pnl_rub(today_iso)
            async with carry_bot.db._lock:
                cur = await carry_bot.db._db.execute(
                    "SELECT COUNT(*) n, "
                    "SUM(CASE WHEN pnl_rub > 0 THEN 1 ELSE 0 END) wins, "
                    "COALESCE(SUM(pnl_rub), 0) total "
                    "FROM pair_trades WHERE exit_time >= ?",
                    (week_ago,),
                )
                row = await cur.fetchone()
            week_c = row[2] if row else 0.0
            n_c = row[0] if row else 0
            wins_c = row[1] if row else 0
            lines.append(_fmt_strategy("Carry", today_c, week_c, n_c, wins_c))
            total_today += today_c
            total_week += week_c

        lines.append(
            f"\n<b>COMBINED:</b>\n"
            f"  Today: {total_today:+.2f} ₽\n"
            f"  7-day: {total_week:+.2f} ₽"
        )
        return "\n".join(lines)

    async def cmd_help():
        return (
            "<b>Available commands:</b>\n"
            "  /status — quick summary of all strategies\n"
            "  /pairs  — detailed pairs subsystem\n"
            "  /trend  — detailed trend subsystem\n"
            "  /carry  — Si calendar-spread carry\n"
            "  /open   — every open position\n"
            "  /pnl    — today + 7-day P&L\n"
            "  /help   — this list"
        )

    return {
        "status": cmd_status,
        "pairs": cmd_pairs,
        "trend": cmd_trend,
        "carry": cmd_carry,
        "open": cmd_open,
        "pnl": cmd_pnl,
        "help": cmd_help,
    }
