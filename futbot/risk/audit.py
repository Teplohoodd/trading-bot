"""Final risk audit — the last veto layer before an order is placed.

Sits AFTER the pipeline (a trade decision has been computed) and BEFORE the
sizer / order placement.  Any of its checks can reject the trade.  Per
TradingAgents convention: risk is a separate concern from "do I have a
signal" — even a perfectly clean signal can be wrong to trade right now
because of portfolio state or market microstructure.

Checks (in order):
  1. Hour-of-day blacklist (MSK = UTC+3)
  2. Days-to-expiry (roll window guard)
  3. Total ГО usage cap across all open positions + this candidate
  4. Per-contract daily P&L floor (don't double-down on a bad day)
  5. Total daily P&L kill-switch
  6. Spread guard (Glosten-Milgrom: if spread is wide, liquidity is fake)
  7. Max open contracts cap
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

logger = logging.getLogger("futbot.audit")


MSK_OFFSET = timedelta(hours=3)


@dataclass
class AuditResult:
    approved: bool
    reason: str
    detail: dict


async def audit(
    *,
    broker,
    db,
    contract,
    direction: str,
    proposed_lots: int,
    initial_margin: float | None,
    order_book: dict | None,
    settings,
) -> AuditResult:
    detail: dict = {}

    # 1 — Hour blacklist
    now_msk = datetime.now(timezone.utc) + MSK_OFFSET
    hour = now_msk.hour
    detail["hour_msk"] = hour
    if hour in list(settings.FUTBOT_BLACKOUT_HOURS_MSK):
        return AuditResult(False, f"blackout hour MSK={hour}", detail)

    # 2 — Roll window
    dte = contract.days_to_expiry
    detail["days_to_expiry"] = dte
    if dte < int(settings.FUTBOT_MIN_DAYS_TO_EXPIRY):
        return AuditResult(False, f"too close to expiry ({dte}d)", detail)

    # 3 — ГО usage cap
    open_rows = await db.open_trades()
    used_go = 0.0
    for r in open_rows:
        if r["initial_margin"] and r["lots"]:
            used_go += float(r["initial_margin"]) * int(r["lots"])
    candidate_go = float(initial_margin or 0) * int(proposed_lots)
    try:
        portfolio_value = float(await broker.get_portfolio_value())
    except Exception:
        portfolio_value = 0.0
    cap_go = portfolio_value * float(settings.FUTBOT_MAX_GO_PCT)
    detail.update(
        {
            "portfolio_value": round(portfolio_value, 2),
            "used_go": round(used_go, 2),
            "candidate_go": round(candidate_go, 2),
            "cap_go": round(cap_go, 2),
        }
    )
    if portfolio_value > 0 and (used_go + candidate_go) > cap_go:
        return AuditResult(
            False,
            f"ГО cap exceeded: used {used_go:.0f} + candidate {candidate_go:.0f} > {cap_go:.0f}",
            detail,
        )

    # 4 — Per-contract daily loss cap (only if contract has prior open)
    # (Optional, applied on the figi: if today's realized loss on this
    # contract already exceeds the per-contract cap, refuse re-entry.)
    pct_cap_contract = float(settings.FUTBOT_MAX_CONTRACT_DAILY_LOSS_PCT)
    if pct_cap_contract > 0:
        today_iso = datetime.utcnow().date().isoformat()
        async with db._lock:
            cur = await db._db.execute(
                "SELECT COALESCE(SUM(pnl),0) FROM trades "
                "WHERE figi=? AND date(exit_time)=? AND pnl IS NOT NULL",
                (contract.figi, today_iso),
            )
            row = await cur.fetchone()
        contract_today_pnl = float(row[0]) if row else 0.0
        contract_loss_cap = portfolio_value * pct_cap_contract
        detail["contract_today_pnl"] = round(contract_today_pnl, 2)
        if portfolio_value > 0 and contract_today_pnl < -contract_loss_cap:
            return AuditResult(
                False,
                f"per-contract daily loss {contract_today_pnl:.0f} < -{contract_loss_cap:.0f}",
                detail,
            )

    # 5 — Total daily kill-switch
    pct_cap_total = float(settings.FUTBOT_MAX_DAILY_LOSS_PCT)
    if pct_cap_total > 0:
        today_iso = datetime.utcnow().date().isoformat()
        total_today = await db.daily_pnl(today_iso)
        total_cap = portfolio_value * pct_cap_total
        detail["total_today_pnl"] = round(total_today, 2)
        if portfolio_value > 0 and total_today < -total_cap:
            return AuditResult(
                False,
                f"daily kill-switch: total today {total_today:.0f} < -{total_cap:.0f}",
                detail,
            )

    # 6 — Spread guard
    # `order_book` arg is optional; when None we skip this check.
    if order_book and order_book.get("bid") and order_book.get("ask"):
        bid = float(order_book["bid"])
        ask = float(order_book["ask"])
        mid = (bid + ask) / 2 if (bid + ask) > 0 else 0
        if mid > 0:
            spread_pct = (ask - bid) / mid * 100
            detail["spread_pct"] = round(spread_pct, 4)
            # We don't have a running median here without extra state; use a
            # hard reasonableness cap of 0.20% for FORTS Si/Br-class names.
            if spread_pct > 0.20:
                return AuditResult(False, f"spread {spread_pct:.3f}% > 0.20% — too wide", detail)

    # 7 — Max open contracts cap
    max_open = int(settings.FUTBOT_MAX_OPEN_CONTRACTS)
    open_count = len(open_rows)
    detail["open_contracts"] = open_count
    if open_count >= max_open:
        # Allow if it's an opposing exit-then-flip on the SAME figi (handled
        # at decision-loop level), but for a fresh contract the cap binds.
        already_in_figi = any(r["figi"] == contract.figi for r in open_rows)
        if not already_in_figi:
            return AuditResult(
                False,
                f"max open contracts cap ({open_count}/{max_open})",
                detail,
            )

    return AuditResult(True, "approved", detail)
