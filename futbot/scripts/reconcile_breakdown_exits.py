"""One-off: reconcile breakdown trades whose exit P&L was fabricated from the
current price (the closed_externally bug, fixed 2026-06-24).  Looks up the REAL
covering-trade price from the broker's operations history and rewrites pnl_rub.

Read-only broker call (get_operations) + a targeted UPDATE on already-closed
rows the running bot won't touch again.  Safe to run alongside the orchestrator.

Usage:  python -m futbot.scripts.reconcile_breakdown_exits          # dry-run
        python -m futbot.scripts.reconcile_breakdown_exits --apply  # write DB
"""

import argparse
import asyncio
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.broker import BrokerClient                       # noqa: E402
from futbot.breakdown.config import BreakdownSettings      # noqa: E402
from futbot.orchestrator.config import OrchSettings        # noqa: E402

TRADE_IDS = [14, 16]   # the two suspect rows; breakdown is always SHORT → BUY


async def real_exit_price(broker, figi, since_iso):
    """Covering BUY trades for this short, via broker.get_operations (cursor
    endpoint, SERVER-SIDE type filter).  The old raw-protobuf hack — needed
    because the abandoned SDK couldn't deserialize this account's operations
    (ValueError: 66 is not a valid OperationType) — is gone: t-tech parses
    cleanly."""
    from t_tech.invest import OperationType
    from t_tech.invest.utils import quotation_to_decimal
    since = datetime.fromisoformat(since_iso)
    if since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    ops = await broker.get_operations(
        from_dt=since,
        operation_types=[OperationType.OPERATION_TYPE_BUY,
                         OperationType.OPERATION_TYPE_BUY_MARGIN])
    out = []
    for op in ops:
        if getattr(op, "figi", None) != figi:
            continue
        px = float(quotation_to_decimal(op.price)) if op.price else None
        out.append((getattr(op, "date", None), px,
                    int(getattr(op, "quantity", 0) or 0)))
    out.sort(key=lambda t: t[0] or datetime.min.replace(tzinfo=timezone.utc),
             reverse=True)
    return out


async def main(apply: bool):
    bset = BreakdownSettings()
    oset = OrchSettings()
    con = sqlite3.connect(bset.BD_DB_PATH)
    con.row_factory = sqlite3.Row
    rows = {r["id"]: r for r in con.execute(
        f"SELECT * FROM bd_trades WHERE id IN ({','.join('?'*len(TRADE_IDS))})",
        TRADE_IDS)}

    broker = BrokerClient(token=oset.T_INVEST_TOKEN,
                          account_id=oset.T_INVEST_ACCOUNT_ID,
                          app_name="bd-reconcile")
    await broker.connect()
    try:
        for tid in TRADE_IDS:
            r = rows.get(tid)
            if r is None:
                print(f"#{tid}: not found"); continue
            print(f"\n#{tid} {r['stock']} {r['fut_ticker']} ×{r['lots']} "
                  f"entry={r['entry_price']:.2f} ({r['entry_time'][:19]})")
            print(f"   recorded: exit={r['exit_price']:.2f} "
                  f"pnl={r['pnl_rub']:+.2f} reason={r['exit_reason']}")
            ops = await real_exit_price(broker, r["fut_figi"], r["entry_time"])
            if not ops:
                print("   NO covering BUY op found in history — leaving as-is "
                      "(may predate retention / different figi).")
                continue
            print("   covering BUY ops since entry (newest first):")
            for d, px, qty in ops[:5]:
                ds = d.isoformat()[:19] if d else "?"
                print(f"     {ds}  price={px}  qty={qty}")
            real_px = ops[0][1]
            if real_px is None or real_px <= 0:
                print("   op price unusable — leaving as-is."); continue
            new_pnl = (r["entry_price"] - real_px) * r["lots"] \
                * r["lot_size"] * r["rpp"]
            print(f"   => REAL exit={real_px:.2f}  pnl={new_pnl:+.2f}  "
                  f"(was {r['pnl_rub']:+.2f}, delta {new_pnl - r['pnl_rub']:+.2f})")
            if apply:
                con.execute(
                    "UPDATE bd_trades SET exit_price=?, pnl_rub=?, "
                    "exit_reason=? WHERE id=?",
                    (real_px, new_pnl,
                     r["exit_reason"].replace("~approx", "") + "_reconciled",
                     tid))
                con.commit()
                print("   ✔ DB updated")
    finally:
        await broker.disconnect()
        con.close()
    if not apply:
        print("\n[dry-run] nothing written. Re-run with --apply to commit.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.apply))
