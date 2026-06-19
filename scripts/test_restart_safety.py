"""Restart-safety integration test.

Simulates a bot restart with active in-memory state (trailing peaks +
cooldowns) and verifies that hydration from DB recovers it correctly.

Steps:
  1. Open a fresh test repository pointed at a temp DB.
  2. Insert a synthetic OPEN trade row.
  3. Persist a position_peak (peak_price, activated, partial_taken).
  4. Persist a cooldown for that figi.
  5. CLOSE the repo (simulates bot shutdown).
  6. Re-open repo, hydrate state via the same paths the engine uses, and
     assert the loaded state matches what we wrote.

Run: python -m scripts.test_restart_safety
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from database.db import Repository, Trade

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("restart_safety")


async def run() -> int:
    failures: list[str] = []

    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "restart_test.db"

        # ---- Phase 1: write state ------------------------------------------
        db = Repository(db_path)
        await db.initialize()

        trade = Trade(
            figi="BBG004730N88",
            ticker="SBER",
            direction="buy",
            lots=10,
            entry_price=300.0,
            strategy="test",
            signal_confidence=0.75,
            entry_time=datetime.now(timezone.utc).isoformat(),
            stop_loss=290.0,
            take_profit=320.0,
            lot_size=10,
        )
        trade_id = await db.insert_trade(trade)
        log.info(f"Inserted synthetic open trade id={trade_id}")

        await db.upsert_position_peak(
            trade_id=trade_id,
            peak_price=315.5,
            activated=True,
            partial_taken=True,
            partial_price=312.0,
        )
        await db.upsert_cooldown("BBG004730N88", "2026-04-30T08:00:00+00:00")
        await db.upsert_cooldown("BBG00BFXMP82", "2026-04-30T09:30:00+00:00")
        log.info("Wrote position_peak + 2 cooldowns")

        await db.close()
        log.info("Phase 1 done — repo closed (simulating bot shutdown)")

        # ---- Phase 2: re-open and hydrate (same paths the engine uses) -----
        db2 = Repository(db_path)
        await db2.initialize()

        peaks = await db2.get_position_peaks()
        cooldowns = await db2.get_cooldowns()
        open_trades = await db2.get_open_trades()

        log.info(
            f"After restart: {len(open_trades)} open trade(s), "
            f"{len(peaks)} peak(s), {len(cooldowns)} cooldown(s)"
        )

        # ---- Assertions ----------------------------------------------------
        if len(open_trades) != 1:
            failures.append(f"open_trades count: expected 1, got {len(open_trades)}")
        elif open_trades[0]["id"] != trade_id:
            failures.append(f"open_trade id: expected {trade_id}, got {open_trades[0]['id']}")

        peak_info = peaks.get(trade_id)
        if peak_info is None:
            failures.append(f"position_peak missing for trade_id={trade_id}")
        else:
            if abs(peak_info["peak_price"] - 315.5) > 1e-6:
                failures.append(f"peak_price: expected 315.5, got {peak_info['peak_price']}")
            if peak_info["activated"] is not True:
                failures.append(f"activated: expected True, got {peak_info['activated']}")
            if peak_info["partial_taken"] is not True:
                failures.append(f"partial_taken: expected True, got {peak_info['partial_taken']}")
            if abs((peak_info["partial_price"] or 0) - 312.0) > 1e-6:
                failures.append(f"partial_price: expected 312.0, got {peak_info['partial_price']}")

        if len(cooldowns) != 2:
            failures.append(f"cooldowns count: expected 2, got {len(cooldowns)}")
        if "BBG004730N88" not in cooldowns:
            failures.append("SBER cooldown missing")
        if "BBG00BFXMP82" not in cooldowns:
            failures.append("second cooldown missing")

        # ---- Test peak update + delete -------------------------------------
        await db2.upsert_position_peak(
            trade_id=trade_id,
            peak_price=320.7,
            activated=True,
            partial_taken=True,
            partial_price=312.0,
        )
        peaks2 = await db2.get_position_peaks()
        if abs(peaks2[trade_id]["peak_price"] - 320.7) > 1e-6:
            failures.append(f"peak update failed: got {peaks2[trade_id]['peak_price']}")

        await db2.delete_position_peak(trade_id)
        peaks3 = await db2.get_position_peaks()
        if trade_id in peaks3:
            failures.append(f"peak delete failed: trade_id={trade_id} still present")

        # ---- Per-direction stats sanity ------------------------------------
        buy_stats = await db2.get_trade_stats(direction="buy")
        sell_stats = await db2.get_trade_stats(direction="sell")
        if "kelly_f" not in buy_stats or "kelly_f" not in sell_stats:
            failures.append("get_trade_stats missing kelly_f")

        await db2.close()

    if failures:
        for f in failures:
            log.error(f"FAIL: {f}")
        return 1
    log.info("PASS: all restart-safety assertions held")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(run()))
