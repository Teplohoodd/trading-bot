"""Kill-switch — checked once per loop, independent of any single trade.

If tripped, the main loop stops opening NEW positions until the next UTC
midnight (when the daily P&L counter resets).  Existing positions remain
open and continue to be managed by the trailing stop / time-cap logic.

This is intentionally separate from `audit.py`'s daily P&L check:
  * audit blocks per-trade approvals;
  * circuit prevents the loop from even *running* the pipeline for new
    entries (saves API calls, telegram noise, log spam).
"""

import logging
from datetime import datetime, timezone, date

logger = logging.getLogger("futbot.circuit")


class CircuitBreaker:
    def __init__(self, settings):
        self.settings = settings
        self._tripped_date: date | None = None
        self._tripped_reason: str = ""

    async def is_tripped(self, *, broker, db) -> tuple[bool, str]:
        # Reset on new UTC day
        today = datetime.now(timezone.utc).date()
        if self._tripped_date is not None and self._tripped_date < today:
            logger.info(
                f"Circuit breaker auto-reset on new UTC day "
                f"(was tripped {self._tripped_date} for: {self._tripped_reason})"
            )
            self._tripped_date = None
            self._tripped_reason = ""

        if self._tripped_date == today:
            return True, self._tripped_reason

        # Test the kill-switch condition
        pct_cap = float(self.settings.FUTBOT_MAX_DAILY_LOSS_PCT)
        if pct_cap <= 0:
            return False, ""
        try:
            portfolio_value = float(await broker.get_portfolio_value())
        except Exception:
            return False, ""
        if portfolio_value <= 0:
            return False, ""

        today_iso = today.isoformat()
        total_today = await db.daily_pnl(today_iso)
        cap = portfolio_value * pct_cap
        if total_today < -cap:
            self._tripped_date = today
            self._tripped_reason = (
                f"daily kill-switch tripped: P&L {total_today:.0f} ₽ < -{cap:.0f} ₽ "
                f"({pct_cap*100:.1f}% of portfolio)"
            )
            logger.error(self._tripped_reason)
            return True, self._tripped_reason
        return False, ""
