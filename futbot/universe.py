"""Resolve tier-1 / tier-2 base tickers to actual front-month FORTS contracts.

FORTS contracts have tickers like SiM6, SiU6, SiZ6 (March, September, December
of 2026) — the "base" is "Si".  We pick the contract whose `expiration_date`
is at least FUTBOT_MIN_DAYS_TO_EXPIRY days out — that's the "front month"
for trading purposes, skipping the about-to-expire one.

Re-resolved at startup and once per day (cheap — one find_instrument call
per base).  Future improvement: detect expiry approaching and trigger a roll.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger("futbot.universe")


@dataclass(frozen=True)
class Contract:
    base: str  # "Si"
    ticker: str  # "SiU6"
    figi: str
    tier: int  # 1 or 2
    expiration: datetime | None  # UTC
    instrument: object  # raw SDK Future object — kept for ГО / step_value lookup

    @property
    def days_to_expiry(self) -> int:
        if not self.expiration:
            return 999
        delta = self.expiration - datetime.now(timezone.utc)
        return max(0, delta.days)


async def resolve_universe(broker, settings) -> list[Contract]:
    """Resolve every base ticker in tier-1 + tier-2 to its front-month figi.

    Uses `get_all_futures()` once (more reliable than per-base find_instrument,
    which can throw on stray instrument-type enum values), then filters
    locally by the configured base prefixes.

    Returns only contracts with days_to_expiry >= FUTBOT_MIN_DAYS_TO_EXPIRY.
    """
    try:
        all_futures = await broker.get_all_futures()
    except Exception as e:
        logger.error(f"get_all_futures failed: {e}")
        return []

    bases_t1 = list(settings.FUTBOT_TIER1_BASES)
    bases_t2 = list(settings.FUTBOT_TIER2_BASES)
    min_dte = int(settings.FUTBOT_MIN_DAYS_TO_EXPIRY)

    out: list[Contract] = []
    for tier, bases in ((1, bases_t1), (2, bases_t2)):
        for base in bases:
            # Filter to futures whose ticker matches the base prefix.
            # Length check: a FORTS futures ticker is base + 2 chars (month+year)
            # — e.g. "Si" + "M6" → "SiM6", "BR" + "Q6" → "BRQ6".  Some tickers
            # are full names (EURRUBF) — accept exact match too.
            candidates = []
            for f in all_futures:
                t = getattr(f, "ticker", "") or ""
                is_prefix_match = t == base or (t.startswith(base) and len(t) == len(base) + 2)
                if not is_prefix_match:
                    continue
                exp = getattr(f, "expiration_date", None)
                if exp is None:
                    continue
                if hasattr(exp, "ToDatetime"):
                    exp = exp.ToDatetime()
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                candidates.append((f, exp))

            if not candidates:
                logger.warning(f"  {base}: no matching futures with expiration")
                continue

            # Sort by expiration ascending, pick the soonest > min_dte days out
            candidates.sort(key=lambda c: c[1])
            picked = None
            for f, exp in candidates:
                dte = (exp - datetime.now(timezone.utc)).days
                if dte >= min_dte:
                    picked = (f, exp)
                    break

            if picked is None:
                logger.info(f"  {base}: all contracts within {min_dte}d of expiry — skipping")
                continue

            f, exp = picked
            out.append(
                Contract(
                    base=base,
                    ticker=f.ticker,
                    figi=f.figi,
                    tier=tier,
                    expiration=exp,
                    instrument=f,
                )
            )

    logger.info(f"Universe resolved: {len(out)} contracts — " f"{', '.join(c.ticker for c in out)}")
    return out
