"""Utility helpers: Quotation/Decimal converters, MOEX schedule, timezone."""

from datetime import datetime, time, timezone, timedelta
from decimal import Decimal

MSK = timezone(timedelta(hours=3))

# MOEX main trading session + evening session
MOEX_OPEN = time(7, 0)  # 07:00 MSK
MOEX_CLOSE = time(23, 50)  # 23:50 MSK

# Weekend trading session (Tinkoff + MOEX "выходные торги")
# Only for shares with weekend_flag=True; no futures, no bonds.
MOEX_WEEKEND_OPEN = time(10, 0)  # 10:00 MSK Sat/Sun
MOEX_WEEKEND_CLOSE = time(19, 0)  # 19:00 MSK Sat/Sun
MOEX_HOLIDAYS_2026 = {
    # Known Russian market holidays (update annually)
    datetime(2026, 1, 1).date(),
    datetime(2026, 1, 2).date(),
    datetime(2026, 1, 3).date(),
    datetime(2026, 1, 7).date(),
    datetime(2026, 1, 8).date(),
    datetime(2026, 2, 23).date(),
    datetime(2026, 3, 8).date(),
    datetime(2026, 5, 1).date(),
    datetime(2026, 5, 9).date(),
    datetime(2026, 6, 12).date(),
    datetime(2026, 11, 4).date(),
    datetime(2026, 12, 31).date(),
}


def quotation_to_decimal(quotation) -> Decimal:
    """Convert tinkoff Quotation(units, nano) to Decimal."""
    if quotation is None:
        return Decimal(0)
    return Decimal(str(quotation.units)) + Decimal(str(quotation.nano)) / Decimal("1000000000")


def decimal_to_quotation(value: Decimal):
    """Convert Decimal to tinkoff Quotation."""
    from tinkoff.invest import Quotation

    units = int(value)
    nano = int((value - units) * Decimal("1000000000"))
    return Quotation(units=units, nano=nano)


def money_to_decimal(money) -> Decimal:
    """Convert tinkoff MoneyValue to Decimal."""
    if money is None:
        return Decimal(0)
    return Decimal(str(money.units)) + Decimal(str(money.nano)) / Decimal("1000000000")


def is_moex_open(dt: datetime | None = None) -> bool:
    """Check if MOEX regular session is open.

    Does NOT account for weekend trading — use ``is_market_tradeable()`` for
    a broader check that also accepts Sat/Sun sessions for weekend-enabled
    shares.
    """
    if dt is None:
        dt = datetime.now(MSK)
    else:
        dt = dt.astimezone(MSK)

    # Weekend check
    if dt.weekday() >= 5:
        return False

    # Holiday check
    if dt.date() in MOEX_HOLIDAYS_2026:
        return False

    # Time check
    current_time = dt.time()
    return MOEX_OPEN <= current_time <= MOEX_CLOSE


def is_weekend_session(dt: datetime | None = None) -> bool:
    """True if we're currently inside the Sat/Sun weekend trading window.

    Only weekend-enabled shares (weekend_flag=True on the Tinkoff instrument)
    trade during this window — futures, bonds and ETFs do not.
    """
    if dt is None:
        dt = datetime.now(MSK)
    else:
        dt = dt.astimezone(MSK)
    if dt.weekday() < 5:  # Mon-Fri: regular week, not a weekend session
        return False
    if dt.date() in MOEX_HOLIDAYS_2026:
        return False
    return MOEX_WEEKEND_OPEN <= dt.time() <= MOEX_WEEKEND_CLOSE


def is_market_tradeable(weekend_flag: bool = False, dt: datetime | None = None) -> bool:
    """True if the instrument can be traded right now.

    Accepts both the regular MOEX session and — if the instrument carries
    ``weekend_flag=True`` — the Sat/Sun weekend session.
    """
    if is_moex_open(dt):
        return True
    if weekend_flag and is_weekend_session(dt):
        return True
    return False


def now_msk() -> datetime:
    """Current datetime in MSK timezone."""
    return datetime.now(MSK)


def format_decimal(value: Decimal, decimals: int = 2) -> str:
    """Format decimal for display."""
    return f"{float(value):,.{decimals}f}"


def pct_change(old: Decimal, new: Decimal) -> float:
    """Calculate percentage change."""
    if old == 0:
        return 0.0
    return float((new - old) / old * 100)
