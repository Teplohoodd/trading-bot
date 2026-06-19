"""Centralised commission logic — Tinkoff «Трейдер» tariff.

Source of truth: `config.Settings.COMMISSION_*_PCT` from the parent
trade_claude project.  Both futbot (4-layer) and futbot.scalp (HFT-lite)
read them through this module so changing the tariff is a one-line edit.

Values (as of 2026-05, verified against the official PDF):
    SHARES   = 0.05 %   per side, 0.10 % round-trip
    FUTURES  = 0.04 %   per side, 0.08 % round-trip       (basic list)
    FUTURES  = 0.08 %   per side, 0.16 % round-trip       (additional list)
    CURRENCY = 0.50 %
    METALS   = 1.50 %

ADDITIONAL list FORTS contracts:
    Tinkoff hands MOEX a separate list of "less liquid" futures that
    carry the 0.08 % rate.  The list isn't fully published — bases below
    are based on common knowledge of the FORTS doc.  When in doubt the
    helper falls back to the basic rate; check `instrument_kind="future_plus"`
    if you want the double rate explicitly.  Override per-base via
    `FUTURES_PLUS_BASES` (env-overridable).

Tariff rule: «Рассчитанная комиссия... не может составлять менее 0.01 ₽».
Each side of every trade is floored at 0.01 ₽ in the math below.

API:
    commission_pct(kind="future")  →  0.0004
    round_trip_pnl(direction, entry, exit, lots, lot_size, rub_per_point, kind)
        →  (net_pnl, pnl_pct, gross_pnl, commission_rub)
"""

import os

# Bases (FORTS prefixes) that fall under the higher 0.08 % "additional"
# list.  Empty by default — most users won't hit these.  Override via env:
#    FUTBOT_FUTURES_PLUS_BASES=ED,UCNY,GLDRUBF
_PLUS_BASES = set(
    b.strip().upper()
    for b in os.environ.get("FUTBOT_FUTURES_PLUS_BASES", "").split(",")
    if b.strip()
)

# Module-level cache so we don't re-read Settings on every trade close.
_CACHED: dict | None = None


def _load() -> dict:
    """Lazily read commission rates from the parent project's Settings.
    Cached after first call."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    try:
        from config.settings import Settings  # trade_claude root

        s = Settings()
        _CACHED = {
            "share": float(getattr(s, "COMMISSION_SHARES_PCT", 0.0005)),
            "bond": float(getattr(s, "COMMISSION_SHARES_PCT", 0.0005)),
            "etf": float(getattr(s, "COMMISSION_SHARES_PCT", 0.0005)),
            "fund": float(getattr(s, "COMMISSION_SHARES_PCT", 0.0005)),
            "future": float(getattr(s, "COMMISSION_FUTURES_PCT", 0.0004)),
            "futures": float(getattr(s, "COMMISSION_FUTURES_PCT", 0.0004)),
            "currency": float(getattr(s, "COMMISSION_CURRENCY_PCT", 0.005)),
            "fx": float(getattr(s, "COMMISSION_CURRENCY_PCT", 0.005)),
            "metal": float(getattr(s, "COMMISSION_METALS_PCT", 0.015)),
            "metals": float(getattr(s, "COMMISSION_METALS_PCT", 0.015)),
            "precious_metal": float(getattr(s, "COMMISSION_METALS_PCT", 0.015)),
        }
    except Exception:
        # Settings unloadable → use Tinkoff Trader defaults
        _CACHED = {
            "share": 0.0005,
            "bond": 0.0005,
            "etf": 0.0005,
            "fund": 0.0005,
            "future": 0.0004,
            "futures": 0.0004,
            "currency": 0.005,
            "fx": 0.005,
            "metal": 0.015,
            "metals": 0.015,
            "precious_metal": 0.015,
        }
    return _CACHED


def commission_pct(instrument_kind: str = "share", base_ticker: str = "") -> float:
    """One-side commission as a fraction.

    For futures, pass `base_ticker` (e.g. "BR", "Si", "RT") so we can apply
    the doubled rate (0.08 %) when the contract is in the FORTS Additional
    list — listed in env `FUTBOT_FUTURES_PLUS_BASES`.  Defaults to the
    basic-list rate (0.04 %) — safer underestimate that matches MOEX's
    most-traded contracts.
    """
    rates = _load()
    kind = (instrument_kind or "share").lower()
    base_upper = (base_ticker or "").strip().upper()

    if kind in ("future", "futures", "future_plus", "futures_plus"):
        if kind in ("future_plus", "futures_plus") or base_upper in _PLUS_BASES:
            return 2 * rates["future"]  # 0.04 % → 0.08 % (additional list)
        return rates["future"]

    return rates.get(kind, rates["share"])


def round_trip_pnl(
    *,
    direction: str,
    entry_price: float,
    exit_price: float,
    lots: int,
    lot_size: int = 1,
    rub_per_point: float = 1.0,
    instrument_kind: str = "future",
    base_ticker: str = ""
) -> tuple[float, float, float, float]:
    """Compute the P&L of a closed trade NET of round-trip commission.

    Works for shares (lot_size = N shares per lot, rub_per_point = 1) AND
    for futures (lot_size = 1, rub_per_point from extract_futures_metadata).

    Returns:
        (net_pnl_rub, pnl_pct, gross_pnl_rub, commission_rub)

    `pnl_pct` is computed on the gross price move (matches what user sees
    on their chart) so it doesn't get distorted by commission.
    `net_pnl_rub` is what actually hits the account.
    """
    if entry_price <= 0 or lots <= 0 or lot_size <= 0:
        return 0.0, 0.0, 0.0, 0.0

    cpct = commission_pct(instrument_kind, base_ticker)
    # Notional one side = price × lots × lot_size × rub_per_point.  For
    # share futures (SR, GZ, LK, MX) lot_size is the multiplier (1 lot =
    # 100 shares of underlying), rub_per_point = 1.  For Si, BR, RT etc.
    # lot_size = 1 contract, rub_per_point = step_value / step_increment.
    notional_entry = entry_price * lots * lot_size * rub_per_point
    notional_exit = exit_price * lots * lot_size * rub_per_point
    # Per Tinkoff Trader rules: "Рассчитанная комиссия... не может
    # составлять менее 0,01 единицы".  Floor each side at 0.01 ₽,
    # round to 2 decimals (mathematical rounding) as the tariff specifies.
    side_entry = max(round(notional_entry * cpct, 2), 0.01)
    side_exit = max(round(notional_exit * cpct, 2), 0.01)
    commission = side_entry + side_exit

    if direction == "buy":
        gross_pnl = (exit_price - entry_price) * lots * lot_size * rub_per_point
        pnl_pct = (exit_price - entry_price) / entry_price * 100
    else:
        gross_pnl = (entry_price - exit_price) * lots * lot_size * rub_per_point
        pnl_pct = (entry_price - exit_price) / entry_price * 100

    net_pnl = gross_pnl - commission
    return round(net_pnl, 4), round(pnl_pct, 4), round(gross_pnl, 4), round(commission, 4)


def estimated_round_trip_cost(
    *,
    price: float,
    lots: int = 1,
    lot_size: int = 1,
    rub_per_point: float = 1.0,
    instrument_kind: str = "future",
    base_ticker: str = ""
) -> float:
    """Estimate round-trip commission BEFORE a trade.  Used by the
    edge-vs-commission gate to refuse entries whose expected profit
    can't beat fees.  Applies the same per-side 0.01 ₽ floor as the
    actual broker rules (Tinkoff Trader).
    """
    if price <= 0:
        return 0.0
    cpct = commission_pct(instrument_kind, base_ticker)
    notional = price * lots * lot_size * rub_per_point
    one_side = max(round(notional * cpct, 2), 0.01)
    return 2 * one_side
