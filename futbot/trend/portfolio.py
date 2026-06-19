"""Walk-forward-validated portfolio of FORTS contracts.

Each entry is a (base, N, k) parameter set that survived the 90d-IS / 90d-OOS
walk-forward test in `scripts/trend_walk_forward.py` (run 2026-05-20).

Fields:
    base      — FORTS ticker prefix, e.g. "GD" → resolves to GDM6 front-month
    n         — Bollinger lookback period (bars)
    k         — Bollinger σ multiplier
    oos_pnl   — out-of-sample NET ₽ over 90 days (reference for monitoring)
    oos_sh    — out-of-sample Sharpe (reference)
    oos_trd   — out-of-sample trades (sample size)
    notes     — what we think the base actually is (best-guess from ticker)

CRITICAL: this list is sorted by OOS Sharpe descending.  The first dozen
are the "core" portfolio (Sharpe > 0.7).  Lower-ranked contracts contribute
marginally — consider trimming if you want fewer concurrent positions.

Recalibrate every 60-90 days by rerunning the walk-forward script and
replacing this table.  β / regime drift WILL happen.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrendEntry:
    base: str
    n: int
    k: float
    oos_pnl: float
    oos_sh: float
    oos_trd: int
    notes: str = ""


# Walk-forward survivors (26 contracts).  Order: by OOS Sharpe desc.
PORTFOLIO: list[TrendEntry] = [
    TrendEntry("PX", 20, 2.5, 3958.38, 1.78, 8, "Polyus Gold (mining)"),
    TrendEntry("USDRU", 50, 2.0, 8.94, 1.77, 3, "USDRUBF currency"),
    TrendEntry("GD", 20, 1.5, 758.29, 1.26, 27, "Gold (workhorse, 9 trd/mo)"),
    TrendEntry("SS", 20, 2.5, 150.78, 1.09, 8, "?"),
    TrendEntry("SZ", 30, 2.0, 75.24, 0.90, 4, "?"),
    TrendEntry("YD", 20, 2.0, 252.64, 0.89, 17, "Yandex (5.7 trd/mo)"),
    TrendEntry("LT", 20, 1.5, 134.45, 0.85, 10, "?"),
    TrendEntry("S1", 20, 2.0, 13.95, 0.85, 21, "?"),
    TrendEntry("SOLUSDper", 30, 1.5, 17.58, 0.84, 34, "Solana perp (11 trd/mo)"),
    TrendEntry("VB", 80, 1.5, 472.20, 0.81, 7, "VTB"),
    TrendEntry("MV", 30, 2.0, 75.25, 0.77, 10, "?"),
    TrendEntry("GK", 30, 2.5, 97.93, 0.61, 8, "GMK Norilsk Nickel?"),
    TrendEntry("AMDper", 30, 2.0, 79.69, 0.60, 11, "AMD perp"),
    TrendEntry("SA", 20, 1.5, 1.17, 0.59, 8, "?"),
    TrendEntry("AK", 20, 1.5, 691.05, 0.54, 14, "?"),
    TrendEntry("GN", 20, 2.0, 143.48, 0.53, 8, "?"),
    TrendEntry("IB", 20, 2.5, 2.19, 0.47, 7, "?"),
    TrendEntry("TT", 50, 1.5, 835.91, 0.35, 4, "Tatneft"),
    TrendEntry("GL", 20, 1.5, 526.93, 0.33, 30, "Gold (TR), 10 trd/mo"),
    TrendEntry("GLDRU", 30, 1.5, 517.72, 0.33, 24, "GLDRUBF (gold-RUB)"),
    TrendEntry("EA", 50, 1.5, 108.75, 0.27, 6, "?"),
    TrendEntry("AN", 20, 1.5, 79.11, 0.20, 26, "?"),
    TrendEntry("SV", 30, 2.0, 3.15, 0.20, 19, "Silver?"),
    TrendEntry("CC", 30, 2.0, 7.82, 0.19, 6, "?"),
    TrendEntry("RN", 80, 1.5, 403.49, 0.10, 5, "Rosneft"),
    TrendEntry("SC", 50, 1.5, 5.89, 0.08, 9, "?"),
]


# ── Neo-asset / perpetual exclusion ─────────────────────────────────────
# "Neo-активы" (AMDper, SOLUSDper, *USDper, *perp) are Tinkoff instruments
# that TRACK US stocks / crypto.  They are technically futures but behave
# very differently and the bot does NOT model their mechanics:
#   • price is in USD, P&L credited in RUB at the CLOSE-day FX rate
#     (so our rub_per_point=1 P&L is wrong by the ~USD/RUB factor);
#   • daily holding commission (4.5 %+CB rate) and funding;
#   • no expiry → no rollover.
# Excluded from trading until proper Neo-asset support is built.  Trade only
# clean FORTS futures where (exit-entry)×point_value is the real RUB P&L.
NEO_EXCLUDE = {"AMDper", "SOLUSDper", "BTCUSDper", "ETHUSDper", "XRPUSDper", "TRXUSDper"}


def _is_neo(base: str) -> bool:
    return base in NEO_EXCLUDE or base.lower().endswith(("per", "perp"))


# Bases OWNED by another strategy — trend must NOT trade them or the two bots
# fight over the same contract (carry trades the GK calendar spread, so trend
# shorting/longing GKM6 breaks the spread's delta-neutrality + reconciliation).
CARRY_OWNED = {"GK"}


def _excluded(base: str) -> bool:
    return _is_neo(base) or base in CARRY_OWNED


# Subsets for tiered deployment:
#   Core: best Sharpe + decent activity (Sharpe>0.6 + ≥7 OOS trades)
#   Extended: everything that passed WF
# Neo assets / perpetuals are filtered out of BOTH (see NEO_EXCLUDE above).
def core_portfolio() -> list[TrendEntry]:
    return [e for e in PORTFOLIO if e.oos_sh >= 0.6 and e.oos_trd >= 7 and not _excluded(e.base)]


def tradeable_portfolio() -> list[TrendEntry]:
    """Full WF portfolio minus Neo assets and carry-owned bases."""
    return [e for e in PORTFOLIO if not _excluded(e.base)]


# ── Neo-asset portfolio ─────────────────────────────────────────────────
# Validated by neo_backtest.py (2026-06: triple_top +196 % on-margin, 66 %
# win, OOS-positive on 90d).  These are the liquid Neo assets with enough
# pattern activity.  Neo mechanics (USD price → RUB P&L at close FX, daily
# holding fee, NO expiry, high leverage) are handled in the trend bot.
# `base` is the full perpA ticker (Neo tickers have no month code).
# Kept only Neo with backtest win-rate ≥ 60% AND positive on-margin total.
# Crypto Neo (ETHUSD/SOLUSD/BTCUSD/XRPUSD/TRXUSD) REQUIRE quals status — see
# Tinkoff docs: "Чтобы торговать неоактивами на криптовалюту, нужно получить
# статус квалифицированного инвестора".  Without it the broker rejects orders
# with FAILED_PRECONDITION 90002.  Excluded until the user is qualified.
# REMOVED also (triple_top fails): BTC 48% win, TSLA 50%, COIN 38%, NFLX 22%.
# (BTC/TSLA still available for DISPATCH channel-signal trades.)
NEO_PORTFOLIO: list[TrendEntry] = [
    TrendEntry("NBISperpA", 0, 0, 42.1, 0.0, 6, "Nebius   100% win, +90% onMargin"),
    TrendEntry("CVNAperpA", 0, 0, 24.4, 0.0, 5, "Carvana  100% win, +97%"),
    TrendEntry("APPperpA", 0, 0, 18.3, 0.0, 6, "Applovin  67% win, +45%"),
    TrendEntry("HOODperpA", 0, 0, 5.2, 0.0, 10, "Robinhood 60% win, +15%"),
    # --- CRYPTO (commented out — re-enable after quals status) -----------
    # TrendEntry("ETHUSDperpA", 0, 0, 25.3, 0.0, 21, "Ethereum  71% win, +67%"),
    # TrendEntry("TRXUSDperpA", 0, 0, 10.0, 0.0, 13, "Tron      69% win, +36%"),
    # TrendEntry("XRPUSDperpA", 0, 0,  5.1, 0.0, 21, "Ripple    67% win, +11%"),
    # TrendEntry("SOLUSDperpA", 0, 0,  1.9, 0.0, 27, "Solana    63% win, +5%"),
]


def neo_portfolio() -> list[TrendEntry]:
    return list(NEO_PORTFOLIO)


def by_base(base: str) -> TrendEntry | None:
    for e in PORTFOLIO:
        if e.base == base:
            return e
    return None
