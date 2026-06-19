"""ScalpSettings — scalp-specific config layered on top of the parent .env.

Reuses the same .env-finding logic as FutSettings, plus a `SCALP_` prefix
namespace so its knobs are independent of the 4-layer pipeline's settings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings

from futbot.config import _ENV_FILE


class ScalpSettings(BaseSettings):
    # Credentials (reused from parent .env)
    T_INVEST_TOKEN: str
    T_INVEST_ACCOUNT_ID: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: int = 0

    # ── Safety: paper mode ON by default ─────────────────────────────────
    SCALP_PAPER_MODE: bool = True

    # ── Universe ─────────────────────────────────────────────────────────
    # Most-liquid FORTS contracts ONLY.  Scalping illiquid names = guaranteed
    # slippage loss.  Keep this short and curated.
    # Default universe — only contracts where commission math AND live
    # results agree:
    #   * Si — market orders disabled at Tinkoff
    #   * SR — 32k₽ notional × 0.08% RT = 26₽ vs 15m-ATR ≈ 4₽ → never makes math
    #   * MX — 267k₽ × 0.08% RT = 213₽.  Math BARELY passes the gate but
    #     live 2026-05-17: 2 trades = −352 ₽ (one had +50pt favourable move
    #     yet still lost to commission because exit hit before TP).  Re-enable
    #     only when we switch to limit-aggressive orders.
    # BR + GZ remain — both showed positive gross moves yesterday but
    # got eaten by signal_flip / time_cap.  Fixing those exits next.
    # scalp v2 (2026-05-20): only BR.  60d backtest of approximated
    # book+tfi proxies showed GZ at −280 ₽ NET / 27 trades, BR at −12 ₽
    # / 111 trades (essentially break-even pre-commission).  BR has the
    # lowest notional (110 ₽) → smallest commission damage per trade,
    # giving microstructure edge the most room to surface.  Add GZ back
    # only after seeing positive paper results on BR.
    SCALP_TIER1_BASES: list = ["BR"]
    SCALP_MIN_DAYS_TO_EXPIRY: int = 14
    SCALP_MAX_OPEN_POSITIONS: int = 2  # across all contracts combined

    # ── Microstructure signal thresholds ─────────────────────────────────
    # Book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).  Range [-1, +1].
    # +0.3 = strong buy pressure; -0.3 = strong sell pressure.  Threshold is
    # the MINIMUM |imbalance| required for an entry signal.
    SCALP_BOOK_IMBALANCE_MIN: float = 0.25
    # Order book depth to subscribe to.  10 levels is the sweet spot — deeper
    # books are mostly stale quotes; shallower miss real liquidity.
    SCALP_BOOK_DEPTH: int = 10
    # Trade flow imbalance (TFI) over rolling N seconds: (buy_vol − sell_vol) /
    # total.  Combined with book imbalance gives strongest signal.
    SCALP_TFI_WINDOW_SEC: int = 30
    SCALP_TFI_MIN: float = 0.15
    # Number of trades required in the rolling window for TFI to be meaningful.
    SCALP_TFI_MIN_TRADES: int = 5

    # ── Indicator confirmations on 1-min bars ────────────────────────────
    # Fast RSI on 1-min bars.  Below buy → long-bias, above sell → short-bias.
    SCALP_RSI_PERIOD: int = 7  # fast RSI, not the textbook 14
    SCALP_RSI_BUY_BELOW: float = 40.0
    SCALP_RSI_SELL_ABOVE: float = 60.0
    # VWAP distance: how far is price from session VWAP, normalized by ATR?
    # Negative for longs (we want to buy below VWAP for mean reversion).
    SCALP_VWAP_DEV_ATR_MAX: float = 0.5
    # EMA(9) > EMA(21) on 1-min = trend bias for long (and inverse).  Optional
    # confirmation, can be disabled by setting weight=0 in strategy.
    SCALP_EMA_FAST: int = 9
    SCALP_EMA_SLOW: int = 21

    # ── Risk per trade ───────────────────────────────────────────────────
    # All in ATR units of the 15-MIN bar (not 1-min).  The 1-min ATR is too
    # small relative to broker commission (0.04% of notional) — TP at
    # 1.5× ATR_1m fails the edge-vs-commission gate on every contract.
    # The 15-min ATR is ~3-4× larger (square-root-of-time scaling),
    # which lets TP targets actually exceed round-trip commission cost.
    # scalp v2 hold/TP/SL (2026-05-20):
    # Microstructure signal (book + tfi) operates on seconds-minutes
    # horizon — holding 15 min washes out the signal.  Tighter all-round:
    #   * Hold 3 min (was 15) — exit before signal stales
    #   * TP 1.0 × ATR_15m (was 2.0) — TP must be reachable in 1-2 min on
    #     a real movement, not require the 15-min ATR to play out
    #   * SL 0.8 × ATR_15m (was 1.2) — tighter for symmetric R:R 0.8 win
    SCALP_INITIAL_STOP_ATR: float = 0.8
    SCALP_TAKE_PROFIT_ATR: float = 1.0
    SCALP_MAX_HOLD_SECONDS: int = 180  # 3 min — microstructure horizon
    SCALP_TRAIL_ACTIVATE_ATR: float = 0.5  # trail activates at +0.5 ATR

    # Early-abandon for flat positions.  Live 2026-05-19: 26 GZ time_cap
    # trades cost −271 ₽ — almost all were positions that drifted in noise
    # for 15 min, then closed at near-zero pnl with commission damage.
    # If after EARLY_ABANDON_AFTER_SEC the position hasn't moved by
    # at least EARLY_ABANDON_PROFIT_ATR in profit, close it now — smaller
    # commission damage than waiting for time_cap on the same outcome.
    SCALP_EARLY_ABANDON_AFTER_SEC: int = 300  # 5 min
    SCALP_EARLY_ABANDON_PROFIT_ATR: float = 0.15  # any profit < 0.15×ATR = "flat"

    # signal_flip exits — minimum conditions before we honour them.
    # Live observation 2026-05-17: signal_flip exits cost ~−212 ₽ across 16
    # trades, because the order-book + TFI signal flips on noise within 1-2
    # minutes of entry and we'd close losers AT entry-price, paying 2×
    # commission for nothing.  Two gates now:
    #   1. Minimum age — no flip during first N seconds (give the move a
    #      chance to start).
    #   2. Minimum profit — must already be in profit by some ATR fraction
    #      before flip is treated as "lock in a winner" instead of "panic
    #      out of a wobble".
    # If neither condition met, the flip is ignored and the position rides
    # to stop / TP / time_cap.
    SCALP_FLIP_MIN_AGE_SEC: int = 90
    SCALP_FLIP_MIN_PROFIT_ATR: float = 0.3

    # Commission gate threshold.  Before placing an entry we require:
    #     TP profit >= MULT × round-trip commission
    # 1.0  = trade is break-even on TP hit (no margin)
    # 1.2  = 20% margin over fees on a perfect trade (default — small but real)
    # 2.0  = strict, refuses most signals (was the original — too restrictive)
    SCALP_COMMISSION_GATE_MULT: float = 1.2

    # ── Daily limits (real risk control) ─────────────────────────────────
    # Removed the hard total-day cap 2026-05-18 — it stopped the bot at
    # 13:26 MSK after 30 trades, missing the rest of the session (which
    # included the +213 ₽ MSK-12h cluster).  Real safety floor is the
    # P&L kill-switch below, not a count.  A soft per-hour rate limit
    # is still applied so a runaway-buggy day can't fire 200 trades.
    SCALP_MAX_TRADES_PER_DAY: int = 0  # 0 = disabled
    SCALP_MAX_TRADES_PER_HOUR: int = 12  # soft cap; avg session
    # is 4-8/h, this leaves headroom
    SCALP_DAILY_LOSS_PCT_LIMIT: float = 0.01  # 1% of portfolio → kill switch
    SCALP_DAILY_WIN_LOCK_PCT: float = 0.015  # +1.5% on the day → also stop
    # (don't give back profits)

    # Trading session blackout (MSK).  Avoid open/close volatility spikes,
    # they cause slippage on stops.
    SCALP_BLACKOUT_HOURS_MSK: list = [9, 18, 23]  # pre-open, evening-cross, eve session noise
    # Spread cap: skip if spread > N × min_price_increment of contract.
    SCALP_MAX_SPREAD_TICKS: int = 3

    # ── Loop intervals (event-driven but with safety polls) ──────────────
    # The bot is mostly event-driven via streams, but a safety tick runs to:
    #   - re-check stops/TP from a fresh last_price snapshot,
    #   - manage positions in case streaming dropped a tick.
    SCALP_SAFETY_TICK_SECONDS: float = 2.0

    # ── Paths ────────────────────────────────────────────────────────────
    SCALP_DB_PATH: Path = Path(__file__).resolve().parent.parent.parent / "data" / "scalp.db"
    SCALP_LOG_PATH: Path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "logs" / "scalp.log"
    )

    model_config = {
        "env_file": _ENV_FILE,
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
