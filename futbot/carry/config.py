"""CarrySettings — Si calendar-spread carry bot config.

Strategy: delta-neutral basis mean-reversion on the front/next Si
(USD/RUB) calendar spread.  Walk-forward validated 2026-05-29 (4/4 time
blocks positive, Sharpe +0.78/+0.36/+0.23/+2.59, win 67-86%).
"""

from pathlib import Path
from pydantic_settings import BaseSettings

from futbot.config import _ENV_FILE


class CarrySettings(BaseSettings):
    # Credentials (reused from parent .env)
    T_INVEST_TOKEN: str
    T_INVEST_ACCOUNT_ID: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: int = 0

    # ── Safety ─────────────────────────────────────────────────────────
    # LIVE again 2026-05-31 (user-authorised) after fixing the two problems
    # that the first live trade exposed:
    #   1. FIXED — P&L now reads ACTUAL fills (post_market_order_with_fill /
    #      post_limit_with_fallback), not get_last_price snapshots.  The trade
    #      mislogged as +39.4₽ was really -73.6₽.
    #   2. FIXED — entry/exit now use PASSIVE LIMIT orders (capture the spread)
    #      via CARRY_USE_LIMIT_ORDERS, with market fallback on the remainder.
    # RESIDUAL RISK: the limit path is unverified live; and if the order book
    # is empty (illiquid hours) it market-falls-back → slippage returns.  Risk
    # is bounded (delta-neutral, 1 lot GK ~189₽ margin, -1% daily kill).
    # Watch the first trades: log shows "OPEN(lim) ... (limit/market)"; verify
    # real P&L via /pnl and the operations API.
    CARRY_PAPER_MODE: bool = False

    # ── Instrument ─────────────────────────────────────────────────────
    # Base ticker whose front/next expiries form the calendar spread.
    # Switched Si→GK 2026-05-29 after a 4-year roll-safe verification
    # (futbot/scripts/carry_verify.py): Si calendar carry is FRAGILE over
    # 2022-2025 (1/4 positive years, -33%) — the earlier "4/4" was 4 blocks
    # of a single recent 6-month window, not 4 years.  GK (Norilsk) is the
    # most robust: 4/4 positive years, 76% win, +102%, ~5 trades/mo.
    # Alternatives also validated 4/4: RN (87% win), LK, GZ.
    CARRY_BASE: str = "GK"

    # ── Execution ──────────────────────────────────────────────────────
    # Carry's edge is thinner than the bid/ask spread, so market orders lose.
    # Post PASSIVE limit orders (buy at bid / sell at ask) to CAPTURE the
    # spread, with a market fallback on the unfilled remainder so no leg hangs.
    CARRY_USE_LIMIT_ORDERS: bool = True
    CARRY_LIMIT_TIMEOUT_SEC: float = 30.0

    # ── Strategy parameters (match the validated backtest) ─────────────
    CARRY_Z_ENTRY: float = 1.5
    CARRY_Z_STOP: float = 3.5
    CARRY_ROLLING_Z_WINDOW_HOURS: int = 240
    CARRY_MAX_HOLD_HOURS: int = 72
    # Close the spread when the FRONT contract is within this many days of
    # expiry (avoid expiry/delivery mechanics; the next pair of expiries is
    # picked up automatically on the following tick).
    CARRY_ROLL_DAYS_BEFORE_EXPIRY: int = 5

    # ── Sizing ─────────────────────────────────────────────────────────
    # Calendar spread is delta-neutral & low-risk, so a modest capital
    # slice is plenty.  Equal lots both legs (1:1) for delta-neutrality.
    CARRY_CAPITAL_PCT: float = 0.10  # 10% of portfolio margin budget
    CARRY_MAX_LOTS: int = 10  # hard cap per leg

    # ── Daily risk cap ─────────────────────────────────────────────────
    CARRY_DAILY_LOSS_PCT_LIMIT: float = 0.01  # 1% of portfolio daily kill

    # ── Loop cadence ───────────────────────────────────────────────────
    CARRY_LOOP_SECONDS: int = 3600  # hourly (matches z-window resolution)

    # ── Paths ──────────────────────────────────────────────────────────
    CARRY_DB_PATH: Path = Path(__file__).resolve().parent.parent.parent / "data" / "carry.db"

    model_config = {
        "env_file": _ENV_FILE,
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
