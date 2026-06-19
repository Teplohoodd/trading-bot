"""futbot configuration.

Reuses the parent project's .env file (same T-Invest + Telegram credentials)
and adds FUTBOT_-prefixed overrides for the futures-specific knobs.

CRITICAL DEFAULT: paper-mode is ON.  Going live requires explicitly setting
`FUTBOT_PAPER_MODE=false` in .env.  This is intentional — `futbot` should
never trade real money on first boot.

The .env file is searched in this order (first hit wins):
  1. Path in FUTBOT_ENV_FILE environment variable (explicit override)
  2. Current working directory
  3. The repo root — one level above the futbot package
This makes both `python -m futbot.main` (from repo root) and
`python futbot/main.py` (from inside the package) work the same way.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


def _find_env_file() -> str:
    """Resolve .env location.  Returns a path string suitable for
    pydantic-settings; '.env' literal as last-resort fallback so the
    settings class still loads even if no .env exists (the required
    fields will fail validation with a clear error)."""
    explicit = os.environ.get("FUTBOT_ENV_FILE")
    if explicit and Path(explicit).is_file():
        return explicit
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",  # repo root
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return ".env"


_ENV_FILE = _find_env_file()


class FutSettings(BaseSettings):
    # ── Credentials (reused from trade_claude) ─────────────────────────────
    T_INVEST_TOKEN: str
    T_INVEST_ACCOUNT_ID: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: int = 0

    # ── Execution mode ─────────────────────────────────────────────────────
    # Paper mode = no real orders.  All "would-be" trades are logged to
    # futbot.db with a paper_* tag and Telegram alerts are prefixed [PAPER].
    # Default TRUE so the bot is safe-by-default; flip via .env.
    FUTBOT_PAPER_MODE: bool = True

    # Per-cycle cadence.  Loop wakes every FUTBOT_LOOP_SECONDS, fetches data,
    # runs pipeline, and (when not in paper mode) places orders.
    FUTBOT_LOOP_SECONDS: int = 60

    # ── Universe ───────────────────────────────────────────────────────────
    # Tier-1: always-traded liquid FORTS contracts.  Specified as base
    # tickers (e.g. "Si") and resolved to the front month at startup +
    # every roll.  Resolver in universe.py picks the contract whose
    # expiration is ≥ FUTURES_MIN_DAYS_TO_EXPIRY days out.
    # Use actual FORTS ticker prefixes (verified via get_all_futures):
    #   Si=USDRUB, BR=Brent, GZ=Gazprom, SR=Sberbank, LK=Lukoil, MX=MOEX-index
    FUTBOT_TIER1_BASES: list = ["Si", "BR", "GZ", "SR", "LK", "MX"]
    # Tier-2: traded only when ATR > 0.5 × 30-day median ATR for that contract.
    #   NG=NatGas, GD=Gold, RT=RTS-index, EURRUBF (currency, single contract)
    FUTBOT_TIER2_BASES: list = ["NG", "GD", "RT", "EURRUBF"]
    # Skip contracts within N days of expiry.  Roll handling itself is
    # deferred to phase 1+; for now we just stop trading near-expiry contracts.
    FUTBOT_MIN_DAYS_TO_EXPIRY: int = 14
    # Hard cap on simultaneous open contracts.
    FUTBOT_MAX_OPEN_CONTRACTS: int = 2  # phase-3 default; can lift later

    # ── Pipeline layer thresholds ──────────────────────────────────────────
    # Layer 1 — trend (1h):
    FUTBOT_TREND_EMA_FAST: int = 20
    FUTBOT_TREND_EMA_SLOW: int = 50
    FUTBOT_TREND_ADX_MIN: float = 18.0  # below = "flat", no trade
    # Layer 2 — regime (1h):
    FUTBOT_REGIME_ATR_LOOKBACK: int = 30 * 24  # 30 days of hourly bars
    FUTBOT_REGIME_TRENDING_R2_MIN: float = 0.55  # rolling-50 linreg R²
    FUTBOT_REGIME_VOL_SPIKE_MULT: float = 2.0  # ATR > 2 × median → SKIP
    # Layer 3 — setup (15m):
    FUTBOT_SETUP_KDJ_BUY_BELOW: float = 25.0
    FUTBOT_SETUP_KDJ_SELL_ABOVE: float = 75.0
    FUTBOT_SETUP_BB_BUY_BELOW: float = 0.2
    FUTBOT_SETUP_BB_SELL_ABOVE: float = 0.8
    # Layer 4 — trigger (5m):
    FUTBOT_TRIGGER_VOL_MULT: float = 1.5  # bar volume > 1.5× 20-bar median
    FUTBOT_TRIGGER_RANGE_MULT: float = 1.2  # bar range > 1.2× 20-bar median
    FUTBOT_TRIGGER_CLOSE_QUARTILE: float = 0.75  # close in top/bottom 25% of bar

    # ── Sizing ─────────────────────────────────────────────────────────────
    # Max fraction of portfolio used as ГО across all open positions.
    FUTBOT_MAX_GO_PCT: float = 0.40
    # Vol-target: each open position contributes this fraction of portfolio
    # as daily P&L 1-σ vol.
    FUTBOT_VOL_TARGET_DAILY_PCT: float = 0.005  # 0.5 %
    # Cold-start / fallback lots when ГО or vol can't be computed.
    FUTBOT_FALLBACK_LOTS: int = 1

    # ── Stops ──────────────────────────────────────────────────────────────
    # Chandelier trailing: peak − k × ATR_1h.
    FUTBOT_STOP_ATR_MULT_INITIAL: float = 1.5  # catastrophe stop at entry
    FUTBOT_STOP_ATR_MULT_TRAIL: float = 3.0  # Chandelier trail distance
    FUTBOT_TRAIL_ACTIVATE_R: float = 1.0  # activate trail after +1 R
    FUTBOT_MAX_HOLD_HOURS: int = 24  # hard time-cap

    # ── Risk audit ─────────────────────────────────────────────────────────
    # Hours MSK (UTC+3) to BLOCK new entries.  Pre-open + evening-quiet.
    FUTBOT_BLACKOUT_HOURS_MSK: list = [9, 19, 23]
    # Per-contract daily loss cap (RUB).  0 = disabled.
    FUTBOT_MAX_CONTRACT_DAILY_LOSS_PCT: float = 0.01
    # Total daily loss kill-switch (fraction of portfolio).
    FUTBOT_MAX_DAILY_LOSS_PCT: float = 0.015
    # Spread guard: skip when current spread > N × 30-day median spread.
    FUTBOT_SPREAD_THRESHOLD_MULT: float = 1.5

    # ── Paths ──────────────────────────────────────────────────────────────
    # Defaults are resolved relative to the repo root (parent of the
    # futbot package), so the bot writes to the same `data/` dir whether
    # you launch as `python -m futbot.main` from the repo root or as
    # `python main.py` from inside futbot/.
    FUTBOT_DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "futbot.db"
    FUTBOT_LOG_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "logs" / "futbot.log"

    model_config = {
        "env_file": _ENV_FILE,
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # ignore trade_claude-only keys in .env
    }
