"""TrendSettings — independent config for trend bot."""

from pathlib import Path
from pydantic_settings import BaseSettings

from futbot.config import _ENV_FILE


class TrendSettings(BaseSettings):
    # Credentials (reused from parent .env)
    T_INVEST_TOKEN: str
    T_INVEST_ACCOUNT_ID: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: int = 0

    # ── Safety ─────────────────────────────────────────────────────────
    # LIVE since 2026-05-29 (user-authorised).  Trend is the most robust edge
    # (9/10 years across regimes).  Starting at verification size — see the
    # MAX_OPEN note below; scale up after live fills are confirmed clean.
    TREND_PAPER_MODE: bool = False

    # Freeze new entries (existing positions still managed: rollover, exits).
    # Set to True while developing/testing replacement logic.  Default is
    # False as of 2026-05-24 — pattern detector is live (see
    # futbot.patterns).  Override via .env: TREND_FREEZE_NEW_ENTRIES=true
    TREND_FREEZE_NEW_ENTRIES: bool = False

    # ── Universe selection ─────────────────────────────────────────────
    # "core"     — only Sharpe ≥ 0.6 AND ≥ 7 OOS trades (~12 contracts)
    # "extended" — all 26 WF survivors
    TREND_UNIVERSE_MODE: str = "core"

    # ── Neo-asset trading ──────────────────────────────────────────────
    # Neo assets (US stocks / crypto) — high leverage, triple_top validated
    # (+196% on-margin, OOS-positive).  USD-priced, RUB P&L at close FX rate,
    # daily holding fee (~4.5%+CB rate), no expiry.  Restricted to triple_top.
    TREND_TRADE_NEO: bool = True
    TREND_NEO_DAILY_FEE_ANNUAL: float = 0.205  # 4.5% + ~16% CB rate
    TREND_NEO_MAX_OPEN: int = 4  # cap concurrent Neo positions
    # Free-margin guard: require free margin ≥ buffer × the new position's ГО
    # (initial margin) before opening a Neo trade.  2.0 leaves a 2× cushion so
    # accumulated Neo positions can't drift into a margin call.
    TREND_NEO_MARGIN_BUFFER: float = 2.0

    # ── Strategy parameters (override per-contract via portfolio.py) ───
    # NO stop-loss by design — OsEngine convention.  Exit only via band-flip
    # or expiry rollover.
    TREND_MIN_DAYS_TO_EXPIRY: int = 14
    TREND_ROLLOVER_DAYS: int = 3  # close position N days before expiry

    # ── Sizing ─────────────────────────────────────────────────────────
    TREND_LOTS_PER_TRADE: int = 1  # fallback fixed lots (used only if vol-target disabled)
    # Volatility-targeted sizing: instead of 1 fixed lot per signal, size each
    # position so that its INITIAL MARGIN (ГО) ≈ TARGET_MARGIN_RUB.  Cheap legs
    # (GD, S1, MM at ~5-7k margin) get 1 lot; very cheap ones get 2-3; expensive
    # ones (LK ~6.5k) stay at 1.  Keeps capital usage even across instruments.
    TREND_VOL_TARGET_SIZING: bool = True
    TREND_TARGET_MARGIN_RUB: float = 7000.0  # target ГО per position
    # Hard ceiling so a price/risk-rate misread can't ladder a position 20x.
    # 10 covers cheap legs (S1, HOOD, CVNA) — they get clipped to ~3-5k ГО
    # rather than 7k, but that's better than over-leveraging on bad metadata.
    TREND_LOTS_MAX_PER_TRADE: int = 10
    # Concurrency cap raised 6→10 after gradual scaling worked.  Peak margin
    # stays < ~70% of the account at 10 positions × 7k = 70k (gated below by
    # the live free-margin check before each open).
    TREND_MAX_OPEN_POSITIONS: int = 10  # cap concurrent risk

    # ── Risk caps ──────────────────────────────────────────────────────
    TREND_DAILY_LOSS_PCT_LIMIT: float = 0.02  # 2% drawdown → stop new entries
    TREND_PER_CONTRACT_DAILY_LOSS_PCT: float = 0.005  # 0.5% per contract daily

    # ── Loop cadence ───────────────────────────────────────────────────
    # Hourly evaluation matches Bollinger lookback resolution.  Faster
    # would just re-process the same closed bar.
    TREND_LOOP_SECONDS: int = 3600

    # ── Candle history depth ───────────────────────────────────────────
    # Need at least MAX(N) + buffer bars.  Largest N in portfolio is 80,
    # so 200h ≈ 8 days lookback is plenty.  Fetching 30 days gives margin
    # for non-trading hours / weekends.
    TREND_CANDLE_HISTORY_DAYS: int = 30

    # ── Paths ──────────────────────────────────────────────────────────
    TREND_DB_PATH: Path = Path(__file__).resolve().parent.parent.parent / "data" / "trend.db"
    TREND_LOG_PATH: Path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "logs" / "trend.log"
    )

    model_config = {
        "env_file": _ENV_FILE,
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
