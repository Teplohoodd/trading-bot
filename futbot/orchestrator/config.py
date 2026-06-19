"""OrchSettings — toggles which strategies the orchestrator runs.

Each strategy has its own *Settings (PairsSettings, TrendSettings) loaded
internally — those still come from the same .env.  This config just
controls WHICH ones the orchestrator activates.
"""

from pathlib import Path
from pydantic_settings import BaseSettings

from futbot.config import _ENV_FILE


class OrchSettings(BaseSettings):
    # Credentials (reused from parent .env)
    T_INVEST_TOKEN: str
    T_INVEST_ACCOUNT_ID: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: int = 0

    # ── Which strategies to enable ─────────────────────────────────────
    # PAIRS DISABLED 2026-05-29: the fair multi-year HOURLY test (stitched
    # MOEX-ISS futures, 2022-2025, real live params) showed ALL pairs lose
    # (LK-Si -44%, GZ-Si -74%, GK-MM -64%, 0-2/4 positive years).  The earlier
    # "good" pairs P&L was the over-leverage bug.  Retired from trading; the
    # code stays for research only.
    ORCH_ENABLE_PAIRS: bool = False
    ORCH_ENABLE_TREND: bool = True
    ORCH_ENABLE_CARRY: bool = True  # Si calendar-spread carry (WF-validated)
    ORCH_ENABLE_SCALP: bool = False  # off by default; user can flip in .env
    # Volume-breakdown shorts on stocks via futures (the IVAT signature).
    # Validated on 90d/27 stocks (2h bars: +138%, 4/4 months) but only ONE
    # falling regime — starts in PAPER (BD_PAPER_MODE) to earn forward stats.
    ORCH_ENABLE_BREAKDOWN: bool = True

    # ── Loop intervals (override per-strategy defaults if needed) ──────
    ORCH_PAIRS_INTERVAL_SECONDS: int = 3600  # hourly (signal scan)
    # Trend: signal bars stay HOURLY (validated geometry) but the scanner
    # ticks every 15 min, so a pattern confirmed on an hourly close is
    # entered minutes later, not up to an hour later (2026-06-12: the LT
    # entry lag).  Detection is idempotent — same closed bars between hourly
    # closes → no signal duplication (plus the open-trade check per base).
    ORCH_TREND_INTERVAL_SECONDS: int = 900  # 15 min scan, hourly bars
    ORCH_CARRY_INTERVAL_SECONDS: int = 3600  # hourly (signal scan)
    # Fast position monitor (between signal ticks): reconcile vs broker +
    # enforce stop/target/timeout.  Catches orphans & exits far faster than
    # the hourly tick.
    ORCH_MONITOR_INTERVAL_SECONDS: int = 300  # 5 min

    # ── Paths ──────────────────────────────────────────────────────────
    ORCH_LOG_PATH: Path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "logs" / "orchestrator.log"
    )

    model_config = {
        "env_file": _ENV_FILE,
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
