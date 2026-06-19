"""PairsSettings — independent config namespace for pairs bot."""

from pathlib import Path
from pydantic_settings import BaseSettings

from futbot.config import _ENV_FILE


class PairsSettings(BaseSettings):
    # Credentials (reused from parent .env)
    T_INVEST_TOKEN: str
    T_INVEST_ACCOUNT_ID: str
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: int = 0

    # ── Safety: paper by default ───────────────────────────────────────
    PAIRS_PAPER_MODE: bool = True

    # ── Universe — pairs vetted in grid search + expansion 2026-05-22 ──
    # Each pair is "y-x" where spread = price(y) - β·price(x).
    # Order matters for sign convention (β re-fit weekly).
    #
    # Original 4 (60-180d backtest, in-sample):
    #   LK-Si, SR-Si, GZ-Si, SR-MX
    # Already validated live 2026-05-22: LK-Si delivered +2181 ₽ in 24h.
    #
    # +3 new from 25-base expansion scan (in-sample 180d):
    #   LK-RN — both Russian oil, Sharpe 3.89 in IS
    #   MX-GK — MOEX index vs Norilsk Nickel, Sharpe 2.82 IS
    #   NG-MM — NatGas vs MOEX index, Sharpe 1.80 IS
    # ⚠️ The +3 new are NOT walk-forward validated; live-only validation now.
    # If after 4 weeks any of them shows < 50% win rate, drop it.
    # MX (price ~265k/lot, margin ~53k) doesn't fit a 50k portfolio at any
    # sane per-pair budget — removed.  Will be re-added when portfolio
    # > 250k (per_leg budget needs to cover at least 1 MX lot margin).
    PAIRS_LIST: list = [
        "LK-Si",  # LK margin ~6.7k, Si margin ~6.7k.  WF-OOS: Sharpe +0.06 win 57% ✅
        "SR-Si",  # SR margin ~7.6k, Si margin ~6.7k.  WF-OOS: Sharpe -0.76 win 29% ⚠ ON WATCH
        "GZ-Si",  # GZ margin ~2k,  Si margin ~6.7k.   WF-OOS: Sharpe +0.19 win 70% ✅
        "LK-RN",  # both Russian oil.                  WF-OOS: Sharpe +0.59 win 60% ✅ (best)
        # ── GK-MM, YD-GK were added then REVERTED 2026-05-29.  The advanced
        #    scanner gave them great IN-SAMPLE Sharpe (2.76, 2.89) but they
        #    FAILED walk-forward out-of-sample (Sharpe -0.47, -0.48) — classic
        #    overfit.  The WF gate (futbot/scripts/pairs_walkforward.py) caught
        #    them before any capital.  Do NOT re-add without OOS validation.
        # NG-MM REMOVED 2026-05-29: NG is priced in USD/mmBtu (~3.17) by
        # T-Invest API; with rub_per_point=1 fallback the notional looks
        # like 3 ₽/lot instead of ~300 000 ₽/lot.  Now caught by the
        # MIN_SANE_NOTIONAL guard in compute_lots, but the pair is
        # economically unmatchable at 50k portfolio anyway.
        # Re-add once the broker exposes correct step_value for NG.
    ]

    # ── Strategy parameters ────────────────────────────────────────────
    # z_entry lowered 2.0 → 1.7 to increase activity (~40% more triggers).
    # Backtest LK-Si at z=1.5 had 17 trades vs 12 at z=2.0; 1.7 is the
    # middle ground keeping decent win-rate.
    PAIRS_Z_ENTRY: float = 1.7
    PAIRS_Z_STOP: float = 4.0
    PAIRS_MAX_HOLD_HOURS: int = 48
    PAIRS_ROLLING_Z_WINDOW_HOURS: int = 240
    PAIRS_REFIT_BETA_HOURS: int = 168
    PAIRS_MAX_ADF_PVALUE: float = 0.15
    PAIRS_FIT_LOOKBACK_DAYS: int = 180

    # ── Sizing ─────────────────────────────────────────────────────────
    # Budget is per-leg MARGIN (since 2026-05-27 fix).  On a 50k portfolio
    # at 30% × 50k = 15k pair_capital → 7.5k per-leg margin.  This fits
    # 1 lot of Si/LK (~6-7k margin).  At 600k portfolio, drop to 8% to
    # avoid over-concentration — same per_leg budget in absolute terms.
    #
    # max_open × capital_per_pair ≤ 1.0 to avoid over-allocation in
    # extreme case where all pairs trigger at once.
    PAIRS_CAPITAL_PER_PAIR_PCT: float = 0.30
    PAIRS_MAX_OPEN_PAIRS: int = 3  # 3 × 30% = 90% capital cap

    # ── Daily risk caps ─────────────────────────────────────────────────
    PAIRS_DAILY_LOSS_PCT_LIMIT: float = 0.02  # 2% of portfolio daily kill
    PAIRS_BLACKOUT_HOURS_MSK: list = []  # pair trading isn't time-of-day
    # sensitive, no blackout by default

    # ── Loop cadence ────────────────────────────────────────────────────
    # Hourly evaluation matches the resolution of the strategy fit.  A
    # smaller interval would just re-evaluate stale data.
    PAIRS_LOOP_SECONDS: int = 3600  # 1 hour

    # ── Paths ──────────────────────────────────────────────────────────
    PAIRS_DB_PATH: Path = Path(__file__).resolve().parent.parent.parent / "data" / "pairs.db"
    PAIRS_LOG_PATH: Path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "logs" / "pairs.log"
    )

    model_config = {
        "env_file": _ENV_FILE,
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
