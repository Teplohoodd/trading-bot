"""Breakdown strategy settings."""

from pydantic_settings import BaseSettings


class BreakdownSettings(BaseSettings):
    # LIVE (user override 2026-06-10): user accepts the one-regime caveat.
    # Was PAPER — flip back if you want a forward track record first.
    BD_PAPER_MODE: bool = False

    # ── Signal (validated params; see breakdown_study.py) ──────────────
    BD_BAR_HOURS: int = 2  # signal timeframe (2h was the sweet spot)
    BD_LOOKBACK_BARS: int = 12  # N: prior-range window (12×2h = 24h)
    BD_VOL_MULT: float = 2.0  # volume ≥ k × prior N-bar median
    BD_SEVERITY: float = -0.01  # breakdown bar return ≤ -1%
    BD_SMA_BARS: int = 120  # regime filter: close < SMA(120×2h=10d)
    BD_USE_SMA_FILTER: bool = True

    # ── Exits ───────────────────────────────────────────────────────────
    BD_RISK_REWARD: float = 3.0  # target = entry - RR × (stop - entry)
    BD_TIMEOUT_BARS: int = 24  # 24×2h = 48h max hold
    BD_MAX_STOP_PCT: float = 0.10  # skip if stop distance > 10% (bad bar)

    # ── Sizing / risk ───────────────────────────────────────────────────
    BD_LOTS_PER_TRADE: int = 1
    BD_MAX_OPEN_POSITIONS: int = 3
    BD_MARGIN_BUFFER: float = 2.0  # free margin ≥ buffer × ГО before entry

    # ── DB ──────────────────────────────────────────────────────────────
    BD_DB_PATH: str = "data/breakdown.db"

    class Config:
        env_file = ".env"
        extra = "ignore"


# Stock (signal) → futures base (execution).  Futures resolve to front month
# at runtime.  Only stocks WITH liquid FORTS futures are tradeable; the rest
# are signal-only (alert, no trade).
STOCK_TO_FUT = {
    "IVAT": "IVAT",  # IVM6 — the motivating case (stock itself unshortable)
    "SBER": "SBRF",
    "GAZP": "GAZR",
    "LKOH": "LKOH",
    "ROSN": "ROSN",
    # GMKN excluded: carry owns the GK calendar spread (GKU6/GKM6) — a
    # breakdown short on GKU6 would break its delta-neutrality.  Signal на
    # GMKN остаётся alert-only (см. STOCK_FIGI).
    "TATN": "TATN",
    "MGNT": "MGNT",
    "YDEX": "YDEX",
    "VTBR": "VTBR",
    "ALRS": "ALRS",
    "CHMF": "CHMF",
    "MTLR": "MTLR",
    "AFLT": "AFLT",
    "MOEX": "MOEX",
    "OZON": "OZON",
    "POSI": "POSI",
    "SOFL": "SOFL",
    "WUSH": "WUSH",
    "SMLT": "SMLT",
    "SGZH": "SGZH",
    "ASTR": "ASTR",
    "HEAD": "HEAD",
    "MTSS": "MTSI",
    "AFKS": "AFKS",
    "PIKK": "PIKK",
}

# Stock FIGIs (signal data source) — same universe as the validation study.
STOCK_FIGI = {
    "SBER": "BBG004730N88",
    "GAZP": "BBG004730RP0",
    "LKOH": "BBG004731032",
    "ROSN": "BBG004731354",
    "GMKN": "BBG004731489",
    "TATN": "BBG004RVFFC0",
    "MGNT": "BBG004RVFCY3",
    "YDEX": "TCS00A107T19",
    "VTBR": "BBG004730ZJ9",
    "ALRS": "BBG004S68B31",
    "CHMF": "BBG00475K6C3",
    "MTLR": "BBG004S68598",
    "AFLT": "BBG004S683W7",
    "MOEX": "BBG004730JJ5",
    "OZON": "TCS00A10CW95",
    "POSI": "TCS00A103X66",
    "SOFL": "TCS00A0ZZBC2",
    "WUSH": "TCS00A105EX7",
    "SMLT": "BBG00F6NKQX3",
    "SGZH": "BBG0100R9963",
    "IVAT": "TCS00A108GD8",
    "ASTR": "TCS00A106T36",
    "HEAD": "TCS20A107662",
    "MTSS": "BBG004S681W1",
    "AFKS": "BBG004S68614",
    "PIKK": "BBG004S68BH6",
}
