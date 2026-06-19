"""Curated watchlists and sector mappings for MOEX."""

# Blue-chip MOEX tickers to always include in scanning
BLUECHIP_TICKERS = [
    "SBER",  # Сбербанк
    "GAZP",  # Газпром
    "LKOH",  # Лукойл
    "GMKN",  # ГМК Норникель
    "YNDX",  # Яндекс
    "ROSN",  # Роснефть
    "NVTK",  # Новатэк
    "TATN",  # Татнефть
    "MGNT",  # Магнит
    "MTSS",  # МТС
    "TCSG",  # ТКС Холдинг (Тинькофф)
    "SNGS",  # Сургутнефтегаз
    "ALRS",  # АЛРОСА
    "POLY",  # Полюс
    "CHMF",  # Северсталь
    "NLMK",  # НЛМК
    "MAGN",  # ММК
    "PHOR",  # ФосАгро
    "FEES",  # ФСК ЕЭС
    "HYDR",  # РусГидро
]

SECTOR_MAP = {
    "SBER": "financials",
    "TCSG": "financials",
    "GAZP": "energy",
    "LKOH": "energy",
    "ROSN": "energy",
    "NVTK": "energy",
    "SNGS": "energy",
    "GMKN": "materials",
    "ALRS": "materials",
    "POLY": "materials",
    "CHMF": "materials",
    "NLMK": "materials",
    "MAGN": "materials",
    "YNDX": "technology",
    "MGNT": "consumer",
    "MTSS": "telecom",
    "TATN": "energy",
    "PHOR": "materials",
    "FEES": "utilities",
    "HYDR": "utilities",
}

# Liquid MOEX futures (FORTS).  Ticker prefixes (без месячной литеры) used
# as substring matches against the future's `ticker` field returned by the
# T-Invest API — the actual contract ticker is e.g. "SiH5", "BRM5", "RIU5".
# Rebuild/extend this list as new futures gain liquidity.
LIQUID_FUTURES_PREFIXES = [
    "Si",  # USD/RUB
    "BR",  # Brent
    "GD",  # Gold
    "RI",  # RTS Index
    "MX",  # MOEX Index (MIX/MXI)
    "NG",  # Natural Gas
    "SR",  # Sber (SBRF)
    "GZ",  # Gazprom (GAZR)
    "LK",  # Lukoil (LKOH)
    "VB",  # VTB (VTBR)
    "GM",  # Norilsk Nickel (GMKR)
    "RN",  # Rosneft (ROSN)
    "TT",  # Tatneft (TATN)
    "MN",  # Magnit (MGNT)
    "MT",  # MTS (MTSI)
    "AL",  # Alrosa (ALRS)
    "CH",  # Severstal (CHMF)
    "NM",  # NLMK
    "EU",  # EUR/RUB
    "CN",  # CNY/RUB
]

# Asset-class codes for LightGBM categorical feature (see analysis/features.py).
# The code is stable — do not renumber after a model has been trained against
# a given mapping, or old categorical splits will silently target the wrong
# asset class.  Unknown futures fall back to OTHER (15).
#
# Design: one code per FORTS asset family.  Single-stock futures share the
# same "underlying" family (Sber → SR, Gazp → GZ, etc.) so the model can
# transfer knowledge across contracts of the same underlying but separate
# them from commodities / FX where dynamics differ.
SHARE_ASSET_CLASS_CODE = 0
OTHER_ASSET_CLASS_CODE = 15
FUTURES_ASSET_CLASS_CODES: dict[str, int] = {
    "Si": 1,
    "EU": 2,
    "CN": 3,  # FX
    "BR": 4,
    "NG": 5,
    "GD": 6,  # commodities
    "RI": 7,
    "MX": 8,  # indices
    "SR": 9,
    "GZ": 10,
    "LK": 11,  # single-stock: SBER/GAZP/LKOH
    "VB": 12,
    "GM": 13,
    "RN": 14,  # single-stock: VTBR/GMKN/ROSN
    "TT": 16,
    "MN": 17,
    "MT": 18,  # single-stock: TATN/MGNT/MTSS
    "AL": 19,
    "CH": 20,
    "NM": 21,  # single-stock: ALRS/CHMF/NLMK
}


def asset_class_code(ticker: str | None, kind: str) -> int:
    """Return the asset-class category code for a given (ticker, kind).

    Shares → ``SHARE_ASSET_CLASS_CODE`` (0).  Futures are mapped by 2-char
    ticker prefix (Tinkoff conventions: SiH6, BRM5, RIU5, …).  Unknown
    tickers → ``OTHER_ASSET_CLASS_CODE`` (15).

    Used by ``analysis.features.build_features`` so a single pooled
    LightGBM model can split on asset class and learn per-family rules
    without needing N separate models.
    """
    if kind != "future":
        return SHARE_ASSET_CLASS_CODE
    if not ticker or len(ticker) < 2:
        return OTHER_ASSET_CLASS_CODE
    return FUTURES_ASSET_CLASS_CODES.get(ticker[:2], OTHER_ASSET_CLASS_CODE)


# Blacklist: instruments to never trade
BLACKLIST_TICKERS: set[str] = set()

# Trading profiles — applied via /profile command in Telegram
TRADING_PROFILES: dict[str, dict] = {
    "conservative": {
        "label": "Conservative",
        "description": "Small positions, tight stops, high signal confidence required",
        "MAX_POSITIONS": 3,
        "MAX_POSITION_PCT": 0.10,
        "MAX_PORTFOLIO_RISK_PCT": 0.01,
        "MAX_DAILY_LOSS_PCT": 0.015,
        "MAX_DRAWDOWN_PCT": 0.06,
        "SIGNAL_THRESHOLD": 0.70,
        "KELLY_FRACTION": 0.15,
        "SCAN_INTERVAL_MINUTES": 60,
        "SPREAD_THRESHOLD": 1.5,
    },
    "moderate": {
        "label": "Moderate",
        "description": "Balanced risk/reward (default)",
        "MAX_POSITIONS": 5,
        "MAX_POSITION_PCT": 0.20,
        "MAX_PORTFOLIO_RISK_PCT": 0.02,
        "MAX_DAILY_LOSS_PCT": 0.03,
        "MAX_DRAWDOWN_PCT": 0.10,
        "SIGNAL_THRESHOLD": 0.60,
        "KELLY_FRACTION": 0.25,
        "SCAN_INTERVAL_MINUTES": 30,
        "SPREAD_THRESHOLD": 2.0,
    },
    "aggressive": {
        "label": "Aggressive",
        "description": "Larger positions, lower thresholds, frequent scans",
        "MAX_POSITIONS": 8,
        "MAX_POSITION_PCT": 0.30,
        "MAX_PORTFOLIO_RISK_PCT": 0.035,
        "MAX_DAILY_LOSS_PCT": 0.05,
        "MAX_DRAWDOWN_PCT": 0.15,
        "SIGNAL_THRESHOLD": 0.50,
        "KELLY_FRACTION": 0.40,
        "SCAN_INTERVAL_MINUTES": 15,
        "SPREAD_THRESHOLD": 2.5,
    },
}
