"""Market regime detector based on ADX and volatility."""

from enum import Enum
import numpy as np
import pandas as pd
from analysis.indicators import compute_indicators


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


class RegimeDetector:
    """Detects market regime to adjust strategy weights."""

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime from OHLCV data."""
        df = compute_indicators(df)
        last = df.iloc[-1]

        adx = last.get("adx_14", 20)
        atr_pct = last.get("atr_14", 0) / last["close"] * 100 if last["close"] > 0 else 0

        # ATR percentile for volatility regime
        atr_series = df["atr_14"].dropna() / df["close"] * 100
        atr_percentile = (atr_series < atr_pct).mean() * 100 if len(atr_series) > 0 else 50

        # High volatility check
        if atr_percentile > 80:
            return MarketRegime.HIGH_VOLATILITY

        # Trend check
        if not np.isnan(adx) and adx > 25:
            # Direction from EMA
            ema9 = last.get("ema_9", 0)
            ema21 = last.get("ema_21", 0)
            if ema9 > ema21:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN

        return MarketRegime.RANGING

    def get_strategy_weights(self, regime: MarketRegime) -> dict[str, float]:
        """Return strategy weights for signal aggregation based on regime."""
        return {
            MarketRegime.TRENDING_UP: {"ml_lightgbm": 0.7, "technical": 0.3},
            MarketRegime.TRENDING_DOWN: {"ml_lightgbm": 0.7, "technical": 0.3},
            MarketRegime.RANGING: {"ml_lightgbm": 0.4, "technical": 0.6},
            MarketRegime.HIGH_VOLATILITY: {"ml_lightgbm": 0.3, "technical": 0.3},
        }[regime]

    def get_position_scale(self, regime: MarketRegime) -> float:
        """Position size multiplier based on regime. < 1.0 means reduce size."""
        return {
            MarketRegime.TRENDING_UP: 1.0,
            MarketRegime.TRENDING_DOWN: 0.8,
            MarketRegime.RANGING: 0.7,
            MarketRegime.HIGH_VOLATILITY: 0.5,
        }[regime]
