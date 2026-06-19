"""Technical indicator computation using pandas-ta."""

import pandas as pd
import numpy as np

try:
    import pandas_ta as ta

    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to OHLCV DataFrame.

    Expects columns: open, high, low, close, volume.
    Returns DataFrame with indicator columns added.
    """
    df = df.copy()

    if HAS_PANDAS_TA:
        return _compute_with_pandas_ta(df)
    return _compute_manual(df)


def _compute_with_pandas_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators via pandas-ta library."""
    # RSI
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["rsi_7"] = ta.rsi(df["close"], length=7)

    # MACD
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    if macd is not None:
        df["macd"] = macd.iloc[:, 0]
        df["macd_histogram"] = macd.iloc[:, 1]
        df["macd_signal"] = macd.iloc[:, 2]

    # Bollinger Bands
    bb = ta.bbands(df["close"], length=20, std=2.0)
    if bb is not None:
        df["bb_upper"] = bb.iloc[:, 0]
        df["bb_mid"] = bb.iloc[:, 1]
        df["bb_lower"] = bb.iloc[:, 2]
        df["bb_width"] = (
            bb.iloc[:, 3] if bb.shape[1] > 3 else (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        )
        df["bb_percent_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ATR
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # OBV
    df["obv"] = ta.obv(df["close"], df["volume"])

    # ADX
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    if adx is not None:
        df["adx_14"] = adx.iloc[:, 0]

    # Stochastic
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    if stoch is not None:
        df["stoch_k"] = stoch.iloc[:, 0]
        df["stoch_d"] = stoch.iloc[:, 1]

    # EMAs and SMAs
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["ema_50"] = ta.ema(df["close"], length=50)
    df["sma_200"] = ta.sma(df["close"], length=200)

    return df


def _compute_manual(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback: compute indicators manually without pandas-ta."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # RSI
    for period in [7, 14]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_mid"] = sma20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_percent_b"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # ATR
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["obv"] = obv

    # ADX (simplified)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr14 = df["atr_14"]
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    df["adx_14"] = dx.rolling(14).mean()

    # Stochastic
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    df["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # EMAs / SMAs
    df["ema_9"] = close.ewm(span=9, adjust=False).mean()
    df["ema_21"] = close.ewm(span=21, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["sma_200"] = close.rolling(200).mean()

    return df
