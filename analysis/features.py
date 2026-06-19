"""ML feature engineering pipeline.

Design follows Jansen's *Machine Learning for Trading* (ML4T) and López de
Prado's *Advances in Financial Machine Learning*:

* **Labels**: triple-barrier method (ATR-scaled profit-take / stop-loss /
  time-out) with sample weights proportional to |realised return|.  This
  adapts to volatility regime and emphasises decisive samples.
* **Features**: combination of classical technical indicators and ML4T-
  style temporal enrichments — multi-horizon returns, rolling moments,
  and cross-time quantile ranks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from analysis.indicators import compute_indicators
from analysis.macro import MACRO_FEATURE_NAMES

logger = logging.getLogger(__name__)


# Feature set.  First 21 entries are ordered by permutation importance from
# an earlier tuning run (Optuna + permutation importance).  The remaining
# entries are new ML4T-style features added after adopting triple-barrier
# labels; their relative importance will be re-ranked on the next tuning run.
#
# Removed (permutation importance = 0, confirmed noise):
#   macd_signal_dist  — redundant with macd_histogram
#   spread_bps        — point-in-time order book; meaningless for daily/hourly aggregates
#   book_imbalance    — same reason
#   hour_of_day       — constant for daily candles; adds noise
FEATURE_NAMES = [
    # --- Classical block (ranked by previous permutation-importance run) ---
    "close_vs_sma200",  # #1: price vs 200-day MA — long-term trend position
    "atr_14",  # #2: absolute volatility
    "atr_pct",  # #3: relative volatility (ATR / price)
    "adx_14",  # #4: trend strength
    "price_momentum_20",  # #5: 20-bar price change %
    "rsi_14",  # #6: RSI 14
    "bb_width",  # #7: Bollinger Band width (squeeze / expansion)
    "price_momentum_5",  # #8: short-term momentum
    "ema9_ema21_cross",  # #9: fast/slow EMA spread
    "macd_histogram",  # #10: MACD histogram
    "rsi_7",  # #11: fast RSI
    "obv_slope_10",  # #12: OBV trend (volume confirmation)
    "stoch_d",  # #13: Stochastic %D
    "returns_volatility_20",  # #14: realised volatility
    "gap_pct",  # #15: overnight gap
    "volume_ratio_20",  # #16: current vs average volume
    "stoch_k",  # #17: Stochastic %K
    "close_vs_ema50",  # #18: price vs 50-bar EMA
    "high_low_range_pct",  # #19: intrabar range (body size)
    "day_of_week",  # #20: weekday seasonality
    "bb_percent_b",  # #21: Bollinger Band %B
    # --- ML4T block (Jansen ch.4, factor research) ---
    "returns_1",  # single-bar return
    "returns_3",  # 3-bar return
    "returns_10",  # 10-bar return
    "returns_21",  # 21-bar (≈ monthly) return
    "returns_skew_20",  # skewness of recent returns — tail asymmetry
    "returns_kurt_20",  # kurtosis — fat-tail regime detector
    "momentum_rank_60",  # 20-bar momentum's percentile within last 60 bars
    "price_position_60",  # price position in 60-bar Donchian channel
    # --- Fractional differentiation (LdP ch.5) ---
    # close_ffd_04 = FFD-transformed log(close) at d=0.4.  Stationary (passes
    # ADF) while preserving long-memory of price level — bridges the gap
    # between raw price (non-stationary, leaks across train/test) and pure
    # returns (d=1, fully memory-less).  Fed alongside returns_* so the
    # model has both short-term momentum AND a stationary level signal.
    "close_ffd_04",
]

# --- Macro block (broad-market / external drivers) ---
# Appended to FEATURE_NAMES so the schema stays one contiguous list.  Even
# when MacroProvider fails to resolve an instrument these columns are still
# emitted (zero-filled), so train/inference schemas match in all failure
# modes.  See analysis/macro.py for the driver list and resolution logic.
#
# EXCLUSIONS (permutation-importance = 0 across 36k-sample universal run
# 2026-04-21):
#   brent_return_1, brent_return_5 — Brent signal already captured by
#     imoex_return_* and imoex_rel_strength_20; incremental info = 0.
#     Still EMITTED by build_features (zero-filled on failure) for schema
#     stability, but excluded from the training + inference feature set.
_EXCLUDED_MACRO = {"brent_return_1", "brent_return_5"}
FEATURE_NAMES = FEATURE_NAMES + [f for f in MACRO_FEATURE_NAMES if f not in _EXCLUDED_MACRO]

# --- Cross-asset block (stock ↔ market / commodity relationships) ---
# These describe HOW the instrument moves relative to the broad market and
# key commodities, not just the absolute levels.  They're cheap to compute
# (no extra API calls — they reuse the aligned macro series) but have been
# shown in factor research to carry information above pure price features:
#
#   * correlation_20 — is the stock currently regime-aligned with the market?
#     A falling market + strongly-correlated stock is a different short setup
#     than a falling market + uncorrelated stock.
#   * relative_strength_20 — (stock_ret − market_ret), aka "alpha" over the
#     lookback.  Positive = outperformer; Jegadeesh & Titman (1993) show this
#     has >1-month persistence in cross-section.
#   * market_volatility_20 — IMOEX realised vol.  Acts as a regime switch —
#     LightGBM will learn "in high-vol regimes the meaning of RSI changes."
#
# brent_corr_20 had permutation-importance = 0 (same run as above) — excluded.
# Zero-filled when macro data is missing, same contract as the macro block.
CROSS_ASSET_FEATURE_NAMES = [
    "imoex_corr_20",
    "imoex_rel_strength_20",
    "imoex_volatility_20",
    # "brent_corr_20"  — excluded: perm-importance = 0.000 (2026-04-21 run)
]
FEATURE_NAMES = FEATURE_NAMES + CROSS_ASSET_FEATURE_NAMES

# --- Instrument-kind block ---
# Re-enabled after INCLUDE_FUTURES flipped to True — futures now contribute
# meaningfully to the training corpus.  The block gives a pooled model enough
# structural features to learn kind/asset-class-specific rules without needing
# N separate models (Gu/Kelly/Xiu 2020 "Empirical Asset Pricing via ML",
# Bianchi 2021 "Bond Risk Premiums with ML" — both show pooled models +
# categorical asset-class beat N separate models when per-asset history is
# short, which is exactly our futures situation).
#
#   is_future         : 0/1 binary flag — cheapest possible kind indicator.
#   asset_class_code  : integer categorical {0=share, 1=Si, 2=EU, 4=BR, 6=GD,
#                       7=RI, 8=MX, 9=SR, …}.  Fed to LightGBM as a
#                       categorical_feature so the model learns splits like
#                       "if asset_class_code in (4, 5, 6) — commodity —
#                       use different RSI thresholds".
#   days_to_expiry    : for futures, days until expiry; 0 for shares.
#                       Captures roll-cycle regime.
#   in_roll_window    : 1 when within FUTURES_ROLL_WINDOW_DAYS of expiry (for
#                       futures only); pure indicator of roll risk.
#   session_flag      : 0=no time, 1=morning (07-13 MSK), 2=main (14-18),
#                       3=evening (19-23).  Futures trade evening session
#                       on MOEX, shares don't — the flag lets the model
#                       condition on session without needing hour_of_day
#                       (which is noisy on daily candles).
#   basis_slope       : (future − spot) / spot over trailing window.  Fallback
#                       = 0 when spot not available to feature builder.
#   oi_momentum_20    : 20-bar % change in open interest.  Fallback = 0 when
#                       OI series not passed in (current v1 for most callers).
INSTRUMENT_KIND_FEATURE_NAMES = [
    "is_future",
    "asset_class_code",
    "days_to_expiry",
    "in_roll_window",
    "session_flag",
    "basis_slope",
    "oi_momentum_20",
]
FEATURE_NAMES = FEATURE_NAMES + INSTRUMENT_KIND_FEATURE_NAMES

# Categorical feature names — LightGBM handles these via `categorical_feature`
# kwarg in fit().  Kept as a constant so trainer + inference agree on which
# columns are categorical without touching every caller.
CATEGORICAL_FEATURE_NAMES = ["asset_class_code"]


def build_features(
    df: pd.DataFrame,
    spread_bps: float = 0.0,
    book_imbalance: float = 0.0,
    macro_df: pd.DataFrame | None = None,
    instrument_kind: str = "share",
    ticker: str | None = None,
    expiration_date=None,
    roll_window_days: int = 7,
    open_interest: pd.Series | None = None,
    basis: pd.Series | None = None,
) -> pd.DataFrame:
    """Build ML features from OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume, and optionally 'time'.
        spread_bps: Current bid-ask spread in basis points.
        book_imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).
        macro_df: Optional macro context DataFrame (DatetimeIndex UTC + one
            ``<name>_close`` column per macro factor) from MacroProvider.
            When None or empty, all MACRO_FEATURE_NAMES columns are
            zero-filled so the feature schema stays constant regardless.
        instrument_kind: "share" or "future" — drives ``is_future``,
            ``asset_class_code`` and the futures-only feature computations.
        ticker: Instrument ticker — used to map asset_class_code (Si, BR, GD, …).
            Optional; when None, shares get code 0, futures get OTHER (15).
        expiration_date: datetime of the futures contract expiry.  Ignored for
            shares.  When None for a future, days_to_expiry / in_roll_window
            are zero-filled (degrades gracefully — the model still has
            is_future + asset_class_code to key off).
        roll_window_days: Half-width in days around expiry that counts as
            "in roll window" for the binary flag.  Matches
            settings.FUTURES_ROLL_WINDOW_DAYS default (7).
        open_interest: Optional aligned OI series for the futures contract
            (same index/length as df).  When provided, oi_momentum_20 =
            20-bar pct_change; otherwise zero-filled.
        basis: Optional aligned (future − spot)/spot series.  When provided,
            basis_slope carries it forward; otherwise zero-filled.

    Returns:
        DataFrame with feature columns (one row per bar, NaN rows at start).
    """
    df = compute_indicators(df)

    features = pd.DataFrame(index=df.index)

    # 1-2. RSI
    features["rsi_14"] = df["rsi_14"]
    features["rsi_7"] = df["rsi_7"]

    # 3-4. MACD
    features["macd_histogram"] = df["macd_histogram"]
    features["macd_signal_dist"] = df["macd"] - df["macd_signal"]

    # 5-6. Bollinger Bands
    features["bb_percent_b"] = df["bb_percent_b"]
    features["bb_width"] = df["bb_width"]

    # 7-8. ATR
    features["atr_14"] = df["atr_14"]
    features["atr_pct"] = df["atr_14"] / df["close"] * 100

    # 9. OBV slope (linear regression over 10 bars)
    obv = df["obv"]
    features["obv_slope_10"] = obv.rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=True
    )

    # 10. Volume ratio
    vol_sma20 = df["volume"].rolling(20).mean()
    features["volume_ratio_20"] = df["volume"] / vol_sma20.replace(0, np.nan)

    # 11-12. Price momentum
    features["price_momentum_5"] = df["close"].pct_change(5) * 100
    features["price_momentum_20"] = df["close"].pct_change(20) * 100

    # 13. EMA crossover
    ema_diff = df["ema_9"] - df["ema_21"]
    features["ema9_ema21_cross"] = ema_diff / df["close"] * 100

    # 14-15. Close vs moving averages
    features["close_vs_ema50"] = (df["close"] - df["ema_50"]) / df["ema_50"] * 100
    features["close_vs_sma200"] = (df["close"] - df["sma_200"]) / df["sma_200"] * 100

    # 16-18. ADX, Stochastic
    features["adx_14"] = df["adx_14"]
    features["stoch_k"] = df["stoch_k"]
    features["stoch_d"] = df["stoch_d"]

    # 19. High-Low range
    features["high_low_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100

    # 20. Gap
    features["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100

    # 21-22. Order book features (scalar for current bar, fill for all bars)
    features["spread_bps"] = spread_bps
    features["book_imbalance"] = book_imbalance

    # 23. Returns volatility
    returns = df["close"].pct_change()
    features["returns_volatility_20"] = returns.rolling(20).std() * 100

    # 24-25. Calendar features
    if "time" in df.columns:
        dt_col = pd.to_datetime(df["time"])
        features["day_of_week"] = dt_col.dt.dayofweek
        # hour_of_day: meaningful for intraday candles only.
        # For daily candles all hours are 0 (or identical) → replace with constant 0
        # so the feature carries no spurious signal.
        hours = dt_col.dt.hour
        hour_std = hours.std()
        features["hour_of_day"] = hours if (hour_std and hour_std > 1) else 0
    else:
        features["day_of_week"] = 2  # Default mid-week
        features["hour_of_day"] = 12

    # === ML4T-style enhancements (Jansen ch.4 "Alpha Factor Research") ========

    # Multi-horizon returns: captures momentum at different frequencies.
    # Jansen: "lagged returns at multiple horizons are the single most
    # information-dense feature family for price prediction on tabular data."
    for h in (1, 3, 10, 21):
        features[f"returns_{h}"] = df["close"].pct_change(h) * 100

    # Higher moments of the return distribution — regime signature
    features["returns_skew_20"] = returns.rolling(20).skew()
    features["returns_kurt_20"] = returns.rolling(20).kurt()

    # Cross-time quantile rank: where does the current 20-bar momentum sit
    # within the last 60 bars of momentum observations?  Normalised to [0,1].
    # Jansen ch.4: "rank-based features are scale-invariant and robust to
    # outliers, making them the standard for factor models."
    mom_20 = df["close"].pct_change(20)
    features["momentum_rank_60"] = mom_20.rolling(60).rank(pct=True)

    # Donchian channel position: price vs 60-bar high/low range.
    # 0.0 = at 60-bar low, 1.0 = at 60-bar high.
    high_60 = df["high"].rolling(60).max()
    low_60 = df["low"].rolling(60).min()
    donchian_range = (high_60 - low_60).replace(0, np.nan)
    features["price_position_60"] = (df["close"] - low_60) / donchian_range

    # Fractional differentiation of log-close at d=0.4 (LdP ch.5).
    # Stationary level signal — see FEATURE_NAMES note above.  The first
    # ~30 rows will be NaN (FFD kernel needs lookback); LightGBM handles
    # NaN natively via missing-value branch.
    try:
        from analysis.frac_diff import frac_diff_ffd

        log_close = np.log(df["close"].clip(lower=1e-9))
        features["close_ffd_04"] = frac_diff_ffd(log_close, d=0.4).to_numpy()
    except Exception as _e:
        # Never let frac-diff break the pipeline — zero-fill keeps schema.
        features["close_ffd_04"] = 0.0

    # === Macro context features ==============================================
    # Always emit the full MACRO_FEATURE_NAMES schema (zero-filled when macro
    # data is unavailable) so train and inference paths produce identical
    # columns — even when find_instrument / get_candles fail for a driver.
    for col in MACRO_FEATURE_NAMES:
        features[col] = 0.0
    # Same contract for the cross-asset block.
    for col in CROSS_ASSET_FEATURE_NAMES:
        features[col] = 0.0

    if macro_df is not None and not macro_df.empty and "time" in df.columns:
        try:
            main_idx = pd.to_datetime(df["time"], utc=True)
            # Align macro to our bar timestamps via ffill-reindex: each bar
            # gets the most recent macro close at or before its timestamp.
            # This handles macros that trade on a different schedule (e.g.,
            # BRENT futures hours vs MOEX equities) without forward-look bias.
            macro_sorted = macro_df.sort_index()

            # Keep per-macro aligned Series around for the cross-asset block
            aligned_by_name: dict[str, pd.Series] = {}

            for close_col in macro_sorted.columns:
                if not close_col.endswith("_close"):
                    continue
                name = close_col[: -len("_close")]
                aligned = macro_sorted[close_col].reindex(main_idx, method="ffill").to_numpy()
                aligned_series = pd.Series(aligned, index=features.index)
                aligned_by_name[name] = aligned_series

                r1_col = f"{name}_return_1"
                r5_col = f"{name}_return_5"
                if r1_col in MACRO_FEATURE_NAMES:
                    features[r1_col] = (aligned_series.pct_change(1) * 100).fillna(0.0).to_numpy()
                if r5_col in MACRO_FEATURE_NAMES:
                    features[r5_col] = (aligned_series.pct_change(5) * 100).fillna(0.0).to_numpy()

            # --- Cross-asset derivatives ------------------------------------
            # These need the ALIGNED macro series + the stock's own returns,
            # so we build them here (not in a second pass) to avoid recomputing
            # the ffill-reindex.
            stock_ret = df["close"].pct_change()
            # IMOEX block — only if we actually resolved IMOEX
            imoex_aligned = aligned_by_name.get("imoex")
            if imoex_aligned is not None:
                imoex_ret = imoex_aligned.pct_change()
                # 20-bar rolling correlation (Pearson) between stock and index
                corr_20 = stock_ret.rolling(20).corr(imoex_ret)
                features["imoex_corr_20"] = corr_20.fillna(0.0).to_numpy()
                # Relative strength: cumulative alpha of stock over market, 20 bars
                # We use sum of log-like returns in % to stay scale-consistent
                rel_strength = (stock_ret - imoex_ret).rolling(20).sum() * 100
                features["imoex_rel_strength_20"] = rel_strength.fillna(0.0).to_numpy()
                # Market regime: IMOEX realised vol over 20 bars (%)
                imoex_vol = imoex_ret.rolling(20).std() * 100
                features["imoex_volatility_20"] = imoex_vol.fillna(0.0).to_numpy()

            # BRENT block — only useful for energy-exposed stocks, but cheap
            brent_aligned = aligned_by_name.get("brent")
            if brent_aligned is not None:
                brent_ret = brent_aligned.pct_change()
                brent_corr = stock_ret.rolling(20).corr(brent_ret)
                features["brent_corr_20"] = brent_corr.fillna(0.0).to_numpy()

        except Exception as e:
            # Leave macro + cross-asset columns zero-filled; never let macro
            # breakage take down the whole feature pipeline.
            logger.warning(f"Macro feature alignment failed, leaving zero-filled: {e}")

    # === Instrument-kind block ===============================================
    # These are the kind/asset-class structural features the pooled model uses
    # to learn per-family rules.  They're ordered here to match
    # INSTRUMENT_KIND_FEATURE_NAMES so any schema drift is caught in tests.
    from config.instruments import asset_class_code as _asset_class_code

    is_future = instrument_kind == "future"
    features["is_future"] = 1.0 if is_future else 0.0
    features["asset_class_code"] = int(_asset_class_code(ticker, instrument_kind))

    # --- Futures-only structural features ---
    n_rows = len(features)

    if is_future and expiration_date is not None and "time" in df.columns:
        try:
            exp_ts = pd.Timestamp(expiration_date)
            if exp_ts.tzinfo is None:
                exp_ts = exp_ts.tz_localize("UTC")
            else:
                exp_ts = exp_ts.tz_convert("UTC")
            bar_ts = pd.to_datetime(df["time"], utc=True)
            days_left = (exp_ts - bar_ts).dt.total_seconds().to_numpy() / 86400.0
            # Clip so past-expiry bars don't emit negative numbers
            days_left = np.clip(days_left, 0.0, None)
            features["days_to_expiry"] = days_left
            features["in_roll_window"] = (days_left <= float(roll_window_days)).astype(int)
        except Exception as e:
            logger.debug(f"days_to_expiry calc failed, zero-filling: {e}")
            features["days_to_expiry"] = 0.0
            features["in_roll_window"] = 0
    else:
        features["days_to_expiry"] = 0.0
        features["in_roll_window"] = 0

    # --- Session flag (MSK-based) ---
    # MOEX trading sessions: morning 07:00-13:59 MSK, main 14:00-18:44,
    # evening 19:00-23:49 (futures only).  Give the model a low-cardinality
    # flag so it can gate features by session without needing the noisier
    # hour_of_day.
    if "time" in df.columns:
        try:
            dt_utc = pd.to_datetime(df["time"], utc=True)
            # All Tinkoff candle times are UTC; convert to Moscow for session math
            hour = dt_utc.dt.tz_convert("Europe/Moscow").dt.hour.to_numpy()
            sess = np.zeros(n_rows, dtype=int)
            sess[(hour >= 7) & (hour <= 13)] = 1  # morning
            sess[(hour >= 14) & (hour <= 18)] = 2  # main
            sess[(hour >= 19) & (hour <= 23)] = 3  # evening (futures)
            features["session_flag"] = sess
        except Exception as e:
            logger.debug(f"session_flag calc failed, zero-filling: {e}")
            features["session_flag"] = 0
    else:
        features["session_flag"] = 0

    # --- Basis slope (fallback-safe) ---
    # Real computation requires an aligned spot price series — not plumbed yet.
    # Zero-fill keeps the schema stable so the feature can be backfilled
    # without retraining the full column layout.
    if is_future and basis is not None and len(basis) == n_rows:
        features["basis_slope"] = np.asarray(basis, dtype=float)
    else:
        features["basis_slope"] = 0.0

    # --- OI momentum (fallback-safe) ---
    if is_future and open_interest is not None and len(open_interest) == n_rows:
        oi = pd.Series(open_interest).astype(float)
        features["oi_momentum_20"] = oi.pct_change(20).fillna(0.0).to_numpy()
    else:
        features["oi_momentum_20"] = 0.0

    return features


def build_target(
    df: pd.DataFrame, horizon: int = 5, threshold_pct: float | None = None
) -> pd.Series:
    """Legacy fixed-horizon target (kept for backwards compatibility).

    Prefer ``build_triple_barrier_target`` for new code — it adapts to
    volatility regime and yields better-separated classes.
    """
    if threshold_pct is None:
        # Adaptive threshold: ~half the typical bar range
        if "atr_14" in df.columns:
            med_atr = df["atr_14"].median()
            last_close = df["close"].iloc[-1]
            if last_close > 0 and med_atr > 0:
                threshold_pct = float(med_atr / last_close * 100) * 0.5
            else:
                threshold_pct = 1.0
        else:
            # Fallback: 1-day ATR estimate from High-Low
            if "high" in df.columns and "low" in df.columns:
                hl_pct = ((df["high"] - df["low"]) / df["close"].replace(0, np.nan) * 100).median()
                threshold_pct = float(hl_pct) * 0.5 if hl_pct > 0 else 1.0
            else:
                threshold_pct = 1.0
        # Clamp: don't let threshold be too tight (noise) or too wide (no signal)
        threshold_pct = max(0.3, min(threshold_pct, 3.0))

    forward_return = df["close"].pct_change(horizon).shift(-horizon) * 100

    target = pd.Series(1, index=df.index)  # Default: hold
    target[forward_return > threshold_pct] = 2  # Buy
    target[forward_return < -threshold_pct] = 0  # Sell

    return target


def _label_uniqueness(label_lifespans: list[tuple[int, int]], n: int) -> np.ndarray:
    """Sample-uniqueness weights (López de Prado AFML ch.4 §4.4).

    For each sample i with label spanning bars [start_i, end_i], count how
    many OTHER samples have labels that overlap any bar in [start_i, end_i].
    Weight = 1 / mean_overlap_over_label_span.

    Implementation note: rather than compute per-bar overlap explicitly
    (O(n²) memory), we use the snippet-4.1 approach — build an indicator
    of "concurrent labels at bar t" then average inverse-concurrency over
    each label's lifespan.

    Args:
        label_lifespans: list of (start_bar, end_bar) inclusive ranges, one
            per labelled sample.
        n: total number of bars in the source series.

    Returns:
        np.ndarray length len(label_lifespans), one uniqueness weight per
        labelled sample, in [0, 1].
    """
    if not label_lifespans:
        return np.array([], dtype=float)
    # Per-bar concurrency count: how many labels are alive at each bar.
    counts = np.zeros(n, dtype=np.int32)
    for s, e in label_lifespans:
        if s < 0 or e >= n or s > e:
            continue
        counts[s : e + 1] += 1
    counts = np.clip(counts, 1, None)  # avoid div-by-zero

    # Average uniqueness per label = mean(1/counts[start:end+1]).
    weights = np.zeros(len(label_lifespans), dtype=float)
    inv = 1.0 / counts
    for i, (s, e) in enumerate(label_lifespans):
        if s < 0 or e >= n or s > e:
            weights[i] = 0.0
            continue
        weights[i] = float(inv[s : e + 1].mean())
    return weights


def build_triple_barrier_target(
    df: pd.DataFrame,
    pt_multiplier: float = 2.0,
    sl_multiplier: float = 1.0,
    max_hold: int = 10,
    min_ret: float = 0.0,
    use_uniqueness: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Triple-barrier labels (López de Prado, *Advances in Financial ML* ch.3).

    For each bar ``i`` we simulate a long-entry at the close and watch the
    next ``max_hold`` bars.  Three barriers compete:

      * **Upper (profit-take)**: ``close_i + pt_multiplier * ATR_i``
      * **Lower (stop-loss)**:   ``close_i - sl_multiplier * ATR_i``
      * **Vertical (time-out)**: ``max_hold`` bars

    Label = 2 (buy) if upper hit first, 0 (sell) if lower hit first, else 1
    (hold) on time-out.  Because barriers scale with the bar's own ATR, the
    labels adapt to the volatility regime — quiet periods get tighter
    barriers, so "buy" still means "meaningful move relative to normal noise."

    Returns
    -------
    labels : pd.Series[float]
        Labels in {0, 1, 2}, with NaN in the last ``max_hold`` rows (no
        forward data available).
    weights : pd.Series[float]
        |realised return| at barrier hit, normalised to mean 1 over valid
        samples.  Used as sample_weight during fitting — ML4T pattern for
        down-weighting indecisive / time-out samples.

    Parameters
    ----------
    pt_multiplier, sl_multiplier
        ATR multipliers for the upper / lower barriers.  Asymmetric default
        (2× / 1×) reflects the risk-reward target a human trader would demand.
    max_hold
        Horizon (in bars) before the vertical barrier terminates the trade.
    min_ret
        If > 0, any timeout with |return| < min_ret is forced to hold=1;
        otherwise timeouts are labelled by realised-return sign.  Default 0
        keeps the pure López de Prado semantics (timeout → hold).
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    if "atr_14" in df.columns:
        atr = df["atr_14"].to_numpy(dtype=float)
    else:
        # Fallback: rough ATR from high-low range
        hl = (df["high"] - df["low"]).rolling(14).mean().to_numpy(dtype=float)
        atr = hl

    labels = np.full(n, np.nan, dtype=float)
    realised_ret = np.zeros(n, dtype=float)
    # Track each label's lifespan [start_bar, end_bar] for uniqueness weighting.
    # end_bar = bar at which the barrier was hit (or i+max_hold on time-out).
    lifespans: list[tuple[int, int]] = []
    label_indices: list[int] = []

    for i in range(n - max_hold):
        c_i = close[i]
        a_i = atr[i]
        if not np.isfinite(c_i) or c_i <= 0:
            continue
        if not np.isfinite(a_i) or a_i <= 0:
            # No volatility info → fall back to fixed 1.5% / 0.75% bands
            pt_px = c_i * 1.015
            sl_px = c_i * 0.9925
        else:
            pt_px = c_i + pt_multiplier * a_i
            sl_px = c_i - sl_multiplier * a_i

        label = 1  # default: time-out → hold
        ret = 0.0
        end_bar = i + max_hold
        for j in range(1, max_hold + 1):
            k = i + j
            if high[k] >= pt_px:
                label = 2
                ret = (pt_px - c_i) / c_i
                end_bar = k
                break
            if low[k] <= sl_px:
                label = 0
                ret = (sl_px - c_i) / c_i
                end_bar = k
                break
        else:
            # No barrier hit; realised return from time-out close
            ret = (close[i + max_hold] - c_i) / c_i
            if min_ret > 0 and abs(ret) >= min_ret:
                # Optional: label timeouts by sign when move is substantial
                label = 2 if ret > 0 else 0

        labels[i] = label
        realised_ret[i] = ret
        lifespans.append((i, end_bar))
        label_indices.append(i)

    labels_series = pd.Series(labels, index=df.index, dtype=float)

    # Sample weights = |realised return| × uniqueness, both normalised.
    # |realised return| (Jansen ch.4) — emphasises decisive PT/SL hits over
    #     tiny time-out drifts.
    # Uniqueness (LdP ch.4 §4.4) — when many labels overlap (high-frequency
    #     labelling) each individual sample carries less unique information;
    #     downweight to avoid inflating the IID assumption that LightGBM
    #     makes about training samples.
    abs_ret = np.abs(realised_ret)
    abs_ret[np.isnan(labels)] = np.nan

    if use_uniqueness and lifespans:
        uniq = _label_uniqueness(lifespans, n)
        # Combine multiplicatively: a sample contributes if it was both
        # decisive AND unique.  Then normalise jointly.
        combined = np.full(n, np.nan, dtype=float)
        for idx, w_u in zip(label_indices, uniq):
            base = abs_ret[idx]
            if np.isnan(base):
                continue
            combined[idx] = base * w_u
        weights_series = pd.Series(combined, index=df.index, dtype=float)
    else:
        weights_series = pd.Series(abs_ret, index=df.index, dtype=float)

    mean_w = weights_series.dropna().mean()
    if mean_w and mean_w > 0:
        weights_series = weights_series / mean_w
    # Floor tiny weights so time-out samples still get meaningful gradient.
    # A floor of 0.25 means the least-decisive sample carries 1/4 the weight
    # of a decisive one — enough signal to learn the "hold" pattern but
    # still emphasising clean PT/SL hits.  (0.1 was too aggressive: hold
    # recall collapsed to ~3 % because hold samples had near-zero weight.)
    weights_series = weights_series.clip(lower=0.25)

    return labels_series, weights_series
