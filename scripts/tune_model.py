"""
Hyperparameter tuning for LightGBM + feature selection.

Usage:
    python scripts/tune_model.py [--trials 40] [--tickers 15] [--output data/tuning_results.json]

What it does
------------
1. Loads training data via the same broker pipeline as the main bot
2. Runs Optuna (Bayesian optimisation) over key LGBM hyperparameters
   using TimeSeriesSplit to avoid lookahead bias
3. Evaluates feature importance via permutation importance on the best model
4. Writes the best hyperparameters + ranked feature list to JSON
   (the main model.py reads this file automatically on next /retrain)

Literature basis
----------------
* Chen & Guestrin (2016)  XGBoost — original paper on gradient boosting regularisation
* Prokhorenkova et al. (2018) CatBoost — ordered boosting insight applied here as
  TimeSeriesSplit CV to prevent temporal leakage
* Brownlees & Engle (2012) on volatility clustering → ATR/ADX as primary features
* Fischer & Krauss (2018) "Deep learning with LSTM..." — confirms LGBM beats LSTM
  on tabular financial data when features are < 50 and samples < 10 000
* Key practical insight (Kaggle M5 / Jane Street winners): for financial tabular data
  subsample=0.7, colsample=0.7, small learning rate + NO early stopping on small
  validation sets outperforms aggressive early stopping.

Horizon tuning mode (--tune-horizon)
------------------------------------
Grid-searches the triple-barrier `max_hold` parameter over several values and
reports blended CV score (F1 + IC) + pure F1 + pure IC per horizon.  Based on:

* López de Prado (2018) "Advances in Financial Machine Learning" ch.3 —
  horizon should match the holding period business logic, not be guessed
  from theory; tune on data.
* Krauss, Do & Huck (2017) "Deep neural networks, gradient-boosted trees,
  random forests: Statistical arbitrage on the S&P 500" — 1-day horizon
  on daily data optimal for swing stats-arb; ≈6-8 hours on hourly.
* Fischer & Krauss (2018) — 5-10 day horizons on daily; ≈ 10-20 hourly bars.
* Zhang, Zohren & Roberts (2020) "Deep learning for portfolio optimization" —
  multi-horizon ensembling outperforms single fixed horizon; use grid
  result as the primary, not the only, label path.

Expected answer band for this repo (180-day hourly dataset, Russian blue-chips):
7-20 bars for shares, 3-7 for futures.  Grid [5, 10, 15, 20, 30] brackets it.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from analysis.features import (
    build_features,
    build_triple_barrier_target,
    FEATURE_NAMES,
)
from analysis.indicators import compute_indicators
from analysis.macro import MacroProvider
from analysis.screener import _candles_to_df
from config.settings import Settings
from core.broker import BrokerClient
from t_tech.invest import CandleInterval

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("tune")


def _align_proba(clf: LGBMClassifier, raw_proba: np.ndarray) -> np.ndarray:
    """Map predict_proba to fixed 3-column layout [sell=0, hold=1, buy=2].

    When training data is missing a class (common in short-horizon folds
    where ALL samples hit the same barrier), LightGBM returns fewer columns.
    This function pads the missing class columns with 0 so downstream
    ``proba[:, 2] - proba[:, 0]`` never raises "index 2 out of bounds".
    """
    classes = [int(c) for c in clf.classes_]
    n_classes = 3
    if raw_proba.ndim == 1:
        if len(classes) == n_classes:
            return raw_proba
        aligned = np.zeros(n_classes)
        for col_idx, cls in enumerate(classes):
            if 0 <= cls < n_classes:
                aligned[cls] = raw_proba[col_idx]
        return aligned
    else:
        if raw_proba.shape[1] == n_classes:
            return raw_proba
        aligned = np.zeros((raw_proba.shape[0], n_classes))
        for col_idx, cls in enumerate(classes):
            if 0 <= cls < n_classes:
                aligned[:, cls] = raw_proba[:, col_idx]
        return aligned


# ── config ────────────────────────────────────────────────────────────────────
N_TRIALS = 40  # Optuna trials
N_TICKERS = 15  # tickers to fetch training data from
N_CV_FOLDS = 5  # TimeSeriesSplit folds
MIN_BARS = 80  # minimum clean samples per ticker
# Triple-barrier parameters (match ml/trainer.py ModelTrainer.TB_*)
# Symmetric barriers keep class distribution balanced — asymmetric 2:1 PT:SL
# collapses ~60 % of samples into "sell" for a random walk (see Jansen ch.4).
TB_PT = 2.0  # upper barrier = 2 × ATR
TB_SL = 2.0  # lower barrier = 2 × ATR (symmetric)
# Per-kind time-out horizon (= embargo size for purged CV).
# Shares mean-revert over ~10h on hourly candles; futures chop much more
# so we shorten the label horizon to 5 bars — otherwise a +1 % → -1 %
# → +0.5 % trip over 10 bars labels both "buy" and "sell" samples, which
# the model can never disambiguate.  Matches ml/trainer.py TB_MAX_HOLD /
# TB_MAX_HOLD_FUTURES in config.Settings.
TB_MAX_HOLD = 20  # shares: 20 bars ≈ 20h (~2 trading days) — IC-optimal
TB_MAX_HOLD_FUTURES = 10  # futures: 10 bars (= 20 × 0.5 ratio)
DAILY_LOOKBACK = 1095  # 3 years of daily candles (to survive SMA-200 warmup)


def _max_hold_for_kind(kind: str) -> int:
    """Pick the right triple-barrier horizon for this instrument kind."""
    return TB_MAX_HOLD_FUTURES if kind == "future" else TB_MAX_HOLD


# ── horizon tuning grid ───────────────────────────────────────────────────────
# Applied to BOTH shares and futures uniformly inside the grid search — we
# want apples-to-apples comparison across horizons, not a fixed per-kind ratio.
# After picking the winning horizon, we scale futures by FUTURES_HORIZON_RATIO
# (< 1.0) to preserve the empirical "futures mean-revert faster" split from
# the compacted ml/trainer defaults.
HORIZON_GRID = [5, 10, 15, 20, 30]
FUTURES_HORIZON_RATIO = 0.5  # futures horizon = round(share_horizon * this)

# Default LGBM params used inside the horizon grid search — we hold hyperparams
# constant so horizon is the only variable.  These mirror ml/trainer.py's
# pre-tuning defaults (before Optuna runs).
DEFAULT_LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
}

OUTPUT = ROOT / "data" / "tuning_results.json"
HORIZON_OUTPUT = ROOT / "data" / "horizon_tuning.json"

# Features to ALWAYS exclude (importance=0 in previous run + domain knowledge)
# macro features (brent/usdrub) are kept in — tickers updated (LCOc1 for
# BRENT, USDRUBF for USD/RUB), need a re-tune to know their importance.
ZERO_IMPORTANCE = {"macd_signal_dist", "spread_bps", "book_imbalance", "hour_of_day"}


# ── data pipeline ─────────────────────────────────────────────────────────────
async def fetch_raw_data(settings: Settings, broker: BrokerClient, n_tickers: int) -> list[dict]:
    """Fetch raw candles + features per ticker WITHOUT building labels.

    Returns a list of dicts, one per ticker that passed the bar/row filters:
        {"ticker", "kind", "df", "feat"}
    where `df` carries the raw OHLCV+indicators (needed by
    build_triple_barrier_target) and `feat` holds the pre-computed feature
    matrix (FEATURE_NAMES columns).

    Separating this step from label generation lets the horizon grid search
    re-use a single expensive API fetch across N horizons.
    """
    from analysis.screener import Screener

    screener = Screener(broker, top_n=n_tickers)
    macro_provider = MacroProvider(broker)

    # Fetch BOTH long and short watchlists so the tuned hyperparameters
    # aren't biased toward bullish-regime features.  Triple-barrier labels
    # are symmetric (PT=SL) so a mixed universe gives the model a more
    # direction-agnostic training set, which matters when the live bot
    # trades both sides (screener.direction="long" + "short" combined).
    logger.info("Fetching watchlist (long + short)...")
    long_wl = await screener.scan_universe(direction="long")
    short_wl = await screener.scan_universe(direction="short")
    # Dedupe by figi, keep higher score
    merged: dict[str, dict] = {}
    for cand in long_wl + short_wl:
        existing = merged.get(cand["figi"])
        if existing is None or cand.get("score", 0) > existing.get("score", 0):
            merged[cand["figi"]] = cand
    watchlist = sorted(merged.values(), key=lambda c: c.get("score", 0), reverse=True)
    tickers = watchlist[:n_tickers]
    long_n = sum(1 for t in tickers if t.get("direction") == "long")
    short_n = sum(1 for t in tickers if t.get("direction") == "short")
    logger.info(
        f"Got {len(tickers)} tickers ({long_n} long + {short_n} short): "
        f"{[t['ticker'] for t in tickers]}"
    )

    raw: list[dict] = []
    now = datetime.now(timezone.utc)

    for item in tickers:
        figi, ticker = item["figi"], item["ticker"]
        kind = item.get("kind", "share") or "share"
        try:
            # Try hourly (180 days)
            used_interval = CandleInterval.CANDLE_INTERVAL_HOUR
            candles = await broker.get_candles(
                figi,
                now - timedelta(days=180),
                now,
                interval=used_interval,
            )
            # Fall back to daily (3 years) — SMA-200 warmup needs ≥200 bars,
            # so 1095 calendar days (~780 trading days) gives ~580 clean rows.
            if not candles:
                used_interval = CandleInterval.CANDLE_INTERVAL_DAY
                candles = await broker.get_candles(
                    figi,
                    now - timedelta(days=DAILY_LOOKBACK),
                    now,
                    interval=used_interval,
                )
            if len(candles) < MIN_BARS:
                logger.info(f"  skip {ticker}: only {len(candles)} bars")
                continue

            df = _candles_to_df(candles)
            df = compute_indicators(df)

            # Fetch macro context at the same interval as the instrument's
            # candles.  MacroProvider caches across tickers, so only the first
            # ticker pays the API cost per interval per 15-min window.
            times = pd.to_datetime(df["time"], utc=True)
            try:
                macro_df = await macro_provider.get_macro_df(
                    from_dt=times.min().to_pydatetime(),
                    to_dt=times.max().to_pydatetime(),
                    interval=used_interval,
                )
            except Exception as e:
                logger.debug(f"  {ticker}: macro fetch skipped ({e})")
                macro_df = None

            feat = build_features(
                df, spread_bps=0.0, book_imbalance=0.0, macro_df=macro_df, instrument_kind=kind
            )

            raw.append(
                {
                    "ticker": ticker,
                    "kind": kind,
                    "df": df,
                    "feat": feat,
                }
            )
            logger.info(f"  {ticker} ({kind}): {len(df)} bars fetched")
        except Exception as e:
            logger.warning(f"  {ticker}: {e}")

    if not raw:
        raise RuntimeError("No training data fetched")
    return raw


def build_pooled_dataset(
    raw: list[dict],
    max_hold_shares: int,
    max_hold_futures: int | None = None,
    pt_multiplier: float = TB_PT,
    sl_multiplier: float = TB_SL,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, int]]:
    """Build triple-barrier labels on top of pre-fetched raw per-ticker data.

    Separating this from fetch_raw_data lets the horizon grid re-use a single
    (expensive) API fetch across many horizon values.  Returns
    (X, y, w, samples_by_kind).
    """
    if max_hold_futures is None:
        max_hold_futures = max_hold_shares

    all_X, all_y, all_w = [], [], []
    samples_by_kind: dict[str, int] = {"share": 0, "future": 0}

    for item in raw:
        ticker = item["ticker"]
        kind = item["kind"]
        df = item["df"]
        feat = item["feat"]
        max_hold_k = max_hold_futures if kind == "future" else max_hold_shares
        try:
            labels, weights = build_triple_barrier_target(
                df,
                pt_multiplier=pt_multiplier,
                sl_multiplier=sl_multiplier,
                max_hold=max_hold_k,
            )
            combined = pd.concat(
                [feat[FEATURE_NAMES], labels.rename("target"), weights.rename("weight")],
                axis=1,
            ).dropna()
            if len(combined) < MIN_BARS:
                if verbose:
                    logger.info(f"  skip {ticker}: only {len(combined)} clean rows")
                continue
            all_X.append(combined[FEATURE_NAMES])
            all_y.append(combined["target"].astype(int))
            all_w.append(combined["weight"].astype(float))
            samples_by_kind[kind] = samples_by_kind.get(kind, 0) + len(combined)
        except Exception as e:
            logger.warning(f"  {ticker}: label build failed: {e}")

    if not all_X:
        raise RuntimeError(
            f"No labelled rows at horizon={max_hold_shares}/"
            f"{max_hold_futures} — try a different horizon"
        )

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)
    w = pd.concat(all_w, ignore_index=True)

    if verbose:
        counts = y.value_counts().sort_index()
        names = {0: "sell", 1: "hold", 2: "buy"}
        dist = "  ".join(f"{names[k]}: {v} ({v/len(y)*100:.0f}%)" for k, v in counts.items())
        kind_dist = "  ".join(f"{k}: {n}" for k, n in samples_by_kind.items() if n > 0)
        logger.info(f"Horizon {max_hold_shares}/{max_hold_futures}: " f"{len(X)} samples | {dist}")
        if kind_dist:
            logger.info(f"  by kind: {kind_dist}")

    return X, y, w, samples_by_kind


async def fetch_data(
    settings: Settings, broker: BrokerClient, n_tickers: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, weights) pooled from top screener tickers.

    Thin wrapper around fetch_raw_data + build_pooled_dataset that preserves
    the original one-call API used by the Optuna path in main().
    """
    raw = await fetch_raw_data(settings, broker, n_tickers)
    X, y, w, _ = build_pooled_dataset(
        raw,
        max_hold_shares=TB_MAX_HOLD,
        max_hold_futures=TB_MAX_HOLD_FUTURES,
    )
    return X, y, w


# ── feature selection ─────────────────────────────────────────────────────────
def select_features(all_features: list[str]) -> list[str]:
    """Remove known-zero-importance features."""
    return [f for f in all_features if f not in ZERO_IMPORTANCE]


# ── purged cross-validation (Jansen ch.6 / López de Prado ch.7) ──────────────
def cv_score(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    features: list[str],
    n_splits: int = N_CV_FOLDS,
    embargo: int = max(TB_MAX_HOLD, TB_MAX_HOLD_FUTURES),
) -> float:
    """Purged TimeSeriesSplit → blended F1 + IC score.

    The objective combines classification quality (F1) with ranking quality
    (Information Coefficient) because Optuna's sampler needs a single scalar
    but trading actually cares more about rank consistency than class match:
        score = 0.6 * f1_weighted + 2.0 * IC_clamped
    IC typically sits in [0, 0.15] so the 2× weight roughly equalises ranges.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    Xf = X[features].fillna(0).values
    yv = y.values
    wv = w.values if w is not None else None
    scores = []

    for train_idx, val_idx in tscv.split(Xf):
        # Purge: drop last `embargo` rows of train so their forward-looking
        # labels don't overlap the validation window.
        if len(train_idx) > embargo:
            train_idx_purged = train_idx[:-embargo]
        else:
            train_idx_purged = train_idx
        if len(train_idx_purged) < 100 or len(val_idx) < 20:
            continue

        clf = LGBMClassifier(**params, verbose=-1, n_jobs=-1)
        fit_kwargs = {}
        if wv is not None:
            fit_kwargs["sample_weight"] = wv[train_idx_purged]
        clf.fit(Xf[train_idx_purged], yv[train_idx_purged], **fit_kwargs)

        pred = clf.predict(Xf[val_idx])
        f1 = f1_score(yv[val_idx], pred, average="weighted", zero_division=0)

        proba = _align_proba(clf, clf.predict_proba(Xf[val_idx]))
        score_exp = proba[:, 2] - proba[:, 0]  # P(buy) - P(sell)
        y_dir = np.where(yv[val_idx] == 2, 1, np.where(yv[val_idx] == 0, -1, 0))
        if len(np.unique(y_dir)) > 1 and np.std(score_exp) > 0:
            ic = spearmanr(score_exp, y_dir).correlation
            ic = 0.0 if np.isnan(ic) else float(ic)
        else:
            ic = 0.0

        # Blend: F1 dominates, IC adds "does the model rank directions correctly?"
        scores.append(0.6 * f1 + 2.0 * max(0.0, ic))

    return float(np.mean(scores)) if scores else 0.0


def cv_metrics(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    features: list[str],
    n_splits: int = N_CV_FOLDS,
    embargo: int = max(TB_MAX_HOLD, TB_MAX_HOLD_FUTURES),
) -> dict:
    """Same CV loop as cv_score but returns blended + raw F1 variants + raw IC.

    Reports BOTH F1_weighted and F1_macro:
    - F1_weighted: weights each class by its sample count.  INFLATED when
      hold% is high (a "predict hold always" model looks great).
    - F1_macro: equal weight per class regardless of frequency.  Honest
      when classes are imbalanced — can't be gamed by always predicting hold.

    For HORIZON SELECTION always prefer IC (Spearman rank correlation of
    P(buy)−P(sell) vs actual direction) — it is completely immune to class
    imbalance and directly measures directional edge, which is all a
    swing-trading bot cares about.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    Xf = X[features].fillna(0).values
    yv = y.values
    wv = w.values if w is not None else None
    f1s_w, f1s_m, ics, blends = [], [], [], []

    for train_idx, val_idx in tscv.split(Xf):
        if len(train_idx) > embargo:
            train_idx_purged = train_idx[:-embargo]
        else:
            train_idx_purged = train_idx
        if len(train_idx_purged) < 100 or len(val_idx) < 20:
            continue

        clf = LGBMClassifier(**params, verbose=-1, n_jobs=-1)
        fit_kwargs = {}
        if wv is not None:
            fit_kwargs["sample_weight"] = wv[train_idx_purged]
        clf.fit(Xf[train_idx_purged], yv[train_idx_purged], **fit_kwargs)

        pred = clf.predict(Xf[val_idx])
        f1_w = f1_score(yv[val_idx], pred, average="weighted", zero_division=0)
        f1_m = f1_score(yv[val_idx], pred, average="macro", zero_division=0)

        proba = _align_proba(clf, clf.predict_proba(Xf[val_idx]))
        score_exp = proba[:, 2] - proba[:, 0]
        y_dir = np.where(yv[val_idx] == 2, 1, np.where(yv[val_idx] == 0, -1, 0))
        if len(np.unique(y_dir)) > 1 and np.std(score_exp) > 0:
            ic = spearmanr(score_exp, y_dir).correlation
            ic = 0.0 if np.isnan(ic) else float(ic)
        else:
            ic = 0.0

        f1s_w.append(float(f1_w))
        f1s_m.append(float(f1_m))
        ics.append(float(ic))
        # Blended still uses F1_weighted for consistency with Optuna objective.
        # Do NOT use blended for horizon selection — use IC directly.
        blends.append(0.6 * f1_w + 2.0 * max(0.0, ic))

    if not f1s_w:
        return {"blended": 0.0, "f1_weighted": 0.0, "f1_macro": 0.0, "ic": 0.0, "n_folds": 0}
    return {
        "blended": float(np.mean(blends)),
        "f1_weighted": float(np.mean(f1s_w)),
        "f1_macro": float(np.mean(f1s_m)),
        "ic": float(np.mean(ics)),
        "n_folds": len(f1s_w),
    }


# ── horizon grid search (López de Prado 2018 ch.3) ────────────────────────────
def run_horizon_grid(
    raw: list[dict],
    features: list[str],
    horizons: list[int] = HORIZON_GRID,
    futures_ratio: float = FUTURES_HORIZON_RATIO,
    params: dict | None = None,
) -> list[dict]:
    """Grid-search triple-barrier max_hold with fixed LGBM hyperparams.

    For each horizon H:
      - shares max_hold=H, futures max_hold=max(2, round(H*ratio))
      - embargo = H (covers the full forward look-ahead per label)
      - purged TimeSeriesSplit CV with constant hyperparams
      - reports IC, F1_macro, F1_weighted, blended, samples, hold%

    **WINNER = highest IC.**  F1_weighted is NOT used for selection because
    it is inflated by hold% (majority-class dominance at short horizons):
    at horizon=1, hold%≈93%, so a "predict hold always" model gets F1=0.93.
    F1_macro is reported as a sanity check (equal weight per class, immune
    to imbalance).  IC (Spearman rank of P(buy)−P(sell) vs actual direction)
    is the only metric that directly measures directional edge.

    Ref: Jansen "Machine Learning for Algorithmic Trading" ch.4 §4.4
         — IC as primary evaluation metric for return-prediction models.

    Returns results sorted by cv_ic_spearman descending.
    """
    params = params or DEFAULT_LGBM_PARAMS
    results = []
    logger.info(f"Grid search over {len(horizons)} horizons: {horizons}")
    logger.info(
        "NOTE: winner is selected by IC (Spearman), NOT by blended/F1_weighted.\n"
        "  F1_weighted is INFLATED when hold%>50% — short horizons look great\n"
        "  on F1 but IC reveals near-zero directional signal at h<5."
    )

    for h in horizons:
        h_fut = max(2, int(round(h * futures_ratio)))
        try:
            X, y, w, samples_by_kind = build_pooled_dataset(
                raw,
                max_hold_shares=h,
                max_hold_futures=h_fut,
                verbose=True,
            )
        except RuntimeError as e:
            logger.warning(f"Horizon {h}: {e}")
            continue

        counts = y.value_counts().sort_index().to_dict()
        names = {0: "sell", 1: "hold", 2: "buy"}
        class_dist = {names[k]: int(v) for k, v in counts.items()}
        hold_pct = class_dist.get("hold", 0) / max(1, len(y))

        inflation_warning = " ⚠ F1_weighted inflated (hold%>50%)" if hold_pct > 0.5 else ""

        metrics = cv_metrics(params, X, y, w, features, embargo=h)
        result = {
            "horizon_shares": h,
            "horizon_futures": h_fut,
            "n_samples": int(len(X)),
            "samples_by_kind": {k: int(v) for k, v in samples_by_kind.items()},
            "class_distribution": class_dist,
            "hold_pct": round(hold_pct, 3),
            "cv_ic_spearman": round(metrics["ic"], 4),  # PRIMARY selector
            "cv_f1_macro": round(metrics["f1_macro"], 4),  # balanced F1
            "cv_f1_weighted": round(metrics["f1_weighted"], 4),  # inflated at short h
            "cv_blended": round(metrics["blended"], 4),
            "cv_folds": metrics["n_folds"],
            "f1_weighted_inflated": hold_pct > 0.5,
        }
        results.append(result)
        logger.info(
            f"  horizon={h:>2} (fut={h_fut}): "
            f"samples={len(X):>5}  hold%={hold_pct*100:>4.1f}  "
            f"IC={metrics['ic']:+.4f}  "
            f"F1_macro={metrics['f1_macro']:.4f}  "
            f"F1_w={metrics['f1_weighted']:.4f}"
            f"{inflation_warning}"
        )

    # Sort by IC — the only unbiased directional-signal metric
    results.sort(key=lambda r: r["cv_ic_spearman"], reverse=True)
    return results


# ── Optuna objective ──────────────────────────────────────────────────────────
def make_objective(X: pd.DataFrame, y: pd.Series, w: pd.Series, features: list[str]):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 10, 50),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 3.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            # NOTE: no class_weight — sample_weight from triple-barrier already
            # encodes both class balance and within-class importance.
            "random_state": 42,
        }
        return cv_score(params, X, y, w, features)

    return objective


# ── feature importance ────────────────────────────────────────────────────────
def rank_features(
    best_params: dict, X: pd.DataFrame, y: pd.Series, w: pd.Series, features: list[str]
) -> list[dict]:
    """Train final model (sample-weighted) and compute permutation importance."""
    clf = LGBMClassifier(**best_params, verbose=-1, n_jobs=-1)
    Xf = X[features].fillna(0)
    clf.fit(Xf.values, y.values, sample_weight=w.values if w is not None else None)

    # 1. LGBM split importance
    lgbm_imp = dict(zip(features, clf.feature_importances_))

    # 2. Permutation importance (on last 20% of data to keep temporal order)
    split_idx = int(len(Xf) * 0.8)
    X_val = Xf.iloc[split_idx:].values
    y_val = y.iloc[split_idx:].values
    perm = permutation_importance(
        clf, X_val, y_val, n_repeats=10, random_state=42, scoring="f1_weighted"
    )

    ranked = []
    for i, feat in enumerate(features):
        ranked.append(
            {
                "feature": feat,
                "lgbm_importance": int(lgbm_imp[feat]),
                "permutation_mean": round(float(perm.importances_mean[i]), 5),
                "permutation_std": round(float(perm.importances_std[i]), 5),
            }
        )

    # Sort by permutation mean (more reliable than split importance)
    ranked.sort(key=lambda x: x["permutation_mean"], reverse=True)
    return ranked


# ── main ──────────────────────────────────────────────────────────────────────
async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--tickers", type=int, default=N_TICKERS)
    parser.add_argument("--output", type=str, default=str(OUTPUT))
    parser.add_argument(
        "--tune-horizon",
        action="store_true",
        help="Grid-search triple-barrier max_hold horizon with fixed LGBM "
        "params (fast; writes to data/horizon_tuning.json) instead of "
        "the full Optuna run.  Use this to pick the best TB_MAX_HOLD "
        "before running the full hyperparameter tune.",
    )
    parser.add_argument(
        "--horizon-grid",
        type=str,
        default=",".join(str(h) for h in HORIZON_GRID),
        help="Comma-separated horizons (shares) to test when --tune-horizon "
        "is passed.  Default: " + ",".join(str(h) for h in HORIZON_GRID),
    )
    parser.add_argument(
        "--horizon-output",
        type=str,
        default=str(HORIZON_OUTPUT),
        help="Output JSON path for --tune-horizon results.",
    )
    args = parser.parse_args()

    settings = Settings()
    broker = BrokerClient(settings.T_INVEST_TOKEN, settings.T_INVEST_ACCOUNT_ID)
    await broker.connect()

    try:
        # Fetch the raw per-ticker data ONCE — both the horizon grid and
        # the Optuna path reuse it, so we avoid hitting the Tinkoff API
        # multiple times for the same candles.
        raw = await fetch_raw_data(settings, broker, args.tickers)
    finally:
        await broker.disconnect()

    features = select_features(FEATURE_NAMES)
    logger.info(f"Feature set: {len(features)} features (removed {ZERO_IMPORTANCE})")

    # ──────────────────────────────────────────────────────────────────────────
    # Horizon grid search path — skip Optuna entirely.
    # ──────────────────────────────────────────────────────────────────────────
    if args.tune_horizon:
        try:
            horizons = [int(h.strip()) for h in args.horizon_grid.split(",") if h.strip()]
        except ValueError:
            raise SystemExit(
                f"--horizon-grid must be comma-separated ints, got: {args.horizon_grid!r}"
            )
        if not horizons:
            raise SystemExit("--horizon-grid produced no horizons")

        logger.info(
            f"Horizon grid search: {horizons} (futures = round(h × {FUTURES_HORIZON_RATIO}))"
        )
        logger.info(f"Using fixed LGBM params for fair comparison: {DEFAULT_LGBM_PARAMS}")

        grid_results = run_horizon_grid(
            raw=raw,
            features=features,
            horizons=horizons,
            futures_ratio=FUTURES_HORIZON_RATIO,
            params=DEFAULT_LGBM_PARAMS,
        )

        if not grid_results:
            raise SystemExit("Horizon grid produced no results (all horizons failed)")

        # Summary table — sorted by IC (primary selector)
        logger.info("\n── Horizon grid summary (sorted by IC — primary selection metric) ──")
        logger.info(
            "  IC = Spearman(P(buy)−P(sell), actual_direction) — unbiased.\n"
            "  F1_macro = equal weight per class — also unbiased.\n"
            "  F1_weighted ⚠ inflated when hold%>50% — DO NOT use for selection."
        )
        logger.info(
            f"{'horizon':>8} {'fut':>4} {'samples':>8} {'hold%':>6} "
            f"{'IC':>8} {'F1_macro':>9} {'F1_w⚠':>7}"
        )
        for r in grid_results:
            warn = "⚠" if r.get("f1_weighted_inflated") else " "
            logger.info(
                f"{r['horizon_shares']:>8} {r['horizon_futures']:>4} "
                f"{r['n_samples']:>8} {r['hold_pct']*100:>5.1f}% "
                f"{r['cv_ic_spearman']:>+8.4f} {r['cv_f1_macro']:>9.4f} "
                f"{r['cv_f1_weighted']:>6.4f}{warn}"
            )
        winner = grid_results[0]
        logger.info(
            f"\nWinner by IC: horizon_shares={winner['horizon_shares']}  "
            f"horizon_futures={winner['horizon_futures']}  "
            f"IC={winner['cv_ic_spearman']}  F1_macro={winner['cv_f1_macro']}"
        )
        logger.info(
            "Next step: update TB_MAX_HOLD / TB_MAX_HOLD_FUTURES in "
            "config/settings.py AND scripts/tune_model.py (and "
            "ml/trainer.py reads the same constants), then re-run "
            "this script without --tune-horizon for the full Optuna "
            "hyperparameter search at the new horizon."
        )

        out_path = Path(args.horizon_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tuned_at": datetime.utcnow().isoformat(),
            "methodology": (
                "Grid search over triple-barrier max_hold horizon with fixed "
                "LGBM params.  Purged TimeSeriesSplit CV (Jansen ch.6) with "
                "embargo = horizon.  Selection criterion: blended 0.6·F1 + "
                "2·max(IC, 0) — same as main Optuna objective.  "
                "Refs: López de Prado 2018 ch.3; Krauss et al. 2017; "
                "Fischer & Krauss 2018."
            ),
            "futures_horizon_ratio": FUTURES_HORIZON_RATIO,
            "fixed_lgbm_params": DEFAULT_LGBM_PARAMS,
            "triple_barrier": {
                "pt_multiplier": TB_PT,
                "sl_multiplier": TB_SL,
            },
            "n_tickers": args.tickers,
            "feature_count": len(features),
            "winner": winner,
            "all_results": grid_results,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"\nHorizon tuning results saved to {out_path}")
        return payload

    # ──────────────────────────────────────────────────────────────────────────
    # Standard Optuna path — use the currently-configured horizons.
    # ──────────────────────────────────────────────────────────────────────────
    X, y, w, _ = build_pooled_dataset(
        raw,
        max_hold_shares=TB_MAX_HOLD,
        max_hold_futures=TB_MAX_HOLD_FUTURES,
    )
    cv_embargo = max(TB_MAX_HOLD, TB_MAX_HOLD_FUTURES)
    logger.info(
        f"Triple-barrier config: PT={TB_PT}×ATR, SL={TB_SL}×ATR, "
        f"max_hold shares={TB_MAX_HOLD} / futures={TB_MAX_HOLD_FUTURES} bars; "
        f"CV embargo={cv_embargo} (max of per-kind horizons)"
    )

    # ── Optuna study ──────────────────────────────────────────────────────────
    logger.info(f"Running {args.trials} Optuna trials (F1 + IC blended objective)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(make_objective(X, y, w, features), n_trials=args.trials, show_progress_bar=True)

    best = study.best_params
    best_score = study.best_value
    logger.info(f"\nBest blended CV score (0.6·F1 + 2·IC): {best_score:.4f}")
    logger.info(f"Best params: {json.dumps(best, indent=2)}")

    # ── Feature ranking ───────────────────────────────────────────────────────
    logger.info("Computing permutation importance on best model...")
    best["random_state"] = 42
    ranked = rank_features(best, X, y, w, features)

    logger.info("\nFeature ranking (by permutation importance):")
    for r in ranked:
        bar = "█" * max(0, int(r["permutation_mean"] * 1000))
        logger.info(
            f"  {r['feature']:30s}  perm={r['permutation_mean']:+.5f} ±{r['permutation_std']:.5f}  {bar}"
        )

    # Features with negative permutation importance → removing them HELPS
    noise_feats = [r["feature"] for r in ranked if r["permutation_mean"] <= 0]
    good_feats = [r["feature"] for r in ranked if r["permutation_mean"] > 0]
    logger.info(f"\nGood features ({len(good_feats)}): {good_feats}")
    logger.info(f"Noise/useless  ({len(noise_feats)}): {noise_feats}")

    # ── Save results ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "tuned_at": datetime.utcnow().isoformat(),
        "methodology": "triple-barrier labels (López de Prado ch.3) + "
        "purged TimeSeriesSplit + sample-weighted fit "
        "(Jansen ML4T ch.4/ch.6)",
        "cv_blended_score": round(best_score, 4),
        "triple_barrier": {
            "pt_multiplier": TB_PT,
            "sl_multiplier": TB_SL,
            "max_hold_shares": TB_MAX_HOLD,
            "max_hold_futures": TB_MAX_HOLD_FUTURES,
            "embargo": cv_embargo,
        },
        "best_params": best,
        "features_ranked": ranked,
        "features_to_use": good_feats,
        "features_excluded": list(ZERO_IMPORTANCE) + noise_feats,
        "n_samples": len(X),
        "n_tickers": args.tickers,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {output_path}")
    logger.info("Run /retrain in the bot to apply the new hyperparameters.")
    return results


if __name__ == "__main__":
    asyncio.run(main())
