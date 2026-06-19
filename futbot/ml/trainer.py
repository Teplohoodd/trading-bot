"""Train a per-concept ML-gate ENSEMBLE bundle.

For each concept we train TWO independent binary classifiers:

    up_model: predicts P(price hits +1.5 ATR upper barrier within horizon)
    dn_model: predicts P(price hits −1.5 ATR lower barrier within horizon)

Each model is itself a 5-seed bag of LightGBM classifiers, with isotonic
calibration fit on out-of-fold predictions.  Validation is purged
walk-forward (López de Prado AFML ch. 7): the dataset is split into K
contiguous folds, each fold's test period gets an EMBARGO of `horizon`
bars excluded from training (otherwise the triple-barrier labels leak
the future into the train set via overlap).

Metrics reported per model:
  * ROC AUC  — discrimination ability, threshold-independent
  * Brier    — probability calibration (lower = better)
  * IC       — Spearman rank corr between predicted prob and realised
              forward return (the actual business metric for a veto gate)
  * F1@best  — F1 at the threshold that maximises validation F1

Bundle layout (data/models/futbot_<concept>.joblib):
    {
        "version": "binary-bag-v1",
        "concept": str,
        "trained_at": iso,
        "feature_names": list[str],
        "label_horizon": int,
        "label_up_mult": float,
        "label_dn_mult": float,
        "up": { "calibrators": [iso, ...], "models": [lgbm, ...], "metrics": {...} },
        "dn": { "calibrators": [iso, ...], "models": [lgbm, ...], "metrics": {...} },
    }

The wrapper in ml/model.py loads this bundle and exposes:
    predict_proba_up_dn(features_row) → (p_up, p_dn)
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from scipy.stats import spearmanr

from futbot.ml import datasets as ds
from futbot.ml import features as feat
from futbot.ml import labels as lab

warnings.filterwarnings("ignore", message="X does not have valid feature names")

logger = logging.getLogger("futbot.ml.trainer")


DEFAULT_HORIZON = 5
DEFAULT_UP_MULT = 1.5
DEFAULT_DN_MULT = 1.5
TEST_FRACTION = 0.20
CV_FOLDS = 4
N_SEEDS = 5
MODEL_DIR = Path("data/models")


# ── LightGBM hyperparams ─────────────────────────────────────────────────────
# Small grid; for ~5-7k daily rows full HP search overfits the CV.  These
# values are conservative defaults that work across the concepts we care about.
# `class_weight=balanced` is intentional — the positive class is the
# minority (~30 % of rows) and we want the model to pay attention to it.
BASE_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.035,
    num_leaves=15,  # was 31 — narrower to avoid overfitting
    max_depth=5,
    min_child_samples=50,  # was 30 — more conservative
    reg_alpha=0.2,
    reg_lambda=0.2,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.85,
    class_weight="balanced",
    verbose=-1,
    n_jobs=-1,
)


# ── Data prep ────────────────────────────────────────────────────────────────
def _prepare(concept: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Load daily OHLCV, build features, triple-barrier label, split into
    binary y_up / y_dn.  Returns (df, tb_label, y_up, y_dn) aligned by index.
    Final `horizon` rows are dropped to avoid label leakage into test."""
    df = ds.load_concept(concept)
    if df.empty or len(df) < 200:
        raise ValueError(f"{concept}: only {len(df)} bars — not enough to train")
    df = feat.build_features(df, dropna=True)
    tb = lab.triple_barrier(
        df,
        horizon=DEFAULT_HORIZON,
        up_mult=DEFAULT_UP_MULT,
        dn_mult=DEFAULT_DN_MULT,
    )
    df = df.iloc[:-DEFAULT_HORIZON].reset_index(drop=True)
    tb = tb.iloc[:-DEFAULT_HORIZON].reset_index(drop=True)
    y_up, y_dn = lab.binary_labels(tb)
    # Also compute the realised forward return — used for IC metric
    return df, tb, y_up, y_dn


def _forward_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Realised log-return from each bar's close to `horizon` bars ahead."""
    fwd = np.log(df["close"].shift(-horizon) / df["close"])
    return fwd


# ── Walk-forward CV with embargo ────────────────────────────────────────────
def _wf_folds(n: int, k: int, embargo: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate (train_idx, test_idx) for purged walk-forward CV.

    For fold i in [0, k):
      * test = block i of n/k contiguous rows
      * train = all rows BEFORE the test block, with `embargo` rows at the
                end of train removed (to prevent the triple-barrier label
                of a train row leaking into the test period).

    Yields fold definitions oldest-first.  Fold 0 has a tiny train; later
    folds have more data — this matches a real "train on history, test on
    near-future" deployment.
    """
    fold_size = n // (k + 1)  # +1 so the first 1/(k+1) of data is always train-only warm-up
    folds = []
    warmup = fold_size
    for i in range(k):
        test_start = warmup + i * fold_size
        test_end = test_start + fold_size
        if test_end > n:
            test_end = n
        train_end = max(0, test_start - embargo)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) > 100 and len(test_idx) > 10:
            folds.append((train_idx, test_idx))
    return folds


# ── Bagged ensemble training ─────────────────────────────────────────────────
def _train_bag(
    X_train: np.ndarray, y_train: np.ndarray, n_seeds: int = N_SEEDS
) -> list[LGBMClassifier]:
    """Train N seeded LGBM classifiers on the same data.  Seed variance is
    the cheapest variance reduction we can do — no extra hyperparameter
    tuning, no extra features needed."""
    models = []
    for seed in range(n_seeds):
        m = LGBMClassifier(**BASE_PARAMS, random_state=seed)
        m.fit(X_train, y_train)
        models.append(m)
    return models


def _predict_bag(models: list[LGBMClassifier], X: np.ndarray) -> np.ndarray:
    """Average the positive-class probability across the bag."""
    probas = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probas, axis=0)


# ── Calibration ──────────────────────────────────────────────────────────────
def _fit_calibrator(p_raw: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
    """Isotonic regression: monotonic mapping from raw prob → calibrated prob.
    Works on OOF predictions to keep calibration honest."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_raw, y_true)
    return iso


# ── Metrics ──────────────────────────────────────────────────────────────────
def _metrics_binary(
    y_true: np.ndarray, p_pred: np.ndarray, fwd_ret: np.ndarray | None = None
) -> dict:
    out: dict = {}
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(float(roc_auc_score(y_true, p_pred)), 4)
    else:
        out["roc_auc"] = None
    out["brier"] = round(float(brier_score_loss(y_true, p_pred)), 4)
    out["pos_rate"] = round(float(y_true.mean()), 4)
    out["mean_pred"] = round(float(p_pred.mean()), 4)

    # F1 / precision / recall at the threshold that maximises F1 on this set
    thresholds = np.arange(0.30, 0.71, 0.025)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        yhat = (p_pred >= t).astype(int)
        if yhat.sum() == 0:
            continue
        f1 = f1_score(y_true, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    out["best_threshold"] = round(float(best_t), 3)
    yhat = (p_pred >= best_t).astype(int)
    out["f1_at_best"] = round(float(f1_score(y_true, yhat, zero_division=0)), 4)
    out["precision_at_best"] = round(float(precision_score(y_true, yhat, zero_division=0)), 4)
    out["recall_at_best"] = round(float(recall_score(y_true, yhat, zero_division=0)), 4)

    # Information Coefficient — Spearman corr of P(positive) vs realised return.
    # For y_up: positive correlation means model gives higher prob when forward
    # return is higher.  For y_dn: should be NEGATIVE (higher prob when return
    # is lower).
    if fwd_ret is not None and len(fwd_ret) == len(p_pred):
        mask = ~np.isnan(fwd_ret)
        if mask.sum() >= 30:
            r, _ = spearmanr(p_pred[mask], fwd_ret[mask])
            out["ic"] = round(float(r), 4) if not np.isnan(r) else None
    return out


# ── Public API ───────────────────────────────────────────────────────────────
def train_concept(concept: str, *, verbose: bool = True) -> dict:
    df, tb, y_up, y_dn = _prepare(concept)
    fwd = _forward_return(df, DEFAULT_HORIZON).values
    X = df[feat.FEATURE_NAMES].values
    n = len(df)

    # Hold-out test = last TEST_FRACTION of the data.  All CV happens in
    # the prefix; final models retrain on prefix and we report metrics on
    # the held-out tail.
    n_test = int(n * TEST_FRACTION)
    n_train = n - n_test
    X_tr, X_te = X[:n_train], X[n_train:]
    fwd_tr, fwd_te = fwd[:n_train], fwd[n_train:]

    bundle = {
        "version": "binary-bag-v1",
        "concept": concept,
        "trained_at": datetime.utcnow().isoformat(),
        "feature_names": list(feat.FEATURE_NAMES),
        "label_horizon": DEFAULT_HORIZON,
        "label_up_mult": DEFAULT_UP_MULT,
        "label_dn_mult": DEFAULT_DN_MULT,
        "n_train": int(n_train),
        "n_test": int(n_test),
    }

    for side, y in (("up", y_up.values), ("dn", y_dn.values)):
        y_tr, y_te = y[:n_train], y[n_train:]
        if verbose:
            logger.info(
                f"{concept} [{side}]: pos_rate(train)={y_tr.mean():.3f} "
                f"pos_rate(test)={y_te.mean():.3f}"
            )

        # ── 1. Walk-forward CV to (a) sanity-check and (b) collect OOF preds
        # for calibrator fitting.  We train a fresh bag at each fold from
        # scratch — slower but means the calibrator sees genuinely
        # out-of-fold predictions across the whole train period.
        folds = _wf_folds(n_train, CV_FOLDS, embargo=DEFAULT_HORIZON)
        oof_pred = np.full(n_train, np.nan)
        cv_metrics_per_fold = []
        for fi, (tr_idx, te_idx) in enumerate(folds):
            bag = _train_bag(X_tr[tr_idx], y_tr[tr_idx])
            p_fold = _predict_bag(bag, X_tr[te_idx])
            oof_pred[te_idx] = p_fold
            m_fold = _metrics_binary(
                y_tr[te_idx],
                p_fold,
                fwd_ret=fwd_tr[te_idx],
            )
            cv_metrics_per_fold.append(m_fold)

        # ── 2. Fit calibrator on the (concatenated) OOF predictions
        oof_mask = ~np.isnan(oof_pred)
        if oof_mask.sum() < 50:
            calibrator = None
            logger.warning(f"{concept} [{side}]: too few OOF preds; skipping calibration")
        else:
            calibrator = _fit_calibrator(oof_pred[oof_mask], y_tr[oof_mask])

        # ── 3. Retrain final bag on the full train set
        final_bag = _train_bag(X_tr, y_tr)

        # ── 4. Test-set metrics (this is what we trust most)
        p_te_raw = _predict_bag(final_bag, X_te)
        p_te = calibrator.transform(p_te_raw) if calibrator is not None else p_te_raw
        test_m = _metrics_binary(y_te, p_te, fwd_ret=fwd_te)

        # ── 5. CV-average metrics (rough generalisation health-check)
        cv_avg = {}
        if cv_metrics_per_fold:
            for k in ("roc_auc", "brier", "f1_at_best", "ic"):
                vals = [m[k] for m in cv_metrics_per_fold if m.get(k) is not None]
                if vals:
                    cv_avg[f"cv_{k}_mean"] = round(float(np.mean(vals)), 4)
                    cv_avg[f"cv_{k}_std"] = round(float(np.std(vals)), 4)
        metrics = {**test_m, **cv_avg, "cv_folds": len(folds)}

        bundle[side] = {
            "models": final_bag,
            "calibrator": calibrator,
            "metrics": metrics,
        }
        if verbose:
            logger.info(
                f"{concept} [{side}]: TEST roc_auc={test_m.get('roc_auc')} "
                f"brier={test_m['brier']} f1={test_m['f1_at_best']} "
                f"ic={test_m.get('ic')} thr={test_m['best_threshold']}"
            )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / f"futbot_{concept}.joblib"
    joblib.dump(bundle, out)
    if verbose:
        logger.info(f"{concept}: saved → {out}")
    return {
        "concept": concept,
        "n_train": n_train,
        "n_test": n_test,
        "up_metrics": bundle["up"]["metrics"],
        "dn_metrics": bundle["dn"]["metrics"],
    }


def train_all() -> dict[str, dict]:
    out = {}
    for c in ds.CONCEPTS:
        try:
            out[c] = train_concept(c)
        except Exception as e:
            logger.exception(f"  {c}: training failed: {e}")
            out[c] = {"error": str(e)}
    return out
