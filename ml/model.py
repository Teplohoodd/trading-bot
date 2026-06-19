"""LightGBM model wrapper: train, predict, save/load."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

logger = logging.getLogger(__name__)

# Path where scripts/tune_model.py writes its results.
_TUNING_RESULTS_PATH = Path(__file__).parent.parent / "data" / "tuning_results.json"


def _load_tuned_params() -> dict | None:
    """Load hyperparameters from the last Optuna tuning run, if available."""
    if _TUNING_RESULTS_PATH.exists():
        try:
            with open(_TUNING_RESULTS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            params = data.get("best_params", {})
            # Remove keys that aren't valid LGBMClassifier __init__ params
            params.pop("class_weight", None)
            params.pop("random_state", None)
            if params:
                logger.info(
                    f"Loaded tuned hyperparameters from {_TUNING_RESULTS_PATH} "
                    f"(cv_f1={data.get('cv_f1', '?')})"
                )
                return params
        except Exception as e:
            logger.warning(f"Could not load tuning results: {e}")
    return None


@dataclass
class ModelMetadata:
    figi: Optional[str]
    version: int
    accuracy: float
    f1: float
    train_samples: int
    feature_names: list[str]
    trained_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def _align_proba(model: LGBMClassifier, raw_proba: np.ndarray) -> np.ndarray:
    """Map predict_proba output to the fixed 3-column layout [sell, hold, buy].

    LightGBM only emits columns for classes that appear in the training data.
    If a class was absent from that ticker's labels (e.g., all bars hit the
    take-profit before the stop could fire → no sell label), the returned
    array has fewer than 3 columns and any indexing into column 2 raises
    "index 2 is out of bounds for axis 1 with size N".

    This function pads missing columns with 0.0 so the caller can always use
    column 0 = P(sell), column 1 = P(hold), column 2 = P(buy).

    Notes
    -----
    After padding, rows still sum to 1.0 (the missing class gets probability 0
    and the other classes keep their exact predict_proba values).
    No re-normalisation is needed.
    """
    classes = [int(c) for c in model.classes_]
    n_classes = 3

    if raw_proba.ndim == 1:
        # Single row returned as a 1-D array of length len(classes)
        if len(classes) == n_classes:
            return raw_proba
        aligned = np.zeros(n_classes)
        for col_idx, cls in enumerate(classes):
            if 0 <= cls < n_classes:
                aligned[cls] = raw_proba[col_idx]
        return aligned
    else:
        # Batch: shape (n_samples, len(classes))
        if raw_proba.shape[1] == n_classes:
            return raw_proba
        aligned = np.zeros((raw_proba.shape[0], n_classes))
        for col_idx, cls in enumerate(classes):
            if 0 <= cls < n_classes:
                aligned[:, cls] = raw_proba[:, col_idx]
        return aligned


class LGBMModel:
    """LightGBM 3-class classifier (sell=0, hold=1, buy=2)."""

    CLASSES = {0: "sell", 1: "hold", 2: "buy"}

    def __init__(self, model_path: Optional[Path] = None):
        self._model: Optional[LGBMClassifier] = None
        self._feature_names: list[str] = []
        self._metadata: Optional[ModelMetadata] = None
        # Per-asset-class isotonic calibrators for P(buy) and P(sell).
        # Keyed by asset_class_code (int).  Populated at train() time if the
        # feature is present + we have enough validation samples per class.
        # Schema: {cls_code: {"buy": IsotonicRegression, "sell": IsotonicRegression}}
        self._calibrators: dict[int, dict[str, IsotonicRegression]] = {}
        # Optional meta-labelling secondary classifier (LdP ch.3).  When set,
        # MLStrategy gates trades on meta_conf and scales position size by it.
        # Lazily imported to avoid circular dep at module-load time.
        self._meta_model = None  # type: ignore[var-annotated]

        if model_path and model_path.exists():
            self.load(model_path)

    @property
    def meta_model(self):
        return self._meta_model

    def set_meta_model(self, meta_model) -> None:
        """Attach a trained MetaLabellingModel.  Stored alongside primary in
        save(); loaded automatically by load()."""
        self._meta_model = meta_model

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def metadata(self) -> Optional[ModelMetadata]:
        return self._metadata

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        figi: Optional[str] = None,
        version: int = 1,
        sample_weight: Optional[pd.Series] = None,
        categorical_features: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Train the model and return validation metrics.

        Parameters
        ----------
        sample_weight
            Optional per-row training weights (ML4T pattern from López de
            Prado).  When using triple-barrier labels, weights ∝ |realised
            return| so clean PT/SL hits dominate over ambiguous time-outs.
        categorical_features
            Names of columns in X_train that LightGBM should treat as
            native categorical (no ordinal assumption).  Typically
            ``["asset_class_code"]`` in the pooled model — lets the tree
            split on asset family without needing one-hot encoding.  Only
            columns actually present in X_train are passed through;
            missing names are silently skipped so this function stays safe
            for callers training on partial feature sets.
        """
        self._feature_names = list(X_train.columns)

        n_train = len(X_train)

        # Prefer Optuna-tuned hyperparameters when available.
        # Fall back to sensible size-adaptive defaults otherwise.
        tuned = _load_tuned_params()
        if tuned:
            hparams = {**tuned}
        else:
            # Size-adaptive defaults (no tuning results yet)
            if n_train < 2000:
                hparams = dict(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    num_leaves=20,
                    min_child_samples=10,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.7,
                    reg_alpha=0.5,
                    reg_lambda=2.0,
                )
            elif n_train < 8000:
                hparams = dict(
                    n_estimators=400,
                    learning_rate=0.04,
                    max_depth=5,
                    num_leaves=28,
                    min_child_samples=15,
                    subsample=0.85,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    reg_alpha=0.2,
                    reg_lambda=1.0,
                )
            else:
                hparams = dict(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.9,
                    subsample_freq=1,
                    colsample_bytree=0.9,
                    reg_alpha=0.0,
                    reg_lambda=0.0,
                )

        # IMPORTANT: n_estimators from Optuna is already the optimal stopping point
        # found by cross-validation.  Do NOT use early_stopping here — it stops
        # training far too early when the validation set is small / noisy (we saw
        # n_iter=14 out of 2000 in previous runs).
        #
        # We keep class_weight="balanced" even when sample_weight is provided.
        # They are complementary, not duplicative:
        #   sample_weight   — ∝ |realised return|, weights clean PT/SL hits
        #                     more than indecisive time-outs (within-class).
        #   class_weight    — inversely ∝ class frequency, compensates for
        #                     natural label imbalance (across classes).
        # In LightGBM the final per-sample weight is their product, so each
        # term addresses a different failure mode.  Without class_weight the
        # model collapses onto the majority class when sample_weight is too
        # noisy to discriminate classes on its own.
        self._model = LGBMClassifier(
            **hparams,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

        # Handle NaN
        X_train = X_train.fillna(0)
        X_val = X_val.fillna(0)

        # Filter categorical_features down to columns actually present in
        # X_train so callers that pass a superset don't crash LightGBM.
        cat_feats = None
        if categorical_features:
            cat_feats = [c for c in categorical_features if c in X_train.columns]
            if not cat_feats:
                cat_feats = None

        fit_kwargs = {}
        if sample_weight is not None:
            sw = np.asarray(sample_weight, dtype=float)
            # Safety: replace NaN/negative weights with 1.0 so fit() never errors
            sw = np.where(np.isfinite(sw) & (sw > 0), sw, 1.0)
            fit_kwargs["sample_weight"] = sw
        if cat_feats is not None:
            fit_kwargs["categorical_feature"] = cat_feats

        self._model.fit(X_train, y_train, **fit_kwargs)

        # Warn early if a class is missing from training labels — common when
        # a short ticker history has all samples hit PT (no sell) or all SL
        # (no buy).  _align_proba() pads the missing column with 0 so prediction
        # still works, but IC will be degraded.
        trained_classes = set(int(c) for c in self._model.classes_)
        missing = {0, 1, 2} - trained_classes
        if missing:
            cls_names = {0: "sell", 1: "hold", 2: "buy"}
            logger.warning(
                f"Training data missing class(es): "
                f"{[cls_names[c] for c in sorted(missing)]} — "
                f"probabilities for those classes will be 0. "
                f"Consider fetching more history for this ticker."
            )

        # --- Validation metrics --------------------------------------------
        y_pred = self._model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")

        # Information Coefficient (Jansen ch.6): Spearman rank correlation
        # between the model's expected-return score (P(buy) - P(sell)) and
        # the true direction encoded as {-1, 0, +1}.  This is the metric
        # that actually matters for trading — a model with 40 % F1 but
        # IC=0.05 is more profitable than one with 50 % F1 but IC=0.0.
        proba = _align_proba(self._model, self._model.predict_proba(X_val))
        expected_score = proba[:, 2] - proba[:, 0]  # P(buy) - P(sell)
        y_dir = np.where(np.asarray(y_val) == 2, 1, np.where(np.asarray(y_val) == 0, -1, 0))
        if len(np.unique(y_dir)) > 1 and np.std(expected_score) > 0:
            ic_stat = spearmanr(expected_score, y_dir)
            ic = float(ic_stat.correlation) if not np.isnan(ic_stat.correlation) else 0.0
        else:
            ic = 0.0

        self._metadata = ModelMetadata(
            figi=figi,
            version=version,
            accuracy=round(acc, 4),
            f1=round(f1, 4),
            train_samples=len(X_train),
            feature_names=self._feature_names,
        )

        logger.info(
            f"Model trained: acc={acc:.4f}, f1={f1:.4f}, ic={ic:+.4f}, " f"samples={len(X_train)}"
        )
        logger.info(
            f"\n{classification_report(y_val, y_pred, target_names=['sell', 'hold', 'buy'], zero_division=0)}"
        )

        # --- Per-asset-class isotonic calibration ---------------------------
        # Raw LightGBM probabilities with class_weight="balanced" + sample_weight
        # are systematically mis-calibrated (they optimise log-loss, not
        # probability fidelity).  Isotonic calibration (Niculescu-Mizil &
        # Caruana 2005) is a monotone, non-parametric map from raw P(buy)
        # to calibrated P(buy) that preserves rank but matches the empirical
        # frequency of the positive class.  Fitting one per asset_class_code
        # lets futures (where raw P(buy) is more dispersed) have a different
        # mapping than shares — addresses the main reason pooled models
        # underperform per-asset models on probability thresholds.
        #
        # Fallback: only calibrate classes with ≥ 50 validation samples AND
        # both positive/negative outcomes.  Otherwise we skip (prediction
        # falls back to raw probabilities), so calibration never makes the
        # model worse, only better.
        self._calibrators = {}
        if "asset_class_code" in X_val.columns and len(X_val) >= 50:
            try:
                y_val_arr = np.asarray(y_val)
                codes = X_val["asset_class_code"].astype(int).to_numpy()
                buy_p = proba[:, 2]
                sell_p = proba[:, 0]
                unique_codes = np.unique(codes)
                for cls_code in unique_codes:
                    mask = codes == cls_code
                    n = int(mask.sum())
                    if n < 50:
                        continue
                    y_buy = (y_val_arr[mask] == 2).astype(float)
                    y_sell = (y_val_arr[mask] == 0).astype(float)
                    entry: dict[str, IsotonicRegression] = {}
                    if 0 < y_buy.sum() < n:  # has both positives + negatives
                        iso_buy = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                        iso_buy.fit(buy_p[mask], y_buy)
                        entry["buy"] = iso_buy
                    if 0 < y_sell.sum() < n:
                        iso_sell = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                        iso_sell.fit(sell_p[mask], y_sell)
                        entry["sell"] = iso_sell
                    if entry:
                        self._calibrators[int(cls_code)] = entry
                if self._calibrators:
                    logger.info(
                        f"Fitted isotonic calibrators for "
                        f"{len(self._calibrators)} asset class(es): "
                        f"{sorted(self._calibrators.keys())}"
                    )
            except Exception as e:
                logger.warning(f"Isotonic calibration failed, skipping: {e}")
                self._calibrators = {}

        return {"accuracy": acc, "f1": f1, "ic": ic}

    # Minimum probability for a directional signal.
    # Below this, we return "hold" — avoids acting on coin-flip predictions.
    #
    # Calibration note: base rate for a balanced 3-class problem is 0.333.
    # With class_weight="balanced" and triple-barrier labels we observe typical
    # max-class probabilities in the 0.40–0.55 range for confident samples.
    # MIN_CONFIDENCE=0.38 means "at least 14 % above base rate" — strict
    # enough to filter coin-flips, loose enough to let real signals through.
    MIN_CONFIDENCE = 0.38

    def _apply_calibration(
        self, proba: np.ndarray, asset_class_codes: np.ndarray | None
    ) -> np.ndarray:
        """Apply per-asset-class isotonic calibration to raw P(sell)/P(buy).

        Returns a new array with calibrated columns 0 + 2 and a re-normalised
        column 1 (hold) so each row still sums to 1.  When no calibrator
        exists for a row's asset class, that row is passed through unchanged.
        Falls back to raw probabilities on any error — calibration must
        never make predictions worse when it fails.
        """
        if not self._calibrators or asset_class_codes is None:
            return proba
        try:
            out = proba.copy()
            codes = asset_class_codes.astype(int)
            unique_codes = np.unique(codes)
            for cls_code in unique_codes:
                entry = self._calibrators.get(int(cls_code))
                if not entry:
                    continue
                mask = codes == cls_code
                if "sell" in entry:
                    out[mask, 0] = entry["sell"].predict(proba[mask, 0])
                if "buy" in entry:
                    out[mask, 2] = entry["buy"].predict(proba[mask, 2])
            # Re-normalise: hold = 1 − sell − buy, clipped to ≥ 0; then
            # renormalise the whole row to sum to 1 for numerical safety.
            sell_col = np.clip(out[:, 0], 0.0, 1.0)
            buy_col = np.clip(out[:, 2], 0.0, 1.0)
            # If sell+buy already exceeds 1 (rare), rescale proportionally
            sb = sell_col + buy_col
            over = sb > 1.0
            if over.any():
                sell_col[over] = sell_col[over] / sb[over]
                buy_col[over] = buy_col[over] / sb[over]
            hold_col = 1.0 - sell_col - buy_col
            hold_col = np.clip(hold_col, 0.0, 1.0)
            out[:, 0] = sell_col
            out[:, 1] = hold_col
            out[:, 2] = buy_col
            row_sum = out.sum(axis=1, keepdims=True)
            row_sum[row_sum <= 0] = 1.0
            out = out / row_sum
            return out
        except Exception as e:
            logger.debug(f"Calibration failed, using raw proba: {e}")
            return proba

    def predict(self, X: pd.DataFrame) -> tuple[str, float]:
        """Predict direction and confidence for a single row or latest row.

        Returns (direction_str, confidence).

        Only signals buy/sell when the winning class probability exceeds
        MIN_CONFIDENCE (40 %).  Below that threshold the prediction is
        returned as "hold" even if argmax points to buy/sell.  This
        prevents acting on near-random predictions especially on small
        training sets.

        When per-asset calibrators are available, probabilities are passed
        through isotonic regression *before* the MIN_CONFIDENCE gate —
        so the gate acts on calibrated (empirical-frequency-matching)
        probabilities, not raw LightGBM scores.
        """
        if not self.is_trained:
            return "hold", 0.0

        X = X.fillna(0)
        if len(X.shape) == 1:
            X = X.to_frame().T

        # Use only last row
        X_last = X[self._feature_names].iloc[[-1]]
        # _align_proba guarantees 3-column output even if model was trained
        # on a 2-class subset (e.g., no "sell" samples hit the barrier).
        raw = _align_proba(self._model, self._model.predict_proba(X_last))
        codes = (
            X_last["asset_class_code"].to_numpy() if "asset_class_code" in X_last.columns else None
        )
        proba = self._apply_calibration(raw, codes)[0]

        sell_p, hold_p, buy_p = float(proba[0]), float(proba[1]), float(proba[2])

        # Directional signal only if clearly above the hold baseline
        if buy_p >= sell_p and buy_p >= hold_p and buy_p >= self.MIN_CONFIDENCE:
            return "buy", buy_p
        if sell_p > buy_p and sell_p >= hold_p and sell_p >= self.MIN_CONFIDENCE:
            return "sell", sell_p

        # Not confident enough → hold, report the best directional prob as context
        best_directional = max(buy_p, sell_p)
        return "hold", best_directional

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Raw probabilities, always shape (n_samples, 3): [P(sell), P(hold), P(buy)].

        Applies per-asset-class isotonic calibration when available — see
        ``_apply_calibration`` for the fallback contract.
        """
        if not self.is_trained:
            return np.array([[0.33, 0.34, 0.33]])
        X = X.fillna(0)[self._feature_names]
        raw = _align_proba(self._model, self._model.predict_proba(X))
        codes = X["asset_class_code"].to_numpy() if "asset_class_code" in X.columns else None
        return self._apply_calibration(raw, codes)

    def feature_importance(self) -> dict[str, float]:
        """Get feature importance (gain-based)."""
        if not self.is_trained:
            return {}
        importance = self._model.feature_importances_
        return dict(sorted(zip(self._feature_names, importance), key=lambda x: x[1], reverse=True))

    def save(self, path: Path):
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Bundle meta-labelling model inline so the .joblib is self-contained.
        # Storing as a serialised sklearn-style object dict avoids the
        # circular-import dance MetaLabellingModel would create at top level.
        meta_bundle = None
        if self._meta_model is not None and getattr(self._meta_model, "is_trained", False):
            meta_bundle = {
                "model": self._meta_model._model,
                "feature_names": self._meta_model._feature_names,
                "threshold": self._meta_model.threshold,
                "metadata": self._meta_model.metadata,
            }
        joblib.dump(
            {
                "model": self._model,
                "feature_names": self._feature_names,
                "metadata": self._metadata,
                "calibrators": self._calibrators,
                "meta_bundle": meta_bundle,
            },
            path,
        )
        logger.info(
            f"Model saved to {path}" f"{' (with meta-labelling model)' if meta_bundle else ''}"
        )

    def load(self, path: Path):
        """Load model from disk."""
        data = joblib.load(path)
        self._model = data["model"]
        self._feature_names = data["feature_names"]
        self._metadata = data["metadata"]
        # Backwards compat: old model files predate calibration; default to empty
        self._calibrators = data.get("calibrators", {}) or {}
        meta_bundle = data.get("meta_bundle")
        if meta_bundle:
            try:
                from ml.meta_model import MetaLabellingModel

                m = MetaLabellingModel(threshold=meta_bundle.get("threshold", 0.55))
                m._model = meta_bundle["model"]
                m._feature_names = meta_bundle["feature_names"]
                m.metadata = meta_bundle.get("metadata")
                self._meta_model = m
            except Exception as e:
                logger.warning(f"Failed to load meta model from bundle: {e}")
                self._meta_model = None
        else:
            self._meta_model = None
        logger.info(
            f"Model loaded from {path} "
            f"(calibrators: {len(self._calibrators)} asset classes, "
            f"meta: {'YES' if self._meta_model else 'no'})"
        )


def _log_evaluation(period: int):
    """LightGBM callback for periodic logging."""

    def callback(env):
        if env.iteration % period == 0:
            logger.debug(f"LGB iteration {env.iteration}")

    callback.order = 10
    return callback


def _early_stopping(stopping_rounds: int):
    """LightGBM early stopping callback."""
    from lightgbm import early_stopping

    return early_stopping(stopping_rounds=stopping_rounds, verbose=False)
