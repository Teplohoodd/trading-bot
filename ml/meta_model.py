"""Meta-labelling secondary classifier (López de Prado, AFML ch.3).

Concept:
  Primary model says: "this looks like a BUY (or SELL or HOLD)."
  Meta model says:    "given the features AND the primary's claim, was the
                      primary likely to be correct on this kind of bar?"

  We trade only when BOTH agree:
      action = primary_direction
      size   ∝ primary_conf × meta_conf

Why it works (Bailey/Hudson & Thames 2019, LdP 2018 §3.3):
  * The primary is forced to predict 3 classes balanced for the triple-
    barrier problem.  It will produce many "buy" calls that look weak
    in retrospect but the model still chose buy because that was the
    least-bad of three options.
  * The meta is a BINARY problem on a smaller dataset (only the bars
    where the primary said buy or sell), which it can learn with
    higher precision than the noisier 3-class primary.
  * Suppressing low-meta-confidence trades trades RECALL for PRECISION
    — exactly the lever a trader wants when each false positive costs
    real money via commission + slippage.

Training data flow:
  1. Fit the primary on the WHOLE training set with purged CV.
  2. Collect the primary's OUT-OF-FOLD predictions (one per training row)
     — these are unbiased estimates of what the primary would say in live
     trading.  THIS IS CRITICAL: training the meta on in-sample primary
     predictions causes massive overfitting (Theory of Bagging, AFML §3.3).
  3. Filter to rows where primary predicted buy or sell (drop "hold" —
     those don't generate trades).
  4. Build meta-target: 1 if primary's directional prediction matched the
     ground-truth triple-barrier label (buy→2, sell→0), 0 otherwise.
  5. Train the meta on the same features + primary-prediction columns,
     using purged CV again, with sample weights ∝ |realised return|.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass
class MetaModelMetadata:
    accuracy: float
    f1: float
    precision: float
    recall: float
    n_train: int
    n_pos: int  # positive class count (primary was correct)
    feature_names: list[str]
    threshold: float = 0.55  # meta-confidence floor used at inference time
    trained_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# Feature names appended ON TOP of the primary feature set when training
# the meta model.  These let it learn "the primary's confidence calibration
# is unreliable when adx_14 < 20", etc.
META_AUGMENT_FEATURES = [
    "primary_pred_buy",  # primary's P(buy)
    "primary_pred_sell",  # primary's P(sell)
    "primary_pred_hold",  # primary's P(hold)
    "primary_direction",  # 1 = buy, -1 = sell  (primary argmax in {buy, sell})
    "primary_conf",  # primary's directional confidence (max of buy/sell)
]


def build_meta_dataset(
    X: pd.DataFrame,
    y_primary_pred: np.ndarray,  # shape (n, 3): [P(sell), P(hold), P(buy)]
    y_truth: pd.Series,  # ground-truth triple-barrier labels {0,1,2}
    sample_weights: pd.Series | None = None,
    primary_min_conf: float = 0.5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Construct the meta-training dataset.

    Returns (X_meta, y_meta, w_meta) where:
        X_meta: original features + META_AUGMENT_FEATURES
        y_meta: 1 if primary directional pred matches truth, 0 otherwise
        w_meta: sample weights (|realised return| × uniqueness, from primary build)

    Rows where the primary predicted "hold" are DROPPED — meta-labelling
    operates on actionable signals only.
    """
    if len(X) != len(y_truth) or len(X) != y_primary_pred.shape[0]:
        raise ValueError("X / y_truth / primary preds length mismatch")

    sell_p = y_primary_pred[:, 0]
    hold_p = y_primary_pred[:, 1]
    buy_p = y_primary_pred[:, 2]

    # Primary's directional argmax constrained to {buy, sell}.  Pick the
    # larger of buy_p / sell_p; resulting "direction" only matters for rows
    # we keep (primary wasn't HOLD).
    direction = np.where(buy_p >= sell_p, 1, -1)
    primary_conf = np.where(direction == 1, buy_p, sell_p)

    # Filter: keep only bars where primary would actually trade, i.e. its
    # directional confidence ≥ primary_min_conf AND its argmax wasn't hold.
    keep_mask = (np.maximum(buy_p, sell_p) >= primary_min_conf) & (
        np.maximum(buy_p, sell_p) >= hold_p
    )
    if keep_mask.sum() == 0:
        raise ValueError("No meta-training rows: primary always preferred 'hold'.")

    # Meta target: 1 if directional prediction matches truth.
    # Truth label 2 = buy, 0 = sell (1 = hold doesn't matter — we filter).
    truth_arr = y_truth.to_numpy()
    correct = ((direction == 1) & (truth_arr == 2)) | ((direction == -1) & (truth_arr == 0))
    y_meta = pd.Series(correct.astype(int), index=X.index, name="meta_target")

    # Build augmented feature frame
    X_meta = X.copy()
    X_meta["primary_pred_sell"] = sell_p
    X_meta["primary_pred_hold"] = hold_p
    X_meta["primary_pred_buy"] = buy_p
    X_meta["primary_direction"] = direction
    X_meta["primary_conf"] = primary_conf

    # Apply the keep mask
    X_meta = X_meta.loc[keep_mask]
    y_meta = y_meta.loc[keep_mask]
    if sample_weights is not None:
        w_meta = sample_weights.loc[keep_mask]
    else:
        w_meta = pd.Series(1.0, index=y_meta.index, dtype=float)

    return X_meta, y_meta, w_meta


class MetaLabellingModel:
    """Binary LightGBM wrapping the meta-labelling logic."""

    DEFAULT_PARAMS = {
        "n_estimators": 200,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
        "objective": "binary",
        "verbose": -1,
    }

    def __init__(self, params: dict | None = None, threshold: float = 0.55):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._model: Optional[LGBMClassifier] = None
        self._feature_names: list[str] = []
        self.threshold = threshold
        self.metadata: Optional[MetaModelMetadata] = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None and self._feature_names != []

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        categorical_features: list[str] | None = None,
    ) -> dict:
        """Train binary classifier; return validation metrics dict."""
        self._feature_names = list(X.columns)
        self._model = LGBMClassifier(**self.params)

        # Class weight balance: meta target is often imbalanced (primary is
        # biased toward "hold" → meta sees mostly the hardest cases, which
        # tend to be majority-incorrect).  Compute class_weight automatically.
        pos = int(y.sum())
        neg = int(len(y) - pos)
        if pos == 0 or neg == 0:
            raise ValueError(f"Meta training set degenerate: pos={pos}, neg={neg}")
        cw = {0: pos / (pos + neg), 1: neg / (pos + neg)}

        fit_kw = {"sample_weight": sample_weight}
        if categorical_features:
            cat_idx = [list(X.columns).index(c) for c in categorical_features if c in X.columns]
            if cat_idx:
                fit_kw["categorical_feature"] = cat_idx
        if X_val is not None and y_val is not None:
            fit_kw["eval_set"] = [(X_val, y_val)]
            fit_kw["callbacks"] = []

        self._model.set_params(class_weight=cw)
        self._model.fit(X, y, **{k: v for k, v in fit_kw.items() if v is not None})

        # Evaluate on val set if provided, else on train (with caveat).
        Xe, ye = (X_val, y_val) if X_val is not None else (X, y)
        proba = self._model.predict_proba(Xe)[:, 1]
        preds = (proba >= self.threshold).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(ye, preds)),
            "f1": float(f1_score(ye, preds, zero_division=0)),
            "precision": float(precision_score(ye, preds, zero_division=0)),
            "recall": float(recall_score(ye, preds, zero_division=0)),
            "n_train": int(len(X)),
            "n_pos": pos,
        }
        self.metadata = MetaModelMetadata(
            accuracy=metrics["accuracy"],
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            n_train=metrics["n_train"],
            n_pos=metrics["n_pos"],
            feature_names=self._feature_names,
            threshold=self.threshold,
        )
        return metrics

    def predict_proba(self, X_meta: pd.DataFrame) -> np.ndarray:
        """Return P(primary correct) for each row.  Shape (n,)."""
        if not self.is_trained:
            return np.full(len(X_meta), 0.5, dtype=float)
        X_aligned = X_meta.fillna(0)[self._feature_names]
        return self._model.predict_proba(X_aligned)[:, 1]

    def predict(self, X_meta: pd.DataFrame) -> tuple[int, float]:
        """Single-row meta prediction: (gate_signal, meta_confidence).

        gate_signal = 1 if meta_confidence >= self.threshold (i.e. take
        the primary's signal), 0 otherwise.
        """
        if not self.is_trained:
            return 1, 0.5  # no meta available → don't block
        proba = float(self.predict_proba(X_meta.iloc[[-1]] if len(X_meta) > 1 else X_meta)[0])
        return (1 if proba >= self.threshold else 0), proba

    def save(self, path: Path) -> None:
        joblib.dump(
            {
                "model": self._model,
                "feature_names": self._feature_names,
                "threshold": self.threshold,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "MetaLabellingModel":
        bundle = joblib.load(path)
        m = cls(threshold=bundle.get("threshold", 0.55))
        m._model = bundle["model"]
        m._feature_names = bundle["feature_names"]
        m.metadata = bundle.get("metadata")
        return m
