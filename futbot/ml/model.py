"""Inference wrapper for the binary-bag ML-gate bundles.

Loads a bundle written by ml/trainer.py, exposes:
    .predict_up_dn(features_row) → (p_up, p_dn)

Bundle structure (binary-bag-v1):
    bundle["up"]["models"]      — list of N LightGBM classifiers (bag)
    bundle["up"]["calibrator"]  — fitted IsotonicRegression (or None)
    bundle["dn"][...]           — same shape, dn target
    bundle["feature_names"]     — feature order

The gate policy (apply_gate) decides vote ∈ {-1, 0, +1}:
    * If P(up) ≥ THRESH_UP                       → vote candidate = +1
    * If P(dn) ≥ THRESH_DN                       → vote candidate = -1
    * If neither side meets its threshold        → no opinion (vote 0)
    * If both sides meet thresholds (rare)       → pick the higher prob

The gate VETOES the chain only when:
    * Inbound vote is +1 and the model says P(up) is low AND P(dn) ≥ VETO
    * Inbound vote is -1 and the model says P(dn) is low AND P(up) ≥ VETO

That asymmetry is deliberate — the model can only KILL trades, not start
them.  Initiation stays with the 4-layer technical chain.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("futbot.ml.model")


MODEL_DIR = Path("data/models")
# Probability thresholds (after isotonic calibration these are real probs).
THRESH_VOTE = 0.45  # below this → "model has no opinion"
THRESH_VETO = 0.55  # opposite side must clear this to veto the chain


@dataclass
class GatePrediction:
    p_up: float
    p_dn: float
    concept: str
    loaded: bool
    version: str = ""


class MLGateModel:
    """One bundle per concept, lazily loaded.  Thread-safe singleton cache."""

    _cache: dict[str, "MLGateModel | None"] = {}
    _lock = Lock()

    def __init__(self, concept: str, bundle: dict):
        self.concept = concept
        self.bundle = bundle
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.version = bundle.get("version", "unknown")

    @classmethod
    def for_concept(cls, concept: str) -> "MLGateModel | None":
        if not concept:
            return None
        with cls._lock:
            if concept in cls._cache:
                return cls._cache[concept]
            path = MODEL_DIR / f"futbot_{concept}.joblib"
            if not path.exists():
                cls._cache[concept] = None
                return None
            try:
                bundle = joblib.load(path)
                ver = bundle.get("version", "")
                if ver != "binary-bag-v1":
                    logger.warning(
                        f"{path} has version='{ver}' (expected 'binary-bag-v1') — "
                        f"retrain with `python -m futbot.scripts.train_ml {concept}`"
                    )
                    cls._cache[concept] = None
                    return None
                m = cls(concept, bundle)
                cls._cache[concept] = m
                up_m = bundle["up"]["metrics"]
                dn_m = bundle["dn"]["metrics"]
                logger.info(
                    f"loaded ML gate {concept} (v{ver}): "
                    f"up_auc={up_m.get('roc_auc')} up_ic={up_m.get('ic')} | "
                    f"dn_auc={dn_m.get('roc_auc')} dn_ic={dn_m.get('ic')}"
                )
                return m
            except Exception as e:
                logger.warning(f"failed to load {path}: {e}")
                cls._cache[concept] = None
                return None

    def _build_x(self, features_row: dict | pd.Series) -> np.ndarray:
        if isinstance(features_row, pd.Series):
            features_row = features_row.to_dict()
        missing = []
        x = []
        for f in self.feature_names:
            v = features_row.get(f)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                missing.append(f)
                v = 0.0
            x.append(float(v))
        if missing:
            logger.debug(f"ML gate inputs missing for {self.concept}: {missing[:5]}...")
        return np.array(x, dtype=float).reshape(1, -1)

    def _predict_side(self, side: str, X: np.ndarray) -> float:
        spec = self.bundle[side]
        bag = spec["models"]
        probs = np.array([m.predict_proba(X)[0, 1] for m in bag])
        raw = float(probs.mean())
        iso = spec.get("calibrator")
        if iso is not None:
            return float(iso.transform([raw])[0])
        return raw

    def predict_up_dn(self, features_row: dict | pd.Series) -> GatePrediction:
        X = self._build_x(features_row)
        try:
            p_up = self._predict_side("up", X)
            p_dn = self._predict_side("dn", X)
        except Exception as e:
            logger.warning(f"predict failed for {self.concept}: {e}")
            return GatePrediction(0.0, 0.0, self.concept, loaded=True, version=self.version)
        return GatePrediction(
            p_up=p_up,
            p_dn=p_dn,
            concept=self.concept,
            loaded=True,
            version=self.version,
        )


def apply_gate(
    *, inbound_vote: int, features_row: dict, concept: str | None
) -> tuple[int, str, dict]:
    """High-level decision function used by pipeline/ml_gate.py.

    Returns (vote_after, reason, detail) where detail is suitable for
    DB / Telegram logging.
    """
    if not concept:
        return inbound_vote, "no concept mapping — pass-through", {"loaded": False}

    mdl = MLGateModel.for_concept(concept)
    if mdl is None:
        return inbound_vote, f"no model for {concept} — pass-through", {"loaded": False}

    pred = mdl.predict_up_dn(features_row)
    detail = {
        "loaded": True,
        "concept": concept,
        "version": pred.version,
        "p_up": round(pred.p_up, 3),
        "p_dn": round(pred.p_dn, 3),
    }

    if inbound_vote == 0:
        return 0, "inbound vote is 0", detail

    # What does the model think the direction is?
    if pred.p_up >= THRESH_VOTE and pred.p_up >= pred.p_dn:
        model_vote, model_p = 1, pred.p_up
    elif pred.p_dn >= THRESH_VOTE and pred.p_dn > pred.p_up:
        model_vote, model_p = -1, pred.p_dn
    else:
        model_vote, model_p = 0, max(pred.p_up, pred.p_dn)
    detail["model_vote"] = model_vote
    detail["model_p"] = round(model_p, 3)

    # Veto only on STRONG opposite disagreement
    if model_vote == -inbound_vote and model_p >= THRESH_VETO:
        return (
            0,
            (f"ML veto: model {model_vote:+d} @ p={model_p:.2f} " f"vs chain {inbound_vote:+d}"),
            detail,
        )

    return (
        inbound_vote,
        (f"ML pass: p_up={pred.p_up:.2f} p_dn={pred.p_dn:.2f} " f"(chain {inbound_vote:+d})"),
        detail,
    )
