"""Apples-to-apples comparison: legacy 3-class model vs new binary-bag-v1.

Both models trained on IDENTICAL feature matrix + walk-forward splits, so
the metrics are directly comparable.  This is the honest version of "did
the new model improve things?".

Computed for each concept:

  Directional accuracy  — on rows where the triple-barrier label is ±1
                          (i.e. there WAS a decisive move), did the model
                          predict the correct sign?  Excludes the "hold"
                          class so both models have a fair shot.
  IC (Spearman)         — corr between the model's "go-long" score and
                          realised forward return.  For 3-class:
                          score = P(+1) - P(-1).  For binary-bag:
                          score = P(up) - P(dn).
  Coverage @ p≥0.55     — what fraction of test rows produce a
                          high-confidence signal?  This is what determines
                          whether the gate actually fires in production.
  Prec @ p≥0.55         — when the gate WOULD fire, how often is the
                          direction right?  This is the only metric that
                          matters for a veto gate.

Usage:
    python -m futbot.scripts.compare_ml          # all concepts
    python -m futbot.scripts.compare_ml oil sber
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futbot.ml import datasets as ds
from futbot.ml import features as feat
from futbot.ml import labels as lab
from futbot.ml import trainer as tr  # for the WF-folds + binary-bag

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)


HIGH_CONF = 0.55


def _legacy_3class_oof(X: np.ndarray, y3: np.ndarray, folds) -> np.ndarray:
    """Replicate the legacy single-LGBM 3-class model's OOF predictions
    using the same WF folds the new model uses.  Returns an (n, 3) array
    aligned to rows; rows not in any test fold are NaN."""
    n = len(y3)
    classes = np.array([-1, 0, 1])
    oof = np.full((n, 3), np.nan)
    for tr_idx, te_idx in folds:
        m = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=6,
            min_child_samples=30,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )
        m.fit(X[tr_idx], y3[tr_idx])
        proba = m.predict_proba(X[te_idx])
        # Align columns to the global [-1, 0, +1] order
        col_map = {c: i for i, c in enumerate(m.classes_)}
        for ci, c in enumerate(classes):
            if c in col_map:
                oof[te_idx, ci] = proba[:, col_map[c]]
            else:
                oof[te_idx, ci] = 0.0
    return oof


def _binary_bag_oof(X: np.ndarray, y_up: np.ndarray, y_dn: np.ndarray, folds):
    """Run the same WF folds as the new trainer would — produces OOF
    predictions for both UP and DN binary models (no calibrator; calibration
    is fit on these very preds in production, so we don't double-fit)."""
    n = len(y_up)
    p_up = np.full(n, np.nan)
    p_dn = np.full(n, np.nan)
    for tr_idx, te_idx in folds:
        bag_up = tr._train_bag(X[tr_idx], y_up[tr_idx])
        bag_dn = tr._train_bag(X[tr_idx], y_dn[tr_idx])
        p_up[te_idx] = tr._predict_bag(bag_up, X[te_idx])
        p_dn[te_idx] = tr._predict_bag(bag_dn, X[te_idx])
    return p_up, p_dn


def _metrics(
    *, score: np.ndarray, y3: np.ndarray, fwd: np.ndarray, y_up: np.ndarray, y_dn: np.ndarray
) -> dict:
    """`score` is a real number where + means model thinks long.
    Returns directional accuracy, IC, coverage / precision at the
    HIGH_CONF threshold (mapped from score via score>=0.55→long etc)."""
    mask_dir = y3 != 0
    correct = ((score > 0) & (y3 == 1)) | ((score < 0) & (y3 == -1))
    dir_acc = float(correct[mask_dir].mean()) if mask_dir.any() else float("nan")
    mask_finite = ~np.isnan(fwd)
    if mask_finite.sum() >= 30:
        ic, _ = spearmanr(score[mask_finite], fwd[mask_finite])
    else:
        ic = float("nan")
    # Coverage / precision at high-conf.  We treat score>=HIGH_CONF as "go long",
    # score<=-(1-HIGH_CONF) i.e. score≤-0.45 as "go short"... actually for
    # comparability we use |score| ≥ HIGH_CONF where score is symmetric around 0.
    # For 3-class: score = P(+1)-P(-1) which lives in [-1, 1].  HIGH_CONF=0.55
    # means "model is at least 55% more confident on one side than the other".
    long_hi = score >= HIGH_CONF
    short_hi = score <= -HIGH_CONF
    coverage = float((long_hi | short_hi).mean())
    precision = None
    if (long_hi | short_hi).any():
        hi_correct = (long_hi & (y3 == 1)) | (short_hi & (y3 == -1))
        # Denominator = high-conf decisions that resolved (excl. hold)
        denom = long_hi | short_hi
        denom_resolved = denom & mask_dir
        precision = float(hi_correct[denom_resolved].mean()) if denom_resolved.any() else None
    return {
        "dir_acc": round(dir_acc, 4),
        "ic": round(float(ic), 4) if not np.isnan(ic) else None,
        "coverage_hi": round(coverage, 4),
        "precision_hi": round(precision, 4) if precision is not None else None,
    }


def compare_concept(concept: str) -> tuple[dict, dict]:
    df = ds.load_concept(concept)
    df = feat.build_features(df, dropna=True)
    tb = lab.triple_barrier(
        df, horizon=tr.DEFAULT_HORIZON, up_mult=tr.DEFAULT_UP_MULT, dn_mult=tr.DEFAULT_DN_MULT
    )
    # Drop the final horizon rows (label peeks future)
    df = df.iloc[: -tr.DEFAULT_HORIZON].reset_index(drop=True)
    tb = tb.iloc[: -tr.DEFAULT_HORIZON].reset_index(drop=True)
    y_up, y_dn = lab.binary_labels(tb)
    fwd = np.log(df["close"].shift(-tr.DEFAULT_HORIZON) / df["close"]).values
    X = df[feat.FEATURE_NAMES].values
    n = len(df)

    n_test = int(n * tr.TEST_FRACTION)
    n_train = n - n_test
    X_tr, X_te = X[:n_train], X[n_train:]
    fwd_tr, fwd_te = fwd[:n_train], fwd[n_train:]
    y3_tr, y3_te = tb.values[:n_train], tb.values[n_train:]
    y_up_tr, y_up_te = y_up.values[:n_train], y_up.values[n_train:]
    y_dn_tr, y_dn_te = y_dn.values[:n_train], y_dn.values[n_train:]

    folds = tr._wf_folds(n_train, tr.CV_FOLDS, embargo=tr.DEFAULT_HORIZON)

    # ── LEGACY 3-class: OOF on train + final-train on full train, score on test
    legacy_oof = _legacy_3class_oof(X_tr, y3_tr, folds)
    # Final retrain
    final_legacy = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        min_child_samples=30,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    final_legacy.fit(X_tr, y3_tr)
    classes = np.array([-1, 0, 1])
    proba_test = final_legacy.predict_proba(X_te)
    col_map = {c: i for i, c in enumerate(final_legacy.classes_)}
    legacy_score_test = np.array(
        [
            proba_test[:, col_map.get(1, -1)] if 1 in col_map else np.zeros(len(X_te)),
            proba_test[:, col_map.get(-1, -1)] if -1 in col_map else np.zeros(len(X_te)),
        ]
    )
    legacy_score = legacy_score_test[0] - legacy_score_test[1]  # P(+1)-P(-1)

    # ── BINARY-BAG: OOF on train, final retrain bag on full train
    p_up_oof, p_dn_oof = _binary_bag_oof(X_tr, y_up_tr, y_dn_tr, folds)
    bag_up = tr._train_bag(X_tr, y_up_tr)
    bag_dn = tr._train_bag(X_tr, y_dn_tr)
    p_up_te = tr._predict_bag(bag_up, X_te)
    p_dn_te = tr._predict_bag(bag_dn, X_te)
    binary_score = p_up_te - p_dn_te

    legacy_m = _metrics(
        score=legacy_score,
        y3=y3_te,
        fwd=fwd_te,
        y_up=y_up_te,
        y_dn=y_dn_te,
    )
    binary_m = _metrics(
        score=binary_score,
        y3=y3_te,
        fwd=fwd_te,
        y_up=y_up_te,
        y_dn=y_dn_te,
    )
    return legacy_m, binary_m


def main():
    targets = sys.argv[1:] or list(ds.CONCEPTS.keys())
    results = {}
    for c in targets:
        try:
            print(f"\nTraining + scoring {c} (both architectures)...")
            results[c] = compare_concept(c)
        except Exception as e:
            print(f"  {c}: failed ({e})")

    print()
    print("=" * 90)
    print(
        f"{'concept':<8} | " f"{'dir_acc':>14} | {'IC':>14} | " f"{'cov_hi':>14} | {'prec_hi':>14}"
    )
    print(
        f"{'':<8} | {'legacy / new':>14} | {'legacy / new':>14} | "
        f"{'legacy / new':>14} | {'legacy / new':>14}"
    )
    print("-" * 90)
    for c, (lg, bn) in results.items():

        def _fmt(a, b, w):
            sa = (
                f"{a:.3f}"
                if a is not None and not (isinstance(a, float) and np.isnan(a))
                else "  —  "
            )
            sb = (
                f"{b:.3f}"
                if b is not None and not (isinstance(b, float) and np.isnan(b))
                else "  —  "
            )
            return f"{sa}/{sb}".rjust(w)

        print(
            f"{c:<8} | "
            f"{_fmt(lg['dir_acc'], bn['dir_acc'], 14)} | "
            f"{_fmt(lg['ic'], bn['ic'], 14)} | "
            f"{_fmt(lg['coverage_hi'], bn['coverage_hi'], 14)} | "
            f"{_fmt(lg['precision_hi'], bn['precision_hi'], 14)}"
        )
    print()
    print("How to read:")
    print("  dir_acc — % correct sign on rows that resolved ±1 (excl. hold).")
    print("            Random baseline ≈ 0.50 (binary problem after dropping hold).")
    print("  IC      — Spearman corr (score, realised fwd return).  >0 is good.")
    print("  cov_hi  — fraction of test rows where |score| ≥ 0.55 (the gate fires).")
    print("  prec_hi — directional accuracy CONDITIONAL on the gate firing.")
    print("            This is what matters for a veto gate.  Random = 0.50.")


if __name__ == "__main__":
    main()
