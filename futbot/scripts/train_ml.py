"""Train per-concept ML-gate models (binary-bag-v1) from Kaggle data.

Usage:
    python -m futbot.scripts.train_ml                # all concepts
    python -m futbot.scripts.train_ml oil gold       # specific ones

Output: data/models/futbot_<concept>.joblib  per concept that succeeded.
Reports per-side ROC AUC / Brier / F1 / IC (Spearman corr with realised
forward return) on a held-out test tail.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from futbot.ml.datasets import CONCEPTS
from futbot.ml.trainer import train_concept


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("futbot.train")

    args = sys.argv[1:]
    targets = list(args) if args else list(CONCEPTS.keys())
    unknown = [t for t in targets if t not in CONCEPTS]
    if unknown:
        logger.error(f"unknown concepts: {unknown}.  Known: {list(CONCEPTS)}")
        sys.exit(2)

    logger.info(f"Training {len(targets)} concept(s): {targets}")
    results: dict[str, dict] = {}
    for c in targets:
        logger.info(f"--- {c} ---")
        try:
            results[c] = train_concept(c)
        except Exception as e:
            logger.exception(f"{c}: training failed: {e}")
            results[c] = {"error": str(e)}

    # Summary table
    print()
    print("=" * 120)
    print(
        f"{'concept':<8} {'n_tr':>5} {'n_te':>5} | "
        f"{'UP auc':>7} {'brier':>6} {'f1':>5} {'IC':>6} {'thr':>5} | "
        f"{'DN auc':>7} {'brier':>6} {'f1':>5} {'IC':>6} {'thr':>5}"
    )
    print("-" * 120)
    for c, m in results.items():
        if "error" in m:
            print(f"{c:<8} ERROR: {m['error'][:80]}")
            continue
        u = m["up_metrics"]
        d = m["dn_metrics"]

        def _fmt(v, fmt):
            return fmt.format(v) if v is not None else "  —  "

        print(
            f"{c:<8} {m['n_train']:>5} {m['n_test']:>5} | "
            f"{_fmt(u.get('roc_auc'), '{:>7.3f}')} "
            f"{_fmt(u.get('brier'), '{:>6.3f}')} "
            f"{_fmt(u.get('f1_at_best'), '{:>5.3f}')} "
            f"{_fmt(u.get('ic'), '{:>+6.3f}')} "
            f"{_fmt(u.get('best_threshold'), '{:>5.2f}')} | "
            f"{_fmt(d.get('roc_auc'), '{:>7.3f}')} "
            f"{_fmt(d.get('brier'), '{:>6.3f}')} "
            f"{_fmt(d.get('f1_at_best'), '{:>5.3f}')} "
            f"{_fmt(d.get('ic'), '{:>+6.3f}')} "
            f"{_fmt(d.get('best_threshold'), '{:>5.2f}')}"
        )
    print()
    print("Reading the table:")
    print("  * ROC AUC > 0.55 = useful signal; 0.50 = coin flip; 0.60+ = strong")
    print("  * IC = Spearman corr between predicted prob and realised forward")
    print("    log-return.  Sign matters: UP IC should be positive, DN IC negative.")
    print("    |IC| > 0.05 is meaningful on noisy daily data.")
    print("  * Brier = mean squared prob error.  Lower = better calibrated.")


if __name__ == "__main__":
    main()
