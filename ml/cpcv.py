"""Combinatorial Purged Cross-Validation (López de Prado AFML ch.12).

Why we need this:
  Walk-forward CV gives ONE estimate of out-of-sample performance per fold,
  using non-overlapping test windows.  CPCV partitions the data into N
  contiguous groups and then *combinatorially* picks k of them as test
  groups (the rest become train), rotating across all C(N, k) combinations.
  This produces O(N) more out-of-sample paths from the same data, giving:

  * Tighter confidence intervals on the chosen metric.
  * A "deflated Sharpe" estimate that corrects for selection bias when
    choosing among many candidate strategies (Bailey & Lopez de Prado 2014).
  * The ability to detect overfitting via the variance of test-fold metrics
    across combinations.

Implementation: snippet 12.4 from AFML, simplified for our use case.
  - Time-ordered samples partitioned into N groups.
  - Each combination uses k groups as test (concatenated, with purge+embargo
    around each test group's boundaries).
  - Train = remaining N-k groups, after purging samples whose label spans
    overlap any test group.

For our trader, we typically use N=6, k=2 → 15 paths.  Embargo equal to
the triple-barrier max_hold prevents label leakage across the boundary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CPCVSplit:
    """One train/test split produced by CPCV."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    test_groups: tuple[int, ...]


def cpcv_splits(
    n_samples: int,
    n_groups: int = 6,
    k_test_groups: int = 2,
    embargo: int = 10,
) -> Iterator[CPCVSplit]:
    """Yield CPCV train/test splits over a time-ordered sample range.

    Args:
        n_samples: total number of samples (time-ordered).
        n_groups: how many contiguous groups to partition the data into.
            Higher → more train/test combinations but smaller test sizes.
        k_test_groups: how many groups go into the test set per combination.
        embargo: number of samples on each side of every test-group boundary
            to drop from the training set.  Should equal the triple-barrier
            ``max_hold`` (label horizon).

    Yields:
        CPCVSplit objects.  Total count = C(n_groups, k_test_groups).
    """
    if n_groups < 2 or k_test_groups < 1 or k_test_groups >= n_groups:
        raise ValueError(f"Invalid CPCV params: n_groups={n_groups}, k_test_groups={k_test_groups}")

    # Group boundaries (slice indices) — last group absorbs any remainder
    bounds = np.linspace(0, n_samples, n_groups + 1, dtype=int)

    for combo in combinations(range(n_groups), k_test_groups):
        # Test = union of selected groups
        test_mask = np.zeros(n_samples, dtype=bool)
        for g in combo:
            test_mask[bounds[g] : bounds[g + 1]] = True

        # Train = complement, MINUS embargo zone around each test group.
        train_mask = ~test_mask
        for g in combo:
            lo, hi = bounds[g], bounds[g + 1]
            # Purge embargo samples on both sides of the test slice
            train_mask[max(0, lo - embargo) : lo] = False
            train_mask[hi : min(n_samples, hi + embargo)] = False

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield CPCVSplit(train_idx=train_idx, test_idx=test_idx, test_groups=combo)


def cpcv_summary(metrics_per_split: list[dict]) -> dict:
    """Reduce per-split metrics to a single robust summary.

    Returns mean, std, min/max for each numeric metric.  Std across splits is
    the key statistic — high variance signals an overfit (model performs
    well on some windows but poorly on others), low variance signals
    a robust estimator.
    """
    if not metrics_per_split:
        return {}
    df = pd.DataFrame(metrics_per_split).select_dtypes(include="number")
    out = {}
    for col in df.columns:
        out[f"{col}_mean"] = float(df[col].mean())
        out[f"{col}_std"] = float(df[col].std())
        out[f"{col}_min"] = float(df[col].min())
        out[f"{col}_max"] = float(df[col].max())
    out["n_splits"] = len(df)
    return out
