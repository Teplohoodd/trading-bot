"""Cointegration helpers — fit β, α and run ADF on residuals."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("futbot.pairs.coint")


@dataclass
class FitResult:
    pair: str
    base_y: str
    base_x: str
    beta: float
    alpha: float
    spread_mean: float
    spread_std: float
    adf_p: float
    n_bars: int


def fit_pair(prices_y: pd.Series, prices_x: pd.Series, *, pair_name: str) -> FitResult | None:
    """Engle-Granger 2-step: OLS for β/α, then ADF on residual.

    Returns None if either series has too few aligned bars or ADF fails.
    """
    # Align on common index
    common = pd.concat([prices_y.rename("y"), prices_x.rename("x")], axis=1, join="inner").dropna()
    if len(common) < 100:
        logger.warning(f"{pair_name}: only {len(common)} aligned bars — skip")
        return None
    y = common["y"].values
    x = common["x"].values

    beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
    alpha = y.mean() - beta * x.mean()
    resid = y - (beta * x + alpha)

    try:
        from statsmodels.tsa.stattools import adfuller

        adf = adfuller(resid, maxlag=5, autolag=None)
        adf_p = float(adf[1])
    except Exception as e:
        logger.warning(f"{pair_name}: ADF failed ({e})")
        adf_p = 1.0

    base_y, base_x = pair_name.split("-")
    return FitResult(
        pair=pair_name,
        base_y=base_y,
        base_x=base_x,
        beta=float(beta),
        alpha=float(alpha),
        spread_mean=float(resid.mean()),
        spread_std=float(resid.std()),
        adf_p=adf_p,
        n_bars=len(common),
    )


def current_zscore(
    *, y_price: float, x_price: float, beta: float, spread_mean: float, spread_std: float
) -> float:
    """Z-score of the current spread given the cached fit."""
    spread = y_price - beta * x_price
    if spread_std <= 0:
        return 0.0
    return (spread - spread_mean) / spread_std
