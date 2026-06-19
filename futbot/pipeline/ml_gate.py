"""Layer 5 — ML gate.

Loads a per-concept (oil/gas/gold/sber/lkoh/moex) LightGBM model trained
on daily Kaggle data via futbot.ml.trainer.  At inference, computes the
same feature vector on the most recent 1h bars (resampled to daily) and
asks the model for a 3-class probability.

Veto policy (see ml.model.apply_gate for details):
  * No model on disk OR no concept mapping → pass-through.
  * Model STRONGLY disagrees (opposite sign, p ≥ CONFIDENCE_VETO) → veto (0).
  * Otherwise → pass-through.

This INTENTIONALLY can only kill trades, never initiate them.  Initiation
is the chain's job; ML is just sanity check against a longer-horizon prior.
"""

import logging

import pandas as pd

from futbot.pipeline.base import LayerResult
from futbot.ml import features as feat
from futbot.ml import model as ml_model
from futbot.ml.datasets import CONTRACT_TO_CONCEPT

logger = logging.getLogger("futbot.layer.ml")


def _resample_1h_to_daily(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly bars to daily so the ML features (trained on daily
    Kaggle data) are comparable.  Tinkoff's hourly timestamps are bar-open,
    UTC.  We resample by UTC date.
    """
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()
    df = df_1h.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")
    daily = (
        df.resample("D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["close"])
        .reset_index()
    )
    return daily


def evaluate(
    *,
    prior_vote: int,
    settings,
    df_1h: pd.DataFrame | None = None,
    base_ticker: str | None = None,
    **_kwargs,
) -> LayerResult:
    """If df_1h + base_ticker are provided, use the trained model.
    Otherwise fall back to pass-through (phase-0 behaviour)."""
    if prior_vote == 0:
        return LayerResult(
            layer="ml_gate",
            vote=0,
            vetoes=True,
            reason="no inbound direction",
        )

    if df_1h is None or base_ticker is None:
        return LayerResult(
            layer="ml_gate",
            vote=prior_vote,
            vetoes=False,
            reason="ml_gate not wired (no data) — passing through",
            detail={"loaded": False},
        )

    concept = CONTRACT_TO_CONCEPT.get(base_ticker)
    if concept is None:
        return LayerResult(
            layer="ml_gate",
            vote=prior_vote,
            vetoes=False,
            reason=f"no concept mapping for base '{base_ticker}' — passing through",
            detail={"loaded": False},
        )

    daily = _resample_1h_to_daily(df_1h)
    if len(daily) < 70:
        return LayerResult(
            layer="ml_gate",
            vote=prior_vote,
            vetoes=False,
            reason=f"not enough daily bars ({len(daily)}) — passing through",
            detail={"loaded": False, "daily_bars": len(daily)},
        )

    feat_df = feat.build_features(daily, dropna=True)
    if feat_df.empty:
        return LayerResult(
            layer="ml_gate",
            vote=prior_vote,
            vetoes=False,
            reason="features all-NaN — passing through",
            detail={"loaded": False},
        )
    features_row = feat_df.iloc[-1].to_dict()

    vote_after, reason, detail = ml_model.apply_gate(
        inbound_vote=prior_vote,
        features_row=features_row,
        concept=concept,
    )
    vetoes = vote_after == 0 and prior_vote != 0
    return LayerResult(
        layer="ml_gate",
        vote=vote_after,
        vetoes=vetoes,
        reason=reason,
        detail=detail,
    )
