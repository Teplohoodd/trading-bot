"""Triple-barrier labelling (López de Prado, AFML Ch. 3).

For each bar t, look forward up to `horizon` bars and label according to
which barrier was hit first:
   +1   upper barrier (entry + up_mult × ATR) hit before lower barrier
   −1   lower barrier (entry − dn_mult × ATR) hit before upper barrier
    0   neither barrier hit within `horizon` bars (time barrier)

For a daily-horizon ML gate, defaults are:
   horizon  = 5 daily bars (~1 week)
   up_mult  = 1.5 × ATR_14
   dn_mult  = 1.5 × ATR_14
This produces a symmetric ±1.5 ATR move requirement, which on daily MOEX /
commodity series typically resolves with ~30 % +1, ~30 % −1, ~40 % 0 (a
balanced 3-class problem).

The ML gate uses just the SIGN — it asks "does the model think price will
move significantly UP or DOWN in the next week?".  The 0-class becomes
"no opinion", which the gate treats as pass-through.

`Note on bias`: triple-barrier with same-bar entry+exit assumed at CLOSE.
Real intraday execution will deviate; that's fine — the gate is a sanity
check, not a P&L predictor.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("futbot.ml.labels")


def triple_barrier(
    df: pd.DataFrame,
    *,
    horizon: int = 5,
    up_mult: float = 1.5,
    dn_mult: float = 1.5,
    atr_col: str = "atr_pct"
) -> pd.Series:
    """Compute triple-barrier labels.

    df must contain: time, open, high, low, close, plus the `atr_col`
    feature (default 'atr_pct' as produced by features.build_features).
    Returns an int Series aligned to df.index with values in {-1, 0, +1}.
    """
    if df.empty:
        return pd.Series([], dtype=int)
    out = np.zeros(len(df), dtype=int)
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atr_pct = df[atr_col].values  # ATR as % of price

    n = len(df)
    for i in range(n - 1):
        entry = closes[i]
        a = atr_pct[i] / 100.0 * entry  # ATR in price units
        if not np.isfinite(a) or a <= 0:
            out[i] = 0
            continue
        up_barrier = entry + up_mult * a
        dn_barrier = entry - dn_mult * a
        end = min(i + horizon + 1, n)
        label = 0
        for j in range(i + 1, end):
            if highs[j] >= up_barrier:
                label = +1
                break
            if lows[j] <= dn_barrier:
                label = -1
                break
        out[i] = label
    return pd.Series(out, index=df.index, name="tb_label")


def class_balance(labels: pd.Series) -> dict:
    """Quick health check on a label series."""
    n = len(labels)
    if n == 0:
        return {}
    vc = labels.value_counts().to_dict()
    return {
        "n": n,
        "pct_up": round(vc.get(1, 0) / n * 100, 1),
        "pct_dn": round(vc.get(-1, 0) / n * 100, 1),
        "pct_hold": round(vc.get(0, 0) / n * 100, 1),
    }


def binary_labels(tb: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Split a {-1, 0, +1} triple-barrier label into TWO binary targets:

        y_up = (tb == +1)   "did price hit the +1.5 ATR upper barrier first?"
        y_dn = (tb == -1)   "did price hit the −1.5 ATR lower barrier first?"

    Why binary instead of 3-class:
      * The 3-class model has to split probability mass across 3 outcomes
        whose boundaries are murky (a near-1 ATR move can end up as +1 or
        as 0 depending on intra-bar wicks). Probability resolution is poor
        in the middle.
      * Two binary models ask cleanly: "given this state, will we get a
        decisive UP move?" / "...DECISIVE DOWN?".  Each model's
        probabilities calibrate sharply because the negative class is
        well-defined (everything else, including both holds and opposite-
        direction breakouts).
      * The bot's gate naturally consumes P(up) and P(dn) — they map
        directly to the +1/−1 votes.  No need for argmax tie-breaking.

    Returns two int Series (0/1).
    """
    y_up = (tb == 1).astype(int)
    y_dn = (tb == -1).astype(int)
    y_up.name = "y_up"
    y_dn.name = "y_dn"
    return y_up, y_dn


def binary_balance(y: pd.Series) -> dict:
    """Positive-class fraction + n for a binary target."""
    if len(y) == 0:
        return {"n": 0, "pct_pos": 0.0}
    return {"n": len(y), "pct_pos": round(float(y.mean()) * 100, 1)}
