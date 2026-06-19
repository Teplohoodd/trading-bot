"""Edge engine — three orthogonal signals retail bots rarely combine.

The current scalp module uses microstructure (book imbalance, trade flow
imbalance) plus textbook indicators (RSI, EMA, MACD).  This works but
showed weak edge in live (NET P&L bouncing ±2% per day).  The three
mechanics below are documented in academic / industry literature and
specifically exploit T-Invest API features that ARE NOT used by the
most-starred GitHub bots (Freqtrade / Jesse / Hummingbot — all crypto-
focused, none expose Open Interest, all use standard TA only):

──────────────────────────────────────────────────────────────────────────
1. OPEN INTEREST DELTA — futures-specific information
──────────────────────────────────────────────────────────────────────────
Every FORTS trade is BETWEEN two parties — every contract has both a
buyer and seller side.  But OI (open interest) tells us WHICH side is
opening new positions:

    Price ↑ AND OI ↑  →  NEW LONGS opening (conviction buy)
    Price ↑ AND OI ↓  →  SHORTS covering   (squeeze, weaker)
    Price ↓ AND OI ↑  →  NEW SHORTS opening (conviction sell)
    Price ↓ AND OI ↓  →  LONGS covering    (capitulation, weaker)

Retail spot bots can't use this — there's no OI in spot markets.  Our
scalp module misses it because we never set `with_open_interest=True`
on the trades subscription.  Adding it gives us a "smart-money proxy"
for each trade as it prints.

──────────────────────────────────────────────────────────────────────────
2. CVD (CUMULATIVE VOLUME DELTA) DIVERGENCE — exhaustion signal
──────────────────────────────────────────────────────────────────────────
CVD = cumulative sum of (buy-aggressor-volume - sell-aggressor-volume)
since session open.  Divergence between CVD and price predicts trend
exhaustion (Bookmap/Hyblock community).

    Bullish divergence: price prints LOWER low, CVD prints HIGHER low
                        → sellers running out of fuel, mean-revert long
    Bearish divergence: price prints HIGHER high, CVD prints LOWER high
                        → buyers running out of fuel, mean-revert short

Our `tfi` feature is INSTANTANEOUS imbalance — CVD is its INTEGRAL.
The integral catches accumulating pressure that single-window TFI misses.

──────────────────────────────────────────────────────────────────────────
3. LEAD-LAG CROSS-INSTRUMENT — front-running the slow contract
──────────────────────────────────────────────────────────────────────────
Empirically on MOEX FORTS:
    * BR (Brent) is a global price → leads RU energy futures by seconds
    * Si (USD/RUB) moves → leads RU exporters' futures
    * GAZP share moves → can lead GZ futures by 5-30 seconds
        (and vice versa during fast moves)

When the LEADING instrument fires a clean directional signal AND the
LAGGING instrument hasn't moved yet, we have a few seconds of edge.
Most retail bots score each instrument independently — this is left
on the table.

Verified empirically by `scripts/edge_backtest.py` which computes the
lagged correlation matrix on historical candles.

──────────────────────────────────────────────────────────────────────────
Integration into the scalp pipeline
──────────────────────────────────────────────────────────────────────────
The composite signal score in scalp/strategy.py becomes:

    base_score = 0.40 × book + 0.25 × tfi + 0.15 × ind + 0.10 × vwap
                 + 0.10 × cvd_div      ← NEW (Mechanism 2)

    veto: refuse entry if OI delta over last 20 trades disagrees with
          direction (Mechanism 1)

    boost: +0.10 to score if leading-instrument also signalled same
           direction in last 60s (Mechanism 3)

Together they form a "Smart Money Footprint" filter: only trade when
flow + integral + cross-asset all agree, and where OI tells us we're
on the same side as the positioning institutions.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger("futbot.scalp.edge")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Open Interest delta
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OIState:
    """Rolling state for OI delta tracking, per instrument."""

    # We store (timestamp, oi, was_buyer_aggressor, qty) for the last N trades
    history: deque = field(default_factory=lambda: deque(maxlen=200))

    def update(self, *, ts: float, oi: int, aggressor_dir: int, qty: int):
        self.history.append(
            {
                "ts": ts,
                "oi": int(oi),
                "dir": int(aggressor_dir),
                "qty": int(qty),
            }
        )

    def conviction(self, window_n: int = 20) -> dict:
        """Read the most recent `window_n` trades and classify the flow.

        Returns dict:
            new_long_strength   ∈ [0, 1]   how much "price↑ AND OI↑" weight
            new_short_strength  ∈ [0, 1]   how much "price↓ AND OI↑" weight
            covering            ∈ [0, 1]   weight of OI-decreasing trades
            sample              int        # trades inspected

        These are SOFT scores summing approximately to 1.  The signal is
        the asymmetry between new_long_strength and new_short_strength.
        """
        if len(self.history) < 5:
            return {"new_long": 0.0, "new_short": 0.0, "covering": 0.0, "sample": 0}
        recent = list(self.history)[-window_n:]
        # Need at least 2 to see a delta
        if len(recent) < 2:
            return {"new_long": 0.0, "new_short": 0.0, "covering": 0.0, "sample": 0}

        new_long = 0
        new_short = 0
        covering = 0
        total = 0
        for i in range(1, len(recent)):
            cur = recent[i]
            prev_oi = recent[i - 1]["oi"]
            d_oi = cur["oi"] - prev_oi
            qty = cur["qty"]
            total += qty
            if d_oi > 0 and cur["dir"] > 0:  # OI up + buy-aggressor
                new_long += qty
            elif d_oi > 0 and cur["dir"] < 0:  # OI up + sell-aggressor
                new_short += qty
            elif d_oi < 0:  # OI down = covering
                covering += qty
            # else: no OI change, skip

        if total <= 0:
            return {"new_long": 0.0, "new_short": 0.0, "covering": 0.0, "sample": 0}
        return {
            "new_long": round(new_long / total, 3),
            "new_short": round(new_short / total, 3),
            "covering": round(covering / total, 3),
            "sample": len(recent),
        }


def oi_veto(
    *, oi_state: OIState, direction: str, min_sample: int = 8, min_conviction: float = 0.30
) -> tuple[bool, str]:
    """Should we VETO this proposed entry based on OI flow?

    Block when the most recent flow says the opposite side is positioning:
      * Long entry  but new_short_strength > min_conviction + 0.10 → veto
      * Short entry but new_long_strength  > min_conviction + 0.10 → veto

    Returns (veto, reason).  When we don't have enough data we DO NOT
    veto — the upstream chain still rules.
    """
    c = oi_state.conviction()
    if c["sample"] < min_sample:
        return False, f"OI sample {c['sample']} < {min_sample} — pass-through"
    if direction == "buy" and c["new_short"] > min_conviction + 0.10:
        return True, (
            f"OI veto: new_short flow {c['new_short']:.2f} > "
            f"new_long {c['new_long']:.2f} — institutions opening shorts"
        )
    if direction == "sell" and c["new_long"] > min_conviction + 0.10:
        return True, (
            f"OI veto: new_long flow {c['new_long']:.2f} > "
            f"new_short {c['new_short']:.2f} — institutions opening longs"
        )
    return False, f"OI flow OK (long={c['new_long']:.2f} short={c['new_short']:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
# 2. CVD divergence
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CVDState:
    """Cumulative Volume Delta running state, per instrument.

    Updated on every trade event: signed_qty = qty × (+1 if buyer-aggressor,
    −1 if seller-aggressor).  CVD is the running sum reset each UTC day.

    For divergence detection we also store a low-res series of
    (timestamp, price, cvd) sampled at most once per second so we can
    look for swing-high/swing-low pivots.
    """

    cvd: float = 0.0
    last_reset_day: str = ""
    samples: deque = field(default_factory=lambda: deque(maxlen=1200))  # ~20 min @ 1Hz
    _last_sample_ts: float = 0.0

    def update(self, *, ts: float, price: float, qty: int, aggressor_dir: int):
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).date().isoformat()
        if today != self.last_reset_day:
            self.cvd = 0.0
            self.last_reset_day = today
            self.samples.clear()
            self._last_sample_ts = 0.0
        self.cvd += aggressor_dir * qty
        # Sample at most once per second
        if ts - self._last_sample_ts >= 1.0:
            self.samples.append({"ts": ts, "price": price, "cvd": self.cvd})
            self._last_sample_ts = ts

    def divergence(self, lookback_sec: float = 600.0) -> dict:
        """Detect price–CVD divergence in the last `lookback_sec`.

        Method: find lowest LOW (and highest HIGH) of price in window.
        Compare the CVD AT THAT BAR vs current CVD.

          Bullish divergence (long signal):
            price made a low at t_lo, current price ≥ that low,
            BUT CVD at t_lo < current CVD  → buyers refusing to push lower
            even though price dipped → mean-revert UP

          Bearish divergence (short signal):
            mirror.

        Returns dict {direction: "buy"/"sell"/None, strength: 0..1}.
        Empty/no-signal returns direction=None.
        """
        if len(self.samples) < 30:
            return {"direction": None, "strength": 0.0, "reason": "not enough samples"}
        now_ts = self.samples[-1]["ts"]
        cur_price = self.samples[-1]["price"]
        cur_cvd = self.samples[-1]["cvd"]
        cutoff = now_ts - lookback_sec
        window = [s for s in self.samples if s["ts"] >= cutoff]
        if len(window) < 20:
            return {"direction": None, "strength": 0.0, "reason": "thin window"}

        # Pivot points: lowest and highest by PRICE within the window
        lo = min(window, key=lambda s: s["price"])
        hi = max(window, key=lambda s: s["price"])

        # Need meaningful price range to call divergence
        px_range = hi["price"] - lo["price"]
        if px_range <= 0:
            return {"direction": None, "strength": 0.0, "reason": "flat price"}
        # Normalise CVD delta by total trade volume in the window
        cvd_range = max(abs(cur_cvd - lo["cvd"]), abs(cur_cvd - hi["cvd"]), 1.0)

        # Bullish: cur_price ≥ low * (1 + 0.0005) — sat near or above low,
        # AND cur_cvd > lo["cvd"] (CVD made HIGHER low than at price low)
        # Strength = how close cur_price is to lo AND how much CVD rose
        if cur_price <= lo["price"] * 1.002:  # within 0.2% of the recent low
            d_cvd = cur_cvd - lo["cvd"]
            if d_cvd > 0:
                strength = min(1.0, d_cvd / cvd_range)
                if strength > 0.20:
                    return {
                        "direction": "buy",
                        "strength": round(strength, 3),
                        "pivot_price": lo["price"],
                        "pivot_cvd": lo["cvd"],
                        "cur_price": cur_price,
                        "cur_cvd": cur_cvd,
                    }

        # Bearish mirror
        if cur_price >= hi["price"] * 0.998:
            d_cvd = cur_cvd - hi["cvd"]
            if d_cvd < 0:
                strength = min(1.0, abs(d_cvd) / cvd_range)
                if strength > 0.20:
                    return {
                        "direction": "sell",
                        "strength": round(strength, 3),
                        "pivot_price": hi["price"],
                        "pivot_cvd": hi["cvd"],
                        "cur_price": cur_price,
                        "cur_cvd": cur_cvd,
                    }

        return {"direction": None, "strength": 0.0, "reason": "no divergence"}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lead-lag cross-instrument signal
# ─────────────────────────────────────────────────────────────────────────────
# This one lives at the orchestrator level — it can't be computed
# from a single instrument's state.  We provide a helper that, given
# the snapshot of one instrument's last-N-second return, returns a
# boolean "agrees with proposed direction" answer.


def leading_instrument_agrees(
    *, leader_returns: deque, direction: str, window_sec: float = 60.0, min_move_pct: float = 0.03
) -> dict:
    """`leader_returns` is a deque of {ts, ret_since_start} where ret is the
    cumulative log-return of the LEADER from a reference point.  We look
    at the change over the last `window_sec` and check if it exceeds
    `min_move_pct` (in %) in our direction.

    Returns {agree: bool, leader_move_pct: float, sample: int}.
    """
    if not leader_returns:
        return {"agree": False, "leader_move_pct": 0.0, "sample": 0}
    now_ts = leader_returns[-1]["ts"]
    cutoff = now_ts - window_sec
    window = [r for r in leader_returns if r["ts"] >= cutoff]
    if len(window) < 5:
        return {"agree": False, "leader_move_pct": 0.0, "sample": len(window)}
    move = (window[-1]["ret_since_start"] - window[0]["ret_since_start"]) * 100
    if direction == "buy" and move >= min_move_pct:
        return {"agree": True, "leader_move_pct": round(move, 4), "sample": len(window)}
    if direction == "sell" and move <= -min_move_pct:
        return {"agree": True, "leader_move_pct": round(move, 4), "sample": len(window)}
    return {"agree": False, "leader_move_pct": round(move, 4), "sample": len(window)}
