"""Chandelier trailing stop logic.

Two-phase stop management for an open position:

Phase 1 — initial catastrophe stop, set at entry:
    long :  stop = entry − STOP_ATR_MULT_INITIAL × ATR_1h
    short:  stop = entry + STOP_ATR_MULT_INITIAL × ATR_1h

Phase 2 — Chandelier trail, activated once P&L ≥ TRAIL_ACTIVATE_R × initial_risk:
    long :  stop = max(prev_stop, peak_high − STOP_ATR_MULT_TRAIL × ATR_1h)
    short:  stop = min(prev_stop, peak_low  + STOP_ATR_MULT_TRAIL × ATR_1h)

Stops never move against the position (monotone in P&L direction).  This is
the standard Le Beau / Lukac Chandelier — wide enough to ride a real trend
through normal pullbacks, but locks in profit once the move has clearly
played out.

State (peak, current_stop, trail_active) lives in db.positions_state and is
re-hydrated on restart.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("futbot.stops")


@dataclass
class StopState:
    direction: str
    entry: float
    initial_risk: float  # |entry − initial_stop|, in price points
    peak: float  # high water mark (long) or low water mark (short)
    current_stop: float
    trail_active: bool


def initial_state(*, direction: str, entry: float, atr_1h: float, settings) -> StopState:
    """Compute the at-entry stop and peak state."""
    init_mult = float(settings.FUTBOT_STOP_ATR_MULT_INITIAL)
    risk = atr_1h * init_mult
    if direction == "buy":
        stop = entry - risk
        peak = entry
    else:
        stop = entry + risk
        peak = entry
    return StopState(
        direction=direction,
        entry=entry,
        initial_risk=risk,
        peak=peak,
        current_stop=stop,
        trail_active=False,
    )


def update(*, state: StopState, last_price: float, atr_1h: float, settings) -> StopState:
    """Roll the trailing stop forward.  Returns a NEW state (does not mutate)."""
    trail_mult = float(settings.FUTBOT_STOP_ATR_MULT_TRAIL)
    activate_r = float(settings.FUTBOT_TRAIL_ACTIVATE_R)

    new_peak = state.peak
    if state.direction == "buy":
        new_peak = max(state.peak, last_price)
        # P&L in R-multiples (positive when in profit)
        r_pnl = (last_price - state.entry) / state.initial_risk if state.initial_risk > 0 else 0
        new_active = state.trail_active or (r_pnl >= activate_r)
        if new_active:
            trail_stop = new_peak - trail_mult * atr_1h
            new_stop = max(state.current_stop, trail_stop)
        else:
            new_stop = state.current_stop
    else:
        new_peak = min(state.peak, last_price)
        r_pnl = (state.entry - last_price) / state.initial_risk if state.initial_risk > 0 else 0
        new_active = state.trail_active or (r_pnl >= activate_r)
        if new_active:
            trail_stop = new_peak + trail_mult * atr_1h
            new_stop = min(state.current_stop, trail_stop)
        else:
            new_stop = state.current_stop

    return StopState(
        direction=state.direction,
        entry=state.entry,
        initial_risk=state.initial_risk,
        peak=new_peak,
        current_stop=new_stop,
        trail_active=new_active,
    )


def is_stopped_out(*, state: StopState, bar_high: float, bar_low: float) -> bool:
    """Check whether the last bar's range touched the stop.  Use intra-bar
    extremes so we don't miss a flash-stop on a wick."""
    if state.direction == "buy":
        return bar_low <= state.current_stop
    return bar_high >= state.current_stop
