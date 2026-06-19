"""Microstructure feature computations from streaming state.

These are the actual alpha sources for short-horizon (seconds-minutes)
prediction.  Unlike daily-bar indicators, microstructure signals exist
ONLY in book + trade flow and have direct intuition:

  * Book imbalance: more bid volume than ask → buy pressure → near-term up bias.
  * Microprice: volume-weighted mid; differs from naïve mid when one side
    is thicker, and that difference predicts where the trade will print next.
  * Trade flow imbalance (TFI): in the last N seconds, was the dollar
    volume buy-initiated or sell-initiated?  Strong directional flow
    persists for a few seconds-minutes ("autocorrelation in order flow"
    is a classic empirical finding).
  * Trade intensity: how many trades / second now vs baseline.  Activity
    spikes precede directional moves.

All functions are pure — no I/O.  They take an `InstrumentState` snapshot
and return floats / dicts.  Run from the scalp main loop on every signal
evaluation.
"""

import time
from dataclasses import dataclass


def book_imbalance(state, levels: int = 5) -> float:
    """(bid_vol − ask_vol) / total.  Range [-1, +1].  Positive = buy side
    thicker.  Uses only top `levels` levels — deep book is often stale."""
    if not state.bids or not state.asks:
        return 0.0
    bid_v = sum(b.volume for b in state.bids[:levels])
    ask_v = sum(a.volume for a in state.asks[:levels])
    total = bid_v + ask_v
    if total <= 0:
        return 0.0
    return (bid_v - ask_v) / total


def microprice(state) -> float:
    """Volume-weighted mid.  When ask is thin, microprice > naive mid,
    indicating the next trade is likely to print at ask side.  Returns
    0.0 if the book is empty."""
    if not state.bids or not state.asks:
        return 0.0
    best_bid = state.bids[0]
    best_ask = state.asks[0]
    if best_bid.volume + best_ask.volume == 0:
        return (best_bid.price + best_ask.price) / 2
    # Microprice formula: bid_p × ask_vol + ask_p × bid_vol  /  total_vol
    return (best_bid.price * best_ask.volume + best_ask.price * best_bid.volume) / (
        best_bid.volume + best_ask.volume
    )


def spread_ticks(state, min_price_increment: float) -> int:
    """Spread in ticks (min_price_increment units).  Wide spread = expensive
    to cross = bad time to enter."""
    if not state.bids or not state.asks or min_price_increment <= 0:
        return 999
    spread = state.asks[0].price - state.bids[0].price
    return int(round(spread / min_price_increment))


def trade_flow_imbalance(state, window_sec: float) -> tuple[float, int]:
    """(buy_qty − sell_qty) / total_qty over the last `window_sec` seconds.
    Returns (tfi, n_trades).  n_trades is exposed so the caller can require
    a minimum sample size before trusting the value."""
    if not state.recent_trades:
        return 0.0, 0
    now = state.recent_trades[-1]["ts"]
    cutoff = now - window_sec
    bv = sv = 0
    n = 0
    for t in reversed(state.recent_trades):
        if t["ts"] < cutoff:
            break
        if t["dir"] > 0:
            bv += t["qty"]
        else:
            sv += t["qty"]
        n += 1
    total = bv + sv
    if total <= 0:
        return 0.0, n
    return (bv - sv) / total, n


def trade_intensity(state, window_sec: float = 30.0) -> float:
    """Trades / second in the last window.  Useful as a regime detector —
    when intensity spikes 3-5× the local average, a move is coming."""
    if not state.recent_trades:
        return 0.0
    now = state.recent_trades[-1]["ts"]
    cutoff = now - window_sec
    n = sum(1 for t in reversed(state.recent_trades) if t["ts"] >= cutoff)
    return n / window_sec


@dataclass
class MicroSnapshot:
    """All microstructure features in one snapshot for logging / debug."""

    book_imb: float
    micro_px: float
    spread_t: int
    tfi: float
    tfi_n: int
    intensity: float
    mid: float


def snapshot(state, min_price_increment: float, tfi_window_sec: float) -> MicroSnapshot:
    bid_p = state.bids[0].price if state.bids else 0.0
    ask_p = state.asks[0].price if state.asks else 0.0
    mid = (bid_p + ask_p) / 2 if (bid_p and ask_p) else 0.0
    bi = book_imbalance(state)
    mp = microprice(state)
    st = spread_ticks(state, min_price_increment)
    tfi, tfi_n = trade_flow_imbalance(state, tfi_window_sec)
    intens = trade_intensity(state, 30.0)
    return MicroSnapshot(
        book_imb=bi,
        micro_px=mp,
        spread_t=st,
        tfi=tfi,
        tfi_n=tfi_n,
        intensity=intens,
        mid=mid,
    )
