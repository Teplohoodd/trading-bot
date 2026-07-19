"""Scalp signal: combine microstructure + 1-min indicators into a vote.

Scoring model:
    score = 0.40 * book_imb_signal
          + 0.30 * tfi_signal              (only if tfi_n >= min)
          + 0.15 * indicator_alignment
          + 0.15 * vwap_pull               (mean-reversion against VWAP)

Each component returns a number in [-1, +1] aligned with direction.
Weights chosen by:
  * book_imb has the strongest short-horizon predictive power (papers:
    Cont, Stoikov, Talreja 2010; Avellaneda & Stoikov 2008)
  * TFI is next — order flow autocorrelates over seconds-minutes
  * 1-min RSI / MACD as soft confirmations
  * VWAP pull provides mean-reversion bias — works when book/TFI agree it's
    a small pullback in a larger trend

Entry rule:
  * |score| >= ENTRY_SCORE_MIN
  * spread <= MAX_SPREAD_TICKS
  * tfi_n >= MIN_TRADES (don't trade on stale flow)
  * atr_1m is fresh and > 0

Exit rule for an open position:
  * stop hit: bar.low <= stop (long) or bar.high >= stop (short)
  * TP hit
  * time cap
  * Signal flip (score in opposite direction with score >= 0.6 * ENTRY_SCORE_MIN)
  * Stale state (no stream events for > 30s)
"""

import logging
import math
from dataclasses import dataclass

from futbot.scalp import microstructure as micro
from futbot.scalp import indicators as ind

logger = logging.getLogger("futbot.scalp.strategy")


# Combined-score thresholds.
# 2026-05-20 data (112 trades, scalp v2 microstructure-only signal):
#     |s|~0.7 → 23% win, −97 ₽    ← worst losers
#     |s|~0.8 → 27% win, −3.2 ₽
#     |s|~0.9 → 37% win, −1.0 ₽   ← break-even territory
#     |s|~1.0 → 33% win, −1.3 ₽   ← break-even
# IC = +0.106 (p=0.27, n=112): signal IS in the right direction, just weak.
# Commission drag (0.08% RT × 112 trades) eats the small edge.
# Solution: only trade extreme signals where edge survives commission.
# Raising 0.70 → 0.85 cuts trade count ~60% and removes the bleeding bucket.
ENTRY_SCORE_MIN = 0.85
EXIT_SCORE_MIN = 0.55  # used to TRIGGER signal_flip exits (gated upstream
# by FLIP_MIN_AGE_SEC + FLIP_MIN_PROFIT_ATR)


@dataclass
class SignalResult:
    direction: str | None  # "buy" | "sell" | None
    score: float
    components: dict  # breakdown for logging
    rejection: str | None  # if direction is None, why


def _book_signal(book_imb: float, threshold: float) -> float:
    """Map book imbalance to a score in [-1, +1] with dead-zone."""
    if abs(book_imb) < threshold:
        return 0.0
    # Linear scaling once over threshold; cap at ±1
    return max(-1.0, min(1.0, book_imb / 0.5))


def _tfi_signal(tfi: float, tfi_n: int, threshold: float, min_n: int) -> float:
    if tfi_n < min_n or abs(tfi) < threshold:
        return 0.0
    return max(-1.0, min(1.0, tfi / 0.5))


def _indicator_signal(snap: ind.IndicatorSnapshot, settings) -> float:
    """Weighted sum of fast-RSI bias + EMA-cross direction + MACD-hist sign,
    each clipped to [-1, +1].  Returns 0 if any underlying value is NaN."""
    # Fast RSI — long bias if RSI < buy threshold, short if > sell threshold
    rsi_score = 0.0
    if not math.isnan(snap.rsi_fast):
        if snap.rsi_fast < settings.SCALP_RSI_BUY_BELOW:
            rsi_score = (settings.SCALP_RSI_BUY_BELOW - snap.rsi_fast) / 20.0
        elif snap.rsi_fast > settings.SCALP_RSI_SELL_ABOVE:
            rsi_score = -(snap.rsi_fast - settings.SCALP_RSI_SELL_ABOVE) / 20.0
        rsi_score = max(-1.0, min(1.0, rsi_score))
    # EMA cross — sign of (fast - slow) normalised by close
    ema_score = 0.0
    if (not math.isnan(snap.ema_diff)) and snap.last_close > 0:
        ema_score = max(-1.0, min(1.0, snap.ema_diff / snap.last_close * 200))
    # MACD histogram sign
    macd_score = 0.0
    if not math.isnan(snap.macd_hist) and snap.last_close > 0:
        macd_score = max(-1.0, min(1.0, snap.macd_hist / snap.last_close * 400))
    return (rsi_score + ema_score + macd_score) / 3


def _vwap_signal(vwap_dev_atr: float, threshold: float) -> float:
    """Mean-reversion: if price is < threshold ATRs ABOVE VWAP, slight short;
    if < threshold ATRs BELOW, slight long.  Beyond ±2 ATR we don't trust
    reversion any more (could be a trend day)."""
    if math.isnan(vwap_dev_atr):
        return 0.0
    if abs(vwap_dev_atr) > 2.0:
        return 0.0
    # Sign is inverse — far below VWAP → buy bias
    return -max(-1.0, min(1.0, vwap_dev_atr / threshold))


def evaluate(*, state, instrument, settings) -> SignalResult:
    """Compute the entry signal for one instrument.  `instrument` is the
    raw SDK Future object (we read its min_price_increment for spread)."""
    min_pi = getattr(instrument, "min_price_increment", None)
    if min_pi is None:
        return SignalResult(None, 0.0, {}, "no min_price_increment on instrument")
    try:
        from t_tech.invest.utils import quotation_to_decimal

        min_pi_f = float(quotation_to_decimal(min_pi))
    except Exception:
        min_pi_f = 0.01

    if not state.bids or not state.asks:
        return SignalResult(None, 0.0, {}, "no book data yet")

    micro_snap = micro.snapshot(state, min_pi_f, settings.SCALP_TFI_WINDOW_SEC)
    if micro_snap.spread_t > settings.SCALP_MAX_SPREAD_TICKS:
        return SignalResult(
            None,
            0.0,
            {"spread_ticks": micro_snap.spread_t},
            f"spread {micro_snap.spread_t} ticks > {settings.SCALP_MAX_SPREAD_TICKS}",
        )

    if len(state.candles_1m) < max(settings.SCALP_EMA_SLOW + 5, 20):
        return SignalResult(
            None,
            0.0,
            {"n_1m_bars": len(state.candles_1m)},
            f"warming up 1m candles ({len(state.candles_1m)})",
        )
    ind_snap = ind.snapshot(
        state.candles_1m,
        rsi_period=settings.SCALP_RSI_PERIOD,
        ema_fast=settings.SCALP_EMA_FAST,
        ema_slow=settings.SCALP_EMA_SLOW,
    )
    if math.isnan(ind_snap.atr_1m) or ind_snap.atr_1m <= 0:
        return SignalResult(
            None,
            0.0,
            {},
            "atr_1m not available",
        )

    book_s = _book_signal(micro_snap.book_imb, settings.SCALP_BOOK_IMBALANCE_MIN)
    tfi_s = _tfi_signal(
        micro_snap.tfi,
        micro_snap.tfi_n,
        settings.SCALP_TFI_MIN,
        settings.SCALP_TFI_MIN_TRADES,
    )
    ind_s = _indicator_signal(ind_snap, settings)
    vwap_s = _vwap_signal(ind_snap.vwap_dev_atr, settings.SCALP_VWAP_DEV_ATR_MAX)

    # ─── scalp v2 (2026-05-20) ─────────────────────────────────────────
    # Historical backtest of the INDICATOR portion (RSI+EMA+MACD+VWAP+CVD)
    # on 60d × 7 contracts × multiple horizons showed NEGATIVE NET P&L
    # everywhere — the indicator features added noise, not signal.
    # New v2 formula uses ONLY microstructure: 0.60 book + 0.40 tfi.
    # Indicator + VWAP are kept in `components` for debug logging but
    # contribute 0 weight to the entry decision.
    score = 0.60 * book_s + 0.40 * tfi_s
    components = {
        "book": round(book_s, 3),
        "tfi": round(tfi_s, 3),
        "ind": round(ind_s, 3),
        "vwap": round(vwap_s, 3),
        "score": round(score, 3),
        "spread_t": micro_snap.spread_t,
        "tfi_n": micro_snap.tfi_n,
        "atr_1m": round(ind_snap.atr_1m, 4),
        "rsi": round(ind_snap.rsi_fast, 1) if not math.isnan(ind_snap.rsi_fast) else None,
        "vwap_dev": (
            round(ind_snap.vwap_dev_atr, 2) if not math.isnan(ind_snap.vwap_dev_atr) else None
        ),
    }

    if abs(score) < ENTRY_SCORE_MIN:
        return SignalResult(
            None,
            score,
            components,
            f"score {score:+.2f} below threshold {ENTRY_SCORE_MIN}",
        )

    # Directional agreement v2: both primary signals (book, tfi) must point
    # the same way as score.  With only 2 components this is stricter than
    # the old "2 of 3" rule — if book and tfi disagree, refuse the trade
    # regardless of composite magnitude.
    score_sign = 1 if score > 0 else -1
    book_agrees = (book_s > 0.05 and score_sign > 0) or (book_s < -0.05 and score_sign < 0)
    tfi_agrees = (tfi_s > 0.05 and score_sign > 0) or (tfi_s < -0.05 and score_sign < 0)
    if not (book_agrees and tfi_agrees):
        return SignalResult(
            None,
            score,
            components,
            f"book/tfi disagree (book={book_s:+.2f}, tfi={tfi_s:+.2f})",
        )

    return SignalResult(
        direction="buy" if score > 0 else "sell",
        score=score,
        components=components,
        rejection=None,
    )


def should_exit(*, state, instrument, settings, position_direction: str) -> SignalResult:
    """Check if an OPEN position should exit because the signal flipped.
    Returns a SignalResult where direction != position_direction means
    we should close.  Otherwise direction is None."""
    sig = evaluate(state=state, instrument=instrument, settings=settings)
    # We only care about *strong* opposite signals as exit triggers.
    if sig.direction is not None and sig.direction != position_direction:
        if abs(sig.score) >= EXIT_SCORE_MIN:
            return sig
    return SignalResult(None, sig.score, sig.components, "no flip")
