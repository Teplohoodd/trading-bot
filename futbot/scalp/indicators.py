"""Fast indicators on the rolling 1-min candle buffer.

Computes a tight set of intraday confirmations to layer ON TOP of the
microstructure signal:

  * VWAP from session-start (UTC midnight is a reasonable proxy for MOEX
    intraday session — main hours are 07:00-18:50 UTC).
  * Fast RSI (period 7 not 14) — short-horizon overbought/oversold.
  * ATR(14) for stop sizing.
  * EMA(9) vs EMA(21) cross for trend bias.
  * MACD(5, 13, 5) — much shorter than the textbook 12/26/9 — for momentum.

All values are scalar (last bar).  Built from the streaming candle deque
in InstrumentState.
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


def _ema(values: list[float], n: int) -> float:
    """Last value of an EMA over `values`.  Returns NaN if not enough data."""
    if len(values) < n:
        return float("nan")
    alpha = 2 / (n + 1)
    e = values[0]
    for v in values[1:]:
        e = alpha * v + (1 - alpha) * e
    return e


def _atr_1m(candles: deque, n: int = 14) -> float:
    if len(candles) < n + 1:
        return float("nan")
    bars = list(candles)
    trs = []
    for i in range(1, len(bars)):
        h = bars[i]["high"]
        l = bars[i]["low"]
        pc = bars[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    # Wilder smoothing
    atr = sum(trs[:n]) / n
    for v in trs[n:]:
        atr = (atr * (n - 1) + v) / n
    return atr


def _rsi(closes: list[float], n: int) -> float:
    if len(closes) < n + 1:
        return float("nan")
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    # Wilder smoothing
    avg_g = sum(gains[:n]) / n
    avg_l = sum(losses[:n]) / n
    for g, l in zip(gains[n:], losses[n:]):
        avg_g = (avg_g * (n - 1) + g) / n
        avg_l = (avg_l * (n - 1) + l) / n
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100 - 100 / (1 + rs)


def _session_vwap(candles: deque) -> float:
    """Cumulative VWAP since the last UTC midnight."""
    if not candles:
        return float("nan")
    today_utc = datetime.now(timezone.utc).date()
    num, den = 0.0, 0.0
    for c in candles:
        bar_t = c["time"]
        if hasattr(bar_t, "date"):
            if bar_t.date() != today_utc:
                continue
        typical = (c["high"] + c["low"] + c["close"]) / 3
        num += typical * c["volume"]
        den += c["volume"]
    if den == 0:
        return float("nan")
    return num / den


def _macd_short(
    closes: list[float], fast: int = 5, slow: int = 13, signal: int = 5
) -> tuple[float, float]:
    """Short-period MACD.  Returns (macd_line, signal_line)."""
    if len(closes) < slow + signal:
        return float("nan"), float("nan")
    macd_series = []
    # We build EMA fast/slow incrementally
    alpha_f = 2 / (fast + 1)
    alpha_s = 2 / (slow + 1)
    ef = es = closes[0]
    for c in closes[1:]:
        ef = alpha_f * c + (1 - alpha_f) * ef
        es = alpha_s * c + (1 - alpha_s) * es
        macd_series.append(ef - es)
    # Signal = EMA of macd_series with `signal` period
    if len(macd_series) < signal:
        return macd_series[-1], float("nan")
    alpha_g = 2 / (signal + 1)
    sig = macd_series[0]
    for v in macd_series[1:]:
        sig = alpha_g * v + (1 - alpha_g) * sig
    return macd_series[-1], sig


@dataclass
class IndicatorSnapshot:
    last_close: float
    atr_1m: float  # in price units
    rsi_fast: float  # 0-100
    vwap: float  # session VWAP
    vwap_dev_atr: float  # (close − vwap) / atr — normalised
    ema_fast: float
    ema_slow: float
    ema_diff: float  # ema_fast − ema_slow, in price units
    macd: float
    macd_signal: float
    macd_hist: float  # macd − signal


def snapshot(candles: deque, *, rsi_period: int, ema_fast: int, ema_slow: int) -> IndicatorSnapshot:
    if not candles:
        return IndicatorSnapshot(
            last_close=float("nan"),
            atr_1m=float("nan"),
            rsi_fast=float("nan"),
            vwap=float("nan"),
            vwap_dev_atr=float("nan"),
            ema_fast=float("nan"),
            ema_slow=float("nan"),
            ema_diff=float("nan"),
            macd=float("nan"),
            macd_signal=float("nan"),
            macd_hist=float("nan"),
        )
    closes = [c["close"] for c in candles]
    last = closes[-1]
    atr = _atr_1m(candles)
    rsi = _rsi(closes, rsi_period)
    vwap = _session_vwap(candles)
    vwap_dev_atr = (
        (last - vwap) / atr
        if (not math.isnan(vwap) and not math.isnan(atr) and atr > 0)
        else float("nan")
    )
    ef = _ema(closes, ema_fast)
    es = _ema(closes, ema_slow)
    ema_diff = (ef - es) if (not math.isnan(ef) and not math.isnan(es)) else float("nan")
    macd, macd_sig = _macd_short(closes)
    macd_hist = (
        (macd - macd_sig) if (not math.isnan(macd) and not math.isnan(macd_sig)) else float("nan")
    )
    return IndicatorSnapshot(
        last_close=last,
        atr_1m=atr,
        rsi_fast=rsi,
        vwap=vwap,
        vwap_dev_atr=vwap_dev_atr,
        ema_fast=ef,
        ema_slow=es,
        ema_diff=ema_diff,
        macd=macd,
        macd_signal=macd_sig,
        macd_hist=macd_hist,
    )
