"""Technical analysis strategy: RSI + MACD + Bollinger Bands + volume confirmation."""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from strategy.base import BaseStrategy, Signal, ExitSignal
from analysis.indicators import compute_indicators


class TechnicalStrategy(BaseStrategy):
    """Composite TA strategy combining multiple indicators with weighted voting."""

    name = "technical"

    def __init__(self, settings=None):
        # settings is optional so the class is still instantiable in tests /
        # scripts where Settings isn't available.  When None, ATR multipliers
        # fall back to the legacy hardcoded 2.0 / 3.0.
        self.settings = settings

    async def generate_signal(
        self, figi: str, ticker: str, df: pd.DataFrame, order_book: dict | None = None
    ) -> Signal:
        df = compute_indicators(df)
        last = df.iloc[-1]

        scores = []
        weights = []

        # 1. RSI (weight 2)
        rsi = last.get("rsi_14", 50)
        if not np.isnan(rsi):
            if rsi < 30:
                scores.append(1.0)  # Oversold -> buy
            elif rsi > 70:
                scores.append(-1.0)  # Overbought -> sell
            elif rsi < 40:
                scores.append(0.3)
            elif rsi > 60:
                scores.append(-0.3)
            else:
                scores.append(0.0)
            weights.append(2.0)

        # 2. MACD histogram (weight 2)
        macd_hist = last.get("macd_histogram", 0)
        macd_hist_prev = df["macd_histogram"].iloc[-2] if len(df) > 1 else 0
        if not np.isnan(macd_hist):
            if macd_hist > 0 and macd_hist > macd_hist_prev:
                scores.append(1.0)  # Bullish momentum increasing
            elif macd_hist > 0:
                scores.append(0.3)
            elif macd_hist < 0 and macd_hist < macd_hist_prev:
                scores.append(-1.0)  # Bearish momentum increasing
            elif macd_hist < 0:
                scores.append(-0.3)
            else:
                scores.append(0.0)
            weights.append(2.0)

        # 3. Bollinger %B (weight 1.5)
        bb_pct = last.get("bb_percent_b", 0.5)
        if not np.isnan(bb_pct):
            if bb_pct < 0.0:
                scores.append(1.0)  # Below lower band
            elif bb_pct > 1.0:
                scores.append(-1.0)  # Above upper band
            elif bb_pct < 0.2:
                scores.append(0.6)
            elif bb_pct > 0.8:
                scores.append(-0.6)
            else:
                scores.append(0.0)
            weights.append(1.5)

        # 4. EMA crossover (weight 1.5)
        ema9 = last.get("ema_9", 0)
        ema21 = last.get("ema_21", 0)
        if ema9 and ema21 and not np.isnan(ema9) and not np.isnan(ema21):
            cross = (ema9 - ema21) / ema21 * 100
            scores.append(np.clip(cross / 2, -1, 1))
            weights.append(1.5)

        # 5. Volume confirmation (weight 1)
        vol_ratio = (
            last.get("volume", 0) / df["volume"].rolling(20).mean().iloc[-1]
            if df["volume"].rolling(20).mean().iloc[-1] > 0
            else 1
        )
        vol_boost = min(vol_ratio / 2, 1.5) if vol_ratio > 1.5 else 1.0

        # 6. ADX trend strength (weight 1)
        adx = last.get("adx_14", 20)
        if not np.isnan(adx):
            if adx > 25:
                scores.append(0.5 if scores and np.mean(scores) > 0 else -0.5)  # Confirm trend
            else:
                scores.append(0.0)  # Weak trend
            weights.append(1.0)

        # Compute weighted score
        if not scores:
            return Signal(
                figi=figi,
                ticker=ticker,
                direction="hold",
                confidence=0.0,
                strategy_name=self.name,
                timestamp=datetime.utcnow(),
            )

        weighted_score = np.average(scores, weights=weights) * vol_boost
        confidence = min(abs(weighted_score), 1.0)

        if weighted_score > 0.15:
            direction = "buy"
        elif weighted_score < -0.15:
            direction = "sell"
        else:
            direction = "hold"

        # ATR-based stop/target — multipliers from settings (defaults 4.0 / 2.0
        # since 2026-05-14; legacy was hardcoded 2.0 / 3.0).
        atr_pct = float(last.get("atr_14", 0) / last["close"] * 100) if last.get("atr_14") else 2.0
        stop_mult = float(getattr(self.settings, "STOP_ATR_MULT", 2.0)) if self.settings else 2.0
        tgt_mult = float(getattr(self.settings, "TARGET_ATR_MULT", 3.0)) if self.settings else 3.0
        stop_pct = max(atr_pct * stop_mult, 1.0)
        target_pct = max(atr_pct * tgt_mult, 2.0)

        return Signal(
            figi=figi,
            ticker=ticker,
            direction=direction,
            confidence=round(confidence, 3),
            strategy_name=self.name,
            timestamp=datetime.utcnow(),
            suggested_stop_pct=round(stop_pct, 2),
            suggested_target_pct=round(target_pct, 2),
            features={
                "rsi_14": round(float(rsi), 1) if not np.isnan(rsi) else None,
                "macd_hist": round(float(macd_hist), 4) if not np.isnan(macd_hist) else None,
                "bb_pct_b": round(float(bb_pct), 3) if not np.isnan(bb_pct) else None,
                "adx": round(float(adx), 1) if not np.isnan(adx) else None,
                "vol_ratio": round(vol_ratio, 2),
                "weighted_score": round(weighted_score, 3),
            },
        )

    async def should_exit(
        self, figi: str, ticker: str, entry_price: float, direction: str, df: pd.DataFrame
    ) -> Optional[ExitSignal]:
        df = compute_indicators(df)
        last = df.iloc[-1]
        current_price = last["close"]

        # P&L check
        if direction == "buy":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100

        rsi = last.get("rsi_14", 50)

        # RSI reversal exit
        if direction == "buy" and rsi > 80:
            return ExitSignal(reason="signal_reversal", urgency="immediate")
        if direction == "sell" and rsi < 20:
            return ExitSignal(reason="signal_reversal", urgency="immediate")

        # MACD reversal
        macd_hist = last.get("macd_histogram", 0)
        if direction == "buy" and macd_hist < 0 and pnl_pct > 0:
            return ExitSignal(reason="signal_reversal", urgency="next_bar")
        if direction == "sell" and macd_hist > 0 and pnl_pct > 0:
            return ExitSignal(reason="signal_reversal", urgency="next_bar")

        return None
