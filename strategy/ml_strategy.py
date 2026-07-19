"""ML-based strategy using LightGBM for signal generation."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from t_tech.invest import CandleInterval

from strategy.base import BaseStrategy, Signal, ExitSignal
from analysis.features import build_features, FEATURE_NAMES
from analysis.indicators import compute_indicators
from analysis.macro import MacroProvider
from ml.model import LGBMModel
from config.settings import Settings

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """LightGBM-based signal strategy with ATR stops."""

    name = "ml_lightgbm"

    def __init__(
        self,
        model: LGBMModel | None,
        settings: Settings,
        macro_provider: MacroProvider | None = None,
    ):
        self.model = model
        self.settings = settings
        self.macro_provider = macro_provider
        # Cache instrument_kind per figi so should_exit() can reuse it
        # without re-plumbing the BaseStrategy signature.  Populated on
        # every generate_signal() call.
        self._kind_cache: dict[str, str] = {}

    def set_model(self, model: LGBMModel):
        """Update the model (after retraining)."""
        self.model = model

    async def _fetch_macro(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Fetch macro context aligned to the main df's time range.

        MacroProvider caches within-scan so this is effectively one API
        call per macro per 15-min window, not per-ticker.  Failures are
        swallowed — build_features zero-fills macro columns.
        """
        if self.macro_provider is None or "time" not in df.columns:
            return None
        try:
            times = pd.to_datetime(df["time"], utc=True)
            if times.empty:
                return None
            return await self.macro_provider.get_macro_df(
                from_dt=times.min().to_pydatetime(),
                to_dt=times.max().to_pydatetime(),
                interval=CandleInterval.CANDLE_INTERVAL_HOUR,
            )
        except Exception as e:
            logger.debug(f"Macro fetch skipped: {e}")
            return None

    async def generate_signal(
        self, figi: str, ticker: str, df: pd.DataFrame, order_book: dict | None = None
    ) -> Signal:
        # Fallback if no model
        if not self.model or not self.model.is_trained:
            return Signal(
                figi=figi,
                ticker=ticker,
                direction="hold",
                confidence=0.0,
                strategy_name=self.name,
                timestamp=datetime.utcnow(),
            )

        # Build features
        spread_bps = order_book.get("spread_bps", 0) if order_book else 0
        book_imbalance = order_book.get("imbalance", 0) if order_book else 0
        instrument_kind = order_book.get("instrument_kind", "share") if order_book else "share"
        expiration_date = order_book.get("expiration_date") if order_book else None
        roll_window = int(getattr(self.settings, "FUTURES_ROLL_WINDOW_DAYS", 7))
        # Cache for should_exit() (same-figi follow-up)
        self._kind_cache[figi] = instrument_kind

        macro_df = await self._fetch_macro(df)
        features_df = build_features(
            df,
            spread_bps=spread_bps,
            book_imbalance=book_imbalance,
            macro_df=macro_df,
            instrument_kind=instrument_kind,
            ticker=ticker,
            expiration_date=expiration_date,
            roll_window_days=roll_window,
        )

        # Predict
        direction, confidence = self.model.predict(features_df[FEATURE_NAMES])

        # Asset-class-specific confidence gate.  Futures have shorter labels
        # (TB_MAX_HOLD_FUTURES) and higher intrabar noise — raise the floor.
        if instrument_kind == "future":
            threshold = float(
                getattr(self.settings, "SIGNAL_THRESHOLD_FUTURES", self.settings.SIGNAL_THRESHOLD)
            )
        else:
            threshold = float(self.settings.SIGNAL_THRESHOLD)

        # Skip low-confidence signals
        if confidence < threshold:
            direction = "hold"

        # ----- Meta-labelling gate (LdP ch.3) ----------------------------
        # When a meta model is bundled with the primary, run it on the same
        # feature row (augmented with the primary's full P-vector and direction).
        # Trade only when meta_conf >= meta.threshold; otherwise force "hold".
        # Confidence reported back to the engine is the GEOMETRIC MEAN of the
        # primary and meta — calibrated for size scaling downstream.
        meta_conf = None
        meta = getattr(self.model, "meta_model", None)
        if direction != "hold" and meta is not None and getattr(meta, "is_trained", False):
            try:
                from ml.meta_model import META_AUGMENT_FEATURES

                # Build augmented row using primary's full predict_proba
                proba_row = self.model.predict_proba(features_df[FEATURE_NAMES].iloc[[-1]])[0]
                aug = features_df[FEATURE_NAMES].iloc[[-1]].copy()
                aug["primary_pred_sell"] = float(proba_row[0])
                aug["primary_pred_hold"] = float(proba_row[1])
                aug["primary_pred_buy"] = float(proba_row[2])
                aug["primary_direction"] = 1 if direction == "buy" else -1
                aug["primary_conf"] = float(confidence)
                gate, mc = meta.predict(aug)
                meta_conf = mc
                if gate == 0:
                    logger.debug(
                        f"{ticker}: meta gate veto (meta_conf={mc:.3f} < "
                        f"thr={meta.threshold:.2f}); primary said {direction} "
                        f"@ {confidence:.3f}"
                    )
                    direction = "hold"
                else:
                    # Combine primary + meta multiplicatively (geometric mean
                    # of two probabilities ∈ [0,1] keeps the right scale).
                    confidence = float((confidence * mc) ** 0.5)
            except Exception as e:
                logger.warning(f"Meta-labelling step failed, ignoring meta: {e}")

        # ATR-based stops.  Multipliers come from settings (defaults 4.0 / 2.0
        # since 2026-05-14; legacy was hardcoded 2.0 / 3.0).  Wider stop lets
        # the model's signal_reversal exit drive most exits before the stop
        # fires; shorter target makes the winner-progress gate (now 0) and
        # the signal_reversal commission gate the practical exit triggers.
        df_ind = compute_indicators(df)
        last = df_ind.iloc[-1]
        atr_pct = (
            float(last.get("atr_14", 0) / last["close"] * 100)
            if last.get("atr_14") and last["close"] > 0
            else 2.0
        )
        stop_mult = float(getattr(self.settings, "STOP_ATR_MULT", 2.0))
        tgt_mult = float(getattr(self.settings, "TARGET_ATR_MULT", 3.0))
        stop_pct = max(atr_pct * stop_mult, 1.0)
        target_pct = max(atr_pct * tgt_mult, 2.0)

        # Extract top features for logging
        top_features = {}
        try:
            importance = self.model.feature_importance()
            top_5 = list(importance.keys())[:5]
            last_features = features_df[FEATURE_NAMES].iloc[-1]
            for f in top_5:
                val = last_features.get(f)
                if val is not None:
                    top_features[f] = round(float(val), 4)
        except Exception:
            pass

        return Signal(
            figi=figi,
            ticker=ticker,
            direction=direction,
            confidence=round(confidence, 3),
            strategy_name=self.name,
            timestamp=datetime.utcnow(),
            suggested_stop_pct=round(stop_pct, 2),
            suggested_target_pct=round(target_pct, 2),
            features=top_features,
        )

    async def should_exit(
        self, figi: str, ticker: str, entry_price: float, direction: str, df: pd.DataFrame
    ) -> Optional[ExitSignal]:
        """Exit if ML signal reverses with high confidence."""
        if not self.model or not self.model.is_trained:
            return None

        # Reuse kind cached during last generate_signal (0 if never scored)
        instrument_kind = self._kind_cache.get(figi, "share")
        roll_window = int(getattr(self.settings, "FUTURES_ROLL_WINDOW_DAYS", 7))
        macro_df = await self._fetch_macro(df)
        features_df = build_features(
            df,
            macro_df=macro_df,
            instrument_kind=instrument_kind,
            ticker=ticker,
            roll_window_days=roll_window,
        )
        pred_direction, confidence = self.model.predict(features_df[FEATURE_NAMES])

        # Strong reversal signal.
        # postmortem 2026-04-25: signal_reversal exits mean +0.24% (hit 55%),
        # best exit category.  Lowered threshold 0.70 → 0.62 to catch more
        # reversals while still filtering noise (below 0.6 = near coin-flip).
        if confidence > 0.62:
            if direction == "buy" and pred_direction == "sell":
                return ExitSignal(reason="signal_reversal", urgency="immediate")
            if direction == "sell" and pred_direction == "buy":
                return ExitSignal(reason="signal_reversal", urgency="immediate")

        return None
