"""Walk-forward model trainer."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.broker import BrokerClient
from analysis.features import (
    build_features,
    build_triple_barrier_target,
    FEATURE_NAMES,
    CATEGORICAL_FEATURE_NAMES,
)
from analysis.macro import MacroProvider
from analysis.screener import _candles_to_df
from database.db import Repository
from ml.model import LGBMModel
from ml.meta_model import MetaLabellingModel, build_meta_dataset, META_AUGMENT_FEATURES
from config.settings import Settings
from tinkoff.invest import CandleInterval

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains LightGBM models with walk-forward validation."""

    def __init__(
        self,
        broker: BrokerClient,
        db: Repository,
        settings: Settings,
        macro_provider: MacroProvider | None = None,
    ):
        self.broker = broker
        self.db = db
        self.settings = settings
        self.macro_provider = macro_provider

    async def _fetch_candles_with_fallback(
        self, figi: str, ticker: str, lookback_days: int
    ) -> tuple[list, CandleInterval | None]:
        """Try hourly candles first; if unavailable (30014), fall back to daily.

        Hourly gives more samples but some tickers/time-ranges return 30014.
        Daily candles are available for almost all listed instruments.

        Returns (candles, interval_used) so callers can request matching-
        interval macro data.  interval_used is None if no candles fetched.
        """
        now = datetime.now(timezone.utc)

        # 1st attempt: hourly candles over lookback_days
        from_dt = now - timedelta(days=lookback_days)
        candles = await self.broker.get_candles(
            figi, from_dt, now, interval=CandleInterval.CANDLE_INTERVAL_HOUR
        )
        if candles:
            logger.info(f"{ticker}: {len(candles)} hourly candles")
            return candles, CandleInterval.CANDLE_INTERVAL_HOUR

        # Fallback: 3 years of daily candles.
        # SMA-200 warmup needs 200 bars; 1095 calendar days ≈ 780 trading days
        # → ~580 clean rows per ticker, enough for meaningful walk-forward CV.
        from_dt_daily = now - timedelta(days=1095)
        candles = await self.broker.get_candles(
            figi, from_dt_daily, now, interval=CandleInterval.CANDLE_INTERVAL_DAY
        )
        if candles:
            logger.info(f"{ticker}: hourly unavailable, using {len(candles)} daily candles")
            return candles, CandleInterval.CANDLE_INTERVAL_DAY

        logger.warning(f"{ticker}: no candle data available (hourly or daily)")
        return candles, None

    # Triple-barrier hyperparameters (Jansen / López de Prado).
    # max_hold defines label horizon → embargo size for purged CV.
    #
    # IMPORTANT: labels are SYMMETRIC (PT = SL) to keep class distribution
    # balanced.  An asymmetric 2:1 PT:SL gives P(lower hits first) ≈ 67 %
    # for a random walk, which collapses ~60 % of samples into the "sell"
    # class and makes classification boundaries degenerate.  The bot's
    # *actual trading* R/R is controlled separately by risk/manager.py —
    # labels answer "did price move N sigma up or down first?", not "would
    # my exit rule have been profitable?"  (Jansen ch.4 §4.5 on label
    # symmetry for balanced multi-class models.)
    TB_PT = 2.0  # upper barrier = entry + 2 × ATR
    TB_SL = 2.0  # lower barrier = entry − 2 × ATR (symmetric)

    # --- Rollback gate thresholds ------------------------------------------
    # New model is rejected (keep the previous active one) when BOTH
    # accuracy and F1 drop by more than these fractions compared to what's
    # stored in model_registry for the currently-active model.  Stops the
    # "retrain-degrades-every-day" pattern observed in live logs
    # (45 % → 40 % → 39 % over three days).
    ROLLBACK_ACC_DROP = 0.02  # 2 % absolute accuracy drop
    ROLLBACK_F1_DROP = 0.02  # 2 % absolute F1 drop

    def _should_rollback(
        self,
        existing: dict | None,
        new_acc: float,
        new_f1: float,
        label: str,
        new_tb_max_hold: int | None = None,
        new_feature_names: list[str] | None = None,
    ) -> bool:
        """Return True if the new CV metrics are materially worse than the
        currently-active model in model_registry.  When True, the caller
        should skip saving and keep the previous version active.

        Bypass conditions (all skip numeric comparison):
        -----------------------------------------------
        1. **Horizon changed** — acc at h=10 and h=20 are not comparable;
           longer horizons yield harder classification tasks.
        2. **Feature schema changed** — old and new models were trained on
           different input spaces.  A drop from 38→45 features where 7 new
           columns are all-zero is expected noise reduction, not degradation.
           Once the new features accumulate real signal (futures data) the
           metrics will recover naturally.  Blocking the update would lock
           the system into a stale schema that can never receive the new
           features.
        """
        if not existing:
            return False

        # --- Bypass 1: label horizon changed ---------------------------------
        prev_hold = existing.get("tb_max_hold") or 10  # old rows default to 10
        curr_hold = new_tb_max_hold or int(getattr(self.settings, "TB_MAX_HOLD", 20))
        if prev_hold != curr_hold:
            logger.info(
                f"[{label}] Horizon changed {prev_hold}→{curr_hold} bars: "
                f"skipping rollback (metrics at different horizons are not "
                f"comparable).  New model will be saved."
            )
            return False

        # --- Bypass 2: feature schema changed --------------------------------
        prev_features = existing.get("feature_names") or []
        curr_features = new_feature_names or []
        if curr_features and set(curr_features) != set(prev_features):
            added = set(curr_features) - set(prev_features)
            removed = set(prev_features) - set(curr_features)
            logger.info(
                f"[{label}] Feature schema changed "
                f"({len(prev_features)}→{len(curr_features)} features; "
                f"+{len(added)} added, -{len(removed)} removed): "
                f"skipping rollback — acc/f1 on different input spaces are "
                f"not comparable.  New model will be saved."
            )
            return False

        prev_acc = float(existing.get("accuracy") or 0.0)
        prev_f1 = float(existing.get("f1_score") or 0.0)
        acc_drop = prev_acc - new_acc
        f1_drop = prev_f1 - new_f1
        # Only rollback when BOTH metrics regress — avoids over-triggering on
        # fold noise.  F1 alone can drop 3-4 % between runs from class-mix
        # shifts even when the model is genuinely as good or better.
        if acc_drop > self.ROLLBACK_ACC_DROP and f1_drop > self.ROLLBACK_F1_DROP:
            logger.warning(
                f"[{label}] ROLLBACK: new acc={new_acc:.4f} (−{acc_drop:.3f}), "
                f"f1={new_f1:.4f} (−{f1_drop:.3f}) vs prev v{existing.get('version')} "
                f"(acc={prev_acc:.4f}, f1={prev_f1:.4f}, horizon={prev_hold}).  "
                f"Keeping previous model."
            )
            return True
        return False

    def _max_hold_for_kind(self, instrument_kind: str) -> int:
        """Return triple-barrier horizon appropriate for this instrument kind.

        Futures mean-revert faster than shares — at hourly resolution, a 10-
        bar window on Si/BR often captures 2-3 full direction flips, washing
        out the directional signal.  Halving the horizon for futures keeps
        label quality comparable.
        """
        if instrument_kind == "future":
            return int(getattr(self.settings, "TB_MAX_HOLD_FUTURES", 10))
        return int(getattr(self.settings, "TB_MAX_HOLD", 20))

    async def _fetch_macro_for_df(
        self, df: pd.DataFrame, interval: CandleInterval
    ) -> pd.DataFrame | None:
        """Fetch macro candles covering the time range of ``df`` at ``interval``.

        Returns None silently on any failure — training must still work if
        macro fetches break, just without those features.
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
                interval=interval,
            )
        except Exception as e:
            logger.debug(f"Macro fetch skipped in trainer: {e}")
            return None

    def _build_dataset(
        self,
        df: pd.DataFrame,
        macro_df: pd.DataFrame | None = None,
        instrument_kind: str = "share",
        ticker: str | None = None,
        expiration_date=None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series] | None:
        """Return (X, y, sample_weights) aligned and cleaned, or None if too sparse.

        Uses triple-barrier labels (ATR-scaled PT/SL/time-out) so the class
        mix adapts to volatility regime — a universal model trained on both
        high-vol and low-vol tickers stays balanced.  ``instrument_kind``
        controls both (a) the label horizon and (b) the ``is_future`` feature
        emitted per row.  For futures, ``expiration_date`` is used to clip
        the dataset so no label reaches across contract expiry (label would
        be meaningless — the contract ceases to trade).
        """
        max_hold = self._max_hold_for_kind(instrument_kind)

        # --- Pre-expiry clip (futures only) -------------------------------
        # Drop rows whose triple-barrier label window would cross expiry.
        # We need at least max_hold+5 bars of post-bar history to resolve
        # the label; rows closer to expiry than that can't be labelled and
        # would just become NaN dropped later anyway — but clipping here
        # keeps the dataset clean and lets us abort early when a contract
        # is too close to expiry to train on.
        roll_window = int(getattr(self.settings, "FUTURES_ROLL_WINDOW_DAYS", 7))
        if instrument_kind == "future" and expiration_date is not None and "time" in df.columns:
            try:
                exp_ts = pd.Timestamp(expiration_date)
                if exp_ts.tzinfo is None:
                    exp_ts = exp_ts.tz_localize("UTC")
                else:
                    exp_ts = exp_ts.tz_convert("UTC")
                bar_ts = pd.to_datetime(df["time"], utc=True)
                diffs = bar_ts.diff().dropna()
                if len(diffs):
                    bar_sec = diffs.median().total_seconds()
                    if bar_sec > 0:
                        bars_to_exp = ((exp_ts - bar_ts).dt.total_seconds() / bar_sec).to_numpy()
                        safe_mask = bars_to_exp >= (max_hold + 5)
                        kept = int(safe_mask.sum())
                        if kept < 100:
                            logger.warning(
                                f"{ticker or '?'}: only {kept} bars before expiry "
                                f"(max_hold={max_hold}) — cannot train"
                            )
                            return None
                        dropped = int((~safe_mask).sum())
                        if dropped > 0:
                            logger.info(
                                f"{ticker or '?'}: dropped {dropped} near-expiry "
                                f"bars ({kept} kept) before labeling"
                            )
                        df = df.loc[safe_mask].reset_index(drop=True)
                        if macro_df is not None and "time" in df.columns:
                            # macro_df uses its own time axis, nothing to clip
                            pass
            except Exception as e:
                logger.debug(f"Pre-expiry clip failed, continuing: {e}")

        features_df = build_features(
            df,
            macro_df=macro_df,
            instrument_kind=instrument_kind,
            ticker=ticker,
            expiration_date=expiration_date,
            roll_window_days=roll_window,
        )
        labels, weights = build_triple_barrier_target(
            df,
            pt_multiplier=self.TB_PT,
            sl_multiplier=self.TB_SL,
            max_hold=max_hold,
        )

        combined = pd.concat(
            [features_df[FEATURE_NAMES], labels.rename("target"), weights.rename("weight")],
            axis=1,
        ).dropna()

        if combined.empty:
            return None

        X = combined[FEATURE_NAMES]
        y = combined["target"].astype(int)
        w = combined["weight"].astype(float)
        return X, y, w

    async def train_model(
        self, figi: str, ticker: str = "UNKNOWN", instrument_kind: str | None = None
    ) -> LGBMModel | None:
        """Train a per-ticker model.  Returns trained model or None if quality/size fails.

        If ``instrument_kind`` is None, it's looked up via broker.get_instrument_info.
        For futures the contract's expiration_date is also fetched so the
        dataset can be clipped before expiry and ``days_to_expiry``/
        ``in_roll_window`` features can be populated.
        """
        logger.info(f"Training model for {ticker} ({figi})...")

        # Resolve kind + futures metadata once
        instrument_obj = None
        expiration_date = None
        if instrument_kind is None:
            try:
                instrument_obj, instrument_kind = await self.broker.get_instrument_info(figi)
                if instrument_kind not in ("share", "future"):
                    instrument_kind = "share"
            except Exception:
                instrument_kind = "share"
        if instrument_kind == "future":
            try:
                if instrument_obj is None:
                    instrument_obj, _ = await self.broker.get_instrument_info(figi)
                meta = self.broker.extract_futures_metadata(instrument_obj)
                expiration_date = meta.get("expiration_date")
            except Exception as e:
                logger.debug(f"Could not fetch futures metadata for {ticker}: {e}")

        candles, interval = await self._fetch_candles_with_fallback(
            figi, ticker, self.settings.ML_LOOKBACK_DAYS
        )

        if len(candles) < self.settings.ML_MIN_SAMPLES:
            logger.warning(
                f"Insufficient data for {ticker}: {len(candles)} < {self.settings.ML_MIN_SAMPLES}"
            )
            return None

        df = _candles_to_df(candles)
        logger.info(f"Fetched {len(df)} candles for {ticker} (kind={instrument_kind})")

        macro_df = await self._fetch_macro_for_df(df, interval) if interval else None
        dataset = self._build_dataset(
            df,
            macro_df=macro_df,
            instrument_kind=instrument_kind,
            ticker=ticker,
            expiration_date=expiration_date,
        )
        if dataset is None:
            logger.warning(f"No clean dataset for {ticker}")
            return None
        X, y, w = dataset

        if len(X) < 200:
            logger.warning(f"Insufficient clean data for {ticker}: {len(X)} < 200")
            return None

        # Purged walk-forward validation — embargo = label horizon.
        # LightGBM's fit() is a CPU-bound sync call (5-60 s per fold).  Running
        # it directly in the async event loop would freeze Telegram polling
        # and every other engine task for minutes.  Off-load to a worker
        # thread so the event loop stays responsive.
        max_hold = self._max_hold_for_kind(instrument_kind)
        metrics = await asyncio.to_thread(
            self._purged_walk_forward_validate,
            X,
            y,
            w,
            5,
            max_hold,
        )
        if not metrics:
            logger.warning(f"No CV folds produced for {ticker}")
            return None
        avg_f1 = float(np.mean([m["f1"] for m in metrics]))
        avg_acc = float(np.mean([m["accuracy"] for m in metrics]))
        avg_ic = float(np.mean([m.get("ic", 0.0) for m in metrics]))

        logger.info(
            f"Walk-forward results for {ticker}: "
            f"avg_f1={avg_f1:.4f}, avg_acc={avg_acc:.4f}, avg_ic={avg_ic:+.4f}"
        )

        # Gate by IC as well as F1: IC > 0.02 is the Jansen-recommended floor
        # for "model has directional edge" on tabular financial data.
        if avg_f1 < 0.35 and avg_ic < 0.02:
            logger.warning(f"Model quality too low for {ticker}: f1={avg_f1:.4f}, ic={avg_ic:+.4f}")
            return None

        current_max_hold = self._max_hold_for_kind(instrument_kind)
        existing = await self.db.get_active_model(figi)
        # Rollback gate — if the new CV is materially worse than the current
        # active model, keep the old one instead of overwriting.
        # Passes current horizon so the gate can bypass comparison when
        # TB_MAX_HOLD changed (different prediction task ≠ worse model).
        if self._should_rollback(
            existing,
            avg_acc,
            avg_f1,
            label=ticker,
            new_tb_max_hold=current_max_hold,
            new_feature_names=FEATURE_NAMES,
        ):
            return None
        version = (existing["version"] + 1) if existing else 1

        # Final fit on all data (85/15 split), retaining sample weights
        split_idx = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        w_train = w.iloc[:split_idx]

        model = LGBMModel()
        # Final fit is also sync + CPU-heavy — off-load to a thread.
        # Pass categorical_features so LightGBM uses native categorical
        # splits on asset_class_code (no one-hot needed; the categorical
        # split tree is invariant to code numbering).
        final_metrics = await asyncio.to_thread(
            model.train,
            X_train,
            y_train,
            X_val,
            y_val,
            figi,
            version,
            w_train,
            CATEGORICAL_FEATURE_NAMES,
        )

        # Meta-labelling: train secondary classifier over OOF predictions.
        # This is a thread-bound CPU job — off-load to keep the event loop
        # responsive while training (~30-90s extra per retrain).
        meta_threshold = float(getattr(self.settings, "META_LABEL_THRESHOLD", 0.55))
        primary_min_conf = float(getattr(self.settings, "META_PRIMARY_MIN_CONF", 0.5))
        meta_model = await asyncio.to_thread(
            self._train_meta_model,
            X,
            y,
            w,
            max_hold,
            primary_min_conf,
            meta_threshold,
        )
        if meta_model is not None:
            model.set_meta_model(meta_model)

        model_path = self.settings.MODELS_DIR / f"{ticker}_{version}.joblib"
        await asyncio.to_thread(model.save, model_path)

        await self.db.register_model(
            figi=figi,
            model_path=str(model_path),
            version=version,
            accuracy=final_metrics["accuracy"],
            f1_score=final_metrics["f1"],
            train_samples=len(X_train),
            feature_names=FEATURE_NAMES,
            tb_max_hold=current_max_hold,
        )

        logger.info(
            f"Model v{version} for {ticker} saved "
            f"(final: acc={final_metrics['accuracy']:.4f}, "
            f"f1={final_metrics['f1']:.4f}, ic={final_metrics.get('ic', 0):+.4f})"
        )
        return model

    def _oof_primary_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None,
        n_folds: int = 3,
        embargo: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate out-of-fold primary predictions over the dataset.

        Strategy: expanding-window walk-forward.  For each fold, train a
        primary on the data BEFORE the test window (purged by ``embargo``)
        and predict on the test window.  Concatenate test predictions →
        unbiased OOF estimates for the *covered* portion of the dataset.

        Returns (oof_proba, oof_indices) where:
            oof_proba : np.ndarray of shape (n_oof, 3) [P(sell), P(hold), P(buy)]
            oof_indices : original positions in X corresponding to the OOF rows

        Rows in the first ``n_folds``-th of the dataset are NOT in OOF —
        the meta dataset filters by oof_indices later.
        """
        total = len(X)
        test_size = total // (n_folds + 1)
        oof_proba_chunks: list[np.ndarray] = []
        oof_idx_chunks: list[np.ndarray] = []

        for fold in range(n_folds):
            train_end = total - (n_folds - fold) * test_size
            test_start = train_end
            test_end = test_start + test_size
            train_end_purged = max(0, train_end - embargo)

            X_train = X.iloc[:train_end_purged]
            y_train = y.iloc[:train_end_purged]
            X_test = X.iloc[test_start:test_end]
            w_train = sample_weights.iloc[:train_end_purged] if sample_weights is not None else None

            if len(X_train) < 100 or len(X_test) < 20:
                continue

            primary_fold = LGBMModel()
            primary_fold.train(
                X_train,
                y_train,
                X_test,
                y.iloc[test_start:test_end],
                sample_weight=w_train,
                categorical_features=CATEGORICAL_FEATURE_NAMES,
            )
            proba = primary_fold.predict_proba(X_test)
            oof_proba_chunks.append(proba)
            oof_idx_chunks.append(np.arange(test_start, test_end))

        if not oof_proba_chunks:
            return np.zeros((0, 3)), np.array([], dtype=int)

        oof_proba = np.vstack(oof_proba_chunks)
        oof_idx = np.concatenate(oof_idx_chunks)
        return oof_proba, oof_idx

    def _train_meta_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series,
        max_hold: int,
        primary_min_conf: float = 0.5,
        meta_threshold: float = 0.55,
    ) -> MetaLabellingModel | None:
        """Train the meta-labelling secondary classifier (LdP ch.3).

        Steps:
            1. Generate OOF primary predictions over X (purged walk-forward).
            2. Build meta dataset (drop hold-predicted rows; meta-target =
               primary's directional prediction matched truth).
            3. Train binary LightGBM with another purged time-series split.
        Returns trained MetaLabellingModel or None if any stage failed.
        """
        try:
            logger.info("Generating OOF primary predictions for meta training...")
            oof_proba, oof_idx = self._oof_primary_predictions(
                X,
                y,
                sample_weights,
                n_folds=3,
                embargo=max_hold,
            )
            if oof_proba.shape[0] < 200:
                logger.warning(
                    f"Meta training aborted: only {oof_proba.shape[0]} OOF samples " f"(need ≥200)"
                )
                return None

            X_oof = X.iloc[oof_idx].reset_index(drop=True)
            y_oof = y.iloc[oof_idx].reset_index(drop=True)
            w_oof = sample_weights.iloc[oof_idx].reset_index(drop=True)

            X_meta, y_meta, w_meta = build_meta_dataset(
                X_oof,
                oof_proba,
                y_oof,
                sample_weights=w_oof,
                primary_min_conf=primary_min_conf,
            )
            if len(X_meta) < 100:
                logger.warning(
                    f"Meta training aborted: only {len(X_meta)} meta samples after filter"
                )
                return None

            # Class balance sanity
            pos_pct = float(y_meta.mean())
            logger.info(
                f"Meta dataset: n={len(X_meta)}, P(primary correct)={pos_pct:.3f}, "
                f"feature_count={X_meta.shape[1]}"
            )
            if pos_pct < 0.05 or pos_pct > 0.95:
                logger.warning(f"Meta target degenerate (pos_pct={pos_pct:.2f}); skipping")
                return None

            # Purged train/val split for meta
            split = int(len(X_meta) * 0.8)
            split = max(50, split - max_hold)  # purge embargo
            X_tr, X_vl = X_meta.iloc[:split], X_meta.iloc[split + max_hold :]
            y_tr, y_vl = y_meta.iloc[:split], y_meta.iloc[split + max_hold :]
            w_tr = w_meta.iloc[:split]

            if len(X_vl) < 30:
                # Not enough validation rows — train on all, evaluate on train
                X_vl, y_vl = X_meta, y_meta

            meta = MetaLabellingModel(threshold=meta_threshold)
            cat_in_meta = [c for c in CATEGORICAL_FEATURE_NAMES if c in X_meta.columns]
            metrics = meta.train(
                X_tr,
                y_tr,
                sample_weight=w_tr,
                X_val=X_vl,
                y_val=y_vl,
                categorical_features=cat_in_meta,
            )
            logger.info(
                f"Meta model trained: acc={metrics['accuracy']:.3f}, "
                f"f1={metrics['f1']:.3f}, prec={metrics['precision']:.3f}, "
                f"recall={metrics['recall']:.3f}, n={metrics['n_train']}"
            )
            return meta
        except Exception as e:
            logger.error(f"Meta-labelling training failed: {e}", exc_info=True)
            return None

    def _purged_walk_forward_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weights: pd.Series | None = None,
        n_folds: int = 5,
        embargo: int = 10,
    ) -> list[dict]:
        """Expanding-window walk-forward CV with embargo (López de Prado ch.7).

        Because a triple-barrier label at time ``t`` depends on prices in
        ``[t+1, t+max_hold]``, training on rows near the test window leaks
        information.  The embargo drops the last ``embargo`` samples of each
        training fold so labels never overlap the next test window.

        This typically drops raw F1 by 2-5% vs naive CV but the remaining
        signal is real and out-of-sample reliable — Jansen ch.6.
        """
        total = len(X)
        test_size = total // (n_folds + 1)
        metrics: list[dict] = []

        for fold in range(n_folds):
            train_end = total - (n_folds - fold) * test_size
            test_start = train_end
            test_end = test_start + test_size

            # Purge: drop rows whose labels could overlap the test window
            train_end_purged = max(0, train_end - embargo)

            X_train = X.iloc[:train_end_purged]
            y_train = y.iloc[:train_end_purged]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            w_train = sample_weights.iloc[:train_end_purged] if sample_weights is not None else None

            if len(X_train) < 100 or len(X_test) < 20:
                continue

            model = LGBMModel()
            m = model.train(
                X_train,
                y_train,
                X_test,
                y_test,
                sample_weight=w_train,
                categorical_features=CATEGORICAL_FEATURE_NAMES,
            )
            metrics.append(m)
            logger.debug(
                f"Fold {fold+1}: acc={m['accuracy']:.4f}, "
                f"f1={m['f1']:.4f}, ic={m.get('ic', 0):+.4f}"
            )

        return metrics

    async def train_universal_model(
        self,
        tickers_figis: list[tuple[str, str]] | list[tuple[str, str, str]],
        force_overwrite: bool = False,
    ) -> LGBMModel | None:
        """Train a universal model on pooled data from multiple tickers.

        Each ticker contributes its own triple-barrier-labelled slice (so
        labels are ATR-scaled per asset).  Within-ticker temporal order is
        preserved; across tickers samples are concatenated.  Walk-forward
        CV is applied per ticker and averaged — prevents cross-asset
        leakage that would arise from a single pooled shuffle-split.

        ``tickers_figis`` accepts either 2-tuples (ticker, figi) for
        backwards compatibility or 3-tuples (ticker, figi, kind) when the
        caller already knows the instrument kind (screener path).  Missing
        kind is resolved via broker.get_instrument_info.
        """
        logger.info(f"Training universal model on {len(tickers_figis)} tickers...")

        per_ticker: list[tuple[str, str, pd.DataFrame, pd.Series, pd.Series]] = []

        for item in tickers_figis:
            if len(item) == 3:
                ticker, figi, kind = item
            else:
                ticker, figi = item
                kind = None
            try:
                instrument_obj = None
                if kind is None:
                    try:
                        instrument_obj, kind = await self.broker.get_instrument_info(figi)
                        if kind not in ("share", "future"):
                            kind = "share"
                    except Exception:
                        kind = "share"

                # Fetch expiration_date for futures so labels don't cross expiry
                expiration_date = None
                if kind == "future":
                    try:
                        if instrument_obj is None:
                            instrument_obj, _ = await self.broker.get_instrument_info(figi)
                        meta = self.broker.extract_futures_metadata(instrument_obj)
                        expiration_date = meta.get("expiration_date")
                    except Exception as e:
                        logger.debug(f"Could not fetch futures metadata for {ticker}: {e}")

                candles, interval = await self._fetch_candles_with_fallback(
                    figi, ticker, self.settings.ML_LOOKBACK_DAYS
                )
                if len(candles) < 200:
                    logger.debug(f"Skip {ticker} for universal model: only {len(candles)} candles")
                    continue

                df = _candles_to_df(candles)
                macro_df = await self._fetch_macro_for_df(df, interval) if interval else None
                dataset = self._build_dataset(
                    df,
                    macro_df=macro_df,
                    instrument_kind=kind,
                    ticker=ticker,
                    expiration_date=expiration_date,
                )
                if dataset is None:
                    continue
                X_t, y_t, w_t = dataset
                if len(X_t) >= 100:
                    per_ticker.append((ticker, kind, X_t, y_t, w_t))
            except Exception as e:
                logger.debug(f"Skip {ticker} for universal model: {e}")

        if not per_ticker:
            logger.warning("No data for universal model")
            return None

        all_X = pd.concat([t[2] for t in per_ticker], ignore_index=True)
        all_y = pd.concat([t[3] for t in per_ticker], ignore_index=True)
        all_w = pd.concat([t[4] for t in per_ticker], ignore_index=True)

        # Class distribution diagnostics + kind breakdown
        counts = all_y.value_counts().sort_index()
        class_names = {0: "sell", 1: "hold", 2: "buy"}
        dist_str = "  ".join(
            f"{class_names.get(k, k)}: {v} ({v/len(all_y)*100:.0f}%)" for k, v in counts.items()
        )
        n_shares = sum(1 for t in per_ticker if t[1] == "share")
        n_futures = sum(1 for t in per_ticker if t[1] == "future")
        logger.info(
            f"Universal model data: {len(all_X)} samples from {len(per_ticker)} tickers "
            f"(shares={n_shares}, futures={n_futures})  |  {dist_str}"
        )

        # --- Per-ticker walk-forward CV, averaged --------------------------
        # Pooling all tickers and splitting on a single time axis would put
        # the whole history of ticker-N into the test fold — lookahead-free
        # but distribution-skewed.  Instead we CV each ticker independently
        # and average the fold metrics (Jansen ch.6 "cross-sectional CV").
        all_metrics: list[dict] = []
        metrics_by_kind: dict[str, list[dict]] = {"share": [], "future": []}
        for ticker, kind, X_t, y_t, w_t in per_ticker:
            if len(X_t) < 200:
                continue
            max_hold_k = self._max_hold_for_kind(kind)
            # Each ticker's CV is a pure CPU job — off-load to a thread so
            # engine + Telegram bot keep responding while it runs.
            m = await asyncio.to_thread(
                self._purged_walk_forward_validate,
                X_t,
                y_t,
                w_t,
                3,
                max_hold_k,
            )
            all_metrics.extend(m)
            metrics_by_kind.setdefault(kind, []).extend(m)

        avg_f1 = avg_acc = avg_ic = 0.0
        if all_metrics:
            avg_f1 = float(np.mean([m["f1"] for m in all_metrics]))
            avg_acc = float(np.mean([m["accuracy"] for m in all_metrics]))
            avg_ic = float(np.mean([m.get("ic", 0.0) for m in all_metrics]))
            logger.info(
                f"Universal CV ({len(all_metrics)} folds): "
                f"acc={avg_acc:.4f}, f1={avg_f1:.4f}, ic={avg_ic:+.4f}"
            )
            # Per-kind breakdown so regressions on either side are visible
            for kind_key, kind_metrics in metrics_by_kind.items():
                if not kind_metrics:
                    continue
                k_f1 = float(np.mean([m["f1"] for m in kind_metrics]))
                k_acc = float(np.mean([m["accuracy"] for m in kind_metrics]))
                k_ic = float(np.mean([m.get("ic", 0.0) for m in kind_metrics]))
                logger.info(
                    f"  [{kind_key:6}] {len(kind_metrics)} folds: "
                    f"acc={k_acc:.4f}, f1={k_f1:.4f}, ic={k_ic:+.4f}"
                )

        # Universal model uses the shares horizon as the canonical label
        # horizon (futures horizon is a derived fraction of it).
        universal_max_hold = int(getattr(self.settings, "TB_MAX_HOLD", 20))
        existing = await self.db.get_active_model(None)
        # Rollback gate — if the new universal CV is materially worse than
        # the currently-active model, skip the final fit + save so the old
        # model keeps serving signals.  Without this, daily retraining was
        # observed to monotonically degrade accuracy (45 → 40 → 39 %).
        if (
            not force_overwrite
            and all_metrics
            and self._should_rollback(
                existing,
                avg_acc,
                avg_f1,
                label="universal",
                new_tb_max_hold=universal_max_hold,
                new_feature_names=FEATURE_NAMES,
            )
        ):
            return None
        version = (existing["version"] + 1) if existing else 1

        # --- Final pooled fit ---------------------------------------------
        split_idx = int(len(all_X) * 0.85)
        X_train, X_val = all_X.iloc[:split_idx], all_X.iloc[split_idx:]
        y_train, y_val = all_y.iloc[:split_idx], all_y.iloc[split_idx:]
        w_train = all_w.iloc[:split_idx]

        model = LGBMModel()
        # Final pooled fit is sync CPU-heavy — off-load.
        # Pass categorical feature names so LightGBM uses native asset_class
        # splits (important for the pooled model — lets it learn
        # per-asset-family rules without needing N separate models).
        metrics = await asyncio.to_thread(
            model.train,
            X_train,
            y_train,
            X_val,
            y_val,
            None,
            version,
            w_train,
            CATEGORICAL_FEATURE_NAMES,
        )

        # Meta-labelling secondary classifier over the pooled dataset.
        # Same off-loading pattern as primary fit.  Uses universal_max_hold
        # (longest applicable horizon) as the embargo for OOF folds —
        # conservatively wider than mixed-kind data needs, but cheaper than
        # tracking per-ticker boundaries.
        meta_threshold = float(getattr(self.settings, "META_LABEL_THRESHOLD", 0.55))
        primary_min_conf = float(getattr(self.settings, "META_PRIMARY_MIN_CONF", 0.5))
        meta_model = await asyncio.to_thread(
            self._train_meta_model,
            all_X,
            all_y,
            all_w,
            universal_max_hold,
            primary_min_conf,
            meta_threshold,
        )
        if meta_model is not None:
            model.set_meta_model(meta_model)
            logger.info(
                f"Universal meta model attached: "
                f"acc={meta_model.metadata.accuracy:.3f}, "
                f"f1={meta_model.metadata.f1:.3f}, "
                f"n={meta_model.metadata.n_train}, "
                f"thr={meta_model.threshold:.2f}"
            )

        model_path = self.settings.MODELS_DIR / f"universal_{version}.joblib"
        await asyncio.to_thread(model.save, model_path)

        await self.db.register_model(
            figi=None,
            model_path=str(model_path),
            version=version,
            accuracy=metrics["accuracy"],
            f1_score=metrics["f1"],
            train_samples=len(X_train),
            feature_names=FEATURE_NAMES,
            tb_max_hold=universal_max_hold,
        )

        return model
