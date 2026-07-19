"""Macro context features: broad-market and external-factor signals.

The core per-instrument features describe price action *within* a single
instrument — indicators, returns, volume.  But a move in SBER often has
nothing to do with SBER itself: the whole market is selling on an oil
crash or a RUB devaluation.  Without macro context the model tries to
reconstruct these broad factors from noisy within-ticker signals and
loses information.

This module pulls candles for a handful of macro drivers (MOEX index,
BRENT oil, USD/RUB, Russian bond index) and exposes them for
`analysis.features.build_features` to turn into short/medium-horizon
return features aligned to the target instrument's time axis.

Each macro instrument contributes 2 returns (1-bar and 5-bar), so
4 drivers × 2 horizons = 8 new features.

**Graceful degradation:** if an instrument can't be resolved (find_instrument
returns nothing) or candles fail to fetch, that column is zero-filled and a
warning is logged once.  Training and inference still work, just without
that particular macro factor — the FEATURE_NAMES schema stays stable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from t_tech.invest import CandleInterval

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Macro instrument resolution.
#
# Each entry: (macro_name, list of search queries in priority order).
# MacroProvider._resolve_figi tries queries in order and uses the first that
# `broker.find_instrument` returns a hit for.  Exact ticker match is preferred
# over fuzzy matches.  BRENT futures rotate monthly — we rely on Tinkoff's
# search returning the active contract; if it fails, the feature simply stays
# zero-filled for that session.
# ---------------------------------------------------------------------------
MACRO_CONFIG: list[tuple[str, list[str]]] = [
    ("imoex", ["IMOEXF", "IMOEX", "TMOS"]),  # MOEX index (futures / index / ETF proxy)
    ("brent", ["LCOc1", "BRENT", "BR"]),  # BRENT oil (Reuters LCOc1 nearest future)
    ("usdrub", ["USDRUBF", "USD000UTSTOM", "USDRUB"]),  # USD/RUB (futures / spot)
    ("rgbi", ["RGBI", "SBGB", "OFZB"]),  # Russian govt bond index / ETF proxy
]


# Feature names produced by build_features for each macro driver.
#
# Kept DETERMINISTIC (not derived from MACRO_CONFIG at runtime) so FEATURE_NAMES
# is stable across runs.  Even if a macro fails to resolve the column is still
# emitted (zero-filled) — this way a model trained with all 8 macros can still
# score against partial data in production without schema errors.
MACRO_FEATURE_NAMES: list[str] = [
    "imoex_return_1",
    "imoex_return_5",
    "brent_return_1",
    "brent_return_5",
    "usdrub_return_1",
    "usdrub_return_5",
    "rgbi_return_1",
    "rgbi_return_5",
]


class MacroProvider:
    """Caching provider for macro candles.

    One instance is shared across strategy + trainer + tuner.  It caches:
    * FIGI resolution (forever per-process — macro tickers don't change)
    * Candle fetches (15-min TTL per (name, interval) pair)

    Within a single autonomous scan all tickers request the same macro window,
    so the candle cache gives us exactly one API call per macro per scan.
    """

    CACHE_TTL_SECONDS = 900  # 15 min — enough for one scan cycle

    def __init__(self, broker):
        self.broker = broker
        self._figi_cache: dict[str, Optional[str]] = {}
        # (name, interval) -> (fetched_at, df)   df has DatetimeIndex, 'close' column
        self._candle_cache: dict[tuple, tuple[datetime, pd.DataFrame]] = {}
        self._warned_missing: set[str] = set()

    async def _resolve_figi(self, name: str, queries: list[str]) -> Optional[str]:
        if name in self._figi_cache:
            return self._figi_cache[name]

        for q in queries:
            try:
                instruments = await self.broker.find_instrument(q)
            except Exception as e:
                logger.debug(f"Macro resolve failed for {name}/{q}: {e}")
                continue
            if not instruments:
                continue

            # Prefer exact ticker match; fall back to the first returned hit.
            exact = [i for i in instruments if getattr(i, "ticker", "") == q]
            chosen = exact[0] if exact else instruments[0]
            figi = getattr(chosen, "figi", None)
            if not figi:
                continue

            ticker = getattr(chosen, "ticker", "?")
            kind = getattr(chosen, "instrument_kind", "?")
            logger.info(f"Macro {name}: resolved to {ticker} ({figi}) kind={kind} via query '{q}'")
            self._figi_cache[name] = figi
            return figi

        # All queries exhausted without a hit.
        if name not in self._warned_missing:
            logger.warning(
                f"Macro {name}: no instrument found (queries tried: {queries}); "
                f"feature will be zero-filled"
            )
            self._warned_missing.add(name)
        self._figi_cache[name] = None
        return None

    async def get_macro_df(
        self, from_dt: datetime, to_dt: datetime, interval: CandleInterval
    ) -> pd.DataFrame:
        """Fetch macro close prices over [from_dt, to_dt] at the given interval.

        Returns DataFrame with DatetimeIndex (UTC) and one `<name>_close`
        column per successfully resolved macro.  Empty DataFrame if no macros
        resolved.  Caller is responsible for aligning to its own time axis.
        """
        # Import here to avoid circular import at module load time
        from analysis.screener import _candles_to_df

        now = datetime.now(timezone.utc)
        frames: list[pd.Series] = []

        for name, queries in MACRO_CONFIG:
            figi = await self._resolve_figi(name, queries)
            if not figi:
                continue

            key = (name, interval)
            cached = self._candle_cache.get(key)
            df: Optional[pd.DataFrame] = None

            if cached:
                fetched_at, cached_df = cached
                if (now - fetched_at).total_seconds() < self.CACHE_TTL_SECONDS:
                    df = cached_df

            if df is None:
                try:
                    candles = await self.broker.get_candles(figi, from_dt, to_dt, interval=interval)
                except Exception as e:
                    logger.debug(f"Macro {name}: candle fetch failed ({e})")
                    # Cache empty to avoid hammering on every ticker in the scan
                    self._candle_cache[key] = (now, pd.DataFrame())
                    continue

                if not candles:
                    logger.debug(f"Macro {name}: no candles returned for interval {interval}")
                    self._candle_cache[key] = (now, pd.DataFrame())
                    continue

                raw = _candles_to_df(candles)
                if "time" not in raw.columns or "close" not in raw.columns:
                    self._candle_cache[key] = (now, pd.DataFrame())
                    continue

                df = raw.copy()
                df["time"] = pd.to_datetime(df["time"], utc=True)
                df = df.set_index("time").sort_index()
                self._candle_cache[key] = (now, df)

            if df is None or df.empty or "close" not in df.columns:
                continue

            frames.append(df["close"].rename(f"{name}_close"))

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=1).sort_index()
