"""Autonomous ticker screener/ranker for MOEX shares."""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pandas as pd

from core.broker import BrokerClient
from analysis.indicators import compute_indicators
from utils.helpers import quotation_to_decimal

logger = logging.getLogger(__name__)


def _candles_to_df(candles: list) -> pd.DataFrame:
    """Convert tinkoff candle objects to DataFrame."""
    rows = []
    for c in candles:
        rows.append(
            {
                "time": c.time.isoformat(),
                "open": float(quotation_to_decimal(c.open)),
                "high": float(quotation_to_decimal(c.high)),
                "low": float(quotation_to_decimal(c.low)),
                "close": float(quotation_to_decimal(c.close)),
                "volume": c.volume,
            }
        )
    return pd.DataFrame(rows)


class Screener:
    """Scans MOEX shares, scores by liquidity + volatility + momentum."""

    def __init__(
        self,
        broker: BrokerClient,
        top_n: int = 30,
        include_futures: bool = True,
        futures_min_days_to_expiry: int = 14,
    ):
        self.broker = broker
        self.top_n = top_n
        self.include_futures = include_futures
        self.futures_min_days_to_expiry = futures_min_days_to_expiry

    async def _collect_universe(self, direction: str) -> list[dict]:
        """Collect shares (+ optionally futures) and normalise to uniform candidate dicts.

        Returns list of dicts:
          {instrument, figi, ticker, name, lot, kind, short_ok}
        where instrument is the raw API object (used only for metadata).
        """
        all_shares = await self.broker.get_all_shares()
        now = datetime.now(timezone.utc)

        candidates: list[dict] = []
        for s in all_shares:
            if s.currency != "rub":
                continue
            if not (s.api_trade_available_flag and s.buy_available_flag and s.sell_available_flag):
                continue
            if s.otc_flag or s.lot <= 0:
                continue
            short_ok = bool(getattr(s, "short_enabled_flag", False))
            if direction == "short" and not short_ok:
                continue
            candidates.append(
                {
                    "instrument": s,
                    "figi": s.figi,
                    "ticker": s.ticker,
                    "name": s.name,
                    "lot": s.lot,
                    "kind": "share",
                    "short_ok": short_ok,
                    "rub_per_point": 1.0,  # shares: 1 price-unit = 1 RUB
                }
            )

        if self.include_futures:
            try:
                all_futures = await self.broker.get_all_futures()
            except Exception as e:
                logger.warning(f"get_all_futures failed: {e}")
                all_futures = []
            from config.instruments import LIQUID_FUTURES_PREFIXES

            prefixes = tuple(LIQUID_FUTURES_PREFIXES)
            for f in all_futures:
                try:
                    if f.currency != "rub":
                        continue
                    if not (
                        f.api_trade_available_flag
                        and f.buy_available_flag
                        and f.sell_available_flag
                    ):
                        continue
                    # Skip futures without a public liquid ticker prefix
                    ticker = getattr(f, "ticker", "") or ""
                    if not ticker.startswith(prefixes):
                        continue
                    # Skip near-expiry
                    exp = getattr(f, "expiration_date", None)
                    if exp is not None:
                        try:
                            days_to_exp = (exp - now).days
                            if days_to_exp < self.futures_min_days_to_expiry:
                                continue
                        except Exception:
                            pass
                    # Extract futures metadata up-front (initial_margin,
                    # step_value, rub_per_point) so downstream scoring &
                    # sizing don't conflate contract notional with capital
                    # requirement.  See core.broker.extract_futures_metadata.
                    meta = self.broker.extract_futures_metadata(f)
                    rpp = float(meta.get("rub_per_point", 1) or 1)
                    # Futures have no overnight short carry — always shortable
                    candidates.append(
                        {
                            "instrument": f,
                            "figi": f.figi,
                            "ticker": ticker,
                            "name": getattr(f, "name", ticker),
                            "lot": getattr(f, "lot", 1) or 1,
                            "kind": "future",
                            "short_ok": True,
                            "initial_margin_buy": float(meta.get("initial_margin_buy", 0) or 0),
                            "initial_margin_sell": float(meta.get("initial_margin_sell", 0) or 0),
                            "step_value": float(meta.get("step_value", 0) or 0),
                            "min_price_increment": float(meta.get("min_price_increment", 0) or 0),
                            "rub_per_point": rpp if rpp > 0 else 1.0,
                            "expiration_date": meta.get("expiration_date"),
                            "asset_type": meta.get("asset_type", ""),
                            "basic_asset": meta.get("basic_asset", ""),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Skip future {getattr(f, 'ticker', '?')}: {e}")
                    continue

        return candidates

    async def scan_universe(
        self, custom_figis: list[str] | None = None, direction: str = "long"
    ) -> list[dict]:
        """Scan MOEX shares (+ optionally futures) and return scored candidates.

        Args:
            custom_figis: List of FIGIs to always include regardless of score.
            direction: "long" — rank by positive momentum (buy candidates);
                       "short" — rank by negative momentum (sell candidates).

        Returns list of dicts: {figi, ticker, name, score, avg_volume, atr_pct, momentum, kind}
        sorted by composite score descending.
        """
        logger.info(
            f"Starting universe scan (direction={direction}, futures={self.include_futures})..."
        )
        custom_figis = set(custom_figis or [])
        candidates = await self._collect_universe(direction)
        logger.info(f"Found {len(candidates)} tradeable candidates (direction={direction})")

        scored = []
        now = datetime.now(timezone.utc)
        from_dt = now - timedelta(days=30)

        for cand in candidates:
            try:
                figi = cand["figi"]
                ticker = cand["ticker"]
                name = cand["name"]
                lot = cand["lot"]
                kind = cand["kind"]
                candles = await self.broker.get_candles(
                    figi,
                    from_dt,
                    now,
                )
                if len(candles) < 10:
                    continue

                df = _candles_to_df(candles)
                df_ind = compute_indicators(df)

                avg_volume = df_ind["volume"].mean()
                last_close = df_ind["close"].iloc[-1]

                # Average hourly volume in RUB.
                #
                # Shares: price is already RUB/share, so
                #   avg_vol_rub = contracts_per_bar × price × lot_size
                #
                # Futures: price is in POINTS, not RUB.  Multiplying by price
                # silently gave wrong numbers (Si ≈ right by accident, BR/GD
                # 10-100× off).  Correct conversion uses rub_per_point
                # (= step_value / min_price_increment) from the Future SDK
                # object — RUB value of moving 1 full point of price.
                if kind == "future":
                    rpp = float(cand.get("rub_per_point", 1.0) or 1.0)
                    avg_vol_rub = avg_volume * last_close * rpp * lot
                else:
                    avg_vol_rub = avg_volume * last_close * lot

                # Futures on MOEX have inherently lower notional per hour but
                # tighter spreads — accept 500k/h floor for futures, 1M/h for shares.
                min_liquidity = 500_000 if kind == "future" else 1_000_000
                if avg_vol_rub < min_liquidity:
                    continue

                # ATR as % of price (volatility)
                atr_val = df_ind["atr_14"].iloc[-1] or 0
                atr_pct = float(atr_val / last_close * 100) if last_close > 0 else 0
                # Median ATR% over the last 60 bars — used by the ATR filter
                # in _scan_cycle to skip entries when vol is below 80% of
                # its recent median (low-vol = noisy model).
                atr_series = df_ind["atr_14"].dropna()
                close_series = df_ind["close"].dropna()
                atr_pct_series = (atr_series / close_series * 100).where(close_series > 0, 0)
                atr_pct_median60 = float(
                    atr_pct_series.tail(60).median() if len(atr_pct_series) >= 10 else 0
                )

                # RSI(14) — measures overbought/oversold
                rsi14 = float(df_ind["rsi_14"].iloc[-1] or 50)

                # ADX(14) — trend strength (>25 = trending, <20 = ranging)
                adx = float(df_ind["adx_14"].iloc[-1] or 0)

                # 20-bar momentum (hourly, ~20 trading hours ≈ 2-3 days)
                momentum = (
                    float(df_ind["close"].pct_change(20).iloc[-1] * 100) if len(df_ind) >= 20 else 0
                )

                # --- Composite score ---
                # 1. Volume weight (log scale)
                volume_score = np.log10(max(avg_vol_rub, 1))

                # 2. Directional momentum — rewards alignment, PENALISES counter-trend
                #    Long:  +6.8% → 1.68;  -3.2% → 0.68 (floor 0.5)
                #    Short:  -3.2% → 1.32;  +6.8% → 0.32 (floor 0.5)
                if direction == "short":
                    directional_boost = max(0.5, 1.0 - momentum / 10)
                else:
                    directional_boost = max(0.5, 1.0 + momentum / 10)

                # 3. RSI entry-timing quality
                #    Long: penalise RSI > 70 (overbought = bad entry timing)
                #    Short: penalise RSI < 30 (oversold = bad entry timing)
                #    Uses a linear ramp: full quality at RSI 50-65 long / 35-50 short
                if direction == "short":
                    rsi_quality = max(0.25, 1.0 - max(0.0, (40.0 - rsi14) / 40.0))
                else:
                    rsi_quality = max(0.25, 1.0 - max(0.0, (rsi14 - 65.0) / 35.0))

                # 4. ADX trend bonus (stronger trend = higher confidence)
                trend_bonus = 1.0 + min(adx, 50) / 100 if adx > 20 else 0.85

                score = (
                    volume_score * max(atr_pct, 0.1) * directional_boost * rsi_quality * trend_bonus
                )

                # Tag overbought/oversold for display
                rsi_tag = ""
                if direction == "long" and rsi14 > 70:
                    rsi_tag = "OB"  # overbought
                elif direction == "short" and rsi14 < 30:
                    rsi_tag = "OS"  # oversold

                scored.append(
                    {
                        "figi": figi,
                        "ticker": ticker,
                        "name": name,
                        "lot_size": lot,
                        "kind": kind,
                        "score": round(score, 2),
                        "avg_volume_rub": round(avg_vol_rub),
                        "atr_pct": round(atr_pct, 2),
                        "atr_pct_median60": round(atr_pct_median60, 3),
                        "momentum": round(momentum, 2),
                        "rsi_14": round(rsi14, 1),
                        "adx_14": round(adx, 1),
                        "rsi_tag": rsi_tag,
                        "last_price": last_close,
                        "direction": direction,
                        # Futures-only metadata (ignored by share path; empty for shares)
                        "initial_margin_buy": cand.get("initial_margin_buy", 0.0),
                        "initial_margin_sell": cand.get("initial_margin_sell", 0.0),
                        "rub_per_point": cand.get("rub_per_point", 1.0),
                        "expiration_date": cand.get("expiration_date"),
                        "asset_type": cand.get("asset_type", ""),
                        "basic_asset": cand.get("basic_asset", ""),
                    }
                )
            except Exception as e:
                logger.debug(f"Skip {cand.get('ticker', '?')}: {e}")
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[: self.top_n]

        # Always include custom tickers even if they didn't make the top-N
        top_figis = {c["figi"] for c in top}
        for c in scored:
            if c["figi"] in custom_figis and c["figi"] not in top_figis:
                c["custom"] = True
                top.append(c)
                top_figis.add(c["figi"])

        logger.info(f"Screener found {len(scored)} candidates, top {len(top)} selected")
        return top

    async def analyze_ticker(
        self, figi: str, ticker: str, ml_strategy=None, tech_strategy=None
    ) -> dict:
        """Deep analysis of a single ticker.

        Returns dict with price, indicators, ML prediction, technical signal,
        fundamentals, order book stats.
        """
        result: dict = {"figi": figi, "ticker": ticker, "error": None}

        now = datetime.now(timezone.utc)
        from_dt = now - timedelta(days=30)  # Same window as screener for consistency

        # 1. Candles + indicators
        try:
            candles = await self.broker.get_candles(figi, from_dt, now)
            if len(candles) < 20:
                result["error"] = "Not enough candles"
                return result
            df = _candles_to_df(candles)
            df_ind = compute_indicators(df)
            last = df_ind.iloc[-1]

            result["price"] = float(last["close"])
            result["indicators"] = {
                "rsi_14": round(float(last.get("rsi_14", 0) or 0), 1),
                "rsi_7": round(float(last.get("rsi_7", 0) or 0), 1),
                "macd": round(float(last.get("macd", 0) or 0), 4),
                "macd_signal": round(float(last.get("macd_signal", 0) or 0), 4),
                "macd_hist": round(float(last.get("macd_histogram", 0) or 0), 4),
                "bb_pct_b": round(float(last.get("bb_percent_b", 0) or 0), 3),
                "bb_width": round(float(last.get("bb_width", 0) or 0), 3),
                "atr_14": round(float(last.get("atr_14", 0) or 0), 4),
                "atr_pct": round(
                    float(last.get("atr_14", 0) or 0) / max(float(last["close"]), 0.01) * 100, 2
                ),
                "adx_14": round(float(last.get("adx_14", 0) or 0), 1),
                "stoch_k": round(float(last.get("stoch_k", 0) or 0), 1),
                "stoch_d": round(float(last.get("stoch_d", 0) or 0), 1),
                "ema_9": round(float(last.get("ema_9", 0) or 0), 2),
                "ema_21": round(float(last.get("ema_21", 0) or 0), 2),
                "ema_50": round(float(last.get("ema_50", 0) or 0), 2),
                "sma_200": round(float(last.get("sma_200", 0) or 0), 2),
            }
            result["momentum_5d"] = round(float(df["close"].pct_change(5).iloc[-1] * 100), 2)
            result["momentum_20d"] = round(float(df["close"].pct_change(20).iloc[-1] * 100), 2)
            avg_vol = df["volume"].mean()
        except Exception as e:
            result["error"] = f"Candles error: {e}"
            return result

        # 2. Order book
        try:
            ob = await self.broker.get_order_book(figi, depth=10)
            if ob.bids and ob.asks:
                from t_tech.invest.utils import quotation_to_decimal as q2d

                bid = float(q2d(ob.bids[0].price))
                ask = float(q2d(ob.asks[0].price))
                mid = (bid + ask) / 2
                spread_bps = (ask - bid) / mid * 10000 if mid > 0 else 0
                total_bid = sum(b.quantity for b in ob.bids)
                total_ask = sum(a.quantity for a in ob.asks)
                total = total_bid + total_ask
                imbalance = (total_bid - total_ask) / total if total > 0 else 0
                result["order_book"] = {
                    "bid": round(bid, 4),
                    "ask": round(ask, 4),
                    "spread_bps": round(spread_bps, 1),
                    "imbalance": round(imbalance, 3),
                    "bid_volume": total_bid,
                    "ask_volume": total_ask,
                }
        except Exception:
            result["order_book"] = None

        # 3. Fundamentals from T-Invest API (shares only — futures have no fundamentals)
        try:
            instrument, kind = await self.broker.get_instrument_info(figi)
            result["kind"] = kind
            result["lot_size"] = getattr(instrument, "lot", 1) if instrument else 1
            result["name"] = getattr(instrument, "name", ticker) if instrument else ticker
            result["sector"] = getattr(instrument, "sector", "") if instrument else ""
            if kind == "share" and instrument is not None:
                asset_uid = getattr(instrument, "asset_uid", None)
                if asset_uid:
                    fundamentals = await self.broker.get_fundamentals(asset_uid)
                    result["fundamentals"] = fundamentals
            else:
                result["fundamentals"] = {}
            avg_vol_rub = avg_vol * result["price"] * result["lot_size"]
            result["avg_volume_rub"] = round(avg_vol_rub)
        except Exception as e:
            logger.debug(f"Fundamentals error for {ticker}: {e}")
            result["fundamentals"] = {}

        # 4. Technical strategy signal
        if tech_strategy:
            try:
                from analysis.indicators import compute_indicators as ci

                ob_data = result.get("order_book") or {}
                sig = await tech_strategy.generate_signal(
                    figi,
                    ticker,
                    df_ind,
                    {
                        "spread_bps": ob_data.get("spread_bps", 0),
                        "imbalance": ob_data.get("imbalance", 0),
                    },
                )
                result["tech_signal"] = {
                    "direction": sig.direction,
                    "confidence": round(sig.confidence, 3),
                    "stop_pct": round(sig.suggested_stop_pct, 2),
                    "target_pct": round(sig.suggested_target_pct, 2),
                }
            except Exception as e:
                logger.debug(f"Tech signal error: {e}")
                result["tech_signal"] = None

        # 5. ML model signal
        if ml_strategy:
            try:
                ob_data = result.get("order_book") or {}
                sig = await ml_strategy.generate_signal(
                    figi,
                    ticker,
                    df_ind,
                    {
                        "spread_bps": ob_data.get("spread_bps", 0),
                        "imbalance": ob_data.get("imbalance", 0),
                    },
                )
                result["ml_signal"] = {
                    "direction": sig.direction,
                    "confidence": round(sig.confidence, 3),
                    "features": sig.features or {},
                }
            except Exception as e:
                logger.debug(f"ML signal error: {e}")
                result["ml_signal"] = None

        return result
