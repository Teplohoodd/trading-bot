"""Backtest the 'edge' mechanics on historical FORTS candles.

What we CAN test offline (using REST `get_candles`):
  * CVD divergence — by approximating buy/sell volume from
    candle (close, open, high, low) using the "trade location" heuristic:
        buy_vol_share = (close - low) / (high - low)   [0..1]
    This is a known approximation used in retail tools when L2 isn't
    available.  Not perfect (real TFI requires tick data) but good enough
    to test whether the divergence concept produces directional edge.
  * Lead-lag — by computing cross-correlation between log-returns of
    paired instruments at various lags.  Lag with peak |corr| is the
    candidate "leader window".  Out-of-sample test: does the leader's
    last move predict the lagger's next move?
  * Cointegration of pairs — for pair-trading edge.  We test if any
    pair of (BR, GZ, SR, LK, MX, Si) is cointegrated on 1h bars and
    if mean-reversion of the spread is profitable.

What we CANNOT test offline:
  * Real Open Interest delta (we don't have historical OI per trade,
    only aggregated daily values via instruments service).
  * True trade-flow imbalance at tick resolution.

For ALL untestable items we mark NOT VERIFIED in the report — only
deploy what we've actually measured.

Usage:
    python -m futbot.scripts.edge_backtest                 # all 3 tests
    python -m futbot.scripts.edge_backtest cvd             # one of:
    python -m futbot.scripts.edge_backtest leadlag         #   cvd / leadlag /
    python -m futbot.scripts.edge_backtest coint           #   coint
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import Settings  # noqa: E402
from core.broker import BrokerClient  # noqa: E402
from tinkoff.invest import CandleInterval  # noqa: E402
from tinkoff.invest.utils import quotation_to_decimal  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("tinkoff").setLevel(logging.WARNING)
logger = logging.getLogger("edge_backtest")


TIER1_BASES = ["BR", "GZ", "SR", "LK", "MX", "Si"]


# ─────────────────────────────────────────────────────────────────────────────
# Data fetch
# ─────────────────────────────────────────────────────────────────────────────
async def _resolve_front_month(broker, base: str):
    """Return the front-month future for the base, with at least 14d to expiry."""
    futs = await broker.get_all_futures()
    cands = []
    for f in futs:
        t = getattr(f, "ticker", "") or ""
        if t == base or (t.startswith(base) and len(t) == len(base) + 2):
            exp = getattr(f, "expiration_date", None)
            if exp is None:
                continue
            if hasattr(exp, "ToDatetime"):
                exp = exp.ToDatetime()
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            cands.append((f, exp))
    if not cands:
        return None, None
    cands.sort(key=lambda x: x[1])
    now = datetime.now(timezone.utc)
    for f, exp in cands:
        if (exp - now).days >= 14:
            return f, exp
    return cands[0][0], cands[0][1]


def _candles_to_df(candles) -> pd.DataFrame:
    rows = [
        {
            "time": c.time,
            "open": float(quotation_to_decimal(c.open)),
            "high": float(quotation_to_decimal(c.high)),
            "low": float(quotation_to_decimal(c.low)),
            "close": float(quotation_to_decimal(c.close)),
            "volume": int(c.volume),
        }
        for c in candles
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


async def _fetch_bases(
    broker, bases: list[str], days: int, interval=CandleInterval.CANDLE_INTERVAL_HOUR
) -> dict[str, pd.DataFrame]:
    out = {}
    now = datetime.now(timezone.utc)
    for base in bases:
        f, _ = await _resolve_front_month(broker, base)
        if f is None:
            logger.warning(f"  {base}: no contract found")
            continue
        try:
            candles = await broker.get_candles(
                f.figi,
                now - timedelta(days=days),
                now,
                interval=interval,
            )
            df = _candles_to_df(candles)
            if len(df) > 50:
                out[base] = df
                logger.info(f"  {base} ({f.ticker}): {len(df)} bars")
        except Exception as e:
            logger.warning(f"  {base}: fetch failed ({e})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — CVD divergence (approximation)
# ─────────────────────────────────────────────────────────────────────────────
def _cvd_from_candles(df: pd.DataFrame) -> pd.Series:
    """Approximate signed volume from OHLCV: trade location heuristic.

    Each bar contributes to CVD according to where its CLOSE sits within
    the (low, high) range:
        buy_share = (close - low) / (high - low)       in [0, 1]
        signed_vol = volume × (2 × buy_share - 1)      in [-vol, +vol]
        cvd[t] = cumsum(signed_vol)
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    buy_share = (df["close"] - df["low"]) / rng
    buy_share = buy_share.fillna(0.5).clip(0, 1)
    signed = df["volume"] * (2 * buy_share - 1)
    return signed.cumsum()


def test_cvd_divergence(df: pd.DataFrame, *, lookback: int = 24, fwd: int = 6) -> dict:
    """Run a vectorised "divergence" backtest on hourly candles.

    For each bar t > lookback we ask:
      * Is current close within 0.2% of the LOWEST low in [t-lookback, t]?
      * If yes, is CVD[t] greater than CVD at that low?  → BULL divergence
        Forward return = (close[t+fwd] − close[t]) / close[t]
      * Mirror for highs → BEAR divergence

    Returns counts and average forward return per signal.
    """
    n = len(df)
    if n < lookback + fwd + 5:
        return {"n_bars": n, "error": "too few bars"}
    close = df["close"].values
    cvd = _cvd_from_candles(df).values

    bull_signals = []
    bear_signals = []
    for t in range(lookback, n - fwd):
        window = slice(t - lookback, t + 1)
        lo_idx = window.start + int(np.argmin(close[window]))
        hi_idx = window.start + int(np.argmax(close[window]))
        # BULLISH: price near recent low + CVD higher than at that low
        if close[t] <= close[lo_idx] * 1.002 and cvd[t] > cvd[lo_idx]:
            fwd_ret = (close[t + fwd] - close[t]) / close[t]
            bull_signals.append(fwd_ret)
        # BEARISH
        if close[t] >= close[hi_idx] * 0.998 and cvd[t] < cvd[hi_idx]:
            fwd_ret = (close[t + fwd] - close[t]) / close[t]
            bear_signals.append(fwd_ret)

    out = {
        "n_bars": n,
        "lookback": lookback,
        "fwd": fwd,
        "bull_signals": len(bull_signals),
        "bear_signals": len(bear_signals),
    }
    if bull_signals:
        out["bull_avg_ret_pct"] = round(float(np.mean(bull_signals)) * 100, 4)
        out["bull_hit_rate"] = round(float(np.mean([r > 0 for r in bull_signals])), 3)
    if bear_signals:
        out["bear_avg_ret_pct"] = round(float(np.mean(bear_signals)) * 100, 4)
        out["bear_hit_rate"] = round(float(np.mean([r < 0 for r in bear_signals])), 3)
    # Edge net: average return × direction sign minus typical round-trip cost (0.08%)
    if bull_signals and bear_signals:
        avg_edge = (np.mean(bull_signals) - np.mean(bear_signals)) / 2 * 100
        out["avg_edge_pct"] = round(float(avg_edge), 4)
        out["edge_net_of_commission_pct"] = round(float(avg_edge - 0.08), 4)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Lead-lag correlations
# ─────────────────────────────────────────────────────────────────────────────
def test_lead_lag(dfs: dict[str, pd.DataFrame], max_lag: int = 5) -> dict:
    """For every ordered pair (A, B), compute corr(ret_A[t], ret_B[t+k]) for
    k in [-max_lag, +max_lag].  A "leader" is one whose past returns
    correlate with B's future returns more than the reverse direction.

    Reports: pairs with peak |corr| > 0.05 and significant directional skew.
    """
    # Align all on common timestamps
    bases = list(dfs.keys())
    if len(bases) < 2:
        return {"error": "need >= 2 series"}
    series_list = [dfs[b].set_index("time")["close"].rename(b) for b in bases]
    common = pd.concat(series_list, axis=1, join="inner").dropna()
    if len(common) < 100:
        return {"error": f"too few aligned bars ({len(common)})"}

    rets = np.log(common).diff().dropna()
    results = []
    for a in bases:
        for b in bases:
            if a == b:
                continue
            best_k, best_c = 0, 0.0
            for k in range(-max_lag, max_lag + 1):
                if k == 0:
                    continue
                if k > 0:
                    c = rets[a].iloc[:-k].corr(rets[b].iloc[k:])
                else:
                    c = rets[a].iloc[-k:].corr(rets[b].iloc[:k])
                if c is None or np.isnan(c):
                    continue
                if abs(c) > abs(best_c):
                    best_k, best_c = k, float(c)
            results.append(
                {
                    "leader": a,
                    "lagger": b,
                    "best_lag_bars": best_k,
                    "corr": round(best_c, 4),
                }
            )

    # Filter to interesting: |corr| > 0.05 AND leader→lagger direction
    interesting = [r for r in results if abs(r["corr"]) >= 0.05 and r["best_lag_bars"] > 0]
    interesting.sort(key=lambda r: -abs(r["corr"]))
    return {
        "n_aligned_bars": len(rets),
        "all_pairs": results,
        "interesting": interesting[:10],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Pair cointegration (mean reversion)
# ─────────────────────────────────────────────────────────────────────────────
def test_cointegration(dfs: dict[str, pd.DataFrame]) -> dict:
    """For every pair (A, B), run the Engle-Granger 2-step:
      1. OLS regression: A_t = β × B_t + α + ε_t
      2. ADF test on residuals ε_t → stationary residual = cointegration

    Reports pairs with ADF p-value < 0.05.  Then for each cointegrated
    pair, backtest a simple mean-reversion of the residual:
       z = (ε_t − mean(ε)) / std(ε)
       go LONG_A SHORT_B when z < -2; close when z crosses 0
       go SHORT_A LONG_B when z > +2; close when z crosses 0
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return {"error": "install statsmodels for cointegration tests"}

    bases = list(dfs.keys())
    if len(bases) < 2:
        return {"error": "need >= 2 series"}
    series_list = [dfs[b].set_index("time")["close"].rename(b) for b in bases]
    common = pd.concat(series_list, axis=1, join="inner").dropna()
    if len(common) < 100:
        return {"error": f"too few aligned bars ({len(common)})"}

    results = []
    for i, a in enumerate(bases):
        for b in bases[i + 1 :]:
            y = common[a].values
            x = common[b].values
            # OLS through statsmodels — but we can do it with numpy
            beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
            alpha = y.mean() - beta * x.mean()
            resid = y - (beta * x + alpha)
            try:
                adf = adfuller(resid, maxlag=5, autolag=None)
                pval = float(adf[1])
            except Exception:
                continue
            entry = {
                "pair": f"{a}-{b}",
                "beta": round(beta, 4),
                "adf_p": round(pval, 4),
                "n": len(y),
            }
            # Backtest if cointegrated
            if pval < 0.10:
                z = (resid - resid.mean()) / resid.std()
                trades = _backtest_spread(z=z, y=y, x=x, beta=beta)
                entry.update(trades)
            results.append(entry)
    results.sort(key=lambda r: r["adf_p"])
    return {"n_aligned_bars": len(common), "pairs": results}


def _backtest_spread(*, z: np.ndarray, y: np.ndarray, x: np.ndarray, beta: float) -> dict:
    """Trade the spread when |z| > 2.  Exit when z crosses 0.

    P&L is normalised by the COMBINED notional of the two legs at entry —
    long-y notional + short-βx notional = |y| + |β|·|x|.  This avoids
    the small-β explosion bug where dividing by |spread| at entry near
    zero gave absurd percentages.

    Also includes a 4×0.04 % = 0.16 % round-trip cost per trade (two legs,
    open+close each).
    """
    pos = 0  # +1 = long y short x;  -1 = short y long x
    entry_idx = None
    pnls = []
    holding_bars = []
    commission_rt = 0.0016  # 0.16 % round trip for the pair
    for t in range(1, len(z)):
        if pos == 0:
            if z[t] > 2:
                pos, entry_idx = -1, t
            elif z[t] < -2:
                pos, entry_idx = +1, t
        else:
            crossed = (pos == +1 and z[t] >= 0) or (pos == -1 and z[t] <= 0)
            if crossed:
                sp_entry = y[entry_idx] - beta * x[entry_idx]
                sp_exit = y[t] - beta * x[t]
                combined_notional = abs(y[entry_idx]) + abs(beta) * abs(x[entry_idx])
                gross_pnl_pct = (
                    pos * (sp_exit - sp_entry) / combined_notional if combined_notional > 0 else 0.0
                )
                net_pnl_pct = gross_pnl_pct - commission_rt
                pnls.append(net_pnl_pct)
                holding_bars.append(t - entry_idx)
                pos, entry_idx = 0, None

    if not pnls:
        return {"n_trades": 0}
    return {
        "n_trades": len(pnls),
        "win_rate": round(float(np.mean([p > 0 for p in pnls])), 3),
        "avg_pnl_pct": round(float(np.mean(pnls)) * 100, 4),
        "total_pnl_pct": round(float(np.sum(pnls)) * 100, 4),
        "avg_hold_bars": round(float(np.mean(holding_bars)), 1),
        "best_trade_pct": round(float(np.max(pnls)) * 100, 4),
        "worst_trade_pct": round(float(np.min(pnls)) * 100, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 90

    settings = Settings()
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="edge-backtest",
    )
    await broker.connect()

    logger.info(f"Fetching {days}d of hourly candles for {TIER1_BASES}…")
    dfs = await _fetch_bases(broker, TIER1_BASES, days=days)
    await broker.disconnect()
    if not dfs:
        logger.error("No data fetched.  Exiting.")
        return

    print()
    print("=" * 90)
    print(f"BACKTEST RESULTS — {days}d × hourly candles × {len(dfs)} contracts")
    print("=" * 90)

    if which in ("all", "cvd"):
        print()
        print("─── Test 1: CVD divergence (per instrument) ───")
        print(
            f"{'base':<6} {'n_bars':>7} {'bull_n':>7} {'bull_ret%':>10} "
            f"{'bull_wr':>8} {'bear_n':>7} {'bear_ret%':>10} {'bear_wr':>8} "
            f"{'edge_net%':>10}"
        )
        for b, df in dfs.items():
            r = test_cvd_divergence(df)
            if "error" in r:
                print(f"  {b:<6} {r['error']}")
                continue
            print(
                f"  {b:<6} {r['n_bars']:>7} "
                f"{r.get('bull_signals', 0):>7} "
                f"{r.get('bull_avg_ret_pct', 0):>+10.4f} "
                f"{r.get('bull_hit_rate', 0):>8.3f} "
                f"{r.get('bear_signals', 0):>7} "
                f"{r.get('bear_avg_ret_pct', 0):>+10.4f} "
                f"{r.get('bear_hit_rate', 0):>8.3f} "
                f"{r.get('edge_net_of_commission_pct', 0):>+10.4f}"
            )
        print()
        print(
            "  How to read:  edge_net% > 0 means signal predicts direction "
            "even after Tinkoff Trader 0.08% round-trip."
        )

    if which in ("all", "leadlag"):
        print()
        print("─── Test 2: Lead-lag cross-instrument ───")
        r = test_lead_lag(dfs)
        if "error" in r:
            print(f"  {r['error']}")
        else:
            print(f"  Aligned bars: {r['n_aligned_bars']}")
            print(f"  Top 10 leader→lagger pairs by |corr|:")
            print(f"  {'leader':<6} → {'lagger':<6}  {'lag(bars)':>10}  {'corr':>8}")
            for p in r["interesting"]:
                print(
                    f"  {p['leader']:<6} → {p['lagger']:<6}  "
                    f"{p['best_lag_bars']:>10}  {p['corr']:>+8.4f}"
                )
            if not r["interesting"]:
                print(
                    "  (no pair has |corr| ≥ 0.05 — lead-lag not exploitable "
                    "at hourly resolution; might still work at 5m or 1m)"
                )

    if which in ("all", "coint"):
        print()
        print("─── Test 3: Cointegration + spread backtest ───")
        r = test_cointegration(dfs)
        if "error" in r:
            print(f"  {r['error']}")
        else:
            print(f"  Aligned bars: {r['n_aligned_bars']}")
            print(
                f"  {'pair':<10} {'beta':>8} {'adf_p':>8} {'n_tr':>5} "
                f"{'win%':>6} {'avg%':>7} {'total%':>8} {'best':>7} {'worst':>7} {'avg_hold':>9}"
            )
            for p in r["pairs"]:
                if "n_trades" in p and p["n_trades"] > 0:
                    print(
                        f"  {p['pair']:<10} {p['beta']:>+8.3f} {p['adf_p']:>8.4f} "
                        f"{p['n_trades']:>5} "
                        f"{p.get('win_rate', 0)*100:>5.1f}% "
                        f"{p.get('avg_pnl_pct', 0):>+7.3f} "
                        f"{p.get('total_pnl_pct', 0):>+8.3f} "
                        f"{p.get('best_trade_pct', 0):>+7.3f} "
                        f"{p.get('worst_trade_pct', 0):>+7.3f} "
                        f"{p.get('avg_hold_bars', 0):>9.1f}"
                    )
                else:
                    print(
                        f"  {p['pair']:<10} {p['beta']:>+8.3f} {p['adf_p']:>8.4f} "
                        f"(not cointegrated)"
                    )
            print()
            print(
                "  How to read: adf_p < 0.05 = significant cointegration. "
                "total% is the cumulative gross return of the spread strategy."
            )

    print()
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
