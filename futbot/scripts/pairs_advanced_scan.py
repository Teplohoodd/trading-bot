"""Advanced stat-arb pair scanner — beyond vanilla cointegration.

The existing `pairs_universe_expand.py` keeps any pair with ADF p < 0.10.
But statistical cointegration is necessary, NOT sufficient, for *tradeable*
mean reversion.  A spread can pass ADF yet revert so slowly (half-life >
weeks) that you'd never realise the reversion inside a sane holding window,
or be so close to a random walk that the "edge" is noise.

This scanner adds the canonical stat-arb quality gauntlet:

  1. Engle-Granger β/α + ADF p-value          (is it cointegrated at all?)
  2. Ornstein-Uhlenbeck half-life (hours)      (how FAST does it revert?)
       Δs_t = λ·s_{t-1} + ε  →  half_life = -ln(2)/λ
       Reject λ ≥ 0 (no reversion) or half_life outside [HL_MIN, HL_MAX].
  3. Hurst exponent (variance-ratio estimator)  (mean-revert vs trend vs RW?)
       H < 0.5 mean-reverting, ≈0.5 random walk, >0.5 trending.
  4. Rolling z-score mean-reversion backtest    (does it actually make money
       net of commission?)  → Sharpe, win-rate, total %, n_trades.

Composite quality score ranks survivors so the BEST pairs float to the top,
not just the ones that happen to have many trades.

Both directions of each pair are equivalent for a spread (y-βx vs x-(1/β)y),
so we only test the upper triangle.

Usage:
    python -u -m futbot.scripts.pairs_advanced_scan
    python -u -m futbot.scripts.pairs_advanced_scan --days 240 --top 25
"""

import argparse
import asyncio
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import Settings
from core.broker import BrokerClient
from t_tech.invest import CandleInterval
from t_tech.invest.utils import quotation_to_decimal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("t_tech").setLevel(logging.WARNING)
logger = logging.getLogger("pairs_adv")


# Broad liquid FORTS base set (front-month resolved at runtime).
BASES = [
    "BR",
    "GZ",
    "SR",
    "LK",
    "MX",
    "Si",
    "NG",
    "GD",
    "RT",
    "USDRU",
    "GLDRU",
    "EURRU",
    "CR",
    "PX",
    "YD",
    "VB",
    "TT",
    "RN",
    "GK",
    "MM",
    "PT",
    "LT",
    "S1",
    "MV",
    "SV",
]

# ── Quality thresholds ─────────────────────────────────────────────────
ADF_P_MAX = 0.10
HL_MIN_H = 2.0  # reject if reverts faster than 2h (likely microstructure noise)
HL_MAX_H = 240.0  # reject if half-life > 10 days (won't revert in 48h hold)
HURST_MAX = 0.55  # allow slight slack above 0.5 (estimator is noisy)
MIN_TRADES = 6
DAYS = 180

# ── Tradeability guards (lessons from the MX-GK β=57 over-leverage bug) ──
# A huge |β| means the two contracts live on wildly different price scales;
# the β-hedge then demands an absurd lot ratio (β=57 → 11,545 lots) that
# either can't be filled or blows the margin budget.  Statistically these
# pass ADF, but they are NOT investable.  Cap |β| to a sane band.
BETA_ABS_MAX = 15.0
# NG (natural gas) is quoted in USD/mmBtu (~3) while our pipeline assumes
# RUB; its notional/rub_per_point can't be resolved correctly, so any NG
# pair is unsizable (see MIN_SANE_NOTIONAL guard in pairs/execution.py).
EXCLUDE_BASES = {"NG"}

# Backtest params (mirror live config)
COMMISSION_RT = 0.0016
Z_ENTRY = 2.0
Z_STOP = 4.0
ROLLING_Z_WINDOW = 240
MAX_HOLD = 48


# ════════════════════════════════════════════════════════════════════════
# Data fetch (front-month resolution + chunked hourly)
# ════════════════════════════════════════════════════════════════════════


async def _resolve_front_month(broker, base: str):
    futs = await broker.get_all_futures()
    cands = []
    now = datetime.now(timezone.utc)
    for f in futs:
        t = getattr(f, "ticker", "") or ""
        if not (t == base or (t.startswith(base) and len(t) == len(base) + 2)):
            continue
        exp = getattr(f, "expiration_date", None)
        if exp is None:
            continue
        if hasattr(exp, "ToDatetime"):
            exp = exp.ToDatetime()
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        cands.append((f, exp))
    if not cands:
        return None
    cands.sort(key=lambda x: x[1])
    for f, exp in cands:
        if (exp - now).days >= 14:
            return f
    return cands[0][0]


def _candles_to_df(candles) -> pd.DataFrame:
    rows = [{"time": c.time, "close": float(quotation_to_decimal(c.close))} for c in candles]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


async def _fetch_chunked(broker, figi: str, days: int) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    chunks = []
    end = now
    days_left = days
    while days_left > 0:
        chunk_days = min(days_left, 89)
        start = end - timedelta(days=chunk_days)
        try:
            candles = await broker.get_candles(
                figi,
                start,
                end,
                interval=CandleInterval.CANDLE_INTERVAL_HOUR,
            )
            df = _candles_to_df(candles)
            if not df.empty:
                chunks.append(df)
        except Exception:
            pass
        end = start
        days_left -= chunk_days
    if not chunks:
        return pd.DataFrame()
    return (
        pd.concat(chunks)
        .drop_duplicates("time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )


# ════════════════════════════════════════════════════════════════════════
# Stat-arb diagnostics
# ════════════════════════════════════════════════════════════════════════


def ou_half_life(spread: np.ndarray) -> float:
    """Ornstein-Uhlenbeck half-life of mean reversion, in bars (hours).

    Regress Δs_t on s_{t-1}:  Δs = λ·s_{t-1} + c + ε.
    If λ < 0 the process is mean-reverting; half_life = -ln(2)/λ.
    Returns np.inf when λ ≥ 0 (no reversion).
    """
    s = spread[:-1]
    ds = np.diff(spread)
    # OLS with intercept
    A = np.vstack([s, np.ones_like(s)]).T
    try:
        coef, *_ = np.linalg.lstsq(A, ds, rcond=None)
    except Exception:
        return np.inf
    lam = coef[0]
    if lam >= 0:
        return np.inf
    return float(-math.log(2.0) / lam)


def hurst_exponent(series: np.ndarray, max_lag: int = 60) -> float:
    """Hurst exponent via the variance-of-differences (variance-ratio) method.

    For lags τ, compute the std of (series_t - series_{t-τ}); the slope of
    log(std) vs log(τ) estimates H.  H<0.5 mean-reverting, 0.5 random walk,
    >0.5 trending.  Robust and dependency-free.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    max_lag = min(max_lag, n // 2)
    if max_lag < 4:
        return 0.5
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diff = series[lag:] - series[:-lag]
        sd = np.std(diff)
        if sd <= 0:
            continue
        tau.append((lag, sd))
    if len(tau) < 4:
        return 0.5
    log_lag = np.log([t[0] for t in tau])
    log_sd = np.log([t[1] for t in tau])
    slope = np.polyfit(log_lag, log_sd, 1)[0]
    return float(slope)


def backtest_spread(y: np.ndarray, x: np.ndarray, beta: float) -> dict:
    """Rolling-z mean-reversion backtest on the spread y - βx."""
    spread = y - beta * x
    n = len(spread)
    if n < ROLLING_Z_WINDOW + 50:
        return {"n_trades": 0}

    z = np.full(n, np.nan)
    for t in range(ROLLING_Z_WINDOW, n):
        w = spread[t - ROLLING_Z_WINDOW : t]
        sd = w.std()
        z[t] = (spread[t] - w.mean()) / sd if sd > 0 else 0.0

    pos = 0
    entry_idx = None
    pnls, holds = [], []
    for t in range(ROLLING_Z_WINDOW, n):
        if np.isnan(z[t]):
            continue
        if pos == 0:
            if z[t] > Z_ENTRY:
                pos, entry_idx = -1, t
            elif z[t] < -Z_ENTRY:
                pos, entry_idx = +1, t
            continue
        crossed_zero = (pos == +1 and z[t] >= 0) or (pos == -1 and z[t] <= 0)
        stopped = abs(z[t]) >= Z_STOP
        timed_out = (t - entry_idx) >= MAX_HOLD
        if not (crossed_zero or stopped or timed_out):
            continue
        sp_e = y[entry_idx] - beta * x[entry_idx]
        sp_x = y[t] - beta * x[t]
        combined = abs(y[entry_idx]) + abs(beta) * abs(x[entry_idx])
        gross = pos * (sp_x - sp_e) / combined if combined > 0 else 0
        pnls.append(gross - COMMISSION_RT)
        holds.append(t - entry_idx)
        pos = 0

    if not pnls:
        return {"n_trades": 0}
    arr = np.array(pnls)
    sharpe = arr.mean() / arr.std() * math.sqrt(len(arr)) if arr.std() > 0 else 0.0
    return {
        "n_trades": len(arr),
        "win_rate": float((arr > 0).mean()),
        "total_pct": float(arr.sum() * 100),
        "avg_pct": float(arr.mean() * 100),
        "sharpe": float(sharpe),
        "median_hold_h": float(np.median(holds)),
    }


def composite_score(c: dict) -> float:
    """Rank survivors.  Sharpe is primary; reward fast reversion + strong
    mean-reversion (low Hurst), lightly penalise too-few trades."""
    sharpe = c["sharpe"]
    hl = c["half_life"]
    hurst = c["hurst"]
    n = c["n_trades"]
    hl_bonus = max(0.0, 1.0 - hl / HL_MAX_H)  # 0..1, faster = better
    hurst_bonus = max(0.0, (0.5 - hurst) * 2)  # 0..1, lower = better
    sample_factor = min(1.0, n / 15.0)  # trust ~15+ trades
    return (sharpe * (1 + 0.3 * hl_bonus + 0.3 * hurst_bonus)) * (0.6 + 0.4 * sample_factor)


# ════════════════════════════════════════════════════════════════════════


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    settings = Settings()
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="pairs-adv",
    )
    await broker.connect()
    logger.info(f"Fetching {args.days}d hourly for {len(BASES)} bases…")

    series_list = []
    for base in BASES:
        f = await _resolve_front_month(broker, base)
        if f is None:
            logger.warning(f"  {base}: no contract")
            continue
        df = await _fetch_chunked(broker, f.figi, args.days)
        if len(df) < 500:
            logger.warning(f"  {base}: only {len(df)} bars — skip")
            continue
        series_list.append(df.set_index("time")["close"].rename(base))
        logger.info(f"  {base:6} ({f.ticker:8}) {len(df)} bars")
    await broker.disconnect()

    if len(series_list) < 2:
        logger.error("Not enough data")
        return

    # Keep series in a dict; align EACH PAIR independently (a joint inner-join
    # across all bases truncates everyone to the shortest series — biases the
    # scan and throws away most data).  Per-pair alignment uses the full
    # overlap of just those two contracts (~1900-2300 bars typically).
    by_base = {s.name: s for s in series_list}
    bases = list(by_base.keys())
    logger.info(f"Loaded {len(bases)} series; aligning per-pair. bases={bases}")

    from statsmodels.tsa.stattools import adfuller

    survivors = []
    rejected = {
        "adf": 0,
        "half_life": 0,
        "hurst": 0,
        "trades": 0,
        "loss": 0,
        "short": 0,
        "beta": 0,
        "excluded": 0,
    }
    n_pairs = 0
    MIN_PAIR_BARS = 600
    for i, a in enumerate(bases):
        for b in bases[i + 1 :]:
            n_pairs += 1
            if a in EXCLUDE_BASES or b in EXCLUDE_BASES:
                rejected["excluded"] += 1
                continue
            aligned = pd.concat(
                [by_base[a].rename("y"), by_base[b].rename("x")], axis=1, join="inner"
            ).dropna()
            if len(aligned) < MIN_PAIR_BARS:
                rejected["short"] += 1
                continue
            y = aligned["y"].values
            x = aligned["x"].values
            beta = np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1)
            if abs(beta) > BETA_ABS_MAX or abs(beta) < (1.0 / BETA_ABS_MAX):
                # Untradeable hedge ratio — see BETA_ABS_MAX note above.
                rejected["beta"] += 1
                continue
            alpha = y.mean() - beta * x.mean()
            resid = y - (beta * x + alpha)

            try:
                adf_p = float(adfuller(resid, maxlag=5, autolag=None)[1])
            except Exception:
                continue
            if adf_p > ADF_P_MAX:
                rejected["adf"] += 1
                continue

            hl = ou_half_life(resid)
            if not (HL_MIN_H <= hl <= HL_MAX_H):
                rejected["half_life"] += 1
                continue

            hurst = hurst_exponent(resid)
            if hurst > HURST_MAX:
                rejected["hurst"] += 1
                continue

            bt = backtest_spread(y, x, beta)
            if bt.get("n_trades", 0) < MIN_TRADES:
                rejected["trades"] += 1
                continue
            if bt["total_pct"] <= 0:
                rejected["loss"] += 1
                continue

            row = {
                "a": a,
                "b": b,
                "beta": beta,
                "adf_p": adf_p,
                "half_life": hl,
                "hurst": hurst,
                **bt,
            }
            row["score"] = composite_score(row)
            survivors.append(row)

    survivors.sort(key=lambda c: -c["score"])

    print()
    print("=" * 132)
    print(
        f"ADVANCED STAT-ARB SCAN — {n_pairs} pairs tested (per-pair aligned, ~{args.days}d hourly)"
    )
    print(
        f"Filters: adf_p≤{ADF_P_MAX}, half_life∈[{HL_MIN_H},{HL_MAX_H}]h, "
        f"hurst≤{HURST_MAX}, n_trades≥{MIN_TRADES}, total>0"
    )
    print(
        f"Rejected: excluded={rejected['excluded']} short={rejected['short']} "
        f"beta={rejected['beta']} adf={rejected['adf']} "
        f"half_life={rejected['half_life']} hurst={rejected['hurst']} "
        f"trades={rejected['trades']} loss={rejected['loss']}"
    )
    print("=" * 132)
    hdr = (
        f"{'pair':<13}{'beta':>9}{'adf_p':>7}{'half_l':>8}{'hurst':>7}"
        f"{'n':>4}{'win%':>6}{'avg%':>7}{'total%':>8}{'sharpe':>7}{'hold':>6}{'SCORE':>8}"
    )
    print(hdr)
    print("-" * 132)
    for c in survivors[: args.top]:
        print(
            f"{c['a']+'-'+c['b']:<13}{c['beta']:>+8.4f}{c['adf_p']:>7.3f}"
            f"{c['half_life']:>7.1f}h{c['hurst']:>7.2f}{c['n_trades']:>4}"
            f"{c['win_rate']*100:>5.0f}%{c['avg_pct']:>+7.2f}{c['total_pct']:>+8.2f}"
            f"{c['sharpe']:>+7.2f}{c['median_hold_h']:>5.0f}h{c['score']:>8.2f}"
        )

    print()
    print("=" * 132)
    print("PROPOSED high-quality PAIRS_LIST (top by composite score):")
    print("=" * 132)
    print("    PAIRS_LIST: list = [")
    for c in survivors[:8]:
        print(
            f'        "{c["a"]}-{c["b"]}",'
            f'  # HL={c["half_life"]:.0f}h H={c["hurst"]:.2f} '
            f'Sharpe={c["sharpe"]:.2f} n={c["n_trades"]}'
        )
    print("    ]")

    # Persist full results
    out = Path(__file__).resolve().parents[2] / "data" / "pairs_adv_scan.csv"
    pd.DataFrame(survivors).to_csv(out, index=False)
    logger.info(f"Saved {len(survivors)} survivors → {out}")


if __name__ == "__main__":
    asyncio.run(main())
