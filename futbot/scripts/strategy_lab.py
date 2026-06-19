"""Strategy lab — backtest proven futures strategies on live FORTS data.

Implements two academically-validated strategies and reports results both on
NOTIONAL and on MARGIN (leverage-adjusted — what actually matters on a
margin-traded futures account):

  1. Time-Series Momentum (TSMOM)
     Moskowitz, Ooi & Pedersen (2012), J. Financial Economics 104(2).
     Signal = sign of trailing return over lookback L; position vol-scaled to
     a target; re-evaluated each bar; transaction cost on flips.  Documented
     across 58 futures incl. currencies.  We test several lookbacks across the
     liquid universe and single out the currency contracts (Si, USDRU, EURRU,
     GLDRU, CR).

  2. Carry / term-structure (Si = USD/RUB)
     Covered Interest Parity: with RUB rates >> USD rates the USD/RUB futures
     curve is in contango (F > S, slope ≈ rate differential).  We measure the
     live annualised basis from multiple Si expiries, and backtest a simple
     "short front-month, roll" carry approximation.

Leverage note: return-on-margin = notional_return / margin_fraction.  With
Si margin ≈ 9.3 % that's ~10.7× leverage; we report both so the real
risk/return on deployed capital is visible.

Usage:
    python -u -m futbot.scripts.strategy_lab --days 180
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
from tinkoff.invest import CandleInterval
from tinkoff.invest.utils import quotation_to_decimal

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("tinkoff").setLevel(logging.WARNING)
logger = logging.getLogger("strat_lab")


UNIVERSE = [
    "BR",
    "GZ",
    "SR",
    "LK",
    "MX",
    "Si",
    "GD",
    "RT",
    "USDRU",
    "GLDRU",
    "EURRU",
    "CR",
    "PX",
    "YD",
    "VB",
    "RN",
    "GK",
    "MM",
    "PT",
    "S1",
    "MV",
    "SV",
]
CURRENCY_BASES = {"Si", "USDRU", "EURRU", "GLDRU", "CR"}

HOURS_PER_YEAR = 24 * 365
COMMISSION_RT = 0.0008  # 0.08 % round-trip (futures)
TSMOM_LOOKBACKS = [120, 240, 480, 720]  # hours (~5,10,20,30 trading days)
VOL_WINDOW = 240  # hours for realised-vol scaling
TARGET_VOL_ANN = 0.15  # 15 % annual vol target per instrument


# ════════════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════════════


async def _all_expiries(broker, base: str):
    """Return [(future, expiry_dt)] sorted by expiry for a base ticker."""
    futs = await broker.get_all_futures()
    now = datetime.now(timezone.utc)
    out = []
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
        out.append((f, exp))
    out.sort(key=lambda x: x[1])
    return out, now


async def _resolve_front(broker, base: str):
    cands, now = await _all_expiries(broker, base)
    if not cands:
        return None
    for f, exp in cands:
        if (exp - now).days >= 14:
            return f
    return cands[0][0]


def _candles_df(candles) -> pd.DataFrame:
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
    left = days
    while left > 0:
        cd = min(left, 89)
        start = end - timedelta(days=cd)
        try:
            c = await broker.get_candles(
                figi, start, end, interval=CandleInterval.CANDLE_INTERVAL_HOUR
            )
            df = _candles_df(c)
            if not df.empty:
                chunks.append(df)
        except Exception:
            pass
        end = start
        left -= cd
    if not chunks:
        return pd.DataFrame()
    return (
        pd.concat(chunks)
        .drop_duplicates("time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )


# ════════════════════════════════════════════════════════════════════════
# Strategy 1 — Time-Series Momentum
# ════════════════════════════════════════════════════════════════════════


def tsmom_backtest(close: np.ndarray, lookback: int) -> dict:
    """Vol-scaled time-series momentum on an hourly close series.

    Position at bar t (applied to return t→t+1):
        raw = sign(close[t] - close[t-lookback])
        scale = target_vol_per_bar / realised_vol(last VOL_WINDOW returns)
        pos = clip(raw * scale, -1, +1)          # cap leverage at 1× notional
    Net bar P&L (notional) = pos[t] * ret[t+1] - cost*|pos[t]-pos[t-1]|.
    """
    n = len(close)
    if n < lookback + VOL_WINDOW + 50:
        return {"n": 0}
    ret = np.zeros(n)
    ret[1:] = close[1:] / close[:-1] - 1.0
    target_per_bar = TARGET_VOL_ANN / math.sqrt(HOURS_PER_YEAR)

    pos = np.zeros(n)
    for t in range(lookback + VOL_WINDOW, n):
        raw = np.sign(close[t] - close[t - lookback])
        rv = ret[t - VOL_WINDOW : t].std()
        scale = (target_per_bar / rv) if rv > 0 else 0.0
        pos[t] = float(np.clip(raw * scale, -1.0, 1.0))

    start = lookback + VOL_WINDOW
    pnl = np.zeros(n)
    for t in range(start, n - 1):
        turn = abs(pos[t] - pos[t - 1])
        pnl[t + 1] = pos[t] * ret[t + 1] - COMMISSION_RT * turn

    active = pnl[start + 1 :]
    if len(active) == 0 or active.std() == 0:
        return {"n": 0}
    sharpe_ann = active.mean() / active.std() * math.sqrt(HOURS_PER_YEAR)
    total = active.sum()
    ann_ret = active.mean() * HOURS_PER_YEAR
    ann_vol = active.std() * math.sqrt(HOURS_PER_YEAR)
    # max drawdown of cumulative notional pnl
    cum = np.cumsum(active)
    peak = np.maximum.accumulate(cum)
    mdd = float((cum - peak).min())
    n_flips = int((np.abs(np.diff(pos[start:])) > 1e-9).sum())
    return {
        "n": len(active),
        "sharpe_ann": float(sharpe_ann),
        "total_notional": float(total),
        "ann_ret_notional": float(ann_ret),
        "ann_vol_notional": float(ann_vol),
        "mdd_notional": mdd,
        "flips": n_flips,
    }


# ════════════════════════════════════════════════════════════════════════
# Strategy 2 — Carry / term structure (Si)
# ════════════════════════════════════════════════════════════════════════


async def carry_term_structure(broker, base: str, days: int) -> dict:
    """Measure the live annualised basis across Si expiries + a simple
    short-front-month roll backtest approximation."""
    cands, now = await _all_expiries(broker, base)
    cands = [(f, e) for f, e in cands if (e - now).days >= 5][:4]
    if len(cands) < 2:
        return {}
    figis = [f.figi for f, _ in cands]
    last = await broker.get_last_prices(",".join(figis)) if False else None
    # get_last_price per figi (broker helper)
    prices = []
    for f, e in cands:
        p = float(await broker.get_last_price(f.figi))
        prices.append((f.ticker, e, p, (e - now).days))

    # Annualised basis between consecutive expiries
    legs = []
    for i in range(len(prices) - 1):
        t0, e0, p0, d0 = prices[i]
        t1, e1, p1, d1 = prices[i + 1]
        dt_years = max((e1 - e0).days, 1) / 365.0
        ann_basis = (p1 / p0 - 1.0) / dt_years if p0 > 0 else 0.0
        legs.append(
            {"front": t0, "next": t1, "p_front": p0, "p_next": p1, "ann_basis_pct": ann_basis * 100}
        )

    # Short-front-month roll backtest: hold a SHORT on the continuous
    # front-month series; in contango the front converges DOWN toward spot,
    # so short earns the decay.  Approximated on the front-month series.
    front = await _resolve_front(broker, base)
    bt = {}
    if front is not None:
        df = await _fetch_chunked(broker, front.figi, days)
        if len(df) > 100:
            c = df["close"].values
            ret = c[1:] / c[:-1] - 1.0
            # constant short, vol of the instrument
            short_ret = -ret
            ann = short_ret.mean() * HOURS_PER_YEAR
            vol = short_ret.std() * math.sqrt(HOURS_PER_YEAR)
            cum = np.cumprod(1 + short_ret) - 1
            bt = {
                "short_ann_ret_notional": float(ann),
                "short_ann_vol": float(vol),
                "short_total_notional": float(cum[-1]),
                "n_bars": len(short_ret),
            }
    return {"curve": prices, "legs": legs, "short_bt": bt}


# ════════════════════════════════════════════════════════════════════════


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()

    s = Settings()
    broker = BrokerClient(
        token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="strat-lab"
    )
    await broker.connect()

    # Fetch series + margin per base
    series, margins = {}, {}
    for base in UNIVERSE:
        f = await _resolve_front(broker, base)
        if f is None:
            continue
        df = await _fetch_chunked(broker, f.figi, args.days)
        if len(df) < 600:
            continue
        series[base] = df["close"].values
        meta = broker.extract_futures_metadata(f)
        dlong = float(meta.get("dlong") or 0.0)
        margins[base] = dlong if dlong > 0 else 0.25
        logger.info(f"  {base:6} {f.ticker:8} {len(df)} bars  margin={margins[base]:.3f}")

    # ── TSMOM ───────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print(
        "STRATEGY 1 — TIME-SERIES MOMENTUM (vol-scaled, MOP-2012)  " "[notional vs MARGIN-adjusted]"
    )
    print("=" * 120)
    rows = []
    for base, c in series.items():
        for lb in TSMOM_LOOKBACKS:
            r = tsmom_backtest(c, lb)
            if r.get("n", 0) == 0:
                continue
            lev = 1.0 / margins[base]
            rows.append(
                {
                    "base": base,
                    "lookback_h": lb,
                    "sharpe": r["sharpe_ann"],
                    "ann_notional_%": r["ann_ret_notional"] * 100,
                    "ann_margin_%": r["ann_ret_notional"] * lev * 100,
                    "mdd_margin_%": r["mdd_notional"] * lev * 100,
                    "leverage": lev,
                    "is_fx": base in CURRENCY_BASES,
                }
            )
    bt = pd.DataFrame(rows)
    if not bt.empty:
        # Best lookback per base by Sharpe
        best = bt.sort_values("sharpe", ascending=False).groupby("base").head(1)
        best = best.sort_values("sharpe", ascending=False)
        print(
            f"\n{'base':6}{'LB(h)':>6}{'sharpe':>8}{'annNotion%':>11}"
            f"{'annMargin%':>11}{'mddMargin%':>11}{'lev':>6}  fx"
        )
        print("-" * 120)
        for _, r in best.iterrows():
            print(
                f"{r['base']:6}{int(r['lookback_h']):>6}{r['sharpe']:>8.2f}"
                f"{r['ann_notional_%']:>11.1f}{r['ann_margin_%']:>11.1f}"
                f"{r['mdd_margin_%']:>11.1f}{r['leverage']:>6.1f}  "
                f"{'FX' if r['is_fx'] else ''}"
            )
        print(
            f"\nMean Sharpe (best-LB per base): all={best['sharpe'].mean():.2f}  "
            f"FX-only={best[best['is_fx']]['sharpe'].mean():.2f}"
        )
        # Diversified equal-weight portfolio at the 480h lookback
        lb_fixed = 480
        port_rets = []
        for base, c in series.items():
            r = tsmom_backtest(c, lb_fixed)
            if r.get("n", 0):
                port_rets.append(r["sharpe_ann"])
        if port_rets:
            print(
                f"Equal-weight TSMOM(480h) mean per-instrument Sharpe: "
                f"{np.mean(port_rets):.2f} across {len(port_rets)} contracts"
            )

    # ── Carry / term structure on Si ────────────────────────────────────
    print("\n" + "=" * 120)
    print("STRATEGY 2 — CARRY / TERM STRUCTURE  (Si = USD/RUB, CIP)")
    print("=" * 120)
    for base in ["Si", "USDRU", "EURRU"]:
        cs = await carry_term_structure(broker, base, args.days)
        if not cs:
            continue
        print(f"\n{base} futures curve (live):")
        for tk, e, p, d in cs["curve"]:
            print(f"   {tk:9} exp {e.date()} ({d:>3}d)  price={p:.2f}")
        for leg in cs["legs"]:
            print(
                f"   {leg['front']}→{leg['next']}: annualised basis = "
                f"{leg['ann_basis_pct']:+.1f}%/yr"
            )
        sb = cs.get("short_bt", {})
        if sb:
            lev = 1.0 / margins.get(base, 0.1)
            print(
                f"   short-front-month roll backtest ({args.days}d): "
                f"notional {sb['short_ann_ret_notional']*100:+.1f}%/yr, "
                f"on-margin {sb['short_ann_ret_notional']*lev*100:+.1f}%/yr "
                f"(lev {lev:.1f}×), vol {sb['short_ann_vol']*100:.0f}%"
            )

    await broker.disconnect()

    out = Path(__file__).resolve().parents[2] / "data" / "strategy_lab_tsmom.csv"
    if not bt.empty:
        bt.to_csv(out, index=False)
        logger.info(f"Saved TSMOM grid → {out}")


if __name__ == "__main__":
    asyncio.run(main())
