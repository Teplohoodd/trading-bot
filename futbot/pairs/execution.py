"""Two-leg order placement.

A spread position has two legs:
  * y-leg (the "base" asset, e.g. LK)
  * x-leg (the "hedge" asset, e.g. Si), sized by β
direction = +1 → BUY y, SELL x
direction = -1 → SELL y, BUY x

Lot sizing is approximate in paper mode (1 lot of y + |β|-rounded lots
of x).  For live execution we floor to integer lots and clamp by per-
pair capital budget.

Failure semantics (live):
  * If leg-1 fills but leg-2 fails → IMMEDIATELY close leg-1 at market.
    Half-hedged exposure is the worst possible state.
"""

import logging
import math
import time

from tinkoff.invest import OrderDirection

from futbot.utils import commissions as comm

logger = logging.getLogger("futbot.pairs.exec")


async def place_two_leg_entry(
    *,
    broker,
    figi_y: str,
    figi_x: str,
    ticker_y: str,
    ticker_x: str,
    direction: int,
    lots_y: int,
    lots_x: int,
    paper: bool,
) -> tuple[str, str, float, float]:
    """Place both legs.  Returns (order_id_y, order_id_x, fill_y, fill_x).

    Convention: direction=+1 → buy y, sell x.  direction=-1 → sell y, buy x.
    """
    side_y_dir = "buy" if direction > 0 else "sell"
    side_x_dir = "sell" if direction > 0 else "buy"

    last_y = float(await broker.get_last_price(figi_y))
    last_x = float(await broker.get_last_price(figi_x))

    if paper:
        ts = int(time.time())
        oid_y = f"paper-pair-{side_y_dir}-{figi_y[-6:]}-{ts}"
        oid_x = f"paper-pair-{side_x_dir}-{figi_x[-6:]}-{ts}"
        logger.info(
            f"[PAPER] OPEN  {ticker_y}/{ticker_x}  "
            f"{side_y_dir.upper()} {lots_y} × {ticker_y} @ {last_y:.4f}  "
            f"+ {side_x_dir.upper()} {lots_x} × {ticker_x} @ {last_x:.4f}"
        )
        return oid_y, oid_x, last_y, last_x

    # ── Live execution ─────────────────────────────────────────────────
    def _sdk_dir(s: str) -> OrderDirection:
        return (
            OrderDirection.ORDER_DIRECTION_BUY
            if s == "buy"
            else OrderDirection.ORDER_DIRECTION_SELL
        )

    # Leg 1 first — Y is usually the more liquid contract; if it fails
    # we never expose ourselves on X.  Use *_with_fill so the recorded entry
    # price is the REAL executed price, not a last-price snapshot (the old
    # behaviour flipped P&L signs on thin spreads).
    try:
        res_y = await broker.post_market_order_with_fill(figi_y, lots_y, _sdk_dir(side_y_dir))
        oid_y = res_y["order_id"]
    except Exception as e:
        logger.error(f"LIVE: leg Y {ticker_y} failed ({e}) — aborting pair entry")
        raise

    # Leg 2 — if this fails, immediately close leg 1
    try:
        res_x = await broker.post_market_order_with_fill(figi_x, lots_x, _sdk_dir(side_x_dir))
        oid_x = res_x["order_id"]
    except Exception as e:
        logger.critical(f"LIVE: leg X {ticker_x} failed AFTER y filled — emergency unwind y")
        # Emergency unwind y
        try:
            unwind_dir = _sdk_dir("sell" if side_y_dir == "buy" else "buy")
            await broker.post_market_order(figi_y, lots_y, unwind_dir)
            logger.info(f"  Emergency unwind y done")
        except Exception as ee:
            logger.critical(f"  UNWIND ALSO FAILED — manual intervention needed: {ee}")
        raise

    # REAL executed fills (fall back to last price only if the venue gave none)
    fill_y = res_y["fill_price"] if res_y["fill_price"] > 0 else last_y
    fill_x = res_x["fill_price"] if res_x["fill_price"] > 0 else last_x
    comm = res_y.get("commission_rub", 0.0) + res_x.get("commission_rub", 0.0)
    logger.info(
        f"[LIVE] OPEN  {ticker_y}/{ticker_x}  "
        f"{side_y_dir.upper()} {lots_y} @ {fill_y:.4f} oid={oid_y}  "
        f"+ {side_x_dir.upper()} {lots_x} @ {fill_x:.4f} oid={oid_x}  "
        f"(real comm {comm:.2f} ₽)"
    )
    return oid_y, oid_x, fill_y, fill_x


async def place_two_leg_exit(
    *,
    broker,
    figi_y: str,
    figi_x: str,
    ticker_y: str,
    ticker_x: str,
    direction: int,
    lots_y: int,
    lots_x: int,
    reason: str,
    paper: bool,
) -> tuple[str, str, float, float]:
    """Close both legs.  direction is the OPEN direction (we reverse it)."""
    # Reverse the original sides
    side_y = "sell" if direction > 0 else "buy"
    side_x = "buy" if direction > 0 else "sell"
    last_y = float(await broker.get_last_price(figi_y))
    last_x = float(await broker.get_last_price(figi_x))

    if paper:
        ts = int(time.time())
        oid_y = f"paper-pair-exit-{figi_y[-6:]}-{ts}"
        oid_x = f"paper-pair-exit-{figi_x[-6:]}-{ts}"
        logger.info(
            f"[PAPER] CLOSE ({reason}) {ticker_y}/{ticker_x}  "
            f"{side_y.upper()} {lots_y} × {ticker_y} @ {last_y:.4f}  "
            f"+ {side_x.upper()} {lots_x} × {ticker_x} @ {last_x:.4f}"
        )
        return oid_y, oid_x, last_y, last_x

    def _sdk_dir(s: str) -> OrderDirection:
        return (
            OrderDirection.ORDER_DIRECTION_BUY
            if s == "buy"
            else OrderDirection.ORDER_DIRECTION_SELL
        )

    # Close both — best effort, log every failure.  Use *_with_fill so the
    # recorded exit price is the REAL executed price (not a last-price guess).
    fill_y, fill_x = last_y, last_x
    comm = 0.0
    try:
        res_y = await broker.post_market_order_with_fill(figi_y, lots_y, _sdk_dir(side_y))
        oid_y = res_y["order_id"]
        if res_y["fill_price"] > 0:
            fill_y = res_y["fill_price"]
        comm += res_y.get("commission_rub", 0.0)
    except Exception as e:
        logger.error(f"LIVE exit: y {ticker_y} failed: {e}")
        oid_y = "?"
    try:
        res_x = await broker.post_market_order_with_fill(figi_x, lots_x, _sdk_dir(side_x))
        oid_x = res_x["order_id"]
        if res_x["fill_price"] > 0:
            fill_x = res_x["fill_price"]
        comm += res_x.get("commission_rub", 0.0)
    except Exception as e:
        logger.error(f"LIVE exit: x {ticker_x} failed: {e}")
        oid_x = "?"

    logger.info(
        f"[LIVE] CLOSE ({reason}) {ticker_y}/{ticker_x}  "
        f"{side_y.upper()} {lots_y} @ {fill_y:.4f} oid={oid_y}  "
        f"+ {side_x.upper()} {lots_x} @ {fill_x:.4f} oid={oid_x}  "
        f"(real comm {comm:.2f} ₽)"
    )
    return oid_y, oid_x, fill_y, fill_x


# ── Limit-order (passive) two-leg execution ─────────────────────────────
#
# For thin-edge spreads (carry) market orders cross the bid/ask on every leg
# and the slippage exceeds the edge.  These post PASSIVE limits (buy at bid /
# sell at ask) to *capture* the spread instead of paying it, via the broker's
# order-book-aware post_limit_with_fallback (which falls back to a market
# order on the unfilled remainder so a leg never hangs half-open).  Entry
# keeps the emergency-unwind safety: if leg X can't be completed we close Y.


async def place_two_leg_limit_entry(
    *,
    broker,
    figi_y: str,
    figi_x: str,
    ticker_y: str,
    ticker_x: str,
    direction: int,
    lots_y: int,
    lots_x: int,
    paper: bool,
    timeout: float = 30.0,
) -> tuple[str, str, float, float]:
    """Passive-limit version of place_two_leg_entry (captures the spread)."""
    side_y_dir = "buy" if direction > 0 else "sell"
    side_x_dir = "sell" if direction > 0 else "buy"
    last_y = float(await broker.get_last_price(figi_y))
    last_x = float(await broker.get_last_price(figi_x))

    if paper:
        # Model PASSIVE fills: buy at bid / sell at ask (capture the spread),
        # so paper P&L reflects the limit-order benefit, not last price.
        py = await broker.get_fair_price(figi_y, side_y_dir, "limit_passive")
        px = await broker.get_fair_price(figi_x, side_x_dir, "limit_passive")
        fill_y = float(py) if py else last_y
        fill_x = float(px) if px else last_x
        ts = int(time.time())
        oid_y = f"paper-pair-lim-{side_y_dir}-{figi_y[-6:]}-{ts}"
        oid_x = f"paper-pair-lim-{side_x_dir}-{figi_x[-6:]}-{ts}"
        logger.info(
            f"[PAPER] OPEN(lim) {ticker_y}/{ticker_x}  "
            f"{side_y_dir.upper()} {lots_y}×{ticker_y} @ {fill_y:.4f}  "
            f"+ {side_x_dir.upper()} {lots_x}×{ticker_x} @ {fill_x:.4f}"
        )
        return oid_y, oid_x, fill_y, fill_x

    def _sdk_dir(s: str) -> OrderDirection:
        return (
            OrderDirection.ORDER_DIRECTION_BUY
            if s == "buy"
            else OrderDirection.ORDER_DIRECTION_SELL
        )

    # Leg Y first (more liquid). post_limit_with_fallback: passive limit,
    # then market-fallback the unfilled remainder → always fully filled.
    try:
        resp_y, fill_y, otype_y, filled_y = await broker.post_limit_with_fallback(
            figi_y,
            lots_y,
            _sdk_dir(side_y_dir),
            mode="limit_passive",
            timeout=timeout,
            fallback_market=True,
        )
        oid_y = getattr(resp_y, "order_id", "?")
        if not fill_y or fill_y <= 0:
            fill_y = last_y
    except Exception as e:
        logger.error(f"LIVE(lim): leg Y {ticker_y} failed ({e}) — aborting entry")
        raise

    # Leg X — on failure, emergency-unwind Y (market) to avoid naked exposure
    try:
        resp_x, fill_x, otype_x, filled_x = await broker.post_limit_with_fallback(
            figi_x,
            lots_x,
            _sdk_dir(side_x_dir),
            mode="limit_passive",
            timeout=timeout,
            fallback_market=True,
        )
        oid_x = getattr(resp_x, "order_id", "?")
        if not fill_x or fill_x <= 0:
            fill_x = last_x
    except Exception as e:
        logger.critical(f"LIVE(lim): leg X {ticker_x} failed AFTER y filled — unwinding y")
        try:
            await broker.post_market_order(
                figi_y, lots_y, _sdk_dir("sell" if side_y_dir == "buy" else "buy")
            )
            logger.info("  Emergency unwind y done")
        except Exception as ee:
            logger.critical(f"  UNWIND ALSO FAILED — manual intervention: {ee}")
        raise

    logger.info(
        f"[LIVE] OPEN(lim) {ticker_y}/{ticker_x}  "
        f"{side_y_dir.upper()} {lots_y} @ {fill_y:.4f} ({otype_y}) oid={oid_y}  "
        f"+ {side_x_dir.upper()} {lots_x} @ {fill_x:.4f} ({otype_x}) oid={oid_x}"
    )
    return oid_y, oid_x, fill_y, fill_x


async def place_two_leg_limit_exit(
    *,
    broker,
    figi_y: str,
    figi_x: str,
    ticker_y: str,
    ticker_x: str,
    direction: int,
    lots_y: int,
    lots_x: int,
    reason: str,
    paper: bool,
    timeout: float = 30.0,
) -> tuple[str, str, float, float]:
    """Passive-limit version of place_two_leg_exit.  direction = OPEN dir."""
    side_y = "sell" if direction > 0 else "buy"
    side_x = "buy" if direction > 0 else "sell"
    last_y = float(await broker.get_last_price(figi_y))
    last_x = float(await broker.get_last_price(figi_x))

    if paper:
        py = await broker.get_fair_price(figi_y, side_y, "limit_passive")
        px = await broker.get_fair_price(figi_x, side_x, "limit_passive")
        fill_y = float(py) if py else last_y
        fill_x = float(px) if px else last_x
        ts = int(time.time())
        oid_y = f"paper-pair-lim-exit-{figi_y[-6:]}-{ts}"
        oid_x = f"paper-pair-lim-exit-{figi_x[-6:]}-{ts}"
        logger.info(
            f"[PAPER] CLOSE(lim,{reason}) {ticker_y}/{ticker_x}  "
            f"{side_y.upper()} {lots_y}×{ticker_y} @ {fill_y:.4f}  "
            f"+ {side_x.upper()} {lots_x}×{ticker_x} @ {fill_x:.4f}"
        )
        return oid_y, oid_x, fill_y, fill_x

    def _sdk_dir(s: str) -> OrderDirection:
        return (
            OrderDirection.ORDER_DIRECTION_BUY
            if s == "buy"
            else OrderDirection.ORDER_DIRECTION_SELL
        )

    fill_y, fill_x = last_y, last_x
    try:
        resp_y, f_y, otype_y, _ = await broker.post_limit_with_fallback(
            figi_y,
            lots_y,
            _sdk_dir(side_y),
            mode="limit_passive",
            timeout=timeout,
            fallback_market=True,
        )
        oid_y = getattr(resp_y, "order_id", "?")
        if f_y and f_y > 0:
            fill_y = f_y
    except Exception as e:
        logger.error(f"LIVE(lim) exit: y {ticker_y} failed: {e}")
        oid_y = "?"
    try:
        resp_x, f_x, otype_x, _ = await broker.post_limit_with_fallback(
            figi_x,
            lots_x,
            _sdk_dir(side_x),
            mode="limit_passive",
            timeout=timeout,
            fallback_market=True,
        )
        oid_x = getattr(resp_x, "order_id", "?")
        if f_x and f_x > 0:
            fill_x = f_x
    except Exception as e:
        logger.error(f"LIVE(lim) exit: x {ticker_x} failed: {e}")
        oid_x = "?"

    logger.info(
        f"[LIVE] CLOSE(lim,{reason}) {ticker_y}/{ticker_x}  "
        f"{side_y.upper()} {lots_y} @ {fill_y:.4f} oid={oid_y}  "
        f"+ {side_x.upper()} {lots_x} @ {fill_x:.4f} oid={oid_x}"
    )
    return oid_y, oid_x, fill_y, fill_x


def compute_two_leg_pnl(
    *,
    direction: int,
    beta: float,
    entry_y: float,
    entry_x: float,
    exit_y: float,
    exit_x: float,
    lots_y: int,
    lots_x: int,
    rpp_y: float,
    rpp_x: float,
    lot_size_y: int,
    lot_size_x: int,
    instrument_kind: str = "future",
    base_y: str = "",
    base_x: str = "",
) -> dict:
    """P&L of both legs together, net of round-trip commission on each.

    Returns dict with gross_rub, commission_rub, net_rub, net_pct.
    """
    # Leg y P&L
    pnl_y, _, gross_y, comm_y = comm.round_trip_pnl(
        direction="buy" if direction > 0 else "sell",
        entry_price=entry_y,
        exit_price=exit_y,
        lots=lots_y,
        lot_size=lot_size_y,
        rub_per_point=rpp_y,
        instrument_kind=instrument_kind,
        base_ticker=base_y,
    )
    # Leg x P&L — opposite direction
    pnl_x, _, gross_x, comm_x = comm.round_trip_pnl(
        direction="sell" if direction > 0 else "buy",
        entry_price=entry_x,
        exit_price=exit_x,
        lots=lots_x,
        lot_size=lot_size_x,
        rub_per_point=rpp_x,
        instrument_kind=instrument_kind,
        base_ticker=base_x,
    )
    net_rub = pnl_y + pnl_x
    gross_rub = gross_y + gross_x
    commission_rub = comm_y + comm_x
    # P&L % of combined notional at entry
    notional_y = abs(entry_y * lots_y * lot_size_y * rpp_y)
    notional_x = abs(entry_x * lots_x * lot_size_x * rpp_x)
    combined = notional_y + notional_x
    pnl_pct = (net_rub / combined * 100) if combined > 0 else 0.0
    return {
        "gross_rub": round(gross_rub, 2),
        "commission_rub": round(commission_rub, 2),
        "net_rub": round(net_rub, 2),
        "pnl_pct": round(pnl_pct, 4),
    }


def compute_lots(
    *,
    portfolio_value: float,
    beta: float,
    price_y: float,
    price_x: float,
    rpp_y: float,
    rpp_x: float,
    lot_size_y: int,
    lot_size_x: int,
    dlong_y: float = 0.0,
    dlong_x: float = 0.0,
    dshort_y: float = 0.0,
    dshort_x: float = 0.0,
    direction: int = +1,
    capital_per_pair_pct: float = 0.10,
    fallback_margin_pct: float = 0.25,
) -> tuple[int, int]:
    """β-hedged sizing with PER-LEG MARGIN budget.

    Why margin (not notional): Russian futures use leverage — 1 lot of
    Si (notional ~72k ₽) only ties up ~6k ₽ margin.  Sizing by notional
    would refuse all but the cheapest contracts on small portfolios.
    Sizing by margin matches what actually gets blocked in the account.

    Margin source (preferred → fallback):
      1. broker-supplied dlong/dshort fraction (from
         extract_futures_metadata) × notional → exact margin
      2. fallback_margin_pct × notional (default 25%, deliberately
         conservative — typical FORTS margin is 5-20%)

    For a LONG pair (direction=+1): y is bought (dlong_y), x is sold (dshort_x).
    For a SHORT pair (direction=-1): y is sold (dshort_y), x is bought (dlong_x).

    Returns (0, 0) — caller MUST treat as "skip this pair" — when even
    1 lot of either leg exceeds the per-leg margin budget.

    Safety guards (in order):
      1. MIN_SANE_NOTIONAL: if notional < 500 ₽/lot the broker is returning
         a foreign-currency price (e.g. NG in USD/mmBtu) and rub_per_point=1
         fallback is completely wrong — reject the pair.
      2. Standard margin-budget rejection: if 1-lot margin > per_leg → (0,0).
      3. SAFETY_CAP: hard ceiling of 25 lots per leg so a miscalibrated
         fallback_margin_pct can never produce catastrophic position size.

    Rules:
      pair_capital = portfolio_value × capital_per_pair_pct
      per_leg = pair_capital / 2  (margin budget per leg)
      lots_y = floor(per_leg / margin_y_per_lot)   capped at SAFETY_CAP
      lots_x_raw = |β| × (notional_y / notional_x) × lots_y
      if lots_x margin > per_leg → scale BOTH legs down proportionally
    """
    # Guard 1: sanity-check notional.  T-Invest API returns prices in the
    # native currency of the underlying; some FORTS contracts (e.g. NG
    # natural gas) quote in USD/mmBtu (~3 USD) while rub_per_point defaults
    # to 1.0.  1 contract × 3.17 × 1.0 = 3.17 "rubles" notional is
    # nonsensical.  500 ₽/lot is a conservative floor that correctly
    # rejects these while accepting MM (~2 600 ₽/lot), Si (~72 000), etc.
    MIN_SANE_NOTIONAL = 500.0  # rubles per lot
    # Guard 3: absolute maximum per leg regardless of budget/margin calc.
    # Protects against any future mis-calibration without hiding correct
    # large-portfolio trades (at 600k, 30% budget → 90k per_leg → ~12 lots
    # of Si = well within 25).
    SAFETY_CAP = 25

    pair_capital = portfolio_value * capital_per_pair_pct
    per_leg = pair_capital / 2

    notional_y_per_lot = price_y * lot_size_y * rpp_y
    notional_x_per_lot = price_x * lot_size_x * rpp_x
    if notional_y_per_lot <= 0 or notional_x_per_lot <= 0 or per_leg <= 0:
        return 0, 0

    # Guard 1
    if notional_y_per_lot < MIN_SANE_NOTIONAL or notional_x_per_lot < MIN_SANE_NOTIONAL:
        logger.warning(
            f"compute_lots: notional too small "
            f"(y={notional_y_per_lot:.2f} x={notional_x_per_lot:.2f} ₽/lot) "
            f"— likely foreign-currency price; skipping pair"
        )
        return 0, 0

    # Pick the correct margin side based on what we're doing with each leg
    dy = float(dlong_y) if direction > 0 else float(dshort_y)
    dx = float(dshort_x) if direction > 0 else float(dlong_x)
    margin_y_per_lot = (
        notional_y_per_lot * dy if dy > 0 else notional_y_per_lot * fallback_margin_pct
    )
    margin_x_per_lot = (
        notional_x_per_lot * dx if dx > 0 else notional_x_per_lot * fallback_margin_pct
    )

    # Guard 2: 1-lot margin budget check
    if margin_y_per_lot > per_leg or margin_x_per_lot > per_leg:
        return 0, 0

    lots_y = min(SAFETY_CAP, max(1, math.floor(per_leg / margin_y_per_lot)))
    ratio = abs(beta) * notional_y_per_lot / notional_x_per_lot
    lots_x_raw = ratio * lots_y
    lots_x = min(SAFETY_CAP, max(1, round(lots_x_raw)))

    # X-leg over-budget guard (β scaling can push X past per_leg)
    if lots_x * margin_x_per_lot > per_leg:
        scale = per_leg / (lots_x * margin_x_per_lot)
        lots_x = max(1, math.floor(lots_x * scale))
        lots_y = max(1, math.floor(lots_y * scale))
        if lots_x * margin_x_per_lot > per_leg or lots_y * margin_y_per_lot > per_leg:
            return 0, 0

    # Guard 3: hard ceiling (final check after all scaling)
    if lots_y > SAFETY_CAP or lots_x > SAFETY_CAP:
        return 0, 0

    return int(lots_y), int(lots_x)
