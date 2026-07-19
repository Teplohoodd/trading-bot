"""Order placement — thin wrapper supporting paper + live modes.

In PAPER mode:
  * No orders are sent to the broker.
  * Entry "fill price" = last traded price (best-effort estimate).
  * Exit  "fill price" = last traded price at the moment of close.
  * Returns a synthetic order_id like "paper-buy-FUTSIxxx-1684252800".

In LIVE mode:
  * Uses broker.post_market_order (simple/predictable for first live release).
  * Returns the real order_id from the broker.

A future enhancement would be limit-aggressive orders matching trade_claude's
post_limit_with_fallback, but for the first live release we keep it boring.
"""

import logging
import time
from decimal import Decimal

from t_tech.invest import OrderDirection

logger = logging.getLogger("futbot.orders")


async def place_entry(
    *, broker, figi: str, ticker: str, direction: str, lots: int, paper: bool
) -> tuple[str, float]:
    """Returns (order_id, fill_price)."""
    last_p = float(await broker.get_last_price(figi))

    if paper:
        oid = f"paper-{direction}-{figi[-6:]}-{int(time.time())}"
        logger.info(
            f"[PAPER] entry {direction.upper()} {ticker} ({figi}) "
            f"× {lots} lot(s) @ {last_p:.4f}  oid={oid}"
        )
        return oid, last_p

    # Live
    side = (
        OrderDirection.ORDER_DIRECTION_BUY
        if direction == "buy"
        else OrderDirection.ORDER_DIRECTION_SELL
    )
    resp = await broker.post_market_order(figi, lots, side)
    oid = getattr(resp, "order_id", None) or getattr(resp, "orderId", "?")
    # Best-effort fill price from the response; fall back to last price.
    fill = last_p
    try:
        if hasattr(resp, "executed_order_price"):
            fill = float(broker_to_decimal(resp.executed_order_price))
    except Exception:
        pass
    logger.info(
        f"[LIVE] entry {direction.upper()} {ticker} ({figi}) "
        f"× {lots} lot(s) @ {fill:.4f}  oid={oid}"
    )
    return oid, fill


async def place_exit(
    *, broker, figi: str, ticker: str, entry_direction: str, lots: int, reason: str, paper: bool
) -> tuple[str, float]:
    """Close an open position.  entry_direction = direction of the position
    being closed; the exit order is the opposite side."""
    last_p = float(await broker.get_last_price(figi))

    if paper:
        oid = f"paper-exit-{figi[-6:]}-{int(time.time())}"
        logger.info(
            f"[PAPER] exit  ({reason}) {ticker} ({figi}) "
            f"× {lots} lot(s) @ {last_p:.4f}  oid={oid}"
        )
        return oid, last_p

    side = (
        OrderDirection.ORDER_DIRECTION_SELL
        if entry_direction == "buy"
        else OrderDirection.ORDER_DIRECTION_BUY
    )
    resp = await broker.post_market_order(figi, lots, side)
    oid = getattr(resp, "order_id", None) or getattr(resp, "orderId", "?")
    fill = last_p
    logger.info(
        f"[LIVE] exit  ({reason}) {ticker} ({figi}) " f"× {lots} lot(s) @ {fill:.4f}  oid={oid}"
    )
    return oid, fill
