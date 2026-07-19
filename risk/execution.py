"""Almgren-Chriss TWAP execution scheduler for large orders."""

import asyncio
import logging
from decimal import Decimal

from t_tech.invest import OrderDirection

from core.broker import BrokerClient

logger = logging.getLogger(__name__)


class ExecutionScheduler:
    """Splits large orders into TWAP slices to minimize market impact.

    Each slice uses the same order execution mode as single orders
    (limit_aggressive / limit_passive / market).
    """

    def __init__(self, broker: BrokerClient):
        self.broker = broker

    def should_use_twap(self, lots: int, lot_size: int, avg_daily_volume: float) -> bool:
        """Use TWAP if order > 1% of daily volume."""
        if avg_daily_volume <= 0:
            return True
        order_shares = lots * lot_size
        participation = order_shares / avg_daily_volume
        return participation > 0.01

    async def execute_twap(
        self,
        figi: str,
        total_lots: int,
        direction: OrderDirection,
        duration_minutes: int = 30,
        n_slices: int = 6,
        exec_mode: str = "limit_aggressive",
        timeout: float = 45.0,
    ) -> list[tuple]:
        """Execute order in TWAP slices using smart order routing.

        Args:
            figi: Instrument FIGI.
            total_lots: Total lots to execute.
            direction: Buy or sell.
            duration_minutes: Total execution window.
            n_slices: Number of slices.
            exec_mode: "market" | "limit_aggressive" | "limit_passive"
            timeout: Per-slice limit order fill timeout (seconds).

        Returns:
            List of (order_response, fill_price, order_type, filled_lots) tuples.
            `filled_lots` is the actual per-slice fill (may be < slice_lots if
            a slice partially filled).  Caller MUST sum these for total filled.
        """
        if total_lots <= 0:
            return []

        # Compute slice sizes
        base_size = total_lots // n_slices
        remainder = total_lots % n_slices
        slices = [base_size + (1 if i < remainder else 0) for i in range(n_slices)]
        slices = [s for s in slices if s > 0]

        # Per-slice timeout should not exceed inter-slice interval
        interval_seconds = (duration_minutes * 60) / len(slices)
        slice_timeout = min(timeout, interval_seconds * 0.8)

        results = []
        logger.info(
            f"TWAP execution: {total_lots} lots in {len(slices)} slices "
            f"over {duration_minutes}min [{exec_mode}]"
        )

        for i, slice_lots in enumerate(slices):
            try:
                if exec_mode == "market":
                    resp = await self.broker.post_market_order(figi, slice_lots, direction)
                    fill_price = float(await self.broker.get_last_price(figi))
                    slice_filled = getattr(resp, "lots_executed", slice_lots) or slice_lots
                    results.append((resp, fill_price, "market", slice_filled))
                else:
                    resp, fill_price, order_type, slice_filled = (
                        await self.broker.post_limit_with_fallback(
                            figi=figi,
                            lots=slice_lots,
                            direction=direction,
                            mode=exec_mode,
                            timeout=slice_timeout,
                            fallback_market=True,
                        )
                    )
                    results.append((resp, fill_price, order_type, slice_filled))

                logger.info(
                    f"TWAP slice {i + 1}/{len(slices)}: "
                    f"{slice_filled}/{slice_lots} lots @ {fill_price:.4f}"
                )

            except Exception as e:
                logger.error(f"TWAP slice {i + 1} failed: {e}")

            if i < len(slices) - 1:
                await asyncio.sleep(interval_seconds)

        filled = len(results)
        total_price = sum(r[1] for r in results) / filled if filled else 0
        logger.info(f"TWAP complete: {filled}/{len(slices)} slices, avg price={total_price:.4f}")
        return results
