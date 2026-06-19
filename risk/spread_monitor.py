"""Glosten-Milgrom spread anomaly detection.

Wide spreads indicate informed trading or low liquidity.
Reject trades when spread > threshold * median historical spread.
"""

import logging
from collections import deque
from decimal import Decimal

logger = logging.getLogger(__name__)


class SpreadMonitor:
    """Monitors bid-ask spreads to detect anomalous conditions."""

    def __init__(self, threshold_multiplier: float = 2.0, window_size: int = 100):
        self._threshold = threshold_multiplier
        self._window = window_size
        self._history: dict[str, deque[float]] = {}

    def record_spread(self, figi: str, bid: Decimal, ask: Decimal):
        """Record a spread observation."""
        if figi not in self._history:
            self._history[figi] = deque(maxlen=self._window)

        mid = (float(bid) + float(ask)) / 2
        if mid > 0:
            spread_bps = (float(ask) - float(bid)) / mid * 10000
            self._history[figi].append(spread_bps)

    def get_spread_bps(self, figi: str, bid: Decimal, ask: Decimal) -> float:
        """Calculate current spread in basis points."""
        mid = (float(bid) + float(ask)) / 2
        if mid <= 0:
            return 0.0
        return (float(ask) - float(bid)) / mid * 10000

    def is_spread_normal(self, figi: str) -> tuple[bool, float]:
        """Check if current spread is within normal range.

        Returns (is_normal, current_ratio_vs_median).
        """
        history = self._history.get(figi)
        if not history or len(history) < 10:
            return True, 1.0  # Not enough data, assume normal

        current = history[-1]
        sorted_spreads = sorted(history)
        median = sorted_spreads[len(sorted_spreads) // 2]

        if median <= 0:
            return True, 1.0

        ratio = current / median
        is_normal = ratio < self._threshold

        if not is_normal:
            logger.warning(
                f"Spread anomaly for {figi}: {current:.1f}bps vs median {median:.1f}bps (ratio {ratio:.1f}x)"
            )

        return is_normal, round(ratio, 2)

    def get_typical_spread(self, figi: str) -> float:
        """Get median spread in bps for instrument."""
        history = self._history.get(figi)
        if not history:
            return 0.0
        sorted_spreads = sorted(history)
        return sorted_spreads[len(sorted_spreads) // 2]

    def get_stats(self, figi: str) -> dict:
        """Get spread statistics for display."""
        history = self._history.get(figi)
        if not history:
            return {"median": 0, "current": 0, "ratio": 1.0, "observations": 0}
        sorted_spreads = sorted(history)
        median = sorted_spreads[len(sorted_spreads) // 2]
        current = history[-1]
        return {
            "median": round(median, 1),
            "current": round(current, 1),
            "ratio": round(current / median, 2) if median > 0 else 1.0,
            "observations": len(history),
        }
