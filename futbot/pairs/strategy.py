"""Entry / exit decision logic for pair trades.

Stateless: each function takes the current state + thresholds and
returns a decision.  Lifecycle:

    z > z_entry  → open SHORT spread (sell y, buy β·x)
    z < -z_entry → open LONG spread  (buy y, sell β·x)
    z crosses 0  → close (mean reversion done)
    |z| > z_stop → close (structural break, give up)
    held > max_hold_hours → close (horizon cap)
"""

from dataclasses import dataclass


@dataclass
class EntryDecision:
    open_side: int | None  # +1 long-spread (buy y), -1 short-spread, None = wait
    reason: str


@dataclass
class ExitDecision:
    close: bool
    reason: str
    detail: str = ""


def should_open(*, z: float, z_entry: float, adf_p: float, max_adf_p: float) -> EntryDecision:
    """Decide entry for a flat pair.  Refuses when cointegration is broken."""
    if adf_p > max_adf_p:
        return EntryDecision(
            None,
            f"cointegration broken (adf_p={adf_p:.3f} > {max_adf_p})",
        )
    if z >= z_entry:
        return EntryDecision(
            -1,
            f"z={z:+.2f} ≥ +{z_entry}: spread overshoot → short spread",
        )
    if z <= -z_entry:
        return EntryDecision(
            +1,
            f"z={z:+.2f} ≤ -{z_entry}: spread undershoot → long spread",
        )
    return EntryDecision(None, f"z={z:+.2f} inside ±{z_entry} — wait")


def should_close(
    *, position_side: int, z: float, held_hours: float, z_stop: float, max_hold_hours: int
) -> ExitDecision:
    """Decide whether to close an open spread position."""
    # 1. Structural-break stop
    if abs(z) >= z_stop:
        return ExitDecision(True, "stop", f"|z|={abs(z):.2f} ≥ {z_stop}")
    # 2. Mean-reversion exit (z crossed 0 from the entry side)
    if position_side == +1 and z >= 0:
        return ExitDecision(True, "mean_rev", f"z={z:+.2f} crossed 0 from below")
    if position_side == -1 and z <= 0:
        return ExitDecision(True, "mean_rev", f"z={z:+.2f} crossed 0 from above")
    # 3. Horizon cap
    if held_hours >= max_hold_hours:
        return ExitDecision(True, "horizon", f"held {held_hours:.1f}h ≥ {max_hold_hours}h cap")
    return ExitDecision(False, "hold", f"z={z:+.2f}, held {held_hours:.1f}h")
