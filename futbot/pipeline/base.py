"""Shared types for the pipeline.

A `LayerResult` is the standardised return from every layer.  The orchestrator
in decision.py walks layers in order; the first one to return `vote=0` with
`vetoes=True` stops the chain.  Layers that have no opinion (e.g. ML gate
that's not loaded yet) return `vetoes=False` so the chain continues.
"""

from dataclasses import dataclass, field
from typing import Literal

Vote = Literal[-1, 0, 1]  # -1 short, 0 hold/no opinion, +1 long


@dataclass
class LayerResult:
    layer: str
    vote: Vote = 0
    vetoes: bool = False  # if True and vote=0 → chain stops here
    reason: str = ""  # human-readable
    detail: dict = field(default_factory=dict)  # numeric fields for logging/DB

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "vote": self.vote,
            "vetoes": self.vetoes,
            "reason": self.reason,
            **self.detail,
        }
