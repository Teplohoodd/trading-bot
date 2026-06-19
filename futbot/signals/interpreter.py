"""Heuristic first-pass interpreter for the "📈" channel's crypto calls.

The trader writes cryptically, so this is NOT a full parser — it's a triage:
  • HIGH confidence  → clear instrument + direction + an actionable entry verb
                       ("шортю", "лонганул", "пробуйте short", "в шорт", limit).
  • LOW  confidence  → cryptic / commentary / multi-step / hindsight ("Готово ✅",
                       "ждем", "weekly review") → ESCALATE to LLM/human judgement
                       (this is exactly where I misread the 2026-06-01 post).

So the bot never acts on a low-confidence parse on its own — those go to the
Claude-in-the-loop session (me) or to the user for confirmation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Instrument detection → Neo perpetual ticker
INSTRUMENTS = {
    "BTC": ["btc", "биток", "битк", "bitcoin"],
    "ETH": ["eth", "эфир", "ethereum"],
    "SOL": ["sol", "солан", "solana"],
}
NEO_TICKER = {"BTC": "BTCUSDperpA", "ETH": "ETHUSDperpA", "SOL": "SOLUSDperpA"}

# Direction cues
SHORT_CUES = ["short", "шорт", "продаж", "в шорт", "по'short", "short'"]
LONG_CUES = ["long", "лонг", "покуп", "в лонг", "лонган", "long'"]

# Actionable-entry verbs (the trader is opening / telling you to open NOW)
ACTION_CUES = [
    "открыл",
    "зашел",
    "зашёл",
    "шортю",
    "шортчу",
    "лонгую",
    "лонганул",
    "лонганул",
    "пробуйте",
    "пробуй",
    "лимитк",
    "поставил",
    "в шорт",
    "в лонг",
    "short'чу",
    "short'ю",
    "откры",
]
# Commentary / hindsight / wait → NOT actionable on their own
NONACTION_CUES = [
    "готово",
    "✅",
    "ждем",
    "ждём",
    "weekly review",
    "реализова",
    "цель достигнут",
    "обратите внимание",
    "так и получилось",
    "было ✅",
    "не перепутайте",
    "продублирую",
    "нарратив",
    "для общего развития",
]


@dataclass
class Signal:
    raw: str
    instrument: str | None = None  # BTC / ETH / SOL
    neo_ticker: str | None = None
    direction: int = 0  # +1 long, -1 short, 0 none
    actionable: bool = False  # is this a fresh entry instruction?
    confidence: str = "low"  # high / low
    levels: list[float] = field(default_factory=list)
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "neo": self.neo_ticker,
            "dir": "LONG" if self.direction > 0 else "SHORT" if self.direction < 0 else "-",
            "actionable": self.actionable,
            "confidence": self.confidence,
            "levels": self.levels,
            "reason": self.reason,
        }


def _find_levels(text: str) -> list[float]:
    out = []
    # $2 300 / $79 000 / 80.303 / 1639.12 / 82-х
    for m in re.findall(r"\$?\s?(\d[\d\s.,]{1,9}\d|\d)", text):
        s = m.replace(" ", "").replace(",", "")
        try:
            v = float(s)
            if 0.01 <= v <= 200000:
                out.append(v)
        except Exception:
            pass
    return out[:6]


def interpret(text: str) -> Signal:
    """First-pass triage of one message."""
    sig = Signal(raw=text or "")
    low = (text or "").lower()
    if not low.strip():
        sig.reason = "empty (likely image-only — needs vision/LLM)"
        return sig

    # instrument
    for sym, cues in INSTRUMENTS.items():
        if any(c in low for c in cues):
            sig.instrument = sym
            sig.neo_ticker = NEO_TICKER[sym]
            break
    # direction
    sh = any(c in low for c in SHORT_CUES)
    lo = any(c in low for c in LONG_CUES)
    if sh and not lo:
        sig.direction = -1
    elif lo and not sh:
        sig.direction = +1
    elif sh and lo:
        sig.direction = 0  # both mentioned → ambiguous
    # actionable vs commentary
    has_action = any(c in low for c in ACTION_CUES)
    is_comment = any(c in low for c in NONACTION_CUES)
    sig.actionable = has_action and not is_comment
    sig.levels = _find_levels(text)

    # confidence
    if sig.instrument and sig.direction != 0 and sig.actionable:
        sig.confidence = "high"
        sig.reason = "clear instrument+direction+entry verb"
    else:
        sig.confidence = "low"
        bits = []
        if not sig.instrument:
            bits.append("no instrument")
        if sig.direction == 0:
            bits.append("ambiguous/!direction")
        if not sig.actionable:
            bits.append("commentary/hindsight" if is_comment else "no entry verb")
        sig.reason = "needs LLM/human: " + ", ".join(bits)
    return sig


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # Smoke test on real channel messages
    samples = [
        "Пробуйте Short'ить. Если не вернемся в FvG ... то это поход под 75k$.",
        "Long'анул ETH в диапазон $2 400 -$2 500.",
        "Short ETH под $2 300.",
        "SOL Short'чу в район $81.5-$82 под начало недели.",
        "До $1540 еще открыл Short на небольшую часть депозита.",
        "Теперь ждем, чтобы BTC дал отскок выше ✅.",  # commentary (I misread this)
        "Готово ✅.",
        "Провели weekly review. Ждем слабости от ETH и SOL ... перед июньским падением.",
    ]
    for s in samples:
        sig = interpret(s)
        print(f"[{sig.confidence:>4}] {sig.as_dict()}\n   « {s[:70]} »")
