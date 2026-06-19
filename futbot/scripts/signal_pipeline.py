"""End-to-end demo of the '📈' signal pipeline on real recent messages.

Flow per message:
  1. heuristic interpret() → triage (high / low-confidence).
  2. low-confidence cryptic posts get an LLM/human override (here I supply the
     interpretation a Claude-in-the-loop session would make, using context like
     the attached PnL images and price levels — e.g. "$1540" ⇒ ETH).
  3. for each ACTIONABLE signal: fetch the Neo price, size 1 lot, check free
     margin (ГО × buffer), suggest a stop, and emit a PROPOSAL (never auto-fire).
  4. log everything to data/signals.db for honest forward stats.

Run:  python -u -m futbot.scripts.signal_pipeline
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import Settings
from core.broker import BrokerClient
from futbot.signals.interpreter import interpret, NEO_TICKER
from futbot.signals.db import SignalDB

# Recent channel messages (id, utc_time, text).  `override` = the resolved
# interpretation a Claude-in-the-loop session supplies for cryptic posts.
MESSAGES = [
    (741, "2026-05-24T16:46", "SOL Short'чу в район $81.5-$82 под начало недели.", None),
    (
        752,
        "2026-05-31T18:52",
        "Провели weekly review. Ждем слабости от ETH и SOL ... BTC рывок вверх перед июньским падением.",
        ("BTC", -1, False, "thesis: June fall — bias SHORT, no entry yet"),
    ),
    (
        755,
        "2026-06-01T15:45",
        "Теперь ждем, чтобы BTC дал отскок выше ✅.",
        ("BTC", 0, False, "commentary: waiting for bounce to short into — NO entry"),
    ),
    (
        759,
        "2026-06-05T14:30",
        "До $1540 еще открыл Short на небольшую часть депозита.",
        ("ETH", -1, True, "image+level $1540 ⇒ ETH short opened"),
    ),
]

MARGIN_BUFFER = 2.0
STOP_PCT = 0.06  # 6% protective stop on the Neo (≈ pattern stop scale)


async def main():
    s = Settings()
    b = BrokerClient(token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="sig-pipe")
    await b.connect()
    db = SignalDB(Path("data/signals.db"))
    await db.initialize()

    futs = await b.get_all_futures()
    by_ticker = {(getattr(f, "ticker", "") or ""): f for f in futs}
    usd = by_ticker.get("USDRUBF")
    fx = float(await b.get_last_price(usd.figi)) if usd else 80.0
    summ, ok = await b.get_margin_summary()
    free = summ.get("available", 0.0) if ok else 0.0

    print("=" * 90)
    print(f"'📈' SIGNAL PIPELINE  (FX {fx:.2f}, free margin {free:,.0f}₽)")
    print("=" * 90)

    for msg_id, t, text, override in MESSAGES:
        sig = interpret(text)
        src = "heuristic"
        if override is not None:
            inst, d, act, why = override
            sig.instrument = inst
            sig.neo_ticker = NEO_TICKER.get(inst)
            sig.direction = d
            sig.actionable = act
            sig.confidence = "high" if act else "low"
            sig.reason = f"LLM: {why}"
            src = "llm"
        print(f"\n[{msg_id}] {t}  «{text[:62]}»")
        print(
            f"   → {sig.instrument or '-'} {'LONG' if sig.direction>0 else 'SHORT' if sig.direction<0 else '-'}"
            f"  actionable={sig.actionable}  conf={sig.confidence}  ({src}: {sig.reason})"
        )

        proposal = None
        if sig.actionable and sig.neo_ticker and sig.direction != 0:
            f = by_ticker.get(sig.neo_ticker)
            if f is not None:
                px = float(await b.get_last_price(f.figi))
                meta = b.extract_futures_metadata(f)
                rr = float(meta.get("dlong") or 0.20)
                go = px * fx * 1 * rr
                need = go * MARGIN_BUFFER
                margin_ok = free >= need
                stop = px * (1 + STOP_PCT) if sig.direction < 0 else px * (1 - STOP_PCT)
                proposal = (
                    f"PROPOSE {('SHORT' if sig.direction<0 else 'LONG')} "
                    f"1× {sig.neo_ticker} @ ${px:.2f}  "
                    f"stop ${stop:.2f} ({STOP_PCT*100:.0f}%)  "
                    f"ГО {go:,.0f}₽ (lev {1/rr:.1f}×)  "
                    f"margin {'OK ✅' if margin_ok else 'BLOCKED ❌'}"
                )
                print(f"   💡 {proposal}")
        await db.log(
            msg_id=msg_id,
            msg_time=t,
            raw_text=text,
            instrument=sig.instrument,
            neo_ticker=sig.neo_ticker,
            direction=sig.direction,
            actionable=int(sig.actionable),
            confidence=sig.confidence,
            interpreted_by=src,
            status=("proposed" if proposal else "skipped"),
            notes=sig.reason,
        )

    st = await db.stats()
    print(f"\nSignal log: data/signals.db  | closed trades: {st['closed']}")
    await db.close()
    await b.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
