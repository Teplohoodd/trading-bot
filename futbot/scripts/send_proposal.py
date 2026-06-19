"""send_proposal.py — push a Neo trade PROPOSAL to the user's Telegram + log it.

Used in DISPATCH mode: the user forwards a (pre-validated) channel message, I
interpret it, then call this to send a clean trade proposal to the Telegram bot
and record it in data/signals.db.  Nothing is auto-executed — the user decides.

It pulls the live Neo price, USD/RUB FX, and free margin from the broker,
computes ГО (initial margin), checks the margin buffer, and suggests a stop.

Examples:
  python -u -m futbot.scripts.send_proposal --instrument SOL --dir short \
      --note "741: SOL Short'чу $81.5-82" --msg-id 741
  python -u -m futbot.scripts.send_proposal --instrument ETH --dir short \
      --lots 1 --stop-pct 6 --note "759: ETH short, target \$1540" --msg-id 759
"""

import argparse
import asyncio
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import Settings
from core.broker import BrokerClient
from futbot.signals.db import SignalDB

NEO_TICKER = {
    "BTC": "BTCUSDperpA",
    "ETH": "ETHUSDperpA",
    "SOL": "SOLUSDperpA",
    "XRP": "XRPUSDperpA",
    "TRX": "TRXUSDperpA",
    "TSLA": "TSLAperpA",
    "NBIS": "NBISperpA",
    "CVNA": "CVNAperpA",
    "APP": "APPperpA",
    "HOOD": "HOODperpA",
}
MARGIN_BUFFER = 2.0


def _send_telegram(token: str, chat_id: int, text: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
    ).encode()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=20) as r:
            return json.loads(r.read()).get("ok", False)
    except Exception as e:
        print(f"Telegram send failed: {e}")
        return False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", required=True, help="BTC/ETH/SOL/XRP/TRX/TSLA/...")
    ap.add_argument("--dir", required=True, choices=["long", "short"])
    ap.add_argument("--lots", type=int, default=1)
    ap.add_argument("--stop-pct", type=float, default=6.0)
    ap.add_argument("--note", default="", help="source message / rationale")
    ap.add_argument("--msg-id", type=int, default=0)
    args = ap.parse_args()

    sym = args.instrument.upper()
    ticker = NEO_TICKER.get(sym)
    if not ticker:
        print(f"Unknown instrument {sym}; known: {list(NEO_TICKER)}")
        return
    direction = +1 if args.dir == "long" else -1

    s = Settings()
    b = BrokerClient(token=s.T_INVEST_TOKEN, account_id=s.T_INVEST_ACCOUNT_ID, app_name="send-prop")
    await b.connect()
    futs = await b.get_all_futures()
    by_t = {(getattr(f, "ticker", "") or ""): f for f in futs}
    f = by_t.get(ticker)
    if f is None:
        print(f"{ticker} not found")
        await b.disconnect()
        return
    px = float(await b.get_last_price(f.figi))
    usd = by_t.get("USDRUBF")
    fx = float(await b.get_last_price(usd.figi)) if usd else 80.0
    meta = b.extract_futures_metadata(f)
    rr = float(meta.get("dlong") or 0.20)
    summ, ok = await b.get_margin_summary()
    free = summ.get("available", 0.0) if ok else 0.0
    suff = summ.get("sufficiency", 0.0) if ok else 0.0
    await b.disconnect()

    go = px * fx * args.lots * rr
    need = go * MARGIN_BUFFER
    margin_ok = ok and free >= need
    stop = px * (1 + args.stop_pct / 100) if direction < 0 else px * (1 - args.stop_pct / 100)
    side = "SHORT" if direction < 0 else "LONG"
    flag = "✅ маржа OK" if margin_ok else ("❌ МАРЖА МАЛА" if ok else "⚠ маржа неизвестна")

    text = (
        f"📡 <b>СИГНАЛ → ПРЕДЛОЖЕНИЕ</b>\n"
        f"<b>{side} {args.lots}× {sym}</b> ({ticker})\n"
        f"Цена: <b>${px:,.2f}</b>  |  стоп: ${stop:,.2f} ({args.stop_pct:.0f}%)\n"
        f"ГО: {go:,.0f}₽  плечо {1/rr:.1f}×  |  своб.маржа {free:,.0f}₽  запас {suff:.1f}×\n"
        f"{flag}\n"
        f"<i>Источник: {args.note}</i>\n\n"
        f"Подтверди вручную в приложении Т-Инвестиции, если согласен."
    )
    sent = _send_telegram(s.TELEGRAM_BOT_TOKEN, int(s.TELEGRAM_CHAT_ID), text)
    print(
        ("SENT ✅" if sent else "SEND FAILED")
        + " | "
        + text.replace("<b>", "")
        .replace("</b>", "")
        .replace("<i>", "")
        .replace("</i>", "")
        .replace("\n", " | ")
    )

    # log
    db = SignalDB(Path("data/signals.db"))
    await db.initialize()
    await db.log(
        msg_id=args.msg_id or int(datetime.utcnow().timestamp()),
        msg_time=datetime.utcnow().isoformat(),
        raw_text=args.note,
        instrument=sym,
        neo_ticker=ticker,
        direction=direction,
        actionable=1,
        confidence="high",
        interpreted_by="dispatch",
        status=("proposed" if margin_ok else "proposed_blocked"),
        proposed_entry=px,
        proposed_lots=args.lots,
        notes=f"stop {stop:.2f}; ГО {go:.0f}; margin_ok={margin_ok}",
    )
    await db.close()


if __name__ == "__main__":
    asyncio.run(main())
