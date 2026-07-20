"""futbot web app — Telegram Mini App backend + static front.

Run:   python -m futbot.webapp.server          (default port 8088)

Serves:
  /                       — the mini app (static/index.html)
  /api/summary            — equity, realized totals, per-strategy stats
  /api/positions          — open positions with live price / stop / target
  /api/trades?days=30     — closed trades feed
  /api/stats              — equity curve points + aggregates
  /api/feed               — recent bot events parsed from orchestrator.log
  /api/candles?stock=SBER — 2h candles (stock) + open-position overlay levels

Read-only: SQLite opened per-request in ro mode; its own BrokerClient
(app_name futbot-webapp) only fetches prices/candles — never orders.

For Telegram Mini App: expose via an HTTPS tunnel, e.g.
  cloudflared tunnel --url http://localhost:8088
put the https URL into .env as WEBAPP_URL=…  then /app in the bot chat
sends the button that opens this UI inside Telegram.
"""

import asyncio
import json
import logging
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import Settings                      # noqa: E402
from core.broker import BrokerClient                      # noqa: E402
from futbot.breakdown.config import STOCK_FIGI            # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | webapp | %(message)s")
log = logging.getLogger("webapp")

DATA = ROOT / "data"
STATIC = Path(__file__).parent / "static"
PORT = 8088


def q(db: str, sql: str, args=()) -> list[dict]:
    con = sqlite3.connect(f"file:{DATA / db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()


class Ctx:
    broker: BrokerClient | None = None
    lock = asyncio.Lock()

    @classmethod
    async def get_broker(cls) -> BrokerClient:
        async with cls.lock:
            if cls.broker is None:
                s = Settings()
                b = BrokerClient(token=s.T_INVEST_TOKEN,
                                 account_id=s.T_INVEST_ACCOUNT_ID,
                                 app_name="futbot-webapp")
                await b.connect()
                cls.broker = b
            return cls.broker


# ── helpers ──────────────────────────────────────────────────────────────
def strategy_stats():
    out = {}
    for name, db, tab, pnl in (("breakdown", "breakdown.db", "bd_trades", "pnl_rub"),
                               ("trend", "trend.db", "trend_trades", "pnl"),
                               ("carry", "carry.db", "pair_trades", "pnl_rub")):
        rows = q(db, f"SELECT {pnl} p, exit_time FROM {tab} "
                     f"WHERE exit_time IS NOT NULL AND COALESCE(paper,0)=0")
        n = len(rows)
        wins = sum(1 for r in rows if (r["p"] or 0) > 0)
        net = sum(r["p"] or 0 for r in rows)
        today = datetime.utcnow().date().isoformat()
        tnet = sum(r["p"] or 0 for r in rows if (r["exit_time"] or "")[:10] == today)
        week = (datetime.utcnow() - timedelta(days=7)).isoformat()
        wnet = sum(r["p"] or 0 for r in rows if (r["exit_time"] or "") >= week)
        out[name] = {"n": n, "win": round(100 * wins / n) if n else 0,
                     "net": round(net), "today": round(tnet), "week": round(wnet)}
    return out


def open_rows():
    rows = []
    for r in q("breakdown.db",
               "SELECT id,stock name,fut_ticker ticker,fut_figi figi,lots,"
               "entry_price,entry_time,stop_price,target_price "
               "FROM bd_trades WHERE exit_time IS NULL"):
        rows.append({**r, "strategy": "breakdown", "direction": "sell"})
    for r in q("trend.db",
               "SELECT id,base name,ticker,figi,lots,direction,"
               "entry_price,entry_time,stop_price,target_price "
               "FROM trend_trades WHERE exit_time IS NULL"):
        rows.append({**r, "strategy": "trend"})
    return rows


LOG_PAT = re.compile(
    r"^(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*?\| (?P<src>orchestrator[^ ]*) \| "
    r"(?P<msg>.*)$")
KEEP = ("SIGNAL", "SHORT", "CLOSE", "heartbeat", "R:R gate", "PANIC",
        "re-anchored", "reconcil", "cooldown", "per-bar", "stop @",
        "Shutdown", "starting")


def parse_feed(limit=80):
    path = DATA / "logs" / "orchestrator.log"
    try:
        # tail ~4000 lines cheaply
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 800_000))
            lines = f.read().decode("utf-8", "replace").splitlines()[1:]
    except Exception:
        return []
    out = []
    for ln in reversed(lines):
        if not any(k in ln for k in KEEP):
            continue
        m = LOG_PAT.match(ln)
        if not m:
            continue
        msg = m.group("msg").strip()
        kind = ("open" if ("SHORT" in msg or "SIGNAL" in msg) else
                "close" if "CLOSE" in msg else
                "guard" if ("gate" in msg or "PANIC" in msg or "cooldown" in msg
                            or "per-bar" in msg) else
                "sys")
        out.append({"ts": m.group("ts"), "src": m.group("src").split(".")[-1],
                    "kind": kind, "msg": msg[:220]})
        if len(out) >= limit:
            break
    return out


# ── API handlers ─────────────────────────────────────────────────────────
async def api_summary(request):
    st = strategy_stats()
    total = sum(v["net"] for v in st.values())
    equity = None
    try:
        b = await Ctx.get_broker()
        summ, ok = await b.get_margin_summary()
        if ok:
            equity = round(summ.get("liquid", 0))
    except Exception as e:
        log.warning(f"equity fetch: {e}")
    return web.json_response({
        "equity": equity, "realized_total": total, "strategies": st,
        "open_count": len(open_rows()),
        "ts": datetime.now(timezone.utc).isoformat()})


async def api_positions(request):
    rows = open_rows()
    prices, upnl = {}, {}
    try:
        b = await Ctx.get_broker()
        detail, ok = await b.get_positions_detail()
        for r in rows:
            try:
                prices[r["figi"]] = float(await b.get_last_price(r["figi"]))
            except Exception:
                prices[r["figi"]] = None
            if ok:
                upnl[r["figi"]] = float(
                    detail.get(r["figi"], {}).get("unrealized", 0.0))
    except Exception as e:
        log.warning(f"positions broker: {e}")
    out = []
    for r in rows:
        px = prices.get(r["figi"])
        e, st, tg = r["entry_price"], r["stop_price"], r["target_price"]
        sign = -1 if r["direction"] == "sell" else 1
        prog = None
        if px and st and tg and st != tg:
            prog = max(0.0, min(1.0, (px - st) / (tg - st)))  # 0 at stop → 1 at target
        out.append({**r, "price": px, "upnl": upnl.get(r["figi"]),
                    "chg_pct": round(sign * (px / e - 1) * 100, 2) if px and e else None,
                    "progress": prog})
    return web.json_response(out)


async def api_trades(request):
    days = int(request.query.get("days", "30"))
    cut = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = []
    for r in q("breakdown.db",
               "SELECT id,stock name,lots,entry_time,entry_price,exit_time,"
               "exit_price,exit_reason,pnl_rub pnl FROM bd_trades "
               "WHERE exit_time>=? AND COALESCE(paper,0)=0 ORDER BY exit_time DESC",
               (cut,)):
        rows.append({**r, "strategy": "breakdown", "direction": "sell"})
    for r in q("trend.db",
               "SELECT id,base name,lots,direction,entry_time,entry_price,"
               "exit_time,exit_price,exit_reason,pnl FROM trend_trades "
               "WHERE exit_time>=? AND COALESCE(paper,0)=0 ORDER BY exit_time DESC",
               (cut,)):
        rows.append({**r, "strategy": "trend"})
    rows.sort(key=lambda r: r["exit_time"] or "", reverse=True)
    return web.json_response(rows)


async def api_stats(request):
    rows = []
    for db, tab, pnl, strat in (("breakdown.db", "bd_trades", "pnl_rub", "breakdown"),
                                ("trend.db", "trend_trades", "pnl", "trend"),
                                ("carry.db", "pair_trades", "pnl_rub", "carry")):
        for r in q(db, f"SELECT exit_time t,{pnl} p FROM {tab} "
                       f"WHERE exit_time IS NOT NULL AND COALESCE(paper,0)=0"):
            rows.append({**r, "s": strat})
    rows.sort(key=lambda r: r["t"] or "")
    curve, cum = [], 0.0
    for r in rows:
        cum += r["p"] or 0
        curve.append({"t": (r["t"] or "")[:16], "v": round(cum), "s": r["s"],
                      "p": round(r["p"] or 0)})
    # reason breakdown
    reasons = {}
    for r in q("breakdown.db", "SELECT exit_reason rr,COUNT(*) n,SUM(pnl_rub) p "
               "FROM bd_trades WHERE exit_time IS NOT NULL AND COALESCE(paper,0)=0 "
               "GROUP BY 1") + \
             q("trend.db", "SELECT exit_reason rr,COUNT(*) n,SUM(pnl) p "
               "FROM trend_trades WHERE exit_time IS NOT NULL AND COALESCE(paper,0)=0 "
               "GROUP BY 1"):
        key = (r["rr"] or "?").split("(")[0]
        a = reasons.setdefault(key, {"n": 0, "p": 0})
        a["n"] += r["n"]; a["p"] += round(r["p"] or 0)
    return web.json_response({"curve": curve, "reasons": reasons,
                              "strategies": strategy_stats()})


async def api_candles(request):
    stock = request.query.get("stock", "SBER").upper()
    days = int(request.query.get("days", "12"))
    figi = STOCK_FIGI.get(stock)
    if not figi:
        return web.json_response({"error": f"unknown stock {stock}"}, status=400)
    from t_tech.invest.schemas import CandleInterval
    from t_tech.invest.utils import quotation_to_decimal
    b = await Ctx.get_broker()
    now = datetime.now(timezone.utc)
    candles = await b.get_candles(figi, now - timedelta(days=days), now,
                                  CandleInterval.CANDLE_INTERVAL_HOUR)
    import pandas as pd
    df = pd.DataFrame([{"time": c.time,
                        "open": float(quotation_to_decimal(c.open)),
                        "high": float(quotation_to_decimal(c.high)),
                        "low": float(quotation_to_decimal(c.low)),
                        "close": float(quotation_to_decimal(c.close)),
                        "volume": c.volume} for c in candles])
    if len(df):
        d = (df.set_index("time").resample("2h")
               .agg({"open": "first", "high": "max", "low": "min",
                     "close": "last", "volume": "sum"}).dropna().reset_index())
    else:
        d = df
    out = [{"time": int(r["time"].timestamp()), "open": r["open"],
            "high": r["high"], "low": r["low"], "close": r["close"],
            "volume": r["volume"]} for _, r in d.iterrows()]
    # overlay: open breakdown position on this stock (stock-space levels)
    lvl = None
    br = q("breakdown.db", "SELECT stock_entry,stop_price,target_price,"
           "entry_price,fut_ticker FROM bd_trades WHERE exit_time IS NULL "
           "AND stock=?", (stock,))
    if br:
        r = br[0]
        # futures-space → stock-space via entry ratio
        k = (r["stock_entry"] / r["entry_price"]) if r["entry_price"] else None
        if k:
            lvl = {"entry": r["stock_entry"],
                   "stop": r["stop_price"] * k,
                   "target": r["target_price"] * k,
                   "ticker": r["fut_ticker"]}
    return web.json_response({"stock": stock, "candles": out, "levels": lvl,
                              "universe": sorted(STOCK_FIGI.keys())})


async def api_feed(request):
    return web.json_response(parse_feed())


async def index(request):
    return web.FileResponse(STATIC / "index.html")


def main():
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/summary", api_summary)
    app.router.add_get("/api/positions", api_positions)
    app.router.add_get("/api/trades", api_trades)
    app.router.add_get("/api/stats", api_stats)
    app.router.add_get("/api/candles", api_candles)
    app.router.add_get("/api/feed", api_feed)
    app.router.add_static("/static", STATIC)
    log.info(f"futbot webapp on http://localhost:{PORT}")
    web.run_app(app, port=PORT, print=None)


if __name__ == "__main__":
    main()
