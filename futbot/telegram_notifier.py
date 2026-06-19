"""Lightweight Telegram notifier for futbot.

Doesn't reuse trade_claude's full TelegramBot (we don't need command
handling here — futbot is autonomous and one-way reporting is enough).
Instead, just uses telegram.Bot.send_message directly.

Messages are non-blocking: pushed to an asyncio.Queue and drained by a
background task.  If the queue fills (network down), oldest messages are
dropped — we never block the main trading loop on Telegram.

Sent message types:
  * TRADE_OPENED   — entry placed (paper or live)
  * TRADE_CLOSED   — exit, with realised P&L
  * CIRCUIT        — kill-switch tripped / reset
  * BOOT           — process started (mode, universe count)
  * ERROR          — exception in a critical path (rare)
  * DAILY_SUMMARY  — once per UTC midnight if there were trades that day

Every message is prefixed [PAPER] or [LIVE] so it's never ambiguous which
bot it's from when futbot + trade_claude run in parallel.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from telegram import Bot

logger = logging.getLogger("futbot.telegram")


QUEUE_MAX = 200


class MsgType(Enum):
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    CIRCUIT = "circuit"
    BOOT = "boot"
    ERROR = "error"
    DAILY_SUMMARY = "daily_summary"
    INFO = "info"


@dataclass
class Msg:
    type: MsgType
    title: str
    body: str = ""
    ts: datetime = field(default_factory=datetime.utcnow)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: int, paper: bool):
        self.token = token
        self.chat_id = int(chat_id) if chat_id else 0
        self.paper = paper
        self._bot: Bot | None = None
        self._queue: asyncio.Queue[Msg] = asyncio.Queue(maxsize=QUEUE_MAX)
        self._task: asyncio.Task | None = None
        self._running = False
        self._prefix = "[PAPER]" if paper else "[LIVE]"

    async def start(self):
        if self.chat_id == 0:
            logger.warning("TELEGRAM_CHAT_ID not set; notifier disabled (logs-only)")
            return
        try:
            self._bot = Bot(token=self.token)
            await self._bot.get_me()  # cheap connectivity test
        except Exception as e:
            logger.warning(f"Telegram bot init failed ({e}); notifier disabled")
            self._bot = None
            return
        self._running = True
        self._task = asyncio.create_task(self._drain(), name="futbot-telegram-drain")
        logger.info(f"Telegram notifier started (chat_id={self.chat_id}, mode={self._prefix})")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    def push(self, msg: Msg):
        """Non-blocking enqueue.  Drops on full so trading loop is never gated."""
        if self.chat_id == 0 or self._bot is None:
            return
        try:
            self._queue.put_nowait(msg)
        except asyncio.QueueFull:
            logger.warning("Telegram queue full; dropping message")

    async def _drain(self):
        while self._running:
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            try:
                text = self._format(msg)
                await self._bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning(f"Telegram send failed ({type(e).__name__}: {e})")
                # Don't crash the drainer; just continue
                await asyncio.sleep(2)

    def _format(self, m: Msg) -> str:
        emoji = {
            MsgType.TRADE_OPENED: "📈",
            MsgType.TRADE_CLOSED: "🏁",
            MsgType.CIRCUIT: "🛑",
            MsgType.BOOT: "🤖",
            MsgType.ERROR: "⚠",
            MsgType.DAILY_SUMMARY: "📊",
            MsgType.INFO: "ℹ",
        }.get(m.type, "•")
        body = f"\n{m.body}" if m.body else ""
        return f"{emoji} <b>{self._prefix} futbot — {m.title}</b>{body}"


# ── Helper builders (used by main.py) ────────────────────────────────────────
def fmt_trade_opened(
    *, ticker: str, direction: str, lots: int, entry: float, stop: float, reason_chain: dict | None
) -> Msg:
    body = (
        f"<b>{ticker}</b>: {direction.upper()} × {lots}\n"
        f"entry: {entry:.4f}\n"
        f"stop:  {stop:.4f}"
    )
    if reason_chain:
        body += "\n\n" + "\n".join(
            f"• {k}: {v.get('reason', '')[:80]}" if isinstance(v, dict) else f"• {k}: {v}"
            for k, v in reason_chain.items()
        )
    return Msg(MsgType.TRADE_OPENED, f"OPEN {ticker} {direction.upper()}", body)


def fmt_trade_closed(
    *,
    ticker: str,
    direction: str,
    lots: int,
    entry: float,
    exit_: float,
    pnl: float,
    pnl_pct: float,
    reason: str,
) -> Msg:
    sign = "+" if pnl >= 0 else ""
    body = (
        f"<b>{ticker}</b>: {direction.upper()} × {lots}\n"
        f"entry: {entry:.4f} → exit: {exit_:.4f}\n"
        f"P&L: <b>{sign}{pnl:.2f} ₽</b> ({sign}{pnl_pct:.2f}%)\n"
        f"reason: {reason}"
    )
    return Msg(MsgType.TRADE_CLOSED, f"CLOSE {ticker}", body)


def fmt_boot(*, mode: str, universe_count: int) -> Msg:
    return Msg(
        MsgType.BOOT,
        "Started",
        body=f"mode: {mode}\nuniverse: {universe_count} contracts",
    )


def fmt_circuit(reason: str) -> Msg:
    return Msg(MsgType.CIRCUIT, "Circuit tripped", body=reason)
