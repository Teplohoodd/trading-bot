"""Notification service: consumes queue and sends formatted alerts via Telegram."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

from telegram_bot.formatters import (
    format_trade_opened,
    format_trade_closed,
    format_signal,
    format_risk_status,
    format_advisory_signal,
)

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    TRADE_OPENED = "trade_opened"
    TRADE_CLOSED = "trade_closed"
    SIGNAL = "signal"
    ADVISORY_SIGNAL = "advisory_signal"
    RISK_ALERT = "risk_alert"
    MODEL_TRAINED = "model_trained"
    SPREAD_ALERT = "spread_alert"
    ERROR = "error"
    INFO = "info"


@dataclass
class Notification:
    type: NotificationType
    title: str
    data: dict = field(default_factory=dict)
    priority: str = "medium"  # low, medium, high, critical
    timestamp: datetime = field(default_factory=datetime.utcnow)


class NotificationService:
    """Consumes notification queue and sends to Telegram."""

    def __init__(self, bot: Bot, chat_id: int, queue: asyncio.Queue):
        self.bot = bot
        self.chat_id = chat_id
        self.queue = queue
        self._running = False

    async def run(self):
        """Main loop: consume queue and send messages."""
        self._running = True
        logger.info("Notification service started")

        while self._running:
            try:
                notification = await asyncio.wait_for(self.queue.get(), timeout=5.0)
                await self._send(notification)
                self.queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Notification error: {e}")
                await asyncio.sleep(1)

    async def stop(self):
        self._running = False

    async def _send(self, notification: Notification):
        """Format and send notification based on type."""
        if self.chat_id == 0:
            logger.warning("No chat_id set, skipping notification")
            return

        try:
            text = self._format(notification)
            kwargs: dict = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"}

            # Advisory signals get Approve/Reject inline keyboard
            if notification.type == NotificationType.ADVISORY_SIGNAL:
                sid = notification.data.get("signal_id", "")
                kwargs["reply_markup"] = InlineKeyboardMarkup(
                    [
                        [
                            InlineKeyboardButton(
                                "✅ Execute", callback_data=f"advisory:execute:{sid}"
                            ),
                            InlineKeyboardButton("❌ Skip", callback_data=f"advisory:skip:{sid}"),
                        ]
                    ]
                )

            await self.bot.send_message(**kwargs)
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def _format(self, n: Notification) -> str:
        """Format notification to HTML text."""
        formatters = {
            NotificationType.TRADE_OPENED: lambda: format_trade_opened(n.data),
            NotificationType.TRADE_CLOSED: lambda: format_trade_closed(n.data),
            NotificationType.SIGNAL: lambda: format_signal(n.data),
            NotificationType.ADVISORY_SIGNAL: lambda: format_advisory_signal(n.data),
            NotificationType.RISK_ALERT: lambda: f"<b>RISK ALERT</b>\n\n{n.title}\n{n.data.get('details', '')}",
            NotificationType.SPREAD_ALERT: lambda: f"<b>SPREAD ALERT</b>\n\n{n.title}",
            NotificationType.MODEL_TRAINED: lambda: (
                f"<b>Model Retrained</b>\n\n"
                f"Ticker: {n.data.get('ticker', 'universal')}\n"
                f"Accuracy: {n.data.get('accuracy', 0):.1%}\n"
                f"F1: {n.data.get('f1', 0):.1%}\n"
                f"Samples: {n.data.get('samples', 0)}"
            ),
            NotificationType.ERROR: lambda: f"<b>ERROR</b>\n\n{n.title}",
            NotificationType.INFO: lambda: f"<b>INFO</b>\n\n{n.title}",
        }

        formatter = formatters.get(n.type, lambda: f"<b>{n.title}</b>")
        return formatter()
