"""Telegram command handler for the scalp bot (and re-usable for futbot).

Why a separate file from `telegram_notifier.py`:
  * notifier sends OUTBOUND only (alerts).  Doesn't poll.
  * this module RECEIVES inbound /commands and replies.  Polls the bot.

The two run side-by-side in the same process.  Using `bot.send_message`
(via plain Bot) does NOT conflict with `application.updater.start_polling`
(via Application) — Tinkoff's getUpdates lock is per-CONNECTION, not
per-token, and the two pieces of code open separate connections.

Public API:
    cmd = TelegramCommandServer(token, chat_id, handlers)
    await cmd.start()    # spawns polling
    ...
    await cmd.stop()

`handlers` is a dict of {command_name: async_callable(update) → str}.
The server registers each as a Telegram CommandHandler and replies with
the returned string.  Errors caught and reported.

Authorization: only the configured `chat_id` is allowed to invoke
commands.  Anyone else gets "Not authorized".  This prevents random
chat IDs from probing your bot.
"""

import asyncio
import logging
from typing import Awaitable, Callable

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

logger = logging.getLogger("futbot.telegram_commands")


# A handler returns a string that will be sent back to the user.
# It can be async or sync.  No special context — keep API minimal.
HandlerFunc = Callable[[], Awaitable[str] | str]


class TelegramCommandServer:
    def __init__(self, *, token: str, chat_id: int, handlers: dict[str, HandlerFunc]):
        self.token = token
        self.chat_id = int(chat_id) if chat_id else 0
        self.handlers = handlers
        self._app: Application | None = None
        self._task: asyncio.Task | None = None

    async def start(self):
        """Non-blocking start.  If Telegram is unreachable right now (typical
        in RU on flaky days), we spawn a background task that keeps retrying
        with exponential backoff so the scalp trading loop never waits on it.

        The bot is fully functional without command access — only inbound
        /commands are unavailable until the polling task connects.
        """
        if self.chat_id == 0:
            logger.warning(
                "TELEGRAM_CHAT_ID not set; command server disabled. "
                "Run: python -m futbot.scalp.get_chat_id"
            )
            return
        if not self.token:
            logger.warning("TELEGRAM_BOT_TOKEN missing; command server disabled.")
            return

        try:
            self._app = Application.builder().token(self.token).build()
        except Exception as e:
            logger.error(f"Telegram Application build failed: {e}")
            return

        # Register handlers
        for cmd_name, handler in self.handlers.items():
            self._app.add_handler(CommandHandler(cmd_name, self._make_wrapper(cmd_name, handler)))
        if "help" not in self.handlers:
            self._app.add_handler(
                CommandHandler("help", self._make_wrapper("help", self._help_text))
            )

        # Background retry-loop until polling is alive
        self._task = asyncio.create_task(self._connect_loop(), name="tg-cmd-connect")

    async def _connect_loop(self):
        backoff = 5.0
        attempt = 0
        while True:
            attempt += 1
            try:
                await self._app.initialize()
                await self._app.start()
                await self._app.updater.start_polling(drop_pending_updates=True)
                logger.info(
                    f"Telegram commands ready (chat_id={self.chat_id}); "
                    f"available: /{' /'.join(sorted(self.handlers))}"
                    + (" /help" if "help" not in self.handlers else "")
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"Telegram polling start attempt {attempt} failed "
                    f"({type(e).__name__}: {e}); retrying in {backoff:.0f}s"
                )
                # Tear down half-initialised state
                try:
                    await self._app.updater.stop()
                except Exception:
                    pass
                try:
                    await self._app.stop()
                except Exception:
                    pass
                try:
                    await self._app.shutdown()
                except Exception:
                    pass
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2, 300.0)

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.warning(f"Command server shutdown error (ignored): {e}")

    # ── Internals ───────────────────────────────────────────────────────
    def _make_wrapper(self, cmd_name: str, fn: HandlerFunc):
        """Wrap a HandlerFunc in a Telegram-PTB CommandHandler callable."""

        async def _wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
            if update.effective_chat is None:
                return
            if update.effective_chat.id != self.chat_id:
                await update.message.reply_text("Not authorized.")
                return
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    result = await result
                reply = str(result) if result is not None else "(empty)"
            except Exception as e:
                logger.exception(f"command /{cmd_name} failed: {e}")
                reply = f"⚠ /{cmd_name} error: {type(e).__name__}: {e}"
            # Telegram has 4096 char limit; truncate just in case
            if len(reply) > 4000:
                reply = reply[:3990] + "\n…(truncated)"
            await update.message.reply_text(
                reply,
                parse_mode="HTML" if "<" in reply else None,
            )

        return _wrapped

    async def _help_text(self) -> str:
        lines = ["Available commands:"]
        for name in sorted(self.handlers):
            lines.append(f"  /{name}")
        lines.append("  /help")
        return "\n".join(lines)
