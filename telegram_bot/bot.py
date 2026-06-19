"""Telegram bot setup and lifecycle management."""

import asyncio
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

from telegram_bot.handlers import (
    start_command,
    help_command,
    status_command,
    portfolio_command,
    pnl_command,
    positions_command,
    buy_command,
    sell_command,
    mode_command,
    risk_command,
    watchlist_command,
    retrain_command,
    stop_command,
    profile_command,
    addticker_command,
    removeticker_command,
    signals_command,
    sync_command,
    analyze_command,
    callback_handler,
)
from telegram_bot.notifications import NotificationService


async def _error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Catch-all handler so exceptions in command handlers never vanish silently.

    Without this, python-telegram-bot logs un-caught exceptions at DEBUG
    level — which our main.py silences to WARNING.  Result: handler crashes
    were invisible in the console ("bot stopped responding, no errors").
    """
    err = context.error
    where = ""
    if isinstance(update, Update):
        if update.message:
            where = f" (msg: {update.message.text[:60] if update.message.text else '?'})"
        elif update.callback_query:
            where = f" (callback: {update.callback_query.data[:60]})"
    logger.error(f"Telegram handler error{where}: {err!r}", exc_info=err)
    # Best-effort user-visible reply
    try:
        if isinstance(update, Update) and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"⚠️ Internal error: {type(err).__name__}: {err}",
            )
    except Exception:
        pass


logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot with inline keyboards and notification service."""

    def __init__(self, token: str, engine, db, notification_queue: asyncio.Queue, settings):
        self.token = token
        self.engine = engine
        self.db = db
        self.notification_queue = notification_queue
        self.settings = settings
        self._app: Application | None = None
        self._notification_service: NotificationService | None = None

    async def start(self):
        """Initialize and start the Telegram bot (non-blocking).

        Network failures during initialization (no internet, blocked DNS,
        proxy down) are caught here and retried — the trading engine is
        running in parallel and must not be killed by a Telegram outage.
        """
        self._app = Application.builder().token(self.token).build()

        # Store shared objects in bot_data for handlers
        self._app.bot_data["engine"] = self.engine
        self._app.bot_data["db"] = self.db
        self._app.bot_data["settings"] = self.settings

        # Register command handlers
        self._app.add_handler(CommandHandler("start", start_command))
        self._app.add_handler(CommandHandler("help", help_command))
        self._app.add_handler(CommandHandler("status", status_command))
        self._app.add_handler(CommandHandler("portfolio", portfolio_command))
        self._app.add_handler(CommandHandler("pnl", pnl_command))
        self._app.add_handler(CommandHandler("positions", positions_command))
        self._app.add_handler(CommandHandler("buy", buy_command))
        self._app.add_handler(CommandHandler("sell", sell_command))
        self._app.add_handler(CommandHandler("mode", mode_command))
        self._app.add_handler(CommandHandler("risk", risk_command))
        self._app.add_handler(CommandHandler("watchlist", watchlist_command))
        self._app.add_handler(CommandHandler("retrain", retrain_command))
        self._app.add_handler(CommandHandler("stop", stop_command))
        self._app.add_handler(CommandHandler("profile", profile_command))
        self._app.add_handler(CommandHandler("addticker", addticker_command))
        self._app.add_handler(CommandHandler("removeticker", removeticker_command))
        self._app.add_handler(CommandHandler("signals", signals_command))
        self._app.add_handler(CommandHandler("sync", sync_command))
        self._app.add_handler(CommandHandler("analyze", analyze_command))

        # Register callback handler for all inline buttons
        self._app.add_handler(CallbackQueryHandler(callback_handler))

        # Catch-all error handler (must be after all other handlers)
        self._app.add_error_handler(_error_handler)

        # Create notification service
        self._notification_service = NotificationService(
            bot=self._app.bot,
            chat_id=self.settings.TELEGRAM_CHAT_ID,
            queue=self.notification_queue,
        )
        self._app.bot_data["notification_service"] = self._notification_service

        # Initialize and start polling with retries — without this, a
        # connect-error at boot crashes the whole telegram task and (under
        # the old supervisor) used to take the engine down with it.
        import asyncio as _asyncio

        boot_backoff = 5
        while True:
            try:
                await self._app.initialize()
                await self._app.start()
                await self._app.updater.start_polling(drop_pending_updates=True)
                break
            except _asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(
                    f"Telegram bootstrap failed ({type(e).__name__}: {e}); "
                    f"retrying in {boot_backoff}s"
                )
                # Tear down whatever started so the next attempt is clean
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
                await _asyncio.sleep(boot_backoff)
                boot_backoff = min(boot_backoff * 2, 300)

        logger.info("Telegram bot started")

        # Run notification service alongside a polling watchdog.
        #
        # Why a heartbeat-based watchdog (not just `updater.running`):
        # python-telegram-bot's polling loop catches NetworkError internally
        # and keeps `running=True` even when getUpdates has been failing for
        # hours.  Symptom: bot looks alive in logs but stops responding to
        # commands after a long network blip.  Solution: ping `bot.get_me()`
        # periodically; on consecutive failures, force a full restart of the
        # updater (stop → start_polling) — this re-creates the underlying
        # httpx pool, recovering from stuck connections.
        import asyncio as _asyncio

        async def _polling_watchdog():
            # consecutive_failures = number of heartbeats failed in a row.
            # restart_backoff = seconds to wait before retrying after a failed
            # restart attempt.  Without backoff, the watchdog used to spam
            # restart() every heartbeat (every 120s) for as long as the
            # network was down — generating thousands of stack traces.
            consecutive_failures = 0
            restart_backoff = 0.0
            await _asyncio.sleep(30)  # initial settle period
            while True:
                # If a previous restart failed, wait extra before next probe.
                # Plain heartbeat interval otherwise.
                await _asyncio.sleep(max(120.0, restart_backoff))
                try:
                    me_task = self._app.bot.get_me()
                    await _asyncio.wait_for(me_task, timeout=15)
                    if consecutive_failures > 0:
                        logger.info(
                            f"Telegram heartbeat recovered after "
                            f"{consecutive_failures} failure(s)"
                        )
                    consecutive_failures = 0
                    restart_backoff = 0.0
                except Exception as e:
                    consecutive_failures += 1
                    logger.warning(
                        f"Telegram heartbeat failed "
                        f"({consecutive_failures}/3): {type(e).__name__}: {e}"
                    )
                    # Only attempt restart at the threshold AND every 3rd failure
                    # afterwards — keeps logs quiet during long outages while
                    # still retrying periodically.  Engine keeps trading.
                    if consecutive_failures >= 3 and consecutive_failures % 3 == 0:
                        logger.error(
                            f"Telegram heartbeat dead {consecutive_failures}× — "
                            "force-restarting polling"
                        )
                        try:
                            await self._app.updater.stop()
                        except Exception as stop_err:
                            logger.warning(f"Updater.stop() error (ignored): {stop_err}")
                        try:
                            await _asyncio.sleep(2)
                            await self._app.updater.start_polling(drop_pending_updates=False)
                            logger.info("Telegram polling restarted by watchdog")
                            consecutive_failures = 0
                            restart_backoff = 0.0
                        except Exception as start_err:
                            # Bump backoff so we don't hammer a dead network.
                            restart_backoff = min(max(restart_backoff * 2, 60.0), 600.0)
                            logger.error(
                                f"Watchdog restart failed: {start_err} — "
                                f"next probe in {restart_backoff:.0f}s"
                            )

        _asyncio.ensure_future(_polling_watchdog())
        await self._notification_service.run()

    async def stop(self):
        """Stop the Telegram bot."""
        if self._notification_service:
            await self._notification_service.stop()
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                logger.error(f"Bot shutdown error: {e}")
        logger.info("Telegram bot stopped")
