"""Entry point: wire all components and run the trading bot."""

import asyncio
import logging
import signal
import sys
from pathlib import Path

from config.settings import Settings
from core.broker import BrokerClient
from core.engine import TradingEngine
from database.db import Repository
from risk.manager import RiskManager
from risk.spread_monitor import SpreadMonitor
from risk.position_sizer import PositionSizer
from risk.execution import ExecutionScheduler
from analysis.screener import Screener
from analysis.macro import MacroProvider
from strategy.ml_strategy import MLStrategy
from strategy.technical_strategy import TechnicalStrategy
from strategy.regime import RegimeDetector
from ml.model import LGBMModel
from ml.trainer import ModelTrainer
from telegram_bot.bot import TelegramBot


def setup_logging(settings: Settings):
    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOGS_DIR / "bot.log", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("grpc").setLevel(logging.WARNING)


async def main():
    # ── 1. Config ──────────────────────────────────────────────────────────────
    settings = Settings()
    setup_logging(settings)
    logger = logging.getLogger("main")
    logger.info("Starting trading bot...")

    # ── 2. Database ────────────────────────────────────────────────────────────
    db = Repository(settings.DB_PATH)
    await db.initialize()
    logger.info("Database initialized")

    # ── 3. Notification queue ──────────────────────────────────────────────────
    notification_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

    # ── 4. Broker ──────────────────────────────────────────────────────────────
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
    )
    await broker.connect()
    logger.info("Broker connected")

    # ── 5. ML model ────────────────────────────────────────────────────────────
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    existing_model_record = await db.get_active_model(None)
    model = None
    if existing_model_record and Path(existing_model_record["model_path"]).exists():
        try:
            model = LGBMModel(Path(existing_model_record["model_path"]))
            logger.info(f"Loaded model v{existing_model_record['version']}")
        except Exception as e:
            logger.warning(f"Could not load saved model: {e}")

    # ── 6. Strategies ──────────────────────────────────────────────────────────
    # MacroProvider is shared across strategy and trainer so they hit the
    # macro cache in lockstep (one API call per macro per 15-min window).
    macro_provider = MacroProvider(broker=broker)
    ml_strategy = MLStrategy(model=model, settings=settings, macro_provider=macro_provider)
    tech_strategy = TechnicalStrategy(settings=settings)
    regime_detector = RegimeDetector()
    strategies = [ml_strategy, tech_strategy]

    # ── 7. Risk management ─────────────────────────────────────────────────────
    spread_monitor = SpreadMonitor(
        threshold_multiplier=settings.SPREAD_THRESHOLD,
        window_size=100,
    )
    position_sizer = PositionSizer(settings=settings)
    execution_scheduler = ExecutionScheduler(broker=broker)
    risk_manager = RiskManager(
        broker=broker,
        spread_monitor=spread_monitor,
        position_sizer=position_sizer,
        execution_scheduler=execution_scheduler,
        db=db,
        settings=settings,
    )

    # ── 8. Screener ────────────────────────────────────────────────────────────
    screener = Screener(
        broker=broker,
        top_n=30,
        include_futures=settings.INCLUDE_FUTURES,
        futures_min_days_to_expiry=settings.FUTURES_MIN_DAYS_TO_EXPIRY,
    )

    # ── 9. Trading engine ──────────────────────────────────────────────────────
    engine = TradingEngine(
        broker=broker,
        risk_manager=risk_manager,
        strategies=strategies,
        screener=screener,
        db=db,
        notification_queue=notification_queue,
        settings=settings,
    )
    engine.set_regime_detector(regime_detector)

    # ── 10. ML trainer ─────────────────────────────────────────────────────────
    trainer = ModelTrainer(broker=broker, db=db, settings=settings, macro_provider=macro_provider)
    engine.set_trainer(trainer)

    # ── 11. Telegram bot ───────────────────────────────────────────────────────
    telegram_bot = TelegramBot(
        token=settings.TELEGRAM_BOT_TOKEN,
        engine=engine,
        db=db,
        notification_queue=notification_queue,
        settings=settings,
    )

    # ── 12. Graceful shutdown ──────────────────────────────────────────────────
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for all signals
            pass

    # ── 13. Run ────────────────────────────────────────────────────────────────
    logger.info("All components ready. Starting event loop...")

    async def run_with_shutdown():
        # Engine is critical — if it dies the bot can't trade and we shut down.
        # Telegram is auxiliary — its death must NOT take the engine with it.
        # Telegram is wrapped in a supervisor that restarts it with exponential
        # backoff so a network blip doesn't end live trading.
        engine_task = asyncio.create_task(engine.run(), name="engine")

        async def telegram_supervisor():
            backoff = 5
            while True:
                try:
                    await telegram_bot.start()
                    logger.warning(
                        "Telegram task ended without exception — "
                        "restarting in 30s (engine continues)"
                    )
                    await asyncio.sleep(30)
                    backoff = 5
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(
                        f"Telegram task crashed: {type(exc).__name__}: {exc} — "
                        f"restarting in {backoff}s "
                        f"(trading engine continues unaffected)",
                    )
                    # Best-effort cleanup of half-initialised PTB state before retry
                    try:
                        await telegram_bot.stop()
                    except Exception:
                        pass
                    try:
                        await asyncio.sleep(backoff)
                    except asyncio.CancelledError:
                        raise
                    backoff = min(backoff * 2, 300)

        telegram_task = asyncio.create_task(telegram_supervisor(), name="telegram_supervisor")

        # Only ENGINE death triggers shutdown.  Telegram failures are logged
        # by the supervisor and recovered in-place.
        def _engine_done(task: asyncio.Task):
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.error(
                    f"Engine task died with exception: {exc!r}",
                    exc_info=exc,
                )
            else:
                logger.warning(
                    f"Engine task finished unexpectedly "
                    f"(returned {task.result()!r}) — shutting down"
                )
            shutdown_event.set()

        engine_task.add_done_callback(_engine_done)

        # Wait for shutdown signal (SIGINT or engine death)
        await shutdown_event.wait()
        logger.info("Shutting down...")

        # Cancel both tasks
        for task in (engine_task, telegram_task):
            task.cancel()
        await asyncio.gather(engine_task, telegram_task, return_exceptions=True)

    try:
        await run_with_shutdown()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        logger.info("Cleaning up...")
        try:
            await engine.emergency_stop()
        except Exception:
            pass
        try:
            await telegram_bot.stop()
        except Exception:
            pass
        try:
            await broker.disconnect()
        except Exception:
            pass
        try:
            await db.close()
        except Exception:
            pass
        logger.info("Bot stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
