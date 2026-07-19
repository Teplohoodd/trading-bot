"""futbot.orchestrator.main — unified bot entry.

Runs pairs + trend (toggleable via ORCH_ENABLE_*) under a single Telegram
session so /commands work.  Each strategy has its own DB; broker and
notifier are shared.

Run:   python -m futbot.orchestrator.main

Each strategy runs as a supervised asyncio task at its own cadence
(ORCH_*_INTERVAL_SECONDS).  On any task crash, supervisor logs and
restarts after a short backoff.

Standalone runners (pairs.main / trend.main) still work for testing —
just don't run them at the same time as the orchestrator (Telegram
poll conflict + broker quota).
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.broker import BrokerClient  # noqa: E402

from futbot.orchestrator.config import OrchSettings  # noqa: E402
from futbot.orchestrator.pairs_bot import PairsBot  # noqa: E402
from futbot.orchestrator.trend_bot import TrendBot  # noqa: E402
from futbot.carry.bot import CarryBot  # noqa: E402
from futbot.orchestrator import commands as cmd_mod  # noqa: E402
from futbot.telegram_notifier import (  # noqa: E402
    TelegramNotifier,
    Msg,
    MsgType,
)
from futbot.telegram_commands import TelegramCommandServer  # noqa: E402


def setup_logging(settings: OrchSettings):
    settings.ORCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Windows console defaults to cp1251 → ₽/─ throw UnicodeEncodeError.
    # Reconfigure stdout to utf-8 so log messages render cleanly.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.ORCH_LOG_PATH, encoding="utf-8"),
        ],
    )
    for n in ("httpx", "telegram", "t_tech", "grpc"):
        logging.getLogger(n).setLevel(logging.WARNING)


async def _supervise_runner(name: str, runner, interval_seconds: int, shutdown: asyncio.Event):
    """Run runner.tick() every `interval_seconds`.  Restart on errors
    with brief backoff so one crash doesn't kill the orchestrator."""
    log = logging.getLogger(f"orchestrator.sup.{name}")
    backoff = 5.0
    while not shutdown.is_set():
        try:
            await runner.tick()
            backoff = 5.0  # reset on successful tick
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception(
                f"runner '{name}' tick crashed: {type(e).__name__}: {e} — "
                f"backing off {backoff:.0f}s"
            )
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=backoff)
                return  # shutdown signalled during backoff
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, 300.0)
            continue

        # Normal sleep until next tick
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=interval_seconds)
            return
        except asyncio.TimeoutError:
            pass


async def _supervise_monitor(runners, interval_seconds: int, shutdown: asyncio.Event):
    """Run runner.monitor() for every runner that has one, every
    `interval_seconds` (fast position-safety loop, between signal ticks)."""
    log = logging.getLogger("orchestrator.sup.monitor")
    monitorable = [(n, r) for n, r, _ in runners if hasattr(r, "monitor")]
    if not monitorable:
        return
    while not shutdown.is_set():
        try:
            await asyncio.wait_for(shutdown.wait(), timeout=interval_seconds)
            return
        except asyncio.TimeoutError:
            pass
        for name, runner in monitorable:
            try:
                await runner.monitor()
            except Exception as e:
                log.exception(f"monitor '{name}' crashed: {type(e).__name__}: {e}")


def _acquire_singleton_lock() -> "object":
    """Prevent a SECOND orchestrator from trading the same account.

    On 2026-06-15 two orchestrators ran at once (a stale 06-12 instance + a
    restart): they raced on the shared DBs and broker, opening/managing
    positions independently (TATN/IVAT chaos, lost stops).  This lock makes a
    second start fail loudly instead of silently double-trading.

    Uses an OS-level exclusive file lock (released automatically if the process
    dies), not just a PID file — robust to crashes.
    """
    import tempfile

    lock_path = Path(tempfile.gettempdir()) / "futbot_orchestrator.lock"
    if sys.platform == "win32":
        import msvcrt

        f = open(lock_path, "w")
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit(
                f"FATAL: another orchestrator is already running "
                f"(lock held: {lock_path}). Refusing to double-trade. "
                f"Kill the other instance first."
            )
    else:
        import fcntl

        f = open(lock_path, "w")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"FATAL: another orchestrator already running ({lock_path}).")
    f.write(str(__import__("os").getpid()))
    f.flush()
    return f  # keep the handle alive for the process lifetime


async def main():
    settings = OrchSettings()
    setup_logging(settings)
    logger = logging.getLogger("orchestrator")

    mode_summary = (
        f"pairs={'on' if settings.ORCH_ENABLE_PAIRS else 'OFF'} "
        f"trend={'on' if settings.ORCH_ENABLE_TREND else 'OFF'} "
        f"carry={'on' if settings.ORCH_ENABLE_CARRY else 'OFF'} "
        f"scalp={'on' if settings.ORCH_ENABLE_SCALP else 'OFF'}"
    )
    logger.info(f"orchestrator starting — {mode_summary}")

    # ── Shared broker ────────────────────────────────────────────────
    broker = BrokerClient(
        token=settings.T_INVEST_TOKEN,
        account_id=settings.T_INVEST_ACCOUNT_ID,
        app_name="futbot-orch",
    )
    await broker.connect()

    # ── Shared outbound notifier ─────────────────────────────────────
    # The [PAPER]/[LIVE] prefix MUST tell the truth.  It was hardcoded paper=
    # True, so REAL trades were labelled [PAPER] — dangerously misleading.
    # Reflect reality: if ANY enabled strategy places real orders → [LIVE].
    from futbot.trend.config import TrendSettings
    from futbot.carry.config import CarrySettings
    from futbot.pairs.config import PairsSettings
    from futbot.breakdown.config import BreakdownSettings

    any_live = (
        (settings.ORCH_ENABLE_TREND and not TrendSettings().TREND_PAPER_MODE)
        or (settings.ORCH_ENABLE_CARRY and not CarrySettings().CARRY_PAPER_MODE)
        or (settings.ORCH_ENABLE_PAIRS and not PairsSettings().PAIRS_PAPER_MODE)
        or (
            getattr(settings, "ORCH_ENABLE_BREAKDOWN", False)
            and not BreakdownSettings().BD_PAPER_MODE
        )
    )
    notifier = TelegramNotifier(
        token=settings.TELEGRAM_BOT_TOKEN,
        chat_id=settings.TELEGRAM_CHAT_ID,
        paper=not any_live,  # [LIVE] if any strategy trades real money
    )
    await notifier.start()
    logger.info(
        f"Notifier mode label: {'LIVE' if any_live else 'PAPER'} " f"(real orders: {any_live})"
    )

    # ── Initialise each enabled strategy ─────────────────────────────
    pairs_bot = None
    trend_bot = None
    carry_bot = None
    runners: list[tuple[str, object, int]] = []

    if settings.ORCH_ENABLE_PAIRS:
        pairs_bot = PairsBot()
        if await pairs_bot.setup(broker, notifier):
            runners.append(("pairs", pairs_bot, settings.ORCH_PAIRS_INTERVAL_SECONDS))

    if settings.ORCH_ENABLE_TREND:
        trend_bot = TrendBot()
        if await trend_bot.setup(broker, notifier):
            runners.append(("trend", trend_bot, settings.ORCH_TREND_INTERVAL_SECONDS))

    if settings.ORCH_ENABLE_CARRY:
        carry_bot = CarryBot()
        if await carry_bot.setup(broker, notifier):
            runners.append(("carry", carry_bot, settings.ORCH_CARRY_INTERVAL_SECONDS))

    breakdown_bot = None
    if getattr(settings, "ORCH_ENABLE_BREAKDOWN", False):
        from futbot.breakdown.bot import BreakdownBot

        breakdown_bot = BreakdownBot()
        if await breakdown_bot.setup(broker, notifier):
            # tick every signal bar (2h); monitor loop covers stops in between
            runners.append(("breakdown", breakdown_bot, breakdown_bot.settings.BD_BAR_HOURS * 3600))

    if settings.ORCH_ENABLE_SCALP:
        logger.warning(
            "Scalp enabled — but scalp orchestrator integration is not "
            "implemented yet; run `python -m futbot.scalp.main` separately"
        )

    if not runners:
        logger.error("No strategies initialised — exiting")
        await notifier.stop()
        await broker.disconnect()
        return

    # ── Single Telegram command server (replaces per-bot servers) ────
    handlers = cmd_mod.build_handlers(pairs_bot=pairs_bot, trend_bot=trend_bot, carry_bot=carry_bot)
    cmd_server = TelegramCommandServer(
        token=settings.TELEGRAM_BOT_TOKEN,
        chat_id=settings.TELEGRAM_CHAT_ID,
        handlers=handlers,
    )
    await cmd_server.start()

    # ── Boot Telegram ────────────────────────────────────────────────
    notifier.push(
        Msg(
            MsgType.BOOT,
            "Orchestrator online",
            f"Strategies: {', '.join(name for name, _, _ in runners)}\n"
            f"Commands: /status /pairs /trend /open /pnl /help",
        )
    )

    # ── Shutdown plumbing ────────────────────────────────────────────
    shutdown = asyncio.Event()

    def _sig(*_):
        logger.info("Shutdown received")
        shutdown.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(s, _sig)
        except NotImplementedError:
            pass  # Windows

    # ── Spawn supervised runner tasks ────────────────────────────────
    tasks = [
        asyncio.create_task(
            _supervise_runner(name, runner, interval, shutdown),
            name=f"sup-{name}",
        )
        for name, runner, interval in runners
    ]
    # Fast position-safety monitor (reconcile + stop/target enforcement)
    tasks.append(
        asyncio.create_task(
            _supervise_monitor(runners, settings.ORCH_MONITOR_INTERVAL_SECONDS, shutdown),
            name="sup-monitor",
        )
    )
    logger.info(
        f"Running {len(tasks)} supervised tasks "
        f"(monitor every {settings.ORCH_MONITOR_INTERVAL_SECONDS}s)"
    )

    # ── Wait for shutdown ───────────────────────────────────────────
    try:
        await shutdown.wait()
    except KeyboardInterrupt:
        shutdown.set()

    logger.info("Shutting down…")

    # Cancel supervisors
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Per-runner teardown
    for name, runner, _ in runners:
        try:
            await runner.shutdown()
        except Exception as e:
            logger.warning(f"shutdown {name} error: {e}")

    # Telegram + broker
    try:
        await cmd_server.stop()
    except Exception:
        pass
    try:
        await notifier.stop()
    except Exception:
        pass
    try:
        await broker.disconnect()
    except Exception:
        pass

    logger.info("Orchestrator stopped.")


if __name__ == "__main__":
    _lock = _acquire_singleton_lock()  # dies here if another instance runs
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
