"""Probe Tinkoff API limits for hourly candles.

Tries progressively larger windows and reports which ones succeed/fail.
Run:  python -m scripts.test_candle_limits
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from config.settings import Settings
from core.broker import BrokerClient
from tinkoff.invest import CandleInterval

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("probe")
# Keep Tinkoff's INFO-level request spam out of the way
logging.getLogger("tinkoff.invest.logging").setLevel(logging.WARNING)


async def main():
    settings = Settings()
    broker = BrokerClient(settings.T_INVEST_TOKEN, settings.T_INVEST_ACCOUNT_ID)
    await broker.connect()

    # Find a reliable liquid ticker
    instruments = await broker.find_instrument("SBER", kind="INSTRUMENT_TYPE_SHARE")
    for inst in instruments[:5]:
        logger.info(f"  candidate: ticker={inst.ticker} figi={inst.figi} name={inst.name}")
    sber = next(
        (i for i in instruments if i.ticker == "SBER" and i.figi.startswith("BBG")),
        next((i for i in instruments if i.ticker == "SBER"), instruments[0]),
    )
    figi = sber.figi
    logger.info(f"Probing candles for SBER ({figi})")

    now = datetime.now(timezone.utc)

    test_windows_days = [90, 100, 110]

    # Also: test older window [200d ago ... 100d ago] to check data exists
    logger.info("--- older slice test ---")
    older_from = now - timedelta(days=200)
    older_to = now - timedelta(days=100)
    try:
        resp = await broker._services.market_data.get_candles(
            figi=figi,
            from_=older_from,
            to=older_to,
            interval=CandleInterval.CANDLE_INTERVAL_HOUR,
        )
        logger.info(f"HOUR older slice (200d..100d) -> {len(resp.candles)} candles")
    except Exception as e:
        logger.error(f"HOUR older slice -> {e!r}")

    # Temporarily override broker's internal chunking by calling raw API directly
    for days in test_windows_days:
        from_dt = now - timedelta(days=days)
        try:
            resp = await broker._services.market_data.get_candles(
                figi=figi,
                from_=from_dt,
                to=now,
                interval=CandleInterval.CANDLE_INTERVAL_HOUR,
            )
            n = len(resp.candles)
            logger.info(f"HOUR  window={days:>3}d  -> OK   ({n:>5} candles)")
        except Exception as e:
            err = str(e)
            if "30014" in err:
                logger.warning(f"HOUR  window={days:>3}d  -> 30014 (window too large)")
            else:
                logger.error(f"HOUR  window={days:>3}d  -> ERROR: {e!r}")

    # Now test full 180 days via broker.get_candles (which chunks internally)
    logger.info("--- broker.get_candles (with internal chunking) ---")
    candles = await broker.get_candles(
        figi,
        now - timedelta(days=180),
        now,
        interval=CandleInterval.CANDLE_INTERVAL_HOUR,
    )
    logger.info(f"broker.get_candles(HOUR, 180d) -> {len(candles)} candles")

    await broker.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
