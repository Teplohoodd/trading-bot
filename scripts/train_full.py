"""End-to-end training: primary LightGBM + meta-labelling secondary classifier.

This is a one-shot retrain script that:
  1. Connects to T-API.
  2. Picks the top-N MOEX shares + futures from the screener.
  3. Trains the universal pooled model with the full LdP-style methodology
     (sample-uniqueness weights, frac-diff features, purged CV, meta-labelling).
  4. Saves the model + meta bundle to data/models/universal_<version>.joblib.
  5. Registers in model_registry so the bot will pick it up on next start.

Run:  python -m scripts.train_full [--top-n 15]
"""

import argparse
import asyncio
import logging
from pathlib import Path

from analysis.macro import MacroProvider
from analysis.screener import Screener
from config.settings import Settings
from core.broker import BrokerClient
from database.db import Repository
from ml.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logging.getLogger("t_tech.invest.logging").setLevel(logging.WARNING)
log = logging.getLogger("train_full")


async def amain(args):
    settings = Settings()
    db = Repository(Path("data/trade_bot.db"))
    await db.initialize()

    broker = BrokerClient(settings.T_INVEST_TOKEN, settings.T_INVEST_ACCOUNT_ID)
    await broker.connect()

    try:
        screener = Screener(
            broker,
            top_n=args.top_n,
            include_futures=settings.INCLUDE_FUTURES,
            futures_min_days_to_expiry=settings.FUTURES_MIN_DAYS_TO_EXPIRY,
        )
        macro = MacroProvider(broker)
        trainer = ModelTrainer(broker, db, settings, macro_provider=macro)

        log.info(f"Building watchlist (top {args.top_n})...")
        watchlist = await screener.scan_universe()
        if not watchlist:
            log.error("Empty watchlist — abort")
            return 1
        tickers_figis = [
            (c["ticker"], c["figi"], c.get("kind", "share")) for c in watchlist[: args.top_n]
        ]
        log.info(f"Training on: {[t[0] for t in tickers_figis]}")

        model = await trainer.train_universal_model(tickers_figis, force_overwrite=args.force)
        if model is None:
            log.error("train_universal_model returned None — see logs above")
            return 1

        log.info("=== Training complete ===")
        log.info(f"Primary metadata: {model.metadata}")
        if model.meta_model is not None and model.meta_model.metadata is not None:
            mm = model.meta_model.metadata
            log.info(
                f"Meta metadata: acc={mm.accuracy:.3f}, f1={mm.f1:.3f}, "
                f"prec={mm.precision:.3f}, rec={mm.recall:.3f}, "
                f"n={mm.n_train}, threshold={mm.threshold}"
            )
        else:
            log.warning("No meta model attached")
        return 0
    finally:
        await broker.disconnect()
        await db.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--top-n", type=int, default=15, help="Top N tickers from screener (default: 15)"
    )
    ap.add_argument(
        "--force", action="store_true", help="Bypass rollback gate and overwrite the current model"
    )
    args = ap.parse_args()
    raise SystemExit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
