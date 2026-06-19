"""Replay last week's trades through the new model + meta-labelling pipeline.

For every closed trade in the analysis window, fetch hourly candles up to
entry_time, build features as the live engine would, run primary + meta,
and report:
  * What the new pipeline would have decided (buy/sell/hold).
  * Whether it agrees with the historical entry direction.
  * Whether the meta would have vetoed the trade.

This is the closest we get to "would the new model have made better
decisions last week?" without a full discrete-event backtest.

Run:  python -m scripts.replay_with_model [--days 7]
"""

import argparse
import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from analysis.features import build_features, FEATURE_NAMES
from analysis.macro import MacroProvider
from analysis.screener import _candles_to_df
from config.settings import Settings
from core.broker import BrokerClient
from database.db import Repository
from ml.model import LGBMModel
from tinkoff.invest import CandleInterval

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("tinkoff.invest.logging").setLevel(logging.WARNING)
log = logging.getLogger("replay_model")


def latest_universal_model(models_dir: Path) -> Path | None:
    candidates = sorted(
        models_dir.glob("universal_*.joblib"),
        key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


async def amain(args):
    settings = Settings()
    db = Repository(Path("data/trade_bot.db"))
    await db.initialize()

    model_path = latest_universal_model(settings.MODELS_DIR)
    if model_path is None:
        log.error("No universal_*.joblib found — train the model first")
        return 1
    log.info(f"Loading model: {model_path}")
    model = LGBMModel(model_path)
    if not model.is_trained:
        log.error("Model failed to load")
        return 1
    has_meta = model.meta_model is not None and getattr(model.meta_model, "is_trained", False)
    log.info(f"Meta-labelling model present: {has_meta}")

    broker = BrokerClient(settings.T_INVEST_TOKEN, settings.T_INVEST_ACCOUNT_ID)
    await broker.connect()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
    conn = sqlite3.connect(settings.DB_PATH)
    trades = pd.read_sql_query(
        "SELECT id, figi, ticker, direction, signal_confidence, entry_time, "
        "entry_price, exit_price, pnl, pnl_pct, exit_reason "
        "FROM trades WHERE status='closed' AND entry_time >= ? "
        "ORDER BY entry_time",
        conn,
        params=(cutoff,),
    )
    conn.close()
    log.info(f"Loaded {len(trades)} closed trades")

    macro = MacroProvider(broker)

    rows = []
    try:
        for _, t in trades.iterrows():
            try:
                figi = t["figi"]
                ticker = t["ticker"]
                entry = pd.to_datetime(t["entry_time"], utc=True)
                from_dt = (entry - pd.Timedelta(days=180)).to_pydatetime()
                to_dt = entry.to_pydatetime()
                candles = await broker.get_candles(
                    figi, from_dt, to_dt, CandleInterval.CANDLE_INTERVAL_HOUR
                )
                if len(candles) < 60:
                    rows.append({"id": t["id"], "ticker": ticker, "skip": "too_few_candles"})
                    continue
                df = _candles_to_df(candles)
                # Macro fetch matched to candle interval
                try:
                    macro_df = await macro.get_macro_df(
                        from_dt=df["time"].iloc[0],
                        to_dt=df["time"].iloc[-1],
                        interval=CandleInterval.CANDLE_INTERVAL_HOUR,
                    )
                except Exception:
                    macro_df = None

                inst, kind = await broker.get_instrument_info(figi)
                features_df = build_features(
                    df,
                    macro_df=macro_df,
                    instrument_kind=kind,
                    ticker=ticker,
                )

                # Primary
                primary_dir, primary_conf = model.predict(features_df[FEATURE_NAMES])

                # Meta gate
                meta_conf = None
                meta_gate = "pass"
                if has_meta and primary_dir != "hold":
                    proba = model.predict_proba(features_df[FEATURE_NAMES].iloc[[-1]])[0]
                    aug = features_df[FEATURE_NAMES].iloc[[-1]].copy()
                    aug["primary_pred_sell"] = float(proba[0])
                    aug["primary_pred_hold"] = float(proba[1])
                    aug["primary_pred_buy"] = float(proba[2])
                    aug["primary_direction"] = 1 if primary_dir == "buy" else -1
                    aug["primary_conf"] = float(primary_conf)
                    gate, mc = model.meta_model.predict(aug)
                    meta_conf = mc
                    meta_gate = "pass" if gate else "VETO"

                final_dir = primary_dir
                if meta_gate == "VETO":
                    final_dir = "hold"

                rows.append(
                    {
                        "id": int(t["id"]),
                        "ticker": ticker,
                        "hist_dir": t["direction"],
                        "hist_conf": float(t["signal_confidence"]),
                        "hist_pnl_pct": float(t["pnl_pct"]),
                        "new_primary_dir": primary_dir,
                        "new_primary_conf": round(float(primary_conf), 3),
                        "meta_conf": round(float(meta_conf), 3) if meta_conf is not None else None,
                        "meta_gate": meta_gate,
                        "new_final_dir": final_dir,
                        "agree_with_hist": final_dir == t["direction"],
                    }
                )
            except Exception as e:
                rows.append({"id": int(t["id"]), "ticker": t["ticker"], "skip": str(e)})
    finally:
        await broker.disconnect()
        await db.close()

    out = pd.DataFrame(rows)
    out_path = (
        Path("analysis/reports") / f"replay_with_model_{datetime.now().strftime('%Y%m%d')}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    log.info(f"Wrote {out_path}")

    # Summary
    if "skip" in out.columns:
        valid = out[out["skip"].isna()]
    else:
        valid = out
    if len(valid):
        print()
        print(f"Total: {len(out)} trades, valid: {len(valid)}")
        if "new_final_dir" in valid.columns:
            agree_mask = valid["agree_with_hist"].astype(bool)
            agree = valid[agree_mask]
            disagree = valid[~agree_mask]
            vetoed = valid[valid["meta_gate"] == "VETO"]
            held = valid[valid["new_final_dir"] == "hold"]
            print(
                f"  agree with historical:    {len(agree):3d} | "
                f"sum_pnl={agree['hist_pnl_pct'].sum():+.2f}% mean={agree['hist_pnl_pct'].mean():+.3f}%"
            )
            print(
                f"  disagree:                 {len(disagree):3d} | "
                f"sum_pnl={disagree['hist_pnl_pct'].sum():+.2f}% mean={disagree['hist_pnl_pct'].mean():+.3f}%"
            )
            print(
                f"  meta vetoed:              {len(vetoed):3d} | "
                f"sum_pnl_avoided={vetoed['hist_pnl_pct'].sum():+.2f}%"
            )
            print(
                f"  new model says HOLD:      {len(held):3d} | "
                f"sum_pnl_avoided={held['hist_pnl_pct'].sum():+.2f}%"
            )

    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    raise SystemExit(asyncio.run(amain(args)))


if __name__ == "__main__":
    main()
