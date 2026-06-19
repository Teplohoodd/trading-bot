# futbot — dedicated futures-only trading agent

Sibling to `trade_claude`. Shares broker, telegram and database
infrastructure via direct imports; ships its own decision pipeline,
ГО-aware sizer, Chandelier trailing stop, ML gate, risk audit, telegram
notifier, backtest harness, and DB schema.

**Status:** all phases shipped. Paper mode default; live requires explicit
flag. Models for `oil`, `gas`, `gold`, `sber` already trained (run
`train_ml` for the rest).

## Running

```bash
# from the repo root (parent of futbot/)
python -m futbot.main                # the bot (paper by default)
python -m futbot.scripts.train_ml    # train all ML-gate models
python -m futbot.scripts.backtest    # historical replay
python -m futbot.scripts.shadow_run  # compare with trade_claude
```

To go live, set in `.env`:

```ini
FUTBOT_PAPER_MODE=false
```

…and restart. Boot log will print `LIVE` plus a warning. **Don't flip this
flag until you've run paper for at least a week** and verified the bot's
behaviour matches your expectations via `data/futbot.db` and Telegram.

## What it does, per tick (default 60s)

1. **Universe refresh** (every 6h): resolve front-month FORTS contracts for
   Si, BR, GZ, SR, LK, MX (tier-1) and NG, GD, RT, EURRUBF (tier-2). Skips
   contracts within 14 days of expiry.
2. **Manage open positions**: refresh the Chandelier trailing stop, check
   if the latest 5m bar's high/low touched the stop, and close if so or
   if the position has been held > 24h. Telegram alert on every close.
3. **Kill-switch check**: if today's realised P&L < −1.5 % of portfolio,
   pause new entries until the next UTC midnight. Telegram alert when
   tripped.
4. **Evaluate every contract** through the 4-layer + ML pipeline:
   - **Trend (1h)**: EMA20 vs EMA50 + ADX ≥ 18 → direction + strength
   - **Regime (1h)**: rolling-50 linreg R² + ATR/median → trending /
     choppy / vol-spike. Vol-spike VETOES; choppy FLIPS the direction
     (mean-reversion mode).
   - **Setup (15m)**: KDJ %K + Bollinger %B in oversold/overbought zone
   - **Trigger (5m)**: bar with > 1.5× median volume, > 1.2× median
     range, closing in the top/bottom 25 % of its range, direction-aligned
   - **ML gate**: per-concept LightGBM (oil/gas/gold/sber/lkoh/moex) trained
     on Kaggle daily data; vetoes only when STRONGLY (p ≥ 0.55) opposite
     the chain. Concepts with no trained model pass through.
5. **Risk audit**: hour-of-day blacklist, days-to-expiry, total ГО cap
   (40 % of portfolio), per-contract daily loss cap, total daily kill,
   spread guard, max open contracts.
6. **Sizer**: min(ГО-cap, vol-target-cap) lots.
7. **Place order**: paper or live. Telegram alert on every entry.

## Layout

```
futbot/
├── main.py               # entry, loop
├── config.py             # FutSettings (FUTBOT_* env keys)
├── universe.py           # front-month FORTS resolver
├── db.py                 # SQLite: trades, decisions, positions_state
├── telegram_notifier.py  # non-blocking alert queue
├── data/candles.py       # multi-TF fetch with in-memory cache
├── pipeline/
│   ├── base.py           # LayerResult
│   ├── trend.py          # Layer 1 — 1h EMA + ADX
│   ├── regime.py         # Layer 2 — trending / choppy / vol-spike
│   ├── setup.py          # Layer 3 — 15m KDJ + Bollinger %B
│   ├── trigger.py        # Layer 4 — 5m breakout bar
│   ├── ml_gate.py        # Layer 5 — per-concept LGBM veto
│   └── decision.py       # orchestrator
├── execution/
│   ├── sizer.py          # ГО + vol-target
│   ├── stops.py          # Chandelier
│   └── orders.py         # paper / live wrapper
├── risk/
│   ├── audit.py          # final veto
│   └── circuit.py        # daily kill-switch
├── ml/
│   ├── datasets.py       # Kaggle download + parsing
│   ├── features.py       # 14 daily features (returns, ATR%, RSI, BB%B,
│   │                     #   EMA-spread, ADX, vol-z, R²-50, dow, month)
│   ├── labels.py         # triple-barrier (LdP ch.3)
│   ├── trainer.py        # walk-forward LGBM train (3-class)
│   └── model.py          # inference wrapper + veto policy
├── backtest/
│   └── replay.py         # replay pipeline on historical Tinkoff bars
└── scripts/
    ├── train_ml.py       # train one or all ML concepts
    ├── backtest.py       # backtest tier-1 contracts
    └── shadow_run.py     # compare futbot vs trade_claude
```

## ML gate

Six concepts mapped to FORTS contract bases:

| Concept | Source dataset                          | FORTS contract |
|---------|-----------------------------------------|----------------|
| `oil`   | guillemservera/fuels-futures-data (CL=F) | BR             |
| `gas`   | guillemservera/fuels-futures-data (NG=F) | NG             |
| `gold`  | guillemservera/precious-metals-data (GC=F) | GD           |
| `sber`  | alexanderkobzar/moex-shares (SBER)       | SR             |
| `lkoh`  | alexanderkobzar/moex-shares (LKOH)       | LK             |
| `moex`  | alexanderkobzar/moex-shares (VWAP all RU)| MX             |

Si, GZ, RT and EURRUBF have no public daily proxy → ML gate passes through
for those (chain decisions only). When you have more data sources, just
extend `CONTRACT_TO_CONCEPT` in `ml/datasets.py`.

The models are 3-class LightGBM with triple-barrier labels (horizon=5 days,
±1.5 ATR barriers). Test-set accuracy on the trained concepts is ~35-40 %
on a 3-class problem (random baseline 33.3 %). **This is by design** —
the gate's role is to VETO setups when its predicted class is the opposite
of the chain with high confidence (default p ≥ 0.55). It does not initiate
trades. Tighten the veto threshold in `ml/model.py` (`CONFIDENCE_VETO`)
to make it more or less opinionated.

To retrain:
```bash
python -m futbot.scripts.train_ml             # all concepts
python -m futbot.scripts.train_ml oil gold    # subset
```

## Backtesting

```bash
python -m futbot.scripts.backtest 90 6        # last 90 days, eval every hour
python -m futbot.scripts.backtest 180 1       # 180 days, every 5m bar (slow but accurate)
```

Outputs a per-ticker summary and saves trade-level detail to
`data/futbot_backtest_trades.csv`. The backtest uses the **same** pipeline
code as live trading — no separate backtest engine that could drift.

## Shadow run (vs trade_claude)

After running paper mode for a few days alongside `trade_claude`:

```bash
python -m futbot.scripts.shadow_run 14    # last 14 days
```

Compares pipeline evaluations, approved trades, opened/closed counts,
realised P&L, unique tickers traded, and figi overlap. Useful for the
"is futbot ready to take over" decision.

## What's intentionally NOT here

| Feature | Why deferred |
|---|---|
| Roll automation | Manual for first month — bot just stops trading contracts ≤ 14d to expiry. |
| Limit-aggressive orders | Live default is `post_market_order` for predictability. Switch to `post_limit_with_fallback` later if slippage shows up in postmortem. |
| News blackouts (CBR rate days) | Easy to add to `risk/audit.py`; deferred until we see evidence it matters. |
| Telegram command handlers | Notifier is one-way. If you want `/futbot status` from Telegram, the existing `trade_claude` TelegramBot can be extended (or vice versa — wire both bots into one app). |

## Killing it

Ctrl-C. Outstanding positions stay open (no auto-flatten on shutdown —
by design; restart picks them up via `db.open_trades()` and
`positions_state`).

## Telemetry

Every pipeline run writes a row to `decisions` whether approved or not,
with each layer's full output as JSON. To see why a contract didn't trade:

```sql
SELECT ts, ticker, proposed_direction, rejected_at_layer, rejection_reason
FROM decisions
WHERE ticker = 'SRM6'
ORDER BY ts DESC LIMIT 20;
```

To see ML-gate output specifically (e.g. when a setup passed but ML vetoed):

```sql
SELECT ts, ticker, rejected_at_layer, layer_ml
FROM decisions
WHERE rejected_at_layer = 'ml_gate'
ORDER BY ts DESC LIMIT 20;
```

## Safety guarantees

- **Paper mode by default.** Live requires explicit env flag.
- **Max 2 open contracts** during phase-3 rollout (`FUTBOT_MAX_OPEN_CONTRACTS`).
- **Daily kill-switch** at 1.5 % drawdown — auto-resets at UTC midnight.
- **Per-contract daily loss cap** at 1 % — blocks doubling-down.
- **ГО cap at 40 %** — keeps free margin for margin-call buffer.
- **Stops use intra-bar extremes** (`bar.low` for longs) — flash-stops on wicks count.
- **Telegram alerts** on every open/close/circuit-trip event. Drops on
  network failure rather than blocking the trading loop.
