# Plan: dedicated futures-only trading agent (`futbot`)

Author: 2026-05-16. Living document — update as the design evolves.

---

## 1. Why split out a futures bot?

The current `trade_claude` bot is a shares-first hybrid that bolts futures onto
the same pipeline. After 5+ weeks live, the structural problems show up:

- **Whipsaw exits dominate losses.** Hold-time analysis (86 `signal_reversal`
  exits): the 5–30 min bucket lost −300 ₽ (n=14); the 2–4 h bucket made +87 ₽
  (n=11, 91 % win). TTM6 on 2026-05-14 was a 5-minute long that lost 318 ₽.
- **Single-timeframe scoring.** The ML model uses hourly bars; the
  `should_exit` loop re-scores the SAME hourly bar every 5 minutes, so the
  model flips on tiny perturbations.
- **Shares features dominate the training set** (ML futures accuracy is unknown
  — no asset-class-stratified validation; the model has no FORTS-specific
  microstructure). Confidence is uncalibrated for futures.
- **Sizing is share-shaped.** `PositionSizer` had to be patched with
  `instrument_kind="future"` branches; ГО / step_value plumbing is fragile.

Futures (FORTS) have meaningfully different mechanics than shares:

| Property | Shares (MOEX TQBR) | Futures (FORTS) |
|---|---|---|
| Capital block | Full notional (price × lot) | Margin only (ГО, ~10-15 % of notional) |
| Short carry | Overnight borrow fee | None (built-in leverage both sides) |
| Lifecycle | Perpetual | Quarterly expiry → roll |
| Best edges | Mean-reversion, factor | Trend, vol-of-vol, calendar spread |
| Spread | Tight on TQBR | Tight on Si/Br/GZ, wide elsewhere |
| Liquidity tail | Long (300+ names) | Short (top ~10 are 95 % of volume) |

A clean rebuild lets us drop the share-side compromises and design from the
start for the FORTS quirks.

## 2. Research take-aways

Reviewed top trading-agent repos in May 2026:

- **[TradingAgents](https://github.com/tauricresearch/tradingagents)** (HKUDS,
  ~30k stars at v0.2.5): hierarchical multi-agent — Analysts (Fundamental,
  Sentiment, News, Technical) → Researchers (Bullish/Bearish debate) → Trader
  → Risk Audit → Portfolio Manager. Pattern: every decision flows through a
  veto layer.
- **[LLM-TradeBot](https://github.com/EthanAlgoX/LLM-TradeBot)** (Binance
  Futures focused): **Four-Layer Filter** — 1h Trend & Fuel → AI Filter
  (LightGBM, retrains every 2h) → 15m Setup (KDJ + BBands) → 5m Trigger
  (pattern + relative volume). Each layer can VETO; nothing trades unless all
  four agree. Conservative-bias dashboard notes the bot frequently outputs
  "WAIT".
- **[Lumibot](https://github.com/Lumiwealth/lumibot)**: same-code
  backtest/live parity by swapping a broker class. Useful pattern even though
  the framework itself is shares-first.
- **[FinMem](https://github.com/pipiku915/FinMem-LLM-StockTrading)**:
  layered memory (short / mid / long term) with character-prompted LLM agent
  — interesting but probably overkill for a quant bot.

The two convergent ideas worth stealing:

1. **Multi-timeframe gating** (LLM-TradeBot): big timeframe says trend, small
   timeframe says trigger. Eliminates whipsaws by construction — a 5-min
   trigger only fires if the 1h trend agrees.
2. **Explicit veto layers** (TradingAgents): make risk a separate
   responsibility that can reject any otherwise-approved trade. Currently
   `risk/manager.py` does this implicitly; making it agentic surfaces
   *why* a trade was killed.

Confirmed by 2026 retail/quant practice
([QuantifiedStrategies](https://www.quantifiedstrategies.com/algorithmic-trading-strategies/),
[Bookmap](https://bookmap.com/blog/key-algorithmic-trading-strategies-from-trend-following-to-mean-reversion-and-beyond)):
~65-70 % of sessions are choppy → mean-reversion dominates; trending sessions
favour breakout. A single strategy cannot win in both regimes; you need a
regime classifier ahead of the entry rule.

## 3. Target universe

**Tier 1 (always-traded):** Si (USD/RUB), Br (Brent), GZ (Gazprom),
SBRF (Sberbank), GAZR, LKOH. 6 names, all with > 1 bln ₽ ADV and
< 0.1 % spread in active hours.

**Tier 2 (regime-permitting):** NG (Natural gas), GD (Gold), MIX (MOEX
index), RTS, USDRUBF, EURRUBF. Trade only when ATR > 0.5 × 30-day median.

**Excluded:** thinly-traded single-stock futures (TTM6, MGM6, etc.). Live
data shows ALM6/TTM6 trades hit slippage and whipsaws on the same bar — not
worth the operational headache.

## 4. Architecture

Reuses the existing **broker / db / telegram / notification** infrastructure
from `trade_claude` — only the **strategy + sizing + decision loop** are
replaced. Sits alongside the existing bot in the same repo, separate process,
separate DB schema (`data/futbot.db`).

```
                     futbot/
                     ├── main.py             # entry: own asyncio loop, own logger
                     ├── config.py           # FutSettings (futures-specific)
                     ├── universe.py         # tier-1 / tier-2 contract resolver + roll
                     ├── pipeline/
                     │   ├── trend_filter.py     # Layer 1: 1h EMA + ADX
                     │   ├── regime_detector.py  # trending / choppy / vol-spike
                     │   ├── ml_gate.py          # Layer 2: futures-only LGBM
                     │   ├── setup.py            # Layer 3: 15m KDJ + BB
                     │   └── trigger.py          # Layer 4: 5m volume + range
                     ├── execution/
                     │   ├── sizer.py        # ГО-aware Kelly, vol-target
                     │   ├── stops.py        # Chandelier trailing stop
                     │   └── orders.py       # T-Invest order placement
                     ├── risk/
                     │   ├── audit.py        # final veto: max-DD, exposure, hour
                     │   └── circuit.py      # kill-switch + per-contract dailyloss
                     ├── data/
                     │   ├── candles.py      # multi-timeframe fetch + cache
                     │   └── features.py     # futures-specific features
                     └── backtest/
                         ├── replay.py
                         └── runner.py
```

### 4.1 Decision pipeline (per bar, per contract)

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐  ┌──────┐
│ 1h trend     │→│ regime       │→│ 15m setup    │→│ 5m trigger   │→│ ML gate│→│ risk │→  ORDER
│ EMA+ADX vote │  │ classifier   │  │ KDJ + BB     │  │ vol + range  │  │ veto   │  │ audit│
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  └────────┘  └──────┘
   ✓ aligned       trending/chop      oversold/over    breakout bar       prob >       within
                                      bought zone                          0.55         limits
```

Every layer can return ✗ → no trade. The bot defaults to "WAIT". This
addresses the whipsaw problem directly — a 5-min flip cannot trade against
a 1-h trend.

### 4.2 Why this is different from the current bot

| | Current `trade_claude` | New `futbot` |
|---|---|---|
| Timeframes | One (1h) | Three nested (1h / 15m / 5m) |
| Decision | ML score → optional TA confirm | 4-layer veto chain |
| Entry threshold | conf > 0.65 | All 4 layers agree |
| Exit | signal_reversal (re-scores same bar) | Chandelier trailing + 2× ATR target + bar-close reversal |
| Sizing | Kelly on pooled stats | Vol-target on per-contract ATR |
| Re-entry cooldown | 60 min (any direction) | 1 bar of trigger TF on same direction; opposite-direction allowed |
| Universe | 30+ tickers screener | 6 tier-1 contracts (10 max) |

### 4.3 Reuse plan from `trade_claude`

Pull and **import directly**, no copy:

- `core/broker.py` — T-Invest client. Add `get_multi_tf_candles()` helper.
- `analysis/macro.py` — RUONIA / IMOEX context (still useful for regime).
- `telegram_bot/` — already split nicely; futbot mounts its own commands.
- `database/db.py` — extend with futbot-prefixed tables, share connection.
- `risk/spread_monitor.py` — Glosten-Milgrom spread guard.
- Token / chat_id config — same `.env`, new prefix `FUTBOT_*`.

**Rewrite from scratch** (the share-shaped parts):

- `risk/position_sizer.py` — too many share branches; futures needs its own.
- `strategy/ml_strategy.py` — single-timeframe; replace with `pipeline/ml_gate.py`.
- `analysis/features.py` — share-heavy features (book imbalance dominated);
  futures needs OI changes, term-structure carry, futures-vs-spot basis.
- `ml/trainer.py` — re-train on a futures-only dataset with asset-class
  stratification (one universal model per asset class is fine, but training
  must NOT mix shares into the futures fold).

### 4.4 Stops & exits (the TTM6 fix)

The structural cause of TTM6 was: bot re-scored the same hourly bar within 5
minutes, flipped, ate −0.42 % into commission. Three counter-measures in the
new bot:

1. **Decisions only on bar close.** No mid-bar re-scoring. If you entered at
   12:07 on a 1h-trigger flag, the next exit decision is at 13:00, not 12:12.
2. **Chandelier trailing stop**: `stop = highest_high(N) − k × ATR`.
   Activates after the trade is at least 1 R in profit. Replaces ATR-fixed
   stop + the broken signal_reversal exit.
3. **Force-exit at bar close** if the 1h trend layer flips. Cleanly separates
   "tactical wiggle" (ignored) from "trend break" (exit). This is much
   stronger than the current `signal_reversal` which fired on any 0.62-conf
   model flip.

### 4.5 ML role

In the current bot, ML is the **primary** signal (other layers gate it). In
futbot, ML is a **veto** — the chain looks for a clean technical setup
(layers 1-4), and ML can reject it if the model says "this setup historically
fails in this regime". This inverts which side carries the noise:

- **Now:** noisy ML score → traded → filtered by TA → still fires too often
  on weak conviction.
- **Futbot:** clean TA setup → ML can only reject it. Default action is
  "trade the setup unless ML strongly disagrees" (model prob < 0.30 against).

ML model: separate LGBM per regime (trending/choppy/vol-spike), trained on
futures bars only, with triple-barrier labels at 10-bar horizon (per current
TB_MAX_HOLD_FUTURES). Retrain weekly, not daily (smaller dataset →
overfitting risk).

### 4.6 Risk audit (last layer)

A trade approved by layers 1-5 must still pass the risk audit:

- Total ГО used + new ГО ≤ MAX_GO_USE_PCT (default 40 % of account)
- Per-contract loss today ≤ MAX_CONTRACT_DAILY_LOSS_PCT (default 1 %)
- Total daily P&L ≤ MAX_DAILY_LOSS_PCT (kill-switch — same as current)
- Spread ≤ 1.5 × 30-day median (Glosten-Milgrom)
- Hour-of-day NOT in {pre-open 09:55-10:05 MSK, evening session quiet 19:00-19:05}
- Not in roll window (`expiration_date - now < FUTURES_MIN_DAYS_TO_EXPIRY`)
- If contract is in tier 2: ATR > 0.5 × 30-day median (regime gate)

## 5. Backtest harness

Reuses `scripts/strategy_bakeoff.py` pattern but extended for futures:

- Fetch 5m, 15m, 1h candles for each tier-1 contract over the last 2 quarters
- Stitch across rolls (drop bars within ±FUTURES_MIN_DAYS_TO_EXPIRY)
- Replay each candidate strategy with realistic ГО + commission
- Compute Sharpe, Sortino, max-DD, profit factor, Calmar
- Walk-forward CV with 30-day train / 7-day test rolling

Acceptance bar before going live:
- Sharpe > 0.5 on 2-quarter walk-forward
- Profit factor > 1.3
- Max-DD < 8 % of account
- ≥ 20 trades per contract per quarter (statistical significance)

## 6. Roll-out plan

**Phase 0 — Scaffolding (1-2 days).** Empty `futbot/` package, FutSettings
loading from existing `.env`, broker connection reuse, "hello world" loop
that just fetches multi-TF candles and prints them.

**Phase 1 — Pipeline w/o ML (3-5 days).** Layers 1, 3, 4 + risk audit. Run
on paper account or in dry-run logging mode (writes hypothetical orders to
`signals` table but doesn't trade). Compare hypothetical trades to current
bot's trades over the same 2-week window.

**Phase 2 — ML gate (1 week).** Train futures-only LGBM with regime
stratification. Plug into pipeline as veto layer. Re-run paper.

**Phase 3 — Sizing + execution (3-5 days).** ГО-aware sizer, Chandelier
trailing stop, real T-Invest orders. Start with **1 lot per contract,
2 contracts max** for the first week of live. Telegram alerts for every
order.

**Phase 4 — Full live (open-ended).** Lift the 2-contract cap once
hypothetical-vs-actual deviation < 5 % and live drawdown < 3 % over the
first month.

**Kill switch:** parallel-run with `trade_claude`. If `futbot` drawdown
exceeds 5 % from peak OR if hypothetical-vs-actual deviation > 15 %,
auto-disable trading (stays in paper mode), Telegram alert, wait for
human override.

## 7. Open questions before coding

1. **Roll automation** — when current contract approaches expiry, close
   open positions and reopen on next contract? Or just stop trading the
   contract 14 days before expiry (current `FUTURES_MIN_DAYS_TO_EXPIRY`)?
   The latter is simpler; the former is more capital-efficient.
2. **Margin call handling** — Tinkoff's margin_attributes API tells us
   `funds_sufficiency_level`. Do we auto-deleverage on level < 1.2, or
   alert and wait?
3. **Calendar/news blackout** — should we skip trading on CBR rate decision
   days (3rd Friday of mar/apr/jun/jul/sep/oct/dec)? Postmortem 2026-04-25
   suggested yes but never wired up.
4. **Shadow mode for current bot** — run new bot alongside current,
   compare P&L for 2 weeks before switching off the old bot?

Most are easy to defer — start with (4) explicit and (3) deferred.

---

## Appendix — referenced sources

- TradingAgents — <https://github.com/tauricresearch/tradingagents>
- LLM-TradeBot — <https://github.com/EthanAlgoX/LLM-TradeBot>
- Lumibot — <https://github.com/Lumiwealth/lumibot>
- FinMem — <https://github.com/pipiku915/FinMem-LLM-StockTrading>
- Funded Futures: 10 best algorithmic futures strategies — <https://fundedfuturesnetwork.com/blog/10-best-futures-algorithmic-trading-strategies/>
- QuantVPS automated futures playbook — <https://www.quantvps.com/blog/automated-futures-trading-strategies>
- Bookmap on trend vs mean reversion regimes — <https://bookmap.com/blog/key-algorithmic-trading-strategies-from-trend-following-to-mean-reversion-and-beyond>
