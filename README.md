# Autonomous Trading Bot — T-Invest (T-Bank) + Telegram

Multi-strategy algorithmic trading bot for the Moscow Exchange (MOEX / FORTS),
built on the official **t-tech-investments** SDK. A single supervised
orchestrator runs several independent strategies against one broker account,
with Telegram control, a Mini App dashboard, and defensive risk management.

> ⚠️ **Not investment advice.** This is a personal research project trading real
> money with leverage (futures). Historical results do not guarantee future
> results. Use at your own risk.

## Contents
- [Overview](#overview)
- [Strategies](#strategies)
- [Breakdown signal](#breakdown-signal)
- [Risk & sizing](#risk--sizing)
- [Exits](#exits)
- [Reliability](#reliability)
- [Telegram & Mini App](#telegram--mini-app)
- [Install](#install)
- [Configuration](#configuration)
- [Run](#run)
- [Repo layout](#repo-layout)

## Overview

The live entry point is **`futbot.orchestrator.main`**. It connects one shared
broker client + one Telegram session, then runs each enabled strategy as a
supervised `asyncio` task at its own cadence. If a strategy tick crashes, the
supervisor logs it and restarts after exponential backoff — one strategy can't
take down the others.

```
orchestrator.main
├── shared BrokerClient (t-tech gRPC)  ── core/broker.py
├── shared TelegramNotifier + command server
├── runner: trend       (pattern breakouts, hourly bars)
├── runner: breakdown   (volume breakdown shorts, 2h bars)
├── runner: carry        (Si calendar spread — disabled by default)
├── monitor loop (5 min): stop/target/timeout + broker reconcile
└── heartbeat (6h) + Mini-App menu button
```

Each strategy has its own SQLite DB (`data/*.db`); the broker and notifier are
shared. A cross-process **singleton lock** prevents a second orchestrator from
trading the same account.

### SDK note (important for RU networks)
Migrated from the abandoned `tinkoff-investments` to the official
**`t-tech-investments`** (namespace `t_tech.invest`). The new SDK defaults to
`invest-public-api.tbank.ru`, whose TLS cert chains to the *Russian Trusted Root
CA* — absent from grpc/certifi trust stores. The broker pins the connection to
the legacy host with a matching SNI override, so verification succeeds:

```python
AsyncClient(token, target="invest-public-api.tinkoff.ru:443",
            options=[("grpc.ssl_target_name_override",
                      "invest-public-api.tinkoff.ru")])
```

## Strategies

| Strategy | Idea | Timeframe | Status |
|---|---|---|---|
| **breakdown** | Short a stock (via its future) when it breaks below a *quiet* consolidation on abnormal volume, in a downtrend | 2h bars | LIVE |
| **trend** | Chart-pattern breakouts (triple-top/bottom etc.) on FORTS futures + a few USD perps | hourly | LIVE |
| **carry** | Si calendar-spread carry (delta-neutral) | hourly | disabled (chronic laggard) |

Signal is computed on the **stock** (cleaner volume structure); execution is on
the **front-month future** (many shares can't be shorted; futures can). Stop and
target are scaled from stock-space onto the future, then **re-anchored to the
actual fill** so slippage on illiquid futures can't distort the 3:1 geometry.

## Breakdown signal

Evaluated on each just-closed 2h bar; all conditions must hold:

1. **Range break** — `close < min(low)` of the prior 12 bars (~24h).
2. **Volume confirmation** — `volume ≥ 3× median` of those 12 bars.
3. **Bar strength** — `close/open − 1 ≤ −1%`.
4. **Downtrend** — `close < SMA(120)` (~10 trading days).
5. **Stop sanity** — skip if the stop is farther than 10% from entry.
6. **Consolidation filter** — the run-up must be *calm*: realized-volatility
   percentile of the prior bars ≤ 0.40. This distinguishes a coiled-spring
   breakout from chasing the bottom of an already-volatile waterfall; on a
   3-year backtest it roughly halves drawdown and turns every year positive.

Two market-regime gates block **new** entries (existing positions ride their
stops):
- **Panic filter** — skip when the universe's 24h realized volatility is above
  its 85th percentile (broad panic → bounce-stops).
- **Per-bar cap** — at most 2 new entries per signal tick (limits correlated
  cluster losses).

## Risk & sizing

- **Risk-based sizing** — lots ≈ `equity × 1.5% / (stop_distance × rub_per_point
  × lot_size)`, so each trade risks a fixed fraction of equity regardless of the
  instrument.
- **Broker-authoritative cap** — the requested lots are capped by
  `orders.get_max_lots` (real margin capacity), so an order is never rejected for
  insufficient margin.
- **Position caps** — ≤ 25% of equity as initial margin per position, an
  absolute lot ceiling, ≤ 5 concurrent positions, and a `2× ГО` free-margin
  buffer before any entry.
- **Liquidity-pool sizing** — half risk when an intact swing low sits between
  entry and target (that trade class historically earns half as much per unit
  risk).
- **Pre-trade commission** — `orders.get_order_price` reports the exact
  commission, shown on the entry notification.

## Exits

- **Stop** at the breakout bar's high (a protective exchange stop-order is placed
  at entry as the hard backstop).
- **Target** at `entry − 3 × risk` (RR 3:1).
- **Timeout** after 48 **trading** hours — weekend hours don't count, so a Friday
  entry isn't dumped Monday before the thesis has trading time to play out.
- **Proactive reconcile** — every manage cycle checks broker truth; a position
  already flat at the broker (stop fired / closed externally) is closed in the DB
  at its **real** covering-trade price (from `get_operations`), not the current
  market price.

## Reliability

Lessons baked in after real incidents:
- **Notifier resilience** — a transient timeout at boot no longer permanently
  disables Telegram alerts; sends retry.
- **Heartbeat** — an "alive + open positions" message to Telegram every 6h, so a
  silent death is noticed.
- **Watchdog** (`scripts/watchdog.ps1`) — restarts the bot if the log goes stale
  (safe via the singleton lock).
- **Deferred order failures** — a market-closed rejection defers the close and
  retries next cycle instead of crash-looping.

## Telegram & Mini App

Commands: `/status` `/open` `/pnl` `/trend` `/stop` (two-step confirm) `/app`
`/help`.

The **Mini App** (`futbot/webapp/`) is a dark, mobile-style dashboard served by a
small aiohttp server: open positions with live price + stop/target progress, P&L
and equity curve, closed-trade feed, an event feed parsed from the log, and 2h
candlestick charts (lightweight-charts) with the open position's entry/stop/
target overlaid. `/app` (or the chat menu button) opens it inside Telegram.

## Install

Requires Python 3.11.

```bash
pip install -r requirements.txt
# the T-Bank SDK comes from the T-Bank package registry:
pip install t-tech-investments \
  --index-url https://opensource.tbank.ru/api/v4/projects/238/packages/pypi/simple
```

## Configuration

Create `.env` (never commit it):

```
T_INVEST_TOKEN=<your T-Invest API token>
T_INVEST_ACCOUNT_ID=<account id>
TELEGRAM_BOT_TOKEN=<bot token>
TELEGRAM_CHAT_ID=<your chat id>
# optional, for the Mini App button:
WEBAPP_URL=https://<your-https-tunnel-or-domain>
```

Strategy toggles live in `futbot/orchestrator/config.py`
(`ORCH_ENABLE_TREND/BREAKDOWN/CARRY`); per-strategy parameters in each
strategy's `config.py`.

## Run

```bash
python -m futbot.orchestrator.main      # the trading bot
python -m futbot.webapp.server          # the Mini App dashboard (port 8088)
```

## Repo layout

```
futbot/
  orchestrator/   main.py, trend_bot.py, breakdown wiring, commands
  breakdown/      volume-breakdown short strategy
  trend/          pattern-breakout strategy
  carry/          calendar-spread carry (disabled)
  pairs/          cointegration pairs strategy (disabled)
  patterns/       chart-pattern detectors used by trend
  webapp/         Telegram Mini App (server + static front)
  telegram_*.py   notifier + command handlers
core/broker.py    async t-tech gRPC wrapper (rate-limited, reconnecting)
config/settings.py   env-driven settings (pydantic)
utils/            shared helpers
scripts/watchdog.ps1   scheduled-task watchdog (restarts on stale log)
data/             per-strategy SQLite DBs + logs (gitignored)
```
