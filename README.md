# Autonomous Trading Bot — T-Invest + Telegram

Автономный торговый робот для Московской биржи через T-Invest API. ML-сигналы (LightGBM с per-asset-class изотонической калибровкой и meta-labelling), академический риск-менеджмент (Kelly + vol-targeting + Kyle-impact), полный цикл управления через Telegram. Работает на акциях и фьючерсах FORTS, лонг и шорт.

> Подробное архитектурное описание с UML-диаграммами — `docs/architecture.tex` (компилируется в PDF через `pdflatex`).

---

## Содержание

1. [Возможности](#возможности)
2. [Архитектура](#архитектура)
3. [Структура файлов](#структура-файлов)
4. [Установка](#установка)
5. [Конфигурация](#конфигурация)
6. [Запуск](#запуск)
7. [Telegram-команды](#telegram-команды)
8. [Режимы и профили](#режимы-и-профили)
9. [ML-модель и фичи](#ml-модель-и-фичи)
10. [Управление рисками](#управление-рисками)
11. [Логика входа и выхода](#логика-входа-и-выхода)
12. [Постмортем-фиксы](#постмортем-фиксы)
13. [База данных](#база-данных)
14. [Известные ограничения](#известные-ограничения)

---

## Возможности

### Торговля
- **Акции** ММВБ (рублёвые, через `share` instrument kind)
- **Фьючерсы** FORTS (BR, GZ, GD, SR, RI, MM и др. — отдельный confidence threshold и position-sizer на ГО)
- **Лонг и шорт** (шорты только на инструменты с `short_enabled_flag=True`, gate-фильтры жёстче)
- Лимитные ордера (limit_aggressive по умолчанию) с фолбэком на маркет через `LIMIT_ORDER_TIMEOUT`
- TWAP-исполнение крупных позиций (Almgren-Chriss `optimal_slices`) — включается автоматически при `participation_rate > 5%`
- Bracket-ордера: SL и TP размещаются как broker-side stop orders, плюс программный fallback в position monitor
- Частичный take-profit (scale-out 50% при MFE ≥ 1.5×ATR)
- Trailing stop (активируется при 70% прогресса к TP, отступ 1×ATR)

### ML / аналитика
- **LightGBM** 3-классовый классификатор (`sell=0 / hold=1 / buy=2`)
- **Triple-barrier labels** (López de Prado, гл. 3) с горизонтом, выбранным по IC grid-search (h=20 для акций, h=10 для фьючерсов)
- **Per-asset-class изотоническая калибровка** P(buy)/P(sell) — отдельные калибраторы на акции, ФОРТС-классы (BR/GZ/GD/SR/RI/MM/Currency/Other)
- **Meta-labelling** — вторичный бинарный классификатор предсказывает корректность направления первичного, итоговая уверенность = √(primary × meta)
- **35+ фичей**: технические индикаторы (RSI, MACD, BB, ATR, OBV, ADX, Williams %R, Stochastic), фрактальное дифференцирование close-серии (Hurst-stationarity), макро (IMOEX, USDRUB, RGBI), order-book (spread_bps, imbalance), futures-специфичные (days_to_expiry, in_roll_window)
- **Walk-forward CV с эмбарго** (López de Prado, гл. 7) — 5 фолдов, expanding window
- **CPCV** (Combinatorial Purged Cross-Validation) опционально
- **Авто-переобучение** каждые 24 часа с rollback-гейтом по acc/F1
- **Универсальная модель** на пуле тикеров (n>50k bars) — лучше overfit-stable одно-тикерных
- **Detect режима** (trending / ranging / high-volatility) — динамические веса стратегий

### Риск-менеджмент
- **Kelly criterion** (дробный — `KELLY_FRACTION` × confidence × regime_scale)
- **Vol-targeting** (Frazzini-Pedersen 2014, Moskowitz 2012) — параллельно с Kelly, выбирается min(Kelly, vol-target)
- **Kyle-Obizhaeva model** для оценки price impact (square-root law)
- **Glosten-Milgrom spread-фильтр** (`SPREAD_THRESHOLD × median_spread`)
- **Daily-loss circuit breaker** (`MAX_DAILY_LOSS_PCT`) и **drawdown breaker** (`MAX_DRAWDOWN_PCT`)
- **StoplossGuard** (freqtrade-pattern) — пауза стороны после N последовательных стоп-лоссов
- **Long auto-pause** — отключает лонги когда realized Kelly f* < 0 (на основе истории по направлению)
- **Carry-free cap** для шортов — лимит лотов под порог Тинькофф, чтобы платить 0 carry
- **Same-ticker cooldown** против whipsaw

### Telegram
- **Три режима**: `autonomous` (полный авто), `advisory` (бот сигналит, юзер апрувит), `interactive` (только ручные сделки)
- **Inline-кнопки** для каждой операции
- Уведомления о signal / entry / exit / circuit-breaker
- **Heartbeat-watchdog** для polling — `bot.get_me()` пинг каждые 2 мин, force-restart при 3 подряд фейлах
- Авторизация по chat_id (один пользователь)
- Команды: `/status`, `/portfolio`, `/pnl`, `/positions`, `/buy`, `/sell`, `/mode`, `/risk`, `/profile`, `/watchlist`, `/retrain`, `/sync`, `/analyze`, `/signals`, `/addticker`, `/removeticker`, `/stop`

---

## Архитектура

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          asyncio event loop                              │
│                                                                          │
│  ┌────────────────┐    asyncio.Queue    ┌───────────────────────────┐    │
│  │ TradingEngine  │◄──notifications────►│        TelegramBot        │    │
│  │                │                     │  Application + Updater    │    │
│  │  • autonomous_loop                   │  HeartbeatWatchdog       │    │
│  │  • position_monitor_loop             │  NotificationService     │    │
│  │  • portfolio_sync_loop               └───────────────────────────┘    │
│  │  • retrain_scheduler                                                  │
│  └────────┬───────────┬───────────┬───────────┬──────────┬────────┐      │
│           │           │           │           │          │        │      │
│      ┌────▼────┐ ┌────▼────┐ ┌────▼────┐ ┌────▼────┐ ┌──▼─────┐ ┌▼────┐  │
│      │Screener │ │Strategy │ │  Risk   │ │ML model │ │ Broker │ │ DB  │  │
│      │ momentum│ │ ML+TA + │ │ Manager │ │LGBM+meta│ │T-Invest│ │SQLite│ │
│      │  vol +  │ │ regime  │ │ 8 gates │ │+isotonic│ │  gRPC  │ │+WAL │  │
│      │ liquidity│ │ vote    │ │+sizers  │ │         │ │  + REST│ │     │  │
│      └─────────┘ └─────────┘ └─────────┘ └─────────┘ └────────┘ └─────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 4 параллельные корутины внутри `TradingEngine.run()`

| Корутина | Период | Задача |
|---|---|---|
| `_autonomous_loop` | `SCAN_INTERVAL_MINUTES` (30/60) | screen → signal → risk gate → execute |
| `_position_monitor_loop` | `PORTFOLIO_CHECK_MINUTES` (5) | software SL/TP, trailing stop, time exit, signal-reversal exit, partial-TP |
| `_portfolio_sync_loop` | 1 мин | реконсиляция с брокером — ловит broker-side stop fills, классифицирует exit_reason |
| `_retrain_scheduler` | `ML_RETRAIN_HOURS` (24) | обучение новой модели → rollback gate → swap |

---

## Структура файлов

```
trade_claude/
├── main.py                         # entry point — собирает все компоненты, async event loop
├── config/
│   ├── settings.py                 # pydantic Settings (env-aware), все тюнабли
│   └── instruments.py              # asset-class коды, blacklist, TRADING_PROFILES
├── core/
│   ├── broker.py                   # AsyncBroker (gRPC + REST T-Invest), retry с jitter
│   └── engine.py                   # TradingEngine — 4 цикла, вход/выход/sync
├── strategy/
│   ├── base.py                     # Signal, ExitSignal, BaseStrategy
│   ├── ml_strategy.py              # MLStrategy — обёртка над LGBMModel + meta
│   ├── technical_strategy.py       # 5 технических правил (MACD cross, BB squeeze, ...)
│   └── regime.py                   # RegimeDetector — ADX/ATR-based, веса стратегий
├── ml/
│   ├── model.py                    # LGBMModel + IsotonicRegression калибраторы
│   ├── trainer.py                  # ModelTrainer — TB labels, walk-forward CV, rollback
│   ├── meta_model.py               # Meta-labeller (бинарный классификатор)
│   └── cpcv.py                     # Combinatorial Purged Cross-Validation
├── analysis/
│   ├── features.py                 # build_features (35+ фичей)
│   ├── indicators.py               # технические индикаторы
│   ├── frac_diff.py                # фрактальное дифференцирование
│   ├── macro.py                    # MacroProvider (IMOEX, USDRUB, RGBI)
│   ├── screener.py                 # Screener — universe scan, top-N momentum
│   └── reports/                    # YAML-постмортемы + fix_proposal.md
├── risk/
│   ├── manager.py                  # RiskManager.approve_trade — 8 гейтов
│   ├── position_sizer.py           # Kelly + vol-targeting + Kyle impact
│   ├── execution.py                # ExecutionScheduler — TWAP, Almgren-Chriss
│   └── spread_monitor.py           # Glosten-Milgrom rolling-median spread
├── telegram_bot/
│   ├── bot.py                      # Application + heartbeat watchdog
│   ├── handlers.py                 # /status /buy /portfolio ... /sync
│   ├── notifications.py            # NotificationService (asyncio.Queue → bot.send_message)
│   ├── keyboards.py                # InlineKeyboardMarkup для всех команд
│   └── formatters.py               # P&L tables, position cards
├── database/
│   └── db.py                       # Repository (aiosqlite) — 6 таблиц
├── utils/
│   └── helpers.py                  # is_moex_open, now_msk, ...
├── scripts/
│   ├── tune_model.py               # GridSearch гипер-параметров + horizon search
│   ├── train_full.py               # offline-обучение универсальной модели
│   ├── weekly_postmortem.py        # YAML отчёты + fix_proposal.md
│   ├── paper_replay.py             # бэктест на исторических свечах
│   └── replay_with_model.py        # шаг-за-шагом replay для дебага модели
├── docs/
│   └── architecture.tex            # LaTeX-документ с UML-диаграммами
├── data/
│   ├── trade_bot.db                # SQLite база (signals, trades, daily_pnl, models, ...)
│   ├── logs/bot.log                # rotating log
│   └── models/universal_v*.joblib  # сохранённые LGBMModel + калибраторы + meta
├── requirements.txt
├── .env                            # секреты + override профиля (см. ниже)
└── README.md
```

---

## Установка

```bash
git clone <repo> trade_claude
cd trade_claude
python -m venv .venv
.venv\Scripts\activate              # Windows
pip install -r requirements.txt
cp .env.example .env                # затем заполнить токены
mkdir data data/logs data/models
```

Зависимости: Python 3.11+, `tinkoff-investments`, `python-telegram-bot[ext]`, `lightgbm`, `pandas`, `numpy`, `scikit-learn`, `pydantic-settings`, `aiosqlite`.

---

## Конфигурация

### `.env` (минимум)

```ini
T_INVEST_TOKEN=t.your_token_here
T_INVEST_ACCOUNT_ID=2263XXXXXX
TELEGRAM_BOT_TOKEN=12345:ABC
TELEGRAM_CHAT_ID=0                  # будет установлен автоматически после первого /start

MODE=autonomous                     # autonomous | advisory | interactive
ACTIVE_PROFILE=conservative         # conservative | moderate | aggressive
INCLUDE_FUTURES=true
ALLOW_SHORTS=true
```

### Профили (см. `config/instruments.py:TRADING_PROFILES`)

| Параметр | conservative | moderate | aggressive |
|---|---|---|---|
| MAX_POSITIONS | 3 | 5 | 8 |
| MAX_POSITION_PCT | 10 % | 20 % | 30 % |
| MAX_PORTFOLIO_RISK_PCT | 1 % | 2 % | 3.5 % |
| MAX_DAILY_LOSS_PCT | 1.5 % | 3 % | 5 % |
| MAX_DRAWDOWN_PCT | 6 % | 10 % | 15 % |
| SIGNAL_THRESHOLD | 0.70 | 0.60 | 0.50 |
| KELLY_FRACTION | 0.15 | 0.25 | 0.40 |
| SCAN_INTERVAL_MINUTES | 60 | 30 | 15 |

Профиль применяется через Telegram `/profile <name>`.

---

## Запуск

```bash
python main.py
```

Логи параллельно идут в `stdout` и `data/logs/bot.log`. Для просмотра последних сделок:

```bash
sqlite3 data/trade_bot.db "SELECT entry_time, ticker, direction, status, pnl_pct, exit_reason FROM trades ORDER BY id DESC LIMIT 20"
```

Скан-цикл резюмируется одной строкой `Scan summary: candidates=30 {hold: 24, executed: 2, max_positions: 1}` — если бот молчит, эта строка покажет точную причину.

---

## Telegram-команды

| Команда | Что делает |
|---|---|
| `/start` | привязка chat_id, главное меню |
| `/status` | текущий режим + market state + позиции + дневной P&L |
| `/portfolio` | таблица текущих позиций с unrealized P&L |
| `/pnl` | сводка P&L day / week / month / total |
| `/positions` | список открытых позиций с inline-кнопками "закрыть" |
| `/buy <TICKER> [lots]` | ручная покупка с подтверждением |
| `/sell <TICKER> [lots]` | ручная продажа/шорт |
| `/mode <name>` | переключить режим работы |
| `/profile <name>` | сменить торговый профиль |
| `/risk` | текущие risk-метрики и состояние circuit-breakers |
| `/watchlist` | топ-30 кандидатов из последнего скана |
| `/signals` | последние сгенерированные сигналы |
| `/retrain` | принудительно запустить retrain |
| `/sync` | реконсиляция с брокером (синхронизация позиций) |
| `/analyze` | запуск forensic-анализа последней недели |
| `/addticker <TICKER>` | добавить кастомный тикер в watchlist |
| `/removeticker <TICKER>` | убрать |
| `/stop` | emergency stop с двойным подтверждением |

---

## Режимы и профили

**`MODE=autonomous`** — бот сам сканирует, открывает и закрывает позиции, юзер только смотрит.

**`MODE=advisory`** — каждый сигнал отправляется в Telegram с inline-кнопками "Approve / Reject", позиция открывается только после нажатия.

**`MODE=interactive`** — авто-скан выключен, торговля только через `/buy /sell`.

---

## ML-модель и фичи

### Triple-barrier labels (López de Prado гл. 3)

Цена входа = current close. Барьеры:
- Profit target (PT) = entry + ATR × `target_mult`
- Stop loss (SL) = entry − ATR × `stop_mult` (симметрично PT для balanced classes)
- Time barrier = entry_bar + `TB_MAX_HOLD` (20 для акций, 10 для фьючерсов)

Класс = первый сработавший барьер: PT→buy, SL→sell, time→hold.

### Calibration (per asset class)

После основного обучения LightGBM, на out-of-fold предсказаниях обучается **`IsotonicRegression`** отдельно для P(buy) и P(sell) **на каждый asset class** (shares, BR, GZ, GD, SR, RI, MM, Currency, Other). Калибратор приводит сырые вероятности к фактической base rate.

### Meta-labelling

После основного обучения собираются OOF-предсказания и обучается бинарный классификатор: `meta_label = 1` если первичный prediction оказался корректным. На инференсе:
```python
final_conf = (primary_conf * meta_conf) ** 0.5
direction = "hold" if final_conf < SIGNAL_THRESHOLD else primary_direction
```

### Rollback gate

После обучения новой модели сравниваются `acc` и `f1_weighted` с предыдущей версией на одном holdout. Если новая хуже на ≥0.02 — rollback (модель НЕ заменяется). Изменение горизонта TB пропускает гейт (метрики на разных горизонтах несравнимы).

---

## Управление рисками

### `RiskManager.approve_trade` — 8 гейтов

1. **Spread filter** — Glosten-Milgrom: текущий spread > `SPREAD_THRESHOLD × median_spread` ⇒ отказ
2. **Daily-loss breaker** — если `daily_pnl_pct < -MAX_DAILY_LOSS_PCT` ⇒ блок до следующего дня
3. **Drawdown breaker** — `current_equity < peak_equity × (1 - MAX_DRAWDOWN_PCT)` ⇒ полный stop
4. **Position concentration** — `MAX_POSITIONS`, `MAX_POSITION_PCT`
5. **Stoploss guard** — `STOP_GUARD_COUNT` стоп-лоссов на стороне за `STOP_GUARD_LOOKBACK_HOURS` ⇒ пауза
6. **Long auto-pause** — realized Kelly f* < 0 на лонг-истории ⇒ блок лонгов кроме `confidence ≥ LONG_MIN_CONFIDENCE_WHEN_BAD_EDGE`
7. **Short permission** — `short_enabled_flag` инструмента + `ALLOW_SHORTS`
8. **Position sizing** — `min(Kelly_lots, vol_target_lots, max_position_lots)` ≥ 1

### Position sizing

```
kelly_lots     = portfolio_value × KELLY_FRACTION × confidence × regime_scale / (lot_value × stop_distance_pct)
vol_target_lots = portfolio_value × VOL_TARGET_DAILY_PCT / (daily_vol × lot_value)
max_lots       = min(kelly, vol_target, MAX_POSITION_PCT × portfolio / lot_value)
```

Для шортов дополнительно умножается на `SHORT_POSITION_SCALE` (0.75) и кэпится `SHORT_CARRY_FREE_THRESHOLD_RUB / lot_value`.

Для фьючерсов используется ГО (initial margin) вместо номинала.

---

## Логика входа и выхода

### Вход (`_run_scan_cycle`)

1. **Cycle-level гейты** (вне per-ticker loop): `is_moex_open`, `SKIP_ENTRY_WEEKDAYS`, `SKIP_ENTRY_HOURS_MSK`
2. Build watchlist: screener возвращает 30 кандидатов (~20 long + 10 short по моментуму, vol, liquidity)
3. По каждому кандидату:
   - skip if уже открыта позиция / в cooldown / max_positions reached
   - skip if shorts gap-open window (первые 30 мин после открытия)
   - `_evaluate_instrument` → fetch candles + order_book + macro → каждая стратегия отдаёт Signal → weighted vote
   - skip if signal direction != candidate thesis
   - skip if confidence < threshold (для шортов — `SHORT_MIN_CONFIDENCE`)
   - skip if `now_msk().hour >= SHORT_ENTRY_CUTOFF_HOUR_MSK` для шортов (carry)
   - `RiskManager.approve_trade` → 8 гейтов
   - `_execute_trade` → лимит-ордер, ждём fill, ставим SL/TP brackets
4. **Scan summary**: одна INFO-строка с counters по причинам skip

### Выход (`_position_monitor_loop`, каждые 5 мин)

Для каждой открытой позиции, в порядке:

1. **Hard SL/TP check** (software fallback) — `current_price <= stop_loss` или `>= take_profit` ⇒ market close immediately. Это страховка на случай потери broker-side stop ордера (постмортем 25-апр-2026).
2. **Trailing stop** — после `TRAILING_STOP_ACTIVATION_FRAC × target_distance` пройдено, обновляется peak; close если `current ≤ peak − TRAILING_STOP_ATR_MULT × ATR`
3. **Partial TP** — при MFE ≥ `PARTIAL_TP_TRIGGER_ATR × ATR` закрыть `PARTIAL_TP_FRAC` лотов (только если `lots ≥ PARTIAL_TP_MIN_LOTS`)
4. **Signal-reversal** — стратегия даёт `should_exit()` с reason="signal_reversal", urgency="immediate" ⇒ close (с проверками `MIN_TARGET_PROGRESS_FRAC` и `MIN_EXIT_PROFIT_MULT_COMMISSION`)
5. **Time exit** — `held > TB_MAX_HOLD` бар И `|pnl_pct| < TIME_EXIT_MAX_PNL_PCT`

### Broker-closed reconciliation (`_sync_portfolio`)

Если позиция исчезла из брокера, но в DB ещё `status='open'`:
- Получить exit_price из operations history
- Сравнить с stop_loss / take_profit / partial-tp ⇒ infer reason (`stop_loss` / `take_profit` / `partial_tp` / `external_close`)
- Записать в DB как closed с правильным reason

---

## Постмортем-фиксы (последний месяц)

| Дата | Найдено | Применено |
|---|---|---|
| 25-апр | Hard SL/TP не срабатывал в коде, держался только broker-side | Software fallback в `_check_exit_conditions` |
| 25-апр | Realized Kelly f* = −0.23 (отрицательный edge) | `KELLY_FRACTION 0.5 → 0.15` |
| 25-апр | Q4 confidence (0.83-0.92) худшая по P&L (инверсия) | `SIGNAL_THRESHOLD 0.65 → 0.72` |
| 25-апр | Часы 7-9, 11-12 МСК систематически плохи | Введён `SKIP_ENTRY_HOURS_MSK` |
| 27-апр | `SKIP_ENTRY_WEEKDAYS=[0,4]` блокировал Mon+Fri, +weekend → 3 дня без сделок | `SKIP_ENTRY_WEEKDAYS=[]` (фильтр был спекулятивен) |
| 27-апр | Все signal rows в DB остались `approved=0` (исполненные тоже) | `update_signal_approval` после risk-check |
| 27-апр | Per-cycle skip-причины не логировались | INFO `Scan summary` в конце цикла |
| 30-апр | 47% лосеров касались +0.5% MFE до разворота | `PARTIAL_TP_ENABLED=True`, scale-out 50% при +1.5×ATR |
| 30-апр | Long realized f* = −0.62 / short f* = +0.57 | `LONG_AUTO_PAUSE`, шорты разрешены, тариф каждой стороны раздельно |
| 30-апр | WUSH/GTRK 3× стопов подряд | `STOP_GUARD_*` (freqtrade-pattern) |
| 04-май | Telegram бот переставал отвечать после долгого network-блипа | Heartbeat-watchdog (`bot.get_me()` каждые 2 мин, force-restart при 3 фейлах) |
| 04-май | `SKIP_ENTRY_HOURS_MSK` имел слабую статистику | Отключён (re-enable при ≥30 trades/hour) |

---

## База данных

`data/trade_bot.db`, SQLite + WAL mode. Основные таблицы:

| Таблица | Назначение |
|---|---|
| `signals` | Каждый сгенерированный сигнал (ticker, direction, confidence, features-JSON, approved, rejection_reason) |
| `trades` | Открытые и закрытые позиции (entry/exit_time, prices, lots, pnl, exit_reason, signal_confidence, stop_loss, take_profit) |
| `daily_pnl` | Дневная сводка (portfolio_value, daily_pnl_pct, peak_equity) для circuit-breakers |
| `models` | Реестр моделей (version, accuracy, f1, train_samples, tb_max_hold) |
| `custom_tickers` | Кастомные тикеры из `/addticker` |
| `position_peaks` | MFE/MAE и peak для trailing stop (восстанавливается при рестарте) |

---

## Известные ограничения

- Облигаций и валюты в торговле НЕТ (только акции + FORTS futures). Облигации можно использовать как пассив.
- Шорты только на share-инструменты с `short_enabled_flag` (требует маржинальный счёт). Фьючерсные шорты бесплатны (нет carry).
- Heart heartbeat-ping в Telegram-watchdog не отслеживает специфические pollyng-зависания внутри `getUpdates` long-poll, только полное падение API.
- Калибраторы тренируются только если в OOF есть ≥50 сэмплов на класс — на холодном старте калибровка пустая, модель использует raw probs.
- `ATR_FILTER_MEDIAN_MULT=0.8` отсекает ~30% кандидатов на низковолатильных днях (по дизайну).
- `is_moex_open()` использует таблицу праздников MOEX зашитую в `utils.helpers` — обновлять вручную раз в год.

---

## Лицензия

Internal / proprietary. Использование на свой риск.
