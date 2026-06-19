# futbot.scalp — частотный скальпинг-модуль

Высокочастотный (по меркам retail) сабмодуль `futbot`. Подписывается на
**streaming** order book + trades + 1-min candles для самых ликвидных FORTS
контрактов и торгует на сочетании microstructure + technical indicators.

## Что НЕ обещаю

Прямо и без оговорок:

- **«Зарабатывать каждый день» — не получится**. Даже у Renaissance
  Medallion ~55-60% профитных дней. Большой winning streak — это удача;
  устойчивый эдж проявляется на горизонте месяца, не дня.
- **Положительная даже-неделя не гарантируется**. Variance на скальпинге
  высокая, отдельная неделя может быть в минусе при положительном EV.
- Скальпинг **прибылен в среднем, не в каждой сделке**. Win-rate обычно
  50-55%; вытягивает асимметрия TP/SL.

## Что обещаю

- **Бот будет торговать 5-30 раз в торговый день** при дефолтных порогах.
  В paper-mode это бесплатный эксперимент — гоняй неделю, смотри
  фактическую статистику.
- **Все сделки логируются в `data/scalp.db`** — можно сделать честный
  postmortem.
- **Telegram-алерты** на каждое открытие/закрытие — видно происходящее в
  реальном времени.
- **Daily kill-switch и win-lock** автоматически останавливают бота при
  -1% drawdown или +1.5% gain — защита от плохих/хороших дней с
  перерасходом.

## Архитектура

```
futbot/scalp/
├── main.py             # entry: python -m futbot.scalp.main
├── config.py           # ScalpSettings (SCALP_* env keys)
├── stream.py           # MarketDataStream подписки + per-instrument state
├── microstructure.py   # book imbalance, microprice, TFI, intensity
├── indicators.py       # fast RSI(7), VWAP, ATR(14) на 1-min, MACD-short
├── strategy.py         # объединение сигналов → entry/exit decision
├── db.py               # scalp_trades таблица
└── README.md           # этот файл
```

## Сигнальная модель

Композитный score в `[-1, +1]`:

```
score = 0.40 × book_imbalance     ← главная альфа (Cont/Stoikov 2010)
      + 0.30 × trade_flow_imbalance (последние 30с buy vs sell volume)
      + 0.15 × indicator_alignment  (RSI(7) + EMA(9/21) + MACD(5/13/5))
      + 0.15 × vwap_pull            (mean-reversion к session VWAP)
```

Если `|score| ≥ 0.45` AND спред ≤ 3 тика AND TFI sample size достаточный
→ открываем 1 лот в сторону score, ставим SL=1.2 ATR, TP=1.5 ATR.

Выход — первое из:
- TP / SL hit
- Время в позиции > 10 минут
- Score флипается в обратную сторону с `|score| ≥ 0.30`
- Stale state (стрим без обновлений > 30с)

## Запуск

```bash
# Из корня репо (или из любой папки — config находит .env сам)
python -m futbot.scalp.main
```

Дефолт — **PAPER mode**. Никаких реальных ордеров. Проверь в логе:

```
scalp bot starting in PAPER mode (env: F:\trade_claude\.env)
Scalp universe: ['SiM6', 'BRM6', 'GZM6', 'SRM6', 'MXM6']
```

Каждые 30 секунд бот печатает **heartbeat** — что видит, какой score у
каждого контракта, почему не открыл сделку:

```
heartbeat:
  SiM6   POS ✓     book=10 trades=247 1m= 38  score=+0.62  book=+0.71  tfi=+0.42 (n=18)  reason=OK
  BRM6       ✓     book=10 trades=183 1m= 38  score=-0.18  book=-0.21  tfi=+0.05 (n=11)  reason=score -0.18 below threshold 0.45
  GZM6       STALE book= 0 trades=  0 1m=  0  score=+0.00  ...  reason=no book data yet
```

POS = открытая позиция. STALE = стрим без обновлений > 30с (не торгуем).

## Переход в live

⚠️ **Только после недели paper-режима** и проверки статистики:

```sql
sqlite3 data/scalp.db "
SELECT
  DATE(entry_time) day,
  COUNT(*) trades,
  ROUND(AVG(pnl), 2) avg_pnl,
  ROUND(SUM(pnl), 2) total_pnl,
  ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 1) win_pct
FROM scalp_trades WHERE exit_time IS NOT NULL
GROUP BY day ORDER BY day DESC;
"
```

Хочешь видеть стабильный плюс на 5+ днях, win rate ≥ 50%, и суммарный
P&L существенно больше комиссии (`COMMISSION_FUTURES_PCT × 2 × n_trades`).

Когда уверен:
```ini
# В .env
SCALP_PAPER_MODE=false
```
и перезапуск.

## Параметры для тюнинга

В `.env` можно переопределить любой `SCALP_*` ключ из `config.py`. Самые
полезные:

| Параметр | Дефолт | Что делает | Куда крутить |
|---|---|---|---|
| `SCALP_BOOK_IMBALANCE_MIN` | 0.25 | Минимум \|imbalance\| для сигнала | ↑ = реже но качественнее |
| `SCALP_TFI_MIN` | 0.15 | Минимум \|TFI\| | ↑ = реже |
| `SCALP_INITIAL_STOP_ATR` | 1.2 | Стоп в ATR | ↑ = меньше стопов, больше потерь когда они срабатывают |
| `SCALP_TAKE_PROFIT_ATR` | 1.5 | TP в ATR | ↑ = реже фиксируем, дольше держим |
| `SCALP_MAX_HOLD_SECONDS` | 600 | Time cap (10 мин) | ↑ = меньше тайм-каута, больше variance |
| `SCALP_MAX_TRADES_PER_DAY` | 30 | Жёсткий лимит | Защита от runaway |
| `SCALP_DAILY_LOSS_PCT_LIMIT` | 0.01 | Daily kill | НЕ повышай — это safety |
| `SCALP_TIER1_BASES` | Si,BR,GZ,SR,MX | Контракты | Добавляй только проверенно ликвидные |

## Известные ограничения

- **Нет лимитных ордеров** — используется `post_market_order`. На
  скальпинге это может стоить нам ~1 тик slippage на трейд. Если в
  paper-mode видишь edge — следующий шаг переделать на limit-aggressive
  через `broker.post_limit_with_fallback`.
- **Нет учёта дивидендов / экспирации** в P&L. Контракт за 14 дней до
  экспирации отфильтровывается, дивиденды на фьючи не платятся, так
  что это OK.
- **Heartbeat в логи раз в 30с** — будет много строк. Хочешь меньше →
  поставь `HEARTBEAT_SEC = 60` в `main.py`, или меняй logging level на
  `WARNING` для `futbot.scalp` логгера.
- **Один процесс на бота**. Если хочешь параллельно гонять
  `futbot.main` (4-слойный) и `futbot.scalp.main` — это два разных
  процесса, две разных БД. Telegram alerts будут идти от обоих в один
  чат.

## Что если ничего не торгует

Возможные причины (проверять по heartbeat):

| Симптом | Причина | Что делать |
|---|---|---|
| `STALE` у всех | Биржа закрыта или нет интернета | Проверить время MSK; проверить сеть |
| `warming up 1m candles (N)` | Бот стартовал недавно | Подождать 20-30 мин |
| `spread X ticks > 3` | Широкий спред — нет ликвидности | Сменить контракт или подождать |
| `score ±0.20 reason=below threshold 0.45` | Сигнал слабый | Снизить `SCALP_BOOK_IMBALANCE_MIN` (но это даёт меньше edge) |
| `Blackout hour MSK=X` | Сейчас 09 / 18 / 23 MSK | Бот специально не входит в эти часы |
| `Daily kill active` | Сегодня уже -1% или +1.5% | Сбросится в полночь UTC |

## Дальнейшее развитие

Реальные апгрейды, которые могут улучшить edge (в порядке окупаемости):

1. **Limit orders на entry** вместо market — экономия ~0.04% × 2 = 0.08%
   per trade. На 30 сделках в день это +2.4% / день экономии edge.
2. **Запись тиковой истории** в БД для офлайн-обучения ML модели,
   аналогичной существующей в `futbot.ml` но на тиковых фичах.
3. **Stat-arb пары** на коинтегрированных контрактах (GZ ↔ SR, BR ↔ NG).
   Spread-trading имеет другую структуру риска и часто живёт там, где
   directional нет.
4. **Адаптивные пороги** — `SCALP_BOOK_IMBALANCE_MIN` зависящий от текущей
   волатильности (в спокойный рынок строже, в волатильном слабее).
