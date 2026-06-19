# futbot.trend — multi-contract Bollinger breakout

Swing trend-following бот через 12-26 FORTS контрактов. Час-эвалюация,
держание днями, выходы только по mechanical band-flip (без SL/TP).

## Запуск

```bash
cd F:\trade_claude
python -m futbot.trend.main           # PAPER mode по умолчанию
```

Для live — `TREND_PAPER_MODE=false` в `.env`, перезапуск.

## Параметры

| Знакомо | По умолчанию | Что делает |
|---|---|---|
| `TREND_UNIVERSE_MODE` | `core` | `core` = 12 контрактов с Sharpe≥0.6 и trades≥7. `extended` = все 26 WF-survivors |
| `TREND_LOTS_PER_TRADE` | 1 | Лотов на сделку (fixed sizing) |
| `TREND_MAX_OPEN_POSITIONS` | 10 | Лимит параллельных позиций |
| `TREND_DAILY_LOSS_PCT_LIMIT` | 0.02 | Дневная просадка → kill |
| `TREND_MIN_DAYS_TO_EXPIRY` | 14 | Не торгуем контракт ближе |
| `TREND_ROLLOVER_DAYS` | 3 | Закрываем позицию за N дней до экспирации |
| `TREND_LOOP_SECONDS` | 3600 | Часовой цикл (= ТФ свечей) |

## Происхождение портфеля

26 контрактов прошли walk-forward validation:
- 180 дней истории
- 90 дней IN-sample → подобрали best (N, k)
- 90 дней OUT-of-sample → применили те же params, проверили
- Avg IS Sharpe +0.83 → Avg OOS Sharpe +0.66 (80% retention = real edge)

См. `scripts/trend_walk_forward.py` — перезапускай каждые 60-90 дней
чтобы пересмотреть list (β/regime drift).

## Топ-контракты в core-режиме (Sharpe≥0.6, trades≥7)

```
PX     Polyus Gold     N=20 k=2.5   OOS Sharpe 1.78
USDRU  USDRUBF         N=50 k=2.0   1.77 (только 3 трейда)
GD     Gold            N=20 k=1.5   1.26  ← workhorse, 9 trd/мес
SS     ?               N=20 k=2.5   1.09
SZ     ?               N=30 k=2.0   0.90
YD     Yandex          N=20 k=2.0   0.89
LT     ?               N=20 k=1.5   0.85
S1     ?               N=20 k=2.0   0.85
SOLper Solana perp     N=30 k=1.5   0.84  ← 11 trd/мес
VB     VTB             N=80 k=1.5   0.81
MV     ?               N=30 k=2.0   0.77
GK     Norilsk Nickel  N=30 k=2.5   0.61
```

## Activity и P&L (realistic forecast)

```
Core portfolio (12 контрактов):
  Trades/180d total:  ~150-200
  Trades/month:       ~25-33
  Trades/day:         ~1-2 (с волатильностью 0-5)

OOS NET (90 дней): +8000-10000 ₽
Live ожидание (slippage срезает 30-50%):
  90 дней: +4000-7000 ₽
  Annualised: +15-30% gross на занятый капитал
  Max DD: 10-25%
```

⚠️ Это **swing**. Каждый трейд держится **дни** (median 47-80 часов).

## Как бот ведёт себя в дневном цикле

Каждый час:
1. Daily kill check (если просадка дня > 2%, не открывает новые)
2. Refresh универсума (раз в 6ч)
3. Для каждого контракта:
   - Если открыта позиция и дней до экспирации ≤ 3 → **rollover close**
   - Загрузить 30 дней почасовых свечей
   - Вычислить Bollinger(N, k)
   - Решить: open / close / hold
4. Telegram alert на каждую сделку

## БД

`data/trend.db`:
- `trend_trades` — закрытые и открытые трейды с лентой Bollinger при входе

Запрос статистики:
```sql
sqlite3 data/trend.db "
SELECT base, direction, lots,
       ROUND(entry_price, 4) AS entry,
       ROUND(exit_price, 4) AS exit_,
       exit_reason,
       ROUND(pnl, 2) AS pnl,
       ROUND((julianday(exit_time)-julianday(entry_time))*24, 1) AS held_h
FROM trend_trades
WHERE exit_time IS NOT NULL
ORDER BY exit_time DESC
LIMIT 30;
"
```

## Reconcile при рестарте

- Контракт в портфеле + позиция в БД → автоматически продолжаем управлять
- Контракт **не в портфеле** + позиция в БД → force-close at market на старте

## Live execution caveat

Сейчас market orders. Для контрактов с limit-only mode (Si family) live
не пойдёт — пройдёт в paper, но в live упадёт при попытке открыть.
Текущий список 26 контрактов прошёл WF на market-order-style backtest;
все включены — но в live режиме контракт-by-contract будут отдельные
краши на запретах.

## Реалистичные ожидания

- **Это НЕ "каждый день в плюсе"** — это «месячный плюс при ~55% профитных дней»
- 1-2 сделки в день в среднем (бывают дни 0, бывают 5)
- Каждая сделка держится 1-7 дней
- Drawdown 10-25% **гарантирован** где-то в году
- После 2-3 месяцев paper'a — если PF > 1.2, можно осторожно в live

## Что НЕ реализовано (TODO)

- Limit-aggressive ордера (сейчас market — даст +0.5-1 тик slippage в live)
- Vol-targeting на размер позиции (сейчас fixed 1 lot)
- Auto-rerun walk-forward каждые 60 дней (сейчас вручную)
- Telegram /команды для trend бота
