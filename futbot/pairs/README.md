# futbot.pairs — статистический арбитраж на коинтегрированных парах FORTS

Swing-стратегия. **Не** скальпинг. Тайминг — часовая оценка, holding
2-7 дней. Edge — mean-reversion спреда между коинтегрированными
фьючерсными контрактами.

## Параметры (выбраны из 180-дневного grid search)

| Параметр | Значение | Почему |
|---|---|---|
| `PAIRS_Z_ENTRY` | 2.0 | Sweet spot между числом сделок и win-rate |
| `PAIRS_Z_STOP` | 4.0 | Структурный пробой коинтеграции |
| `PAIRS_MAX_HOLD_HOURS` | 48 | На сетке 24-168h работает; 48h — баланс активности и Sharpe |
| `PAIRS_ROLLING_Z_WINDOW_HOURS` | 240 | 10 дней — устойчиво при дрейфующих β |
| `PAIRS_REFIT_BETA_HOURS` | 168 | Раз в неделю — β меняется медленно |
| `PAIRS_LIST` | LK-Si, SR-Si, GZ-Si, SR-MX | adf_p ≤ 0.10 на 180 днях |
| `PAIRS_CAPITAL_PER_PAIR_PCT` | 10% | На 4 пары = 40% портфеля одновременно |

## Запуск

```bash
cd F:\trade_claude
python -m futbot.pairs.main           # PAPER mode по умолчанию
```

Для live — `PAIRS_PAPER_MODE=false` в `.env`, перезапуск. **Минимум 30
дней paper** перед live (стратегия имеет малую частоту: ~5 сделок/месяц на
пару; нужно собрать выборку).

## Что бот делает каждый час

1. **Daily kill check** — если просадка дня > 2% → не открывает новых.
2. **Скачивает 240h хвост часовых свечей** для всех контрактов.
3. **Раз в неделю** перефитит β + α + spread stats для каждой пары
   (Engle-Granger ADF). Сохраняет в `pair_state` таблице.
4. **Для каждой пары** считает текущий z-score:
   - Нет позиции AND \|z\| ≥ 2.0 → открыть spread в направлении
     против отклонения.
   - Есть позиция AND (z пересёк 0 OR \|z\| > 4 OR held > 48h) → закрыть.
5. **Telegram alert** на каждое открытие/закрытие.

## Логика спред-позиций

`spread = price(y) − β × price(x)`

- z > +2 → spread overshoot вверх → **short spread**:
  SELL y, BUY β·x lots
- z < −2 → spread undershoot вниз → **long spread**:
  BUY y, SELL β·x lots

Лот-сайзинг (`compute_lots`) равно-нотиональный с β-хеджированием:
`lots_y × notional_y_per_lot ≈ lots_x × notional_x_per_lot / β`.

## БД

Отдельная `data/pairs.db`:
- **pair_trades** — каждая закрытая сделка с обеими ногами, β, z@entry/exit
- **pair_state** — кэш текущих β/α/spread_stats для каждой пары

Можно смотреть так:
```sql
sqlite3 data/pairs.db "
SELECT pair, direction, lots_y, lots_x,
       entry_z, exit_z, exit_reason,
       ROUND(pnl_rub, 2) AS pnl_rub,
       ROUND((julianday(exit_time)-julianday(entry_time))*24, 1) AS held_h
FROM pair_trades WHERE exit_time IS NOT NULL
ORDER BY exit_time DESC LIMIT 20;
"
```

## Reconcile на старте

Если бот рестартил с открытой парой — она будет:
- **re-hydrated** в активную обработку (если age ≤ 4× max_hold = 192h)
- **force-closed** по last_price если старше (предотвращает leaked positions)

## Live execution safety

При live-запуске две ноги отправляются последовательно:
1. Если **leg-Y filled, leg-X failed** → бот немедленно делает emergency
   unwind leg-Y (полу-хеджированная позиция = худший возможный исход).
2. Если **обе failed** на exit — лог CRITICAL, нужна ручная разборка.

## Реалистичные ожидания

| | Что обещают 180-day grid | Что реально может быть |
|---|---|---|
| Annualised return per pair | +15-17 % | +5-10 % (после real slippage + drift) |
| Sharpe | 1.3-6 | 0.7-1.5 (multi-comparison bias adjusted) |
| Win rate | 75-100 % | 60-70 % |
| Trades/month per pair | 1-3 | 1-2 |
| Max drawdown | 0-5 % за 180 дней | 5-15 % на годовой выборке |

**Не** обещаю что цифры из backtest повторятся live. Sample 6-17 сделок
на пару — мало для уверенности. Paper-режим месяц — обязательный шаг.

## Что НЕ реализовано (пока)

- **Limit-aggressive ордера** — сейчас market на обеих ногах. На Si это
  упадёт (limit-only). Si стоит торговать только в paper до перевода на
  limit.
- **Walk-forward β** — β считается на полной 180d истории. Это
  in-sample. Реалистичнее переходить на rolling β.
- **News blackouts** — не учитываем дни CBR rate decisions и
  крупных гео-событий.
- **Portfolio-level VaR** — каждая пара управляется независимо. На
  макро-шоках все 4 могут просесть одновременно (на Si все три "Si"-пары
  скоррелированы).
