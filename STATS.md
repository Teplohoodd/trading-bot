# Trading Statistics -- 6-20 June 2026

> Period covers the first two weeks of live trading.
> All trades prior to June 2026 were parameter tuning and system validation.

## Summary

| Strategy | Trades | Win Rate | P&L | Kelly f* |
|----------|--------|----------|-----|----------|
| Breakdown | 8 | 75% | **+4878 RUB** | +0.628 |
| Trend (patterns) | 5 | 60% | **-1037 RUB** | -0.426 |
| **Total** | **13** | **69%** | **+3841 RUB** | **+0.365** |

---

## Breakdown strategy

Shorts the September futures contract when the underlying stock breaks down.
Entry triggers on a 5%+ intraday move with elevated volume. R/R target 2.5x+.

| Stock | Future | Opened | Closed | Entry | Exit | P&L (RUB) | Exit reason | Hold |
|-------|--------|--------|--------|-------|------|-----------|-------------|------|
| IVAT | IVU6 | 12 Jun | 15 Jun | 7941 | 8292 | **-351** | `timeout` | 61h |
| TATN | TTU6 | 15 Jun | 16 Jun | 58305 | 55816 | **+2489** | `target` | 24h |
| POSI | PSU6 | 16 Jun | 18 Jun | 862 | 845 | **+17** | `timeout` | 48h |
| ROSN | RNU6 | 16 Jun | 18 Jun | 35734 | 33857 | **+1877** | `target` | 43h |
| SBER | SRU6 | 16 Jun | 18 Jun | 29766 | 29188 | **+578** | `timeout` | 48h |
| IVAT | IVU6 | 18 Jun | 19 Jun | 7543 | 7044 | **+499** | `target(closed_externally)` | 35h |
| GAZP | GZU6 | 18 Jun | 19 Jun | 11187 | 10825 | **+362** | `target` | 35h |
| LKOH | LKU6 | 18 Jun | 19 Jun | 45610 | 46203 | **-593** | `stop(closed_externally)` | 17h |

**Avg win:** +970 RUB | **Avg loss:** -472 RUB | **R/R:** 2.06x | **Kelly:** +0.628

---

## Trend strategy

Trades classic chart patterns (triple top/bottom) on stocks and perpetual contracts.

| Ticker | Dir | Opened | Closed | Entry | Exit | P&L (RUB) | P&L% | Exit reason | Hold | Pattern |
|--------|-----|--------|--------|-------|------|-----------|------|-------------|------|---------|
| LTU6 | buy | 11 Jun | 14 Jun | 1733.80 | 1754.60 | **+1395** | +1.20% | `pattern_timeout` | 61h | triple_bottom |
| NBISperpA | sell | 16 Jun | 16 Jun | 253.80 | 268.11 | **-1036** | -5.64% | `pattern_stop` | 1h | triple_top |
| HOODperpA | sell | 17 Jun | 17 Jun | 96.78 | 103.44 | **-1462** | -6.88% | `pattern_stop(stop_filled)` | 6h | triple_top |
| APPperpA | sell | 17 Jun | 18 Jun | 493.16 | 469.58 | **+23** | +4.78% | `reconciled_gone_at_broker` | 25h | triple_top |
| PXU6 | sell | 17 Jun | 19 Jun | 20630.00 | 20592.00 | **+43** | +0.18% | `pattern_timeout` | 48h | triple_top |

**Avg win:** +487 RUB | **Avg loss:** -1249 RUB | **R/R:** 0.39x | **Kelly:** -0.426

---

## Open positions (as of 2026-06-20)

| Stock | Future | Opened (UTC) | Entry | Stop | Target |
|-------|--------|--------------|-------|------|--------|
| SBER | SRU6 | 19 Jun 13:35 | 29127 | 29426 | 28225 |

---

## Observations

**Breakdown** is the only strategy with a statistically positive Kelly (+0.63).
The R/R structure (stop ~1.5%, target ~4%) provides resilience: one stop-out is recovered by a single target hit.

**Trend** is structurally broken in its current configuration. Two perpetual-contract trades
(NBISperpA, HOODperpA) each lost >5% in under 6 hours, wiping out gains from the other three combined.
Stop placements are too tight for perpetuals with high overnight volatility.

Next steps:
- Scale Breakdown incrementally (currently 1 lot per signal).
- Pause Trend on perpetuals until stop-distance is recalibrated.
- Reactivate Carry only when 3+ uncorrelated pairs are validated.
