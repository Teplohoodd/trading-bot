# Trade Postmortem Fix Proposal
**Window:** 2026-04-18 → 2026-04-25 | **Trades:** 61 closed shares | **Mean P&L:** -0.277%

---

## Executive Summary

Five systematic defects found, ranked by estimated weekly P&L impact:

1. 🔴 **CRITICAL — Hard stop/TP never fires** — Engine has no software fallback for stop_loss/take_profit prices; relies solely on broker-side stop orders that can fail or be lost on restart. 10/23 `external_close` trades exited PAST their stop level. Manual closes (mean -1.13%) vs ML-reversal exits (mean +0.24%) — 1.37 pp gap × 23 trades = **~31% of weekly losses** from this defect.

2. 🔴 **CONFIRMED — Negative realized Kelly (f\* = −0.23)** — Strategy has no statistical edge at current signal threshold. `KELLY_FRACTION = 0.5` is sizing into a losing system. Every RUB committed amplifies losses.

3. 🟡 **CONFIRMED — High-confidence signals are the WORST performers** — Q4 (confidence 0.83–0.92) has mean P&L = −0.55%, while Q1 (0.60–0.68) = −0.16%. Spearman(confidence, pnl) = −0.08. The model's confidence ranking is inverted. `SIGNAL_THRESHOLD = 0.65` is too low AND signals above 0.80 should be treated with suspicion.

4. 🟡 **CONFIRMED — Early-morning entries destroy P&L** — Entries at 7–9 MSK average −0.50 to −0.81% (20 trades = 33% of volume). Entries at 10 MSK, 19–23 MSK average +0.21 to +0.99%. Structural: pre-market candles are gapped/illiquid, ATR unreliable.

5. 🟢 **REJECTED — Slippage** — Zero slippage (limit orders, confirmed). No action needed.

---

## Defect Detail

### Defect 1: Hard Stop/TP Missing from Position Monitor (CRITICAL)

**Evidence:** `02_exit_pathology.yaml` — 0/61 exits via `stop_loss`, `take_profit`, `trailing_stop`, `time_exit`. 10/23 `external_close` trades exited at prices past their `stop_loss` column.

**Root cause:** `_check_exit_conditions()` in `core/engine.py:834` checks:
- trailing stop (only activates at 70% progress — rarely reached)  
- time_exit (after 5 days)  
- ML signal reversal  
- **Missing: explicit `current_price <= stop_loss` / `current_price >= take_profit` check**

Broker-side stop orders (lines 646–654) are placed but can be lost on bot restart or network error. No software fallback exists.

| Parameter | Current | Proposed | File:Line |
|---|---|---|---|
| (code fix) | missing check | add SL/TP guard | `core/engine.py:959` |

**Go/no-go gate:** Apply unconditionally — this is a bug fix, not a parameter tuning.

**Expected impact:** Eliminate 10 past-stop rides. At avg -1.1% vs stop target ~-0.8%, saves ~0.3% × 10 trades/week = +3 pp weekly.

---

### Defect 2: KELLY_FRACTION Too High for Current Edge

**Evidence:** `04_sizing.yaml` — win_rate=39.3%, b=0.971, f\*= −0.231.

Realized Kelly f\* is **negative** — the strategy loses more on losers than it wins on winners, and wins less than 50% of the time. `KELLY_FRACTION = 0.5` actively amplifies this.

| Parameter | Current | Proposed | File:Line |
|---|---|---|---|
| `KELLY_FRACTION` | `0.5` | `0.15` | `config/settings.py:40` |
| `KELLY_FALLBACK_PCT` | `0.05` | `0.03` | `config/settings.py:46` |

**Go/no-go gate:** Apply when `realized_kelly_f_star < 0` (confirmed). Restore to 0.25 once win_rate ≥ 45%.

**Expected impact:** Reduce position sizes by 70%, capping downside while strategy is recalibrated.

---

### Defect 3: Signal Threshold Too Low / Model Confidence Inverted

**Evidence:** `01_entry_quality.yaml` — Q4 confidence quartile (0.83–0.92) mean pnl = −0.55% vs Q1 (0.60–0.68) = −0.16%. Spearman = −0.08 (negative, non-significant but directionally wrong).

The model's probability estimates do not rank outcomes correctly. Two responses:  
(a) Raise threshold to cut the worst signals  
(b) Trigger model retraining (the isotonic calibrator needs more data)

| Parameter | Current | Proposed | File:Line |
|---|---|---|---|
| `SIGNAL_THRESHOLD` | `0.65` | `0.72` | `config/settings.py:62` |
| `SIGNAL_THRESHOLD_FUTURES` | `0.68` | `0.75` | `config/settings.py:180` |

**Go/no-go gate:** Apply only after confirming Q1/Q4 pnl pattern persists in next 50 trades. Also trigger `/retrain` to refresh calibration.

**Expected impact:** Cut ~30% of signals (Q3-Q4 band), retaining cleaner Q1-Q2 signals. If Q1-Q2 have edge, net win_rate improves.

---

### Defect 4: Early Morning Entry Filter

**Evidence:** `01_entry_quality.yaml` hour analysis.

| Hour (MSK) | Trades | Mean P&L |
|---|---|---|
| 7 | 14 | −0.46% |
| 8 | 4 | −0.81% |
| 9 | 6 | −0.45% |
| 10 | 5 | +0.34% |
| 11 | 3 | −0.68% |
| 12 | 3 | −1.08% |
| 19 | 6 | +0.75% |
| 21 | 2 | +0.66% |

Worst 4 hours (7, 8, 9, 12 MSK) = 27 trades, mean −0.59%. Best 4 hours (10, 19, 21, 23 MSK) = 14 trades, mean +0.56%.

| Parameter | Current | Proposed | File:Line |
|---|---|---|---|
| `SKIP_ENTRY_HOURS_MSK` (new) | `[]` | `[7, 8, 9, 11, 12]` | `config/settings.py` (add) |

**Go/no-go gate:** Apply with ≥100 trades confirmation. For now, add the setting and enforce it in engine.py `_evaluate_instrument`.

**Expected impact:** Block 27 bad-hour trades/week, redirect to better hours. Estimated +0.59% - (-0.56%) = 1.15 pp swing on redirected trades.

---

## Parameter Diff Block

Paste into `config/settings.py`:

```python
# === FIX 2026-04-25: Kelly reduction (negative f*=-0.23) ===
KELLY_FRACTION: float = 0.15           # was 0.5; restore to 0.25 when win_rate >= 45%
KELLY_FALLBACK_PCT: float = 0.03       # was 0.05

# === FIX 2026-04-25: Threshold raise (confidence quartile inversion) ===
SIGNAL_THRESHOLD: float = 0.72         # was 0.65
SIGNAL_THRESHOLD_FUTURES: float = 0.75 # was 0.68

# === FIX 2026-04-25: Hour filter (pre-market bad performance) ===
SKIP_ENTRY_HOURS_MSK: list = [7, 8, 9, 11, 12]  # new field
```

---

## Rejected Hypotheses

| Hypothesis | Reason Rejected |
|---|---|
| Slippage from market orders | REJECTED: bot uses limit orders, 0 bps slippage (log confirmed) |
| Whipsaw rapid re-entries | REJECTED: 0 whipsaws detected, cooldown filter not binding |
| Sector concentration | N/A: candle cache empty, correlation matrix unavailable |
| Futures-specific issues | N/A: 0 futures trades in window (all shares) |

---

## Implementation Order

1. **Now:** Fix `_check_exit_conditions` to add hard SL/TP check (bug fix, no downside)
2. **Now:** Reduce `KELLY_FRACTION` to 0.15 (caps loss during strategy recalibration)  
3. **Now:** Raise `SIGNAL_THRESHOLD` to 0.72
4. **Now:** Add `SKIP_ENTRY_HOURS_MSK` and enforce in engine
5. **After 50 more trades:** Review confidence quartile pattern; adjust threshold direction
6. **Run `/retrain`:** Fresh calibration with accumulated data
