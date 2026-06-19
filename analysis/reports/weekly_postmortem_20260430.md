# Weekly Postmortem — last 7 days
_Generated 2026-04-30T22:44:49 | window: since 2026-04-23 | trades: 44_

## 1. Overview

- closed trades: **44**
- mean P&L%: **+0.136%**, median: +0.610%
- sum P&L: **+29.32 ₽**
- win rate: **52.3%** (23/44)
- avg win: 1.310% | avg loss: 1.150% | R/R: 1.14 | Kelly f*: **+0.104**

## 2. Long / Short asymmetry

| direction   |   n |   mean_pct |   median_pct |   sum_rub |   hit |
|:------------|----:|-----------:|-------------:|----------:|------:|
| buy         |  17 |     -0.672 |        -1.46 |    -46.02 | 0.353 |
| sell        |  27 |      0.644 |         0.89 |     75.34 | 0.63  |

Per-direction Kelly (using only same-direction trades):

- **sell**: n=27, win_rate=63.0%, avg_win=1.162%, avg_loss=0.236%, R/R=4.92, Kelly f*= **+0.554**
- **buy**: n=17, win_rate=35.3%, avg_win=1.728%, avg_loss=1.981%, R/R=0.87, Kelly f*= **-0.389**

## 3. Exit-reason histogram

| exit_reason     |   n |   mean_pct |   median_pct |   sum_rub |   hit |   hold_h |
|:----------------|----:|-----------:|-------------:|----------:|------:|---------:|
| external_close  |   7 |      0.281 |        0.93  |     18.12 | 0.571 |   10.304 |
| signal_reversal |  27 |      0.591 |        0.74  |     57.79 | 0.63  |    5.741 |
| stop_loss       |   8 |     -2.116 |       -2.015 |    -61.08 | 0     |   18.053 |
| take_profit     |   1 |      3.43  |        3.43  |      9.07 | 1     |    1.994 |
| trailing_stop   |   1 |      1.53  |        1.53  |      5.42 | 1     |   18.05  |

**Read:** signal_reversal hit-rate vs external_close hit-rate gap is the main asymmetry.

## 4. Confidence calibration

- Spearman(conf, pnl_pct) = **+0.3409** (p=0.024)
- Pearson(conf, pnl_pct) = +0.3641

Confidence quartiles:

| q   |   conf_min |   conf_max |   n |   mean_pct |   hit |
|:----|-----------:|-----------:|----:|-----------:|------:|
| q1  |      0.504 |      0.678 |  11 |     -0.724 | 0.364 |
| q2  |      0.685 |      0.714 |  11 |     -0.149 | 0.455 |
| q3  |      0.716 |      0.813 |  11 |      0.734 | 0.727 |
| q4  |      0.816 |      0.977 |  11 |      0.682 | 0.545 |

Reliability (binned win-rate vs confidence):

| cb          |   n |   mean_conf |   win_rate |   mean_pnl |    gap |
|:------------|----:|------------:|-----------:|-----------:|-------:|
| [0.5, 0.6)  |   1 |       0.504 |      0     |     -1.9   | -0.504 |
| [0.6, 0.65) |   5 |       0.627 |      0.4   |     -0.122 | -0.227 |
| [0.65, 0.7) |   7 |       0.673 |      0.286 |     -1.307 | -0.387 |
| [0.7, 0.75) |  14 |       0.717 |      0.571 |      0.345 | -0.146 |
| [0.75, 0.8) |   5 |       0.772 |      0.8   |      0.874 |  0.028 |
| [0.8, 0.85) |   3 |       0.82  |      0.667 |      0.99  | -0.153 |
| [0.85, 0.9) |   4 |       0.87  |      0.5   |      0.415 | -0.37  |
| [0.9, 1.01) |   5 |       0.929 |      0.6   |      0.76  | -0.329 |

Expected Calibration Error (ECE) = **0.230**  (lower = better; >0.10 = poorly calibrated)

## 5. MAE / MFE pathology

MAE/MFE quantiles (% of entry, where MAE is signed adverse, MFE is signed favorable):

|       |   mae_pct |   mfe_pct |   sl_dist_pct |   tp_dist_pct |
|:------|----------:|----------:|--------------:|--------------:|
| count |    44     |    44     |        44     |        44     |
| mean  |    -0.988 |     1.145 |         1.93  |         2.858 |
| std   |     1.052 |     1.189 |         0.703 |         1.077 |
| min   |    -5.121 |    -0.404 |         1.01  |         1.49  |
| 25%   |    -1.414 |     0.176 |         1.404 |         2.128 |
| 50%   |    -0.562 |     0.922 |         1.776 |         2.54  |
| 75%   |    -0.275 |     1.652 |         2.199 |         3.4   |
| max   |    -0.027 |     4.79  |         4.644 |         6.967 |

### 5.1 Stops touched intraday but exit_reason ≠ stop_loss
- n = **2** of 44 trades (4.5%)
- mean realized P&L%: -1.925%  vs  others +0.234%
- by exit_reason: {'external_close': 2}

**Hypothesis:** broker stop fired late (>=0.2% past level) OR position monitor checked stale hourly close while 5-min low pierced the level.

### 5.2 TP touched intraday but exit_reason ≠ take_profit
- n = **2** of 44 trades (4.5%)
- mean realized P&L%: +1.510%  (left on table = mean MFE−exit_pnl)
- mean give-back from MFE → exit: **+0.926 pp**

### 5.3 Did losers ever show profit?
- losers n=21, mean MFE = **+0.356%** (max favorable)
- winners n=23, mean MAE = **-0.486%** (max adverse)
- losers that touched ≥+0.5% MFE before turning: **7** (33.3%)
  → these would have been winners if a partial-TP at 0.5% had been in place.

## 6. Time-of-day (MSK)

By hour of entry (MSK):

|   hour_msk |   n |   mean_pct |   hit |   sum_rub |
|-----------:|----:|-----------:|------:|----------:|
|          9 |   5 |      0.282 | 0.6   |     21.65 |
|         10 |   5 |     -0.856 | 0.4   |    -13.58 |
|         11 |   6 |      0.323 | 0.5   |     12.46 |
|         13 |   5 |      0.534 | 0.4   |      7.04 |
|         14 |   6 |      0.153 | 0.667 |      8.46 |
|         15 |   7 |      0.816 | 0.714 |     20.1  |
|         16 |   2 |      0.77  | 1     |      7.56 |
|         17 |   3 |     -0.26  | 0.333 |    -12.5  |
|         18 |   3 |     -0.447 | 0     |    -16.03 |
|         19 |   1 |      1.3   | 1     |      2.83 |
|         23 |   1 |     -3.12  | 0     |     -8.67 |

**Worst 4 hours (n≥3):** [10, 18, 17, 14] | means: [-0.856, -0.447, -0.26, 0.153]

By day of week:

| dow_msk   |   n |   mean_pct |
|:----------|----:|-----------:|
| Monday    |  22 |      0.39  |
| Tuesday   |  14 |     -0.196 |
| Wednesday |   8 |      0.015 |

## 7. Stop / TP distance vs outcome

Stop-distance (in ATR proxy) summary:

|       |   sl_in_atr |   tp_in_atr |
|:------|------------:|------------:|
| count |       44    |       44    |
| mean  |        0.71 |        1.03 |
| std   |        0.28 |        0.39 |
| min   |        0.27 |        0.4  |
| 25%   |        0.48 |        0.71 |
| 50%   |        0.66 |        0.99 |
| 75%   |        0.91 |        1.33 |
| max   |        1.39 |        2    |

Win rate by stop-distance (in ATR proxy):

| sl_bin   |   n |   mean_pct |   hit |
|:---------|----:|-----------:|------:|
| <1       |  39 |      0.031 | 0.487 |
| 1-1.5    |   5 |      0.952 | 0.8   |

## 8. Per-ticker breakdown

Tickers with ≥3 trades:

| ticker   |   n |   mean_pct |   sum_rub |   hit |
|:---------|----:|-----------:|----------:|------:|
| ASTR     |   3 |      0.993 |      7.11 | 0.667 |
| TGKN     |   3 |      0.723 |     10.37 | 0.667 |

Top 5 worst tickers by sum P&L:

| ticker   |   n |   mean_pct |   sum_rub |   hit |
|:---------|----:|-----------:|----------:|------:|
| GTRK     |   2 |     -1.85  |    -29.5  |     0 |
| CBOM     |   1 |     -2.28  |    -15.75 |     0 |
| WUSH     |   2 |     -1.775 |     -9.03 |     0 |
| MVID     |   1 |     -3.12  |     -8.67 |     0 |
| MDMG     |   1 |     -0.54  |     -8.54 |     0 |

Top 5 best tickers by sum P&L:

| ticker   |   n |   mean_pct |   sum_rub |   hit |
|:---------|----:|-----------:|----------:|------:|
| MRKZ     |   2 |      0.565 |     11.37 |   0.5 |
| SMLT     |   2 |      1.075 |     11.61 |   1   |
| TTLK     |   1 |      1.81  |     11.8  |   1   |
| MSNG     |   1 |      0.74  |     12.54 |   1   |
| MRKC     |   2 |      1.005 |     16.4  |   1   |

## 9. Top-10 worst trades — detail

|   id | ticker   | direction   |   signal_confidence |   entry_price |   exit_price |   stop_loss |   take_profit |    pnl |   pnl_pct | exit_reason     |   hold_hours |   mae_pct |   mfe_pct | stop_touched_intraday   | tp_touched_intraday   |   min_to_mae |
|-----:|:---------|:------------|--------------------:|--------------:|-------------:|------------:|--------------:|-------:|----------:|:----------------|-------------:|----------:|----------:|:------------------------|:----------------------|-------------:|
|  116 | GTRK     | buy         |               0.696 |        75.2   |       73.6   |      73.6   |        77.4   | -16.74 |     -2.13 | stop_loss       |        3.258 |    -2.394 |     0     | True                    | False                 |        192.1 |
|  127 | CBOM     | buy         |               0.901 |         6.62  |        6.469 |       6.453 |         6.87  | -15.75 |     -2.28 | external_close  |        1.739 |    -2.81  |     1.511 | True                    | False                 |        100.3 |
|  104 | MRKZ     | buy         |               0.604 |         0.133 |        0.132 |       0.13  |         0.138 | -14.83 |     -1.01 | signal_reversal |        0.375 |    -1.012 |     0.075 | False                   | False                 |         21   |
|  126 | GTRK     | buy         |               0.685 |        76.4   |       75.2   |      74.7   |        79     | -12.76 |     -1.57 | external_close  |        2.325 |    -2.356 |    -0.262 | True                    | False                 |        135.5 |
|  108 | MVID     | buy         |               0.657 |        67.4   |       65.3   |      65.4   |        70.35  |  -8.67 |     -3.12 | stop_loss       |       10.547 |    -3.338 |     0.074 | True                    | False                 |        630.5 |
|  122 | MDMG     | sell        |               0.707 |      1336.6   |     1343.8   |    1355.2   |      1308.7   |  -8.54 |     -0.54 | signal_reversal |       13.805 |    -1.055 |    -0.09  | False                   | False                 |         54.2 |
|   88 | BTBR     | buy         |               0.678 |       132.59  |      128.5   |     129.26  |       137.61  |  -8.44 |     -3.08 | stop_loss       |       52.76  |    -5.121 |     0.875 | True                    | False                 |       3156.2 |
|  121 | SNGSP    | buy         |               0.67  |        43.125 |       42.395 |      42.49  |        44.065 |  -7.73 |     -1.69 | stop_loss       |       22.177 |    -1.67  |     0.557 | True                    | False                 |       1329.3 |
|  114 | POSI     | sell        |               0.949 |      1020     |     1025     |    1038.6   |       992.2   |  -6.02 |     -0.49 | signal_reversal |       23.519 |    -0.588 |     1.471 | False                   | False                 |       1406.1 |
|  123 | ASTR     | buy         |               0.504 |       287.5   |      282.05  |     281.5   |       295.95  |  -5.73 |     -1.9  | stop_loss       |        3.999 |    -2.365 |     0.313 | True                    | False                 |        235.7 |
