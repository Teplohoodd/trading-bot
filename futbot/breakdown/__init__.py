"""Breakdown strategy — volume-confirmed range-breakdown shorts on MOEX stocks,
executed via the corresponding FORTS single-stock future (stocks like IVAT
can't be shorted directly; their futures can).

Hypothesis (validated 2026-06-10 on 27 stocks × 90d, see
futbot/scripts/breakdown_study.py):
  after a quiet consolidation, a close below the prior N-bar low on abnormal
  volume with a severe bar (≤ -1%) CONTINUES down for hours/days
  (stop cascades + margin calls — the IVAT 2026-05-29 / 2026-06-08 signature).

Backtest (2h bars, N=12, vol≥2×, sev≤-1%, RR 3:1, timeout 24 bars):
  281 trades, 48% win, +0.49%/trade, +138% sum, 4/4 months positive.
Caveat: 90-day falling market (-22.9%) — edge is partly beta; live pilot
starts in PAPER mode and must prove itself forward before real money.
"""
