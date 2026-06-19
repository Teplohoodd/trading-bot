"""futbot.trend — multi-contract Bollinger breakout (swing trend-following).

Entry point: python -m futbot.trend.main

Built from walk-forward validation results (2026-05-20):
  * 171 FORTS contracts scanned
  * 68 had positive in-sample edge
  * 26 survived 90-day out-of-sample validation
  * Avg IS Sharpe +0.83 → OOS Sharpe +0.66 (80% retention = real edge)

Strategy: classic OsEngine-style Bollinger band breakout.
  Long  on close > MA(N) + k·σ
  Short on close < MA(N) − k·σ
  Exit  on close crossing the opposite band
  NO stop-loss (mechanical band-flip only)
  NO take-profit
  Auto-flatten 3 days before contract expiry

Per-contract parameters cherry-picked from the WF survivors —
see `portfolio.py` for the table.

Paper mode by default.  Set TREND_PAPER_MODE=false in .env for live.
"""
