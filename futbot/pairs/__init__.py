"""futbot.pairs — cointegration-based pair trading on MOEX FORTS.

Entry point:  python -m futbot.pairs.main

Sibling to `futbot.scalp` (microstructure scalping) and the 4-layer
`futbot` main bot.  Shares broker/telegram/commission infrastructure
but operates on a DIFFERENT timeframe (hourly evaluation, multi-day
holds) and DIFFERENT alpha source (statistical-arbitrage spread
mean-reversion between cointegrated futures pairs).

Parameters chosen from the 180-day grid search (see
`scripts/pairs_grid_search.py`):
  * z_entry = 2.0
  * max_hold = 48 h (2 trading days)
  * z_stop = 4.0 (structural break)
  * rolling z-score window = 240 h (10 days)
  * Pairs: LK-Si, SR-Si, GZ-Si, SR-MX (all adf_p ≤ 0.10 over 180d)

Paper mode is the default — set PAIRS_PAPER_MODE=false in .env to go live.
"""
