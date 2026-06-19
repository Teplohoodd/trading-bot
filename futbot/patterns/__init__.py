"""futbot.patterns — chart-pattern detection + backtest.

Replaces the failed Bollinger trend logic with classic price patterns:
Double Top/Bottom, Triple Top/Bottom, Head & Shoulders (+ inverse),
Rectangle breakout.

All detectors are PROGRAMMATIC — no subjective "looks like" tolerance.
Each pattern has strict numeric criteria + measured-move target +
pattern-invalidation stop.  Backtest first, deploy only if edge survives
commission on out-of-sample data.
"""
