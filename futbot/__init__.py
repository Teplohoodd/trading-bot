"""futbot — dedicated futures-only trading agent.

Sibling package to `trade_claude`. Imports broker / db / telegram / macro
from the parent package directly (no copy-paste of infrastructure) and adds
its own decision pipeline, sizer, risk audit, and DB schema for futures.

Entry point: `python -m futbot.main`
"""
