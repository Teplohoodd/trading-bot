"""futbot.orchestrator — unified bot runner with single Telegram session.

Entry point: python -m futbot.orchestrator.main

Runs pairs + trend (and optionally scalp) under one process so that:
  * Telegram commands work (only one process can poll a token)
  * Each strategy reports through a single notifier queue
  * Boot/shutdown is coordinated
  * Shared broker connection reduces API setup overhead

Strategies remain in separate DB files and have their own configs.
Standalone runners (pairs/main.py, trend/main.py) still work for testing
and debug — they just can't run while the orchestrator does.
"""
