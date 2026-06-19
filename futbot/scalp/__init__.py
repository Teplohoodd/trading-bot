"""futbot.scalp — high-frequency scalping subsystem.

Architectural separation from the 4-layer pipeline:
  * Different alpha source: order book microstructure + 1-min indicators,
    not multi-timeframe trend confirmation.
  * Different cadence: streaming subscriptions, event-driven (no polling loop).
  * Different exit logic: quick TP/SL within minutes, time-cap in seconds-minutes.
  * Different sizing: small fixed lots, many entries.
  * Separate DB table (`scalp_trades`) and log file.

Reuses: broker, ГО metadata extraction, telegram notifier, .env config loader.

Entry point: `python -m futbot.scalp.main`
"""
