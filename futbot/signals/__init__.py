"""Telegram signal pipeline — interpret a discretionary trader's crypto calls
into structured, actionable Neo-asset trade proposals.

Runtime = Claude-in-the-loop: a scheduled/looped session reads the channel via
telegram-mcp, interprets each new message (the heuristic here does a first
pass + flags cryptic ones for LLM/human judgement), checks margin via t-invest,
and proposes a Neo trade for confirmation.  Nothing auto-fires.
"""
