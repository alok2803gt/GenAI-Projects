#!/bin/bash
# Housekeeping agent -- keeps the books agreeing with the broker.
#
# Runs with --fix, which applies ONLY the SAFE_FIX class: repairs where IBKR (or
# its own execution reports in the commission ledger) is the authority and the
# disagreement can have exactly one meaning. It never places, modifies or
# cancels an order, and never marks a position closed on its own judgement --
# over-eager auto-closing is what corrupted the journal on 2026-09-29, so
# anything ambiguous is FLAGGED to Telegram and left alone.
#
# Every write is preceded by a timestamped copy in housekeeping_backups/.
#
# Schedule: 08:45 ET (before the 09:15 accumulation and the 09:30 open, so the
# books are right before anything trades), 12:30 ET, and 16:30 ET (after the
# close and after the 16:25 commission-ledger snapshot). A TWS outage is a
# clean no-op, not an alert.
cd "$(dirname "$0")"
exec ../venv/bin/python -u housekeeping_agent.py --fix
