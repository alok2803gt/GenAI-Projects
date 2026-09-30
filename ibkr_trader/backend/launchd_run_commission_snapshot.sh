#!/bin/bash
# Per-execution commission snapshot (intended: 16:25 and 17:30 ET weekdays).
#
# READ-ONLY with respect to the broker: it calls reqExecutions and writes the
# results to the local ledger. It places, modifies and cancels nothing.
#
# Why it is scheduled independently of the traders: IBKR's execution API serves
# the CURRENT TRADING DAY ONLY (verified 2026-09-27 -- back-dated filters all
# returned 0 fills), so a session that is never snapshotted can never be
# recovered. The traders each snapshot on connect and at shutdown, but a trader
# that crashes, or a fill placed manually in TWS, would otherwise be missed.
#
# Two runs: 16:25 catches the close, 17:30 catches commissions IBKR reports late.
cd "$(dirname "$0")"
exec ../venv/bin/python -u commission_ledger.py --snapshot
