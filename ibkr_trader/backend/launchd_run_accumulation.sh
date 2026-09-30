#!/bin/bash
# Valuation-gated accumulation for Inside Day Reversal names (intended: 09:15 ET weekdays).
#
# Screens every IDR holding on SEC-filed fundamentals and reports the next due
# tranche for names that qualify. Guards inside accumulation.py: max 3 names,
# 10% target weight split into 3 tranches, adds only at -5%/-10% below the first
# tranche, 5 trading days minimum spacing, and any tranche costing more than
# available cash is skipped.
#
# Retimed 15:30 -> 09:15 ET on 2026-09-29: at 15:30 an Ashley 0DTE option is
# often still open (she force-closes at 15:55), and a near-the-money 0DTE makes
# IBKR project assignment margin -- which rejected the TFC tranche with
# Error 201 "PROJECTED POST EXPIRATION MARGIN DEFICIT IS 18070.61 USD". At 09:15
# yesterday's 0DTE has expired and today's is not yet bought (Ashley executes
# from 09:30), and a tif="OPG" order placed pre-open fills at that day's open.
#
# LIVE (CEO instruction 2026-09-25): places the next due tranche each run.
# To make it report-only again, drop --execute below.
cd "$(dirname "$0")"
exec ../venv/bin/python -u accumulation.py --execute
