#!/bin/bash
# One-shot: restart the two PERSISTENT traders so they load the commission-ledger
# wiring added 2026-09-28, then delete itself.
#
# Why this is needed at all: com.ibkrtrader.ashley and com.ibkrtrader.daytrader
# both run KeepAlive=1 and are NOT covered by market_close_restart.sh (which
# only kickstarts backend/scanner/frontend, and is itself a self-deleting
# one-shot). Long-lived processes therefore keep running whatever code they
# started with, so an edit to either file never takes effect until something
# restarts them.
#
# Why not immediately: day_trader_agent.main() calls connectAsync as its first
# act, so restarting it while TWS is logged out (weekends, nightly restart)
# would crash-loop under KeepAlive. This fires at 08:45 ET Monday, when TWS is
# up and before the 09:30 execution window.
#
# Both traders are restart-safe by design: the day trader rehydrates working
# orders via reqAllOpenOrdersAsync, and Ashley reconciles against real IBKR
# positions plus its persisted fired/setups state (verified live 2026-09-25).
LOGF="/Users/aanydubey/SUMMER BRIDGE ACADEMY/GenAI-Projects/ibkr_trader/backend/market_close_restart.log"
UID_N=$(id -u)

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') commission-wiring restart firing" >> "$LOGF"

if ! nc -z -w3 127.0.0.1 7496; then
  echo "$(date '+%Y-%m-%d %H:%M:%S %Z') TWS API port closed -- NOT restarting, leaving job in place to retry" >> "$LOGF"
  exit 0          # keep the job loaded; it will fire again tomorrow
fi

launchctl kickstart -k gui/$UID_N/com.ibkrtrader.ashley    >> "$LOGF" 2>&1
launchctl kickstart -k gui/$UID_N/com.ibkrtrader.daytrader >> "$LOGF" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') ashley + daytrader restarted for commission wiring; removing one-shot" >> "$LOGF"

# Detach cleanup so bootout doesn't kill this script before it exits.
( sleep 2
  launchctl bootout gui/$UID_N/com.ibkrtrader.commissionwiringrestart >/dev/null 2>&1
  rm -f "/Users/aanydubey/Library/LaunchAgents/com.ibkrtrader.commissionwiringrestart.plist"
) &
disown
exit 0
