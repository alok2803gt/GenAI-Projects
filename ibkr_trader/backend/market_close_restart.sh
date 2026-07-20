#!/bin/bash
LOGF="/Users/aanydubey/SUMMER BRIDGE ACADEMY/GenAI-Projects/ibkr_trader/backend/market_close_restart.log"
UID_N=$(id -u)

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') Market-close restart firing" >> "$LOGF"

launchctl kickstart -k gui/$UID_N/com.ibkrtrader.backend  >> "$LOGF" 2>&1
launchctl kickstart -k gui/$UID_N/com.ibkrtrader.scanner  >> "$LOGF" 2>&1
launchctl kickstart -k gui/$UID_N/com.ibkrtrader.frontend >> "$LOGF" 2>&1

echo "$(date '+%Y-%m-%d %H:%M:%S %Z') Restart commands issued; removing one-shot job" >> "$LOGF"

# Detach cleanup so bootout doesn't kill this script before it exits.
( sleep 2
  launchctl bootout gui/$UID_N/com.ibkrtrader.marketcloserestart >/dev/null 2>&1
  rm -f "/Users/aanydubey/Library/LaunchAgents/com.ibkrtrader.marketcloserestart.plist"
) &
disown
exit 0
