#!/bin/bash
# SPY 0DTE iron condor V2.
# Quotes: IBKR TWS (real NBBO; Alpaca's free indicative option feed is ~5x too wide).
# Orders: Alpaca PAPER account. Switch to --broker ibkr --mode live only after
# the paper phase passes (see spy_0dte/TRADE_PLAN_condor.md).
cd "$(dirname "$0")"
exec ../venv/bin/python -u condor_v2_trader.py --ticker spy --mode paper --broker ibkr "$@"
