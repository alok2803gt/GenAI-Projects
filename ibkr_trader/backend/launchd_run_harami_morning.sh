#!/bin/bash
cd "$(dirname "$0")"
exec ../venv/bin/python -u harami_daily_trader.py --mode morning
