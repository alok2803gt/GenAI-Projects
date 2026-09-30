#!/bin/bash
# PRIMARY harami signal source since 2026-09-23 (IBKR daily bars).
# Writes the harami_scanner_alert entries harami_daily_trader.py --mode entry reads.
cd "$(dirname "$0")"
exec ../venv/bin/python -u candlestick_pattern_research/harami_scanner_ibkr.py --primary
