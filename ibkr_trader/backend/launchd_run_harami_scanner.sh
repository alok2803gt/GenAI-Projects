#!/bin/bash
# CROSS-CHECK only since 2026-09-23: yfinance alerts to Telegram for comparison
# but writes NO harami_scanner_alert entries (IBKR is the live source).
cd "$(dirname "$0")"
exec ../venv/bin/python -u candlestick_pattern_research/harami_scanner.py --shadow
