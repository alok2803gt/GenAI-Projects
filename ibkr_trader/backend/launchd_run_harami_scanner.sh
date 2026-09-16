#!/bin/bash
cd "$(dirname "$0")"
exec ../venv/bin/python -u candlestick_pattern_research/harami_scanner.py
