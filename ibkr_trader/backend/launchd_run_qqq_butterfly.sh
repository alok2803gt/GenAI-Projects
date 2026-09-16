#!/bin/bash
cd "$(dirname "$0")"
exec ../venv/bin/python -u alpaca_0dte_butterfly_trader.py --ticker qqq --entry-mode window --fire
