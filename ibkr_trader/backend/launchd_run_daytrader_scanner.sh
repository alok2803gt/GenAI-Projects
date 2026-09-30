#!/bin/bash
cd "$(dirname "$0")"
exec ../venv/bin/python -u daytrader_scanner.py
