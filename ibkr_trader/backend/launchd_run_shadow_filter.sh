#!/bin/bash
cd "$(dirname "$0")/breakout_research"
exec ../../venv/bin/python -u shadow_filter_monitor.py
