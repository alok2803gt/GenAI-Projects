#!/bin/bash
cd "$(dirname "$0")"
exec ../venv/bin/python -u archive_gex_daily.py
