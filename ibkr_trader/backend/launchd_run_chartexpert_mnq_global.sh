#!/bin/bash
cd "$(dirname "$0")"
exec ../venv/bin/python -u chartexpert_auto_trader.py --symbols MNQ
