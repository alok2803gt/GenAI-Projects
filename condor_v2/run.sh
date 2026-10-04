#!/bin/bash
# Runs the trader with config.json settings. Pass --dry-run to place nothing.
cd "$(dirname "$0")"
exec ./.venv/bin/python -u condor_v2.py "$@"
