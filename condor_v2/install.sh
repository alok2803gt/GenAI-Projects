#!/bin/bash
# One-time setup: creates .venv next to this script and installs dependencies.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3)
echo "using $PY ($($PY --version))"
"$PY" -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt
echo "installed. next: ./run.sh --dry-run"
