#!/bin/bash
# OPTIONAL: build a single-file executable ON THE TARGET MACHINE
# (PyInstaller cannot cross-compile: build on the OS you will run on).
set -e
cd "$(dirname "$0")"
./.venv/bin/pip install --quiet pyinstaller
./.venv/bin/pyinstaller --onefile --name condor_v2 \
  --hidden-import=ib_insync --hidden-import=eventkit condor_v2.py
echo "binary at dist/condor_v2 -- keep config.json beside it"
