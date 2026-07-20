#!/bin/bash
cd "$(dirname "$0")"
PY="../venv/bin/python"

if ! "$PY" -m py_compile main.py; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') SYNTAX ERROR in main.py — refusing to start" >> backend_watchdog.log
  sleep 60
  exit 1
fi
cp main.py main_last_good.py 2>/dev/null

exec "$PY" -u -m uvicorn main:app --host 0.0.0.0 --port 8000
