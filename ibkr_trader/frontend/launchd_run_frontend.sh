#!/bin/bash
cd "$(dirname "$0")"
exec ../venv/bin/python -m http.server 8001
