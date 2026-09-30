#!/bin/bash
# Per-strategy cap raised to $300 (CEO, 2026-09-23): the default 5%-of-net-liq
# cap was $105 on a $2,080 account while real wing_step=4 debits run $161-264,
# so every butterfly was rejected on cost alone, never on signal.
# NOTE: the separate portfolio-headroom check (50% total risk budget) is NOT
# overridden by this flag and can still reject -- see oversight_log.
cd "$(dirname "$0")"
exec ../venv/bin/python -u alpaca_0dte_butterfly_trader.py --ticker qqq --entry-mode window --fire --max-risk-override 300
