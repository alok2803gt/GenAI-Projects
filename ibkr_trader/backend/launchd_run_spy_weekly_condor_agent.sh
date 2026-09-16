#!/bin/bash
cd "$(dirname "$0")"
exec ../venv/bin/python -u spy_weekly_condor_agent.py
