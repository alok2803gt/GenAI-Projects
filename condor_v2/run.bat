@echo off
cd /d "%~dp0"
.venv\Scripts\python -u condor_v2.py %*
