@echo off
REM One-time setup on Windows: creates .venv and installs dependencies.
cd /d "%~dp0"
python -m venv .venv
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
echo installed. next: run.bat --dry-run
