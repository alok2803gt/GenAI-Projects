@echo off
REM ─────────────────────────────────────────────────────────────
REM  XGBoost IBKR Trader — Windows Setup & Launch Script
REM  Kills any existing backend, clears cache, then starts fresh
REM ─────────────────────────────────────────────────────────────

SET VENV_DIR=%~dp0venv
SET BACKEND_DIR=%~dp0backend

echo.
echo ============================================================
echo  XGBoost IBKR Day Trading Signal Engine
echo ============================================================
echo.

REM ── Stop any existing backend on port 8000 ───────────────────
echo [CLEAN] Stopping any existing backend on port 8000...
FOR /F "tokens=5" %%P IN ('netstat -ano 2^>nul ^| findstr ":8000 "') DO (
    taskkill /PID %%P /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM ── Clear runtime cache files (keep trade_journal.db) ────────
echo [CLEAN] Clearing cache files...
IF EXIST "%BACKEND_DIR%\iv_history.json"      del /f "%BACKEND_DIR%\iv_history.json"
IF EXIST "%BACKEND_DIR%\autotrader_state.json" del /f "%BACKEND_DIR%\autotrader_state.json"
echo [CLEAN] Done.
echo.

REM ── Check Python ─────────────────────────────────────────────
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

REM ── Create venv if not exists ─────────────────────────────────
IF NOT EXIST "%VENV_DIR%" (
    echo [SETUP] Creating Python virtual environment...
    python -m venv "%VENV_DIR%"
)

REM ── Activate venv ────────────────────────────────────────────
call "%VENV_DIR%\Scripts\activate.bat"

REM ── Install / update requirements ────────────────────────────
echo [SETUP] Checking Python packages...
pip install -q -r "%BACKEND_DIR%\requirements.txt"

echo.
echo [INFO]  Make sure TWS or IB Gateway is running on port 7497
echo [INFO]  Enable API access: TWS → Edit → Global Config → API → Enable ActiveX and Socket Clients
echo [INFO]  Tick "Allow connections from localhost only" for security
echo.

REM ── Start FastAPI backend ─────────────────────────────────────
echo [START] Starting FastAPI backend on http://localhost:8000 ...
start "IBKR Backend" cmd /k "call %VENV_DIR%\Scripts\activate.bat && cd /d %BACKEND_DIR% && python main.py"

REM ── Wait then open dashboard ──────────────────────────────────
timeout /t 3 /nobreak >nul
echo [START] Opening dashboard in browser...
start "" "%~dp0frontend\index.html"

echo.
echo ============================================================
echo  Backend running at  http://localhost:8000
echo  Dashboard opened in your default browser
echo  Close the "IBKR Backend" window to stop the server
echo ============================================================
echo.
pause
