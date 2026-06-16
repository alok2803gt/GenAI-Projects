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

REM ── Stop any existing processes on ports 8000 and 8001 ──────
echo [CLEAN] Stopping any existing processes on ports 8000 / 8001...
FOR /F "tokens=5" %%P IN ('netstat -ano 2^>nul ^| findstr ":8000 "') DO (
    taskkill /PID %%P /F >nul 2>&1
)
FOR /F "tokens=5" %%P IN ('netstat -ano 2^>nul ^| findstr ":8001 "') DO (
    taskkill /PID %%P /F >nul 2>&1
)
timeout /t 1 /nobreak >nul
echo [CLEAN] Done. (iv_history and autotrader_state preserved)
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

REM ── Start frontend HTTP server on port 8001 ──────────────────
echo [START] Starting frontend server on http://localhost:8001 ...
start "IBKR Frontend" cmd /k "call %VENV_DIR%\Scripts\activate.bat && python -m http.server 8001 --directory %~dp0frontend"

REM ── Wait then open dashboard ──────────────────────────────────
timeout /t 3 /nobreak >nul
echo [START] Opening dashboard in browser...
start "" "http://localhost:8001/index.html"

echo.
echo ============================================================
echo  Backend  running at  http://localhost:8000
echo  Frontend running at  http://localhost:8001/index.html
echo  Close both cmd windows to stop
echo ============================================================
echo.
pause
