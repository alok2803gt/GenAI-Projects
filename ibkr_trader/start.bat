@echo off
REM ─────────────────────────────────────────────────────────────
REM  XGBoost IBKR Trader — Windows Setup & Launch Script
REM  Run this once to install, then use again to start the server
REM ─────────────────────────────────────────────────────────────

SET VENV_DIR=%~dp0venv
SET BACKEND_DIR=%~dp0backend

echo.
echo ============================================================
echo  XGBoost IBKR Day Trading Signal Engine
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

REM Create venv if not exists
IF NOT EXIST "%VENV_DIR%" (
    echo [SETUP] Creating Python virtual environment...
    python -m venv "%VENV_DIR%"
)

REM Activate venv
call "%VENV_DIR%\Scripts\activate.bat"

REM Install requirements
echo [SETUP] Installing Python packages (first run takes ~2 min)...
pip install -q -r "%BACKEND_DIR%\requirements.txt"

echo.
echo [INFO]  Make sure TWS or IB Gateway is running on port 7497
echo [INFO]  Enable API access: TWS → Edit → Global Config → API → Enable ActiveX and Socket Clients
echo [INFO]  Tick "Allow connections from localhost only" for security
echo.

REM Start FastAPI backend
echo [START] Starting FastAPI backend on http://localhost:8000 ...
start "IBKR Backend" cmd /k "call %VENV_DIR%\Scripts\activate.bat && cd /d %BACKEND_DIR% && python main.py"

REM Wait a moment then open the frontend
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
