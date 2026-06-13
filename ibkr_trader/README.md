# XGBoost IBKR Day Trading Signal Engine

Live signals from Interactive Brokers → XGBoost model → React dashboard.

---

## Prerequisites

### 1. Python 3.10 or 3.11
Download from https://www.python.org/downloads/
- During install, check **"Add Python to PATH"**
- Verify: open Command Prompt → `python --version`

### 2. Interactive Brokers TWS (Trader Workstation)
Download from https://www.interactivebrokers.com/en/trading/tws.php
- Log in with your IBKR credentials
- You can use **Paper Trading** account for safe testing

### 3. Enable TWS API
Inside TWS:
1. Go to **Edit → Global Configuration → API → Settings**
2. Check **"Enable ActiveX and Socket Clients"**
3. Set Socket port to **7497** (paper) or **7496** (live)
4. Check **"Allow connections from localhost only"**
5. Uncheck **"Read-Only API"** (needed to receive bar data)
6. Click **Apply → OK**
7. Restart TWS after changing these settings

---

## Project Structure

```
ibkr_trader/
├── backend/
│   ├── main.py           ← FastAPI server + IBKR + XGBoost
│   └── requirements.txt  ← Python dependencies
├── frontend/
│   └── index.html        ← React dashboard (no build needed)
├── start.bat             ← Double-click to install & launch
└── README.md
```

---

## Running (First Time)

1. Start **TWS** and log in (paper trading recommended first)
2. Enable API as described above
3. **Double-click `start.bat`**
   - Creates a Python virtual environment
   - Installs all packages (~2 min first time)
   - Starts the FastAPI backend
   - Opens the dashboard in your browser

On subsequent runs, just double-click `start.bat` again.

---

## Dashboard Features

| Feature | Description |
|---|---|
| Ticker tabs | Switch between AAPL, MSFT, NVDA, SPY |
| + Ticker | Add any US stock symbol |
| Candlestick chart | Live 5-min OHLCV bars from IBKR |
| Signal badge | BUY / HOLD / SELL with probability |
| Feature panel | RSI, momentum, volume ratio, etc. |
| Retrain button | Re-trains XGBoost on latest data |
| Signal history | Last 50 signals across all tickers |
| Auto-refresh | Updates every 15 seconds |

---

## Configuration

Edit `backend/main.py` to change defaults:

```python
TWS_PORT = 7497          # 7497=paper TWS | 7496=live TWS | 4002=IB Gateway paper
BAR_SIZE = "5 mins"      # 1 min / 5 mins / 15 mins / 1 hour
HISTORY_DURATION = "3 D" # how far back to fetch
POLL_INTERVAL = 30       # seconds between refreshes
BUY_THRESHOLD = 0.55     # prob > this → BUY
SELL_THRESHOLD = 0.45    # prob < this → SELL

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]
```

---

## Ports & Firewall

- Backend: `http://localhost:8000`
- The backend and dashboard both run locally — no internet connection needed after setup
- If Windows Firewall prompts you, allow access for `python.exe`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Python not found" | Reinstall Python, check "Add to PATH" |
| "IBKR connection lost" | Check TWS is running, API enabled, port matches |
| "No bars returned" | Market may be closed; TWS shows 0 bars outside RTH by default |
| Dashboard shows "DISCONNECTED" | Backend not running; rerun start.bat |
| Bars stop updating | TWS session expired; re-login to TWS |
| Port 8000 in use | Edit `main.py` → `uvicorn.run(..., port=8001)` and update `index.html` `API` constant |

---

## Architecture

```
TWS / IB Gateway (port 7497)
        │
        │ ib_insync (WebSocket)
        ▼
  main.py (FastAPI)
   ├─ fetch_bars()       → OHLCV DataFrame
   ├─ build_features()   → 9 technical features
   ├─ train_model()      → XGBClassifier (walk-forward CV)
   └─ predict()          → prob, label, confidence
        │
        │ HTTP (localhost:8000)
        ▼
  index.html (React)
   ├─ Candlestick chart
   ├─ Signal badge
   ├─ Feature panel
   └─ Signal history
```

---

## Important Disclaimer

This tool is for **research and paper trading only**.
Do not use signals from this tool for real trading decisions without
thorough backtesting, risk management, and professional financial advice.
Past model accuracy does not predict future market performance.
