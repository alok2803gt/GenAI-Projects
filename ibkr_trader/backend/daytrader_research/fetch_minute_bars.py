"""
Fetches real IBKR minute bars for every (date, ticker) in candidates.csv via
the backend's own /market/history/minute endpoint -- same real data source
and pacing convention (server-side ~5 req/15s) daytrader_intraday_backtest.py
already established and validated. Read-only against the backend; doesn't
touch any live trading state.

820 candidates at ~5/15s server-side pacing is a real, long fetch (~40min) --
run in background, generous client timeout.
"""
import json
import requests
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BACKEND_URL = "http://localhost:8000"

candidates = pd.read_csv(HERE / "candidates.csv")
pairs = list(zip(candidates["ticker"], candidates["date"]))
print(f"Fetching real minute bars for {len(pairs)} (ticker, date) pairs...")

payload = [{"ticker": tk, "date": d} for tk, d in pairs]
r = requests.post(f"{BACKEND_URL}/market/history/minute", json=payload, timeout=3600)
r.raise_for_status()
d = r.json()
print(f"Requested {d['requested']}, returned {d['returned']}")

with open(HERE / "minute_bars.json", "w") as f:
    json.dump(d["data"], f)
print("Saved minute_bars.json")
