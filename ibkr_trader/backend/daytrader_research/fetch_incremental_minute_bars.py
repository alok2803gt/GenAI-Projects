"""
Fetches real IBKR minute bars for the incremental (ticker, date) pairs
needed by the no-sector-cap and 52-week-range candidate variants (not
already covered by the existing minute_bars.json). Same real backend
endpoint/pacing convention as fetch_minute_bars.py. Saves to a SEPARATE
file so the original minute_bars.json is never touched.
"""
import json
import requests
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BACKEND_URL = "http://localhost:8000"

pairs_df = pd.read_csv(HERE / "_incremental_pairs_needed.csv")
pairs = list(zip(pairs_df["ticker"], pairs_df["date"]))
print(f"Fetching real minute bars for {len(pairs)} incremental (ticker, date) pairs...")

payload = [{"ticker": tk, "date": d} for tk, d in pairs]
r = requests.post(f"{BACKEND_URL}/market/history/minute", json=payload, timeout=3600)
r.raise_for_status()
d = r.json()
print(f"Requested {d['requested']}, returned {d['returned']}")

with open(HERE / "minute_bars_incremental.json", "w") as f:
    json.dump(d["data"], f)
print("Saved minute_bars_incremental.json")
