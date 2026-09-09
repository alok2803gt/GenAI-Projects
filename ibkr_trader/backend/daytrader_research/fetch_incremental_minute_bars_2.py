"""
Batch 2: fetches real IBKR minute bars for the (ticker, date) pairs needed
by the 7 new S/R/SMA/EMA factor variants, deduped against batch 1 (the
52-week-range/no-sector-cap fetch) to avoid redundant work. Same real
backend endpoint/pacing convention as fetch_minute_bars.py. Run AFTER
batch 1 completes -- avoids two concurrent heavy fetches contending for
the backend's own real IBKR pacing budget.
"""
import json
import requests
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BACKEND_URL = "http://localhost:8000"

pairs_df = pd.read_csv(HERE / "_incremental_pairs_needed_2_dedup.csv")
pairs = list(zip(pairs_df["ticker"], pairs_df["date"]))
print(f"Fetching real minute bars for {len(pairs)} incremental (ticker, date) pairs (batch 2)...")

payload = [{"ticker": tk, "date": d} for tk, d in pairs]
r = requests.post(f"{BACKEND_URL}/market/history/minute", json=payload, timeout=3600)
r.raise_for_status()
d = r.json()
print(f"Requested {d['requested']}, returned {d['returned']}")

# Merge into the SAME incremental file batch 1 wrote, rather than a third
# separate file -- simpler downstream loading.
inc_path = HERE / "minute_bars_incremental.json"
existing = {}
if inc_path.exists():
    with open(inc_path) as f:
        existing = json.load(f)
existing.update(d["data"])
with open(inc_path, "w") as f:
    json.dump(existing, f)
print(f"Merged into minute_bars_incremental.json -- now {len(existing)} total incremental keys")
