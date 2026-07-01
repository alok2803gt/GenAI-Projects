"""One-shot trigger: run EOD watchlist scan from today's close via IBKR data and send to Telegram.

Usage: python trigger_eod_watchlist.py [--days N]
  --days N   trading days of history to fetch (default 90; needs ≥50 for SMA50)
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import requests
import pandas as pd
from datetime import datetime

from breakout_scanner import (
    load_config, fetch_regime,
    run_eod_watchlist_scan, CURATED_TICKERS,
    _hist_cache as _cache,
)

parser = argparse.ArgumentParser()
parser.add_argument("--days", type=int, default=90)
args = parser.parse_args()

cfg         = load_config()
token       = cfg["telegram_token"]
chat_id     = cfg["telegram_chat_id"]
backend_url = cfg.get("backend_url", "http://localhost:8000")

print(f"Fetching {args.days} days of IBKR history for {len(CURATED_TICKERS)} tickers…")
print("(This can take 30-60s — IBKR pacing)")

BATCH = 50
tickers = CURATED_TICKERS

for i in range(0, len(tickers), BATCH):
    batch = tickers[i:i + BATCH]
    print(f"  Batch {i//BATCH + 1}: {len(batch)} tickers…", end=" ", flush=True)
    try:
        r = requests.post(
            f"{backend_url.rstrip('/')}/market/history/bulk",
            json=batch,
            params={"days": args.days},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json().get("data", {})
        loaded = 0
        for tk, bars in data.items():
            if not bars:
                continue
            df = pd.DataFrame(bars)
            df["Date"] = pd.to_datetime(df["date"])
            df = df.set_index("Date").rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })[["Open", "High", "Low", "Close", "Volume"]]
            df.index = df.index.tz_localize(None)
            _cache[tk] = df
            loaded += 1
        print(f"{loaded}/{len(batch)} loaded")
    except Exception as e:
        print(f"FAILED: {e}")

filled = sum(1 for v in _cache.values() if len(v) >= 30)
print(f"\nCache ready: {filled}/{len(tickers)} tickers with ≥30 bars")

if filled == 0:
    print("ERROR: No data loaded — is the backend running and IBKR connected?")
    sys.exit(1)

print("Fetching regime…")
regime = fetch_regime(backend_url)
print(f"Regime: {regime.get('regime_ok')} | SPY above SMA200: {regime.get('spy_above_sma200')}")

print("Running EOD watchlist scan…")
run_eod_watchlist_scan(tickers, cfg, token, chat_id, regime=regime)
print("Done — check Telegram.")
