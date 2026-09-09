"""
Pulls real 5-min intraday bars for SPY/QQQ/IWM across the exact 45 real
trading days already used in butterfly_0dte_v2_rows.csv (2026-04-23 to
2026-08-24), for butterfly_intraday_stoploss_test.py's real stop-loss
simulation. Cached to butterfly_intraday_cache/<TICKER>.pkl so the
simulation script can be iterated on without re-fetching.
"""
import json
import pickle
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

HERE = Path(__file__).parent
CACHE = HERE / "butterfly_intraday_cache"
CACHE.mkdir(exist_ok=True)

with open(HERE / "scanner_config.json") as f:
    cfg = json.load(f)

client = StockHistoricalDataClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"])

TICKERS = ["SPY", "QQQ", "IWM"]
START = "2026-04-23"
END = "2026-08-25"  # exclusive-ish buffer

for ticker in TICKERS:
    out_path = CACHE / f"{ticker}.pkl"
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=pd.Timestamp(START, tz="America/New_York"),
        end=pd.Timestamp(END, tz="America/New_York"),
    )
    bars = client.get_stock_bars(req).df
    if isinstance(bars.index, pd.MultiIndex):
        bars = bars.xs(ticker, level="symbol")
    bars.index = bars.index.tz_convert("America/New_York")
    bars.to_pickle(out_path)
    print(f"{ticker}: {len(bars)} 5-min bars, {bars.index.min()} to {bars.index.max()} -> {out_path}")

print("DONE")
