"""
Real 2.5-year 1-min bars via Alpaca (credentials added to scanner_config.json
2026-09-15), for the same 12 tickers already tested on ~6 months of IBKR
data. This is the real fix for this research effort's recurring limitation
(RESEARCH_LOG.md notes it at every stage): every result so far was found
AND checked on the same one 6-month window, no genuine out-of-sample data.

Alpaca's minute-bar endpoint accepts multi-symbol requests and has none of
IBKR's pacing/chunking constraints -- confirmed directly: 12 tickers x ~1
year in a single request took 127s. Fetched per-ticker here anyway (not one
giant combined request) so a single ticker failing doesn't lose everything
already fetched, and so progress can be checked/resumed.

Saved to bars_1min_cache_alpaca/<TICKER>.pkl, same schema as the existing
IBKR cache (open/high/low/close/volume, tz-aware US/Eastern index) so every
existing function in poc_pullback_engine.py works unmodified against it --
verified column/dtype/tz match against bars_1min_cache_ibkr/SPY.pkl before
writing this.

Run: python fetch_minute_data_alpaca.py
"""
import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "GOOGL", "AMZN", "TSLA", "META", "AMD", "CRWD", "QQQ", "IWM"]
YEARS_BACK = 2.5
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "bars_1min_cache_alpaca")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    cfg = json.load(open(os.path.join(HERE, "..", "scanner_config.json")))
    client = StockHistoricalDataClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"])

    end = datetime.now() - timedelta(days=1)
    start = end - timedelta(days=int(YEARS_BACK * 365))

    for ticker in TICKERS:
        t0 = time.time()
        req = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=TimeFrame.Minute, start=start, end=end)
        bars = client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            print(f"  {ticker}: NO DATA, skipping")
            continue
        df = df.droplevel("symbol")
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = df.index.tz_convert("US/Eastern")
        df.index.name = "date"
        path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
        df.to_pickle(path)
        print(f"  {ticker}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}, "
              f"{time.time()-t0:.1f}s, saved to {path}")

    print("Done.")


if __name__ == "__main__":
    main()
