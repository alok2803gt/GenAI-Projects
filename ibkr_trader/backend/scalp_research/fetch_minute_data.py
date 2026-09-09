"""
Pulls ~2.5 years of REAL 1-minute bars for AAPL/MSFT/NVDA/SPY from Alpaca
(split+dividend adjusted -- see signals_research/RESEARCH_LOG.md's data
note for why that matters: the first pass of that project used
unadjusted prices and produced a phantom +90% "trade" across NVDA's real
2024-06-07 stock split). Written fresh for this strategy -- does not
import signals_research/fetch_data.py, per instruction to build this
independently -- but applies the same lesson.

Caches to scalp_research/bars_1min_cache/<TICKER>.pkl. No session
resampling here (unlike signals_research's 5-min bars) -- VWAP needs the
raw 1-min bars to compute a genuine intraday volume-weighted average.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

ET = ZoneInfo("America/New_York")
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(HERE)
CACHE_DIR = os.path.join(HERE, "bars_1min_cache")

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]
MINUTE_YEARS = 2.5


def _load_cfg():
    with open(os.path.join(BACKEND_DIR, "scanner_config.json")) as f:
        return json.load(f)


def fetch_and_cache(force: bool = False) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cfg = _load_cfg()
    client = StockHistoricalDataClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"])

    end = datetime.now(ET) - timedelta(hours=1)  # free-tier Alpaca blocks querying "recent" SIP data
    start = end - timedelta(days=int(MINUTE_YEARS * 365))

    result = {}
    for tk in TICKERS:
        cache_path = os.path.join(CACHE_DIR, f"{tk}.pkl")
        if not force and os.path.exists(cache_path):
            print(f"[{tk}] using cached 1-min bars: {cache_path}")
            result[tk] = pd.read_pickle(cache_path)
            continue

        print(f"[{tk}] pulling {MINUTE_YEARS}y of 1-min bars from Alpaca (split+dividend adjusted)...")
        t0 = time.time()
        req = StockBarsRequest(symbol_or_symbols=[tk], timeframe=TimeFrame.Minute,
                                start=start, end=end, adjustment=Adjustment.ALL)
        raw = client.get_stock_bars(req).df
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.loc[tk]
        raw.index = pd.to_datetime(raw.index).tz_convert(ET)
        raw = raw[["open", "high", "low", "close", "volume"]]
        raw["date"] = raw.index.date

        raw.to_pickle(cache_path)
        result[tk] = raw
        print(f"[{tk}] {len(raw)} 1-min bars ({raw.index.min()} .. {raw.index.max()}) in {time.time()-t0:.1f}s")

    return result


if __name__ == "__main__":
    force = "--force" in sys.argv
    data = fetch_and_cache(force=force)
    for tk, df in data.items():
        print(f"{tk}: {len(df)} bars, {df.index.min()} .. {df.index.max()}")
