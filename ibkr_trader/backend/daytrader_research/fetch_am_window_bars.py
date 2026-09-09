"""
Pulls real 1-min bars for the full DISPERSION_COMBO_UNIVERSE (112 tickers)
across the same ~5-year window as target_combo_multiregime_rows.csv
(2021-09 to 2026-08), via Alpaca -- the same source/precedent already
used for this account's other intraday-faithful backtests (see
breakout_intraday_faithful_backtest.py). Immediately filters each
ticker's pull down to the 9:30-10:30 ET morning window per day before
caching, rather than keeping the full-day history -- that's all the next
script needs (a genuine "what would you know by 10:30am" dispersion
read), and discarding the rest keeps 112 tickers x 5 years from becoming
an unreasonable amount of cached data.

Parallelized (ThreadPoolExecutor, small worker count) since 112
sequential ~20-40s pulls would take 40-75 minutes single-threaded --
kept modest to stay reasonable against Alpaca's real rate limits, not
maximally aggressive.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
CACHE_DIR = os.path.join(HERE, "am_window_cache")
WORKERS = 6

with open(os.path.join(BACKEND_DIR, "scanner_config.json")) as f:
    CFG = json.load(f)

# Same 112-ticker DISPERSION_COMBO_UNIVERSE as daytrader_scanner.py -- loaded from the
# already-validated daily-bar cache's own key set rather than re-typed by hand, so this
# can never silently drift out of sync with what the real scanner actually uses.
import pickle
with open(os.path.join(BACKEND_DIR, "breakout_research", "universe_5y_ohlcv.pkl"), "rb") as f:
    TICKERS = sorted(pickle.load(f).keys())
print(f"Universe: {len(TICKERS)} tickers")

START = datetime(2021, 9, 1, tzinfo=ET)
END = datetime.now(ET) - timedelta(hours=1)  # free-tier Alpaca blocks querying "recent" SIP data


def fetch_one(ticker: str) -> tuple[str, int, str]:
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)
        return ticker, len(df), "cached"

    client = StockHistoricalDataClient(CFG["alpaca_api_key"], CFG["alpaca_secret_key"])
    req = StockBarsRequest(symbol_or_symbols=[ticker], timeframe=TimeFrame.Minute,
                            start=START, end=END, adjustment=Adjustment.ALL)
    raw = client.get_stock_bars(req).df
    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.loc[ticker]
    raw.index = pd.to_datetime(raw.index).tz_convert(ET)

    # Keep only 9:30:00-10:30:00 ET bars -- the morning window this research needs.
    t = raw.index.time
    mask = (t >= pd.Timestamp("09:30").time()) & (t <= pd.Timestamp("10:30").time())
    am = raw.loc[mask, ["open", "high", "low", "close", "volume"]]

    os.makedirs(CACHE_DIR, exist_ok=True)
    am.to_pickle(cache_path)
    return ticker, len(am), "fetched"


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_one, tk): tk for tk in TICKERS}
        for fut in as_completed(futures):
            tk = futures[fut]
            try:
                ticker, n, status = fut.result()
                done += 1
                print(f"[{done}/{len(TICKERS)}] {ticker}: {n} AM-window bars ({status})")
            except Exception as exc:
                done += 1
                print(f"[{done}/{len(TICKERS)}] {tk}: FAILED -- {exc}")
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
