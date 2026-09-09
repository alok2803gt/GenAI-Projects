"""
Pulls ~2.5 years of 1-min bars for AAPL/MSFT/NVDA/SPY from Alpaca (the one
proven precedent in this codebase for multi-regime intraday history --
see breakout_intraday_faithful_backtest.py; IBKR's own per-day-pacing
approach only reaches 15-120 days) and resamples to 5-min bars to match
main.py's live BAR_SIZE = "5 mins".

Resampling: grouped by calendar date so a 5-min bin never straddles a
midnight boundary, and bins with zero volume (no real trade in that
5-min slice) are dropped -- this mirrors how the live system only ever
has a bar where a real print occurred, rather than synthesizing flat
bars across dead periods. Session is NOT filtered to RTH (matches
main.py:928's useRTH=False for the live subscription), so pre/post
market prints are included same as production.

Caches to signals_research/bars_cache/<TICKER>.pkl (pandas pickle -- no
pyarrow/fastparquet available in this environment) so repeated runs of
the backtest don't re-pull from Alpaca every time.

Requests split+dividend-adjusted prices (Adjustment.ALL). First run of
this backtest used Alpaca's default RAW (unadjusted) prices and produced
a phantom +90% "trade" on NVDA priced straight across its real 2024-06-07
10-for-1 split boundary (~$1208 pre-split close vs ~$120 post-split
open, same underlying value, zero real return) -- confirmed live
2026-09-07 by inspecting the actual trade record, not assumed.
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
CACHE_DIR = os.path.join(HERE, "bars_cache")

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]
MINUTE_YEARS = 2.5


def _load_cfg():
    with open(os.path.join(BACKEND_DIR, "scanner_config.json")) as f:
        return json.load(f)


def _resample_5min(df_1min: pd.DataFrame) -> pd.DataFrame:
    df = df_1min.copy()
    df.index = df.index.tz_convert(ET)
    df["date"] = df.index.date
    out = []
    for _, day_df in df.groupby("date"):
        day_df = day_df.drop(columns="date")
        r = day_df.resample("5min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        })
        r = r[r["volume"] > 0]  # drop empty bins -- no real print in that slice
        out.append(r)
    if not out:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(out).sort_index()


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
            print(f"[{tk}] using cached bars: {cache_path}")
            result[tk] = pd.read_pickle(cache_path)
            continue

        print(f"[{tk}] pulling {MINUTE_YEARS}y of 1-min bars from Alpaca...")
        t0 = time.time()
        req = StockBarsRequest(symbol_or_symbols=[tk], timeframe=TimeFrame.Minute,
                                start=start, end=end, adjustment=Adjustment.ALL)
        raw = client.get_stock_bars(req).df
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.loc[tk]
        raw.index = pd.to_datetime(raw.index)

        bars5 = _resample_5min(raw)
        bars5.to_pickle(cache_path)
        result[tk] = bars5
        print(f"[{tk}] {len(raw)} 1-min bars -> {len(bars5)} 5-min bars "
              f"({bars5.index.min()} .. {bars5.index.max()}) in {time.time()-t0:.1f}s")

    return result


if __name__ == "__main__":
    force = "--force" in sys.argv
    data = fetch_and_cache(force=force)
    for tk, df in data.items():
        print(f"{tk}: {len(df)} bars, {df.index.min()} .. {df.index.max()}")
