"""
Pulls ~6 months of REAL 1-minute RTH bars for AAPL/MSFT/NVDA/SPY via IBKR
(ib_insync reqHistoricalData), not Alpaca -- this account has no Alpaca
market-data credentials configured, and sweep_engine.py's original 2.5-year
Alpaca-sourced cache (scalp_research/bars_1min_cache/) is gitignored and was
never regenerated on this machine. This is a real, disclosed methodology
difference from the original ict_research/ sweep-reversal backtest: a
~6-month IBKR sample instead of 2.5 years of Alpaca data, chosen because a
single reqHistoricalData call for 1-min bars beyond ~3 months reliably times
out against this account's live TWS connection (measured directly: 1M=7s,
2M=43s, 3M=64s, 6M in one shot=timeout/0 bars) -- so history is built from
three sequential 2-month chunks per ticker instead of one large pull.

Caches to ict_research/bars_1min_cache_ibkr/<TICKER>.pkl, kept separate
from scalp_research's Alpaca cache so the two data sources are never
silently mixed.

Usage: python fetch_minute_data_ibkr.py
"""
import asyncio
import os
import time

import pandas as pd
from ib_insync import IB, Stock

TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]
CHUNK_DURATION = "2 M"
N_CHUNKS = 3   # 3 x 2M ~= 6 months total
TWS_PORT = 7496
CLIENT_ID = 1953

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "bars_1min_cache_ibkr")


def bars_to_df(bars) -> pd.DataFrame:
    df = pd.DataFrame([{
        "date": b.date, "open": b.open, "high": b.high, "low": b.low,
        "close": b.close, "volume": b.volume,
    } for b in bars])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


async def fetch_ticker(ib: IB, ticker: str) -> pd.DataFrame:
    stock = Stock(ticker, "SMART", "USD")
    await ib.qualifyContractsAsync(stock)
    frames = []
    end_dt = ""  # "" = now, for the first (most recent) chunk
    for i in range(N_CHUNKS):
        t0 = time.time()
        bars = await ib.reqHistoricalDataAsync(
            stock, endDateTime=end_dt, durationStr=CHUNK_DURATION, barSizeSetting="1 min",
            whatToShow="TRADES", useRTH=True, formatDate=1, timeout=90,
        )
        if not bars:
            print(f"  {ticker} chunk {i+1}/{N_CHUNKS}: 0 bars (stopping further back-chunks)")
            break
        df_chunk = bars_to_df(bars)
        print(f"  {ticker} chunk {i+1}/{N_CHUNKS}: {len(df_chunk)} bars "
              f"[{df_chunk.index[0]} -> {df_chunk.index[-1]}] in {time.time()-t0:.1f}s")
        frames.append(df_chunk)
        # Next chunk ends right before this chunk's earliest bar.
        end_dt = df_chunk.index[0].strftime("%Y%m%d %H:%M:%S")
        await asyncio.sleep(2)  # be polite to IBKR pacing between chunks

    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    return full


async def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    ib = IB()
    await ib.connectAsync("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    print("Connected to IBKR.")
    try:
        for ticker in TICKERS:
            print(f"=== {ticker} ===")
            df = await fetch_ticker(ib, ticker)
            if df.empty:
                print(f"  {ticker}: NO DATA, skipping cache write")
                continue
            path = os.path.join(CACHE_DIR, f"{ticker}.pkl")
            df.to_pickle(path)
            print(f"  {ticker}: {len(df)} total bars, {df.index[0]} -> {df.index[-1]}, saved to {path}")
    finally:
        ib.disconnect()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
