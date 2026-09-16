"""
Second batch of real IBKR 1-min bars, same method as fetch_minute_data_ibkr.py
(reuses its fetch_ticker/bars_to_df directly rather than duplicating the
logic) -- adds more tickers to reduce how much a single name (SPY, in the
FVG-confluence result) can dominate the average, and to test whether SPY's
weak performance there is SPY-specific or shared by other index ETFs.

Added: GOOGL, AMZN, TSLA, META, AMD, CRWD (liquid single-stock names, same
profile as MSFT/NVDA which performed well) + QQQ, IWM (2 more index ETFs,
to check if the FVG config's SPY weakness generalizes to index products or
is one ticker's noise).

Run: python fetch_minute_data_ibkr_batch2.py
"""
import asyncio
import os

from ib_insync import IB
from fetch_minute_data_ibkr import fetch_ticker, TWS_PORT, CACHE_DIR

TICKERS = ["GOOGL", "AMZN", "TSLA", "META", "AMD", "CRWD", "QQQ", "IWM"]
CLIENT_ID = 1954  # different from fetch_minute_data_ibkr.py's 1953 -- avoid clientId collision if run overlapping


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
