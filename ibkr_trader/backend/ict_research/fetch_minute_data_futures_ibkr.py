"""
Real 1-min futures bars via IBKR continuous contracts (ContFuture) for
ES (E-mini S&P 500) and NQ (E-mini Nasdaq), CME -- the instruments ICT
methodology is most commonly taught and practiced on, given their near-
24h session structure (Asian/London/NY), unlike the US-equity-only
universe every prior stage in this file used.

Chunk size measured directly against this account's live connection: a
2-week request took 51.7s and returned real bars; a 1-month request timed
out at 90s (futures trade ~23h/day vs equities' 6.5h RTH, so the same
calendar duration is ~3.5x more bars). ~13 sequential 2-week chunks per
contract for 6 months, working backwards via endDateTime, matching the
chunking pattern in fetch_minute_data_ibkr.py.

useRTH=False (deliberately) -- the whole point of testing futures is the
near-24h session structure equities don't have; restricting to RTH would
throw away the Asian/London sessions before ever testing them.

Timestamps come back in US/Central (CME's local exchange timezone) --
converted to US/Eastern here to match every other cache file in this
directory, so poc_pullback_engine.py's functions work unmodified.

Saved to bars_1min_cache_futures_ibkr/<SYMBOL>.pkl.

Real methodology disclosure: the 3 quarterly contracts are concatenated
RAW, with no roll-gap price adjustment (no back-splicing). Different
contract months trade at slightly different absolute price levels
(cost-of-carry/calendar-spread), so there is a real, unadjusted price
gap at each of the 2 roll seams in this data. Given the engine computes
swing levels on a rolling intraday window and resets per session, this
mostly self-heals within a `lookback` window of the seam -- except
compute_swing_structure's daily-fractal HH/HL detection, which spans
across days and could register a spurious swing high/low exactly at a
seam day. 2 seam points in ~4-6 months of 1-min data is a small, disclosed
distortion, not adjusted for here rather than introducing a splicing
methodology of its own.

Run: python fetch_minute_data_futures_ibkr.py
"""
import asyncio
import os
import time

import pandas as pd
from ib_insync import IB, Future

# Real IBKR limitation hit live: ContFuture (the auto-rollover continuous
# contract) rejects any historical request with an end date/time set
# (Error 10339) -- it can only ever return "now minus duration", not walk
# backward in chunks. Real data only goes back 2 weeks that way. Fixed by
# using the SPECIFIC expiry contract per quarter instead (which DOES
# support backward chunking) and stitching 3 quarterly contracts together
# to reach ~6 months: Sept 2026 (current front month, about to expire),
# June 2026, March 2026. CME equity index futures expire the 3rd Friday
# of March/June/September/December.
SYMBOLS = [("ES", "CME"), ("NQ", "CME")]
CONTRACT_MONTHS = ["20260918", "20260619", "20260320"]  # newest first
CHUNK_DURATION = "2 W"
N_CHUNKS_PER_CONTRACT = 5  # ~5 x 2W ~= 2.5 months of real trading per contract
TWS_PORT = 7496
CLIENT_ID = 1963

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "bars_1min_cache_futures_ibkr")


def bars_to_df(bars) -> pd.DataFrame:
    df = pd.DataFrame([{
        "date": b.date, "open": b.open, "high": b.high, "low": b.low,
        "close": b.close, "volume": b.volume,
    } for b in bars])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df.index.tz is not None:
        df.index = df.index.tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_localize("US/Central").tz_convert("US/Eastern")
    return df


async def fetch_one_contract(ib: IB, symbol: str, exchange: str, expiry: str) -> pd.DataFrame:
    contract = Future(symbol, expiry, exchange)
    await ib.qualifyContractsAsync(contract)
    frames = []
    end_dt = ""
    for i in range(N_CHUNKS_PER_CONTRACT):
        t0 = time.time()
        bars = await ib.reqHistoricalDataAsync(
            contract, endDateTime=end_dt, durationStr=CHUNK_DURATION, barSizeSetting="1 min",
            whatToShow="TRADES", useRTH=False, formatDate=1, timeout=90,
        )
        if not bars:
            print(f"    {symbol} {expiry} chunk {i+1}/{N_CHUNKS_PER_CONTRACT}: 0 bars (stopping)")
            break
        df_chunk = bars_to_df(bars)
        print(f"    {symbol} {expiry} chunk {i+1}/{N_CHUNKS_PER_CONTRACT}: {len(df_chunk)} bars "
              f"[{df_chunk.index[0]} -> {df_chunk.index[-1]}] in {time.time()-t0:.1f}s")
        frames.append(df_chunk)
        end_dt = df_chunk.index[0].tz_convert("US/Central").strftime("%Y%m%d %H:%M:%S")
        await asyncio.sleep(2)

    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    return full


async def fetch_symbol(ib: IB, symbol: str, exchange: str) -> pd.DataFrame:
    all_frames = []
    for expiry in CONTRACT_MONTHS:
        print(f"  === {symbol} {expiry} ===")
        df = await fetch_one_contract(ib, symbol, exchange, expiry)
        if not df.empty:
            all_frames.append(df)
    if not all_frames:
        return pd.DataFrame()
    full = pd.concat(all_frames).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    return full


async def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    ib = IB()
    await ib.connectAsync("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    print("Connected to IBKR.")
    try:
        for symbol, exchange in SYMBOLS:
            print(f"=== {symbol} ===")
            df = await fetch_symbol(ib, symbol, exchange)
            if df.empty:
                print(f"  {symbol}: NO DATA, skipping cache write")
                continue
            path = os.path.join(CACHE_DIR, f"{symbol}.pkl")
            df.to_pickle(path)
            print(f"  {symbol}: {len(df)} total bars, {df.index[0]} -> {df.index[-1]}, saved to {path}")
    finally:
        ib.disconnect()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
