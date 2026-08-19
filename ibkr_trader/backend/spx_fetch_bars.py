"""
Fetch real SPX 1-min RTH bars for a set of trading days, directly via IBKR
(not the backend's /market/history/minute endpoint -- that one hardcodes a
Stock contract, but SPX is a CBOE index and needs Index("SPX","CBOE"),
matching the exact pattern already used in main.py's own EVC/SPX 0DTE code).

Writes the same bar format as spy_0dte_bars.json so spy_0dte_backtest.py's
simulate_day() can run against it unchanged (ticker-agnostic already).

Usage: python spx_fetch_bars.py
"""
import json
from datetime import date, datetime, timedelta

from ib_insync import IB, Index

TWS_PORT = 7496
CLIENT_ID = 1555
N_DAYS = 40
OUT_FILE = "spx_0dte_bars.json"


def trading_days_back(n):
    days = []
    d = date.today() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d -= timedelta(days=1)
    return list(reversed(days))


def main():
    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    print("Connected to IBKR.")

    contract = Index("SPX", "CBOE")
    ib.qualifyContracts(contract)
    if not contract.conId:
        print("ERROR: could not qualify SPX index contract.")
        return
    print(f"Qualified SPX contract: conId={contract.conId}")

    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")

    days = trading_days_back(N_DAYS)
    print(f"Fetching {len(days)} real trading days of SPX 1-min bars via IBKR...")
    all_bars = {}
    for i, d in enumerate(days):
        try:
            end_dt = datetime.strptime(d, "%Y-%m-%d").replace(hour=16, minute=0, second=0, tzinfo=ET)
            bars = ib.reqHistoricalData(
                contract, endDateTime=end_dt, durationStr="1 D",
                barSizeSetting="1 min", whatToShow="TRADES",
                useRTH=True, keepUpToDate=False,
            )
            if bars:
                all_bars[d] = [
                    {"time": str(b.date), "open": b.open, "high": b.high,
                     "low": b.low, "close": b.close, "volume": b.volume}
                    for b in bars
                ]
                print(f"  [{i+1}/{len(days)}] {d}: {len(bars)} bars")
            else:
                print(f"  [{i+1}/{len(days)}] {d}: no data")
        except Exception as e:
            print(f"  [{i+1}/{len(days)}] {d}: ERROR {e}")
        ib.sleep(3)   # pace conservatively, same discipline as the backend's own endpoint

    ib.disconnect()
    with open(OUT_FILE, "w") as f:
        json.dump(all_bars, f)
    print(f"\nSaved {len(all_bars)} days of real SPX bars to {OUT_FILE}")


if __name__ == "__main__":
    main()
