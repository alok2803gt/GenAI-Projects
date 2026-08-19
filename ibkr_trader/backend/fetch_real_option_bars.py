"""
Download real historical 0DTE option minute-bars from Massive/Polygon's
flat-files S3 bucket, for the same 40 real trading days already used in
spy_0dte_backtest.py / spx via spx_fetch_bars.py.

This replaces the Black-Scholes premium MODEL with real observed trade
prices -- confirmed dense enough to use near the money (~95%+ of the 390
minutes in a session have a real trade for near-ATM strikes, verified on
2026-06-17 SPY before building this).

Caveat stated up front: these are last-TRADE prices (OHLC), not bid/ask
quotes. A real fill would cross the spread; this still assumes fills at
the traded price with no slippage, same limitation as the BS version but
now grounded in real prints instead of a model.

Ticker roots: SPY 0DTE contracts are "O:SPY{YYMMDD}..."; SPX 0DTE trades
under the SPXW (SPX Weeklys) root, "O:SPXW{YYMMDD}...", not plain SPX
(confirmed via flat-file inspection 2026-08-13 -- plain O:SPX{date}...
does not carry daily 0DTE expiries, SPXW does).

Usage:
  python fetch_real_option_bars.py --ticker spy --bars-file spy_0dte_bars.json
  python fetch_real_option_bars.py --ticker spx --bars-file spx_0dte_bars.json
"""
import argparse
import gzip
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3
from botocore.config import Config

ET = ZoneInfo("America/New_York")
OPT_ROOT = {"spy": "SPY", "spx": "SPXW"}


def s3_client():
    with open("scanner_config.json") as f:
        cfg = json.load(f)
    session = boto3.Session(
        aws_access_key_id=cfg["polygon_s3_access_key"],
        aws_secret_access_key=cfg["polygon_s3_secret_key"],
    )
    return session.client("s3", endpoint_url="https://files.massive.com",
                           config=Config(signature_version="s3v4"))


def fetch_day(s3, day, root):
    """day: 'YYYY-MM-DD'. Returns {contract_ticker: [bars]} for 0DTE contracts only."""
    yymmdd = day[2:4] + day[5:7] + day[8:10]
    prefix = f"O:{root}{yymmdd}"
    y, m = day[:4], day[5:7]
    key = f"us_options_opra/minute_aggs_v1/{y}/{m}/{day}.csv.gz"
    try:
        obj = s3.get_object(Bucket="flatfiles", Key=key)
    except Exception as e:
        print(f"  {day}: ERROR fetching {key}: {e}")
        return {}
    data = gzip.decompress(obj["Body"].read())
    lines = data.decode().splitlines()
    contracts = {}
    for line in lines[1:]:
        if not line.startswith(prefix):
            continue
        parts = line.split(",")
        ticker, volume, o, c, h, l, window_start_ns, transactions = parts
        t = datetime.fromtimestamp(int(window_start_ns) / 1e9, tz=ET)
        bar = {
            "time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(o), "high": float(h), "low": float(l), "close": float(c),
            "volume": int(volume),
        }
        contracts.setdefault(ticker, []).append(bar)
    for ticker in contracts:
        contracts[ticker].sort(key=lambda b: b["time"])
    return contracts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, choices=["spy", "spx"])
    ap.add_argument("--bars-file", required=True, help="existing underlying bars file, gives the day list")
    args = ap.parse_args()

    with open(args.bars_file) as f:
        underlying_bars = json.load(f)
    days = sorted(underlying_bars.keys())
    root = OPT_ROOT[args.ticker]
    out_file = f"{args.ticker}_0dte_real_option_bars.json"

    print(f"Fetching real 0DTE option minute-bars for {args.ticker.upper()} "
          f"(root {root}) across {len(days)} real trading days...")
    s3 = s3_client()
    all_days = {}
    t0 = time.time()
    for i, day in enumerate(days):
        d0 = time.time()
        contracts = fetch_day(s3, day, root)
        all_days[day] = contracts
        n_contracts = len(contracts)
        n_bars = sum(len(v) for v in contracts.values())
        print(f"  [{i+1}/{len(days)}] {day}: {n_contracts} contracts, "
              f"{n_bars} total bars ({time.time()-d0:.1f}s)")
        # checkpoint every 5 days in case of a long-running interruption
        if (i + 1) % 5 == 0:
            with open(out_file, "w") as f:
                json.dump(all_days, f)

    with open(out_file, "w") as f:
        json.dump(all_days, f)
    print(f"\nDone in {time.time()-t0:.1f}s. Saved {len(all_days)} days to {out_file}")


if __name__ == "__main__":
    main()
