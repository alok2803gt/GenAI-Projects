"""
Download 1-minute bars for same-day-expiry (0DTE) SPY options from Alpaca.

    python fetch_alpaca_options.py --keys-from path/config.json            # 2024-02-01 .. latest SPY session
    python fetch_alpaca_options.py --start 2025-01-02 --end 2025-01-31 --band 0.03

For each session in the SPY bar file: every listed-dollar strike within +/-band of
that session's 09:30 open, calls and puts, expiring that day. One Parquet file per
session in data/SPY_0dte_option_bars/ (resumable; an empty file marks a day with
no data). Alpaca has option BARS/TRADES from early 2024 but no historical
bid/ask QUOTES -- prices are traded prints; the bid/ask width stays modeled.

The strike band is chosen from the 09:30 open only to decide what to download;
trading decisions never see it. A day that moves further than the band will
have missing contracts, which the engine treats as "no quote" (no trade).
Credentials are read at run time and never written to disk.
"""
import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://data.alpaca.markets/v1beta1/options/bars"
NY = "America/New_York"
log = logging.getLogger("options")


def keys(path):
    k, s = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if path:
        cfg = json.loads(Path(path).read_text())
        k, s = cfg.get("alpaca_api_key"), cfg.get("alpaca_secret_key")
    if not k or not s:
        sys.exit("no Alpaca credentials: set APCA_API_KEY_ID / APCA_API_SECRET_KEY or pass --keys-from")
    return k, s


def occ(session, right, strike, root="SPY"):
    return f"{root}{session:%y%m%d}{right}{int(round(strike * 1000)):08d}"


def get(sess, params):
    for attempt in range(1, 9):
        r = sess.get(URL, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(60, 2 ** attempt))
            continue
        if r.status_code in (401, 403):
            sys.exit(f"Alpaca refused the request ({r.status_code}): {r.text[:200]}")
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Alpaca kept failing after retries")


def fetch_day(sess, session, symbols):
    start = (session + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC").isoformat()
    end = (session + pd.Timedelta(hours=16)).tz_convert("UTC").isoformat()
    rows = []
    for i in range(0, len(symbols), 100):
        params = dict(symbols=",".join(symbols[i:i + 100]), timeframe="1Min", start=start, end=end, limit=10000)
        token = None
        while True:
            if token:
                params["page_token"] = token
            body = get(sess, params)
            for sym, bars in (body.get("bars") or {}).items():
                for b in bars:
                    rows.append(dict(symbol=sym, time=b["t"], open=b["o"], high=b["h"], low=b["l"], close=b["c"],
                                     volume=b["v"], trade_count=b.get("n"), vwap=b.get("vw")))
            token = body.get("next_page_token")
            if not token:
                break
    if not rows:
        return pd.DataFrame(columns=["symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(NY)
    return df.set_index("time").sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="SPY", help="underlying symbol (SPY, QQQ, IWM)")
    ap.add_argument("--spy", default=None, help="underlying 1-min bars (default data/{root}_1min_sip.parquet)")
    ap.add_argument("--start", default="2024-02-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--band", type=float, default=0.03)
    ap.add_argument("--keys-from", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.spy = args.spy or f"data/{args.root}_1min_sip.parquet"
    args.out_dir = args.out_dir or f"data/{args.root}_0dte_option_bars"
    spy = pd.read_parquet(args.spy)
    rth = spy.between_time("09:30", "09:30")
    opens = rth["open"].groupby(rth.index.normalize()).first()
    opens = opens[opens.index >= pd.Timestamp(args.start, tz=NY)]
    if args.end:
        opens = opens[opens.index <= pd.Timestamp(args.end, tz=NY)]

    k, s = keys(args.keys_from)
    sess = requests.Session()
    sess.headers.update({"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    done = empty = 0
    for session, S in opens.items():
        f = out / f"{session:%Y-%m-%d}.parquet"
        if f.exists():
            continue
        lo, hi = math.floor(S * (1 - args.band)), math.ceil(S * (1 + args.band))
        symbols = [occ(session, r, k_, args.root) for k_ in range(lo, hi + 1) for r in ("C", "P")]
        df = fetch_day(sess, session, symbols)
        df.to_parquet(f)
        done += 1
        empty += df.empty
        if done % 20 == 0 or df.empty:
            log.info("%s: %d bars, %d contracts with data (%d days done, %d empty)",
                     session.date(), len(df), df["symbol"].nunique() if len(df) else 0, done, empty)
    files = sorted(out.glob("*.parquet"))
    nonempty = [p for p in files if pd.read_parquet(p, columns=["symbol"]).shape[0] > 0]
    log.info("finished: %d session files, %d with data (%s .. %s)", len(files), len(nonempty),
             nonempty[0].stem if nonempty else "-", nonempty[-1].stem if nonempty else "-")


if __name__ == "__main__":
    main()
