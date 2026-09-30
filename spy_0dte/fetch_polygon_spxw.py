"""
Download 1-minute bars for same-day-expiry SPXW options from Polygon.

For each session 2024-02-01 .. 2026-09-21: every $5 strike within +/-band of an
approximate SPX open (SPY 09:30 open x prior-day SPX/SPY close ratio -- used
only to decide what to download), calls and puts, expiring that day.
Saved per session to data/SPX_0dte_option_bars/ with the same schema as the
Alpaca files. Symbols are stored with root "SPX" (e.g. SPX250303C05900000) so
the 3-letter-root parser reads them. Resumable. Key read at run time only.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

NY = "America/New_York"
OUT = Path("data/SPX_0dte_option_bars")
BAND = 0.025


def get_json(sess, url, params):
    for attempt in range(8):
        try:
            r = sess.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(60, 2 ** attempt))
            continue
        if r.status_code in (401, 403):
            sys.exit(f"Polygon refused ({r.status_code}): {r.text[:200]}")
        return r.json()
    return {}


def fetch_contract(args):
    sess, key, day, right, k = args
    sym = f"O:SPXW{day:%y%m%d}{right}{int(k * 1000):08d}"
    d = f"{day:%Y-%m-%d}"
    body = get_json(sess, f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/minute/{d}/{d}",
                    dict(adjusted="true", sort="asc", limit=50000, apiKey=key))
    rows = []
    for b in body.get("results") or []:
        rows.append(dict(symbol=f"SPX{day:%y%m%d}{right}{int(k * 1000):08d}", time=b["t"], open=b["o"], high=b["h"],
                         low=b["l"], close=b["c"], volume=b["v"], trade_count=b.get("n"), vwap=b.get("vw")))
    return rows


def main():
    key = json.loads(Path(sys.argv[1]).read_text())["polygon_api_key"]
    OUT.mkdir(parents=True, exist_ok=True)
    spx = pd.read_csv("data/SPX_History.csv", parse_dates=["DATE"]).set_index("DATE")["SPX"]
    spy = pd.read_parquet("data/SPY_1min_sip.parquet")
    spy_open = spy.between_time("09:30", "09:30")["open"]
    spy_open.index = spy_open.index.tz_convert(NY).normalize().tz_localize(None)
    spy_close = spy.between_time("15:59", "15:59")["close"]
    spy_close.index = spy_close.index.tz_convert(NY).normalize().tz_localize(None)
    ratio = (spx / spy_close).dropna().shift(1)
    sess = requests.Session()
    days = [d for d in spy_open.index if pd.Timestamp("2024-09-23") <= d <= pd.Timestamp("2026-09-21")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, day in enumerate(days):
            f = OUT / f"{day:%Y-%m-%d}.parquet"
            if f.exists():
                continue
            r = ratio.asof(day)
            if not np.isfinite(r):
                continue
            S = spy_open[day] * r
            ks = np.arange(np.floor(S * (1 - BAND) / 5) * 5, np.ceil(S * (1 + BAND) / 5) * 5 + 1, 5.0)
            jobs = [(sess, key, day, right, k) for k in ks for right in ("C", "P")]
            rows = [row for part in pool.map(fetch_contract, jobs) for row in part]
            df = pd.DataFrame(rows)
            if len(df):
                df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert(NY)
                df = df.set_index("time").sort_index()
            else:
                df = pd.DataFrame(columns=["symbol", "open", "high", "low", "close", "volume", "trade_count", "vwap"])
            df.to_parquet(f)
            if i % 20 == 0:
                print(f"{day.date()}: {len(df)} bars, {df['symbol'].nunique() if len(df) else 0} contracts ({i + 1}/{len(days)})",
                      flush=True)
    print("done")


if __name__ == "__main__":
    main()
