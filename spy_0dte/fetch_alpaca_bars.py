"""
Download 1-minute stock bars from Alpaca's market-data API into Parquet.

    python fetch_alpaca_bars.py --symbol SPY --years 5
    python fetch_alpaca_bars.py --symbol SPY --start 2021-09-01 --end 2026-09-21

Credentials (never written to disk by this script):
    APCA_API_KEY_ID / APCA_API_SECRET_KEY environment variables, or
    --keys-from path/to/config.json  (reads 'alpaca_api_key' / 'alpaca_secret_key')

Design notes
    * feed=sip (consolidated tape). The IEX-only feed carries a small fraction of
      real volume, which would distort every volume-based feature.
    * adjustment=raw. 0DTE options are struck against raw prices; dividend-adjusted
      history would shift the underlying relative to the strikes being modeled.
    * Downloads one calendar month per file under data/<symbol>_1min/, so an
      interrupted run resumes where it stopped. The current, still-open month is
      always re-fetched. The final file keeps ALL hours; prepare_bars() in the
      framework filters to the regular session.
    * Timestamps are each bar's START, converted to America/New_York.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.alpaca.markets/v2/stocks/{symbol}/bars"
NY = "America/New_York"
log = logging.getLogger("fetch")


def load_keys(keys_from: str | None) -> tuple[str, str]:
    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if keys_from:
        cfg = json.loads(Path(keys_from).read_text())
        key, secret = cfg.get("alpaca_api_key"), cfg.get("alpaca_secret_key")
    if not key or not secret:
        sys.exit("no Alpaca credentials: set APCA_API_KEY_ID / APCA_API_SECRET_KEY or pass --keys-from")
    return key, secret


def fetch_range(session: requests.Session, symbol: str, start: pd.Timestamp, end: pd.Timestamp,
                feed: str) -> pd.DataFrame:
    """All 1-min bars in [start, end), following next_page_token."""
    params = dict(timeframe="1Min", start=start.isoformat(), end=end.isoformat(),
                  limit=10000, adjustment="raw", feed=feed, sort="asc")
    rows, token, attempt = [], None, 0
    while True:
        if token:
            params["page_token"] = token
        r = session.get(BASE.format(symbol=symbol), params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            attempt += 1
            if attempt > 8:
                r.raise_for_status()
            wait = min(60, 2 ** attempt)
            log.warning("HTTP %s, retrying in %ss", r.status_code, wait)
            time.sleep(wait)
            continue
        if r.status_code in (401, 403):
            sys.exit(f"Alpaca refused the request ({r.status_code}): {r.text[:200]}")
        r.raise_for_status()
        attempt = 0
        body = r.json()
        rows.extend(body.get("bars") or [])
        token = body.get("next_page_token")
        if not token:
            break
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "trade_count", "vwap"])
    df = pd.DataFrame(rows).rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close",
                                            "v": "volume", "n": "trade_count", "vw": "vwap"})
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(NY)
    return df.set_index("time").sort_index()


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    first = start.tz_convert(NY).tz_localize(None).to_period("M").to_timestamp().tz_localize(NY)
    out, cur = [], first
    while cur < end:
        out.append(cur)
        cur = (cur + pd.offsets.MonthBegin(1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--years", type=float, default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--feed", default="sip", choices=["sip", "iex"])
    ap.add_argument("--keys-from", default=None)
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # The free data plan excludes the most recent 15 minutes; stay clear of it.
    end = pd.Timestamp(args.end, tz=NY) if args.end else pd.Timestamp.now(tz=NY) - pd.Timedelta(minutes=20)
    if args.start:
        start = pd.Timestamp(args.start, tz=NY)
    else:
        start = end - pd.DateOffset(years=args.years or 5)

    key, secret = load_keys(args.keys_from)
    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    part_dir = Path(args.out_dir) / f"{args.symbol}_1min"
    part_dir.mkdir(parents=True, exist_ok=True)
    this_month = pd.Timestamp.now(tz=NY).tz_localize(None).to_period("M")

    for m in month_starts(start, end):
        m_end = min(m + pd.offsets.MonthBegin(1), end)
        m_start = max(m, start)
        f = part_dir / f"{m:%Y-%m}.parquet"
        if f.exists() and m.tz_localize(None).to_period("M") != this_month:
            continue
        df = fetch_range(session, args.symbol, m_start.tz_convert("UTC"), m_end.tz_convert("UTC"), args.feed)
        df.to_parquet(f)
        log.info("%s %s: %d bars", args.symbol, f"{m:%Y-%m}", len(df))

    parts = sorted(part_dir.glob("*.parquet"))
    full = pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    full = full[(full.index >= start) & (full.index < end)]
    out = Path(args.out_dir) / f"{args.symbol}_1min_{args.feed}.parquet"
    full.to_parquet(out)

    rth = full[(full.index.hour * 60 + full.index.minute >= 570) & (full.index.hour * 60 + full.index.minute < 960)]
    per_day = rth.groupby(rth.index.normalize()).size()
    log.info("saved %s: %d bars total, %d regular-session bars over %d sessions (%s .. %s)",
             out, len(full), len(rth), len(per_day), full.index.min(), full.index.max())
    log.info("regular-session bars per day: median %d, min %d; days under 380 bars: %d",
             per_day.median(), per_day.min(), int((per_day < 380).sum()))


if __name__ == "__main__":
    main()
