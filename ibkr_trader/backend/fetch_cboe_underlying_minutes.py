"""
Real historical SPY underlying 1-min RTH bars, reconstructed from CBOE
DataShop's real tick-level time-and-sales data -- removes the IBKR
2-month depth cap entirely (CBOE ticks confirmed real back to at least
2020, likely much further). Needed to extend the real-price 0DTE
backtest to genuine stress windows (COVID crash, 2022 bear market) since
neither IBKR nor Massive/Polygon (stocks tier not subscribed, 403) can
supply underlying bars that far back.

Endpoint: /allaccess/time-and-sales/underlying-trades. Paginated via a
real, confirmed-working cursor: pass `seq_no` = the LAST row's own
seq_no from the previous page to get the next page (first row of each
new page duplicates the last row of the previous one -- skipped here).
Confirmed max page size 5000 rows/request (50000 was rejected).

Usage: python fetch_cboe_underlying_minutes.py --start 2020-02-24 --end 2020-04-17 --out spy_covid_bars.json
"""
import argparse
import base64
import json
import time
from datetime import date, datetime, timedelta

import requests

PAGE_LIMIT = 5000
RTH_START = "09:30:00"
RTH_END = "16:00:00"


def get_token(cfg):
    client_id = cfg["cboe_datashop_client_id"]
    client_secret = cfg["cboe_datashop_client_secret"]
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = requests.post(
        "https://id.livevol.com/connect/token",
        headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials"}, timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def trading_days(start_str, end_str, holidays):
    d = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    days = []
    while d <= end:
        if d.weekday() < 5 and d not in holidays:
            days.append(d.isoformat())
        d += timedelta(days=1)
    return days


def get_with_retry(url, headers, params, max_retries=6):
    delay = 2.0
    for attempt in range(max_retries):
        resp = requests.get(url, headers=headers, params=params, timeout=60)
        if resp.status_code == 429:
            time.sleep(delay)
            delay = min(delay * 1.8, 30.0)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def fetch_day_minute_bars(token, symbol, date_str, page_pause_s=0.6):
    """Paginate all real ticks for the day, aggregate into 1-min RTH OHLCV bars."""
    headers = {"Authorization": f"Bearer {token}"}
    buckets = {}   # "HH:MM" -> {open, high, low, close, volume}
    last_seq = None
    page_num = 0
    done = False
    while not done:
        params = {"symbol": symbol, "date": date_str, "limit": PAGE_LIMIT}
        if last_seq is not None:
            params["seq_no"] = last_seq
        resp = get_with_retry(
            "https://api.livevol.com/v1/delayed/allaccess/time-and-sales/underlying-trades",
            headers, params,
        )
        rows = resp.json()
        if not isinstance(rows, list) or not rows:
            break
        page_num += 1
        start_idx = 1 if last_seq is not None else 0   # skip dup overlap row after page 1
        past_close = False
        for r in rows[start_idx:]:
            ts = r["timestamp"]   # "HH:MM:SS.mmm"
            hhmmss = ts[:8]
            if hhmmss < RTH_START:
                continue
            if hhmmss >= RTH_END:
                past_close = True
                continue
            hhmm = ts[:5]
            px = r.get("underlying_trade_price")
            sz = r.get("underlying_trade_size") or 0
            if px is None or px <= 0:
                continue
            b = buckets.get(hhmm)
            if b is None:
                buckets[hhmm] = {"open": px, "high": px, "low": px, "close": px, "volume": sz}
            else:
                b["high"] = max(b["high"], px)
                b["low"] = min(b["low"], px)
                b["close"] = px
                b["volume"] += sz
        last_seq = rows[-1]["seq_no"]
        # Stop once we've paged past regular-session close, or the page wasn't full (end of day's data)
        if past_close or len(rows) < PAGE_LIMIT:
            done = True
        else:
            time.sleep(page_pause_s)
    return buckets, page_num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--holidays", default="", help="comma-separated YYYY-MM-DD known holidays to skip")
    args = ap.parse_args()

    with open("scanner_config.json") as f:
        cfg = json.load(f)
    holidays = {datetime.strptime(d, "%Y-%m-%d").date() for d in args.holidays.split(",") if d}
    days = trading_days(args.start, args.end, holidays)
    print(f"Fetching real {args.symbol} underlying minute bars for {len(days)} real trading days "
          f"({args.start} to {args.end}) via CBOE tick pagination...")

    token = get_token(cfg)
    token_fetched_at = time.time()
    all_bars = {}
    t0 = time.time()
    for i, d in enumerate(days):
        if time.time() - token_fetched_at > 3300:   # refresh before the 3600s TTL
            token = get_token(cfg)
            token_fetched_at = time.time()
        d0 = time.time()
        try:
            buckets, n_pages = fetch_day_minute_bars(token, args.symbol, d)
            bars = [
                {"time": f"{d} {hhmm}:00", **v}
                for hhmm, v in sorted(buckets.items())
            ]
            all_bars[d] = bars
            print(f"  [{i+1}/{len(days)}] {d}: {len(bars)} minute bars from {n_pages} pages "
                  f"({time.time()-d0:.1f}s)")
        except Exception as e:
            print(f"  [{i+1}/{len(days)}] {d}: ERROR {e}")
        if (i + 1) % 3 == 0:
            with open(args.out, "w") as f:
                json.dump(all_bars, f)

    with open(args.out, "w") as f:
        json.dump(all_bars, f)
    print(f"\nDone in {time.time()-t0:.1f}s. Saved {len(all_bars)} days to {args.out}")


if __name__ == "__main__":
    main()
