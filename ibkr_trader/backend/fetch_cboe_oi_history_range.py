"""
Generalized version of fetch_cboe_oi_history.py -- fetches real SPY option
chains (real OI + gamma + daily underlying OHLC) for an explicit date
range, not just the 40-day recent backtest window. Used to extend the
real-GEX regime test to genuinely diverse, non-confounded historical
periods (2026-08-13 finding: the recent 40-day window's positive_gamma
days were ALL the same August rally, negative_gamma days the same
choppier late-June/July range -- GEX regime and trend were confounded,
so the "negative gamma outperformed" result couldn't be trusted as a
real GEX effect vs. a trend effect).

Uses ONLY the single-request-per-day option-and-underlying-quotes
endpoint (NOT the paginated tick-level time-and-sales endpoint, which
hit a persistent 429 rate-limit wall on 2026-08-13) -- this endpoint has
shown no rate-limit issues across dozens of calls this session, and
gives real daily OHLC for the underlying for free alongside the OI/greeks
needed for GEX, without needing minute-level ticks at all.

Usage: python fetch_cboe_oi_history_range.py --start 2020-01-02 --end 2020-04-30 --out spy_cboe_chains_2020.json
"""
import argparse
import base64
import json
import time
from datetime import datetime, timedelta

import requests

KEEP_FIELDS = ("expiry", "strike", "option_type", "open_interest", "gamma", "delta")


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


def fetch_day(token, symbol, date_str, max_retries=5):
    headers = {"Authorization": f"Bearer {token}"}
    delay = 2.0
    for attempt in range(max_retries):
        resp = requests.get(
            "https://api.livevol.com/v1/delayed/allaccess/market/option-and-underlying-quotes",
            headers=headers, params={"symbol": symbol, "date": date_str}, timeout=60,
        )
        if resp.status_code == 429:
            time.sleep(delay)
            delay = min(delay * 1.7, 20.0)
            continue
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        data = resp.json()
        return {
            "underlying_open": data.get("underlying_open"),
            "underlying_high": data.get("underlying_high"),
            "underlying_low": data.get("underlying_low"),
            "underlying_close": data.get("underlying_close"),
            "iv30": data.get("iv30"),
            "contracts": [{k: o.get(k) for k in KEEP_FIELDS} for o in data.get("options", [])],
        }
    resp.raise_for_status()


def trading_days(start_str, end_str):
    d = datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_str, "%Y-%m-%d").date()
    days = []
    while d <= end:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d += timedelta(days=1)
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--symbol", default="SPY")
    args = ap.parse_args()

    with open("scanner_config.json") as f:
        cfg = json.load(f)
    days = trading_days(args.start, args.end)
    print(f"Fetching {len(days)} real {args.symbol} option chains ({args.start} to {args.end})...")

    token = get_token(cfg)
    token_fetched_at = time.time()
    chains = {}
    t0 = time.time()
    for i, d in enumerate(days):
        if time.time() - token_fetched_at > 3300:
            token = get_token(cfg)
            token_fetched_at = time.time()
        try:
            day_data = fetch_day(token, args.symbol, d)
            if day_data is None:
                print(f"  [{i+1}/{len(days)}] {d}: no data (204, likely a holiday)")
                continue
            chains[d] = day_data
            print(f"  [{i+1}/{len(days)}] {d}: {len(day_data['contracts'])} contracts, "
                  f"O={day_data['underlying_open']} H={day_data['underlying_high']} "
                  f"L={day_data['underlying_low']} C={day_data['underlying_close']}")
        except Exception as e:
            print(f"  [{i+1}/{len(days)}] {d}: ERROR {e}")
        if (i + 1) % 10 == 0:
            with open(args.out, "w") as f:
                json.dump(chains, f)
        time.sleep(0.4)

    with open(args.out, "w") as f:
        json.dump(chains, f)
    print(f"\nDone in {time.time()-t0:.1f}s. Saved {len(chains)} days to {args.out}")


if __name__ == "__main__":
    main()
