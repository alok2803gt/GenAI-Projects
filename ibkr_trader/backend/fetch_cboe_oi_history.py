"""
Fetch real historical SPY option chains (full chain, all expiries, real
open interest + CBOE-computed greeks) via CBOE DataShop's REST API, for
real GEX reconstruction -- confirmed working 2026-08-13 (unlike Massive/
Polygon, which only has current-state OI, see oversight_log.jsonl).

Covers the day BEFORE each of the 40 real 0DTE backtest days (spy_0dte_bars.json)
plus the 40 days themselves -- GEX is inherently an end-of-day figure (OI is
only reported once daily), so the real, honest way to use it as a regime
filter is: yesterday's close GEX informs today's regime, mirroring exactly
how the earlier VIX-proxy regime filter worked (spy_0dte_regime_backtest.py)
and how real GEX-aware traders actually use it.

Usage: python fetch_cboe_oi_history.py
Writes spy_cboe_chains.json: {date: [contracts]} (only the fields needed
for GEX, not the full raw response, to keep the cache a reasonable size).
"""
import json
import time
from datetime import datetime, timedelta

import requests
import base64

OUT_FILE = "spy_cboe_chains.json"
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


def fetch_day(token, symbol, date_str):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        "https://api.livevol.com/v1/delayed/allaccess/market/option-and-underlying-quotes",
        headers=headers, params={"symbol": symbol, "date": date_str}, timeout=60,
    )
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    data = resp.json()
    opts = data.get("options", [])
    return {
        "underlying_close": data.get("underlying_close"),
        "underlying_open": data.get("underlying_open"),
        "iv30": data.get("iv30"),
        "contracts": [{k: o.get(k) for k in KEEP_FIELDS} for o in opts],
    }


def main():
    with open("scanner_config.json") as f:
        cfg = json.load(f)
    with open("spy_0dte_bars.json") as f:
        bars = json.load(f)
    days = sorted(bars.keys())

    # Add the trading day immediately before the first backtest day
    d0 = datetime.strptime(days[0], "%Y-%m-%d")
    prior = d0 - timedelta(days=1)
    while prior.weekday() >= 5:
        prior -= timedelta(days=1)
    all_days = [prior.strftime("%Y-%m-%d")] + days

    token = get_token(cfg)
    print(f"Fetching {len(all_days)} real SPY option chains from CBOE DataShop...")
    chains = {}
    for i, d in enumerate(all_days):
        try:
            day_data = fetch_day(token, "SPY", d)
            if day_data is None:
                print(f"  [{i+1}/{len(all_days)}] {d}: no data (204)")
                continue
            chains[d] = day_data
            print(f"  [{i+1}/{len(all_days)}] {d}: {len(day_data['contracts'])} contracts, "
                  f"close=${day_data['underlying_close']}")
        except Exception as e:
            print(f"  [{i+1}/{len(all_days)}] {d}: ERROR {e}")
        if (i + 1) % 10 == 0:
            with open(OUT_FILE, "w") as f:
                json.dump(chains, f)
        time.sleep(0.3)

    with open(OUT_FILE, "w") as f:
        json.dump(chains, f)
    print(f"\nSaved {len(chains)} days to {OUT_FILE}")


if __name__ == "__main__":
    main()
