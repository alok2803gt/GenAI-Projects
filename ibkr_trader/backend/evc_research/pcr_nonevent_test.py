"""
Real test of the PCR framework in a NON-EVENT context (regular weekly SPY
premium selling, no earnings catalyst) -- using SPY's own real daily
option-chain OI (CBOE DataShop) as a market-wide PCR proxy, since no free
historical CBOE PCR INDEX exists (checked: yfinance has no ^PCC/^PCCE/^CPC/
^CPCE data). This is the same CBOE DataShop access last night's earnings
backtest used, quota confirmed reset this morning (497/500 points, 2026-08-26).

Budget-conscious this time: last night burned 498/500 points on one run
without checking the header first. This pulls only N_DAYS days (~3 points
each) and checks remaining quota before continuing past a safety margin.

Method:
  1. Real daily SPY chain (nearest 5-9 DTE expiry) for the last N_DAYS
     trading days -> real put OI, call OI -> daily PCR.
  2. Real forward 5-trading-day return from each day (a stand-in for "would
     a weekly condor centered here have been contained").
  3. Test: does PCR LEVEL predict forward realized move? Does PCR relative
     to its own trailing mean (peak vs floor) predict anything about the
     forward week, the way the framework's "timing" claim suggests?
"""
import base64
import json
import sys
import time
from datetime import date, timedelta

import numpy as np
import requests
import yfinance as yf

N_DAYS = 110
POINTS_SAFETY_MARGIN = 60  # stop pulling new days once remaining points drop below this


def get_token(cfg):
    basic = base64.b64encode(f"{cfg['cboe_datashop_client_id']}:{cfg['cboe_datashop_client_secret']}".encode()).decode()
    resp = requests.post(
        "https://id.livevol.com/connect/token",
        headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials"}, timeout=20,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def fetch_chain(token, symbol, date_str):
    resp = requests.get(
        "https://api.livevol.com/v1/delayed/allaccess/market/option-and-underlying-quotes",
        headers={"Authorization": f"Bearer {token}"},
        params={"symbol": symbol, "date": date_str}, timeout=60,
    )
    points_left = resp.headers.get("X-Monthly-Points-Left")
    if resp.status_code != 200:
        return None, points_left
    return resp.json(), points_left


def trading_days_back(n: int) -> list[str]:
    days, d = [], date.today() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.strftime("%Y-%m-%d"))
        d -= timedelta(days=1)
    return sorted(days)


def main():
    with open("scanner_config.json") as f:
        cfg = json.load(f)
    token = get_token(cfg)

    hist = yf.Ticker("SPY").history(period="1y", interval="1d")
    hist.index = hist.index.tz_localize(None)
    close_by_date = {d.date().isoformat(): c for d, c in hist["Close"].items()}
    all_dates = sorted(close_by_date.keys())

    try:
        with open("evc_research/pcr_nonevent_results.json") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = []
    have_dates = {r["date"] for r in existing}

    days = [d for d in trading_days_back(N_DAYS) if d not in have_dates]
    print(f"Already have {len(existing)} real days; pulling {len(days)} NEW real trading days...")

    rows = list(existing)
    for i, d in enumerate(days):
        chain, points_left = fetch_chain(token, "SPY", d)
        if points_left is not None:
            pl = int(points_left)
            if pl < POINTS_SAFETY_MARGIN:
                print(f"  STOPPING early: only {pl} points left (safety margin {POINTS_SAFETY_MARGIN})")
                break
        if not chain or not chain.get("options"):
            print(f"  [{i+1}/{len(days)}] {d}: no data")
            continue
        opts = chain["options"]
        spot = chain.get("underlying_close")

        # Near-term expiry: 5-9 real DTE from this date (weekly-ish window)
        d_date = date.fromisoformat(d)
        expiries = sorted({o["expiry"] for o in opts if o.get("expiry")})
        near = [e for e in expiries if 5 <= (date.fromisoformat(e) - d_date).days <= 12]
        if not near:
            print(f"  [{i+1}/{len(days)}] {d}: no near-term expiry in range")
            continue
        expiry = near[0]
        call_oi = sum(o.get("open_interest") or 0 for o in opts if o.get("expiry") == expiry and o.get("option_type") == "C")
        put_oi  = sum(o.get("open_interest") or 0 for o in opts if o.get("expiry") == expiry and o.get("option_type") == "P")
        if call_oi <= 0:
            print(f"  [{i+1}/{len(days)}] {d}: no call OI")
            continue
        pcr = round(put_oi / call_oi, 3)

        # Real forward 5-trading-day return from this date
        future = [dd for dd in all_dates if dd > d]
        if len(future) < 5 or d not in close_by_date:
            print(f"  [{i+1}/{len(days)}] {d}: not enough forward data yet")
            continue
        fwd_close = close_by_date[future[4]]
        fwd_ret = round((fwd_close / close_by_date[d] - 1) * 100, 3)

        rows.append({"date": d, "spot": spot, "expiry": expiry, "pcr": pcr,
                      "call_oi": call_oi, "put_oi": put_oi, "fwd_5d_ret_pct": fwd_ret})
        print(f"  [{i+1}/{len(days)}] {d}: PCR={pcr}  fwd_5d_ret={fwd_ret:+.2f}%  (points_left={points_left})")
        time.sleep(0.2)

    with open("evc_research/pcr_nonevent_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {len(rows)} real days to evc_research/pcr_nonevent_results.json")


if __name__ == "__main__":
    main()
