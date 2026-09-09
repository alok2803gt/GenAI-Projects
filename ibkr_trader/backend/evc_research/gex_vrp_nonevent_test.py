"""
Real test of the two higher-viability, non-event condor filters proposed
after PCR failed twice tonight (once on earnings condors, once on SPY
weekly volatility -- see gex_wall_backtest.py and pcr_nonevent_test.py):

  1. GEX regime: deploy condors only when net gamma is POSITIVE (dealer
     hedging dampens moves) -- avoid when negative (hedging amplifies moves).
  2. VRP (implied vs realized vol spread): deploy only when IV is elevated
     relative to recent REALIZED volatility -- the classic vol-risk-premium
     harvesting thesis.

Budget-conscious (CBOE DataShop quota: 149 points left as of this run,
~3 points/request -- burned nearly the whole month on one careless run two
nights ago). Reuses the forward-5-day-return numbers already computed in
pcr_nonevent_results.json (same underlying, same dates, no need to refetch
price data) and samples a SPREAD of ~40 dates across the existing 101-day,
3-regime range (Mar-Aug 2026) rather than a fresh contiguous pull, to keep
real regime diversity within a much smaller point budget.

Net GEX uses the same real formula as gex-vex-calculator/calc_gex_vex.py
and last night's gex_wall_backtest.py: dollar_gamma = gamma * OI * 100 *
spot^2 * 0.01, net = sum(call side) - sum(put side), summed across the
SAME near-term (5-12 DTE) expiry PCR was computed from -- the expiry
relevant to an actual weekly condor, not calc_gex_vex.py's 25-45 DTE window.

Realized vol: real trailing 5-day close-to-close stdev of SPY, annualized
(x sqrt(252)) for comparability with IV. Implied vol: CBOE's own iv30
field from the chain response (a real, delivered number, not derived).
"""
import base64
import json
import time
from datetime import date

import numpy as np
import requests
import yfinance as yf

N_TARGET = 40
POINTS_SAFETY_MARGIN = 25


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


def main():
    with open("scanner_config.json") as f:
        cfg = json.load(f)
    token = get_token(cfg)

    with open("evc_research/pcr_nonevent_results.json") as f:
        pcr_rows = json.load(f)
    pcr_rows.sort(key=lambda r: r["date"])
    by_date = {r["date"]: r for r in pcr_rows}

    # Spread ~N_TARGET dates evenly across the existing 101-day, 3-regime range
    all_dates = [r["date"] for r in pcr_rows]
    stride = max(1, len(all_dates) // N_TARGET)
    sample_dates = all_dates[::stride][:N_TARGET]
    print(f"Sampling {len(sample_dates)} dates spread across {all_dates[0]} to {all_dates[-1]}")

    hist = yf.Ticker("SPY").history(period="1y", interval="1d")
    hist.index = hist.index.tz_localize(None)
    closes = hist["Close"]
    date_list = [d.date().isoformat() for d in closes.index]

    def realized_vol_5d(as_of_date: str):
        if as_of_date not in date_list:
            return None
        idx = date_list.index(as_of_date)
        if idx < 5:
            return None
        window = closes.iloc[idx - 5: idx + 1]
        rets = np.diff(np.log(window.values))
        return round(float(np.std(rets) * np.sqrt(252) * 100), 2)  # annualized %

    results = []
    for i, d in enumerate(sample_dates):
        chain, points_left = fetch_chain(token, "SPY", d)
        if points_left is not None and int(points_left) < POINTS_SAFETY_MARGIN:
            print(f"  STOPPING early: only {points_left} points left")
            break
        if not chain or not chain.get("options"):
            print(f"  [{i+1}/{len(sample_dates)}] {d}: no data")
            continue
        opts = chain["options"]
        spot = chain.get("underlying_close")
        iv30 = chain.get("iv30")
        expiry = by_date[d]["expiry"]

        exp_opts = [o for o in opts if o.get("expiry") == expiry]
        net_gex = 0.0
        for o in exp_opts:
            oi = o.get("open_interest") or 0
            g = o.get("gamma") or 0
            dollar_gamma = g * oi * 100 * (spot ** 2) * 0.01
            net_gex += dollar_gamma if o.get("option_type") == "C" else -dollar_gamma

        rv5 = realized_vol_5d(d)
        rows_ref = by_date[d]
        results.append({
            "date": d, "spot": spot, "expiry": expiry,
            "net_gex": round(net_gex, 0),
            "gex_regime": "positive" if net_gex >= 0 else "negative",
            "iv30": iv30, "realized_vol_5d": rv5,
            "vrp": round(iv30 - rv5, 2) if (iv30 is not None and rv5 is not None) else None,
            "fwd_5d_ret_pct": rows_ref["fwd_5d_ret_pct"],
        })
        print(f"  [{i+1}/{len(sample_dates)}] {d}: net_gex={net_gex:+,.0f} ({results[-1]['gex_regime']})  "
              f"iv30={iv30}  rv5={rv5}  vrp={results[-1]['vrp']}  fwd_ret={rows_ref['fwd_5d_ret_pct']:+.2f}%  "
              f"(points_left={points_left})")
        time.sleep(0.2)

    with open("evc_research/gex_vrp_nonevent_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} real days to evc_research/gex_vrp_nonevent_results.json")


if __name__ == "__main__":
    main()
