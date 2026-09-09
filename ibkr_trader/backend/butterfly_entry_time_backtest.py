"""
Tests whether 9:45 ET is actually the best entry time, or just an inherited
assumption (borrowed from this account's SPX 0DTE convention, never
independently validated). Reuses butterfly_0dte_backtest_v2.py's exact
real-data methodology (real Unusual Whales intraday tape for option
entry prices, real Polygon minute bars for the morning spot, real
close-of-day intrinsic exit) -- the ONLY thing that changes is the
entry timestamp: 9:35, 9:45, and 10:00 ET, each run independently.

Restricted to wing_step=4 (the validated width) and the near-S/R filter
(prior trading day's high/low within 0.3%) -- the actual live config --
using the near-S/R spot check at EACH candidate entry time itself, not a
fixed 9:45 reference, since a real 9:35 or 10:00 strategy would check its
own spot at its own decision moment.

Output: butterfly_entry_time_rows.csv
"""
import sys
import io
import json
from datetime import date, datetime, timedelta, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd
import yfinance as yf

import butterfly_0dte_backtest_v2 as v2

ENTRY_TIMES = {"09:35": (9, 35), "09:45": (9, 45), "10:00": (10, 0)}
WING_STEP = 4


def get_morning_underlying_price_at(ticker, d, hh, mm):
    data = v2._polygon_get(f"/v2/aggs/ticker/{ticker}/range/1/minute/{d.isoformat()}/{d.isoformat()}",
                            {"adjusted": "true", "sort": "asc", "limit": 500})
    if not data or data.get("status") not in ("OK", "OK "):
        return None
    results = data.get("results") or []
    if not results:
        return None
    target = datetime(d.year, d.month, d.day, hh, mm, tzinfo=timezone.utc) + timedelta(hours=4)  # ET->UTC (EDT)
    best, best_diff = None, None
    for row in results:
        ts = datetime.fromtimestamp(row["t"] / 1000, tz=timezone.utc)
        diff = abs((ts - target).total_seconds())
        if diff > v2.ENTRY_SEARCH_WINDOW_MIN * 60:
            continue
        if best_diff is None or diff < best_diff:
            best, best_diff = row, diff
    return float(best["c"]) if best else None


def get_morning_option_price_at(occ_ticker, d, hh, mm):
    uw_id = occ_ticker[2:] if occ_ticker.startswith("O:") else occ_ticker
    data = v2._uw_get(f"/api/option-contract/{uw_id}/intraday", {"date": d.isoformat()})
    if not data:
        return None
    rows = data.get("data") or []
    if not rows:
        return None
    target = datetime(d.year, d.month, d.day, hh, mm, tzinfo=timezone.utc) + timedelta(hours=4)
    best, best_diff = None, None
    for row in rows:
        try:
            ts = datetime.fromisoformat(row["start_time"].replace("Z", "+00:00"))
        except Exception:
            continue
        diff = abs((ts - target).total_seconds())
        if diff > v2.ENTRY_SEARCH_WINDOW_MIN * 60:
            continue
        if best_diff is None or diff < best_diff:
            best, best_diff = row, diff
    return float(best["close"]) if best else None


def main():
    earliest = v2.uw_earliest_available_date()
    sr_hist_cache = {}
    rows = []

    for ticker in v2.ETF_UNIVERSE:
        print(f"\n=== {ticker} ===", flush=True)
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=False, actions=False)
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        sr_hist_cache[ticker] = hist
        trading_days = [d for d in hist.index.date if earliest <= d < date.today()]
        close_map = dict(zip(hist.index.date, hist["Close"].values))

        for i, d in enumerate(trading_days, 1):
            if i % 15 == 0:
                print(f"  [{i}/{len(trading_days)}] {d}", flush=True)
            prior = hist[hist.index.date < d]
            if prior.empty:
                continue
            prior_high, prior_low = float(prior.iloc[-1]["High"]), float(prior.iloc[-1]["Low"])

            for label, (hh, mm) in ENTRY_TIMES.items():
                spot = get_morning_underlying_price_at(ticker, d, hh, mm)
                if spot is None:
                    continue
                dist_high = abs(prior_high - spot) / spot
                dist_low = abs(spot - prior_low) / spot
                near_sr = dist_high < 0.003 or dist_low < 0.003

                call_map = v2.get_contract_grid_0dte(ticker, d)
                if len(call_map) < 2 * WING_STEP + 1:
                    continue
                strikes_sorted = sorted(call_map.keys())
                k2 = min(strikes_sorted, key=lambda s: abs(s - spot))
                k2_pos = strikes_sorted.index(k2)
                lo_pos, hi_pos = k2_pos - WING_STEP, k2_pos + WING_STEP
                if lo_pos < 0 or hi_pos >= len(strikes_sorted):
                    continue
                k1, k3 = strikes_sorted[lo_pos], strikes_sorted[hi_pos]

                p1 = get_morning_option_price_at(call_map[k1], d, hh, mm)
                p2 = get_morning_option_price_at(call_map[k2], d, hh, mm)
                p3 = get_morning_option_price_at(call_map[k3], d, hh, mm)
                if p1 is None or p2 is None or p3 is None:
                    continue
                entry_debit = p1 + p3 - 2 * p2
                if entry_debit <= 0.005:
                    continue

                close_spot = float(close_map[d])
                exit_value = max(0.0, close_spot - k1) - 2 * max(0.0, close_spot - k2) + max(0.0, close_spot - k3)
                payoff = exit_value - entry_debit

                rows.append(dict(ticker=ticker, entry_date=d.isoformat(), entry_time=label,
                                  spot=spot, near_sr=near_sr, entry_debit=entry_debit,
                                  payoff_dollar=payoff * 100, win=payoff > 0))

    df = pd.DataFrame(rows)
    df.to_csv("butterfly_entry_time_rows.csv", index=False)
    print(f"\nSaved {len(df)} rows")

    print("\n=== ALL DAYS, by entry time ===")
    for label in ENTRY_TIMES:
        g = df[df.entry_time == label]
        print(f"  {label}: n={len(g)} mean=${g.payoff_dollar.mean():+.2f} "
              f"median=${g.payoff_dollar.median():+.2f} win_rate={g.win.mean():.1%}")

    print("\n=== NEAR S/R ONLY (matches live config), by entry time ===")
    for label in ENTRY_TIMES:
        g = df[(df.entry_time == label) & (df.near_sr)]
        if len(g) == 0:
            print(f"  {label}: no near-S/R rows")
            continue
        print(f"  {label}: n={len(g)} mean=${g.payoff_dollar.mean():+.2f} "
              f"median=${g.payoff_dollar.median():+.2f} win_rate={g.win.mean():.1%}")

    print("\n=== per ticker, near S/R only ===")
    near = df[df.near_sr]
    for ticker in v2.ETF_UNIVERSE:
        print(f" -- {ticker} --")
        for label in ENTRY_TIMES:
            g = near[(near.ticker == ticker) & (near.entry_time == label)]
            if len(g) == 0:
                print(f"    {label}: no rows")
                continue
            print(f"    {label}: n={len(g)} mean=${g.payoff_dollar.mean():+.2f} win_rate={g.win.mean():.1%}")


if __name__ == "__main__":
    main()
