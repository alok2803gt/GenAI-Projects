"""
0DTE long butterfly backtest, v2 -- fixes the v1 bug where entry and exit
both referenced the SAME closing price (spot == exit_spot in 100% of v1's
0DTE rows), making the "82% win rate" a near-tautology, not a real edge.

Real fix, real data, three sources combined:
  1. Real 9:45 AM ET underlying price (Polygon minute bars -- confirmed
     this plan has real minute-level equity/ETF bars; the known gap is
     options NBBO/quotes specifically, not underlying bars) -> picks the
     REAL morning ATM strike, not the closing strike.
  2. Real ~9:45 AM ET option prices for K1/K2/K3 (Unusual Whales
     /api/option-contract/{id}/intraday -- real 1-min trade-level tape,
     confirmed live; this account's plan has a 90-trading-day lookback,
     so the window is ~2026-04-23 onward, not the 2-year Polygon window
     used elsewhere -- smaller real sample, honestly smaller, not padded).
  3. Real close-of-day underlying price (existing yfinance daily history)
     -> exact intrinsic value at expiry, unchanged from v1 (this part was
     never wrong -- only the entry side was contaminated).

9:45 AM ET matches this account's own SPX 0DTE strategy's entry_start_time
convention, not an arbitrary choice.

Output: butterfly_0dte_v2_rows.csv
"""
from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

ETF_UNIVERSE = ["SPY", "QQQ", "IWM"]
WING_STEPS = [2, 4]
ENTRY_TARGET_ET_HOUR = 9
ENTRY_TARGET_ET_MIN = 45
ENTRY_SEARCH_WINDOW_MIN = 20   # accept a real trade/bar within +/-20min of 9:45 ET

with open("scanner_config.json") as f:
    _cfg = json.load(f)
POLYGON_API_KEY = _cfg["polygon_api_key"]
UW_API_KEY = _cfg["unusual_whales_api_key"]
_session = requests.Session()


def _polygon_get(path: str, params: dict) -> dict | None:
    params = dict(params)
    params["apiKey"] = POLYGON_API_KEY
    for attempt in range(5):
        try:
            r = _session.get(f"https://api.polygon.io{path}", params=params, timeout=20)
        except requests.RequestException:
            time.sleep(1.5)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(2.0)
            continue
        return None
    return None


def _uw_get(path: str, params: dict | None = None) -> dict | None:
    headers = {"Authorization": f"Bearer {UW_API_KEY}", "Accept": "application/json"}
    for attempt in range(5):
        try:
            r = _session.get(f"https://api.unusualwhales.com{path}", headers=headers,
                              params=params or {}, timeout=20)
        except requests.RequestException:
            time.sleep(1.5)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            time.sleep(2.0)
            continue
        return None
    return None


def get_contract_grid_0dte(ticker: str, d: date):
    data = _polygon_get(
        "/v3/reference/options/contracts",
        {"underlying_ticker": ticker, "contract_type": "call",
         "expiration_date.gte": d.isoformat(), "expiration_date.lte": d.isoformat(),
         "as_of": d.isoformat(), "limit": 1000},
    )
    if not data or data.get("status") not in ("OK", "OK "):
        return {}
    call_map = {}
    for c in data.get("results") or []:
        occ = c.get("ticker")
        if occ:
            call_map[float(c["strike_price"])] = occ
    return call_map


def get_morning_underlying_price(ticker: str, d: date):
    """Real Polygon minute bar closest to 9:45 AM ET on date d."""
    data = _polygon_get(f"/v2/aggs/ticker/{ticker}/range/1/minute/{d.isoformat()}/{d.isoformat()}",
                         {"adjusted": "true", "sort": "asc", "limit": 500})
    if not data or data.get("status") not in ("OK", "OK "):
        return None
    results = data.get("results") or []
    if not results:
        return None
    target = datetime(d.year, d.month, d.day, 13, 45, tzinfo=timezone.utc)  # 9:45 ET = 13:45 UTC (EDT)
    best, best_diff = None, None
    for row in results:
        ts = datetime.fromtimestamp(row["t"] / 1000, tz=timezone.utc)
        diff = abs((ts - target).total_seconds())
        if diff > ENTRY_SEARCH_WINDOW_MIN * 60:
            continue
        if best_diff is None or diff < best_diff:
            best, best_diff = row, diff
    return float(best["c"]) if best else None


def get_morning_option_price(occ_ticker: str, d: date):
    """Real UW 1-min intraday trade closest to 9:45 AM ET on date d."""
    uw_id = occ_ticker[2:] if occ_ticker.startswith("O:") else occ_ticker
    data = _uw_get(f"/api/option-contract/{uw_id}/intraday", {"date": d.isoformat()})
    if not data:
        return None
    rows = data.get("data") or []
    if not rows:
        return None
    target = datetime(d.year, d.month, d.day, 13, 45, tzinfo=timezone.utc)
    best, best_diff = None, None
    for row in rows:
        try:
            ts = datetime.fromisoformat(row["start_time"].replace("Z", "+00:00"))
        except Exception:
            continue
        diff = abs((ts - target).total_seconds())
        if diff > ENTRY_SEARCH_WINDOW_MIN * 60:
            continue
        if best_diff is None or diff < best_diff:
            best, best_diff = row, diff
    if best is None:
        return None
    return float(best["close"])


def get_morning_prices_concurrent(occ_map: dict, d: date) -> dict:
    names = list(occ_map.keys())
    with ThreadPoolExecutor(max_workers=min(6, max(1, len(names)))) as ex:
        results = list(ex.map(lambda n: get_morning_option_price(occ_map[n], d), names))
    return dict(zip(names, results))


@dataclass
class Row:
    ticker: str
    wing_step: int
    entry_date: str
    status: str
    reason: str = ""
    morning_spot: float = np.nan
    close_spot: float = np.nan
    k1: float = np.nan
    k2: float = np.nan
    k3: float = np.nan
    entry_debit: float = np.nan
    long_exit_value: float = np.nan
    long_payoff: float = np.nan
    long_win: object = None


def uw_earliest_available_date() -> date:
    """Probe UW's real access window rather than hardcoding it."""
    r = _session.get(
        "https://api.unusualwhales.com/api/option-contract/SPY260101C00500000/intraday",
        headers={"Authorization": f"Bearer {UW_API_KEY}"},
        params={"date": "2020-01-01"}, timeout=20,
    )
    try:
        msg = r.json().get("message", "")
        import re as _re
        m = _re.search(r"earliest date currently available.*?is (\d{4}-\d{2}-\d{2})", msg)
        if m:
            return date.fromisoformat(m.group(1))
    except Exception:
        pass
    return date.today() - timedelta(days=130)


def main():
    earliest = uw_earliest_available_date()
    print(f"Real UW intraday access window starts: {earliest}")

    rows = []
    for ticker in ETF_UNIVERSE:
        print(f"\n=== {ticker} ===", flush=True)
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=False, actions=False)
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        trading_days = [d for d in hist.index.date if earliest <= d < date.today()]
        close_map = dict(zip(hist.index.date, hist["Close"].values))

        for i, d in enumerate(trading_days, 1):
            if i % 10 == 0:
                print(f"  [{i}/{len(trading_days)}] {d}", flush=True)
            morning_spot = get_morning_underlying_price(ticker, d)
            if morning_spot is None:
                continue
            call_map = get_contract_grid_0dte(ticker, d)
            if len(call_map) < 9:
                continue
            strikes_sorted = sorted(call_map.keys())
            k2 = min(strikes_sorted, key=lambda s: abs(s - morning_spot))
            k2_pos = strikes_sorted.index(k2)
            close_spot = float(close_map.get(d, np.nan))

            for wing_step in WING_STEPS:
                base = dict(ticker=ticker, wing_step=wing_step, entry_date=d.isoformat())
                lo_pos, hi_pos = k2_pos - wing_step, k2_pos + wing_step
                if lo_pos < 0 or hi_pos >= len(strikes_sorted):
                    rows.append(Row(status="skip", reason="wing_out_of_range", morning_spot=morning_spot, **base))
                    continue
                k1, k3 = strikes_sorted[lo_pos], strikes_sorted[hi_pos]
                occ_map = {"k1": call_map[k1], "k2": call_map[k2], "k3": call_map[k3]}
                prices = get_morning_prices_concurrent(occ_map, d)
                if any(prices[k] is None for k in ("k1", "k2", "k3")):
                    rows.append(Row(status="skip", reason="no_morning_option_data", morning_spot=morning_spot,
                                     k1=k1, k2=k2, k3=k3, **base))
                    continue
                entry_debit = prices["k1"] + prices["k3"] - 2 * prices["k2"]
                if entry_debit <= 0.005 or np.isnan(close_spot):
                    rows.append(Row(status="skip", reason="nonpositive_debit_or_no_close",
                                     morning_spot=morning_spot, k1=k1, k2=k2, k3=k3,
                                     entry_debit=entry_debit, **base))
                    continue
                long_exit_value = (max(0.0, close_spot - k1) - 2 * max(0.0, close_spot - k2)
                                   + max(0.0, close_spot - k3))
                long_payoff = long_exit_value - entry_debit
                rows.append(Row(status="ok", morning_spot=morning_spot, close_spot=close_spot,
                                 k1=k1, k2=k2, k3=k3, entry_debit=entry_debit,
                                 long_exit_value=long_exit_value, long_payoff=long_payoff,
                                 long_win=bool(long_payoff > 0), **base))

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv("butterfly_0dte_v2_rows.csv", index=False)
    print(f"\nSaved {len(df)} rows")
    ok = df[df.status == "ok"].copy()
    print(f"Fully priced (real morning + real close data): n={len(ok)}")
    if len(ok) == 0:
        print("No usable rows.")
        return
    ok["payoff_dollar"] = ok.long_payoff * 100
    g = ok.groupby("wing_step").agg(
        n=("long_win", "size"), win_rate=("long_win", "mean"),
        mean_payoff_dollar=("payoff_dollar", "mean"), median_payoff_dollar=("payoff_dollar", "median"),
        min_payoff_dollar=("payoff_dollar", "min"), max_payoff_dollar=("payoff_dollar", "max"),
        pooled_return=("long_payoff", lambda s: s.sum() / ok.loc[s.index, "entry_debit"].sum()),
        mean_debit_dollar=("entry_debit", lambda s: s.mean() * 100),
    )
    print("\n=== SUMMARY (real morning entry, real close exit) ===")
    print(g.to_string())
    print("\nBy ticker:")
    g2 = ok.groupby(["wing_step", "ticker"]).agg(n=("long_win", "size"), win_rate=("long_win", "mean"),
                                                   mean_payoff_dollar=("payoff_dollar", "mean"))
    print(g2.to_string())


if __name__ == "__main__":
    main()
