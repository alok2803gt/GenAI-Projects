"""
Butterfly spread backtest -- real Polygon.io options data, 2026-09-01.

Two structures tested, but only ONE actually needs to be backtested: a
"reverse" (short) butterfly is the exact mirror image of a long butterfly
built from the identical three strikes -- same legs, bought vs sold. Its
entry credit is the negative of the long butterfly's entry debit, and its
expiry payoff is the negative of the long butterfly's expiry payoff. So
this script prices the LONG call butterfly with real data and derives the
reverse butterfly's stats by sign-flipping -- exact, not an approximation,
and it halves the real API load.

  long_butterfly:    buy 1x K1 call, sell 2x K2 call, buy 1x K3 call (debit)
                      -> bets the underlying PINS near K2 (low realized move)
  reverse_butterfly: sell 1x K1 call, buy 2x K2 call, sell 1x K3 call (credit)
                      -> bets the underlying makes a BIG move away from K2
                      (cheaper, capped alternative to the long straddle
                      already backtested this session, which found a real
                      +7.0% pooled edge on big-move days)

Entry: real closing prices for K1/K2/K3 calls on entry_date (Polygon
/v2/aggs on the actual OCC contract ticker -- this plan has real daily
option-contract closes, just no historical NBBO/quotes).
Exit: analytic intrinsic value at real expiry using the underlying's real
closing price on expiry_date (exact at expiration, not an approximation --
avoids the "was this OTM contract still trading on its last day" data gap
that would otherwise plague a expiry-date option-price fetch).

Strikes are the REAL listed strikes at the chosen expiry (not a rounded
guess) -- K2 = nearest real listed strike to spot, K1/K3 = N real strikes
away in the actual sorted strike ladder for that expiry.

DTE buckets:
  0DTE    -- SPY/QQQ/IWM only (real daily-expiry ETFs; most single names
             don't list same-day expiries historically)
  weekly  -- target ~7 DTE, broad 54-ticker universe (same universe as
             evc_long_straddle_backtest.py, for direct comparability)
  monthly -- target ~30 DTE, same broad universe

Outputs: butterfly_backtest_rows.csv, butterfly_backtest_summary.csv
"""
from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

BROAD_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC",
    "QCOM", "AVGO", "MU", "CRM", "ADBE", "NOW", "PLTR", "UBER",
    "JPM", "GS", "MS", "AXP", "V", "MA",
    "UNH", "JNJ", "PFE", "LLY", "ISRG",
    "XOM", "CVX", "SLB",
    "DE", "CAT", "BA", "HON",
    "WMT", "COST", "TGT", "HD", "LOW", "TJX", "ROST",
    "MCD", "SBUX", "NKE", "DIS", "NFLX",
    "BABA", "JD", "SHOP", "SNAP", "COIN", "F", "GM",
]
ETF_UNIVERSE = ["SPY", "QQQ", "IWM"]

DTE_BUCKETS = {
    "0DTE":    dict(target=0,  universe=ETF_UNIVERSE,    entry_stride_days=3,  search_days=3),
    "weekly":  dict(target=7,  universe=BROAD_UNIVERSE,  entry_stride_days=14, search_days=12),
    "monthly": dict(target=30, universe=BROAD_UNIVERSE,  entry_stride_days=14, search_days=45),
}
WING_STEPS = [2, 4]   # real strikes away from center, in the actual listed strike ladder

REQUEST_TIMEOUT = 15
MAX_RETRIES = 6
AGGS_WORKERS = 8
PLAN_AGGS_LOOKBACK_DAYS = 730
PLAN_BOUNDARY_SAFETY_DAYS = 5
POLY_BASE = "https://api.polygon.io"

with open("scanner_config.json", "r") as f:
    _cfg = json.load(f)
POLYGON_API_KEY = _cfg["polygon_api_key"]
_session = requests.Session()


def _polygon_get(path: str, params: dict) -> dict | None:
    params = dict(params)
    params["apiKey"] = POLYGON_API_KEY
    url = f"{POLY_BASE}{path}"
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            resp = _session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES - 1:
                print(f"    [POLY ERROR] {path}: {e}")
                return None
            time.sleep(delay)
            delay = min(delay * 1.7, 20.0)
            continue
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                return None
        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delay
            time.sleep(wait)
            delay = min(delay * 1.7, 20.0)
            continue
        if 500 <= resp.status_code < 600:
            time.sleep(delay)
            delay = min(delay * 1.7, 20.0)
            continue
        return None
    return None


def get_contract_grid_for_dte(ticker: str, entry_date: date, target_dte: int, search_days: int):
    """Real listed call strikes at the expiry closest to entry_date + target_dte."""
    lo = entry_date if target_dte == 0 else entry_date + timedelta(days=1)
    hi = entry_date + timedelta(days=max(target_dte + search_days, 1))
    data = _polygon_get(
        "/v3/reference/options/contracts",
        {
            "underlying_ticker": ticker,
            "contract_type": "call",
            "expiration_date.gte": lo.isoformat(),
            "expiration_date.lte": hi.isoformat(),
            "as_of": entry_date.isoformat(),
            "order": "asc", "sort": "expiration_date", "limit": 1000,
        },
    )
    if not data or data.get("status") not in ("OK", "OK "):
        return None, {}
    results = data.get("results") or []
    if not results:
        return None, {}
    target_date = entry_date + timedelta(days=target_dte)
    expiries = sorted({c["expiration_date"] for c in results})
    expiry_used = min(expiries, key=lambda e: abs((pd.Timestamp(e).date() - target_date).days))
    call_map = {}
    for c in results:
        if c.get("expiration_date") != expiry_used:
            continue
        occ = c.get("ticker")
        if occ:
            call_map[float(c["strike_price"])] = occ
    return expiry_used, call_map


def get_daily_close(occ_ticker: str, date_str: str):
    data = _polygon_get(f"/v2/aggs/ticker/{occ_ticker}/range/1/day/{date_str}/{date_str}", {"adjusted": "true"})
    if not data or data.get("resultsCount", 0) == 0:
        return None
    results = data.get("results") or []
    if not results:
        return None
    close = results[0].get("c")
    return float(close) if close is not None else None


def get_closes_concurrent(occ_tickers: dict, date_str: str) -> dict:
    names = list(occ_tickers.keys())
    with ThreadPoolExecutor(max_workers=min(AGGS_WORKERS, max(1, len(names)))) as ex:
        results = list(ex.map(lambda n: get_daily_close(occ_tickers[n], date_str), names))
    return dict(zip(names, results))


def fetch_ticker_history(ticker: str, retries: int = 3):
    last_err = None
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5y", interval="1d", auto_adjust=False, actions=False)
            if hist is None or hist.empty:
                raise ValueError("empty history")
            hist = hist[["Close"]].copy()
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
            hist = hist[~hist.index.duplicated(keep="last")].sort_index()
            splits = t.splits
            if splits is not None and not splits.empty:
                splits = splits.copy()
                splits.index = pd.to_datetime(splits.index).tz_localize(None).normalize()
            return hist, splits
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  [SKIP TICKER] {ticker}: could not fetch history ({last_err})")
    return None, None


def make_unadjust_fn(splits):
    if splits is None or len(splits) == 0:
        return lambda d: 1.0
    split_list = [(idx.date(), float(ratio)) for idx, ratio in splits.items() if ratio and ratio > 0]
    if not split_list:
        return lambda d: 1.0

    def factor(d):
        f = 1.0
        for sdate, ratio in split_list:
            if sdate > d:
                f *= ratio
        return f
    return factor


@dataclass
class Row:
    ticker: str
    dte_bucket: str
    wing_step: int
    entry_date: str
    status: str
    reason: str = ""
    expiry_used: str = ""
    days_to_expiry: float = np.nan
    spot: float = np.nan
    k1: float = np.nan
    k2: float = np.nan
    k3: float = np.nan
    entry_debit: float = np.nan   # long butterfly's real debit; reverse = -entry_debit (credit)
    exit_spot: float = np.nan
    long_exit_value: float = np.nan
    long_payoff: float = np.nan
    long_win: object = None


def process_ticker(ticker: str, bucket_name: str, bucket_cfg: dict, rows: list, stats: dict):
    hist, splits = fetch_ticker_history(ticker)
    if hist is None:
        return
    unadjust = make_unadjust_fn(splits)
    trading_days = list(hist.index.date)
    td_index = {d: i for i, d in enumerate(trading_days)}
    closes = hist["Close"].values

    def trading_day_on_or_after(d):
        for td in trading_days:
            if td >= d:
                return td
        return None

    plan_cutoff = date.today() - timedelta(days=PLAN_AGGS_LOOKBACK_DAYS + PLAN_BOUNDARY_SAFETY_DAYS)
    stride = bucket_cfg["entry_stride_days"]
    target_dte = bucket_cfg["target"]
    search_days = bucket_cfg["search_days"]
    n_ok = 0

    entry_candidates = [d for d in trading_days if plan_cutoff <= d < date.today() - timedelta(days=target_dte + 3)]
    entry_dates = entry_candidates[::max(stride // 1, 1)]
    # thin by calendar stride, not index stride, so cadence is real days not real trading-day count
    thinned = []
    last_kept = None
    for d in entry_candidates:
        if last_kept is None or (d - last_kept).days >= stride:
            thinned.append(d)
            last_kept = d
    entry_dates = thinned

    for entry_date in entry_dates:
        stats["total_attempts"] += 1
        entry_idx = td_index[entry_date]
        spot_adj = float(closes[entry_idx])
        if spot_adj <= 0:
            continue
        spot = spot_adj * unadjust(entry_date)
        entry_date_str = entry_date.isoformat()

        expiry_used, call_map = get_contract_grid_for_dte(ticker, entry_date, target_dte, search_days)
        if expiry_used is None or len(call_map) < 5:
            stats["no_contracts"] += 1
            continue
        strikes_sorted = sorted(call_map.keys())
        k2 = min(strikes_sorted, key=lambda s: abs(s - spot))
        k2_pos = strikes_sorted.index(k2)
        days_to_expiry = (pd.Timestamp(expiry_used).date() - entry_date).days

        for wing_step in WING_STEPS:
            base_kwargs = dict(ticker=ticker, dte_bucket=bucket_name, wing_step=wing_step,
                                entry_date=entry_date_str)
            lo_pos, hi_pos = k2_pos - wing_step, k2_pos + wing_step
            if lo_pos < 0 or hi_pos >= len(strikes_sorted):
                rows.append(Row(status="skip", reason="wing_out_of_range", expiry_used=expiry_used,
                                 spot=spot, k2=k2, **base_kwargs))
                continue
            k1, k3 = strikes_sorted[lo_pos], strikes_sorted[hi_pos]
            occs = {"k1": call_map[k1], "k2": call_map[k2], "k3": call_map[k3]}

            entry_closes = get_closes_concurrent(occs, entry_date_str)
            if any(entry_closes[k] is None for k in ("k1", "k2", "k3")):
                stats["no_entry_data"] += 1
                rows.append(Row(status="skip", reason="no_entry_data", expiry_used=expiry_used,
                                 days_to_expiry=days_to_expiry, spot=spot, k1=k1, k2=k2, k3=k3,
                                 **base_kwargs))
                continue
            entry_debit = entry_closes["k1"] + entry_closes["k3"] - 2 * entry_closes["k2"]
            if entry_debit <= 0.005:
                stats["nonpositive_debit"] += 1
                rows.append(Row(status="skip", reason="nonpositive_debit", expiry_used=expiry_used,
                                 days_to_expiry=days_to_expiry, spot=spot, k1=k1, k2=k2, k3=k3,
                                 entry_debit=entry_debit, **base_kwargs))
                continue

            expiry_td = trading_day_on_or_after(pd.Timestamp(expiry_used).date())
            if expiry_td is None or expiry_td not in td_index:
                stats["no_expiry_trading_day"] += 1
                continue
            exit_spot_adj = float(closes[td_index[expiry_td]])
            exit_spot = exit_spot_adj * unadjust(expiry_td)

            long_exit_value = (max(0.0, exit_spot - k1) - 2 * max(0.0, exit_spot - k2)
                               + max(0.0, exit_spot - k3))
            long_payoff = long_exit_value - entry_debit

            rows.append(Row(
                status="ok", expiry_used=expiry_used, days_to_expiry=days_to_expiry, spot=spot,
                k1=k1, k2=k2, k3=k3, entry_debit=entry_debit, exit_spot=exit_spot,
                long_exit_value=long_exit_value, long_payoff=long_payoff, long_win=bool(long_payoff > 0),
                **base_kwargs,
            ))
            n_ok += 1
            stats["ok"] += 1

    print(f"  {ticker} [{bucket_name}]: {n_ok} fully priced with real data", flush=True)


def main():
    rows: list = []
    stats = dict(total_attempts=0, no_contracts=0, no_entry_data=0, nonpositive_debit=0,
                 no_expiry_trading_day=0, ok=0)

    for bucket_name, bucket_cfg in DTE_BUCKETS.items():
        universe = bucket_cfg["universe"]
        print(f"\n=== DTE bucket: {bucket_name} (target={bucket_cfg['target']}d, "
              f"{len(universe)} tickers) ===", flush=True)
        for i, ticker in enumerate(universe, 1):
            print(f"[{i}/{len(universe)}] {ticker}", flush=True)
            try:
                process_ticker(ticker, bucket_name, bucket_cfg, rows, stats)
            except Exception as e:
                print(f"  [ERROR] {ticker}: {e}", flush=True)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv("butterfly_backtest_rows.csv", index=False)
    print(f"\nSaved {len(df)} rows. Stats: {stats}")

    ok = df[df.status == "ok"].copy()
    print(f"\nFully priced (real data): n={len(ok)}")
    if len(ok) == 0:
        print("No usable rows -- stopping.")
        return

    # Reverse butterfly = exact mirror: credit = -entry_debit, payoff = -long_payoff.
    ok["reverse_payoff"] = -ok["long_payoff"]
    ok["reverse_win"] = ok["reverse_payoff"] > 0
    ok["long_payoff_pct"] = ok["long_payoff"] / ok["entry_debit"]
    ok["reverse_payoff_pct"] = ok["reverse_payoff"] / ok["entry_debit"]  # % of long's debit == % of reverse's credit

    summary_rows = []
    for (bucket, wing), grp in ok.groupby(["dte_bucket", "wing_step"]):
        n = len(grp)
        summary_rows.append(dict(
            dte_bucket=bucket, wing_step=wing, structure="long_butterfly", n=n,
            win_rate=grp.long_win.mean(), avg_payoff_pct=grp.long_payoff_pct.mean(),
            median_payoff_pct=grp.long_payoff_pct.median(),
            pooled_return=grp.long_payoff.sum() / grp.entry_debit.sum(),
            avg_entry_debit=grp.entry_debit.mean(), avg_days_to_expiry=grp.days_to_expiry.mean(),
        ))
        summary_rows.append(dict(
            dte_bucket=bucket, wing_step=wing, structure="reverse_butterfly", n=n,
            win_rate=grp.reverse_win.mean(), avg_payoff_pct=grp.reverse_payoff_pct.mean(),
            median_payoff_pct=(-grp.long_payoff_pct).median(),
            pooled_return=grp.reverse_payoff.sum() / grp.entry_debit.sum(),
            avg_entry_debit=grp.entry_debit.mean(), avg_days_to_expiry=grp.days_to_expiry.mean(),
        ))

    summary = pd.DataFrame(summary_rows).sort_values(["dte_bucket", "wing_step", "structure"])
    summary.to_csv("butterfly_backtest_summary.csv", index=False)
    with pd.option_context("display.max_rows", None, "display.width", 160, "display.float_format", "{:.3f}".format):
        print("\n=== SUMMARY (by DTE bucket x wing width x structure) ===")
        print(summary.to_string(index=False))

    print("\n=== PER-TICKER, long_butterfly, n>=5 ===")
    grp = ok.groupby(["dte_bucket", "wing_step", "ticker"]).agg(
        n=("long_win", "size"), win_rate=("long_win", "mean"), avg_payoff_pct=("long_payoff_pct", "mean"))
    grp = grp[grp.n >= 5].sort_values("avg_payoff_pct", ascending=False)
    with pd.option_context("display.max_rows", 60, "display.width", 160):
        print(grp.head(60))

    print("\nDone.")


if __name__ == "__main__":
    main()
