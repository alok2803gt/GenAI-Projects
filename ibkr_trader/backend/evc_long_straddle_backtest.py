"""
Long ATM straddle/strangle into earnings -- real Polygon.io options data,
2026-09-01. The mirror image of EVC's iron condor: instead of SELLING
premium and betting the real move stays inside the wings, this BUYS the
ATM straddle and bets the real move exceeds what's priced in.

Reuses the SAME proven real-data pipeline as evc_condor_backtest_real_data.py
(identical universe, identical earnings-date/entry-date/outcome-date logic,
identical real Polygon ATM-straddle pricing for entry cost) -- copied
verbatim, not re-derived, so results are directly comparable. The ONLY
structural difference: no wing legs needed (a straddle is just the ATM
call + put), and the exit is priced from a REAL closing price on the
outcome date for the SAME two contracts, rather than an intrinsic-value
approximation -- this is actually MORE faithful than the condor backtest,
since real option prices are available on both the entry and outcome date
for this trade's only two legs.

Entry: buy 1x ATM call + 1x ATM put at the real closing price on entry_date.
Exit: sell both at their REAL closing price on outcome_date (next trading
day) -- captures real remaining time value, not just intrinsic value.
payoff = (exit_call + exit_put) - (entry_call + entry_put)

Outputs (same directory):
  evc_long_straddle_backtest_rows.csv
  evc_long_straddle_backtest_summary.csv
"""
from __future__ import annotations

import json
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

UNIVERSE = [
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

MIN_MOVE = 0.03   # same floor as EVC -- below this, premium is too cheap to bother pricing
MAX_EXPIRY_SEARCH_DAYS = 45
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


def round_strike(price: float) -> float:
    if price < 50:
        step = 1.0
    elif price < 200:
        step = 2.5
    else:
        step = 5.0
    return float(round(price / step) * step)


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


def get_contract_grid(ticker: str, entry_date: date):
    data = _polygon_get(
        "/v3/reference/options/contracts",
        {
            "underlying_ticker": ticker,
            "expiration_date.gte": entry_date.isoformat(),
            "expiration_date.lte": (pd.Timestamp(entry_date) + pd.Timedelta(days=MAX_EXPIRY_SEARCH_DAYS)).date().isoformat(),
            "as_of": entry_date.isoformat(),
            "order": "asc", "sort": "expiration_date", "limit": 1000,
        },
    )
    if not data or data.get("status") not in ("OK", "OK "):
        return None, {}, {}
    results = data.get("results") or []
    if not results:
        return None, {}, {}
    expiry_used = min(c["expiration_date"] for c in results)
    call_map, put_map = {}, {}
    for c in results:
        if c.get("expiration_date") != expiry_used:
            continue
        strike = float(c["strike_price"])
        occ = c.get("ticker")
        if not occ:
            continue
        if c.get("contract_type") == "call":
            call_map[strike] = occ
        elif c.get("contract_type") == "put":
            put_map[strike] = occ
    return expiry_used, call_map, put_map


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


def fetch_ticker_data(ticker: str, retries: int = 3):
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
            edf = t.get_earnings_dates(limit=20)
            if edf is None or edf.empty:
                raise ValueError("no earnings dates")
            splits = t.splits
            if splits is not None and not splits.empty:
                splits = splits.copy()
                splits.index = pd.to_datetime(splits.index).tz_localize(None).normalize()
            return hist, edf, splits
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  [SKIP TICKER] {ticker}: could not fetch data ({last_err})")
    return None, None, None


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
    earnings_date: str
    timing: str
    entry_date: str
    outcome_date: str
    status: str
    reason: str = ""
    expiry_used: str = ""
    days_to_expiry: float = np.nan
    spot: float = np.nan
    atm_strike: float = np.nan
    entry_call_price: float = np.nan
    entry_put_price: float = np.nan
    entry_cost: float = np.nan
    im_pct: float = np.nan
    exit_call_price: float = np.nan
    exit_put_price: float = np.nan
    exit_value: float = np.nan
    real_price: float = np.nan
    real_move_pct: float = np.nan
    payoff: float = np.nan
    payoff_pct_cost: float = np.nan
    win: object = None


def process_ticker(ticker: str, rows: list, stats: dict):
    hist, edf, splits = fetch_ticker_data(ticker)
    if hist is None:
        return
    unadjust = make_unadjust_fn(splits)
    trading_days = list(hist.index.date)
    td_index = {d: i for i, d in enumerate(trading_days)}
    closes = hist["Close"].values

    def next_trading_day(d):
        for td in trading_days:
            if td > d:
                return td
        return None

    def prev_trading_day(d):
        prev = None
        for td in trading_days:
            if td >= d:
                break
            prev = td
        return prev

    def trading_day_on_or_after(d):
        for td in trading_days:
            if td >= d:
                return td
        return None

    today = date.today()
    n_ok = 0
    for ts, erow in edf.iterrows():
        edate = pd.Timestamp(ts)
        edate_local = edate.date()
        if edate_local >= today:
            continue
        reported_eps = erow.get("Reported EPS", np.nan)
        if pd.isna(reported_eps):
            continue

        hour = edate.hour
        timing = "BMO" if hour < 12 else "AMC"
        if timing == "BMO":
            base_day = edate_local if edate_local in td_index else trading_day_on_or_after(edate_local)
            if base_day is None:
                continue
            entry_date = prev_trading_day(base_day)
        else:
            entry_date = edate_local if edate_local in td_index else prev_trading_day(
                trading_day_on_or_after(edate_local) or edate_local)
        if entry_date is None or entry_date not in td_index:
            continue
        outcome_date = next_trading_day(entry_date)
        if outcome_date is None:
            continue

        stats["total_events"] += 1
        entry_idx = td_index[entry_date]
        outcome_idx = td_index[outcome_date]
        base_kwargs = dict(ticker=ticker, earnings_date=str(edate_local), timing=timing,
                            entry_date=str(entry_date), outcome_date=str(outcome_date))

        spot_adj = float(closes[entry_idx])
        real_price_adj = float(closes[outcome_idx])
        if spot_adj <= 0 or real_price_adj <= 0:
            rows.append(Row(status="skip", reason="bad_price", **base_kwargs))
            continue
        spot = spot_adj * unadjust(entry_date)
        real_price = real_price_adj * unadjust(outcome_date)
        real_move_pct = (real_price - spot) / spot
        entry_date_str = entry_date.isoformat()
        outcome_date_str = outcome_date.isoformat()

        plan_cutoff = date.today() - timedelta(days=PLAN_AGGS_LOOKBACK_DAYS + PLAN_BOUNDARY_SAFETY_DAYS)
        if entry_date < plan_cutoff:
            stats["outside_plan_window"] += 1
            rows.append(Row(status="skip", reason="outside_polygon_plan_window", spot=spot,
                             real_price=real_price, real_move_pct=real_move_pct, **base_kwargs))
            continue

        expiry_used, call_map, put_map = get_contract_grid(ticker, entry_date)
        if expiry_used is None:
            stats["no_contracts_listed"] += 1
            rows.append(Row(status="skip", reason="no_contracts_listed", spot=spot,
                             real_price=real_price, real_move_pct=real_move_pct, **base_kwargs))
            continue
        days_to_expiry = (pd.Timestamp(expiry_used).date() - entry_date).days

        atm_strike = round_strike(spot)
        atm_call_occ = call_map.get(atm_strike)
        atm_put_occ = put_map.get(atm_strike)
        if atm_call_occ is None or atm_put_occ is None:
            stats["no_atm_contract"] += 1
            rows.append(Row(status="skip", reason="no_atm_contract", expiry_used=expiry_used, spot=spot,
                             real_price=real_price, real_move_pct=real_move_pct, **base_kwargs))
            continue

        entry_closes = get_closes_concurrent({"call": atm_call_occ, "put": atm_put_occ}, entry_date_str)
        entry_call_price, entry_put_price = entry_closes["call"], entry_closes["put"]
        if entry_call_price is None or entry_put_price is None:
            stats["no_trade_atm_leg"] += 1
            rows.append(Row(status="skip", reason="no_trade_atm_leg", expiry_used=expiry_used, spot=spot,
                             real_price=real_price, real_move_pct=real_move_pct, **base_kwargs))
            continue

        entry_cost = entry_call_price + entry_put_price
        im_pct = entry_cost / spot if spot > 0 else np.nan
        stats["usable_atm_data"] += 1

        if not math.isfinite(im_pct) or im_pct < MIN_MOVE:
            rows.append(Row(status="reject", reason="below_min_move", expiry_used=expiry_used, spot=spot,
                             atm_strike=atm_strike, entry_call_price=entry_call_price,
                             entry_put_price=entry_put_price, entry_cost=entry_cost, im_pct=im_pct,
                             real_price=real_price, real_move_pct=real_move_pct, **base_kwargs))
            continue
        stats["passed_move_filter"] += 1

        # Real exit: same two contracts' real closing price on the outcome date
        exit_closes = get_closes_concurrent({"call": atm_call_occ, "put": atm_put_occ}, outcome_date_str)
        exit_call_price, exit_put_price = exit_closes["call"], exit_closes["put"]
        if exit_call_price is None or exit_put_price is None:
            stats["no_trade_exit_leg"] += 1
            rows.append(Row(status="skip", reason="no_real_data:no_trade_exit_leg", expiry_used=expiry_used,
                             days_to_expiry=days_to_expiry, spot=spot, atm_strike=atm_strike,
                             entry_call_price=entry_call_price, entry_put_price=entry_put_price,
                             entry_cost=entry_cost, im_pct=im_pct, real_price=real_price,
                             real_move_pct=real_move_pct, **base_kwargs))
            continue

        exit_value = exit_call_price + exit_put_price
        payoff = exit_value - entry_cost

        rows.append(Row(
            status="ok", reason="", expiry_used=expiry_used, days_to_expiry=days_to_expiry,
            spot=spot, atm_strike=atm_strike, entry_call_price=entry_call_price,
            entry_put_price=entry_put_price, entry_cost=entry_cost, im_pct=im_pct,
            exit_call_price=exit_call_price, exit_put_price=exit_put_price, exit_value=exit_value,
            real_price=real_price, real_move_pct=real_move_pct, payoff=payoff,
            payoff_pct_cost=payoff / entry_cost if entry_cost else np.nan, win=bool(payoff > 0),
            **base_kwargs,
        ))
        n_ok += 1
        stats["ok"] += 1

    print(f"  {ticker}: {n_ok} fully priced with real data", flush=True)


def main():
    rows: list = []
    stats = dict(total_events=0, outside_plan_window=0, no_contracts_listed=0, no_atm_contract=0,
                 no_trade_atm_leg=0, usable_atm_data=0, passed_move_filter=0, no_trade_exit_leg=0, ok=0)
    print(f"Processing {len(UNIVERSE)} tickers -- LONG ATM straddle into earnings, real Polygon data")
    for i, ticker in enumerate(UNIVERSE, 1):
        print(f"[{i}/{len(UNIVERSE)}] {ticker}", flush=True)
        try:
            process_ticker(ticker, rows, stats)
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}", flush=True)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv("evc_long_straddle_backtest_rows.csv", index=False)
    print(f"\nSaved {len(df)} rows")

    ok = df[df.status == "ok"].copy()
    print(f"\nFully priced (real data): n={len(ok)}")
    if len(ok) == 0:
        print("No usable rows -- stopping.")
        return

    win_rate = ok.win.mean()
    avg_payoff_pct = ok.payoff_pct_cost.mean()
    med_payoff_pct = ok.payoff_pct_cost.median()
    sum_payoff = ok.payoff.sum()
    sum_cost = ok.entry_cost.sum()
    pooled_return = sum_payoff / sum_cost

    print(f"\n=== AGGREGATE (n={len(ok)}) ===")
    print(f"  win_rate (exit > entry cost) = {win_rate:.1%}")
    print(f"  avg payoff / cost            = {avg_payoff_pct:+.1%}")
    print(f"  median payoff / cost         = {med_payoff_pct:+.1%}")
    print(f"  pooled return (sum payoff / sum cost) = {pooled_return:+.1%}")
    print(f"  avg im_pct (straddle cost as % of spot) = {ok.im_pct.mean():.1%}")
    print(f"  avg |real_move_pct|          = {ok.real_move_pct.abs().mean():.1%}")
    print(f"  avg days_to_expiry           = {ok.days_to_expiry.mean():.1f}")

    summary = dict(n=len(ok), win_rate=win_rate, avg_payoff_pct_cost=avg_payoff_pct,
                   median_payoff_pct_cost=med_payoff_pct, pooled_return=pooled_return,
                   avg_im_pct=ok.im_pct.mean(), avg_abs_real_move_pct=ok.real_move_pct.abs().mean())
    pd.DataFrame([summary]).to_csv("evc_long_straddle_backtest_summary.csv", index=False)

    print("\n=== PER-TICKER (n>=5) ===")
    grp = ok.groupby("ticker").agg(n=("win", "size"), win_rate=("win", "mean"),
                                     avg_payoff_pct=("payoff_pct_cost", "mean"))
    grp = grp[grp.n >= 5].sort_values("avg_payoff_pct", ascending=False)
    with pd.option_context("display.max_rows", None, "display.width", 150):
        print(grp)

    print("\nDone.")


if __name__ == "__main__":
    main()
