"""
EVC (Earnings Vol Crush) iron-condor historical backtest -- REAL Polygon.io
options-chain data version.

This is a companion to evc_condor_backtest.py (the modeled Black-Scholes
version). It replicates that script's universe, earnings-date source,
entry/outcome-date logic, strike-selection convention (round_strike(),
short=1.0x EM, long=1.5x EM / WING_MULT), payoff formula, and MIN_MOVE /
MAX_MOVE_BOUNDS filters EXACTLY -- the ONLY thing that changes is where the
option premiums (and therefore the expected move) come from:

  * evc_condor_backtest.py:            Black-Scholes estimate off
                                        realized_vol_20d * {1.3, 1.6}
  * evc_condor_backtest_real_data.py:  REAL market-observed option closing
                                        prices from Polygon.io's historical
                                        options aggregates, for real listed
                                        contracts found via Polygon's options
                                        contracts reference endpoint
                                        (point-in-time, as_of=entry_date).

Per-event pipeline:
  1. Same entry/outcome date logic as the original (day before BMO earnings,
     day-of for AMC; outcome = next trading day).
  2. Find the nearest REAL listed options expiration >= entry_date for this
     ticker, as of entry_date (point-in-time correct -- no lookahead bias),
     via GET /v3/reference/options/contracts. This is necessarily the real
     market's answer to "the entry date's expiry" -- for tickers without
     daily/weekly listings this may be more than 1 trading day out, unlike
     the original's fixed 1-trading-day Black-Scholes assumption. This is a
     real divergence from the modeled version and is reported honestly
     (see days_to_expiry column and the printed report).
  3. ATM strike = round_strike(spot) (same function as the original). Look
     up the real listed call+put contracts at that exact strike on that real
     expiry. If either leg isn't listed -> skip (no fallback to BS).
  4. Pull REAL closing prices for both ATM legs on entry_date via
     GET /v2/aggs/ticker/O:.../range/1/day/{date}/{date}. Sum = real
     market-implied expected move (this literally IS the ATM straddle
     definition of "expected move"). If either leg has no trade that day
     (resultsCount == 0) -> skip (no fallback).
  5. Apply MIN_MOVE (3%) / MAX_MOVE_BOUNDS (12%/25%) filters to the REAL
     im_pct, exactly as the original.
  6. Compute short/long strikes the same way (round_strike(spot +/- EM),
     round_strike(spot +/- 1.5*EM)). Look up real listed contracts + real
     closing prices for all 4 wing legs on entry_date. Any missing listing
     or missing trade -> skip (no fallback).
  7. net_credit, wing widths, max_risk: same arithmetic as the original.
  8. Payoff at the REAL forward stock price (yfinance close on outcome_date):
     IDENTICAL graduated payoff formula, byte-for-byte copied from the
     original script.

Every skip due to missing/illiquid real data uses status="skip",
reason="no_real_data" (or a more specific no_* reason) -- it is NEVER
silently backfilled with a modeled price. The fraction of events that make
it to a real, fully-priced condor (status="ok") vs. get skipped for missing
real data is tracked explicitly and printed/saved as its own metric, because
that coverage rate is itself an important finding.

Run: python evc_condor_backtest_real_data.py
Outputs (same directory):
  evc_condor_backtest_real_data_rows.csv     (row-level, one row per event)
  evc_condor_backtest_real_data_summary.csv  (aggregate by max_move_bound)
  evc_condor_backtest_real_data_per_ticker.csv (per-ticker breakdown)
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

# ----------------------------------------------------------------------------
# Universe -- IDENTICAL to evc_condor_backtest.py
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Config -- MIN_MOVE / MAX_MOVE_BOUNDS / WING_MULT IDENTICAL to the original.
# No IV_MULTS axis here -- there is only one real market, not a sensitivity
# sweep over an assumed IV multiplier.
# ----------------------------------------------------------------------------
MAX_MOVE_BOUNDS = [0.12, 0.25]
MIN_MOVE = 0.03
WING_MULT = 1.5

MAX_EXPIRY_SEARCH_DAYS = 45     # how far past entry_date to look for a listed expiry
REQUEST_TIMEOUT = 15
MAX_RETRIES = 6
AGGS_WORKERS = 8                # parallel workers for the per-leg aggs fetches

# EMPIRICALLY CONFIRMED (via bisection against real API responses before this
# run): this Polygon.io subscription's options AGGREGATES endpoint (real
# historical closing prices) only covers a rolling ~2-year window, even
# though the options-contracts REFERENCE/metadata endpoint covers the full
# 5+ years. Probing AAPL contracts found the exact boundary: a request for
# 2024-08-20 returns 403 NOT_AUTHORIZED ("Your plan doesn't include this
# data timeframe"), while 2024-08-21 (exactly 2 years before this run's
# date) returns 200 OK. This means the real-data backtest can only ever
# cover ~40% of the original 5-year lookback window -- this is reported
# explicitly as its own finding (reason="outside_polygon_plan_window"), not
# folded into "no trade that day". A small safety margin is kept around the
# boundary in case it shifts by a day or two; real 403s are still handled
# defensively regardless.
PLAN_AGGS_LOOKBACK_DAYS = 730
PLAN_BOUNDARY_SAFETY_DAYS = 5

POLY_BASE = "https://api.polygon.io"

with open("scanner_config.json", "r") as f:
    _cfg = json.load(f)
POLYGON_API_KEY = _cfg["polygon_api_key"]

_session = requests.Session()


def round_strike(price: float) -> float:
    """Exact replica of round_strike() in evc_condor_backtest.py."""
    if price < 50:
        step = 1.0
    elif price < 200:
        step = 2.5
    else:
        step = 5.0
    return float(round(price / step) * step)


# ----------------------------------------------------------------------------
# Polygon.io helpers -- explicit retry/backoff on 429 and transient 5xx.
# ----------------------------------------------------------------------------
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
                print(f"    [POLY ERROR] {path} {params.get('underlying_ticker') or params.get('ticker','')}: {e}")
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
            print(f"    [429] backing off {wait:.1f}s")
            time.sleep(wait)
            delay = min(delay * 1.7, 20.0)
            continue
        if 500 <= resp.status_code < 600:
            time.sleep(delay)
            delay = min(delay * 1.7, 20.0)
            continue
        # 4xx other than 429: not retryable
        return None
    return None


def get_contract_grid(ticker: str, entry_date: date):
    """
    Find the nearest real listed options expiration >= entry_date for this
    ticker, as of entry_date (point-in-time, no lookahead). Returns:
      (expiry_used: str|None, call_map: {strike: occ_ticker}, put_map: {strike: occ_ticker})
    """
    data = _polygon_get(
        "/v3/reference/options/contracts",
        {
            "underlying_ticker": ticker,
            "expiration_date.gte": entry_date.isoformat(),
            "expiration_date.lte": (
                pd.Timestamp(entry_date) + pd.Timedelta(days=MAX_EXPIRY_SEARCH_DAYS)
            ).date().isoformat(),
            "as_of": entry_date.isoformat(),
            "order": "asc",
            "sort": "expiration_date",
            "limit": 1000,
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
    """Real closing price for one option contract on one date, or None."""
    data = _polygon_get(
        f"/v2/aggs/ticker/{occ_ticker}/range/1/day/{date_str}/{date_str}",
        {"adjusted": "true"},
    )
    if not data or data.get("resultsCount", 0) == 0:
        return None
    results = data.get("results") or []
    if not results:
        return None
    close = results[0].get("c")
    return float(close) if close is not None else None


def get_closes_concurrent(occ_tickers: dict, date_str: str) -> dict:
    """occ_tickers: {name: occ_ticker}. Returns {name: close|None}, parallelized."""
    names = list(occ_tickers.keys())
    with ThreadPoolExecutor(max_workers=min(AGGS_WORKERS, max(1, len(names)))) as ex:
        results = list(ex.map(lambda n: get_daily_close(occ_tickers[n], date_str), names))
    return dict(zip(names, results))


# ----------------------------------------------------------------------------
# yfinance data source -- IDENTICAL logic to evc_condor_backtest.py
# ----------------------------------------------------------------------------
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

            # IMPORTANT: yfinance's Close column is ALWAYS retroactively
            # split-adjusted (regardless of auto_adjust), but Polygon's real
            # historical options strike grid reflects the actual AS-TRADED
            # (unadjusted) price at the time. For any ticker that split
            # within the lookback window (e.g. NFLX 10:1 in Nov 2025), using
            # yfinance's split-adjusted Close directly against Polygon's
            # real, dated strike grid silently mismatches by the split
            # ratio -- e.g. NFLX priced ~$650 pre-split shows as ~$65 in
            # yfinance history, so round_strike(spot) lands nowhere near the
            # real $600s strikes that were actually listed that day. This
            # is corrected below by reconstructing an "as-traded" price
            # series using the ticker's real split history.
            splits = t.splits
            if splits is not None and not splits.empty:
                splits = splits.copy()
                splits.index = pd.to_datetime(splits.index).tz_localize(None).normalize()
            return hist, edf, splits
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  [SKIP TICKER] {ticker}: could not fetch data ({last_err})")
    return None, None, None


def make_unadjust_fn(splits):
    """
    Returns a function f(d: date) -> float that converts a yfinance
    split-adjusted price on date d back to the real as-traded price that
    was actually quoted on the market (and therefore comparable to
    Polygon's real, point-in-time options strike grid). Multiplies by the
    cumulative ratio of every split whose ex-date is AFTER d (undoing
    yfinance's retroactive adjustment for those later splits).
    """
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
    atm_call_price: float = np.nan
    atm_put_price: float = np.nan
    expected_move: float = np.nan
    im_pct: float = np.nan
    short_put: float = np.nan
    long_put: float = np.nan
    short_call: float = np.nan
    long_call: float = np.nan
    sp_price: float = np.nan
    lp_price: float = np.nan
    sc_price: float = np.nan
    lc_price: float = np.nan
    put_width: float = np.nan
    call_width: float = np.nan
    wing_width: float = np.nan
    net_credit: float = np.nan
    max_risk: float = np.nan
    breakeven_win_rate_trade: float = np.nan
    real_price: float = np.nan
    real_move_pct: float = np.nan
    payoff: float = np.nan
    payoff_pct_spot: float = np.nan
    payoff_pct_credit: float = np.nan
    win: object = None
    qualifies_12: bool = False
    qualifies_25: bool = False


def process_ticker(ticker: str, rows: list, stats: dict):
    hist, edf, splits = fetch_ticker_data(ticker)
    if hist is None:
        return

    unadjust = make_unadjust_fn(splits)

    trading_days = list(hist.index.date)
    td_index = {d: i for i, d in enumerate(trading_days)}
    closes = hist["Close"].values

    def next_trading_day(d: date):
        for td in trading_days:
            if td > d:
                return td
        return None

    def prev_trading_day(d: date):
        prev = None
        for td in trading_days:
            if td >= d:
                break
            prev = td
        return prev

    def trading_day_on_or_after(d: date):
        for td in trading_days:
            if td >= d:
                return td
        return None

    today = date.today()
    n_events = 0
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
                trading_day_on_or_after(edate_local) or edate_local
            )
        if entry_date is None or entry_date not in td_index:
            continue
        outcome_date = next_trading_day(entry_date)
        if outcome_date is None:
            continue

        n_events += 1
        stats["total_events"] += 1
        entry_idx = td_index[entry_date]
        outcome_idx = td_index[outcome_date]

        base_kwargs = dict(
            ticker=ticker,
            earnings_date=str(edate_local),
            timing=timing,
            entry_date=str(entry_date),
            outcome_date=str(outcome_date),
        )

        spot_adj = float(closes[entry_idx])
        real_price_adj = float(closes[outcome_idx])
        if spot_adj <= 0 or real_price_adj <= 0:
            rows.append(Row(status="skip", reason="bad_price", **base_kwargs))
            continue

        # Un-adjust for any splits that happened AFTER this historical date,
        # so spot/real_price are in the same as-traded scale as Polygon's
        # real, point-in-time options strike grid (see make_unadjust_fn).
        spot = spot_adj * unadjust(entry_date)
        real_price = real_price_adj * unadjust(outcome_date)

        real_move_pct = (real_price - spot) / spot
        atm_strike = round_strike(spot)
        entry_date_str = entry_date.isoformat()

        # --- Step 0: this Polygon plan's options AGGREGATES only cover a
        # rolling ~2-year window (see PLAN_AGGS_LOOKBACK_DAYS comment above).
        # Skip immediately (no wasted API calls) for events clearly before
        # that boundary; events near the boundary are still attempted so the
        # exact cutoff is empirically reflected in the real API responses.
        plan_cutoff = date.today() - timedelta(days=PLAN_AGGS_LOOKBACK_DAYS + PLAN_BOUNDARY_SAFETY_DAYS)
        if entry_date < plan_cutoff:
            stats["outside_plan_window"] += 1
            rows.append(Row(
                status="skip", reason="outside_polygon_plan_window",
                spot=spot, atm_strike=atm_strike, real_price=real_price,
                real_move_pct=real_move_pct, **base_kwargs,
            ))
            continue

        # --- Step 1: find real listed expiry + strike grid, as of entry_date
        expiry_used, call_map, put_map = get_contract_grid(ticker, entry_date)
        if expiry_used is None:
            stats["no_contracts_listed"] += 1
            rows.append(Row(
                status="skip", reason="no_contracts_listed",
                spot=spot, atm_strike=atm_strike, real_price=real_price,
                real_move_pct=real_move_pct, **base_kwargs,
            ))
            continue

        days_to_expiry = (pd.Timestamp(expiry_used).date() - entry_date).days

        atm_call_occ = call_map.get(atm_strike)
        atm_put_occ = put_map.get(atm_strike)
        if atm_call_occ is None or atm_put_occ is None:
            stats["no_atm_contract"] += 1
            rows.append(Row(
                status="skip", reason="no_atm_contract",
                expiry_used=expiry_used, days_to_expiry=days_to_expiry,
                spot=spot, atm_strike=atm_strike, real_price=real_price,
                real_move_pct=real_move_pct, **base_kwargs,
            ))
            continue

        # --- Step 2: real ATM straddle closing prices -> real expected move
        atm_closes = get_closes_concurrent(
            {"call": atm_call_occ, "put": atm_put_occ}, entry_date_str
        )
        atm_call_price, atm_put_price = atm_closes["call"], atm_closes["put"]
        if atm_call_price is None or atm_put_price is None:
            stats["no_trade_atm_leg"] += 1
            rows.append(Row(
                status="skip", reason="no_trade_atm_leg",
                expiry_used=expiry_used, days_to_expiry=days_to_expiry,
                spot=spot, atm_strike=atm_strike, real_price=real_price,
                real_move_pct=real_move_pct,
                atm_call_price=atm_call_price, atm_put_price=atm_put_price,
                **base_kwargs,
            ))
            continue

        expected_move = atm_call_price + atm_put_price
        im_pct = expected_move / spot if spot > 0 else np.nan
        stats["usable_atm_data"] += 1

        r = Row(
            status="ok", reason="",
            expiry_used=expiry_used, days_to_expiry=days_to_expiry,
            spot=spot, atm_strike=atm_strike,
            atm_call_price=atm_call_price, atm_put_price=atm_put_price,
            expected_move=expected_move, im_pct=im_pct,
            real_price=real_price, real_move_pct=real_move_pct,
            **base_kwargs,
        )

        if not math.isfinite(im_pct) or im_pct < MIN_MOVE:
            r.status = "reject"
            r.reason = "below_min_move"
            rows.append(r)
            continue
        if im_pct > max(MAX_MOVE_BOUNDS):
            r.status = "reject"
            r.reason = "above_max_move_bound"
            rows.append(r)
            continue
        stats["passed_move_filter"] += 1

        short_put = round_strike(spot - expected_move)
        short_call = round_strike(spot + expected_move)
        long_put = round_strike(spot - WING_MULT * expected_move)
        long_call = round_strike(spot + WING_MULT * expected_move)

        if not (long_put < short_put < short_call < long_call):
            r.status = "skip"
            r.reason = "invalid_strikes"
            rows.append(r)
            continue

        # --- Step 3: real listed wing contracts must exist at these exact strikes
        missing = []
        if short_put not in put_map:
            missing.append("short_put")
        if long_put not in put_map:
            missing.append("long_put")
        if short_call not in call_map:
            missing.append("short_call")
        if long_call not in call_map:
            missing.append("long_call")
        if missing:
            stats["no_real_data_wing_strike"] += 1
            r.status = "skip"
            r.reason = f"no_real_data:missing_strike:{','.join(missing)}"
            r.short_put, r.long_put = short_put, long_put
            r.short_call, r.long_call = short_call, long_call
            rows.append(r)
            continue

        # --- Step 4: real wing closing prices
        wing_occ = {
            "sp": put_map[short_put], "lp": put_map[long_put],
            "sc": call_map[short_call], "lc": call_map[long_call],
        }
        wing_closes = get_closes_concurrent(wing_occ, entry_date_str)
        sp_price, lp_price = wing_closes["sp"], wing_closes["lp"]
        sc_price, lc_price = wing_closes["sc"], wing_closes["lc"]

        r.short_put, r.long_put = short_put, long_put
        r.short_call, r.long_call = short_call, long_call
        r.sp_price, r.lp_price, r.sc_price, r.lc_price = sp_price, lp_price, sc_price, lc_price

        if any(p is None for p in (sp_price, lp_price, sc_price, lc_price)):
            stats["no_real_data_wing_trade"] += 1
            r.status = "skip"
            r.reason = "no_real_data:no_trade_wing_leg"
            rows.append(r)
            continue

        net_credit = round((sp_price + sc_price) - (lp_price + lc_price), 4)
        if net_credit <= 0:
            r.status = "skip"
            r.reason = "non_positive_credit"
            r.net_credit = net_credit
            rows.append(r)
            continue

        put_width = short_put - long_put
        call_width = long_call - short_call
        wing_width = max(put_width, call_width)
        max_risk = wing_width - net_credit
        if max_risk <= 0:
            r.status = "skip"
            r.reason = "non_positive_risk"
            r.net_credit = net_credit
            rows.append(r)
            continue

        # --- Step 5: payoff -- IDENTICAL formula to evc_condor_backtest.py
        if short_put <= real_price <= short_call:
            payoff = net_credit
        elif real_price < short_put:
            breach = short_put - real_price
            if real_price >= long_put:
                payoff = net_credit - breach
            else:
                payoff = net_credit - (put_width - net_credit)
        else:
            breach = real_price - short_call
            if real_price <= long_call:
                payoff = net_credit - breach
            else:
                payoff = net_credit - (call_width - net_credit)

        r.put_width, r.call_width, r.wing_width = put_width, call_width, wing_width
        r.net_credit = net_credit
        r.max_risk = max_risk
        r.breakeven_win_rate_trade = max_risk / (max_risk + net_credit)
        r.payoff = payoff
        r.payoff_pct_spot = payoff / spot
        r.payoff_pct_credit = payoff / net_credit
        r.win = bool(payoff > 0)
        r.qualifies_12 = im_pct <= 0.12
        r.qualifies_25 = im_pct <= 0.25
        rows.append(r)
        n_ok += 1
        stats["ok"] += 1

    print(f"  {ticker}: {n_events} historical earnings events processed, {n_ok} fully priced with real data")


def main():
    rows: list = []
    stats = dict(
        total_events=0, outside_plan_window=0, no_contracts_listed=0, no_atm_contract=0,
        no_trade_atm_leg=0, usable_atm_data=0, passed_move_filter=0, no_real_data_wing_strike=0,
        no_real_data_wing_trade=0, ok=0,
    )
    print(f"Processing {len(UNIVERSE)} tickers against REAL Polygon.io options data...")
    for i, ticker in enumerate(UNIVERSE, 1):
        print(f"[{i}/{len(UNIVERSE)}] {ticker}")
        try:
            process_ticker(ticker, rows, stats)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {ticker}: {e}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    out_path = "evc_condor_backtest_real_data_rows.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # -------------------------------------------------------------------
    # Data coverage / skip-rate funnel -- reported honestly, not papered over
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("REAL-DATA COVERAGE FUNNEL")
    print("=" * 100)
    te = stats["total_events"]
    within_window = te - stats["outside_plan_window"]
    print(f"  total historical earnings events (5y window, matches original) : {te}")
    print(f"  -> outside Polygon plan's ~2y aggs window (skip)                : {stats['outside_plan_window']}  ({stats['outside_plan_window']/max(1,te):.1%} of total)")
    print(f"  = events WITHIN the plan's real-data window                     : {within_window}  ({within_window/max(1,te):.1%} of total)")
    print(f"  -> no options contracts listed near entry (skip)                : {stats['no_contracts_listed']}")
    print(f"  -> ATM strike not listed as real contract (skip)                : {stats['no_atm_contract']}")
    print(f"  -> ATM leg had no real trade that day (skip)                    : {stats['no_trade_atm_leg']}")
    print(f"  = events with a USABLE real expected move                       : {stats['usable_atm_data']}  ({stats['usable_atm_data']/max(1,within_window):.1%} of in-window events)")
    print(f"  -> passed MIN_MOVE/MAX_MOVE_BOUNDS filters                      : {stats['passed_move_filter']}  ({stats['passed_move_filter']/max(1,stats['usable_atm_data']):.1%} of usable-EM events)")
    print(f"  -> wing strike not listed (skip, no_real_data)                  : {stats['no_real_data_wing_strike']}")
    print(f"  -> wing leg had no real trade that day (skip)                   : {stats['no_real_data_wing_trade']}")
    print(f"  = FULLY PRICED real condors (status=ok)                         : {stats['ok']}")
    if stats["passed_move_filter"] > 0:
        print(f"  real-data coverage GIVEN filter pass (ok / passed)              : {stats['ok']/stats['passed_move_filter']:.1%}")
    print(f"  real-data coverage of in-window events (ok / within_window)     : {stats['ok']/max(1,within_window):.1%}")
    print(f"  overall coverage vs full 5y original window (ok / total_events) : {stats['ok']/max(1,te):.1%}")

    coverage_df = pd.DataFrame([stats])
    coverage_df.to_csv("evc_condor_backtest_real_data_coverage.csv", index=False)

    # -------------------------------------------------------------------
    # Aggregate result by max_move_bound (no iv_mult axis -- real is real)
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS (REAL DATA)")
    print("=" * 100)

    ok = df[df["status"] == "ok"].copy()
    summary_rows = []
    for bound in MAX_MOVE_BOUNDS:
        qcol = "qualifies_12" if bound == 0.12 else "qualifies_25"
        sub = ok[(ok[qcol] == True) & (ok["im_pct"] >= MIN_MOVE)]  # noqa: E712
        n = len(sub)
        if n == 0:
            summary_rows.append(dict(max_move_bound=bound, n=0))
            continue
        win_rate = sub["win"].mean()
        avg_payoff_pct_spot = sub["payoff_pct_spot"].mean()
        avg_payoff_pct_credit = sub["payoff_pct_credit"].mean()
        sum_risk = sub["max_risk"].sum()
        sum_credit = sub["net_credit"].sum()
        pooled_breakeven = sum_risk / (sum_risk + sum_credit)
        mean_rr = (sub["max_risk"] / sub["net_credit"]).mean()
        avg_days_to_expiry = sub["days_to_expiry"].mean()
        summary_rows.append(dict(
            max_move_bound=bound, n=n,
            win_rate=win_rate, avg_payoff_pct_spot=avg_payoff_pct_spot,
            avg_payoff_pct_credit=avg_payoff_pct_credit,
            pooled_breakeven_win_rate=pooled_breakeven,
            mean_risk_reward_ratio=mean_rr,
            margin_vs_breakeven=win_rate - pooled_breakeven,
            avg_days_to_expiry=avg_days_to_expiry,
        ))
        print(f"\nmax_move_bound={bound:.0%}  N={n}  (avg real days-to-expiry={avg_days_to_expiry:.1f})")
        print(f"  win_rate                = {win_rate:.1%}")
        print(f"  pooled breakeven needed  = {pooled_breakeven:.1%}  (mean R:R = {mean_rr:.2f}:1)")
        print(f"  margin vs breakeven      = {(win_rate - pooled_breakeven)*100:+.1f} pts")
        print(f"  avg payoff (% of spot)   = {avg_payoff_pct_spot:+.3%}")
        print(f"  avg payoff (% of credit) = {avg_payoff_pct_credit:+.1%}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("evc_condor_backtest_real_data_summary.csv", index=False)

    # -------------------------------------------------------------------
    # Per-ticker breakdown (N >= 8 in at least one scenario)
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PER-TICKER BREAKDOWN (tickers with >=8 qualifying real-data trades)")
    print("=" * 100)
    per_ticker_records = []
    for bound in MAX_MOVE_BOUNDS:
        qcol = "qualifies_12" if bound == 0.12 else "qualifies_25"
        sub = ok[(ok[qcol] == True) & (ok["im_pct"] >= MIN_MOVE)]  # noqa: E712
        if len(sub) == 0:
            continue
        grp = sub.groupby("ticker").agg(n=("win", "size"), win_rate=("win", "mean"))
        grp["max_move_bound"] = bound
        per_ticker_records.append(grp.reset_index())
    if per_ticker_records:
        per_ticker_df = pd.concat(per_ticker_records, ignore_index=True)
    else:
        per_ticker_df = pd.DataFrame(columns=["ticker", "n", "win_rate", "max_move_bound"])
    per_ticker_df.to_csv("evc_condor_backtest_real_data_per_ticker.csv", index=False)

    if len(per_ticker_df) > 0:
        eligible_tickers = per_ticker_df[per_ticker_df["n"] >= 8]["ticker"].unique()
        pivot = per_ticker_df[per_ticker_df["ticker"].isin(eligible_tickers)].pivot_table(
            index="ticker", columns=["max_move_bound"], values=["n", "win_rate"]
        )
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(pivot)
    else:
        print("  (no tickers with usable real-data trades)")

    print("\nDone.")


if __name__ == "__main__":
    main()
