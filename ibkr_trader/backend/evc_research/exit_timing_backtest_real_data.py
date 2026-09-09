"""
Larger, REAL-DATA version of exit_timing_backtest.py -- CEO instruction
2026-08-28: "run larger test using actual data not black scholes."

Three real data sources, one per stage:
  1. yfinance    -- real earnings dates + daily close history (entry/outcome
                    date logic, split history for price un-adjustment).
  2. Polygon.io  -- REAL listed option contracts and REAL closing prices on
                    entry_date, for the ATM straddle (expected move) and all
                    4 condor legs (net_credit) -- no Black-Scholes anywhere
                    in this version. Reuses evc_condor_backtest_real_data.py's
                    own contract-grid/pricing functions verbatim.
  3. IBKR        -- REAL 1-minute bars for the outcome_date, to price the
                    payoff at 8 candidate exit times instead of just the
                    daily close (the only thing any prior EVC backtest ever
                    measured against).

Strikes use the account's REAL LIVE cushions as of 2026-08-27
(put_cushion_mult=1.3, call_cushion_mult=1.0, wing_mult=1.5) -- adapted from
evc_condor_backtest_real_data.py's symmetric 1.0/1.0, since the point is to
validate the exit timing for the structure actually traded today.

Sampled at every 2nd ticker (vs. every 3rd in the first pass) for a larger,
more defensible sample size, per the CEO's explicit ask.
"""
import asyncio
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import json
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from ib_insync import IB, Stock

# ── Config -- matches this account's REAL live EVC config as of 2026-08-27 ──
PUT_CUSHION_MULT = 1.3
CALL_CUSHION_MULT = 1.0
WING_MULT = 1.5
MIN_MOVE = 0.03
MAX_MOVE_BOUND = 0.12
IBKR_MINUTE_BAR_CUTOFF = date.today() - timedelta(days=270)
EXIT_TIMES = ["09:30", "09:35", "09:45", "10:00", "10:30", "11:00", "12:00", "close"]
CLIENT_ID = 986

MAX_RETRIES = 4
REQUEST_TIMEOUT = 15
AGGS_WORKERS = 6
MAX_EXPIRY_SEARCH_DAYS = 10
PLAN_AGGS_LOOKBACK_DAYS = 730
PLAN_BOUNDARY_SAFETY_DAYS = 5

POLY_BASE = "https://api.polygon.io"
with open("../scanner_config.json") as f:
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


def _polygon_get(path: str, params: dict):
    params = dict(params)
    params["apiKey"] = POLYGON_API_KEY
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            resp = _session.get(f"{POLY_BASE}{path}", params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(delay); delay = min(delay * 1.7, 20.0); continue
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                return None
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", delay))
            time.sleep(wait); delay = min(delay * 1.7, 20.0); continue
        if 500 <= resp.status_code < 600:
            time.sleep(delay); delay = min(delay * 1.7, 20.0); continue
        return None
    return None


def get_contract_grid(ticker: str, entry_date: date):
    data = _polygon_get("/v3/reference/options/contracts", {
        "underlying_ticker": ticker,
        "expiration_date.gte": entry_date.isoformat(),
        "expiration_date.lte": (pd.Timestamp(entry_date) + pd.Timedelta(days=MAX_EXPIRY_SEARCH_DAYS)).date().isoformat(),
        "as_of": entry_date.isoformat(), "order": "asc", "sort": "expiration_date", "limit": 1000,
    })
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
        (call_map if c.get("contract_type") == "call" else put_map)[strike] = occ
    return expiry_used, call_map, put_map


def get_daily_close(occ_ticker: str, date_str: str):
    data = _polygon_get(f"/v2/aggs/ticker/{occ_ticker}/range/1/day/{date_str}/{date_str}", {"adjusted": "true"})
    if not data or data.get("resultsCount", 0) == 0:
        return None
    results = data.get("results") or []
    return float(results[0]["c"]) if results and results[0].get("c") is not None else None


def get_closes_concurrent(occ_tickers: dict, date_str: str) -> dict:
    names = list(occ_tickers.keys())
    with ThreadPoolExecutor(max_workers=min(AGGS_WORKERS, max(1, len(names)))) as ex:
        results = list(ex.map(lambda n: get_daily_close(occ_tickers[n], date_str), names))
    return dict(zip(names, results))


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


def payoff_at_price(price, short_put, long_put, short_call, long_call, net_credit, put_width, call_width):
    if short_put <= price <= short_call:
        return net_credit
    elif price < short_put:
        breach = short_put - price
        return net_credit - breach if price >= long_put else net_credit - (put_width - net_credit)
    else:
        breach = price - short_call
        return net_credit - breach if price <= long_call else net_credit - (call_width - net_credit)


@dataclass
class Event:
    ticker: str
    earnings_date: str
    timing: str
    entry_date: date
    outcome_date: date
    spot: float
    expected_move: float
    short_put: float
    long_put: float
    short_call: float
    long_call: float
    net_credit: float
    put_width: float
    call_width: float
    prices: dict = field(default_factory=dict)
    payoffs: dict = field(default_factory=dict)


def fetch_ticker_data(ticker, retries=3):
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
            edf = t.get_earnings_dates(limit=8)
            if edf is None or edf.empty:
                raise ValueError("no earnings dates")
            splits = t.splits
            if splits is not None and not splits.empty:
                splits = splits.copy()
                splits.index = pd.to_datetime(splits.index).tz_localize(None).normalize()
            return hist, edf, splits
        except Exception as e:
            last_err = e
            time.sleep(1.2 * (attempt + 1))
    print(f"  [SKIP TICKER] {ticker}: {last_err}")
    return None, None, None


def build_candidates_real(tickers: list[str]) -> tuple[list[Event], dict]:
    events = []
    stats = {"total_events": 0, "outside_plan_window": 0, "no_contracts_listed": 0,
              "no_atm_contract": 0, "no_trade_atm_leg": 0, "passed_move_filter": 0,
              "missing_wing_contract": 0, "no_trade_wing_leg": 0, "non_positive_credit": 0, "ok": 0}
    plan_cutoff = date.today() - timedelta(days=PLAN_AGGS_LOOKBACK_DAYS + PLAN_BOUNDARY_SAFETY_DAYS)

    for ti, ticker in enumerate(tickers):
        hist, edf, splits = fetch_ticker_data(ticker)
        if hist is None:
            continue
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
        for ts, erow in edf.iterrows():
            edate = pd.Timestamp(ts)
            edate_local = edate.date()
            if edate_local >= today:
                continue
            if pd.isna(erow.get("Reported EPS", np.nan)):
                continue
            timing = "BMO" if edate.hour < 12 else "AMC"
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
            if outcome_date is None or outcome_date < IBKR_MINUTE_BAR_CUTOFF:
                continue
            if entry_date < plan_cutoff:
                stats["outside_plan_window"] += 1
                continue

            entry_idx, outcome_idx = td_index[entry_date], td_index[outcome_date]
            spot_adj, real_price_adj = float(closes[entry_idx]), float(closes[outcome_idx])
            if spot_adj <= 0 or real_price_adj <= 0:
                continue
            spot = spot_adj * unadjust(entry_date)
            atm_strike = round_strike(spot)
            stats["total_events"] += 1

            expiry_used, call_map, put_map = get_contract_grid(ticker, entry_date)
            if expiry_used is None:
                stats["no_contracts_listed"] += 1
                continue
            atm_call_occ, atm_put_occ = call_map.get(atm_strike), put_map.get(atm_strike)
            if atm_call_occ is None or atm_put_occ is None:
                stats["no_atm_contract"] += 1
                continue
            entry_date_str = entry_date.isoformat()
            atm_closes = get_closes_concurrent({"call": atm_call_occ, "put": atm_put_occ}, entry_date_str)
            atm_call_price, atm_put_price = atm_closes["call"], atm_closes["put"]
            if atm_call_price is None or atm_put_price is None:
                stats["no_trade_atm_leg"] += 1
                continue

            expected_move = atm_call_price + atm_put_price
            im_pct = expected_move / spot if spot > 0 else np.nan
            if not math.isfinite(im_pct) or im_pct < MIN_MOVE or im_pct > MAX_MOVE_BOUND:
                continue
            stats["passed_move_filter"] += 1

            short_put = round_strike(spot - PUT_CUSHION_MULT * expected_move)
            short_call = round_strike(spot + CALL_CUSHION_MULT * expected_move)
            long_put = round_strike(spot - WING_MULT * expected_move)
            long_call = round_strike(spot + WING_MULT * expected_move)
            if not (long_put < short_put < short_call < long_call):
                continue

            missing = [name for name, strike, m in [
                ("short_put", short_put, put_map), ("long_put", long_put, put_map),
                ("short_call", short_call, call_map), ("long_call", long_call, call_map),
            ] if strike not in m]
            if missing:
                stats["missing_wing_contract"] += 1
                continue

            wing_occ = {
                "short_put": put_map[short_put], "long_put": put_map[long_put],
                "short_call": call_map[short_call], "long_call": call_map[long_call],
            }
            wing_closes = get_closes_concurrent(wing_occ, entry_date_str)
            if any(v is None for v in wing_closes.values()):
                stats["no_trade_wing_leg"] += 1
                continue

            net_credit = round((wing_closes["short_put"] + wing_closes["short_call"])
                                - (wing_closes["long_put"] + wing_closes["long_call"]), 4)
            put_width, call_width = short_put - long_put, long_call - short_call
            if net_credit <= 0 or max(put_width, call_width) - net_credit <= 0:
                stats["non_positive_credit"] += 1
                continue

            stats["ok"] += 1
            events.append(Event(
                ticker=ticker, earnings_date=str(edate_local), timing=timing,
                entry_date=entry_date, outcome_date=outcome_date, spot=spot,
                expected_move=expected_move,
                short_put=short_put, long_put=long_put, short_call=short_call, long_call=long_call,
                net_credit=net_credit, put_width=put_width, call_width=call_width,
            ))
        if (ti + 1) % 20 == 0:
            print(f"  ... {ti+1}/{len(tickers)} tickers processed, {len(events)} real events so far")
    return events, stats


async def fill_real_prices(events: list[Event]) -> None:
    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=CLIENT_ID, timeout=15)
    ok, failed = 0, 0
    for i, ev in enumerate(events):
        try:
            c = Stock(ev.ticker, "SMART", "USD")
            await ib.qualifyContractsAsync(c)
            end_dt = f"{ev.outcome_date.strftime('%Y%m%d')} 16:00:00 US/Eastern"
            bars = await ib.reqHistoricalDataAsync(
                c, endDateTime=end_dt, durationStr="1 D", barSizeSetting="1 min",
                whatToShow="TRADES", useRTH=True, keepUpToDate=False,
            )
            if not bars or len(bars) < 300:
                failed += 1
                continue
            by_time = {b.date.strftime("%H:%M"): b for b in bars}

            def price_at(hhmm):
                if hhmm in by_time:
                    return float(by_time[hhmm].open)
                target = datetime.strptime(hhmm, "%H:%M").time()
                cands = [b for b in bars if b.date.time() >= target]
                return float(cands[0].open) if cands else None

            ev.prices["09:30"] = float(bars[0].open)
            for hhmm in ["09:35", "09:45", "10:00", "10:30", "11:00", "12:00"]:
                ev.prices[hhmm] = price_at(hhmm)
            ev.prices["close"] = float(bars[-1].close)
            for exit_t, px in ev.prices.items():
                if px is None:
                    continue
                ev.payoffs[exit_t] = payoff_at_price(
                    px, ev.short_put, ev.long_put, ev.short_call, ev.long_call,
                    ev.net_credit, ev.put_width, ev.call_width)
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  [SKIP BARS] {ev.ticker} {ev.outcome_date}: {exc}")
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(events)} processed (ok={ok} failed={failed})")
        await asyncio.sleep(1.2)
    ib.disconnect()
    print(f"Real minute-bar fetch done: {ok} ok, {failed} failed/skipped")


def summarize(events: list[Event]) -> pd.DataFrame:
    rows = []
    for exit_t in EXIT_TIMES:
        payoffs = [ev.payoffs[exit_t] for ev in events if exit_t in ev.payoffs]
        credits = [ev.net_credit for ev in events if exit_t in ev.payoffs]
        spots = [ev.spot for ev in events if exit_t in ev.payoffs]
        if not payoffs:
            rows.append({"exit_time": exit_t, "n": 0})
            continue
        wins = [p for p in payoffs if p > 0]
        pct_credit = [p / c for p, c in zip(payoffs, credits) if c]
        pct_spot = [p / s for p, s in zip(payoffs, spots) if s]
        rows.append({
            "exit_time": exit_t, "n": len(payoffs),
            "win_rate_pct": round(len(wins) / len(payoffs) * 100, 1),
            "avg_payoff": round(sum(payoffs) / len(payoffs), 4),
            "total_payoff": round(sum(payoffs), 2),
            "avg_payoff_pct_credit": round(sum(pct_credit) / len(pct_credit) * 100, 1) if pct_credit else None,
            "avg_payoff_pct_spot": round(sum(pct_spot) / len(pct_spot) * 100, 3) if pct_spot else None,
            "worst": round(min(payoffs), 2), "best": round(max(payoffs), 2),
        })
    return pd.DataFrame(rows)


def main():
    main_src = open("../main.py", encoding="utf-8").read()
    m = re.search(r'CANDIDATE_POOL: List\[str\] = \[(.*?)\]\n', main_src, re.DOTALL)
    all_tickers = re.findall(r'"([A-Z\.]+)"', m.group(1))
    tickers = all_tickers[::2]   # every 2nd ticker -- larger sample than the first pass's every-3rd
    print(f"Universe: {len(all_tickers)} total, sampling {len(tickers)} (every 2nd)")

    print("Building candidate events with REAL Polygon option prices (no Black-Scholes)...")
    events, stats = build_candidates_real(tickers)
    print(f"\nPipeline funnel: {json.dumps(stats, indent=2)}")
    print(f"\n{len(events)} qualifying real-priced events within IBKR's minute-bar window "
          f"(outcome_date >= {IBKR_MINUTE_BAR_CUTOFF})")

    if not events:
        print("No events -- nothing to backtest.")
        return

    print("Fetching REAL IBKR 1-minute bars for each event's outcome date...")
    asyncio.run(fill_real_prices(events))

    df_events = pd.DataFrame([{
        "ticker": e.ticker, "earnings_date": e.earnings_date, "timing": e.timing,
        "outcome_date": str(e.outcome_date), "spot": e.spot, "expected_move": e.expected_move,
        "net_credit": e.net_credit, "short_put": e.short_put, "long_put": e.long_put,
        "short_call": e.short_call, "long_call": e.long_call,
        **{f"price_{k}": v for k, v in e.prices.items()},
        **{f"payoff_{k}": v for k, v in e.payoffs.items()},
    } for e in events])
    df_events.to_csv("exit_timing_backtest_real_data_events.csv", index=False)

    summary = summarize(events)
    summary.to_csv("exit_timing_backtest_real_data_summary.csv", index=False)
    print("\n=== EVC exit-timing backtest: REAL option prices + real IBKR 1-min bars ===")
    print(summary.to_string(index=False))
    print("\nSaved exit_timing_backtest_real_data_events.csv and exit_timing_backtest_real_data_summary.csv")


if __name__ == "__main__":
    main()
