"""
EVC iron-condor backtest of the CURRENT LIVE strike/wing-selection mechanism
-- real Polygon.io options closing-price data, 2026-09-01.

Direct response to a real gap found this session: the existing real-data
backtest (evc_condor_backtest_real_data.py, run 2026-08-21) modeled the
OLD mechanism (short = spot +/- 1.0x expected-move, flat wing = spot +/-
1.5x EM). Since then, TWO real live changes were made that were never
backtested:
  1. 2026-08-26: put_cushion_mult/call_cushion_mult live-configured to 1.3
     (was 1.0) -- short strikes pushed further OTM.
  2. 2026-08-27 (CEO instruction, the MRVL/AFRM/ULTA night): wing (long)
     strike selection changed from a flat 1.5x-EM target to walking every
     REAL strike between the short strike and the 1.5x fallback, tightest
     first, taking the first one whose per-side credit is still positive --
     cut max_risk 79-91% on that night's live entries vs. the old flat rule.

This script replicates #1 and #2 using the SAME real, proven Polygon
pipeline as evc_condor_backtest_real_data.py (same universe, same
earnings-date/entry-date logic, same MIN_MOVE/MAX_MOVE_BOUNDS filters, same
payoff formula -- copied verbatim, not re-derived, so any difference in
results comes only from the strike-selection mechanism itself).

HONEST, DISCLOSED GAP: the live wing-selection logic and the separate
2026-08-30 liquidity gate both use CONSERVATIVE (bid/ask-based worst-fill)
credit, not mid/close price. Checked directly before writing this script:
this Polygon.io plan does not include historical quotes/NBBO data (real
API test against /v3/quotes returned 403 NOT_AUTHORIZED, confirmed
2026-09-01). There is no real bid/ask history available to this account for
this test. This script uses real CLOSING TRADE prices as the best
available proxy for the per-side credit check during the wing walk -- this
is a genuine, real approximation, not the exact live mechanism, and is
reported as such. The liquidity gate itself (reject if conservative_credit
<= 0 or < 30% of mid credit) is NOT modeled at all here; it remains
real-money-untested by any backtest, full stop.

Outputs (same directory):
  evc_condor_backtest_current_mechanism_rows.csv
  evc_condor_backtest_current_mechanism_summary.csv
"""
from __future__ import annotations

import json
import math
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

# ---------------------------------------------------------------------------
# IDENTICAL to evc_condor_backtest_real_data.py -- universe, filters, helpers
# ---------------------------------------------------------------------------
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

MIN_MOVE = 0.03
MAX_MOVE_BOUNDS = [0.12, 0.25]

# The two real mechanism changes under test -- current live values,
# confirmed 2026-09-01 via GET /earnings-vol-crush/status.
PUT_CUSHION_MULT = 1.3
CALL_CUSHION_MULT = 1.3
WING_MULT_FALLBACK = 1.5   # unchanged -- still the outer bound if no tighter strike qualifies

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
    spot: float = np.nan
    expected_move: float = np.nan
    im_pct: float = np.nan
    # OLD mechanism (1.0x short / flat 1.5x wing) -- for direct side-by-side comparison
    old_short_put: float = np.nan
    old_long_put: float = np.nan
    old_short_call: float = np.nan
    old_long_call: float = np.nan
    old_net_credit: float = np.nan
    old_max_risk: float = np.nan
    old_payoff: float = np.nan
    old_win: object = None
    # NEW mechanism (1.3x short / tight-walk wing)
    new_short_put: float = np.nan
    new_long_put: float = np.nan
    new_short_call: float = np.nan
    new_long_call: float = np.nan
    new_put_tightened: bool = False
    new_call_tightened: bool = False
    new_net_credit: float = np.nan
    new_max_risk: float = np.nan
    new_payoff: float = np.nan
    new_win: object = None
    real_price: float = np.nan
    real_move_pct: float = np.nan
    qualifies_12: bool = False
    qualifies_25: bool = False


def payoff_formula(short_put, long_put, short_call, long_call, net_credit, real_price):
    put_width = short_put - long_put
    call_width = long_call - short_call
    if short_put <= real_price <= short_call:
        return net_credit
    elif real_price < short_put:
        breach = short_put - real_price
        if real_price >= long_put:
            return net_credit - breach
        return net_credit - (put_width - net_credit)
    else:
        breach = real_price - short_call
        if real_price <= long_call:
            return net_credit - breach
        return net_credit - (call_width - net_credit)


def wing_walk(short_strike, fallback_strike, side_map, side_short_price, is_put: bool):
    """Real replica of _evc_quote_condor's tight-wing walk (main.py:16267-
    16300): candidates are every REAL listed strike strictly between the
    fallback and the short strike, walked TIGHTEST (closest to short) first.
    Uses real closing-price credit as the best available proxy for the
    live conservative (bid/ask) credit check -- see module docstring for
    why real bid/ask isn't available. Returns (chosen_strike, chosen_price,
    tightened: bool) -- tightened=False means it fell through to fallback,
    matching the live code's own guarantee of never being wider than before.
    """
    # Matches main.py:16288-16290 exactly: union with the fallback via <=/>=
    # so a degenerate case (fallback rounds to the same or an inverted
    # strike vs the short target, e.g. a high-priced stock with a modest
    # expected move) still yields at least one real candidate instead of
    # an empty range.
    if is_put:
        candidates = sorted({s for s in side_map if fallback_strike <= s < short_strike} | {fallback_strike}, reverse=True)
    else:
        candidates = sorted({s for s in side_map if short_strike < s <= fallback_strike} | {fallback_strike})
    for cand in candidates:
        occ = side_map.get(cand)
        if not occ:
            continue
        px = get_daily_close(occ, wing_walk.entry_date_str)
        if px is None:
            continue
        side_credit = side_short_price - px
        if side_credit > 0:
            return cand, px, True
    # fallback
    occ = side_map.get(fallback_strike)
    if not occ:
        return None, None, False
    px = get_daily_close(occ, wing_walk.entry_date_str)
    return (fallback_strike, px, False) if px is not None else (None, None, False)


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

        atm_strike = round_strike(spot)
        atm_call_occ = call_map.get(atm_strike)
        atm_put_occ = put_map.get(atm_strike)
        if atm_call_occ is None or atm_put_occ is None:
            stats["no_atm_contract"] += 1
            rows.append(Row(status="skip", reason="no_atm_contract", expiry_used=expiry_used, spot=spot,
                             real_price=real_price, real_move_pct=real_move_pct, **base_kwargs))
            continue

        atm_closes = get_closes_concurrent({"call": atm_call_occ, "put": atm_put_occ}, entry_date_str)
        atm_call_price, atm_put_price = atm_closes["call"], atm_closes["put"]
        if atm_call_price is None or atm_put_price is None:
            stats["no_trade_atm_leg"] += 1
            rows.append(Row(status="skip", reason="no_trade_atm_leg", expiry_used=expiry_used, spot=spot,
                             real_price=real_price, real_move_pct=real_move_pct, **base_kwargs))
            continue

        expected_move = atm_call_price + atm_put_price
        im_pct = expected_move / spot if spot > 0 else np.nan
        stats["usable_atm_data"] += 1

        r = Row(status="ok", reason="", expiry_used=expiry_used, spot=spot, expected_move=expected_move,
                im_pct=im_pct, real_price=real_price, real_move_pct=real_move_pct, **base_kwargs)

        if not math.isfinite(im_pct) or im_pct < MIN_MOVE:
            r.status, r.reason = "reject", "below_min_move"
            rows.append(r)
            continue
        if im_pct > max(MAX_MOVE_BOUNDS):
            r.status, r.reason = "reject", "above_max_move_bound"
            rows.append(r)
            continue
        stats["passed_move_filter"] += 1

        # ---------------- OLD mechanism: 1.0x short, flat 1.5x wing ----------------
        old_sp = round_strike(spot - expected_move)
        old_sc = round_strike(spot + expected_move)
        old_lp = round_strike(spot - WING_MULT_FALLBACK * expected_move)
        old_lc = round_strike(spot + WING_MULT_FALLBACK * expected_move)

        # ---------------- NEW mechanism: 1.3x short, tight-walk wing ----------------
        new_sp_target = round_strike(spot - PUT_CUSHION_MULT * expected_move)
        new_sc_target = round_strike(spot + CALL_CUSHION_MULT * expected_move)
        fallback_lp = round_strike(spot - WING_MULT_FALLBACK * expected_move)
        fallback_lc = round_strike(spot + WING_MULT_FALLBACK * expected_move)

        # Real edge case found 2026-09-01 (AAPL smoke test): for a high-priced
        # stock with a modest expected move, round_strike()'s $5 step can
        # round the 1.3x and 1.5x EM targets to the SAME strike -- the live
        # code (main.py:16264-16290) never requires fallback strictly beyond
        # the short target either; it just unions the fallback into the
        # candidate set via <=. Only the OLD mechanism's already-proven
        # ordering and the basic short-put-below-short-call sanity check are
        # real requirements here.
        if not (old_lp < old_sp < old_sc < old_lc) or not (new_sp_target < new_sc_target):
            r.status, r.reason = "skip", "invalid_strikes"
            rows.append(r)
            continue

        missing = []
        for label, strike, m in (("old_sp", old_sp, put_map), ("old_lp", old_lp, put_map),
                                  ("old_sc", old_sc, call_map), ("old_lc", old_lc, call_map),
                                  ("new_sp", new_sp_target, put_map), ("new_sc", new_sc_target, call_map),
                                  ("fallback_lp", fallback_lp, put_map), ("fallback_lc", fallback_lc, call_map)):
            if strike not in m:
                missing.append(label)
        if missing:
            stats["no_real_data_wing_strike"] += 1
            r.status, r.reason = "skip", f"no_real_data:missing_strike:{','.join(missing)}"
            rows.append(r)
            continue

        # Real closes for the OLD mechanism's 4 fixed legs
        old_occ = {"sp": put_map[old_sp], "lp": put_map[old_lp], "sc": call_map[old_sc], "lc": call_map[old_lc]}
        old_closes = get_closes_concurrent(old_occ, entry_date_str)
        if any(v is None for v in old_closes.values()):
            stats["no_real_data_wing_trade"] += 1
            r.status, r.reason = "skip", "no_real_data:no_trade_old_leg"
            rows.append(r)
            continue

        # Real closes for the NEW mechanism's short legs
        new_short_occ = {"sp": put_map[new_sp_target], "sc": call_map[new_sc_target]}
        new_short_closes = get_closes_concurrent(new_short_occ, entry_date_str)
        if any(v is None for v in new_short_closes.values()):
            stats["no_real_data_wing_trade"] += 1
            r.status, r.reason = "skip", "no_real_data:no_trade_new_short_leg"
            rows.append(r)
            continue

        wing_walk.entry_date_str = entry_date_str
        new_lp, new_lp_price, put_tightened = wing_walk(
            new_sp_target, fallback_lp, put_map, new_short_closes["sp"], is_put=True)
        new_lc, new_lc_price, call_tightened = wing_walk(
            new_sc_target, fallback_lc, call_map, new_short_closes["sc"], is_put=False)
        if new_lp is None or new_lc is None or new_lp_price is None or new_lc_price is None:
            stats["no_real_data_wing_trade"] += 1
            r.status, r.reason = "skip", "no_real_data:wing_walk_failed"
            rows.append(r)
            continue

        old_net_credit = round((old_closes["sp"] + old_closes["sc"]) - (old_closes["lp"] + old_closes["lc"]), 4)
        new_net_credit = round((new_short_closes["sp"] + new_short_closes["sc"]) - (new_lp_price + new_lc_price), 4)

        if old_net_credit <= 0 or new_net_credit <= 0:
            r.status, r.reason = "skip", "non_positive_credit"
            rows.append(r)
            continue

        old_put_width, old_call_width = old_sp - old_lp, old_lc - old_sc
        old_wing_width = max(old_put_width, old_call_width)
        old_max_risk = old_wing_width - old_net_credit

        new_put_width, new_call_width = new_sp_target - new_lp, new_lc - new_sc_target
        new_wing_width = max(new_put_width, new_call_width)
        new_max_risk = new_wing_width - new_net_credit

        if old_max_risk <= 0 or new_max_risk <= 0:
            r.status, r.reason = "skip", "non_positive_risk"
            rows.append(r)
            continue

        old_payoff = payoff_formula(old_sp, old_lp, old_sc, old_lc, old_net_credit, real_price)
        new_payoff = payoff_formula(new_sp_target, new_lp, new_sc_target, new_lc, new_net_credit, real_price)

        r.old_short_put, r.old_long_put, r.old_short_call, r.old_long_call = old_sp, old_lp, old_sc, old_lc
        r.old_net_credit, r.old_max_risk, r.old_payoff, r.old_win = old_net_credit, old_max_risk, old_payoff, bool(old_payoff > 0)
        r.new_short_put, r.new_long_put, r.new_short_call, r.new_long_call = new_sp_target, new_lp, new_sc_target, new_lc
        r.new_put_tightened, r.new_call_tightened = put_tightened, call_tightened
        r.new_net_credit, r.new_max_risk, r.new_payoff, r.new_win = new_net_credit, new_max_risk, new_payoff, bool(new_payoff > 0)
        r.qualifies_12 = im_pct <= 0.12
        r.qualifies_25 = im_pct <= 0.25
        rows.append(r)
        n_ok += 1
        stats["ok"] += 1

    print(f"  {ticker}: {n_ok} fully priced with real data (both mechanisms)", flush=True)


def main():
    rows: list = []
    stats = dict(total_events=0, outside_plan_window=0, no_contracts_listed=0, no_atm_contract=0,
                 no_trade_atm_leg=0, usable_atm_data=0, passed_move_filter=0,
                 no_real_data_wing_strike=0, no_real_data_wing_trade=0, ok=0)
    print(f"Processing {len(UNIVERSE)} tickers -- CURRENT mechanism (1.3x cushion + tight wing) vs OLD (1.0x/flat 1.5x)")
    for i, ticker in enumerate(UNIVERSE, 1):
        print(f"[{i}/{len(UNIVERSE)}] {ticker}", flush=True)
        try:
            process_ticker(ticker, rows, stats)
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}", flush=True)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv("evc_condor_backtest_current_mechanism_rows.csv", index=False)
    print(f"\nSaved {len(df)} rows")

    ok = df[df.status == "ok"].copy()
    print(f"\nFully priced (both mechanisms, real data): n={len(ok)}")

    summary_rows = []
    for bound in MAX_MOVE_BOUNDS:
        qcol = "qualifies_12" if bound == 0.12 else "qualifies_25"
        sub = ok[(ok[qcol] == True) & (ok.im_pct >= MIN_MOVE)]
        if len(sub) == 0:
            continue
        for label, win_col, credit_col, risk_col, payoff_col in [
            ("OLD (1.0x short, flat 1.5x wing)", "old_win", "old_net_credit", "old_max_risk", "old_payoff"),
            ("NEW (1.3x short, tight wing)", "new_win", "new_net_credit", "new_max_risk", "new_payoff"),
        ]:
            n = len(sub)
            win_rate = sub[win_col].mean()
            sum_risk = sub[risk_col].sum()
            sum_credit = sub[credit_col].sum()
            pooled_breakeven = sum_risk / (sum_risk + sum_credit)
            avg_payoff_pct_credit = (sub[payoff_col] / sub[credit_col]).mean()
            med_payoff_pct_credit = (sub[payoff_col] / sub[credit_col]).median()
            print(f"\nmax_move_bound={bound:.0%}  {label}  N={n}")
            print(f"  win_rate={win_rate:.1%}  pooled_breakeven={pooled_breakeven:.1%}  margin={(win_rate-pooled_breakeven)*100:+.1f}pts")
            print(f"  avg_payoff/credit={avg_payoff_pct_credit:+.1%}  median={med_payoff_pct_credit:+.1%}")
            print(f"  avg_max_risk=${sub[risk_col].mean()*100:,.0f}  avg_net_credit=${sub[credit_col].mean()*100:,.0f}")
            summary_rows.append(dict(max_move_bound=bound, mechanism=label, n=n, win_rate=win_rate,
                                      pooled_breakeven=pooled_breakeven, margin=win_rate - pooled_breakeven,
                                      avg_payoff_pct_credit=avg_payoff_pct_credit,
                                      median_payoff_pct_credit=med_payoff_pct_credit,
                                      avg_max_risk_dollars=sub[risk_col].mean() * 100,
                                      avg_net_credit_dollars=sub[credit_col].mean() * 100))

    pd.DataFrame(summary_rows).to_csv("evc_condor_backtest_current_mechanism_summary.csv", index=False)

    print(f"\nPut wing tightened (vs flat 1.5x fallback): {ok.new_put_tightened.mean()*100:.0f}% of events")
    print(f"Call wing tightened (vs flat 1.5x fallback): {ok.new_call_tightened.mean()*100:.0f}% of events")
    print("\nDone.")


if __name__ == "__main__":
    main()
