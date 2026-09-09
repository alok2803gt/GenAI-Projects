"""
EVC-adjacent jade lizard historical backtest -- REAL Polygon.io options-chain
data version. Companion to evc_condor_backtest_real_data.py (read that file's
docstring for the two real bugs it already solved -- split-unadjustment for
tickers like NFLX, and the ~2yr rolling coverage window on this Polygon
plan's options AGGREGATES endpoint -- both are reused verbatim here).

Same 54-ticker/5-year real earnings-event universe, same entry/outcome-date
logic, same round_strike() convention, same real listed-contract lookup +
real closing-price pattern as the condor script. The ONLY thing that changes
is the STRUCTURE:

  Condor (reference):  short put (1.0x EM) + long put (1.5x EM)   [downside, capped]
                        short call (1.0x EM) + long call (1.5x EM) [upside, capped]

  Jade lizard (this):  short put (1.0x EM), NO LONG PUT            [downside, UNCAPPED
                                                                      down to ~0, same
                                                                      risk shape as a
                                                                      naked/cash-secured
                                                                      short put]
                        short call (1.0x EM) + long call (1.5x EM) [upside, capped --
                                                                      normal credit spread]

A jade lizard is "true" (upside-risk-free by construction) only when the
TOTAL credit received (put credit + call-spread credit) is >= the call
spread's width -- then even a max-adverse rally past the long call can never
produce a net loss, because the excess put credit absorbs the capped call
loss. This is a real-data question, not a design choice we can force: each
historical event's real premiums either satisfy that inequality or they
don't. Both outcomes are tracked (`is_true_jade_lizard` column) and reported
separately.

Payoff scoring (this is the important correctness point vs. a symmetric
condor): the put side has NO wing, so a real breach scales the loss
continuously down toward real_price -> 0 (bounded only by the stock itself
being worth nothing), not capped at any wing width. The call side is a
normal capped credit spread. Because true unbounded-to-zero risk cannot be
expressed as a single "max_risk" breakeven number the way a symmetric
condor's max_risk/(max_risk+credit) can, this script additionally computes a
PRACTICAL max-risk bound per ticker: that ticker's own largest real
historical single-earnings-event drawdown (from the full 5y yfinance
history, independent of Polygon's narrower coverage window, so the tail
estimate uses the most real data available) applied to that event's spot
price. This is reported explicitly as a practical worst-case proxy, NOT a
claim that true max loss is bounded -- true max loss on a naked short put is
technically unbounded down to zero.

Capital feasibility (this account is real money, ~$4.2k combined net liq,
5% per-strategy cap): the short put leg needs cash-secured collateral
(strike x 100) absent portfolio margin, which this account does not have on
Alpaca (options Level 3, not margin-approved). main() pulls live account
numbers from the running backend (GET /account/summary, GET
/alpaca/positions) and checks, for every real short-put strike this
backtest actually selected, what fraction would need MORE collateral than
the account's entire combined net liq, and what fraction exceeds the 5%
per-strategy cap. Reported explicitly, not buried under the win-rate math.

Every skip due to missing/illiquid real data uses status="skip" with a
specific reason -- never silently backfilled with a modeled price. Coverage
funnel tracked and printed exactly like the condor reference script.

Run: python evc_jade_lizard_real_data.py
Outputs (same directory):
  evc_jade_lizard_real_data_rows.csv
  evc_jade_lizard_real_data_summary.csv
  evc_jade_lizard_real_data_per_ticker.csv
  evc_jade_lizard_real_data_coverage.csv
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
# Universe -- IDENTICAL to evc_condor_backtest_real_data.py
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
# Config -- MIN_MOVE / MAX_MOVE_BOUNDS / WING_MULT IDENTICAL to the condor
# reference (short leg at 1.0x EM, long call at WING_MULT=1.5x EM -- same
# convention, for consistency, applied only to the call side here since
# there is no long put).
# ----------------------------------------------------------------------------
MAX_MOVE_BOUNDS = [0.12, 0.25]
MIN_MOVE = 0.03
WING_MULT = 1.5

MAX_EXPIRY_SEARCH_DAYS = 45
REQUEST_TIMEOUT = 15
MAX_RETRIES = 6
AGGS_WORKERS = 8

# Same empirically-confirmed ~2yr rolling window on this Polygon plan's
# options AGGREGATES endpoint -- see evc_condor_backtest_real_data.py for
# the bisection that established this boundary (2024-08-20 -> 403
# NOT_AUTHORIZED, 2024-08-21 -> 200 OK, exactly 2y before that run's date).
PLAN_AGGS_LOOKBACK_DAYS = 730
PLAN_BOUNDARY_SAFETY_DAYS = 5

# Capital feasibility check -- current real account policy for this account.
CASH_SECURED_MULTIPLIER = 100.0   # strike x 100 = full cash-secured collateral, per contract
PER_STRATEGY_CAP_PCT = 0.05       # this account's standing 5% per-strategy risk cap

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
# Polygon.io helpers -- identical to evc_condor_backtest_real_data.py
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
        return None
    return None


def get_contract_grid(ticker: str, entry_date: date):
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
    names = list(occ_tickers.keys())
    with ThreadPoolExecutor(max_workers=min(AGGS_WORKERS, max(1, len(names)))) as ex:
        results = list(ex.map(lambda n: get_daily_close(occ_tickers[n], date_str), names))
    return dict(zip(names, results))


# ----------------------------------------------------------------------------
# yfinance data source -- identical to evc_condor_backtest_real_data.py
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

            # See evc_condor_backtest_real_data.py: yfinance Close is ALWAYS
            # split-adjusted; reconstruct as-traded prices via real split
            # history so strike lookups match Polygon's point-in-time grid.
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
    short_call: float = np.nan
    long_call: float = np.nan
    sp_price: float = np.nan
    sc_price: float = np.nan
    lc_price: float = np.nan
    call_width: float = np.nan
    net_credit: float = np.nan
    is_true_jade_lizard: object = None       # net_credit >= call_width -> upside-risk-free by construction
    call_side_max_loss: float = np.nan       # capped call-spread loss, 0 if true jade lizard
    put_collateral_required: float = np.nan  # short_put * 100 -- cash-secured collateral, this account's policy
    ticker_worst_drawdown_pct: float = np.nan  # this ticker's own worst real historical earnings-day move (full 5y)
    worst_case_price: float = np.nan           # spot * (1 + ticker_worst_drawdown_pct), floored near 0
    practical_put_max_loss: float = np.nan     # (short_put - worst_case_price) - net_credit, floored at 0
    practical_max_risk: float = np.nan         # max(practical_put_max_loss, call_side_max_loss)
    practical_breakeven_win_rate: float = np.nan
    real_price: float = np.nan
    real_move_pct: float = np.nan
    payoff: float = np.nan
    payoff_pct_spot: float = np.nan
    payoff_pct_credit: float = np.nan
    win: object = None
    qualifies_12: bool = False
    qualifies_25: bool = False


def _entry_outcome_dates(edf, td_index, trading_days):
    """
    Replicates the condor reference's BMO/AMC entry/outcome-date logic.
    Returns a list of (edate_local, timing, entry_date, outcome_date) for
    every real historical earnings event with a reported EPS, regardless of
    Polygon coverage -- used both for the worst-drawdown pass (pass 1, pure
    yfinance) and the real-data pricing pass (pass 2, Polygon).
    """
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
    out = []
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
        out.append((edate_local, timing, entry_date, outcome_date))
    return out


def process_ticker(ticker: str, rows: list, stats: dict):
    hist, edf, splits = fetch_ticker_data(ticker)
    if hist is None:
        return

    unadjust = make_unadjust_fn(splits)
    trading_days = list(hist.index.date)
    td_index = {d: i for i, d in enumerate(trading_days)}
    closes = hist["Close"].values

    events = _entry_outcome_dates(edf, td_index, trading_days)

    # ---- Pass 1: this ticker's own worst real historical single-earnings-
    # event drawdown, from the FULL 5y yfinance window (not limited by
    # Polygon's narrower ~2y aggs coverage) -- used as a practical
    # worst-case bound for the naked short put's technically-unbounded
    # downside. Pure price data, no API calls.
    worst_drawdown = 0.0  # most negative real_move_pct seen; 0.0 if none found (no crash on record)
    for (edate_local, timing, entry_date, outcome_date) in events:
        entry_idx = td_index[entry_date]
        outcome_idx = td_index[outcome_date]
        spot_adj = float(closes[entry_idx])
        real_price_adj = float(closes[outcome_idx])
        if spot_adj <= 0 or real_price_adj <= 0:
            continue
        spot_i = spot_adj * unadjust(entry_date)
        real_price_i = real_price_adj * unadjust(outcome_date)
        move_pct = (real_price_i - spot_i) / spot_i
        if move_pct < worst_drawdown:
            worst_drawdown = move_pct

    # ---- Pass 2: real Polygon-priced jade lizard construction per event.
    n_events = 0
    n_ok = 0
    for (edate_local, timing, entry_date, outcome_date) in events:
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

        spot = spot_adj * unadjust(entry_date)
        real_price = real_price_adj * unadjust(outcome_date)
        real_move_pct = (real_price - spot) / spot
        atm_strike = round_strike(spot)
        entry_date_str = entry_date.isoformat()

        plan_cutoff = date.today() - timedelta(days=PLAN_AGGS_LOOKBACK_DAYS + PLAN_BOUNDARY_SAFETY_DAYS)
        if entry_date < plan_cutoff:
            stats["outside_plan_window"] += 1
            rows.append(Row(
                status="skip", reason="outside_polygon_plan_window",
                spot=spot, atm_strike=atm_strike, real_price=real_price,
                real_move_pct=real_move_pct, ticker_worst_drawdown_pct=worst_drawdown,
                **base_kwargs,
            ))
            continue

        expiry_used, call_map, put_map = get_contract_grid(ticker, entry_date)
        if expiry_used is None:
            stats["no_contracts_listed"] += 1
            rows.append(Row(
                status="skip", reason="no_contracts_listed",
                spot=spot, atm_strike=atm_strike, real_price=real_price,
                real_move_pct=real_move_pct, ticker_worst_drawdown_pct=worst_drawdown,
                **base_kwargs,
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
                real_move_pct=real_move_pct, ticker_worst_drawdown_pct=worst_drawdown,
                **base_kwargs,
            ))
            continue

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
                ticker_worst_drawdown_pct=worst_drawdown,
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
            ticker_worst_drawdown_pct=worst_drawdown,
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
        long_call = round_strike(spot + WING_MULT * expected_move)

        if not (0 < short_put < short_call < long_call):
            r.status = "skip"
            r.reason = "invalid_strikes"
            rows.append(r)
            continue

        missing = []
        if short_put not in put_map:
            missing.append("short_put")
        if short_call not in call_map:
            missing.append("short_call")
        if long_call not in call_map:
            missing.append("long_call")
        if missing:
            stats["no_real_data_wing_strike"] += 1
            r.status = "skip"
            r.reason = f"no_real_data:missing_strike:{','.join(missing)}"
            r.short_put, r.short_call, r.long_call = short_put, short_call, long_call
            rows.append(r)
            continue

        wing_occ = {
            "sp": put_map[short_put],
            "sc": call_map[short_call],
            "lc": call_map[long_call],
        }
        wing_closes = get_closes_concurrent(wing_occ, entry_date_str)
        sp_price, sc_price, lc_price = wing_closes["sp"], wing_closes["sc"], wing_closes["lc"]

        r.short_put, r.short_call, r.long_call = short_put, short_call, long_call
        r.sp_price, r.sc_price, r.lc_price = sp_price, sc_price, lc_price

        if any(p is None for p in (sp_price, sc_price, lc_price)):
            stats["no_real_data_wing_trade"] += 1
            r.status = "skip"
            r.reason = "no_real_data:no_trade_wing_leg"
            rows.append(r)
            continue

        # Jade lizard net credit: short put premium + (short call - long call) call-spread credit.
        net_credit = round(sp_price + (sc_price - lc_price), 4)
        if net_credit <= 0:
            r.status = "skip"
            r.reason = "non_positive_credit"
            r.net_credit = net_credit
            rows.append(r)
            continue

        call_width = long_call - short_call
        is_true_jl = bool(net_credit >= call_width)
        call_side_max_loss = max(call_width - net_credit, 0.0)

        put_collateral_required = short_put * CASH_SECURED_MULTIPLIER

        # Practical (not true) worst-case bound for the naked short put:
        # this ticker's own worst real historical earnings-event move,
        # applied to THIS event's spot. True max loss is unbounded down to
        # zero -- this is an explicit practical proxy, not a hard cap.
        worst_case_price = max(spot * (1.0 + worst_drawdown), 0.01)
        practical_put_max_loss = max((short_put - worst_case_price) - net_credit, 0.0)
        practical_max_risk = max(practical_put_max_loss, call_side_max_loss)
        practical_breakeven_win_rate = (
            practical_max_risk / (practical_max_risk + net_credit)
            if (practical_max_risk + net_credit) > 0 else np.nan
        )

        # --- Payoff: put side UNCAPPED down toward real_price->0 (no long put);
        # call side a normal capped credit spread. Byte-for-byte the correct
        # graduated shape for this asymmetric structure.
        if real_price < short_put:
            payoff = net_credit - (short_put - real_price)
        elif real_price <= short_call:
            payoff = net_credit
        elif real_price <= long_call:
            payoff = net_credit - (real_price - short_call)
        else:
            payoff = net_credit - call_width

        r.call_width = call_width
        r.net_credit = net_credit
        r.is_true_jade_lizard = is_true_jl
        r.call_side_max_loss = call_side_max_loss
        r.put_collateral_required = put_collateral_required
        r.worst_case_price = worst_case_price
        r.practical_put_max_loss = practical_put_max_loss
        r.practical_max_risk = practical_max_risk
        r.practical_breakeven_win_rate = practical_breakeven_win_rate
        r.payoff = payoff
        r.payoff_pct_spot = payoff / spot
        r.payoff_pct_credit = payoff / net_credit
        r.win = bool(payoff > 0)
        r.qualifies_12 = im_pct <= 0.12
        r.qualifies_25 = im_pct <= 0.25
        rows.append(r)
        n_ok += 1
        stats["ok"] += 1

    print(f"  {ticker}: {n_events} historical earnings events processed, {n_ok} fully priced with real data "
          f"(ticker worst historical earnings-day move: {worst_drawdown:+.1%})")


def main():
    rows: list = []
    stats = dict(
        total_events=0, outside_plan_window=0, no_contracts_listed=0, no_atm_contract=0,
        no_trade_atm_leg=0, usable_atm_data=0, passed_move_filter=0, no_real_data_wing_strike=0,
        no_real_data_wing_trade=0, ok=0,
    )
    print(f"Processing {len(UNIVERSE)} tickers against REAL Polygon.io options data (jade lizard structure)...")
    for i, ticker in enumerate(UNIVERSE, 1):
        print(f"[{i}/{len(UNIVERSE)}] {ticker}")
        try:
            process_ticker(ticker, rows, stats)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {ticker}: {e}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    out_path = "evc_jade_lizard_real_data_rows.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # -------------------------------------------------------------------
    # Data coverage / skip-rate funnel
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("REAL-DATA COVERAGE FUNNEL")
    print("=" * 100)
    te = stats["total_events"]
    within_window = te - stats["outside_plan_window"]
    print(f"  total historical earnings events (5y window)                    : {te}")
    print(f"  -> outside Polygon plan's ~2y aggs window (skip)                : {stats['outside_plan_window']}  ({stats['outside_plan_window']/max(1,te):.1%} of total)")
    print(f"  = events WITHIN the plan's real-data window                     : {within_window}  ({within_window/max(1,te):.1%} of total)")
    print(f"  -> no options contracts listed near entry (skip)                : {stats['no_contracts_listed']}")
    print(f"  -> ATM strike not listed as real contract (skip)                : {stats['no_atm_contract']}")
    print(f"  -> ATM leg had no real trade that day (skip)                    : {stats['no_trade_atm_leg']}")
    print(f"  = events with a USABLE real expected move                       : {stats['usable_atm_data']}  ({stats['usable_atm_data']/max(1,within_window):.1%} of in-window events)")
    print(f"  -> passed MIN_MOVE/MAX_MOVE_BOUNDS filters                      : {stats['passed_move_filter']}  ({stats['passed_move_filter']/max(1,stats['usable_atm_data']):.1%} of usable-EM events)")
    print(f"  -> wing strike not listed (skip, no_real_data)                  : {stats['no_real_data_wing_strike']}")
    print(f"  -> wing leg had no real trade that day (skip)                   : {stats['no_real_data_wing_trade']}")
    print(f"  = FULLY PRICED real jade lizards (status=ok)                    : {stats['ok']}")
    if stats["passed_move_filter"] > 0:
        print(f"  real-data coverage GIVEN filter pass (ok / passed)              : {stats['ok']/stats['passed_move_filter']:.1%}")
    print(f"  real-data coverage of in-window events (ok / within_window)     : {stats['ok']/max(1,within_window):.1%}")
    print(f"  overall coverage vs full 5y window (ok / total_events)          : {stats['ok']/max(1,te):.1%}")

    coverage_df = pd.DataFrame([stats])
    coverage_df.to_csv("evc_jade_lizard_real_data_coverage.csv", index=False)

    # -------------------------------------------------------------------
    # Aggregate results by max_move_bound
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS (REAL DATA, JADE LIZARD)")
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
        sum_risk = sub["practical_max_risk"].sum()
        sum_credit = sub["net_credit"].sum()
        pooled_practical_breakeven = sum_risk / (sum_risk + sum_credit) if (sum_risk + sum_credit) > 0 else np.nan
        mean_rr = (sub["practical_max_risk"] / sub["net_credit"]).mean()
        avg_days_to_expiry = sub["days_to_expiry"].mean()

        true_jl = sub[sub["is_true_jade_lizard"] == True]  # noqa: E712
        resid = sub[sub["is_true_jade_lizard"] == False]  # noqa: E712

        summary_rows.append(dict(
            max_move_bound=bound, n=n,
            win_rate=win_rate, avg_payoff_pct_spot=avg_payoff_pct_spot,
            avg_payoff_pct_credit=avg_payoff_pct_credit,
            pooled_practical_breakeven_win_rate=pooled_practical_breakeven,
            mean_practical_risk_reward_ratio=mean_rr,
            margin_vs_practical_breakeven=win_rate - pooled_practical_breakeven,
            avg_days_to_expiry=avg_days_to_expiry,
            pct_true_jade_lizard=(sub["is_true_jade_lizard"] == True).mean(),  # noqa: E712
            n_true_jade_lizard=len(true_jl),
            win_rate_true_jade_lizard=true_jl["win"].mean() if len(true_jl) else np.nan,
            n_residual_call_risk=len(resid),
            win_rate_residual_call_risk=resid["win"].mean() if len(resid) else np.nan,
        ))
        print(f"\nmax_move_bound={bound:.0%}  N={n}  (avg real days-to-expiry={avg_days_to_expiry:.1f})")
        print(f"  win_rate                          = {win_rate:.1%}")
        print(f"  pooled PRACTICAL breakeven needed  = {pooled_practical_breakeven:.1%}  (mean practical R:R = {mean_rr:.2f}:1)")
        print(f"  margin vs practical breakeven      = {(win_rate - pooled_practical_breakeven)*100:+.1f} pts")
        print(f"  avg payoff (% of spot)             = {avg_payoff_pct_spot:+.3%}")
        print(f"  avg payoff (% of credit)           = {avg_payoff_pct_credit:+.1%}")
        print(f"  TRUE jade lizard (credit>=call width, upside-risk-free): {len(true_jl)}/{n} ({len(true_jl)/n:.1%}), win_rate={true_jl['win'].mean():.1%}" if len(true_jl) else "  TRUE jade lizard: 0")
        print(f"  residual call risk (credit<call width): {len(resid)}/{n} ({len(resid)/n:.1%}), win_rate={resid['win'].mean():.1%}" if len(resid) else "  residual call risk: 0")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("evc_jade_lizard_real_data_summary.csv", index=False)

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
        grp = sub.groupby("ticker").agg(
            n=("win", "size"), win_rate=("win", "mean"),
            pct_true_jade_lizard=("is_true_jade_lizard", "mean"),
            avg_put_collateral_required=("put_collateral_required", "mean"),
        )
        grp["max_move_bound"] = bound
        per_ticker_records.append(grp.reset_index())
    if per_ticker_records:
        per_ticker_df = pd.concat(per_ticker_records, ignore_index=True)
    else:
        per_ticker_df = pd.DataFrame(columns=["ticker", "n", "win_rate", "max_move_bound"])
    per_ticker_df.to_csv("evc_jade_lizard_real_data_per_ticker.csv", index=False)

    if len(per_ticker_df) > 0:
        eligible_tickers = per_ticker_df[per_ticker_df["n"] >= 8]["ticker"].unique()
        pivot = per_ticker_df[per_ticker_df["ticker"].isin(eligible_tickers)].pivot_table(
            index="ticker", columns=["max_move_bound"], values=["n", "win_rate"]
        )
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(pivot)
    else:
        print("  (no tickers with usable real-data trades)")

    # -------------------------------------------------------------------
    # REAL CAPITAL FEASIBILITY CHECK -- explicit, not skipped.
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("REAL CAPITAL FEASIBILITY CHECK (this account, real live numbers)")
    print("=" * 100)
    combined_net_liq = None
    ibkr_net_liq = None
    alpaca_equity = None
    try:
        ibkr_resp = requests.get("http://localhost:8000/account/summary", timeout=10)
        ibkr_data = ibkr_resp.json()
        ibkr_net_liq = float(ibkr_data.get("net_liquidation", np.nan))
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] could not fetch IBKR /account/summary: {e}")
    try:
        alp_resp = requests.get("http://localhost:8000/alpaca/positions", timeout=10)
        alp_data = alp_resp.json()
        alpaca_equity = float(alp_data.get("account", {}).get("equity", np.nan))
    except Exception as e:  # noqa: BLE001
        print(f"  [WARN] could not fetch /alpaca/positions: {e}")

    if ibkr_net_liq is not None and alpaca_equity is not None and math.isfinite(ibkr_net_liq) and math.isfinite(alpaca_equity):
        combined_net_liq = ibkr_net_liq + alpaca_equity
        per_strategy_cap = combined_net_liq * PER_STRATEGY_CAP_PCT
        print(f"  IBKR net liquidation (U17469343)      : ${ibkr_net_liq:,.2f}")
        print(f"  Alpaca equity                          : ${alpaca_equity:,.2f}")
        print(f"  COMBINED net liq                       : ${combined_net_liq:,.2f}")
        print(f"  Per-strategy cap (5% of combined)      : ${per_strategy_cap:,.2f}")

        ok_priced = ok.dropna(subset=["put_collateral_required"])
        n_priced = len(ok_priced)
        if n_priced > 0:
            over_net_liq = (ok_priced["put_collateral_required"] > combined_net_liq).sum()
            over_cap = (ok_priced["put_collateral_required"] > per_strategy_cap).sum()
            print(f"\n  Real short-put strikes selected across the universe (N={n_priced}, status=ok events):")
            print(f"    min collateral required (strike x 100)  : ${ok_priced['put_collateral_required'].min():,.0f}")
            print(f"    median collateral required               : ${ok_priced['put_collateral_required'].median():,.0f}")
            print(f"    max collateral required                  : ${ok_priced['put_collateral_required'].max():,.0f}")
            print(f"    -> collateral > ENTIRE combined net liq   : {over_net_liq}/{n_priced}  ({over_net_liq/n_priced:.1%})")
            print(f"    -> collateral > 5% per-strategy cap       : {over_cap}/{n_priced}  ({over_cap/n_priced:.1%})")
            n_affordable = n_priced - over_cap
            print(f"    -> events that WOULD fit under the cap    : {n_affordable}/{n_priced}  ({n_affordable/n_priced:.1%})")
            if n_affordable > 0:
                afford_tickers = sorted(ok_priced[ok_priced["put_collateral_required"] <= per_strategy_cap]["ticker"].unique())
                print(f"    tickers with ANY affordable strike        : {afford_tickers}")
            print("\n  HONEST BOTTOM LINE: cash-secured short-put collateral (strike x 100) structurally excludes")
            print("  most/all of this universe from this account's real capital, the same way plain cash-secured")
            print("  puts already do -- a jade lizard's short put is not a smaller-capital alternative to a naked")
            print("  CSP, it needs the SAME full collateral. Winning the backtest's win-rate math does not make")
            print("  the structure deployable at this account's current size.")
        else:
            print("  No fully priced (status=ok) events available to check collateral against.")
        feas_df = pd.DataFrame([dict(
            ibkr_net_liq=ibkr_net_liq, alpaca_equity=alpaca_equity, combined_net_liq=combined_net_liq,
            per_strategy_cap=per_strategy_cap, n_priced=n_priced,
            over_net_liq=int(over_net_liq) if n_priced else None,
            over_cap=int(over_cap) if n_priced else None,
        )])
        feas_df.to_csv("evc_jade_lizard_real_data_capital_feasibility.csv", index=False)
    else:
        print("  [WARN] could not compute combined net liq -- backend endpoints unreachable. Capital check skipped.")

    print("\nDone.")


if __name__ == "__main__":
    main()
