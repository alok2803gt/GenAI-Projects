"""
EVC broken-wing butterfly -- REAL Polygon.io options-chain data backtest.

This is a variant of evc_condor_backtest_real_data.py (the symmetric
iron-condor real-data re-validation: N=255, win_rate=68.6%, breakeven=70.7%,
margin=-2.1pts). It reuses that script's Polygon-fetching pattern EXACTLY
(contracts reference lookup, real ATM straddle pricing, real wing-leg
pricing, coverage/skip tracking, the two real bugs it already solved --
yfinance's always-split-adjusted Close needing as-traded reconstruction, and
this Polygon plan's ~2-year options-aggregates window) -- see that file's
module docstring for the full explanation of those two issues, not repeated
here.

THE ONE THING THAT CHANGES: wing placement becomes ASYMMETRIC per ticker,
based on that ticker's own empirical up-move vs down-move skew around
earnings, instead of both wings sitting at a fixed 1.5x EM. Short strikes
stay at the identical 1.0x EM convention as the symmetric baseline -- only
the long (wing) strikes move -- so this is a clean, isolated test of "does
asymmetric wing-sizing improve on the symmetric real-data baseline" and
nothing else changes in the comparison.

Skew methodology (see compute_skew_ratio() below):
  - For each ticker, real historical earnings-event forward price moves
    (yfinance Close, un-split-adjusted the same way as the reference
    script) are split into "up moves" and "down moves".
  - skew_ratio = median(|down move %|) / median(|up move %|), computed
    using an EXPANDING window of only the events STRICTLY BEFORE the
    current entry_date (i.e. what would have actually been knowable at
    trade time -- no lookahead). Median, not mean, is used because a
    single large earnings gap in a <=20-event-per-ticker sample would
    otherwise dominate the wing-sizing decision; median is the more
    robust choice for a small, fat-tailed per-ticker sample.
  - Requires >=MIN_SKEW_N (3) real up-moves AND >=3 real down-moves of
    prior history before trusting the empirical ratio; earlier events
    (or tickers that are one-sided in their prior sample) fall back to
    skew_ratio=1.0 (symmetric -- i.e. behave exactly like the baseline
    condor for that one event). This is reported per-row (skew_source)
    and aggregated, since a result built mostly on the symmetric
    fallback isn't really testing the asymmetric idea.
  - skew_ratio is clipped to [0.5, 2.0] (SKEW_RATIO_CLIP) so one noisy
    ticker/event can't blow the wing allocation out to a degenerate
    extreme (e.g. all width on one side).

Wing allocation (the isolated change vs the symmetric baseline):
  Symmetric baseline: long_put = spot - 1.5*EM, long_call = spot + 1.5*EM
  -> in EM units, put_width = call_width = 0.5 each (total wing "budget"
  beyond the short strikes = 1.0 EM units, split 50/50).
  Broken-wing here: same TOTAL budget (1.0 EM units), but split
  proportionally to skew_ratio instead of 50/50:
      put_width_em  = skew_ratio / (skew_ratio + 1)
      call_width_em = 1 / (skew_ratio + 1)
      long_put  = round_strike(spot - (1 + put_width_em)  * EM)
      long_call = round_strike(spot + (1 + call_width_em) * EM)
  skew_ratio > 1 (bigger real downside moves historically) -> put side
  gets the WIDER wing (put_width_em > 0.5), call side gets the narrower
  one, and vice versa. skew_ratio == 1 (or the default-symmetric
  fallback) reproduces the exact symmetric baseline for that event --
  by construction this is a true isolated A/B, not a different total
  risk budget.
  HONEST MECHANICAL NOTE: a WIDER wing means the protective long leg
  sits further from the money -> it is CHEAPER -> more of the short
  premium is retained as credit, but the worst-case loss on that side
  (wing_width - net_credit) is LARGER if fully breached. A NARROWER
  wing means a more expensive, closer-in long leg -> less credit, but a
  SMALLER worst-case loss if breached. So this design puts more credit
  AND a bigger worst-case tail loss on the side that already breaches
  more often historically -- whether that trade-off nets out favorably
  is exactly the empirical question this script answers; it is not
  assumed favorable up front.

max_risk / payoff formulas are copied byte-for-byte from the reference
script (they already generalize correctly to unequal put_width/call_width
-- max_risk = max(put_width, call_width) - net_credit, and the payoff
branches already use put_width/call_width independently per side).

Every skip due to missing/illiquid real data uses status="skip" with an
explicit reason -- never silently backfilled or modeled. Coverage is
tracked and reported exactly as honestly as the reference script does.

Run: python evc_broken_wing_butterfly_real_data.py
Outputs (same directory):
  evc_broken_wing_real_data_rows.csv       (row-level, one row per event)
  evc_broken_wing_real_data_summary.csv    (aggregate by max_move_bound)
  evc_broken_wing_real_data_per_ticker.csv (per-ticker breakdown, N>=8)
  evc_broken_wing_real_data_coverage.csv   (data-coverage funnel)
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
# Config -- MIN_MOVE / MAX_MOVE_BOUNDS / short-strike convention IDENTICAL to
# the symmetric baseline so the comparison is apples-to-apples. WING_MULT is
# kept as a named constant purely to document that the *total* wing budget
# (put_width_em + call_width_em == 1.0, i.e. long strikes at 1.5x EM on
# average) matches the baseline's fixed 1.5x -- only its 50/50 split changes.
# ----------------------------------------------------------------------------
MAX_MOVE_BOUNDS = [0.12, 0.25]
MIN_MOVE = 0.03
WING_MULT = 1.5                      # baseline long-strike multiplier (for reference/logging only)
TOTAL_WING_BUDGET_EM = 2 * (WING_MULT - 1.0)  # = 1.0 EM units, split asymmetrically below

MIN_SKEW_N = 3            # need >=3 real prior up-moves AND >=3 real prior down-moves
SKEW_RATIO_CLIP = (0.5, 2.0)   # cap asymmetry at ~2:1 wing-width allocation

MAX_EXPIRY_SEARCH_DAYS = 45
REQUEST_TIMEOUT = 15
MAX_RETRIES = 6
AGGS_WORKERS = 8

# Same empirically-confirmed Polygon plan window as the reference script.
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
# Polygon.io helpers -- copied verbatim from evc_condor_backtest_real_data.py
# (same retry/backoff pattern, same endpoints).
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
    """Nearest real listed options expiration >= entry_date, as of entry_date."""
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
# yfinance data source -- IDENTICAL logic to evc_condor_backtest_real_data.py
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

            # Same split-adjustment correction as the reference script: yfinance
            # Close is always retroactively split-adjusted; Polygon's real
            # point-in-time strike grid reflects the as-traded price. Reconstruct
            # as-traded prices via real split history before comparing to
            # Polygon strikes (see reference script's fetch_ticker_data for the
            # full NFLX-10:1 example).
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
    """Identical to the reference script's make_unadjust_fn()."""
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


def compute_skew_ratio(prior_moves: list[float]):
    """
    Expanding-window (no-lookahead) empirical skew ratio from a ticker's own
    real prior earnings-event forward moves (real_move_pct).

    ratio = median(|down move|) / median(up move), i.e. >1 means downside
    moves have historically been bigger in magnitude than upside moves for
    this ticker (at this point in its history). Requires >=MIN_SKEW_N real
    prior up-moves AND >=MIN_SKEW_N real prior down-moves; otherwise falls
    back to 1.0 (symmetric -- behaves exactly like the baseline condor for
    that event). Clipped to SKEW_RATIO_CLIP to bound the asymmetry.

    Returns (ratio, n_up, n_down, source) where source is "empirical" or
    "default_symmetric".
    """
    ups = [m for m in prior_moves if m > 0]
    downs = [m for m in prior_moves if m < 0]
    n_up, n_down = len(ups), len(downs)
    if n_up < MIN_SKEW_N or n_down < MIN_SKEW_N:
        return 1.0, n_up, n_down, "default_symmetric"
    up_mag = float(np.median(ups))
    down_mag = float(np.median([abs(m) for m in downs]))
    if not math.isfinite(up_mag) or up_mag <= 0:
        return 1.0, n_up, n_down, "default_symmetric"
    ratio = down_mag / up_mag
    ratio = min(max(ratio, SKEW_RATIO_CLIP[0]), SKEW_RATIO_CLIP[1])
    return ratio, n_up, n_down, "empirical"


@dataclass
class Row:
    ticker: str
    earnings_date: str
    timing: str
    entry_date: str
    outcome_date: str
    status: str
    reason: str = ""
    skew_source: str = ""
    skew_ratio: float = np.nan
    skew_n_up: int = 0
    skew_n_down: int = 0
    put_width_em: float = np.nan
    call_width_em: float = np.nan
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

    # -------------------------------------------------------------------
    # Pass 1: build ALL earnings events using real yfinance price data only
    # (no Polygon options calls yet) -- this is the raw material for the
    # per-ticker skew estimate, and it is NOT limited by Polygon's ~2y
    # options-aggregates window, since it only needs underlying closes.
    # Same entry/outcome date logic as the reference script.
    # -------------------------------------------------------------------
    events = []
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

        entry_idx = td_index[entry_date]
        outcome_idx = td_index[outcome_date]
        spot_adj = float(closes[entry_idx])
        real_price_adj = float(closes[outcome_idx])

        ev = dict(
            earnings_date=str(edate_local), timing=timing,
            entry_date=entry_date, outcome_date=outcome_date,
        )
        if spot_adj <= 0 or real_price_adj <= 0:
            ev["valid"] = False
            events.append(ev)
            continue

        spot = spot_adj * unadjust(entry_date)
        real_price = real_price_adj * unadjust(outcome_date)
        ev.update(
            valid=True, spot=spot, real_price=real_price,
            real_move_pct=(real_price - spot) / spot,
        )
        events.append(ev)

    events.sort(key=lambda e: e["entry_date"])

    # -------------------------------------------------------------------
    # Pass 2: iterate chronologically. Compute this ticker's expanding,
    # no-lookahead skew ratio from STRICTLY PRIOR valid events, then price
    # the real broken-wing butterfly via Polygon for events within the
    # data window.
    # -------------------------------------------------------------------
    prior_moves: list[float] = []
    n_events = 0
    n_ok = 0
    for ev in events:
        if not ev.get("valid", True):
            rows.append(Row(
                status="skip", reason="bad_price", ticker=ticker,
                earnings_date=ev["earnings_date"], timing=ev["timing"],
                entry_date=str(ev["entry_date"]), outcome_date=str(ev["outcome_date"]),
            ))
            continue

        n_events += 1
        stats["total_events"] += 1

        skew_ratio, n_up, n_down, skew_source = compute_skew_ratio(prior_moves)
        prior_moves.append(ev["real_move_pct"])  # only AFTER computing this event's skew -- no lookahead
        if skew_source == "empirical":
            stats["skew_empirical"] += 1
        else:
            stats["skew_default_symmetric"] += 1

        entry_date = ev["entry_date"]
        outcome_date = ev["outcome_date"]
        spot = ev["spot"]
        real_price = ev["real_price"]
        real_move_pct = ev["real_move_pct"]
        atm_strike = round_strike(spot)
        entry_date_str = entry_date.isoformat()

        base_kwargs = dict(
            ticker=ticker, earnings_date=ev["earnings_date"], timing=ev["timing"],
            entry_date=str(entry_date), outcome_date=str(outcome_date),
            skew_source=skew_source, skew_ratio=skew_ratio,
            skew_n_up=n_up, skew_n_down=n_down,
        )

        plan_cutoff = date.today() - timedelta(days=PLAN_AGGS_LOOKBACK_DAYS + PLAN_BOUNDARY_SAFETY_DAYS)
        if entry_date < plan_cutoff:
            stats["outside_plan_window"] += 1
            rows.append(Row(
                status="skip", reason="outside_polygon_plan_window",
                spot=spot, atm_strike=atm_strike, real_price=real_price,
                real_move_pct=real_move_pct, **base_kwargs,
            ))
            continue

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

        # Asymmetric wing-width allocation from this event's (no-lookahead)
        # skew_ratio. skew_ratio==1.0 reproduces the symmetric baseline
        # exactly (put_width_em == call_width_em == 0.5).
        put_width_em = skew_ratio / (skew_ratio + 1.0)
        call_width_em = 1.0 - put_width_em

        r = Row(
            status="ok", reason="",
            expiry_used=expiry_used, days_to_expiry=days_to_expiry,
            spot=spot, atm_strike=atm_strike,
            atm_call_price=atm_call_price, atm_put_price=atm_put_price,
            expected_move=expected_move, im_pct=im_pct,
            real_price=real_price, real_move_pct=real_move_pct,
            put_width_em=put_width_em, call_width_em=call_width_em,
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

        # Short strikes: IDENTICAL 1x EM convention to the symmetric baseline.
        short_put = round_strike(spot - expected_move)
        short_call = round_strike(spot + expected_move)
        # Long (wing) strikes: asymmetric per-side multiplier, same total
        # (1.0 EM units) budget as the baseline's fixed 1.5x on both sides.
        long_put = round_strike(spot - (1.0 + put_width_em) * expected_move)
        long_call = round_strike(spot + (1.0 + call_width_em) * expected_move)

        if not (long_put < short_put < short_call < long_call):
            r.status = "skip"
            r.reason = "invalid_strikes"
            r.short_put, r.long_put = short_put, long_put
            r.short_call, r.long_call = short_call, long_call
            rows.append(r)
            continue

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

        # Payoff -- IDENTICAL formula to the reference script. It already
        # generalizes correctly to unequal put_width/call_width: the
        # "blown through the wing" branch uses each side's own width.
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
        no_real_data_wing_trade=0, ok=0, skew_empirical=0, skew_default_symmetric=0,
    )
    print(f"Processing {len(UNIVERSE)} tickers against REAL Polygon.io options data (broken-wing butterfly)...")
    for i, ticker in enumerate(UNIVERSE, 1):
        print(f"[{i}/{len(UNIVERSE)}] {ticker}")
        try:
            process_ticker(ticker, rows, stats)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {ticker}: {e}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    out_path = "evc_broken_wing_real_data_rows.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # -------------------------------------------------------------------
    # Data coverage / skip-rate funnel
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("REAL-DATA COVERAGE FUNNEL (broken-wing butterfly)")
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
    print(f"  = FULLY PRICED real broken-wing butterflies (status=ok)         : {stats['ok']}")
    if stats["passed_move_filter"] > 0:
        print(f"  real-data coverage GIVEN filter pass (ok / passed)              : {stats['ok']/stats['passed_move_filter']:.1%}")
    print(f"  real-data coverage of in-window events (ok / within_window)     : {stats['ok']/max(1,within_window):.1%}")
    print(f"  overall coverage vs full 5y window (ok / total_events)          : {stats['ok']/max(1,te):.1%}")
    skew_total = stats["skew_empirical"] + stats["skew_default_symmetric"]
    print(f"\n  events priced with EMPIRICAL (no-lookahead) skew                : {stats['skew_empirical']}  ({stats['skew_empirical']/max(1,skew_total):.1%} of all processed events)")
    print(f"  events falling back to default_symmetric (insufficient history) : {stats['skew_default_symmetric']}  ({stats['skew_default_symmetric']/max(1,skew_total):.1%} of all processed events)")

    coverage_df = pd.DataFrame([stats])
    coverage_df.to_csv("evc_broken_wing_real_data_coverage.csv", index=False)

    # -------------------------------------------------------------------
    # Aggregate result by max_move_bound
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS (REAL DATA, broken-wing butterfly)")
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
        pct_empirical_skew = (sub["skew_source"] == "empirical").mean()
        avg_skew_ratio = sub["skew_ratio"].mean()
        avg_put_width_em = sub["put_width_em"].mean()
        avg_call_width_em = sub["call_width_em"].mean()
        summary_rows.append(dict(
            max_move_bound=bound, n=n,
            win_rate=win_rate, avg_payoff_pct_spot=avg_payoff_pct_spot,
            avg_payoff_pct_credit=avg_payoff_pct_credit,
            pooled_breakeven_win_rate=pooled_breakeven,
            mean_risk_reward_ratio=mean_rr,
            margin_vs_breakeven=win_rate - pooled_breakeven,
            avg_days_to_expiry=avg_days_to_expiry,
            pct_empirical_skew=pct_empirical_skew,
            avg_skew_ratio=avg_skew_ratio,
            avg_put_width_em=avg_put_width_em,
            avg_call_width_em=avg_call_width_em,
            sum_net_credit=sum_credit,
            sum_max_risk=sum_risk,
        ))
        print(f"\nmax_move_bound={bound:.0%}  N={n}  (avg real days-to-expiry={avg_days_to_expiry:.1f}, {pct_empirical_skew:.1%} empirical skew)")
        print(f"  win_rate                = {win_rate:.1%}")
        print(f"  pooled breakeven needed  = {pooled_breakeven:.1%}  (mean R:R = {mean_rr:.2f}:1)")
        print(f"  margin vs breakeven      = {(win_rate - pooled_breakeven)*100:+.1f} pts")
        print(f"  avg payoff (% of spot)   = {avg_payoff_pct_spot:+.3%}")
        print(f"  avg payoff (% of credit) = {avg_payoff_pct_credit:+.1%}")
        print(f"  avg wing split (put/call EM units) = {avg_put_width_em:.3f} / {avg_call_width_em:.3f}  (avg skew_ratio={avg_skew_ratio:.2f})")
        print(f"  total real $ credit / $ max-risk (per 1 contract, sum across N) = {sum_credit:.2f} / {sum_risk:.2f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("evc_broken_wing_real_data_summary.csv", index=False)

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
            avg_skew_ratio=("skew_ratio", "mean"),
            pct_empirical_skew=("skew_source", lambda s: (s == "empirical").mean()),
            sum_net_credit=("net_credit", "sum"), sum_max_risk=("max_risk", "sum"),
        )
        grp["max_move_bound"] = bound
        per_ticker_records.append(grp.reset_index())
    if per_ticker_records:
        per_ticker_df = pd.concat(per_ticker_records, ignore_index=True)
    else:
        per_ticker_df = pd.DataFrame(columns=["ticker", "n", "win_rate", "max_move_bound"])
    per_ticker_df.to_csv("evc_broken_wing_real_data_per_ticker.csv", index=False)

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
