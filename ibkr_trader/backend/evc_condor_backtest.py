"""
EVC (Earnings Vol Crush) iron-condor historical backtest.

Standalone, offline analysis script -- does NOT import from main.py (per task
instructions). Replicates the exact condor-construction methodology of
_evc_quote_condor() in main.py, using yfinance history + earnings dates as the
real historical data source, and a Black-Scholes premium ESTIMATE (since no
real historical options-chain data exists) driven by a realized-vol-times-
multiplier implied-vol proxy.

See the "METHODOLOGY AND CAVEATS" comment block near the bottom for the full
list of approximations. The single biggest one: option premiums are a MODELED
Black-Scholes estimate off an assumed IV = realized_vol_20d * {1.3, 1.6}, not
observed market prices. Everything downstream (expected move, strikes,
credit, payoff) inherits that uncertainty.

Run: python evc_condor_backtest.py
Output: evc_condor_backtest_rows.csv (row-level) in this same directory,
plus a printed aggregate summary.
"""
from __future__ import annotations

import math
import time
import warnings
from datetime import date, datetime
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Universe (~55 tickers, diverse sectors, real earnings-reaction history)
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
# Config -- mirrors main.py's evc config exactly where noted
# ----------------------------------------------------------------------------
IV_MULTS = [1.3, 1.6]                 # sensitivity range for the IV-vs-RV premium
MAX_MOVE_BOUNDS = [0.12, 0.25]        # matches production cfg["max_move_pct"]=12 (current)
                                       # and the hardcoded absolute cap im_pct>0.25 in main.py
MIN_MOVE = 0.03                       # matches main.py: reject if im_pct < 3%
WING_MULT = 1.5                       # matches main.py cfg["wing_mult"]
RISK_FREE = 0.045                     # matches this account's gex-vex-calculator convention
DIV_YIELD = 0.0
T_YEARS = 1.0 / 252.0                 # 1 trading day to expiry (straddle + condor legs)
VOL_LOOKBACK_DAYS = 20                # trailing realized-vol window before entry


def round_strike(price: float) -> float:
    """Exact replica of _evc_round_strike() in main.py (default step=0.0 branch)."""
    if price < 50:
        step = 1.0
    elif price < 200:
        step = 2.5
    else:
        step = 5.0
    return float(round(price / step) * step)


def bs_call_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    """Black-Scholes European call & put premiums."""
    if S <= 0 or K <= 0:
        return 0.0, 0.0
    if T <= 0 or sigma <= 0:
        call = max(0.0, S - K)
        put = max(0.0, K - S)
        return call, put
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return max(call, 0.0), max(put, 0.0)


def fetch_ticker_data(ticker: str, retries: int = 3):
    """Fetch 5y unadjusted-close history + up to 20 real earnings dates."""
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
            return hist, edf
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"  [SKIP TICKER] {ticker}: could not fetch data ({last_err})")
    return None, None


@dataclass
class Row:
    ticker: str
    earnings_date: str
    timing: str
    entry_date: str
    outcome_date: str
    iv_mult: float
    status: str
    reason: str = ""
    spot: float = np.nan
    atm_strike: float = np.nan
    realized_vol_20d: float = np.nan
    iv_estimate: float = np.nan
    expected_move: float = np.nan
    im_pct: float = np.nan
    short_put: float = np.nan
    long_put: float = np.nan
    short_call: float = np.nan
    long_call: float = np.nan
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


def process_ticker(ticker: str, rows: list):
    hist, edf = fetch_ticker_data(ticker)
    if hist is None:
        return

    trading_days = list(hist.index.date)  # sorted ascending python date objects
    td_index = {d: i for i, d in enumerate(trading_days)}
    closes = hist["Close"].values
    log_ret = np.diff(np.log(closes))  # log_ret[i] = return from day i to day i+1

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
    for ts, erow in edf.iterrows():
        edate = pd.Timestamp(ts)
        edate_local = edate.date()
        if edate_local >= today:
            continue  # future / not-yet-happened
        reported_eps = erow.get("Reported EPS", np.nan)
        if pd.isna(reported_eps):
            continue  # not a real, already-reported earnings event

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
        entry_idx = td_index[entry_date]
        outcome_idx = td_index[outcome_date]

        base_row_kwargs = dict(
            ticker=ticker,
            earnings_date=str(edate_local),
            timing=timing,
            entry_date=str(entry_date),
            outcome_date=str(outcome_date),
        )

        if entry_idx < VOL_LOOKBACK_DAYS:
            for m in IV_MULTS:
                rows.append(Row(iv_mult=m, status="skip", reason="insufficient_history", **base_row_kwargs))
            continue

        spot = float(closes[entry_idx])
        real_price = float(closes[outcome_idx])
        if spot <= 0 or real_price <= 0:
            for m in IV_MULTS:
                rows.append(Row(iv_mult=m, status="skip", reason="bad_price", **base_row_kwargs))
            continue

        # realized vol over the 20 trading days BEFORE entry: use the 20 close-to-close
        # log returns ending at entry_date (log_ret indices entry_idx-20 .. entry_idx-1)
        window = log_ret[entry_idx - VOL_LOOKBACK_DAYS: entry_idx]
        if len(window) < VOL_LOOKBACK_DAYS or np.any(np.isnan(window)):
            for m in IV_MULTS:
                rows.append(Row(iv_mult=m, status="skip", reason="bad_vol_window", **base_row_kwargs))
            continue
        realized_vol = float(np.std(window, ddof=1) * math.sqrt(252))
        if realized_vol <= 0 or not math.isfinite(realized_vol):
            for m in IV_MULTS:
                rows.append(Row(iv_mult=m, status="skip", reason="zero_realized_vol", **base_row_kwargs))
            continue

        atm_strike = round_strike(spot)
        real_move_pct = (real_price - spot) / spot

        for iv_mult in IV_MULTS:
            iv_est = realized_vol * iv_mult
            call_atm, put_atm = bs_call_put(spot, atm_strike, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
            expected_move = call_atm + put_atm
            im_pct = expected_move / spot if spot > 0 else np.nan

            r = Row(
                iv_mult=iv_mult, status="ok", reason="",
                spot=spot, atm_strike=atm_strike, realized_vol_20d=realized_vol,
                iv_estimate=iv_est, expected_move=expected_move, im_pct=im_pct,
                real_price=real_price, real_move_pct=real_move_pct,
                **base_row_kwargs,
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

            short_put = round_strike(spot - expected_move)
            short_call = round_strike(spot + expected_move)
            long_put = round_strike(spot - WING_MULT * expected_move)
            long_call = round_strike(spot + WING_MULT * expected_move)

            if not (long_put < short_put < short_call < long_call):
                r.status = "skip"
                r.reason = "invalid_strikes"
                rows.append(r)
                continue

            sc_call, sc_put_dummy = bs_call_put(spot, short_call, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
            _, sp_put = bs_call_put(spot, short_put, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
            lc_call, _ = bs_call_put(spot, long_call, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
            _, lp_put = bs_call_put(spot, long_put, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)

            sc_prem = sc_call
            sp_prem = sp_put
            lc_prem = lc_call
            lp_prem = lp_put

            net_credit = round((sp_prem + sc_prem) - (lp_prem + lc_prem), 4)
            if net_credit <= 0:
                r.status = "skip"
                r.reason = "non_positive_credit"
                rows.append(r)
                continue

            put_width = short_put - long_put
            call_width = long_call - short_call
            wing_width = max(put_width, call_width)
            max_risk = wing_width - net_credit
            if max_risk <= 0:
                r.status = "skip"
                r.reason = "non_positive_risk"
                rows.append(r)
                continue

            # payoff at real outcome price
            if short_put <= real_price <= short_call:
                payoff = net_credit
            elif real_price < short_put:
                breach = short_put - real_price
                if real_price >= long_put:
                    payoff = net_credit - breach
                else:
                    payoff = net_credit - (put_width - net_credit)
            else:  # real_price > short_call
                breach = real_price - short_call
                if real_price <= long_call:
                    payoff = net_credit - breach
                else:
                    payoff = net_credit - (call_width - net_credit)

            r.short_put, r.long_put = short_put, long_put
            r.short_call, r.long_call = short_call, long_call
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

    print(f"  {ticker}: {n_events} historical earnings events processed")


def main():
    rows: list = []
    print(f"Processing {len(UNIVERSE)} tickers...")
    for i, ticker in enumerate(UNIVERSE, 1):
        print(f"[{i}/{len(UNIVERSE)}] {ticker}")
        try:
            process_ticker(ticker, rows)
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {ticker}: {e}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    out_path = "evc_condor_backtest_rows.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # -------------------------------------------------------------------
    # Aggregate: 4 scenario rows (iv_mult x max_move_bound)
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS")
    print("=" * 100)

    ok = df[df["status"] == "ok"].copy()
    summary_rows = []
    for iv_mult in IV_MULTS:
        for bound in MAX_MOVE_BOUNDS:
            qcol = "qualifies_12" if bound == 0.12 else "qualifies_25"
            sub = ok[(ok["iv_mult"] == iv_mult) & (ok[qcol] == True) & (ok["im_pct"] >= MIN_MOVE)]  # noqa: E712
            n = len(sub)
            if n == 0:
                summary_rows.append(dict(iv_mult=iv_mult, max_move_bound=bound, n=0))
                continue
            win_rate = sub["win"].mean()
            avg_payoff_pct_spot = sub["payoff_pct_spot"].mean()
            avg_payoff_pct_credit = sub["payoff_pct_credit"].mean()
            sum_risk = sub["max_risk"].sum()
            sum_credit = sub["net_credit"].sum()
            pooled_breakeven = sum_risk / (sum_risk + sum_credit)
            mean_rr = (sub["max_risk"] / sub["net_credit"]).mean()
            summary_rows.append(dict(
                iv_mult=iv_mult, max_move_bound=bound, n=n,
                win_rate=win_rate, avg_payoff_pct_spot=avg_payoff_pct_spot,
                avg_payoff_pct_credit=avg_payoff_pct_credit,
                pooled_breakeven_win_rate=pooled_breakeven,
                mean_risk_reward_ratio=mean_rr,
                margin_vs_breakeven=win_rate - pooled_breakeven,
            ))
            print(f"\niv_mult={iv_mult}x  max_move_bound={bound:.0%}  N={n}")
            print(f"  win_rate                = {win_rate:.1%}")
            print(f"  pooled breakeven needed  = {pooled_breakeven:.1%}  (mean R:R = {mean_rr:.2f}:1)")
            print(f"  margin vs breakeven      = {(win_rate - pooled_breakeven)*100:+.1f} pts")
            print(f"  avg payoff (% of spot)   = {avg_payoff_pct_spot:+.3%}")
            print(f"  avg payoff (% of credit) = {avg_payoff_pct_credit:+.1%}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("evc_condor_backtest_summary.csv", index=False)

    # -------------------------------------------------------------------
    # Per-ticker breakdown (N >= 8 in at least one scenario)
    # -------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("PER-TICKER BREAKDOWN (tickers with >=8 qualifying trades in at least one scenario)")
    print("=" * 100)
    per_ticker_records = []
    for iv_mult in IV_MULTS:
        for bound in MAX_MOVE_BOUNDS:
            qcol = "qualifies_12" if bound == 0.12 else "qualifies_25"
            sub = ok[(ok["iv_mult"] == iv_mult) & (ok[qcol] == True) & (ok["im_pct"] >= MIN_MOVE)]  # noqa: E712
            grp = sub.groupby("ticker").agg(n=("win", "size"), win_rate=("win", "mean"))
            grp["iv_mult"] = iv_mult
            grp["max_move_bound"] = bound
            per_ticker_records.append(grp.reset_index())
    per_ticker_df = pd.concat(per_ticker_records, ignore_index=True)
    per_ticker_df.to_csv("evc_condor_backtest_per_ticker.csv", index=False)

    eligible_tickers = per_ticker_df[per_ticker_df["n"] >= 8]["ticker"].unique()
    pivot = per_ticker_df[per_ticker_df["ticker"].isin(eligible_tickers)].pivot_table(
        index="ticker", columns=["iv_mult", "max_move_bound"], values=["n", "win_rate"]
    )
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(pivot)

    print("\nDone.")


if __name__ == "__main__":
    main()
