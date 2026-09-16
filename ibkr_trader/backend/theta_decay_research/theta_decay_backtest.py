"""
Theta-decay / credit-accumulation backtest, per CEO request 2026-09-11:
"I find quite often opportunities where theta decay happens quickly even
without any catalyst for certain tickers ... backtest iteratively, pick 5
tickers at a time from our universe, find the pattern."

Universe: Safe Income Trader's real, current, already-vol-filtered universe
(safe_income_auto.py's UNIVERSE, 15 tickers -- NOT the original pre-2026-08-24
top-25, which was narrowed down for real vol-safety reasons documented in
that file). Run in 3 batches of 5, per CEO's explicit "5 at a time" request.

Structure: short iron condor, matching this account's own already-validated
conventions rather than inventing new ones --
  - Short strikes 12% OTM both sides (= Safe Income's own MIN_CUSHION_PCT,
    itself real-backtested via backtest_cushion.py on this exact universe,
    2026-08-24: 97.1% put / 90.4% call win@exp at 12%).
  - Protective long wings at 1.5x that distance (18% OTM) -- same WING_MULT
    convention as cpi_aftermath_monitor.py / EVC.
  - 35 DTE (midpoint of the 25-45 DTE range this account's CSP/LEAP and EVC
    strategies already use).
  - 50%-of-max-credit profit target (same convention as EVC/GOOG Condor).

Premium modeling (STATED APPROXIMATION, same discipline as every other
backtest in this codebase -- no historical options-chain data exists for
any of this):
  - "Implied vol" at entry = trailing 20-trading-day realized vol (stdev of
    daily log returns, annualized) x VRP_MULT (1.15 -- a commonly-cited
    average equity vol-risk-premium multiplier; stated explicitly, not
    fitted).
  - That vol assumption is held FLAT for the life of each simulated trade
    (we have no historical IV *surface* to evolve it against) -- this
    isolates exactly what the CEO asked about: does theta + the REAL price
    path decay this structure's value quickly on its own, not "did IV also
    crush," which we can't honestly claim to model historically.
  - Position is repriced daily via Black-Scholes along the REAL historical
    underlying price path (yfinance daily closes) -- this part is NOT an
    approximation, it's actual price history.

Walk-forward: one trial every 10 trading days across ~3 years of real
history per ticker (not one cherry-picked entry) -- same "real parameter
grid / multiple regimes" discipline as the SOXL CSP analysis precedent.

Secondary diagnostic (separate from the backtest, clearly labeled as
CURRENT/live, not historical): today's real ATM-ish implied vol (yfinance
live option chain) vs. this ticker's own trailing realized-vol distribution
-- answers "is this ticker paying rich premium RIGHT NOW relative to its
own normal decay speed," which the walk-forward backtest alone can't (no
historical IV surface exists to backtest that specific question).

Usage:
  python theta_decay_backtest.py --tickers SPY,QQQ,NVDA,AAPL,GOOGL
"""
import argparse
import json
import math
from datetime import datetime, timedelta

import numpy as np
import yfinance as yf
from scipy.stats import norm

DTE = 35
SHORT_OTM_PCT = 0.12
WING_MULT = 1.5           # default long-wing distance = SHORT_OTM_PCT * WING_MULT (18%)
WING_OTM_PCT_OVERRIDE = None  # set via --wing-pct to use a direct wing distance instead
                               # of the multiplier -- added 2026-09-11 after live quotes on
                               # D/DUK/PEP showed ZERO real bid on the 18%-OTM long leg (no
                               # two-sided market that far out on these low-vol names) --
                               # narrowing to 14-15% trades some protection for an actually
                               # fillable market.
HISTORY_PERIOD = "3y"
VRP_MULT = 1.15
PROFIT_TARGET_PCT = 0.50
STOP_LOSS_CREDIT_MULT = 2.0  # stop when liability = 2x entry credit (captured_pct <= -1.0)
                             # -- same "defined risk, don't ride to max loss" discipline
                             # this account already applies everywhere else (EVC
                             # max_loss_pct, GOOG Condor, Safe Income's own stop rules).
                             # Without this, a handful of full-width breaches produce
                             # unbounded -20x to -60x tail outcomes that don't reflect
                             # how this account would actually run the trade live.
RV_LOOKBACK = 20
ENTRY_STEP_DAYS = 10
RISK_FREE = 0.04  # stated approximation, roughly current short-rate environment
EXCLUDE_EARNINGS = False  # set via --exclude-earnings, added 2026-09-11 for a fair
                           # comparison against the EVC-style variant's earnings exclusion
EARNINGS_BUFFER_DAYS = 1


def get_earnings_dates(ticker: str) -> list:
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=60)
        return [d.date() for d in ed.index]
    except Exception:
        return []


def overlaps_earnings(entry_date, dte_days, earnings_dates) -> bool:
    from datetime import timedelta as _td
    window_start = entry_date - _td(days=EARNINGS_BUFFER_DAYS)
    window_end = entry_date + _td(days=dte_days + EARNINGS_BUFFER_DAYS)
    return any(window_start <= ed <= window_end for ed in earnings_dates)


def bs_put(S, K, T, sigma, r=RISK_FREE):
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        sigma = 0.01
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call(S, K, T, sigma, r=RISK_FREE):
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        sigma = 0.01
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def ic_value(S, short_put_k, long_put_k, short_call_k, long_call_k, T, sigma):
    """Net value of the SHORT iron condor position (what we'd pay to buy it
    back / close it) -- i.e. our liability, shrinking toward 0 is the win."""
    put_spread = bs_put(S, short_put_k, T, sigma) - bs_put(S, long_put_k, T, sigma)
    call_spread = bs_call(S, short_call_k, T, sigma) - bs_call(S, long_call_k, T, sigma)
    return max(put_spread, 0.0) + max(call_spread, 0.0)


def realized_vol(returns_window: np.ndarray) -> float:
    return float(np.std(returns_window, ddof=1) * math.sqrt(252))


def backtest_ticker(ticker: str) -> dict:
    hist = yf.Ticker(ticker).history(period=HISTORY_PERIOD, interval="1d", auto_adjust=True)
    if hist.empty or len(hist) < RV_LOOKBACK + DTE + 10:
        return {"ticker": ticker, "error": "insufficient history"}

    dates = [d.date() for d in hist.index]
    earnings_dates = get_earnings_dates(ticker) if EXCLUDE_EARNINGS else []
    closes = hist["Close"].values
    log_ret = np.diff(np.log(closes))
    n = len(closes)

    trials = []
    skipped_earnings = 0
    i = RV_LOOKBACK
    while i + DTE < n:
        if EXCLUDE_EARNINGS and overlaps_earnings(dates[i], DTE, earnings_dates):
            skipped_earnings += 1
            i += ENTRY_STEP_DAYS
            continue
        S0 = closes[i]
        rv = realized_vol(log_ret[i - RV_LOOKBACK:i])
        iv = rv * VRP_MULT

        wing_pct = WING_OTM_PCT_OVERRIDE if WING_OTM_PCT_OVERRIDE is not None else SHORT_OTM_PCT * WING_MULT
        short_put_k = round(S0 * (1 - SHORT_OTM_PCT), 2)
        long_put_k = round(S0 * (1 - wing_pct), 2)
        short_call_k = round(S0 * (1 + SHORT_OTM_PCT), 2)
        long_call_k = round(S0 * (1 + wing_pct), 2)

        T0 = DTE / 365.0
        entry_credit = ic_value(S0, short_put_k, long_put_k, short_call_k, long_call_k, T0, iv)
        if entry_credit <= 0.01:
            i += ENTRY_STEP_DAYS
            continue

        outcome = "expired_flat"
        days_to_target = None
        days_to_stop = None
        breached = False
        captured_pct_final = None
        for d in range(1, DTE + 1):
            idx = i + d
            if idx >= n:
                break
            S = closes[idx]
            T = max((DTE - d) / 365.0, 1e-6)
            cur_value = ic_value(S, short_put_k, long_put_k, short_call_k, long_call_k, T, iv)
            captured_pct = 1 - (cur_value / entry_credit)
            if S <= short_put_k or S >= short_call_k:
                breached = True
            if captured_pct >= PROFIT_TARGET_PCT:
                days_to_target = d
                outcome = "hit_target"
                captured_pct_final = PROFIT_TARGET_PCT
                break
            if captured_pct <= -STOP_LOSS_CREDIT_MULT + 1:  # liability >= STOP_LOSS_CREDIT_MULT x credit
                days_to_stop = d
                outcome = "stopped_out"
                captured_pct_final = round(captured_pct, 4)
                break

        if captured_pct_final is None:
            final_idx = min(i + DTE, n - 1)
            S_final = closes[final_idx]
            final_value = ic_value(S_final, short_put_k, long_put_k, short_call_k, long_call_k, 0.0, iv)
            captured_pct_final = 1 - (final_value / entry_credit)
            outcome = "breach_at_expiry" if breached and final_value > entry_credit * 0.1 else (
                "recovered_by_expiry" if breached else "expired_flat")

        trials.append({
            "entry_idx": i,
            "days_to_target": days_to_target,
            "days_to_stop": days_to_stop,
            "outcome": outcome,
            "breached_intraday": breached,
            "captured_pct": round(captured_pct_final, 4),
            "entry_credit": round(entry_credit, 2),
            "iv_used": round(iv, 4),
        })
        i += ENTRY_STEP_DAYS

    if not trials:
        return {"ticker": ticker, "error": "no valid trials generated"}

    n_trials = len(trials)
    hit_target = [t for t in trials if t["outcome"] == "hit_target"]
    stopped_out = [t for t in trials if t["outcome"] == "stopped_out"]
    breach_never_recovered = [t for t in trials if t["outcome"] == "breach_at_expiry"]
    win_rate = sum(1 for t in trials if t["outcome"] in ("hit_target", "expired_flat", "recovered_by_expiry")) / n_trials
    stop_rate = len(stopped_out) / n_trials
    avg_days_to_target = float(np.mean([t["days_to_target"] for t in hit_target])) if hit_target else None
    pct_hit_target_by_half_dte = sum(1 for t in hit_target if t["days_to_target"] <= DTE * 0.5) / n_trials

    # A plain mean of captured_pct is dominated by rare catastrophic breach
    # trades (a single blown-through wing can show captured_pct in the -20x
    # to -30x range, since IC max loss is many multiples of the credit
    # collected) -- that outlier can swamp 60+ otherwise-fine trades and
    # make an 80%+ win-rate strategy look like a guaranteed loser on a
    # simple average. Report median (outlier-robust) alongside the mean,
    # PLUS the actual dollar-weighted expectancy (sum of P&L in credit-
    # dollars / n_trials, i.e. "real expectancy per $1 of credit sold"),
    # PLUS the single worst trial, so the tail risk is visible rather than
    # hidden inside one misleading number.
    captured = [t["captured_pct"] for t in trials]
    pnl_dollars = [t["captured_pct"] * t["entry_credit"] for t in trials]
    worst = min(trials, key=lambda t: t["captured_pct"])

    return {
        "ticker": ticker,
        "n_trials": n_trials,
        "skipped_earnings_windows": skipped_earnings,
        "win_rate": round(win_rate, 4),
        "stop_loss_hit_rate": round(stop_rate, 4),
        "breach_no_recover_rate": round(len(breach_never_recovered) / n_trials, 4),
        "pct_trials_hit_50pct_target": round(len(hit_target) / n_trials, 4),
        "avg_days_to_50pct_target": round(avg_days_to_target, 1) if avg_days_to_target else None,
        "pct_hit_target_within_half_dte": round(pct_hit_target_by_half_dte, 4),
        "median_captured_pct": round(float(np.median(captured)), 4),
        "mean_captured_pct": round(float(np.mean(captured)), 4),
        "expectancy_per_dollar_credit_sold": round(float(np.mean(pnl_dollars) / np.mean([t["entry_credit"] for t in trials])), 4),
        "worst_trial_captured_pct": round(worst["captured_pct"], 4),
        "worst_trial_outcome": worst["outcome"],
    }


def live_iv_vs_rv_snapshot(ticker: str) -> dict:
    """Secondary diagnostic -- CURRENT/live only, not historical."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="6mo", interval="1d", auto_adjust=True)
        closes = hist["Close"].values
        log_ret = np.diff(np.log(closes))
        rv_hist = np.array([realized_vol(log_ret[max(0, j - RV_LOOKBACK):j])
                             for j in range(RV_LOOKBACK, len(log_ret))])
        current_rv = realized_vol(log_ret[-RV_LOOKBACK:])

        exps = tk.options
        if not exps:
            return {"ticker": ticker, "error": "no live option chain"}
        today = datetime.now().date()
        target_exp = None
        for e in exps:
            dte = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
            if 25 <= dte <= 45:
                target_exp = e
                break
        target_exp = target_exp or exps[min(2, len(exps) - 1)]
        chain = tk.option_chain(target_exp)
        spot = hist["Close"].iloc[-1]
        calls = chain.calls
        atm_call = calls.iloc[(calls["strike"] - spot).abs().argsort().iloc[0]]
        live_iv = float(atm_call["impliedVolatility"])

        pctile = float((rv_hist < current_rv).mean() * 100) if len(rv_hist) else None
        return {
            "ticker": ticker,
            "live_iv_atm": round(live_iv, 4),
            "current_realized_vol_20d": round(current_rv, 4),
            "live_iv_over_rv_ratio": round(live_iv / current_rv, 2) if current_rv > 0 else None,
            "current_rv_percentile_vs_6mo_own_history": round(pctile, 1) if pctile is not None else None,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def main():
    global WING_OTM_PCT_OVERRIDE, HISTORY_PERIOD, EXCLUDE_EARNINGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True, help="comma-separated")
    ap.add_argument("--wing-pct", type=float, default=None,
                     help="long-wing OTM distance as a fraction (e.g. 0.145 for 14.5%%); "
                          "overrides the default SHORT_OTM_PCT*WING_MULT=18%% wing")
    ap.add_argument("--years", type=str, default="3y",
                     help="yfinance history period, e.g. '3y' or '5y'")
    ap.add_argument("--exclude-earnings", action="store_true",
                     help="skip any trial whose holding window overlaps a real earnings date")
    ap.add_argument("--tag", type=str, default=None, help="suffix for the output filename")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    WING_OTM_PCT_OVERRIDE = args.wing_pct
    EXCLUDE_EARNINGS = args.exclude_earnings
    HISTORY_PERIOD = args.years

    results = []
    for t in tickers:
        print(f"--- {t} ---")
        bt = backtest_ticker(t)
        print(json.dumps(bt, indent=2))
        diag = live_iv_vs_rv_snapshot(t)
        print(json.dumps(diag, indent=2))
        results.append({"backtest": bt, "live_diagnostic": diag})
        print()

    tag = f"_{args.tag}" if args.tag else ""
    out_path = f"batch_{'_'.join(tickers)}{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
