"""
EVC-rule variant of the theta-decay backtest, per CEO request 2026-09-11.

Difference from theta_decay_backtest.py (the flat-12%/18%-OTM version):
  - Short strikes sized off a MODELED expected move (EM), not a flat % of
    spot -- same real rule main.py's EVC uses (~16692-16722): EM = ATM
    straddle price, short_put/call = spot -/+ cushion_mult * EM
    (cushion_mult=1.0, EVC's own default). EM here is BS-modeled (realized
    vol x VRP_MULT, same stated approximation as the rest of this research)
    since no historical option-chain data exists to pull a real straddle
    price from on any past date -- this is the modeled analog of what the
    live evc_style_wings.py script did with a REAL market-quoted straddle.
  - Wing (long) strikes: walk simulated real strike increments (approximate
    per-ticker granularity, matching what was actually observed live)
    tightest-first from the short strike toward the wing_mult*EM fallback,
    take the first whose MODELED credit is still positive -- the backtest
    analog of EVC's real "walk real strikes, take tightest with positive
    conservative bid/ask credit" rule.
  - EARNINGS EXCLUSION (CEO instruction, this run): any trial whose
    [entry, entry+DTE] window overlaps a real historical earnings date
    (yfinance get_earnings_dates, +/-1 day buffer) is skipped entirely.
    This isolates the no-catalyst theta-decay question from event-risk
    gap moves, which is a fundamentally different (and separately well-
    studied, see the earnings_vol_crush / candlestick research desks)
    phenomenon.

Everything else (day-by-day BS repricing along the real price path, 50%
profit target, 2x-credit stop-loss, 5y walk-forward) is identical to
theta_decay_backtest.py -- see that file's docstring for the full stated-
approximation discipline.
"""
import argparse
import json
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

DTE = 35
CUSHION_MULT = 1.0
WING_MULT = 1.5
VRP_MULT = 1.15
PROFIT_TARGET_PCT = 0.50
STOP_LOSS_CREDIT_MULT = 2.0
RV_LOOKBACK = 20
ENTRY_STEP_DAYS = 10
RISK_FREE = 0.04
EARNINGS_BUFFER_DAYS = 1
MIN_MODELED_CREDIT = 0.01  # per share, same floor as the flat-% version

# Approximate real strike granularity near the money, observed live
# 2026-09-11 (evc_style_wings.py) -- used only to simulate realistic
# "walk real strikes" behavior; not claimed exact for every historical date.
STRIKE_GRANULARITY = {"D": 2.5, "KO": 1.0, "DUK": 2.5, "PEP": 1.0}
DEFAULT_GRANULARITY = 1.0


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
    put_spread = bs_put(S, short_put_k, T, sigma) - bs_put(S, long_put_k, T, sigma)
    call_spread = bs_call(S, short_call_k, T, sigma) - bs_call(S, long_call_k, T, sigma)
    return max(put_spread, 0.0) + max(call_spread, 0.0)


def realized_vol(returns_window: np.ndarray) -> float:
    return float(np.std(returns_window, ddof=1) * math.sqrt(252))


def get_earnings_dates(ticker: str) -> list:
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=60)
        return [d.date() for d in ed.index]
    except Exception:
        return []


def overlaps_earnings(entry_date, dte_days, earnings_dates) -> bool:
    window_start = entry_date - timedelta(days=EARNINGS_BUFFER_DAYS)
    window_end = entry_date + timedelta(days=dte_days + EARNINGS_BUFFER_DAYS)
    return any(window_start <= ed <= window_end for ed in earnings_dates)


def find_wing_strike(spot, short_k, wing_fallback, side, T, sigma, granularity, short_leg_price):
    """Walk simulated real strikes tightest-first from short_k toward
    wing_fallback; return the first whose modeled credit is still positive.
    Falls back to wing_fallback itself if nothing tighter clears the bar
    -- exactly EVC's real fallback behavior."""
    price_fn = bs_put if side == "put" else bs_call
    direction = -1 if side == "put" else 1
    n_steps = max(1, int(round(abs(short_k - wing_fallback) / granularity)))
    for step in range(1, n_steps + 1):
        candidate = round(short_k + direction * step * granularity, 2)
        if side == "put" and candidate <= 0:
            break
        cand_price = price_fn(spot, candidate, T, sigma)
        credit = short_leg_price - cand_price
        if credit > MIN_MODELED_CREDIT:
            return candidate, credit
    return round(wing_fallback, 2), None


def backtest_ticker(ticker: str, years: str = "5y") -> dict:
    hist = yf.Ticker(ticker).history(period=years, interval="1d", auto_adjust=True)
    if hist.empty or len(hist) < RV_LOOKBACK + DTE + 10:
        return {"ticker": ticker, "error": "insufficient history"}

    earnings_dates = get_earnings_dates(ticker)
    dates = [d.date() for d in hist.index]
    closes = hist["Close"].values
    log_ret = np.diff(np.log(closes))
    n = len(closes)
    granularity = STRIKE_GRANULARITY.get(ticker, DEFAULT_GRANULARITY)

    trials = []
    skipped_earnings = 0
    i = RV_LOOKBACK
    while i + DTE < n:
        entry_date = dates[i]
        if overlaps_earnings(entry_date, DTE, earnings_dates):
            skipped_earnings += 1
            i += ENTRY_STEP_DAYS
            continue

        S0 = closes[i]
        rv = realized_vol(log_ret[i - RV_LOOKBACK:i])
        iv = rv * VRP_MULT
        T0 = DTE / 365.0

        em = bs_call(S0, S0, T0, iv) + bs_put(S0, S0, T0, iv)  # modeled ATM straddle
        short_put_target = S0 - CUSHION_MULT * em
        short_call_target = S0 + CUSHION_MULT * em
        # snap targets to the simulated strike grid
        short_put_k = round(round(short_put_target / granularity) * granularity, 2)
        short_call_k = round(round(short_call_target / granularity) * granularity, 2)
        if short_put_k <= 0 or short_put_k >= S0 or short_call_k <= S0:
            i += ENTRY_STEP_DAYS
            continue

        short_put_price = bs_put(S0, short_put_k, T0, iv)
        short_call_price = bs_call(S0, short_call_k, T0, iv)

        put_wing_fallback = S0 - WING_MULT * em
        call_wing_fallback = S0 + WING_MULT * em
        long_put_k, _ = find_wing_strike(S0, short_put_k, put_wing_fallback, "put", T0, iv, granularity, short_put_price)
        long_call_k, _ = find_wing_strike(S0, short_call_k, call_wing_fallback, "call", T0, iv, granularity, short_call_price)

        entry_credit = ic_value(S0, short_put_k, long_put_k, short_call_k, long_call_k, T0, iv)
        if entry_credit <= MIN_MODELED_CREDIT:
            i += ENTRY_STEP_DAYS
            continue

        outcome = "expired_flat"
        days_to_target = None
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
            if captured_pct <= -STOP_LOSS_CREDIT_MULT + 1:
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
            "entry_date": str(entry_date), "days_to_target": days_to_target,
            "outcome": outcome, "captured_pct": round(captured_pct_final, 4),
            "entry_credit": round(entry_credit, 2),
            "short_put_k": short_put_k, "long_put_k": long_put_k,
            "short_call_k": short_call_k, "long_call_k": long_call_k,
            "em_pct_of_spot": round(em / S0 * 100, 2),
        })
        i += ENTRY_STEP_DAYS

    if not trials:
        return {"ticker": ticker, "error": "no valid trials generated", "skipped_earnings_windows": skipped_earnings}

    n_trials = len(trials)
    hit_target = [t for t in trials if t["outcome"] == "hit_target"]
    stopped_out = [t for t in trials if t["outcome"] == "stopped_out"]
    breach_never_recovered = [t for t in trials if t["outcome"] == "breach_at_expiry"]
    win_rate = sum(1 for t in trials if t["outcome"] in ("hit_target", "expired_flat", "recovered_by_expiry")) / n_trials
    avg_days_to_target = float(np.mean([t["days_to_target"] for t in hit_target])) if hit_target else None
    captured = [t["captured_pct"] for t in trials]
    pnl_dollars = [t["captured_pct"] * t["entry_credit"] for t in trials]
    worst = min(trials, key=lambda t: t["captured_pct"])
    avg_em_pct = float(np.mean([t["em_pct_of_spot"] for t in trials]))

    return {
        "ticker": ticker,
        "n_trials": n_trials,
        "skipped_earnings_windows": skipped_earnings,
        "avg_expected_move_pct_of_spot": round(avg_em_pct, 2),
        "win_rate": round(win_rate, 4),
        "stop_loss_hit_rate": round(len(stopped_out) / n_trials, 4),
        "breach_no_recover_rate": round(len(breach_never_recovered) / n_trials, 4),
        "pct_trials_hit_50pct_target": round(len(hit_target) / n_trials, 4),
        "avg_days_to_50pct_target": round(avg_days_to_target, 1) if avg_days_to_target else None,
        "median_captured_pct": round(float(np.median(captured)), 4),
        "mean_captured_pct": round(float(np.mean(captured)), 4),
        "expectancy_per_dollar_credit_sold": round(float(np.mean(pnl_dollars) / np.mean([t["entry_credit"] for t in trials])), 4),
        "worst_trial_captured_pct": round(worst["captured_pct"], 4),
        "avg_entry_credit_$": round(float(np.mean([t["entry_credit"] for t in trials]) * 100), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--years", default="5y")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    results = []
    for t in tickers:
        print(f"--- {t} ---")
        bt = backtest_ticker(t, args.years)
        print(json.dumps(bt, indent=2))
        results.append(bt)
        print()

    out_path = f"evc_style_batch_{'_'.join(tickers)}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
