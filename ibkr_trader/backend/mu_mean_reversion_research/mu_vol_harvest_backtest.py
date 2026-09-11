"""
The simple daily-bar price-mean-reversion test (mu_dip_run_backtest.py)
found NOTHING real -- 0/80 dip-buy signals significant, and the one
significant run-sell result actually showed MOMENTUM CONTINUATION (MU
kept running after overbought RSI), the opposite of what naive "sell the
rip" price-reversal timing would need. That rules out the naive
directional-timing explanation for how people profit from MU's
dip/run pattern.

This tests the more sophisticated, realistic alternative: professionals
harvesting a VOLATILITY risk premium, not betting on price direction at
all. A big move (either direction) mechanically spikes near-term implied
vol; if that vol then decays faster than the realized move continues to
justify, SELLING an ATM straddle right after the move and holding it
captures that decay regardless of whether price keeps trending -- this
is fully consistent with the momentum finding above (price doesn't need
to revert for this to work).

Method: for each real MU single-day dip (<=-5%) and run (>=+5%) trigger
day, plus a matched random baseline sample (all from the 2025+ window,
so price scale is roughly comparable), find a REAL, ACTUALLY-TRADED
MU option contract near-the-money with 25-50 calendar days to expiry.
Get its real close price (and its matching put at the same strike/expiry)
on the entry day, then the SAME contracts' real close price ~10 trading
days later. Short-straddle P&L = entry premium collected - exit value
paid to close (per share; x100/contract, no transaction costs -- stated
approximation, same as this account's other backtests when real
bid/ask isn't available). Also computes BS-implied vol at entry/exit to
show the mechanism, not just the raw P&L number.
"""
import gzip
import math
import pickle
import re
from datetime import timedelta
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.config import Config
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import norm

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
UNIVERSE_PKL = BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl"

TARGET_DTE_LO, TARGET_DTE_HI = 25, 50
HOLD_TRADING_DAYS = 10
RISK_FREE_RATE = 0.04
_OCC_RE = re.compile(r"^O:MU(\d{6})([CP])(\d{8})$")


def s3_client():
    import json
    with open(BACKEND_DIR / "scanner_config.json") as f:
        cfg = json.load(f)
    session = boto3.Session(
        aws_access_key_id=cfg["polygon_s3_access_key"],
        aws_secret_access_key=cfg["polygon_s3_secret_key"],
    )
    return session.client("s3", endpoint_url="https://files.massive.com",
                           config=Config(signature_version="s3v4"))


def fetch_mu_contracts_for_day(s3, day: str) -> dict:
    """Returns {(expiry_yymmdd, right, strike): close_price} for every real
    MU contract that traded that day, using the day's last real trade
    price per contract (same stated approximation fetch_real_option_bars.py
    already uses)."""
    y, m = day[:4], day[5:7]
    key = f"us_options_opra/minute_aggs_v1/{y}/{m}/{day}.csv.gz"
    try:
        obj = s3.get_object(Bucket="flatfiles", Key=key)
    except Exception as e:
        return {"_error": str(e)}
    data = gzip.decompress(obj["Body"].read())
    lines = data.decode().splitlines()
    header = lines[0].split(",")
    ticker_idx, close_idx, ts_idx = header.index("ticker"), header.index("close"), header.index("window_start")
    bars = {}  # (expiry, right, strike) -> list of (ts, close)
    for line in lines[1:]:
        parts = line.split(",")
        ticker = parts[ticker_idx]
        m2 = _OCC_RE.match(ticker)
        if not m2:
            continue
        expiry, right, strike_str = m2.groups()
        strike = int(strike_str) / 1000.0
        key2 = (expiry, right, strike)
        ts = int(parts[ts_idx])
        close = float(parts[close_idx])
        bars.setdefault(key2, []).append((ts, close))
    last_close = {k: sorted(v)[-1][1] for k, v in bars.items()}
    return last_close


def bs_call_put(S, K, T, r, sigma, q=0.0):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return max(0.0, S - K), max(0.0, K - S)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return max(call, 0.0), max(put, 0.0)


def implied_vol(price, S, K, T, r, is_call, q=0.0):
    if price <= 0 or T <= 0:
        return np.nan
    def f(sigma):
        c, p = bs_call_put(S, K, T, r, sigma, q)
        return (c if is_call else p) - price
    try:
        return brentq(f, 1e-4, 5.0, maxiter=100)
    except Exception:
        return np.nan


def load_mu_close():
    with open(UNIVERSE_PKL, "rb") as f:
        universe = pickle.load(f)
    df = universe["MU"].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df["Close"]


def find_atm_straddle(contracts_day, spot, entry_date):
    """Pick the real, actually-traded expiry closest to the target DTE
    window, then the real strike closest to ATM within that expiry, for
    BOTH call and put at that same (expiry, strike)."""
    candidates = {}
    for (expiry, right, strike), close in contracts_day.items():
        exp_date = pd.Timestamp(f"20{expiry[:2]}-{expiry[2:4]}-{expiry[4:6]}")
        dte = (exp_date - entry_date).days
        if TARGET_DTE_LO <= dte <= TARGET_DTE_HI:
            candidates.setdefault(expiry, []).append((strike, right, close, dte))
    if not candidates:
        return None
    # Prefer the expiry whose median DTE is closest to the midpoint of the target range
    target_mid = (TARGET_DTE_LO + TARGET_DTE_HI) / 2
    best_expiry = min(candidates, key=lambda e: abs(candidates[e][0][3] - target_mid))
    rows = candidates[best_expiry]
    strikes_with_both = {}
    for strike, right, close, dte in rows:
        strikes_with_both.setdefault(strike, {})[right] = close
    complete = {k: v for k, v in strikes_with_both.items() if "C" in v and "P" in v}
    if not complete:
        return None
    best_strike = min(complete, key=lambda k: abs(k - spot))
    dte_val = [dte for strike, right, close, dte in rows if strike == best_strike][0]
    return {"expiry": best_expiry, "strike": best_strike, "dte": dte_val,
            "call_px": complete[best_strike]["C"], "put_px": complete[best_strike]["P"]}


def get_exit_price(contracts_day, expiry, strike, right):
    return contracts_day.get((expiry, right, strike))


def main():
    close = load_mu_close()
    ret1d = close.pct_change()
    trading_days = close.index

    dip_days = [d for d in ret1d[ret1d <= -0.05].index if d >= pd.Timestamp("2025-01-01")]
    run_days = [d for d in ret1d[ret1d >= 0.05].index if d >= pd.Timestamp("2025-01-01")]

    rng = np.random.default_rng(11)
    post2025 = trading_days[trading_days >= pd.Timestamp("2025-01-01")]
    # exclude trigger days themselves and keep enough runway for the hold period
    eligible_baseline = post2025[:-HOLD_TRADING_DAYS - 5]
    baseline_days = list(pd.DatetimeIndex(rng.choice(eligible_baseline, size=15, replace=False)))

    sample_dip = list(pd.DatetimeIndex(dip_days))[::max(1, len(dip_days) // 10)][:10]
    sample_run = list(pd.DatetimeIndex(run_days))[::max(1, len(run_days) // 10)][:10]

    all_targets = [(d, "dip") for d in sample_dip] + [(d, "run") for d in sample_run] + \
                  [(d, "baseline") for d in baseline_days]
    print(f"Testing {len(all_targets)} trigger points ({len(sample_dip)} dip, {len(sample_run)} run, "
          f"{len(baseline_days)} baseline), target DTE {TARGET_DTE_LO}-{TARGET_DTE_HI}d, "
          f"hold={HOLD_TRADING_DAYS} trading days")

    s3 = s3_client()
    results = []
    for i, (entry_date, group) in enumerate(sorted(all_targets)):
        loc = trading_days.get_loc(entry_date)
        if loc + HOLD_TRADING_DAYS >= len(trading_days):
            continue
        exit_date = trading_days[loc + HOLD_TRADING_DAYS]
        entry_str, exit_str = entry_date.strftime("%Y-%m-%d"), exit_date.strftime("%Y-%m-%d")
        spot_entry = close.loc[entry_date]
        spot_exit = close.loc[exit_date]

        entry_contracts = fetch_mu_contracts_for_day(s3, entry_str)
        if "_error" in entry_contracts:
            print(f"  [{i+1}/{len(all_targets)}] {entry_str} ({group}): ENTRY FETCH ERROR {entry_contracts['_error']}")
            continue
        straddle = find_atm_straddle(entry_contracts, spot_entry, entry_date)
        if straddle is None:
            print(f"  [{i+1}/{len(all_targets)}] {entry_str} ({group}): no valid ATM straddle found -- skipping")
            continue

        exit_contracts = fetch_mu_contracts_for_day(s3, exit_str)
        if "_error" in exit_contracts:
            print(f"  [{i+1}/{len(all_targets)}] {entry_str} ({group}): EXIT FETCH ERROR {exit_contracts['_error']}")
            continue
        exit_call = get_exit_price(exit_contracts, straddle["expiry"], straddle["strike"], "C")
        exit_put = get_exit_price(exit_contracts, straddle["expiry"], straddle["strike"], "P")
        if exit_call is None or exit_put is None:
            print(f"  [{i+1}/{len(all_targets)}] {entry_str} ({group}): contract didn't trade on exit day -- skipping")
            continue

        T_entry = straddle["dte"] / 365.0
        T_exit = max((straddle["dte"] - HOLD_TRADING_DAYS * 1.4), 1) / 365.0  # ~calendar days for the trading-day hold
        iv_entry_call = implied_vol(straddle["call_px"], spot_entry, straddle["strike"], T_entry, RISK_FREE_RATE, True)
        iv_entry_put = implied_vol(straddle["put_px"], spot_entry, straddle["strike"], T_entry, RISK_FREE_RATE, False)
        iv_exit_call = implied_vol(exit_call, spot_exit, straddle["strike"], T_exit, RISK_FREE_RATE, True)
        iv_exit_put = implied_vol(exit_put, spot_exit, straddle["strike"], T_exit, RISK_FREE_RATE, False)

        entry_premium = straddle["call_px"] + straddle["put_px"]
        exit_value = exit_call + exit_put
        pnl_per_contract = (entry_premium - exit_value) * 100

        row = {
            "entry_date": entry_str, "group": group, "spot_entry": spot_entry, "spot_exit": spot_exit,
            "strike": straddle["strike"], "dte_entry": straddle["dte"],
            "entry_premium": entry_premium, "exit_value": exit_value, "pnl_per_contract": pnl_per_contract,
            "iv_entry_call": iv_entry_call, "iv_entry_put": iv_entry_put,
            "iv_exit_call": iv_exit_call, "iv_exit_put": iv_exit_put,
            "iv_entry_avg": np.nanmean([iv_entry_call, iv_entry_put]),
            "iv_exit_avg": np.nanmean([iv_exit_call, iv_exit_put]),
        }
        results.append(row)
        print(f"  [{i+1}/{len(all_targets)}] {entry_str} ({group}): strike={straddle['strike']} dte={straddle['dte']} "
              f"entry_prem=${entry_premium:.2f} exit_val=${exit_value:.2f} pnl=${pnl_per_contract:+.2f} "
              f"iv_entry={row['iv_entry_avg']:.1%} iv_exit={row['iv_exit_avg']:.1%}")

    df = pd.DataFrame(results)
    df.to_csv(HERE / "mu_vol_harvest_results.csv", index=False)
    print(f"\nSaved {len(df)} rows to mu_vol_harvest_results.csv")

    print("\n=== Short-straddle P&L and IV crush by group ===")
    for group in ["dip", "run", "baseline"]:
        sub = df[df["group"] == group]
        if len(sub) == 0:
            continue
        iv_crush = sub["iv_entry_avg"] - sub["iv_exit_avg"]
        print(f"  {group:10s}: n={len(sub):2d}  mean_pnl=${sub['pnl_per_contract'].mean():+.2f}  "
              f"mean_iv_entry={sub['iv_entry_avg'].mean():.1%}  mean_iv_exit={sub['iv_exit_avg'].mean():.1%}  "
              f"mean_iv_crush={iv_crush.mean():+.1%}")

    dip_pnl, run_pnl, base_pnl = df[df["group"]=="dip"]["pnl_per_contract"], df[df["group"]=="run"]["pnl_per_contract"], df[df["group"]=="baseline"]["pnl_per_contract"]
    if len(dip_pnl) >= 3 and len(base_pnl) >= 3:
        t1, p1 = stats.ttest_ind(dip_pnl, base_pnl, equal_var=False)
        print(f"\nDip-day short-straddle P&L vs baseline: t={t1:.2f} p={p1:.4f}")
    if len(run_pnl) >= 3 and len(base_pnl) >= 3:
        t2, p2 = stats.ttest_ind(run_pnl, base_pnl, equal_var=False)
        print(f"Run-day short-straddle P&L vs baseline: t={t2:.2f} p={p2:.4f}")

    dip_crush = (df[df["group"]=="dip"]["iv_entry_avg"] - df[df["group"]=="dip"]["iv_exit_avg"])
    run_crush = (df[df["group"]=="run"]["iv_entry_avg"] - df[df["group"]=="run"]["iv_exit_avg"])
    base_crush = (df[df["group"]=="baseline"]["iv_entry_avg"] - df[df["group"]=="baseline"]["iv_exit_avg"])
    if len(dip_crush) >= 3 and len(base_crush) >= 3:
        t3, p3 = stats.ttest_ind(dip_crush.dropna(), base_crush.dropna(), equal_var=False)
        print(f"Dip-day IV crush vs baseline: t={t3:.2f} p={p3:.4f}")
    if len(run_crush) >= 3 and len(base_crush) >= 3:
        t4, p4 = stats.ttest_ind(run_crush.dropna(), base_crush.dropna(), equal_var=False)
        print(f"Run-day IV crush vs baseline: t={t4:.2f} p={p4:.4f}")


if __name__ == "__main__":
    main()
