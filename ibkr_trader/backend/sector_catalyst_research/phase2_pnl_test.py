"""
Phase 2: real $ P&L for the actual tradeable structure -- a cheap, far-OTM,
1-DTE call debit spread on the LAGGARD member of the Memory/Storage basket
(MU/WDC/STX/SNDK) on days the other members pop but it hasn't yet
(spillover signal, validated in phase1_signal_test.py: +11.75pp tail-rate
lift vs baseline).

VOL ASSUMPTION (stated explicitly): no historical intraday options data
exists for these tickers (same constraint as every other backtest in this
codebase). Calibrated a flat IV from the ONE real data point available --
the user's actual 2026-09-03 MU $1000/$1020 1-DTE call spread, entered at
MU's real close ($958.16), which cost a real $71 (=$0.71/spread). Solving
Black-Scholes for the IV that reproduces that price at T=1 trading day
gives 45.8%. MU's own trailing 20-day realized vol on that exact date was
50.4% -- an IV/RV multiplier of 0.91. That multiplier (not a flat vol
level) is applied to each ticker's OWN trailing 20-day realized vol on
each historical signal day, so the vol assumption moves with genuine
day-to-day/ticker-to-ticker conditions instead of using one constant
number for all of history. This is one calibration point applied
systematically -- a real anchor, not a guess, but still an approximation
worth stating plainly.

Entry: at signal day T's close, on the laggard ticker, spot = close_T.
Strikes: long = spot*(1+long_otm), short = long*(1+width) [see grid below].
Exit: T+1 close, INTRINSIC value (real spot, no more BS modeling needed
one day out from a already-short-dated spread) -- matches how the real
MU trade was economically valued (held to/near expiry).

Compares the SIGNAL-conditioned trades against the SAME structure entered
on every regular day for the same tickers (unconditional baseline), to
isolate whether the signal adds real value beyond "these are just volatile
enough stocks that far-OTM spreads pay off sometimes anyway."
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).parent
MEMORY = ["MU", "WDC", "STX", "SNDK"]

POP_RET_THRESH = 0.03
POP_VOL_MULT = 1.0
MIN_MOVERS = 2

IV_RV_MULTIPLIER = 0.9086  # calibrated from the real 2026-09-03 MU trade -- see docstring.
                             # SANITY CHECK FAILED: at this multiplier, even the BASELINE
                             # (buy this spread on a random day, no signal) shows huge positive
                             # mean returns (+69% to +150% on debit) -- a random cheap OTM
                             # spread should NOT have strongly positive expected value in a
                             # functioning market (real options carry a positive vol risk
                             # premium on average, IV > subsequent RV, not the reverse). The
                             # real MU trade was very likely a genuinely cheap outlier fill,
                             # not representative pricing -- calibrating the WHOLE model off
                             # one lucky data point was a mistake. Testing a realistic range
                             # instead (1.0x-1.3x, the standard empirical IV>RV relationship)
                             # and checking whether the SIGNAL's edge over baseline survives.
R = 0.0


def bs_call(spot, strike, t, vol, r=R):
    if t <= 1e-8 or vol <= 1e-6:
        return max(spot - strike, 0.0)
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    return spot * norm.cdf(d1) - strike * np.exp(-r * t) * norm.cdf(d2)


def spread_value(spot, k1, k2, t, vol):
    return bs_call(spot, k1, t, vol) - bs_call(spot, k2, t, vol)


def prep(df):
    df = df.copy()
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df["ret"] = df["Close"].pct_change()
    df["vol_avg20"] = df["Volume"].rolling(20, min_periods=15).mean()
    df["rel_vol"] = df["Volume"] / df["vol_avg20"].shift(1)
    df["popped"] = (df["ret"] >= POP_RET_THRESH) & (df["rel_vol"] >= POP_VOL_MULT)
    df["rv20"] = df["ret"].rolling(20, min_periods=15).std() * np.sqrt(252)
    df["assumed_iv"] = df["rv20"] * IV_RV_MULTIPLIER
    df["fwd_close_1d"] = df["Close"].shift(-1)
    return df


def price_trade(spot, iv, long_otm, width_pct, t_days=1.0):
    k1 = round(spot * (1 + long_otm), 0)
    k2 = round(k1 * (1 + width_pct), 0)
    debit = spread_value(spot, k1, k2, t_days / 252, iv)
    return k1, k2, debit


def simulate(days_by_ticker, frames, long_otm, width_pct, label, iv_mult):
    rows = []
    for t, f in frames.items():
        sub_days = days_by_ticker[t].intersection(f.index)
        for day in sub_days:
            row = f.loc[day]
            spot, rv, fwd = row["Close"], row["rv20"], row["fwd_close_1d"]
            iv = rv * iv_mult if pd.notna(rv) else np.nan
            if pd.isna(iv) or pd.isna(fwd) or iv <= 0:
                continue
            k1, k2, debit = price_trade(spot, iv, long_otm, width_pct)
            if debit <= 0.005:
                continue
            terminal = max(min(fwd, k2) - k1, 0.0)
            payoff = terminal - debit
            rows.append({"ticker": t, "date": day, "debit": debit, "payoff": payoff,
                         "k1": k1, "k2": k2, "spot": spot})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"  {label}: no trades")
        return df
    df["win"] = df["payoff"] > 0
    df["ret_on_debit"] = df["payoff"] / df["debit"]
    print(f"    {label:22s} n={len(df):5d}  win={100*df.win.mean():5.1f}%  "
          f"mean_debit=${100*df.debit.mean():6.2f}  mean_payoff=${100*df.payoff.mean():+7.2f}  "
          f"mean_ret_on_debit={100*df.ret_on_debit.mean():+7.1f}%  median_ret_on_debit={100*df.ret_on_debit.median():+6.1f}%")
    return df


def main():
    with open(HERE / "semis_universe_5y_ohlcv.pkl", "rb") as f:
        universe = pickle.load(f)
    frames = {t: prep(universe[t]) for t in MEMORY}

    pop_counts = pd.DataFrame({t: f["popped"] for t, f in frames.items()}).sum(axis=1)
    hot_days = pop_counts[pop_counts >= MIN_MOVERS].index

    spillover_by_ticker = {}
    for t, f in frames.items():
        hot = hot_days.intersection(f.index)
        was_popper = f.loc[hot, "popped"]
        spillover_by_ticker[t] = hot[~was_popper.values]

    grid = [(0.04, 0.02), (0.05, 0.02), (0.04, 0.03), (0.05, 0.03)]
    iv_mults = [1.0, 1.15, 1.3]

    all_days_by_ticker = {t: f.dropna(subset=["rv20", "fwd_close_1d"]).index for t, f in frames.items()}

    for iv_mult in iv_mults:
        print(f"\n{'#'*80}\nIV = {iv_mult}x trailing 20d realized vol\n{'#'*80}")
        for long_otm, width in grid:
            print(f"  --- long_otm={long_otm*100:.0f}%  width={width*100:.0f}% ---")
            simulate(spillover_by_ticker, frames, long_otm, width, "SIGNAL (laggard)", iv_mult)
            simulate(all_days_by_ticker, frames, long_otm, width, "BASELINE (every day)", iv_mult)


if __name__ == "__main__":
    main()
