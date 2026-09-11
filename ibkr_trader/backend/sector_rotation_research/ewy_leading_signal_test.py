"""
Tests whether EWY's overnight gap is a genuine, tradeable PRE-MARKET
leading signal for MU, vs. just "these two happen to move together"
(the same-day correlation already found in macro_micro_round2.py).

Key distinction: EWY's overnight gap (today's open vs yesterday's close)
is fully known by 9:30am ET, before MU's regular session even starts.
MU's OWN overnight gap reflects roughly the same overnight information
(pre-market/futures react to similar macro/Asia-session news) -- so a
simple "EWY gap vs MU full-day return" test would partly just be
detecting that both react to the same overnight news simultaneously, not
proving EWY tells you anything NEW about MU once you can already see MU's
own pre-market indication.

The cleaner, real test: does EWY's overnight gap predict MU's INTRADAY
return (today's open -> today's close, i.e. what happens AFTER the open,
which is NOT yet knowable from MU's own gap alone)? That isolates whether
EWY's move adds independent forward-looking information about how MU
trades DURING the day, beyond whatever MU's own gap already reflects.

Three tests:
  1. EWY overnight gap vs MU overnight gap -- how much do they already
     move together overnight (same info, simultaneous)?
  2. EWY overnight gap vs MU INTRADAY return (open->close) -- the real
     leading-signal test.
  3. Same as #2, but conditioned on the RESIDUAL EWY gap after removing
     MU's own gap (i.e. does EWY still add information on top of what
     MU's own pre-market move already told you?).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

HERE = Path(__file__).parent


def main():
    # Use the same window as the rest of this study (SNDK/Memory-basket window)
    start = pd.Timestamp("2025-02-14")
    end = pd.Timestamp("2026-08-24")

    ewy = yf.Ticker("EWY").history(start=start, end=end + pd.Timedelta(days=1))
    mu = yf.Ticker("MU").history(start=start, end=end + pd.Timedelta(days=1))
    ewy.index = pd.to_datetime(ewy.index).tz_localize(None)
    mu.index = pd.to_datetime(mu.index).tz_localize(None)

    ewy_prior_close = ewy["Close"].shift(1)
    mu_prior_close = mu["Close"].shift(1)

    ewy_gap = (ewy["Open"] / ewy_prior_close - 1).rename("ewy_gap")
    mu_gap = (mu["Open"] / mu_prior_close - 1).rename("mu_gap")
    mu_intraday = (mu["Close"] / mu["Open"] - 1).rename("mu_intraday")
    mu_full_day = (mu["Close"] / mu_prior_close - 1).rename("mu_full_day")

    df = pd.concat([ewy_gap, mu_gap, mu_intraday, mu_full_day], axis=1).dropna()
    print(f"n={len(df)} trading days, {df.index[0].date()} -> {df.index[-1].date()}")

    # ── 1. Do EWY and MU gap together (same overnight info)? ────────
    print("\n=== Test 1: EWY overnight gap vs MU overnight gap (same-info check) ===")
    r1 = df["ewy_gap"].corr(df["mu_gap"])
    print(f"r={r1:+.3f}  (high = they already move together pre-market, as expected if both "
          f"react to the same overnight news)")

    # ── 2. Does EWY's gap predict MU's INTRADAY (open->close) move? ──
    print("\n=== Test 2: EWY overnight gap vs MU INTRADAY return (open->close) -- the real test ===")
    r2 = df["ewy_gap"].corr(df["mu_intraday"])
    slope, intercept, r_val, p_val, se = stats.linregress(df["ewy_gap"], df["mu_intraday"])
    print(f"r={r2:+.3f}  p={p_val:.4f}  slope={slope:.3f}  (slope = expected MU intraday move per "
          f"1.0 unit of EWY overnight gap)")
    baseline_intraday = df["mu_intraday"].mean()
    top_q = df["ewy_gap"].quantile(0.8)
    bot_q = df["ewy_gap"].quantile(0.2)
    top_days = df[df["ewy_gap"] >= top_q]
    bot_days = df[df["ewy_gap"] <= bot_q]
    print(f"MU's own unconditional mean intraday return: {baseline_intraday*100:+.4f}%")
    print(f"MU intraday return on EWY's TOP-quintile gap-up days (n={len(top_days)}): "
          f"{top_days['mu_intraday'].mean()*100:+.4f}%")
    print(f"MU intraday return on EWY's BOTTOM-quintile gap-down days (n={len(bot_days)}): "
          f"{bot_days['mu_intraday'].mean()*100:+.4f}%")
    t_top, p_top = stats.ttest_1samp(top_days["mu_intraday"], baseline_intraday)
    t_bot, p_bot = stats.ttest_1samp(bot_days["mu_intraday"], baseline_intraday)
    print(f"  top-quintile vs baseline: t={t_top:.2f} p={p_top:.4f}")
    print(f"  bottom-quintile vs baseline: t={t_bot:.2f} p={p_bot:.4f}")

    # ── 3. Does EWY add info BEYOND MU's own gap? ────────────────────
    print("\n=== Test 3: Does EWY's gap add info beyond MU's OWN gap already? ===")
    slope_mg, int_mg, *_ = stats.linregress(df["mu_gap"], df["ewy_gap"])
    ewy_resid = df["ewy_gap"] - (int_mg + slope_mg * df["mu_gap"])
    r3 = ewy_resid.corr(df["mu_intraday"])
    slope3, int3, r3v, p3, se3 = stats.linregress(ewy_resid, df["mu_intraday"])
    print(f"EWY gap residual (after removing what's explained by MU's own gap) vs MU intraday: "
          f"r={r3:+.3f}  p={p3:.4f}")
    print("(If this is still meaningfully non-zero and significant, EWY tells you something REAL "
          "and NEW beyond MU's own pre-market indication -- a genuine incremental signal, not just "
          "duplicate information you'd already see by watching MU's own gap.)")


if __name__ == "__main__":
    main()
