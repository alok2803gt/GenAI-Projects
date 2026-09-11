"""
Follow-up to rotation_analysis.py. That script found same-day correlations
strongly positive (0.37-0.81) across every basket pair and ZERO significant
next-day lead-lag effects -- i.e. no evidence of true zero-sum capital
rotation (money leaving one sub-sector for another). But the CEO's
observation ("some days hyperscalers run, some days memory runs") doesn't
require zero-sum rotation to be real -- it could still be true in a
RELATIVE sense: on a given day, the AI complex as a whole moves on shared
beta, but WHICH basket leads that day's move varies, and *that* day-to-day
leadership rotation could itself have structure (mean-reversion or
momentum in relative performance) even though the raw levels are
positively correlated.

Method: for each day, compute each basket's return MINUS the equal-weight
average return across all 5 baskets that day (the "relative return" --
what's left after removing the common AI-theme factor). Then:
  1. Same-day correlation of relative returns across basket pairs -- if
     true relative rotation exists, this should be MORE negative than the
     raw-return correlations in rotation_analysis.py.
  2. Day-over-day autocorrelation of each basket's OWN relative return
     (does today's leader tend to be tomorrow's leader too [momentum] or
     tomorrow's laggard [mean-reversion]?) -- lag-1 autocorrelation +
     Welch t-test of next-day relative return conditional on being in the
     top tercile of relative return today, vs the basket's own
     unconditional relative-return mean.
"""
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
PKL = HERE / "sector_rotation_ohlcv.pkl"

BASKETS = {
    "Hyperscalers": ["MSFT", "GOOGL", "AMZN", "META", "ORCL"],
    "Chipmakers":   ["NVDA", "AMD", "AVGO", "MRVL", "TSM"],
    "Memory":       ["MU", "SNDK"],
    "Storage":      ["STX", "WDC"],
    "SemiCapEquip": ["AMAT", "LRCX", "KLAC"],
}


def load_returns():
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    returns = {}
    for ticker, df in data.items():
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        returns[ticker] = df["Close"].pct_change().dropna()
    return returns


def basket_returns(returns, tickers):
    aligned = pd.concat([returns[t].rename(t) for t in tickers], axis=1, join="inner")
    return aligned.mean(axis=1)


def main():
    returns = load_returns()
    basket_ret = {name: basket_returns(returns, tickers) for name, tickers in BASKETS.items()}

    # Common date range across ALL 5 baskets (Memory/SNDK is the binding constraint)
    all_df = pd.concat(basket_ret, axis=1, join="inner")
    print(f"Common window across all 5 baskets: {len(all_df)} days, "
          f"{all_df.index[0].date()} -> {all_df.index[-1].date()}")

    # Relative return = basket return - that day's equal-weight average across all 5
    daily_avg = all_df.mean(axis=1)
    rel = all_df.sub(daily_avg, axis=0)

    print("\n=== Same-day RELATIVE-return correlation (common factor removed) ===")
    names = list(BASKETS.keys())
    rel_corr = pd.DataFrame(index=names, columns=names, dtype=float)
    for a, b in combinations(names, 2):
        c = rel[a].corr(rel[b])
        rel_corr.loc[a, b] = c
        rel_corr.loc[b, a] = c
        print(f"  {a:14s} vs {b:14s}: r={c:+.3f}")
    for n in names:
        rel_corr.loc[n, n] = 1.0

    print("\n(Compare to rotation_analysis.py's raw-return correlations, all +0.37 to +0.81 --"
          " if these are notably more negative, that's the relative-rotation signature.)")

    # Day-over-day: does today's relative leader tend to lead or lag tomorrow?
    print("\n=== Own-basket relative-return persistence (lag-1) ===")
    n_tests = len(names)
    alpha_bonf = 0.05 / n_tests
    persistence_rows = []
    for name in names:
        s = rel[name].dropna()
        lag1_autocorr = s.autocorr(lag=1)

        # Top-tercile "today's leader" days -> next-day relative return vs own baseline
        thresh = s.quantile(2 / 3)
        top_days = s.index[s >= thresh][:-1]  # drop last if no next day
        next_vals = []
        for d in top_days:
            loc = s.index.get_loc(d)
            if loc + 1 < len(s):
                next_vals.append(s.iloc[loc + 1])
        next_vals = np.array(next_vals)
        baseline = s.mean()
        t, p = stats.ttest_1samp(next_vals, baseline) if len(next_vals) >= 5 else (np.nan, np.nan)
        persistence_rows.append({
            "basket": name, "lag1_autocorr": lag1_autocorr,
            "n_top_days": len(top_days), "baseline_rel_pct": baseline * 100,
            "next_day_rel_pct_after_leading": next_vals.mean() * 100 if len(next_vals) else np.nan,
            "p_value": p, "significant": (p < alpha_bonf) if not np.isnan(p) else False,
        })

    pers_df = pd.DataFrame(persistence_rows)
    print(f"Bonferroni-corrected alpha: {alpha_bonf:.5f} ({n_tests} tests)")
    with pd.option_context("display.width", 160):
        print(pers_df.round(4).to_string(index=False))

    rel_corr.to_csv(HERE / "relative_correlation_matrix.csv")
    pers_df.to_csv(HERE / "relative_persistence_results.csv", index=False)
    print("\nSaved relative_correlation_matrix.csv and relative_persistence_results.csv")


if __name__ == "__main__":
    main()
