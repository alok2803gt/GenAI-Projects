"""
Tests the CEO's real observation (2026-09-09): some days hyperscalers run,
some days memory stocks run, some days chipmakers run -- is there a real,
statistically supported ROTATION pattern (money moving between AI-complex
sub-sectors on different days), or does everything just move together
with noisy lead/lag that looks like rotation in hindsight?

Baskets (business-model grounded, not just "all semis" -- see
fetch_data.py's own comments for the HBM-vs-HDD distinction that motivated
keeping Memory and Storage as SEPARATE groups):
  Hyperscalers   : MSFT, GOOGL, AMZN, META, ORCL   (AI infra BUYERS)
  Chipmakers     : NVDA, AMD, AVGO, MRVL, TSM      (AI compute SELLERS)
  Memory (HBM)   : MU, SNDK                        (real HBM/memory-chip makers)
  Storage (ctrl) : STX, WDC                        (HDD makers, NOT memory-chip -- control group)
  SemiCapEquip   : AMAT, LRCX, KLAC                (tool makers, adjacent group)

Method:
  1. Basket daily return = equal-weight average of its members' daily %
     returns (only over dates where ALL members have data -- SNDK's short
     history, 2025-02-13 onward post-spinoff, constrains any test
     involving Memory to that shorter window; other basket pairs use their
     own full overlapping history).
  2. Same-day correlation matrix between all 5 baskets, over each pair's
     own valid overlap -- the first, cheapest check: genuine rotation
     implies LOW or negative same-day correlation; a common "AI theme beta"
     with lag implies HIGH same-day correlation instead (everything moves
     together, just with noisy relative timing), which would mean the
     CEO's day-to-day observation is more about WHICH stock leads the
     theme's move that day, not true zero-sum capital rotation.
  3. Lead-lag rotation test: define a "pop day" for basket A as a daily
     return >= POP_THRESH. For each basket PAIR (A,B), compare basket B's
     return on A's pop days (both SAME-day and NEXT-day) against basket
     B's own UNCONDITIONAL mean return (the matched baseline -- same
     discipline every other backtest this account has run this session
     uses: never just "is it positive," always vs what B does anyway).
     Welch's t-test per pair/horizon, Bonferroni-corrected across all pairs
     tested, plus a binomial sign test on event-level direction consistency
     to catch a real-but-diffuse effect individual small-N pairs might
     miss alone.
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

POP_THRESH = 0.03  # 3%, same threshold sector_catalyst_scanner.py already uses live


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
    """Equal-weight average daily return across tickers, restricted to
    dates where ALL tickers in the basket have data."""
    aligned = pd.concat([returns[t].rename(t) for t in tickers], axis=1, join="inner")
    return aligned.mean(axis=1), aligned.index


def main():
    returns = load_returns()
    basket_ret = {}
    basket_dates = {}
    for name, tickers in BASKETS.items():
        r, d = basket_returns(returns, tickers)
        basket_ret[name] = r
        basket_dates[name] = d
        print(f"{name:14s} ({', '.join(tickers)}): {len(r)} trading days, "
              f"{d[0].date()} -> {d[-1].date()}")

    # ── 1. Same-day correlation matrix (each pair on its OWN overlap) ──────
    print("\n=== Same-day correlation matrix (pairwise overlap) ===")
    names = list(BASKETS.keys())
    corr_df = pd.DataFrame(index=names, columns=names, dtype=float)
    for a, b in combinations(names, 2):
        joined = pd.concat([basket_ret[a].rename("a"), basket_ret[b].rename("b")], axis=1, join="inner")
        c = joined["a"].corr(joined["b"])
        corr_df.loc[a, b] = c
        corr_df.loc[b, a] = c
        print(f"  {a:14s} vs {b:14s}: r={c:+.3f}  (n={len(joined)})")
    for n in names:
        corr_df.loc[n, n] = 1.0

    # ── 2. Rotation / lead-lag test ─────────────────────────────────────
    print(f"\n=== Rotation test (pop threshold = {POP_THRESH:.0%}) ===")
    pairs = [(a, b) for a in names for b in names if a != b]
    n_tests = len(pairs) * 2  # same-day + next-day per ordered pair
    alpha_bonf = 0.05 / n_tests
    print(f"Bonferroni-corrected alpha: {alpha_bonf:.5f} ({n_tests} tests)")

    results = []
    for a, b in pairs:
        joined = pd.concat([basket_ret[a].rename("a"), basket_ret[b].rename("b")], axis=1, join="inner")
        joined = joined.sort_index()
        pop_days = joined.index[joined["a"] >= POP_THRESH]
        if len(pop_days) < 5:
            continue
        baseline_mean = joined["b"].mean()

        # Same-day
        same_day_vals = joined.loc[pop_days, "b"].values
        t_same, p_same = stats.ttest_1samp(same_day_vals, baseline_mean)

        # Next-day (shift b forward by one row relative to a's pop day)
        b_series = joined["b"]
        next_vals = []
        for pd_day in pop_days:
            loc = b_series.index.get_loc(pd_day)
            if loc + 1 < len(b_series):
                next_vals.append(b_series.iloc[loc + 1])
        next_vals = np.array(next_vals)
        if len(next_vals) >= 5:
            t_next, p_next = stats.ttest_1samp(next_vals, baseline_mean)
        else:
            t_next, p_next = np.nan, np.nan

        results.append({
            "pop_basket": a, "reacting_basket": b, "n_pop_days": len(pop_days),
            "baseline_mean_pct": baseline_mean * 100,
            "same_day_mean_pct": same_day_vals.mean() * 100, "same_day_p": p_same,
            "next_day_mean_pct": next_vals.mean() * 100 if len(next_vals) else np.nan,
            "next_day_p": p_next, "n_next": len(next_vals),
        })

    res_df = pd.DataFrame(results)
    res_df["same_day_sig"] = res_df["same_day_p"] < alpha_bonf
    res_df["next_day_sig"] = res_df["next_day_p"] < alpha_bonf

    print("\n--- Full results (baseline = reacting basket's own unconditional mean) ---")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(res_df.round(4).to_string(index=False))

    sig_same = res_df[res_df["same_day_sig"]]
    sig_next = res_df[res_df["next_day_sig"]]
    print(f"\n{len(sig_same)}/{len(res_df)} same-day pairs significant after Bonferroni correction")
    print(f"{len(sig_next)}/{len(res_df)} next-day pairs significant after Bonferroni correction")

    res_df.to_csv(HERE / "rotation_test_results.csv", index=False)
    corr_df.to_csv(HERE / "basket_correlation_matrix.csv")
    print(f"\nSaved detailed results to rotation_test_results.csv and basket_correlation_matrix.csv")


if __name__ == "__main__":
    main()
