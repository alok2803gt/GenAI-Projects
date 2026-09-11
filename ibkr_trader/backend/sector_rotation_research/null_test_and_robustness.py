"""
Deeper validity check on relative_rotation.py's headline finding
(Hyperscalers vs Memory relative-return correlation = -0.71). Three real
risks that finding alone doesn't rule out:

1. MECHANICAL ARTIFACT: with only 5 baskets, subtracting the daily
   cross-sectional average forces the 5 relative returns to sum to ~0
   every day. That alone induces SOME negative correlation among
   differently-volatile series even with ZERO true economic
   relationship -- a classic compositional-data / demeaning artifact.
   Test: Monte Carlo null -- simulate 5 independent random return series
   matched to each real basket's own volatility (no true correlation
   between them by construction), demean the same way, and see what
   relative-correlation pattern falls out of the mechanical constraint
   ALONE. If the real -0.71 sits far outside this null distribution,
   that's real evidence of rotation, not an artifact.

2. BETA ARTIFACT: Memory (MU/SNDK) is small-cap/high-beta; Hyperscalers
   are mega-cap/low-beta. On a big common-factor day, a low-beta group's
   raw return undershoots the average (negative relative return) while a
   high-beta group's overshoots (positive relative return) almost by
   construction -- looks like "rotation" but is really just beta
   dispersion around a shared move. Test: regress each basket's raw
   return on the common factor (OLS beta), then check whether the
   relative-return correlation survives after controlling for beta
   differences (partial correlation / regress relative returns on
   |common factor| and check residuals).

3. OUTLIER-DRIVEN: check if a handful of extreme days are doing all the
   work (Spearman rank correlation vs Pearson; correlation with the top
   5 |common-factor| days removed).
"""
import pickle
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
N_SIMS = 5000


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
    all_df = pd.concat(basket_ret, axis=1, join="inner")
    names = list(BASKETS.keys())
    n_days = len(all_df)
    print(f"Common window: {n_days} days")

    daily_avg = all_df.mean(axis=1)
    rel = all_df.sub(daily_avg, axis=0)
    real_corr = rel["Hyperscalers"].corr(rel["Memory"])
    print(f"\nReal Hyperscalers-vs-Memory relative correlation: {real_corr:+.4f}")

    # ── 1. Monte Carlo null: mechanical artifact check ─────────────────
    print(f"\n=== Test 1: Monte Carlo null ({N_SIMS} sims) ===")
    print("Simulating 5 INDEPENDENT random series, each matched to the real basket's own "
          "vol, demeaned the same way -- what correlation does the zero-sum constraint alone produce?")
    vols = all_df.std().values
    rng = np.random.default_rng(42)
    null_corrs = np.empty(N_SIMS)
    hyp_idx, mem_idx = names.index("Hyperscalers"), names.index("Memory")
    for i in range(N_SIMS):
        sim = rng.normal(loc=0, scale=vols, size=(n_days, len(names)))
        sim_avg = sim.mean(axis=1, keepdims=True)
        sim_rel = sim - sim_avg
        null_corrs[i] = np.corrcoef(sim_rel[:, hyp_idx], sim_rel[:, mem_idx])[0, 1]

    null_mean, null_std = null_corrs.mean(), null_corrs.std()
    pctile = (null_corrs < real_corr).mean() * 100
    z = (real_corr - null_mean) / null_std
    print(f"Null distribution (mechanical artifact only): mean={null_mean:+.4f}  std={null_std:.4f}")
    print(f"Real value {real_corr:+.4f} sits at the {pctile:.2f}th percentile of the null "
          f"(z={z:.2f})")
    if pctile < 1 or pctile > 99:
        print(">>> Real correlation is far outside what the mechanical zero-sum constraint alone "
              "produces -- this is NOT just a demeaning artifact.")
    else:
        print(">>> Real correlation is WITHIN the range the mechanical constraint alone would "
              "produce -- cannot rule out this being a pure artifact.")

    # ── 2. Beta artifact check ───────────────────────────────────────
    print("\n=== Test 2: Beta-to-common-factor check ===")
    betas = {}
    for name in names:
        slope, intercept, r, p, se = stats.linregress(daily_avg, all_df[name])
        betas[name] = slope
        print(f"  {name:14s}: beta to common factor = {slope:.3f}")
    print(f"\nHyperscalers beta ({betas['Hyperscalers']:.3f}) vs Memory beta ({betas['Memory']:.3f}) --"
          f" if Memory's beta is much higher, some of the relative-return divergence on big days"
          f" is mechanical beta dispersion, not necessarily 'rotation'.")

    # Partial correlation: control for |common factor| magnitude (vol-of-the-day proxy)
    abs_factor = daily_avg.abs()
    resid = {}
    for name in ["Hyperscalers", "Memory"]:
        slope, intercept, r, p, se = stats.linregress(abs_factor, rel[name])
        resid[name] = rel[name] - (intercept + slope * abs_factor)
    partial_corr = resid["Hyperscalers"].corr(resid["Memory"])
    print(f"\nRelative-return correlation AFTER controlling for |common factor| magnitude: "
          f"{partial_corr:+.4f} (raw relative corr was {real_corr:+.4f})")
    if abs(partial_corr) > abs(real_corr) * 0.7:
        print(">>> Survives controlling for common-factor magnitude -- not primarily a beta-dispersion artifact.")
    else:
        print(">>> Substantially weakens once controlling for common-factor magnitude -- "
              "beta dispersion explains a meaningful part of the raw finding.")

    # ── 3. Outlier sensitivity ───────────────────────────────────────
    print("\n=== Test 3: Outlier sensitivity ===")
    spearman_corr, sp_p = stats.spearmanr(rel["Hyperscalers"], rel["Memory"])
    print(f"Pearson: {real_corr:+.4f}   Spearman (rank, outlier-robust): {spearman_corr:+.4f}")

    top5_days = abs_factor.nlargest(5).index
    rel_trimmed = rel.drop(index=top5_days)
    trimmed_corr = rel_trimmed["Hyperscalers"].corr(rel_trimmed["Memory"])
    print(f"With top-5 highest-|common-factor| days removed: {trimmed_corr:+.4f} "
          f"(full sample was {real_corr:+.4f})")


if __name__ == "__main__":
    main()
