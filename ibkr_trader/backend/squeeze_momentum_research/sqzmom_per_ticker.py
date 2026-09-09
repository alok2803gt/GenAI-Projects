"""
Per-ticker breakdown of the (unconditional, non-trend-filtered) squeeze-fire
LONG signal, each compared against that SAME ticker's own baseline forward
return -- not the universe-wide baseline, since some tickers just drift up
more than others regardless of any signal.

Explicit guard against data-mining: with 112 tickers, ranking by raw lift
and picking "winners" is close to the definition of overfitting -- some
will look good by pure chance even with zero true effect anywhere. This
script reports a one-sample t-test p-value on each ticker's per-trade
returns AND requires a minimum trade count before a ticker is even shown,
specifically so a "the top 5 look great" finding doesn't get reported as
if it were discovered evidence rather than what it actually is: the
expected noise ceiling from testing 112 things at once.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sqzmom_backtest import compute_sqzmom, MAX_HOLDS

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
MIN_TRADES = 15


def backtest_ticker_returns(df, max_hold):
    """Same mechanics as sqzmom_backtest.backtest_ticker, LONG only,
    returns just the list of trade returns for this one ticker."""
    df = compute_sqzmom(df)
    df = df.dropna(subset=["val", "sqzOn", "sqzOff"]).reset_index(drop=True)
    rets = []
    i, n = 1, len(df)
    while i < n - 1:
        fired = bool(df["sqzOn"].iloc[i - 1]) and bool(df["sqzOff"].iloc[i])
        if not fired or df["val"].iloc[i] <= 0:
            i += 1
            continue
        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry_price = df["Open"].iloc[entry_idx]
        exit_idx = None
        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
            if df["val"].iloc[j - 1] > 0 and df["val"].iloc[j] <= 0:
                exit_idx = j
                break
        if exit_idx is None:
            exit_idx = min(entry_idx + max_hold, n - 1)
        ret_pct = (df["Close"].iloc[exit_idx] / entry_price - 1) * 100
        rets.append(ret_pct)
        i = exit_idx + 1
    return rets


def main():
    with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
        universe = pickle.load(f)

    for max_hold in MAX_HOLDS:
        rows = []
        for ticker, df in universe.items():
            df = df.copy()
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            df = df.reset_index(drop=True)

            close = df["Close"]
            baseline = ((close.shift(-max_hold) / close - 1) * 100).dropna()

            try:
                sig_rets = backtest_ticker_returns(df, max_hold)
            except Exception:
                continue
            if len(sig_rets) < MIN_TRADES:
                continue

            sig_rets = np.array(sig_rets)
            lift = sig_rets.mean() - baseline.mean()
            # One-sample t-test: is this ticker's signal mean different from
            # ITS OWN baseline mean? (tests the lift, not just "is return > 0")
            tstat, pval = stats.ttest_1samp(sig_rets - baseline.mean(), 0)
            rows.append({
                "ticker": ticker, "n": len(sig_rets),
                "signal_mean": sig_rets.mean(), "signal_win": (sig_rets > 0).mean() * 100,
                "baseline_mean": baseline.mean(), "lift": lift, "p_value": pval,
            })

        rdf = pd.DataFrame(rows).sort_values("lift", ascending=False)
        n_tested = len(rdf)
        # Bonferroni: with n_tested independent-ish tickers tested at once,
        # the "would survive multiple-comparisons correction" bar
        bonferroni_alpha = 0.05 / max(n_tested, 1)

        print(f"\n{'='*100}\nmax_hold={max_hold}d LONG signal, per-ticker vs OWN baseline "
              f"(n>={MIN_TRADES} trades required, {n_tested} tickers qualify)")
        print(f"Bonferroni-corrected significance bar for {n_tested} simultaneous tests: p < {bonferroni_alpha:.5f}")
        print(f"\n{'TICKER':8s}{'n':>5s}{'sig_mean':>10s}{'sig_win%':>10s}{'base_mean':>10s}{'lift':>9s}{'p_value':>10s}")
        for _, r in rdf.iterrows():
            flag = " ***" if r["p_value"] < bonferroni_alpha else (" *" if r["p_value"] < 0.05 else "")
            print(f"{r['ticker']:8s}{r['n']:5.0f}{r['signal_mean']:+9.2f}%{r['signal_win']:9.1f}%"
                  f"{r['baseline_mean']:+9.2f}%{r['lift']:+8.2f}%{r['p_value']:9.4f}{flag}")

        survivors = rdf[rdf["p_value"] < bonferroni_alpha]
        print(f"\n{len(survivors)}/{n_tested} tickers survive Bonferroni correction "
              f"(the real bar for 'don't just be looking at noise')")
        rdf.to_csv(HERE / f"sqzmom_per_ticker_maxhold{max_hold}.csv", index=False)


if __name__ == "__main__":
    main()
