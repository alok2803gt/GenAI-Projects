"""
Per-ticker robustness check for the bullish harami + downtrend-context
signal, same discipline as squeeze_momentum_research/sqzmom_per_ticker.py:
each ticker compared against ITS OWN downtrend-matched baseline (not the
universe-wide one), with a minimum trade count and Bonferroni correction
for testing many tickers at once. The aggregate result already showed a
real, significant lift (5d: p=0.00007) -- this checks whether that's
broad-based across the universe or concentrated in a handful of names.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from harami_backtest import backtest_ticker

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
MIN_TRADES = 8
HOLD_PERIODS = [5, 10, 20]


def main():
    with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
        universe = pickle.load(f)

    for hold in HOLD_PERIODS:
        rows = []
        for ticker, df in universe.items():
            df = df.copy()
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            df = df.reset_index(drop=True)

            close = df["Close"]
            sma20, sma50 = close.rolling(20).mean(), close.rolling(50).mean()
            downtrend = (close < sma20) & (close < sma50)
            fwd = (close.shift(-hold) / close - 1) * 100
            own_baseline = fwd[downtrend].dropna()
            if len(own_baseline) < 20:
                continue

            try:
                sig_rets = backtest_ticker(df, hold, True)
            except Exception:
                continue
            if len(sig_rets) < MIN_TRADES:
                continue

            sig_rets = np.array(sig_rets)
            base_mean = own_baseline.mean()
            lift = sig_rets.mean() - base_mean
            tstat, pval = stats.ttest_1samp(sig_rets - base_mean, 0)
            rows.append({
                "ticker": ticker, "n": len(sig_rets),
                "signal_mean": sig_rets.mean(), "signal_win": (sig_rets > 0).mean() * 100,
                "baseline_mean": base_mean, "lift": lift, "p_value": pval,
            })

        rdf = pd.DataFrame(rows).sort_values("lift", ascending=False)
        n_tested = len(rdf)
        bonf_alpha = 0.05 / max(n_tested, 1)

        print(f"\n{'='*95}\nhold={hold}d -- per-ticker vs OWN downtrend-matched baseline "
              f"(n>={MIN_TRADES} trades required, {n_tested} tickers qualify)")
        print(f"Bonferroni bar for {n_tested} simultaneous tests: p < {bonf_alpha:.5f}\n")
        print(f"{'TICKER':8s}{'n':>5s}{'sig_mean':>10s}{'sig_win%':>10s}{'base_mean':>10s}{'lift':>9s}{'p_value':>10s}")
        for _, r in rdf.iterrows():
            flag = " ***" if r["p_value"] < bonf_alpha else (" *" if r["p_value"] < 0.05 else "")
            print(f"{r['ticker']:8s}{r['n']:5.0f}{r['signal_mean']:+9.2f}%{r['signal_win']:9.1f}%"
                  f"{r['baseline_mean']:+9.2f}%{r['lift']:+8.2f}%{r['p_value']:9.4f}{flag}")

        n_pos_lift = (rdf["lift"] > 0).sum()
        n_nominal = (rdf["p_value"] < 0.05).sum()
        n_survive = (rdf["p_value"] < bonf_alpha).sum()
        print(f"\n{n_pos_lift}/{n_tested} tickers show positive lift (broad-based check)")
        print(f"{n_nominal}/{n_tested} nominally significant (p<0.05, uncorrected)")
        print(f"{n_survive}/{n_tested} survive Bonferroni correction")
        rdf.to_csv(HERE / f"harami_per_ticker_hold{hold}.csv", index=False)


if __name__ == "__main__":
    main()
