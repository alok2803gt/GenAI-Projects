"""
Tests the already-built and validated VWAP-deviation engine against a
genuinely different ticker universe: TSLA/COIN/PLTR, chosen for higher
volatility and less mega-cap market-efficiency than AAPL/MSFT/NVDA/SPY
(the original universe, among the most heavily-traded, closely-analyzed
names in the entire market). Reuses vwap_scalp_engine.py unchanged --
same strategy logic, same cost assumption -- only the input tickers
differ, to isolate whether tonight's negative result is a property of
these 4 specific highly-efficient names or of the strategy itself.

Runs a focused set of configs around the least-bad settings found on the
original universe (k=1.5/stop=1.5, plus two neighbors), not the full
24-combo grid -- this is a targeted follow-up check, not a fresh
from-scratch parameter search.
"""
import json
import pandas as pd

from vwap_scalp_engine import compute_vwap_deviation, simulate, summarize, ScalpConfig

TICKERS = ["TSLA", "COIN", "PLTR"]
CONFIGS = [
    ScalpConfig(entry_k=1.5, stop_mult=1.5, max_hold_bars=15),
    ScalpConfig(entry_k=2.0, stop_mult=1.5, max_hold_bars=15),
    ScalpConfig(entry_k=2.5, stop_mult=2.0, max_hold_bars=30),
]


def main():
    data = {}
    for tk in TICKERS:
        df = pd.read_pickle(f"bars_1min_cache/{tk}.pkl")
        data[tk] = df
        print(f"{tk}: {len(df)} bars")

    results = {}
    for cfg in CONFIGS:
        key = f"k{cfg.entry_k}_stop{cfg.stop_mult}_hold{cfg.max_hold_bars}"
        all_trades = []
        by_ticker = {}
        for tk, df in data.items():
            dev = compute_vwap_deviation(df)
            trades = simulate(dev, tk, cfg)
            all_trades.extend(trades)
            by_ticker[tk] = summarize(trades)
        results[key] = {"overall": summarize(all_trades), "by_ticker": by_ticker}
        print(f"{key}: {results[key]['overall']}")
        for tk, s in by_ticker.items():
            print(f"  {tk}: {s}")

    with open("highvol_universe_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved highvol_universe_results.json")


if __name__ == "__main__":
    main()
