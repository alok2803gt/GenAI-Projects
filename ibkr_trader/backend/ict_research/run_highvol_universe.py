"""
Tests the already-built and validated liquidity-sweep engine against
TSLA/COIN/PLTR (higher volatility, less mega-cap-efficient than the
original AAPL/MSFT/NVDA/SPY universe) -- same reasoning as
scalp_research/run_highvol_universe.py. Reuses sweep_engine.py unchanged.

Focused configs around the least-bad settings found on the original
universe, not a fresh full grid search.
"""
import json
import pandas as pd

from sweep_engine import compute_levels, simulate, summarize, SweepConfig

TICKERS = ["TSLA", "COIN", "PLTR"]
CACHE_DIR = "../scalp_research/bars_1min_cache"
CONFIGS = [
    SweepConfig(lookback=20, buffer_mult=0.5, confirm_window=5, target_fraction=0.75),
    SweepConfig(lookback=20, buffer_mult=0.5, confirm_window=5, target_fraction=0.5),
    SweepConfig(lookback=20, buffer_mult=0.3, confirm_window=5, target_fraction=0.75),
]


def main():
    data = {}
    for tk in TICKERS:
        df = pd.read_pickle(f"{CACHE_DIR}/{tk}.pkl")
        data[tk] = df
        print(f"{tk}: {len(df)} bars")

    results = {}
    for cfg in CONFIGS:
        key = f"lb{cfg.lookback}_buf{cfg.buffer_mult}_cw{cfg.confirm_window}_tf{cfg.target_fraction}"
        all_trades = []
        by_ticker = {}
        for tk, df in data.items():
            lvl = compute_levels(df, cfg.lookback)
            trades = simulate(lvl, tk, cfg)
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
