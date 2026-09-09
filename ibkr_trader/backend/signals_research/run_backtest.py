"""
Orchestrates the full backtest grid: 2 variants (shared, per_ticker) x 3
training-window sizes (5, 20, 60 trading days) = 6 runs, each a full
walk-forward simulation across AAPL/MSFT/NVDA/SPY. Writes per-run and
per-ticker summaries plus all raw trades to results.json, and prints a
progress log as it goes (each run can take several minutes).

Run: python run_backtest.py [--force-refetch]
"""
import json
import sys
import time
from dataclasses import asdict

sys.path.insert(0, ".")
from fetch_data import fetch_and_cache
from backtest_engine import BacktestConfig, run_walkforward, summarize

WINDOW_SIZES = [5, 20, 60]
VARIANTS = ["shared", "per_ticker"]


def main():
    force = "--force-refetch" in sys.argv
    print("=== Loading data ===")
    data = fetch_and_cache(force=force)
    for tk, df in data.items():
        print(f"  {tk}: {len(df)} bars, {df.index.min()} .. {df.index.max()}")

    results = {}
    t0 = time.time()
    for variant in VARIANTS:
        for window in WINDOW_SIZES:
            key = f"{variant}_{window}d"
            print(f"\n=== Running {key} ===")
            cfg = BacktestConfig(variant=variant, window_size_days=window)
            trades = run_walkforward(data, cfg)

            by_ticker = {}
            for tk in data.keys():
                tk_trades = [t for t in trades if t.ticker == tk]
                by_ticker[tk] = {
                    "summary": summarize(tk_trades),
                    "by_regime": {
                        regime: summarize([t for t in tk_trades if t.regime == regime])
                        for regime in ("HIGH_VOL", "LOW_VOL")
                    },
                }

            results[key] = {
                "variant": variant,
                "window_size_days": window,
                "overall": summarize(trades),
                "overall_by_regime": {
                    regime: summarize([t for t in trades if t.regime == regime])
                    for regime in ("HIGH_VOL", "LOW_VOL")
                },
                "by_ticker": by_ticker,
                "trades": [asdict(t) | {"entry_time": str(t.entry_time), "exit_time": str(t.exit_time)}
                           for t in trades],
            }
            print(f"  {key}: {results[key]['overall']}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved results.json")


if __name__ == "__main__":
    main()
