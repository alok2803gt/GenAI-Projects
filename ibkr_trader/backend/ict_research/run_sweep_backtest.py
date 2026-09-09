"""
Runs the liquidity-sweep reversal strategy across a parameter grid on
real 1-min RTH bars. Reads the same cached bars scalp_research/ already
pulled from Alpaca (real data reuse -- no strategy code imported from
that project, just the raw price history, to avoid a redundant API pull
of identical historical bars).

Run: python run_sweep_backtest.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

from sweep_engine import compute_levels, simulate, summarize, SweepConfig

SCALP_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scalp_research", "bars_1min_cache")
TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]

LOOKBACK = [20, 60]
BUFFER_MULT = [0.3, 0.5]
CONFIRM_WINDOW = [3, 5]
TARGET_FRACTION = [0.5, 0.75]


def load_bars() -> dict:
    data = {}
    for tk in TICKERS:
        path = os.path.join(SCALP_CACHE_DIR, f"{tk}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run scalp_research/fetch_minute_data.py first")
        data[tk] = pd.read_pickle(path)
    return data


def spy_realized_vol(spy_close: pd.Series) -> pd.Series:
    ret = np.log(spy_close).diff()
    return ret.rolling(20 * 390).std()


def tag_regime(trades: list, vol_series: pd.Series) -> None:
    if vol_series.empty:
        for t in trades:
            t.regime = "UNKNOWN"
        return
    median_vol = vol_series.median()
    vol_at = vol_series.reindex(vol_series.index.union([t.exit_time for t in trades])).sort_index().ffill()
    for t in trades:
        v = vol_at.asof(t.exit_time)
        t.regime = "HIGH_VOL" if (pd.notna(v) and v >= median_vol) else "LOW_VOL"


def main():
    print("=== Loading cached 1-min bars (real Alpaca data, reused from scalp_research/) ===")
    data = load_bars()
    for tk, df in data.items():
        print(f"  {tk}: {len(df)} bars")

    results = {}
    t0 = time.time()
    for lookback in LOOKBACK:
        print(f"=== Computing levels for lookback={lookback} ===")
        lvl = {tk: compute_levels(df, lookback) for tk, df in data.items()}
        spy_vol = spy_realized_vol(lvl["SPY"]["close"]) if "SPY" in lvl else pd.Series(dtype=float)

        for buffer_mult in BUFFER_MULT:
            for confirm_window in CONFIRM_WINDOW:
                for target_fraction in TARGET_FRACTION:
                    key = f"lb{lookback}_buf{buffer_mult}_cw{confirm_window}_tf{target_fraction}"
                    cfg = SweepConfig(lookback=lookback, buffer_mult=buffer_mult,
                                       confirm_window=confirm_window, target_fraction=target_fraction)
                    all_trades = []
                    for tk, d in lvl.items():
                        trades = simulate(d, tk, cfg)
                        all_trades.extend(trades)
                    tag_regime(all_trades, spy_vol)

                    by_ticker = {tk: summarize([t for t in all_trades if t.ticker == tk]) for tk in lvl.keys()}
                    results[key] = {
                        "lookback": lookback, "buffer_mult": buffer_mult,
                        "confirm_window": confirm_window, "target_fraction": target_fraction,
                        "overall": summarize(all_trades),
                        "by_regime": {
                            r: summarize([t for t in all_trades if t.regime == r])
                            for r in ("HIGH_VOL", "LOW_VOL")
                        },
                        "by_ticker": by_ticker,
                    }
                    print(f"{key}: {results[key]['overall']}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    with open("sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved sweep_results.json")


if __name__ == "__main__":
    main()
