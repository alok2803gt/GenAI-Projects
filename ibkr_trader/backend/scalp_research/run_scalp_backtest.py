"""
Runs the VWAP-deviation mean-reversion scalp across a real parameter
grid (entry threshold x stop multiple x max hold time) on all 4 tickers,
using real 1-min RTH bars. Regime split uses SPY's own rolling realized
volatility (same objective, data-driven method as signals_research/,
implemented fresh here rather than imported).

Run: python run_scalp_backtest.py
"""
import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from fetch_minute_data import fetch_and_cache
from vwap_scalp_engine import compute_vwap_deviation, simulate, summarize, ScalpConfig

ENTRY_K = [1.5, 2.0, 2.5, 3.0]
STOP_MULT = [1.5, 2.0]
MAX_HOLD = [15, 30, 60]


def spy_realized_vol(spy_dev: pd.DataFrame) -> pd.Series:
    ret = np.log(spy_dev["close"]).diff()
    return ret.rolling(20 * 390).std()  # ~390 RTH 1-min bars/trading day, 20-day window


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
    print("=== Loading 1-min bars ===")
    data = fetch_and_cache()
    for tk, df in data.items():
        print(f"  {tk}: {len(df)} bars, {df.index.min()} .. {df.index.max()}")

    print("=== Computing VWAP deviation (RTH-only, session-anchored) ===")
    dev = {tk: compute_vwap_deviation(df) for tk, df in data.items()}
    for tk, d in dev.items():
        print(f"  {tk}: {len(d)} RTH bars")

    spy_vol = spy_realized_vol(dev["SPY"]) if "SPY" in dev else pd.Series(dtype=float)

    results = {}
    t0 = time.time()
    for entry_k in ENTRY_K:
        for stop_mult in STOP_MULT:
            for max_hold in MAX_HOLD:
                key = f"k{entry_k}_stop{stop_mult}_hold{max_hold}"
                cfg = ScalpConfig(entry_k=entry_k, stop_mult=stop_mult, max_hold_bars=max_hold)
                all_trades = []
                for tk, d in dev.items():
                    trades = simulate(d, tk, cfg)
                    all_trades.extend(trades)
                tag_regime(all_trades, spy_vol)

                by_ticker = {tk: summarize([t for t in all_trades if t.ticker == tk]) for tk in dev.keys()}
                results[key] = {
                    "entry_k": entry_k, "stop_mult": stop_mult, "max_hold_bars": max_hold,
                    "overall": summarize(all_trades),
                    "by_regime": {
                        r: summarize([t for t in all_trades if t.regime == r])
                        for r in ("HIGH_VOL", "LOW_VOL")
                    },
                    "by_ticker": by_ticker,
                    "trades": [
                        {**asdict(t), "entry_time": str(t.entry_time), "exit_time": str(t.exit_time)}
                        for t in all_trades
                    ],
                }
                print(f"{key}: {results[key]['overall']}")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    with open("scalp_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved scalp_results.json")


if __name__ == "__main__":
    main()
