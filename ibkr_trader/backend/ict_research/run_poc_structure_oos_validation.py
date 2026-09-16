"""
The real test this whole research effort has been missing: a genuinely
out-of-sample check. Every prior result (base framework, FVG, SMA trend,
HH/HL structure) was found AND checked on the same single ~6-month IBKR
window (2026-03-17 -> 2026-09-14). This runs the HH/HL structure gate --
the most promising lead so far -- against 24 months of real Alpaca data
that NONE of those tests ever touched: 2024-03-18 -> 2026-03-16, the
portion of the new 2.5-year Alpaca pull that doesn't overlap the IBKR
window at all.

Scoped to the 3 best-performing configs from the original structure test
(buf0.5, pw 3/7/10 -- all lb60/tight5.0, the only viable region) rather
than the full 8-config grid, to keep runtime reasonable (~24 months of
data is ~8.5x the bar count of the original window). struct=False/True
paired for each, same comparison method as every other stage.

No parameters were tuned on this data -- the filter's logic and every
config tested here were fixed before this data was ever fetched.

Run: python run_poc_structure_oos_validation.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

from poc_pullback_engine import (
    compute_levels, apply_consolidation_gate, apply_structure_gate,
    compute_swing_structure, simulate, summarize, PocConfig,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "bars_1min_cache_alpaca")
TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "GOOGL", "AMZN", "TSLA", "META", "AMD", "CRWD", "QQQ", "IWM"]
INDEX_ETFS = {"SPY", "QQQ", "IWM"}
OOS_CUTOFF = pd.Timestamp("2026-03-17", tz="US/Eastern")  # exclude anything the IBKR window already covered

LOOKBACK = 60
RANGE_TIGHTNESS = 5.0
CONFIRM_WINDOW = 5
PULLBACK_TOLERANCE = 0.25
TARGET_FRACTION = 0.75
CONFIGS = [
    # (buffer_mult, pullback_window) -- the 3 best-performing configs from the original structure test
    (0.5, 3),
    (0.5, 7),
    (0.5, 10),
]
REQUIRE_STRUCTURE = [False, True]
FRACTAL_N = 2


def load_oos_bars() -> dict:
    data = {}
    for tk in TICKERS:
        path = os.path.join(CACHE_DIR, f"{tk}.pkl")
        df = pd.read_pickle(path)
        df_oos = df[df.index < OOS_CUTOFF]
        data[tk] = df_oos
    return data


def spy_realized_vol(spy_close: pd.Series) -> pd.Series:
    ret = np.log(spy_close).diff()
    return ret.rolling(20 * 390).std()


def tag_regime(trades: list, vol_series: pd.Series) -> None:
    if vol_series.empty or vol_series.dropna().empty:
        for t in trades:
            t.regime = "UNKNOWN"
        return
    median_vol = vol_series.median()
    vol_at = vol_series.reindex(vol_series.index.union([t.exit_time for t in trades])).sort_index().ffill()
    for t in trades:
        v = vol_at.asof(t.exit_time)
        t.regime = "HIGH_VOL" if (pd.notna(v) and v >= median_vol) else "LOW_VOL"


def main():
    print("=== Loading Alpaca bars, filtered to the out-of-sample window (before 2026-03-17) ===")
    data = load_oos_bars()
    for tk, df in data.items():
        print(f"  {tk}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    print("=== Computing HH/HL swing structure ===")
    structure_by_ticker = {tk: compute_swing_structure(df, FRACTAL_N) for tk, df in data.items()}
    for tk, s in structure_by_ticker.items():
        n_valid = len(s)
        n_bull = s.sum()
        print(f"  {tk}: {n_bull}/{n_valid} sessions bullish structure ({100*n_bull/max(n_valid,1):.0f}%)")

    print("=== Computing levels (lookback=60, once per ticker) ===")
    t0 = time.time()
    lvl_base = {}
    for tk, df in data.items():
        tl0 = time.time()
        lvl = compute_levels(df, LOOKBACK)
        lvl = apply_structure_gate(lvl, structure_by_ticker[tk])
        lvl = apply_consolidation_gate(lvl, RANGE_TIGHTNESS)
        lvl_base[tk] = lvl
        print(f"  {tk}: {time.time()-tl0:.1f}s")
    spy_vol = spy_realized_vol(lvl_base["SPY"]["close"]) if "SPY" in lvl_base else pd.Series(dtype=float)
    print(f"  levels done, elapsed {time.time()-t0:.1f}s")

    results = {}
    for buffer_mult, pullback_window in CONFIGS:
        for require_structure in REQUIRE_STRUCTURE:
            key = f"buf{buffer_mult}_pw{pullback_window}_struct{require_structure}"
            cfg = PocConfig(
                lookback=LOOKBACK, buffer_mult=buffer_mult, confirm_window=CONFIRM_WINDOW,
                pullback_window=pullback_window, pullback_tolerance=PULLBACK_TOLERANCE,
                target_fraction=TARGET_FRACTION, require_bullish_structure=require_structure,
            )
            all_trades = []
            for tk, d in lvl_base.items():
                trades = simulate(d, tk, cfg)
                all_trades.extend(trades)
            tag_regime(all_trades, spy_vol)

            long_trades = [t for t in all_trades if t.side == "LONG"]
            short_trades = [t for t in all_trades if t.side == "SHORT"]
            by_ticker = {tk: summarize([t for t in all_trades if t.ticker == tk]) for tk in lvl_base}
            results[key] = {
                "buffer_mult": buffer_mult, "pullback_window": pullback_window,
                "require_bullish_structure": require_structure,
                "overall": summarize(all_trades),
                "long_only": summarize(long_trades),
                "short_only": summarize(short_trades),
                "by_ticker": by_ticker,
            }
            print(f"{key}: overall={results[key]['overall']}")
            print(f"  long_only={results[key]['long_only']}")

    with open(os.path.join(HERE, "poc_structure_oos_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved poc_structure_oos_results.json")

    print("\n=== Summary: struct=False vs struct=True, out-of-sample (24mo, never touched before) ===")
    for buffer_mult, pullback_window in CONFIGS:
        base = f"buf{buffer_mult}_pw{pullback_window}"
        off = results[f"{base}_structFalse"]
        on = results[f"{base}_structTrue"]
        print(f"  {base}:")
        print(f"    overall    struct=False n={off['overall']['n_trades']} win={off['overall']['win_rate_pct']}% avg={off['overall']['avg_ret_pct']}%"
              f"   |  struct=True n={on['overall']['n_trades']} win={on['overall']['win_rate_pct']}% avg={on['overall']['avg_ret_pct']}%")
        print(f"    LONG-only  struct=False n={off['long_only']['n_trades']} win={off['long_only']['win_rate_pct']}% avg={off['long_only']['avg_ret_pct']}%"
              f"   |  struct=True n={on['long_only']['n_trades']} win={on['long_only']['win_rate_pct']}% avg={on['long_only']['avg_ret_pct']}%")


if __name__ == "__main__":
    main()
