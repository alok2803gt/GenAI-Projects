"""
Runs the full Accumulation -> Consolidation -> Manipulation -> Pullback-to-
POC -> Confirmation -> Entry framework (poc_pullback_engine.py) across a
parameter grid on real IBKR 1-min RTH bars (see fetch_minute_data_ibkr.py
for why IBKR instead of Alpaca, and the real ~6-month-vs-2.5-year sample
size tradeoff that implies).

Grid structure: lookback drives the expensive per-bar POC/tightness
computation (run once per ticker per lookback via compute_levels), every
other parameter -- including range_tightness_mult, the consolidation gate
-- is cheap to sweep afterward (apply_consolidation_gate + simulate).

Run: python run_poc_backtest.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

from poc_pullback_engine import compute_levels, apply_consolidation_gate, simulate, summarize, PocConfig

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "bars_1min_cache_ibkr")
TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]

LOOKBACK = [20, 60]
RANGE_TIGHTNESS_MULT = [1.5, 2.5, 5.0]   # 5.0 ~= effectively loose / near-no consolidation filter
BUFFER_MULT = [0.3, 0.5]
CONFIRM_WINDOW = [3, 5]
PULLBACK_WINDOW = [5, 10]
PULLBACK_TOLERANCE = [0.15, 0.25]
TARGET_FRACTION = [0.5, 0.75]


def load_bars() -> dict:
    data = {}
    for tk in TICKERS:
        path = os.path.join(CACHE_DIR, f"{tk}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run fetch_minute_data_ibkr.py first")
        data[tk] = pd.read_pickle(path)
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
    print("=== Loading cached 1-min bars (real IBKR data) ===")
    data = load_bars()
    for tk, df in data.items():
        print(f"  {tk}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    results = {}
    t0 = time.time()
    for lookback in LOOKBACK:
        print(f"=== Computing levels for lookback={lookback} (expensive, once per ticker) ===")
        lvl_base = {}
        for tk, df in data.items():
            tl0 = time.time()
            lvl_base[tk] = compute_levels(df, lookback)
            print(f"  {tk}: {time.time()-tl0:.1f}s")
        spy_vol = spy_realized_vol(lvl_base["SPY"]["close"]) if "SPY" in lvl_base else pd.Series(dtype=float)

        for tightness in RANGE_TIGHTNESS_MULT:
            lvl = {tk: apply_consolidation_gate(d, tightness) for tk, d in lvl_base.items()}
            for tk in lvl:
                pct_consolidating = lvl[tk]["is_consolidating"].mean() * 100
                print(f"    tightness={tightness}: {tk} consolidating {pct_consolidating:.1f}% of bars")

            for buffer_mult in BUFFER_MULT:
                for confirm_window in CONFIRM_WINDOW:
                    for pullback_window in PULLBACK_WINDOW:
                        for pullback_tolerance in PULLBACK_TOLERANCE:
                            for target_fraction in TARGET_FRACTION:
                                key = (f"lb{lookback}_tight{tightness}_buf{buffer_mult}_cw{confirm_window}_"
                                       f"pw{pullback_window}_pt{pullback_tolerance}_tf{target_fraction}")
                                cfg = PocConfig(
                                    lookback=lookback, buffer_mult=buffer_mult,
                                    confirm_window=confirm_window, pullback_window=pullback_window,
                                    pullback_tolerance=pullback_tolerance, target_fraction=target_fraction,
                                )
                                all_trades = []
                                for tk, d in lvl.items():
                                    trades = simulate(d, tk, cfg)
                                    all_trades.extend(trades)
                                tag_regime(all_trades, spy_vol)

                                by_ticker = {tk: summarize([t for t in all_trades if t.ticker == tk]) for tk in lvl}
                                results[key] = {
                                    "lookback": lookback, "range_tightness_mult": tightness,
                                    "buffer_mult": buffer_mult, "confirm_window": confirm_window,
                                    "pullback_window": pullback_window, "pullback_tolerance": pullback_tolerance,
                                    "target_fraction": target_fraction,
                                    "overall": summarize(all_trades),
                                    "by_regime": {
                                        r: summarize([t for t in all_trades if t.regime == r])
                                        for r in ("HIGH_VOL", "LOW_VOL")
                                    },
                                    "by_ticker": by_ticker,
                                }

        print(f"  lookback={lookback} grid done, elapsed {time.time()-t0:.1f}s")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s, {len(results)} configs")
    with open(os.path.join(HERE, "poc_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved poc_results.json")

    # Quick leaderboard by overall avg_ret_pct among configs with a real sample size.
    scored = [(k, v["overall"]) for k, v in results.items() if v["overall"].get("n_trades", 0) >= 30]
    scored.sort(key=lambda kv: kv[1]["avg_ret_pct"], reverse=True)
    print("\n=== Top 10 configs by avg_ret_pct (n_trades >= 30) ===")
    for k, s in scored[:10]:
        print(f"  {k}: n={s['n_trades']} win={s['win_rate_pct']}% avg={s['avg_ret_pct']}%")
    print("\n=== Bottom 5 configs by avg_ret_pct (n_trades >= 30) ===")
    for k, s in scored[-5:]:
        print(f"  {k}: n={s['n_trades']} win={s['win_rate_pct']}% avg={s['avg_ret_pct']}%")


if __name__ == "__main__":
    main()
