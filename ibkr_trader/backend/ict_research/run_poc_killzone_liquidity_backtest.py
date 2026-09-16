"""
Stage 9: tests the two real methodology gaps flagged in review -- generic
swing-point liquidity targets instead of real ICT pools (prior day/week
high/low), and no session/killzone time restriction -- TOGETHER, not as
another loose independent filter layered onto an already-long chain.

4 combinations, crossed:
  - swing source:  rolling intraday lookback (every prior test in this
                    file) vs real PDH/PDL (compute_prior_day_levels +
                    apply_prior_day_liquidity)
  - killzone:       off (any RTH hour, every prior test) vs on (NY AM
                    09:30-11:00 ET + NY PM 13:30-16:00 ET, both directions)

Run on the FULL 2.5-year Alpaca dataset (not just the 24mo out-of-sample
slice) -- fair, since neither of these two ideas was designed or tuned
using any look at this data; there's no "already contaminated" subset to
avoid. lb60/tight5.0 (the only non-degenerate region across this whole
effort), buf0.5 (the consistently better buffer), pw 5/10.

Run: python run_poc_killzone_liquidity_backtest.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

from poc_pullback_engine import (
    compute_levels, apply_consolidation_gate, apply_killzone_gate,
    compute_prior_day_levels, apply_prior_day_liquidity,
    simulate, summarize, PocConfig,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "bars_1min_cache_alpaca")
TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "GOOGL", "AMZN", "TSLA", "META", "AMD", "CRWD", "QQQ", "IWM"]
INDEX_ETFS = {"SPY", "QQQ", "IWM"}
KILLZONE_WINDOWS = [("09:30", "11:00"), ("13:30", "16:00")]

LOOKBACK = 60
RANGE_TIGHTNESS = 5.0
BUFFER_MULT = 0.5
CONFIRM_WINDOW = 5
PULLBACK_WINDOWS = [5, 10]
PULLBACK_TOLERANCE = 0.25
TARGET_FRACTION = 0.75
SWING_SOURCES = ["rolling", "pdh_pdl"]
KILLZONE_OPTS = [False, True]


def load_bars() -> dict:
    return {tk: pd.read_pickle(os.path.join(CACHE_DIR, f"{tk}.pkl")) for tk in TICKERS}


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
    print("=== Loading full 2.5yr Alpaca bars (12 tickers) ===")
    data = load_bars()
    for tk, df in data.items():
        print(f"  {tk}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    print("=== Computing levels + prior-day liquidity + killzone flags (once per ticker) ===")
    t0 = time.time()
    variants: dict[str, dict[str, pd.DataFrame]] = {"rolling": {}, "pdh_pdl": {}}
    for tk, df in data.items():
        tl0 = time.time()
        lvl = compute_levels(df, LOOKBACK)
        lvl = apply_consolidation_gate(lvl, RANGE_TIGHTNESS)
        lvl = apply_killzone_gate(lvl, KILLZONE_WINDOWS)
        variants["rolling"][tk] = lvl

        prior = compute_prior_day_levels(df)
        lvl_pd = apply_prior_day_liquidity(lvl, prior)
        variants["pdh_pdl"][tk] = lvl_pd
        print(f"  {tk}: {time.time()-tl0:.1f}s")
    spy_vol = spy_realized_vol(variants["rolling"]["SPY"]["close"])
    print(f"  levels done, elapsed {time.time()-t0:.1f}s")

    results = {}
    for swing_source in SWING_SOURCES:
        lvl_set = variants[swing_source]
        for pullback_window in PULLBACK_WINDOWS:
            for require_killzone in KILLZONE_OPTS:
                key = f"swing{swing_source}_pw{pullback_window}_kz{require_killzone}"
                cfg = PocConfig(
                    lookback=LOOKBACK, buffer_mult=BUFFER_MULT, confirm_window=CONFIRM_WINDOW,
                    pullback_window=pullback_window, pullback_tolerance=PULLBACK_TOLERANCE,
                    target_fraction=TARGET_FRACTION, require_killzone=require_killzone,
                )
                all_trades = []
                for tk, d in lvl_set.items():
                    trades = simulate(d, tk, cfg)
                    all_trades.extend(trades)
                tag_regime(all_trades, spy_vol)

                by_ticker = {tk: summarize([t for t in all_trades if t.ticker == tk]) for tk in lvl_set}
                index_trades = [t for t in all_trades if t.ticker in INDEX_ETFS]
                stock_trades = [t for t in all_trades if t.ticker not in INDEX_ETFS]
                results[key] = {
                    "swing_source": swing_source, "pullback_window": pullback_window,
                    "require_killzone": require_killzone,
                    "overall": summarize(all_trades),
                    "index_etfs": summarize(index_trades),
                    "single_stocks": summarize(stock_trades),
                    "by_regime": {r: summarize([t for t in all_trades if t.regime == r]) for r in ("HIGH_VOL", "LOW_VOL")},
                    "by_ticker": by_ticker,
                }
                print(f"{key}: {results[key]['overall']}")

    with open(os.path.join(HERE, "poc_killzone_liquidity_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved poc_killzone_liquidity_results.json")

    print("\n=== Summary: the 4-way comparison, both pullback windows ===")
    for pw in PULLBACK_WINDOWS:
        print(f"  --- pullback_window={pw} ---")
        for swing_source in SWING_SOURCES:
            for kz in KILLZONE_OPTS:
                k = f"swing{swing_source}_pw{pw}_kz{kz}"
                o = results[k]["overall"]
                print(f"    swing={swing_source:8s} killzone={str(kz):5s} n={o['n_trades']:<5} win={o['win_rate_pct']}% avg={o['avg_ret_pct']}%")


if __name__ == "__main__":
    main()
