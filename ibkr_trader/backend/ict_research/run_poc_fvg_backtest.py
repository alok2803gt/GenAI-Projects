"""
Re-runs a focused slice of the POC-pullback grid (run_poc_backtest.py) with
the new require_fvg confluence gate (poc_pullback_engine.py) on and off, to
test whether requiring a genuine 3-bar Fair Value Gap along the sweep ->
pullback path changes the already-negative result.

Grid narrowed from the original 192 configs down to 32, on purpose, for two
reasons: (1) runtime -- the full 192-config grid took ~82 minutes, doubling
it for the require_fvg axis would run ~2.75 hours; (2) relevance -- this
drops range_tightness_mult=1.5 (0% of bars ever qualified as consolidating
at that setting, confirmed dead in the original run) and fixes
pullback_tolerance=0.25 / target_fraction=0.75, the values every top-10
config from the original leaderboard used, so this tests FVG confluence in
the part of the parameter space that already looked least bad, not a
re-litigation of settings already shown to be worse.

Run: python run_poc_fvg_backtest.py
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
RANGE_TIGHTNESS_MULT = [2.5, 5.0]
BUFFER_MULT = [0.3, 0.5]
CONFIRM_WINDOW = [5]
PULLBACK_WINDOW = [5, 10]
PULLBACK_TOLERANCE = [0.25]
TARGET_FRACTION = [0.75]
REQUIRE_FVG = [False, True]


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

            for buffer_mult in BUFFER_MULT:
                for confirm_window in CONFIRM_WINDOW:
                    for pullback_window in PULLBACK_WINDOW:
                        for pullback_tolerance in PULLBACK_TOLERANCE:
                            for target_fraction in TARGET_FRACTION:
                                for require_fvg in REQUIRE_FVG:
                                    key = (f"lb{lookback}_tight{tightness}_buf{buffer_mult}_cw{confirm_window}_"
                                           f"pw{pullback_window}_pt{pullback_tolerance}_tf{target_fraction}_"
                                           f"fvg{require_fvg}")
                                    cfg = PocConfig(
                                        lookback=lookback, buffer_mult=buffer_mult,
                                        confirm_window=confirm_window, pullback_window=pullback_window,
                                        pullback_tolerance=pullback_tolerance, target_fraction=target_fraction,
                                        require_fvg=require_fvg,
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
                                        "target_fraction": target_fraction, "require_fvg": require_fvg,
                                        "overall": summarize(all_trades),
                                        "by_regime": {
                                            r: summarize([t for t in all_trades if t.regime == r])
                                            for r in ("HIGH_VOL", "LOW_VOL")
                                        },
                                        "by_ticker": by_ticker,
                                    }

        print(f"  lookback={lookback} grid done, elapsed {time.time()-t0:.1f}s")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s, {len(results)} configs")
    with open(os.path.join(HERE, "poc_fvg_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved poc_fvg_results.json")

    # Paired comparison: for every (params) pair, show require_fvg=False vs True side by side.
    pairs = {}
    for k, v in results.items():
        base_key = k.rsplit("_fvg", 1)[0]
        pairs.setdefault(base_key, {})[v["require_fvg"]] = v["overall"]

    print("\n=== FVG confluence: False vs True, paired by config ===")
    for base_key, pair in pairs.items():
        f_off = pair.get(False, {})
        f_on = pair.get(True, {})
        print(f"  {base_key}:")
        print(f"    fvg=False -> n={f_off.get('n_trades')} win={f_off.get('win_rate_pct')}% avg={f_off.get('avg_ret_pct')}%")
        print(f"    fvg=True  -> n={f_on.get('n_trades')} win={f_on.get('win_rate_pct')}% avg={f_on.get('avg_ret_pct')}%")

    scored_on = [(k, v["overall"]) for k, v in results.items() if v["require_fvg"] and v["overall"].get("n_trades", 0) >= 20]
    scored_on.sort(key=lambda kv: kv[1]["avg_ret_pct"], reverse=True)
    print("\n=== Best fvg=True configs (n_trades >= 20) ===")
    for k, s in scored_on[:10]:
        print(f"  {k}: n={s['n_trades']} win={s['win_rate_pct']}% avg={s['avg_ret_pct']}%")


if __name__ == "__main__":
    main()
