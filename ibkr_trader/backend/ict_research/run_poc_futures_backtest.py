"""
Stage 10: does this framework behave differently on ES/NQ futures --
the instruments ICT methodology is most commonly taught and practiced
on, given their near-24h session structure -- versus the 12-equity
universe every prior stage used?

Real methodology simplification, disclosed plainly: this reuses
_rth_only (9:30-16:00 ET) unchanged, i.e. day-session-only. Futures
actually trade ~23h/day across Asian/London/NY sessions; testing the
full session would need the day-boundary/EOD-flat logic in
poc_pullback_engine.py's simulate() redesigned around a real CME
settlement-to-settlement trading day (~18:00-17:00 ET), which wasn't
attempted here. This is real information lost, not a free pass -- it
means the specific overnight/London-open killzone behavior ICT
attributes much of its edge to on these instruments is NOT being tested,
only the NY day session. Reusing the equity engine unmodified on the day
session was chosen over a rushed session-boundary rewrite that could
introduce new bugs.

Grid re-opened rather than assumed: lookback and tightness were NOT
fixed to the equity-tuned lb60/tight5.0 region, since there's no reason
to assume that tuning transfers to a structurally different instrument.
lb{20,60} x tight{2.5,5.0} x buf{0.3,0.5} x pw{5,10} x
require_bullish_structure{False,True} (the stage-7 filter, the one
concept that ever looked real, even though it didn't survive equity
out-of-sample testing) = 32 configs x 2 contracts (ES, NQ).

Real data-access constraint hit live, disclosed rather than hidden: the
original plan was 6 months, stitching 3 quarterly contracts. Only the
current front-month contract (Sept 2026) was actually queryable -- IBKR
returned "No security definition found" for the June and March 2026
contracts, already purged from historical lookup after rolling off.
Actual sample: ~2.3 months (2026-07-08 -> 2026-09-15), one contract per
symbol, ~67,500 raw (all-hours) bars each. Meaningfully smaller than
every equity test in this file -- reported as a real limitation on how
much this stage's result can be trusted, not smoothed over.

Run: python run_poc_futures_backtest.py
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
CACHE_DIR = os.path.join(HERE, "bars_1min_cache_futures_ibkr")
SYMBOLS = ["ES", "NQ"]

LOOKBACK = [20, 60]
RANGE_TIGHTNESS_MULT = [2.5, 5.0]
BUFFER_MULT = [0.3, 0.5]
CONFIRM_WINDOW = 5
PULLBACK_WINDOW = [5, 10]
PULLBACK_TOLERANCE = 0.25
TARGET_FRACTION = 0.75
REQUIRE_STRUCTURE = [False, True]
FRACTAL_N = 2


def load_bars() -> dict:
    return {sym: pd.read_pickle(os.path.join(CACHE_DIR, f"{sym}.pkl")) for sym in SYMBOLS}


def realized_vol(close: pd.Series) -> pd.Series:
    ret = np.log(close).diff()
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
    print("=== Loading futures bars (ES, NQ) ===")
    data = load_bars()
    for sym, df in data.items():
        print(f"  {sym}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    results = {}
    t0 = time.time()
    for lookback in LOOKBACK:
        print(f"=== Computing levels for lookback={lookback} ===")
        lvl_base = {}
        structure_by_sym = {}
        for sym, df in data.items():
            tl0 = time.time()
            structure_by_sym[sym] = compute_swing_structure(df, FRACTAL_N)
            lvl = compute_levels(df, lookback)
            lvl = apply_structure_gate(lvl, structure_by_sym[sym])
            lvl_base[sym] = lvl
            print(f"  {sym}: {time.time()-tl0:.1f}s")
        vol_ref = realized_vol(lvl_base["ES"]["close"])

        for tightness in RANGE_TIGHTNESS_MULT:
            lvl = {sym: apply_consolidation_gate(d, tightness) for sym, d in lvl_base.items()}
            for sym in lvl:
                pct_c = lvl[sym]["is_consolidating"].mean() * 100
                print(f"    tightness={tightness}: {sym} consolidating {pct_c:.1f}% of bars")

            for buffer_mult in BUFFER_MULT:
                for pullback_window in PULLBACK_WINDOW:
                    for require_structure in REQUIRE_STRUCTURE:
                        key = (f"lb{lookback}_tight{tightness}_buf{buffer_mult}_pw{pullback_window}_"
                               f"struct{require_structure}")
                        cfg = PocConfig(
                            lookback=lookback, buffer_mult=buffer_mult, confirm_window=CONFIRM_WINDOW,
                            pullback_window=pullback_window, pullback_tolerance=PULLBACK_TOLERANCE,
                            target_fraction=TARGET_FRACTION, require_bullish_structure=require_structure,
                        )
                        all_trades = []
                        for sym, d in lvl.items():
                            all_trades.extend(simulate(d, sym, cfg))
                        tag_regime(all_trades, vol_ref)

                        by_symbol = {sym: summarize([t for t in all_trades if t.ticker == sym]) for sym in lvl}
                        long_trades = [t for t in all_trades if t.side == "LONG"]
                        results[key] = {
                            "lookback": lookback, "range_tightness_mult": tightness,
                            "buffer_mult": buffer_mult, "pullback_window": pullback_window,
                            "require_bullish_structure": require_structure,
                            "overall": summarize(all_trades),
                            "long_only": summarize(long_trades),
                            "by_regime": {r: summarize([t for t in all_trades if t.regime == r]) for r in ("HIGH_VOL", "LOW_VOL")},
                            "by_symbol": by_symbol,
                        }

        print(f"  lookback={lookback} done, elapsed {time.time()-t0:.1f}s")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s, {len(results)} configs")
    with open(os.path.join(HERE, "poc_futures_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved poc_futures_results.json")

    scored = [(k, v["overall"]) for k, v in results.items() if v["overall"].get("n_trades", 0) >= 15]
    scored.sort(key=lambda kv: kv[1]["avg_ret_pct"], reverse=True)
    print("\n=== Top 10 configs by avg_ret_pct (n_trades >= 15) ===")
    for k, s in scored[:10]:
        print(f"  {k}: n={s['n_trades']} win={s['win_rate_pct']}% avg={s['avg_ret_pct']}%")
    print("\n=== Bottom 5 configs ===")
    for k, s in scored[-5:]:
        print(f"  {k}: n={s['n_trades']} win={s['win_rate_pct']}% avg={s['avg_ret_pct']}%")

    print(f"\nTotal configs (any n): {len(results)}, cleared n>=15 bar: {len(scored)}")
    positive = [s for _, s in scored if s["avg_ret_pct"] > 0]
    print(f"Positive among those: {len(positive)}")


if __name__ == "__main__":
    main()
