"""
Stage 12: the one piece of ICT's own methodology stage 11 explicitly
flagged as untested -- the real ~23h futures session (Asian/London/NY),
not just the NY day session reused from the equity engine.

Requires poc_pullback_engine.py's compute_levels(..., rth_only=False) and
the session_date fix in simulate() (a real trading session spans
midnight; using raw calendar date for day-boundary detection would
incorrectly treat one session as two days mid-session -- fixed and
verified directly: a bar at 23:59 ET and the bar at 00:00 ET one minute
later both land in the session that started the prior evening at 18:00
ET, and the session correctly rolls at the real 17:00->18:00 ET
maintenance break, not at midnight).

Killzone windows tested (standard ICT global sessions, ET):
  - Asian:   19:00-00:00
  - London:  02:00-05:00
  - NY AM:   08:00-11:00 (wider than the equity-only 09:30 start used in
             every prior stage -- futures pre-equity-open activity is
             part of the real NY session)
  - NY PM:   13:30-16:00
Tested individually, all 4 combined, and against no killzone restriction
at all (every hour of the real session eligible), crossed with the
stage-7/11 structure filter. Core params fixed to stage 11's best
config (lb20/tight5.0/buf0.5/pw10) -- this is a focused test of the
session-structure question specifically, not a fresh full grid search.

Same ~2.3-month ES/NQ sample as stage 11 (this account's real IBKR
historical-data ceiling for expired futures contracts).

Run: python run_poc_futures_fullsession_backtest.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

from poc_pullback_engine import (
    compute_levels, apply_consolidation_gate, apply_killzone_gate,
    apply_structure_gate, compute_swing_structure, simulate, summarize, PocConfig,
)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "bars_1min_cache_futures_ibkr")
SYMBOLS = ["ES", "NQ"]

LOOKBACK = 20
RANGE_TIGHTNESS = 5.0
BUFFER_MULT = 0.5
CONFIRM_WINDOW = 5
PULLBACK_WINDOW = 10
PULLBACK_TOLERANCE = 0.25
TARGET_FRACTION = 0.75
FRACTAL_N = 2

# Asian killzone (19:00-00:00 ET) ends AT midnight rather than wrapping past
# it, per the standard ICT definition -- no cross-midnight window needed.
# apply_killzone_gate compares same-day (start,end) pairs, so a window
# would need lo<hi within one calendar day anyway; 23:59 instead of 00:00
# as the end loses only the literal last minute, negligible.
KILLZONES = {
    "asian": [("19:00", "23:59")],
    "london": [("02:00", "05:00")],
    "ny_am": [("08:00", "11:00")],
    "ny_pm": [("13:30", "16:00")],
}
ALL_FOUR = [("19:00", "23:59"), ("02:00", "05:00"), ("08:00", "11:00"), ("13:30", "16:00")]

REQUIRE_STRUCTURE = [False, True]


def load_bars() -> dict:
    return {sym: pd.read_pickle(os.path.join(CACHE_DIR, f"{sym}.pkl")) for sym in SYMBOLS}


def main():
    print("=== Loading futures bars (ES, NQ), full session (rth_only=False) ===")
    data = load_bars()
    for sym, df in data.items():
        print(f"  {sym}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    print("=== Computing levels (full ~23h session, lookback=20) ===")
    t0 = time.time()
    lvl_base = {}
    for sym, df in data.items():
        tl0 = time.time()
        lvl = compute_levels(df, LOOKBACK, rth_only=False)
        lvl = apply_structure_gate(lvl, compute_swing_structure(df, FRACTAL_N))
        lvl = apply_consolidation_gate(lvl, RANGE_TIGHTNESS)
        lvl_base[sym] = lvl
        print(f"  {sym}: {time.time()-tl0:.1f}s, {lvl['is_consolidating'].mean()*100:.1f}% consolidating")
    print(f"  levels done, elapsed {time.time()-t0:.1f}s")

    kz_conditions = {"none": None, **KILLZONES, "all_4_combined": ALL_FOUR}

    results = {}
    for kz_name, windows in kz_conditions.items():
        lvl_kz = {}
        for sym, lvl in lvl_base.items():
            if windows is None:
                lvl_kz[sym] = lvl
            else:
                lvl_kz[sym] = apply_killzone_gate(lvl, windows)

        for require_structure in REQUIRE_STRUCTURE:
            key = f"kz{kz_name}_struct{require_structure}"
            cfg = PocConfig(
                lookback=LOOKBACK, buffer_mult=BUFFER_MULT, confirm_window=CONFIRM_WINDOW,
                pullback_window=PULLBACK_WINDOW, pullback_tolerance=PULLBACK_TOLERANCE,
                target_fraction=TARGET_FRACTION, require_bullish_structure=require_structure,
                require_killzone=(windows is not None),
            )
            all_trades = []
            for sym, d in lvl_kz.items():
                all_trades.extend(simulate(d, sym, cfg))

            by_symbol = {sym: summarize([t for t in all_trades if t.ticker == sym]) for sym in lvl_kz}
            results[key] = {
                "killzone": kz_name, "require_bullish_structure": require_structure,
                "overall": summarize(all_trades),
                "by_symbol": by_symbol,
            }
            print(f"{key}: {results[key]['overall']}")

    with open(os.path.join(HERE, "poc_futures_fullsession_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved poc_futures_fullsession_results.json")

    print("\n=== Summary: killzone condition x structure filter ===")
    for kz_name in kz_conditions:
        off = results[f"kz{kz_name}_structFalse"]["overall"]
        on = results[f"kz{kz_name}_structTrue"]["overall"]
        print(f"  {kz_name:16s} structFalse: n={off.get('n_trades')} win={off.get('win_rate_pct')}% avg={off.get('avg_ret_pct')}%"
              f"   |  structTrue: n={on.get('n_trades')} win={on.get('win_rate_pct')}% avg={on.get('avg_ret_pct')}%")


if __name__ == "__main__":
    main()
