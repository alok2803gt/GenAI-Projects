"""
Stage 8: real ICT market structure (higher highs + higher lows) instead of
an SMA crossover, gating LONG setups. Requested specifically because the
stage-6 SMA-based trend filter isn't how ICT actually defines trend --
this replaces that definition with the swing-structure one, tested the
same isolated way (not combined with FVG or the SMA filter).

Same 12-ticker universe, same narrowed grid region (lookback=60,
tightness=5.0x ATR) as every other follow-up in this directory.
require_bullish_structure paired False/True.

See poc_pullback_engine.py's module docstring for the exact definition:
2-day-fractal daily swing highs/lows, bullish = last two confirmed highs
ascending AND last two confirmed lows ascending, both required, confirmed
strictly before today. LONG-only, same as the SMA filter; SHORT untouched.

Run: python run_poc_structure_backtest.py
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
CACHE_DIR = os.path.join(HERE, "bars_1min_cache_ibkr")
TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "GOOGL", "AMZN", "TSLA", "META", "AMD", "CRWD", "QQQ", "IWM"]
INDEX_ETFS = {"SPY", "QQQ", "IWM"}

LOOKBACK = [60]
RANGE_TIGHTNESS_MULT = [5.0]
BUFFER_MULT = [0.3, 0.5]
CONFIRM_WINDOW = [5]
PULLBACK_WINDOW = [3, 5, 7, 10]
PULLBACK_TOLERANCE = [0.25]
TARGET_FRACTION = [0.75]
REQUIRE_STRUCTURE = [False, True]
FRACTAL_N = 2


def load_bars() -> dict:
    data = {}
    for tk in TICKERS:
        path = os.path.join(CACHE_DIR, f"{tk}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
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
    print("=== Loading cached 1-min bars (real IBKR data, 12 tickers) ===")
    data = load_bars()
    for tk, df in data.items():
        print(f"  {tk}: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")

    print("=== Computing HH/HL swing structure (2-day fractal) ===")
    structure_by_ticker = {tk: compute_swing_structure(df, FRACTAL_N) for tk, df in data.items()}
    for tk, s in structure_by_ticker.items():
        print(f"  {tk}: {s.sum()}/{len(s)} sessions bullish structure ({100*s.sum()/len(s):.0f}%)")

    results = {}
    t0 = time.time()
    for lookback in LOOKBACK:
        print(f"=== Computing levels for lookback={lookback} (once per ticker) ===")
        lvl_base = {}
        for tk, df in data.items():
            tl0 = time.time()
            lvl = compute_levels(df, lookback)
            lvl = apply_structure_gate(lvl, structure_by_ticker[tk])
            lvl_base[tk] = lvl
            print(f"  {tk}: {time.time()-tl0:.1f}s")
        spy_vol = spy_realized_vol(lvl_base["SPY"]["close"]) if "SPY" in lvl_base else pd.Series(dtype=float)

        for tightness in RANGE_TIGHTNESS_MULT:
            lvl = {tk: apply_consolidation_gate(d, tightness) for tk, d in lvl_base.items()}

            for buffer_mult in BUFFER_MULT:
                for confirm_window in CONFIRM_WINDOW:
                    for pullback_window in PULLBACK_WINDOW:
                        for pullback_tolerance in PULLBACK_TOLERANCE:
                            for target_fraction in TARGET_FRACTION:
                                for require_structure in REQUIRE_STRUCTURE:
                                    key = (f"lb{lookback}_tight{tightness}_buf{buffer_mult}_cw{confirm_window}_"
                                           f"pw{pullback_window}_pt{pullback_tolerance}_tf{target_fraction}_"
                                           f"struct{require_structure}")
                                    cfg = PocConfig(
                                        lookback=lookback, buffer_mult=buffer_mult,
                                        confirm_window=confirm_window, pullback_window=pullback_window,
                                        pullback_tolerance=pullback_tolerance, target_fraction=target_fraction,
                                        require_bullish_structure=require_structure,
                                    )
                                    all_trades = []
                                    for tk, d in lvl.items():
                                        trades = simulate(d, tk, cfg)
                                        all_trades.extend(trades)
                                    tag_regime(all_trades, spy_vol)

                                    long_trades = [t for t in all_trades if t.side == "LONG"]
                                    short_trades = [t for t in all_trades if t.side == "SHORT"]
                                    by_ticker = {tk: summarize([t for t in all_trades if t.ticker == tk]) for tk in lvl}
                                    index_trades = [t for t in all_trades if t.ticker in INDEX_ETFS]
                                    stock_trades = [t for t in all_trades if t.ticker not in INDEX_ETFS]
                                    results[key] = {
                                        "lookback": lookback, "range_tightness_mult": tightness,
                                        "buffer_mult": buffer_mult, "confirm_window": confirm_window,
                                        "pullback_window": pullback_window, "pullback_tolerance": pullback_tolerance,
                                        "target_fraction": target_fraction, "require_bullish_structure": require_structure,
                                        "overall": summarize(all_trades),
                                        "long_only": summarize(long_trades),
                                        "short_only": summarize(short_trades),
                                        "index_etfs": summarize(index_trades),
                                        "single_stocks": summarize(stock_trades),
                                        "by_regime": {
                                            r: summarize([t for t in all_trades if t.regime == r])
                                            for r in ("HIGH_VOL", "LOW_VOL")
                                        },
                                        "by_ticker": by_ticker,
                                    }

        print(f"  lookback={lookback} grid done, elapsed {time.time()-t0:.1f}s")

    print(f"\nTotal runtime: {time.time()-t0:.1f}s, {len(results)} configs")
    with open(os.path.join(HERE, "poc_structure_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved poc_structure_results.json")

    pairs = {}
    for k, v in results.items():
        base_key = k.rsplit("_struct", 1)[0]
        pairs.setdefault(base_key, {})[v["require_bullish_structure"]] = v

    print("\n=== HH/HL structure gate: False vs True, paired by config (overall / LONG-only / SHORT-only) ===")
    for base_key, pair in pairs.items():
        f_off = pair.get(False, {})
        f_on = pair.get(True, {})
        o_off, o_on = f_off.get("overall", {}), f_on.get("overall", {})
        l_off, l_on = f_off.get("long_only", {}), f_on.get("long_only", {})
        s_off, s_on = f_off.get("short_only", {}), f_on.get("short_only", {})
        print(f"  {base_key}:")
        print(f"    overall    struct=False n={o_off.get('n_trades')} win={o_off.get('win_rate_pct')}% avg={o_off.get('avg_ret_pct')}%"
              f"   |  struct=True n={o_on.get('n_trades')} win={o_on.get('win_rate_pct')}% avg={o_on.get('avg_ret_pct')}%")
        print(f"    LONG-only  struct=False n={l_off.get('n_trades')} win={l_off.get('win_rate_pct')}% avg={l_off.get('avg_ret_pct')}%"
              f"   |  struct=True n={l_on.get('n_trades')} win={l_on.get('win_rate_pct')}% avg={l_on.get('avg_ret_pct')}%")
        print(f"    SHORT-only struct=False n={s_off.get('n_trades')} win={s_off.get('win_rate_pct')}% avg={s_off.get('avg_ret_pct')}%"
              f"   |  struct=True n={s_on.get('n_trades')} win={s_on.get('win_rate_pct')}% avg={s_on.get('avg_ret_pct')}% (should be ~unchanged -- SHORT is ungated)")

    scored_on = [(k, v["overall"]) for k, v in results.items() if v["require_bullish_structure"] and v["overall"].get("n_trades", 0) >= 15]
    scored_on.sort(key=lambda kv: kv[1]["avg_ret_pct"], reverse=True)
    print("\n=== Best struct=True configs overall (n_trades >= 15) ===")
    for k, s in scored_on[:10]:
        print(f"  {k}: n={s['n_trades']} win={s['win_rate_pct']}% avg={s['avg_ret_pct']}%")


if __name__ == "__main__":
    main()
