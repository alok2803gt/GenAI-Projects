"""
Second-pass POC-pullback + FVG-confluence backtest, narrowed and widened at
the same time: narrowed to just the one region of the grid that actually
showed a positive result (lookback=60, range_tightness=5.0x ATR -- every
other combination at lookback=60 produced zero consolidating bars, and
lookback=20 never got FVG-confluence positive anywhere), widened from 4
tickers to 12 to reduce how much any single name's noise can swing the
average, and to directly test whether the first pass's SPY weakness (0-10%
win rate under FVG confluence) is SPY-specific or shared across index ETFs.

Added tickers (see fetch_minute_data_ibkr_batch2.py): GOOGL, AMZN, TSLA,
META, AMD, CRWD (single stocks, same liquid/high-momentum profile as MSFT/
NVDA which performed well) + QQQ, IWM (2 more index ETFs alongside SPY).

Pullback window swept finer (3/5/7/10 instead of just 5/10) since pw5 was
the better half of every pair in the first pass.

Run: python run_poc_fvg_backtest_v2.py
"""
import json
import os
import time

import numpy as np
import pandas as pd

from poc_pullback_engine import compute_levels, apply_consolidation_gate, simulate, summarize, PocConfig

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
REQUIRE_FVG = [False, True]


def load_bars() -> dict:
    data = {}
    for tk in TICKERS:
        path = os.path.join(CACHE_DIR, f"{tk}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found -- run fetch_minute_data_ibkr_batch2.py first")
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

    results = {}
    t0 = time.time()
    for lookback in LOOKBACK:
        print(f"=== Computing levels for lookback={lookback} (once per ticker) ===")
        lvl_base = {}
        for tk, df in data.items():
            tl0 = time.time()
            lvl_base[tk] = compute_levels(df, lookback)
            print(f"  {tk}: {time.time()-tl0:.1f}s")
        spy_vol = spy_realized_vol(lvl_base["SPY"]["close"]) if "SPY" in lvl_base else pd.Series(dtype=float)

        for tightness in RANGE_TIGHTNESS_MULT:
            lvl = {tk: apply_consolidation_gate(d, tightness) for tk, d in lvl_base.items()}
            for tk in lvl:
                pct_c = lvl[tk]["is_consolidating"].mean() * 100
                print(f"    tightness={tightness}: {tk} consolidating {pct_c:.1f}% of bars")

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
                                    index_trades = [t for t in all_trades if t.ticker in INDEX_ETFS]
                                    stock_trades = [t for t in all_trades if t.ticker not in INDEX_ETFS]
                                    results[key] = {
                                        "lookback": lookback, "range_tightness_mult": tightness,
                                        "buffer_mult": buffer_mult, "confirm_window": confirm_window,
                                        "pullback_window": pullback_window, "pullback_tolerance": pullback_tolerance,
                                        "target_fraction": target_fraction, "require_fvg": require_fvg,
                                        "overall": summarize(all_trades),
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
    with open(os.path.join(HERE, "poc_fvg_results_v2.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved poc_fvg_results_v2.json")

    pairs = {}
    for k, v in results.items():
        base_key = k.rsplit("_fvg", 1)[0]
        pairs.setdefault(base_key, {})[v["require_fvg"]] = v

    print("\n=== FVG confluence: False vs True, paired by config (overall / index-etfs / single-stocks) ===")
    for base_key, pair in pairs.items():
        f_off = pair.get(False, {})
        f_on = pair.get(True, {})
        o_off, o_on = f_off.get("overall", {}), f_on.get("overall", {})
        i_off, i_on = f_off.get("index_etfs", {}), f_on.get("index_etfs", {})
        s_off, s_on = f_off.get("single_stocks", {}), f_on.get("single_stocks", {})
        print(f"  {base_key}:")
        print(f"    overall        fvg=False n={o_off.get('n_trades')} win={o_off.get('win_rate_pct')}% avg={o_off.get('avg_ret_pct')}%"
              f"   |  fvg=True n={o_on.get('n_trades')} win={o_on.get('win_rate_pct')}% avg={o_on.get('avg_ret_pct')}%")
        print(f"    index_etfs     fvg=False n={i_off.get('n_trades')} win={i_off.get('win_rate_pct')}% avg={i_off.get('avg_ret_pct')}%"
              f"   |  fvg=True n={i_on.get('n_trades')} win={i_on.get('win_rate_pct')}% avg={i_on.get('avg_ret_pct')}%")
        print(f"    single_stocks  fvg=False n={s_off.get('n_trades')} win={s_off.get('win_rate_pct')}% avg={s_off.get('avg_ret_pct')}%"
              f"   |  fvg=True n={s_on.get('n_trades')} win={s_on.get('win_rate_pct')}% avg={s_on.get('avg_ret_pct')}%")

    scored_on = [(k, v["overall"]) for k, v in results.items() if v["require_fvg"] and v["overall"].get("n_trades", 0) >= 30]
    scored_on.sort(key=lambda kv: kv[1]["avg_ret_pct"], reverse=True)
    print("\n=== Best fvg=True configs overall (n_trades >= 30) ===")
    for k, s in scored_on[:10]:
        print(f"  {k}: n={s['n_trades']} win={s['win_rate_pct']}% avg={s['avg_ret_pct']}%")


if __name__ == "__main__":
    main()
