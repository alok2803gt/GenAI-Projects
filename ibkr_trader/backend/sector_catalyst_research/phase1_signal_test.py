"""
Phase 1: does a real, ex-ante "sector is heating up" signal predict further
upside over the next 1-2 trading days, at a rate/magnitude that would
matter for a cheap far-OTM short-DTE call spread? Pure price/volume test,
no options pricing yet -- no point building the options-P&L machinery on
top of a signal that turns out to be noise.

Signal definition: on day T, a basket "pops" if >= MIN_MOVERS of its
members close up >= POP_RET_THRESH with same-day volume >=
POP_VOL_MULT x its own trailing 20-day average volume. This is knowable by
end of day T (or even intraday) -- no lookahead.

For every basket member (whether or not it was one of the day's movers),
measure its OWN forward return over T+1 and T+1..T+2 (cumulative), and
compare the conditional (signal-day) distribution against that same
ticker's unconditional distribution -- both the mean AND the right-tail
frequency of a big move, since a call-spread's payoff is driven by the
tail, not the average.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent

MEMORY_BASKET = ["MU", "WDC", "STX", "SNDK"]
SEMIS_BASKET = ["NVDA", "AMD", "AVGO", "MRVL", "QCOM", "TXN", "INTC",
                "AMAT", "LRCX", "SMCI", "KLAC", "ON", "MCHP", "ADI", "ASML"]

POP_RET_THRESH = 0.03      # a member "pops" if up >= 3% on the day
POP_VOL_MULT = 1.0         # AND volume >= its own 20-day average (loosened from 1.5x --
                            # 1.5x missed the real 2026-09-04 MU/WDC/STX catalyst entirely:
                            # all 3 popped 6%+ together on only 0.7-1.3x normal volume, a
                            # re-rating move, not a volume-spike event)
MIN_MOVERS = 2             # basket "heats up" if >= 2 members pop same day
TAIL_MOVE_THRESH = 0.04    # what counts as a "big enough for a cheap OTM spread" forward move


def load_universe():
    with open(HERE / "semis_universe_5y_ohlcv.pkl", "rb") as f:
        return pickle.load(f)


def per_ticker_frame(df):
    df = df.copy()
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    df["ret"] = df["Close"].pct_change()
    df["vol_avg20"] = df["Volume"].rolling(20, min_periods=15).mean()
    df["rel_vol"] = df["Volume"] / df["vol_avg20"].shift(1)
    df["popped"] = (df["ret"] >= POP_RET_THRESH) & (df["rel_vol"] >= POP_VOL_MULT)
    df["fwd_ret_1d"] = df["Close"].shift(-1) / df["Close"] - 1
    df["fwd_ret_2d"] = df["Close"].shift(-2) / df["Close"] - 1
    return df


def run_basket(name, tickers, universe):
    frames = {t: per_ticker_frame(universe[t]) for t in tickers if t in universe}
    pop_counts = pd.DataFrame({t: f["popped"] for t, f in frames.items()}).sum(axis=1)
    basket_hot = pop_counts >= MIN_MOVERS
    hot_days = basket_hot[basket_hot].index

    print(f"\n{'='*70}\nBASKET: {name}  ({list(frames.keys())})")
    print(f"{len(hot_days)} real 'basket heating up' days out of {len(pop_counts)} "
          f"trading days ({100*len(hot_days)/len(pop_counts):.1f}%)")

    rows = []
    for t, f in frames.items():
        common_hot = hot_days.intersection(f.index)
        cond = f.loc[common_hot].dropna(subset=["fwd_ret_1d", "fwd_ret_2d"])
        uncond = f.dropna(subset=["fwd_ret_1d", "fwd_ret_2d"])
        if len(cond) < 5:
            continue
        rows.append({
            "ticker": t,
            "n_cond": len(cond),
            "mean_fwd1_cond": cond["fwd_ret_1d"].mean(),
            "mean_fwd1_uncond": uncond["fwd_ret_1d"].mean(),
            "tail_rate_1d_cond": (cond["fwd_ret_1d"] >= TAIL_MOVE_THRESH).mean(),
            "tail_rate_1d_uncond": (uncond["fwd_ret_1d"] >= TAIL_MOVE_THRESH).mean(),
            "mean_fwd2_cond": cond["fwd_ret_2d"].mean(),
            "mean_fwd2_uncond": uncond["fwd_ret_2d"].mean(),
            "tail_rate_2d_cond": (cond["fwd_ret_2d"] >= TAIL_MOVE_THRESH).mean(),
            "tail_rate_2d_uncond": (uncond["fwd_ret_2d"] >= TAIL_MOVE_THRESH).mean(),
        })
    res = pd.DataFrame(rows)
    if res.empty:
        print("  not enough hot-day observations per ticker to report")
        return res

    print(f"\n  {'ticker':6s} {'n':>4s} {'fwd1_cond':>10s} {'fwd1_uncond':>12s} "
          f"{'tail1_cond':>11s} {'tail1_uncond':>13s} {'fwd2_cond':>10s} {'fwd2_uncond':>12s} "
          f"{'tail2_cond':>11s} {'tail2_uncond':>13s}")
    for _, r in res.iterrows():
        print(f"  {r['ticker']:6s} {r['n_cond']:4.0f} "
              f"{100*r['mean_fwd1_cond']:9.2f}% {100*r['mean_fwd1_uncond']:11.2f}% "
              f"{100*r['tail_rate_1d_cond']:10.1f}% {100*r['tail_rate_1d_uncond']:12.1f}% "
              f"{100*r['mean_fwd2_cond']:9.2f}% {100*r['mean_fwd2_uncond']:11.2f}% "
              f"{100*r['tail_rate_2d_cond']:10.1f}% {100*r['tail_rate_2d_uncond']:12.1f}%")

    print(f"\n  BASKET AVERAGE (across members, n-weighted):")
    tot_n = res["n_cond"].sum()
    for col_c, col_u, label in [
        ("mean_fwd1_cond", "mean_fwd1_uncond", "mean fwd 1d return"),
        ("tail_rate_1d_cond", "tail_rate_1d_uncond", f"P(fwd 1d >= {TAIL_MOVE_THRESH*100:.0f}%)"),
        ("mean_fwd2_cond", "mean_fwd2_uncond", "mean fwd 2d return"),
        ("tail_rate_2d_cond", "tail_rate_2d_uncond", f"P(fwd 2d >= {TAIL_MOVE_THRESH*100:.0f}%)"),
    ]:
        c = (res[col_c] * res["n_cond"]).sum() / tot_n
        u = (res[col_u] * res["n_cond"]).sum() / tot_n
        print(f"    {label:28s}  conditional={100*c:6.2f}%   unconditional={100*u:6.2f}%   "
              f"lift={100*(c-u):+6.2f}pp")
    return res


def main():
    universe = load_universe()
    run_basket("Memory/Storage", MEMORY_BASKET, universe)
    run_basket("Broader Semis/AI", SEMIS_BASKET, universe)


if __name__ == "__main__":
    main()
