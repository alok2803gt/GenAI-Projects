"""
Does exiting on a BEARISH harami beat the validated fixed 5-day exit?

Entry is unchanged from the live rule: bullish harami + prior downtrend
context, bought at the NEXT day's open. Only the exit changes:

    fixed_5d      sell at the close 5 trading days later (live rule today)
    bearish_har   hold until a bearish harami prints, sell the NEXT open
    bearish+ctx   same, but only count a bearish harami in an uptrend
                  (close above both SMA20 and SMA50) -- the mirror of the
                  entry's own downtrend filter
    bearish_cap20 bearish harami, but forced out after 20 days if none prints

Same universe, data and detection code as harami_backtest.py (112 tickers,
5y daily). Every variant is measured on the SAME signals, so the comparison
is paired: differences are the exit rule, not the sample.

Reported against a matched baseline: the same number of days held, starting
on a random day, so "the market drifts up" is not mistaken for an edge.

    ../../venv/bin/python harami_exit_signal_backtest.py
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from harami_backtest import BODY_LOOKBACK, find_harami_signals  # noqa: E402

MAX_HOLD_CAP = 20


def add_bearish_harami(df):
    """Mirror image of the entry pattern: a large BULLISH prior body, then a
    bearish day whose body sits inside it."""
    body = df["Close"] - df["Open"]
    abs_body = body.abs()
    prior_open, prior_close = df["Open"].shift(1), df["Close"].shift(1)
    body_median = abs_body.rolling(BODY_LOOKBACK).median()
    df["bear_harami"] = ((prior_close > prior_open)                       # prior day bullish
                         & (df["Close"] < df["Open"])                     # today bearish
                         & (df["Open"] < prior_close) & (df["Close"] > prior_open)   # inside prior body
                         & (abs_body.shift(1) >= body_median.shift(1)))   # prior body was real
    df["uptrend_ctx"] = (df["Close"] > df["sma20"]) & (df["Close"] > df["sma50"])
    return df


def trades_for_ticker(df):
    d = add_bearish_harami(find_harami_signals(df)).reset_index(drop=True)
    n = len(d)
    out = []
    for i in range(BODY_LOOKBACK + 1, n - 2):
        if not (bool(d["harami"].iloc[i]) and bool(d["downtrend_ctx"].iloc[i])):
            continue
        e = i + 1                                   # buy the next open
        if e + 1 >= n:
            continue
        entry = d["Open"].iloc[e]
        row = {"entry_i": e}
        # fixed 5-day exit (today's live rule)
        row["fixed_5d"] = ((d["Close"].iloc[e + 5] / entry - 1) * 100, 5) if e + 5 < n else None
        # exit on the next bearish harami, with and without uptrend context
        for label, need_ctx, cap in (("bearish_har", False, None), ("bearish_ctx", True, None),
                                     ("bearish_cap20", False, MAX_HOLD_CAP)):
            res = None
            for j in range(e + 1, n - 1):
                if cap is not None and j - e >= cap:
                    res = ((d["Close"].iloc[j] / entry - 1) * 100, j - e)
                    break
                if bool(d["bear_harami"].iloc[j]) and (not need_ctx or bool(d["uptrend_ctx"].iloc[j])):
                    res = ((d["Open"].iloc[j + 1] / entry - 1) * 100, j + 1 - e)
                    break
            row[label] = res
        out.append(row)
    return out


def baseline_for_holds(universe, holds):
    """Mean return of holding a random day for the same number of days."""
    by_hold = {}
    for h in sorted(set(int(x) for x in holds if x and x > 0)):
        rets = []
        for df in universe.values():
            c = df["Close"]
            fwd = (c.shift(-h) / c - 1) * 100
            rets.extend(fwd.dropna().tolist())
        by_hold[h] = float(np.mean(rets)) if rets else float("nan")
    return by_hold


def main():
    with open(HERE.parent / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
        universe = pickle.load(f)
    print(f"universe: {len(universe)} tickers\n")
    rows = []
    for t, df in universe.items():
        for r in trades_for_ticker(df):
            rows.append({"ticker": t, **r})
    print(f"signals: {len(rows)} (bullish harami + downtrend, entry at next open)\n")

    variants = ["fixed_5d", "bearish_har", "bearish_ctx", "bearish_cap20"]
    all_holds = [r[v][1] for r in rows for v in variants if r.get(v)]
    base = baseline_for_holds(universe, all_holds)

    print(f"{'exit rule':<16}{'n':>6}{'mean%':>8}{'median%':>9}{'win':>7}{'avg hold':>10}{'baseline%':>11}{'edge pp':>9}{'t':>7}")
    stats = {}
    for v in variants:
        vals = [(r[v][0], r[v][1]) for r in rows if r.get(v)]
        if not vals:
            continue
        rets = np.array([x[0] for x in vals])
        holds = np.array([x[1] for x in vals])
        b = float(np.mean([base.get(int(h), np.nan) for h in holds]))
        edge = rets.mean() - b
        t = rets.mean() / (rets.std(ddof=1) / math.sqrt(len(rets)))
        stats[v] = (rets, holds, edge)
        print(f"{v:<16}{len(rets):>6}{rets.mean():>8.3f}{np.median(rets):>9.3f}"
              f"{(rets > 0).mean():>7.1%}{holds.mean():>10.1f}{b:>11.3f}{edge:>9.3f}{t:>7.2f}")

    print("\npaired vs the live fixed_5d rule (same signals):")
    for v in variants[1:]:
        pairs = [(r["fixed_5d"][0], r[v][0]) for r in rows if r.get("fixed_5d") and r.get(v)]
        d = np.array([b - a for a, b in pairs])
        t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
        print(f"  {v:<14} n={len(d):>5}  mean diff {d.mean():+.3f}pp  t={t:+.2f}"
              f"{'  <- significant' if abs(t) >= 2 else ''}")

    print("\nhold-length distribution for the bearish-harami exit:")
    h = stats["bearish_har"][1]
    print(f"  median {np.median(h):.0f}d | mean {h.mean():.1f}d | 25th {np.percentile(h,25):.0f}d | "
          f"75th {np.percentile(h,75):.0f}d | max {h.max():.0f}d")
    print(f"  exits within 5 days: {(h <= 5).mean():.0%} | beyond 20 days: {(h > 20).mean():.0%}")

    print("\nper-day efficiency (edge divided by average days held):")
    for v in variants:
        if v in stats:
            rets, holds, edge = stats[v]
            print(f"  {v:<14} {edge/holds.mean():+.4f} pp/day")


if __name__ == "__main__":
    main()
