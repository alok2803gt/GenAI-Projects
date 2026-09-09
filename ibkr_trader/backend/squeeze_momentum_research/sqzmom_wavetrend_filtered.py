"""
Tests the OTHER pairing LazyBear himself recommended -- squeeze-fire +
WaveTrend Oscillator (also a LazyBear indicator, "WaveTrend Oscillator
[LazyBear]", one of TradingView's other most-copied scripts). Faithful port
of its standard formula:

    ap  = hlc3 = (H+L+C)/3
    esa = EMA(ap, 10)
    d   = EMA(|ap - esa|, 10)
    ci  = (ap - esa) / (0.015 * d)
    wt1 = EMA(ci, 21)
    wt2 = SMA(wt1, 4)

Standard bullish confirmation reads: wt1 > wt2 (bullish crossover state)
and/or wt1 > 0 (above the zero line, bullish territory) -- tested
separately and combined, same as the ADX variants, against the same
baseline-comparison discipline as every other test in this research.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sqzmom_backtest import compute_sqzmom, MAX_HOLDS

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
WT_N1, WT_N2, WT_SMA = 10, 21, 4


def compute_wavetrend(df):
    ap = (df["High"] + df["Low"] + df["Close"]) / 3
    esa = ap.ewm(span=WT_N1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=WT_N1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d.replace(0, np.nan))
    wt1 = ci.ewm(span=WT_N2, adjust=False).mean()
    wt2 = wt1.rolling(WT_SMA).mean()
    return wt1, wt2


def backtest_ticker_wt(df, max_hold, require_cross, require_positive):
    df = compute_sqzmom(df)
    df["wt1"], df["wt2"] = compute_wavetrend(df)
    df = df.dropna(subset=["val", "sqzOn", "sqzOff", "wt1", "wt2"]).reset_index(drop=True)
    trades = []
    i, n = 1, len(df)
    while i < n - 1:
        fired = bool(df["sqzOn"].iloc[i - 1]) and bool(df["sqzOff"].iloc[i])
        if not fired or df["val"].iloc[i] <= 0:
            i += 1
            continue
        if require_cross and not (df["wt1"].iloc[i] > df["wt2"].iloc[i]):
            i += 1
            continue
        if require_positive and not (df["wt1"].iloc[i] > 0):
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry_price = df["Open"].iloc[entry_idx]
        exit_idx = None
        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
            if df["val"].iloc[j - 1] > 0 and df["val"].iloc[j] <= 0:
                exit_idx = j
                break
        if exit_idx is None:
            exit_idx = min(entry_idx + max_hold, n - 1)
        ret_pct = (df["Close"].iloc[exit_idx] / entry_price - 1) * 100
        trades.append({"ret_pct": ret_pct})
        i = exit_idx + 1
    return trades


def main():
    with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
        universe = pickle.load(f)

    variants = [
        ("no WT filter (bare)",        False, False),
        ("wt1 > wt2 (bullish cross)",  True,  False),
        ("wt1 > 0 (bullish zone)",     False, True),
        ("both combined",              True,  True),
    ]

    for max_hold in MAX_HOLDS:
        print(f"\n{'='*90}\nmax_hold={max_hold}d LONG -- squeeze-fire + WaveTrend (LazyBear's other recommended pairing)\n{'='*90}")
        for label, req_cross, req_pos in variants:
            all_trades = []
            for ticker, df in universe.items():
                df = df.copy()
                df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                df = df.reset_index(drop=True)
                try:
                    all_trades.extend(backtest_ticker_wt(df, max_hold, req_cross, req_pos))
                except Exception:
                    continue
            tdf = pd.DataFrame(all_trades)
            if tdf.empty:
                print(f"  {label:28s}  n=0")
                continue
            win = (tdf["ret_pct"] > 0).mean() * 100
            print(f"  {label:28s}  n={len(tdf):5d}  win={win:5.1f}%  "
                  f"mean={tdf['ret_pct'].mean():+.3f}%  median={tdf['ret_pct'].median():+.3f}%")

        baseline = []
        for ticker, df in universe.items():
            close = df["Close"]
            fwd = (close.shift(-max_hold) / close - 1) * 100
            baseline.append(fwd.dropna())
        base = pd.concat(baseline)
        print(f"  {'BASELINE (any random day)':28s}  n={len(base):5d}  win={(base>0).mean()*100:5.1f}%  "
              f"mean={base.mean():+.3f}%  median={base.median():+.3f}%")


if __name__ == "__main__":
    main()
