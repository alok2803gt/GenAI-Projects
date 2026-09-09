"""
See RESEARCH_LOG.md for the full finding synthesis (bare test, matched
baseline, per-ticker significance + binomial robustness check, and the
live harami_scanner.py deployment).

Bullish Harami backtest -- same universe (112 tickers, 5y daily,
breakout_research/universe_5y_ohlcv.pkl) and same baseline-comparison
discipline as squeeze_momentum_research/.

Standard definition (Nison-style, the textbook version):
  - Prior bar: bearish (close < open), a "large" real body.
  - Current bar: bullish (close > open), body FULLY contained inside the
    prior bar's body: current_open > prior_close AND current_close < prior_open.

Classic theory says this only means something as a REVERSAL signal in the
context of a preceding downtrend -- tested bare (pattern alone) and with a
downtrend-context filter (price below its own 20/50-day SMA, or has
declined over the prior N days), matching exactly how the squeeze research
tested "does context matter" for ADX/WaveTrend.

No lookahead: pattern completes at bar t's close, entry at bar t+1's open.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
HOLD_PERIODS = [5, 10, 20]
BODY_SIZE_PCTILE = 0.5   # prior bar's body must be >= its own trailing median (a real, "large" body, not tiny noise)
BODY_LOOKBACK = 60


def find_harami_signals(df):
    df = df.copy()
    body = (df["Close"] - df["Open"])
    abs_body = body.abs()
    prior_open, prior_close = df["Open"].shift(1), df["Close"].shift(1)
    prior_abs_body = abs_body.shift(1)
    body_median = abs_body.rolling(BODY_LOOKBACK).median()

    prior_bearish = prior_close < prior_open
    current_bullish = df["Close"] > df["Open"]
    contained = (df["Open"] > prior_close) & (df["Close"] < prior_open)
    large_prior_body = prior_abs_body >= body_median.shift(1)

    df["harami"] = prior_bearish & current_bullish & contained & large_prior_body
    df["sma20"] = df["Close"].rolling(20).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    df["downtrend_ctx"] = (df["Close"] < df["sma20"]) & (df["Close"] < df["sma50"])
    return df


def backtest_ticker(df, hold, require_downtrend):
    df = find_harami_signals(df).reset_index(drop=True)
    trades = []
    n = len(df)
    for i in range(BODY_LOOKBACK + 1, n - 1):
        if not bool(df["harami"].iloc[i]):
            continue
        if require_downtrend and not bool(df["downtrend_ctx"].iloc[i]):
            continue
        entry_idx = i + 1
        if entry_idx + hold >= n:
            continue
        entry_price = df["Open"].iloc[entry_idx]
        exit_price = df["Close"].iloc[entry_idx + hold]
        ret_pct = (exit_price / entry_price - 1) * 100
        trades.append(ret_pct)
    return trades


def baseline_returns(universe, hold):
    rets = []
    for ticker, df in universe.items():
        close = df["Close"]
        fwd = (close.shift(-hold) / close - 1) * 100
        rets.append(fwd.dropna())
    return pd.concat(rets)


def main():
    with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
        universe = pickle.load(f)
    print(f"Loaded {len(universe)} tickers\n")

    for hold in HOLD_PERIODS:
        print(f"{'='*90}\nhold={hold}d\n{'='*90}")
        for label, require_dt in [("bare (pattern anywhere)", False), ("+ prior downtrend context", True)]:
            all_rets = []
            for ticker, df in universe.items():
                df = df.copy()
                df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                df = df.reset_index(drop=True)
                try:
                    all_rets.extend(backtest_ticker(df, hold, require_dt))
                except Exception:
                    continue
            r = pd.Series(all_rets)
            if r.empty:
                print(f"  {label:28s}  n=0")
                continue
            print(f"  {label:28s}  n={len(r):5d}  win={(r>0).mean()*100:5.1f}%  "
                  f"mean={r.mean():+.3f}%  median={r.median():+.3f}%")
        base = baseline_returns(universe, hold)
        print(f"  {'BASELINE (any random day)':28s}  n={len(base):6d}  win={(base>0).mean()*100:5.1f}%  "
              f"mean={base.mean():+.3f}%  median={base.median():+.3f}%")


if __name__ == "__main__":
    main()
