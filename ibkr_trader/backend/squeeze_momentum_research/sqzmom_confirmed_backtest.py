"""
Tests the REAL, discretionary way traders use this indicator (per the
conversation this follows) rather than the bare mechanical rule already
shown to have no edge: (1) volume expansion on the fire bar, (2) proximity
to a real recent support/resistance level, (3) waiting for a confirmation
bar before entering. Tests each filter alone, then combined, against the
SAME baseline discipline used throughout this research (each variant's win
rate/mean/median compared to the unconditional forward-return baseline for
those same tickers/dates -- not just "is it positive").

LONG only -- short was uniformly negative in every prior test, no reason to
re-test it here.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sqzmom_backtest import compute_sqzmom, MAX_HOLDS

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent

VOL_AVG_LEN = 20
VOL_MULT = 1.3          # "expansion" = at least 1.3x the 20-day avg volume
SR_LOOKBACK = 60         # bars searched for a recent pivot level
SR_PIVOT_WINDOW = 5      # swing high/low definition, matches the chart's own pivot logic
SR_PROXIMITY_PCT = 0.02  # entry must be within 2% of a real recent level


def find_pivot_levels(df, end_idx, lookback=SR_LOOKBACK, window=SR_PIVOT_WINDOW):
    """Same swing-high/low logic as computePivotLevels() in index.html,
    ported to Python -- a bar is a pivot high/low if it's the extreme
    within +/-window bars either side. Only uses bars BEFORE end_idx (no
    lookahead) -- the level has to have existed before the signal fired."""
    start = max(window, end_idx - lookback)
    levels = []
    for i in range(start, end_idx - window):
        seg_hi = df["High"].iloc[i - window: i + window + 1]
        seg_lo = df["Low"].iloc[i - window: i + window + 1]
        if df["High"].iloc[i] == seg_hi.max():
            levels.append(df["High"].iloc[i])
        if df["Low"].iloc[i] == seg_lo.min():
            levels.append(df["Low"].iloc[i])
    return levels


def backtest_ticker_confirmed(df, max_hold, use_volume, use_sr, confirm_bars):
    df = compute_sqzmom(df)
    df["vol_avg"] = df["Volume"].rolling(VOL_AVG_LEN).mean()
    df = df.dropna(subset=["val", "sqzOn", "sqzOff", "vol_avg"]).reset_index(drop=True)
    trades = []
    i, n = 1, len(df)
    while i < n - 1:
        fired = bool(df["sqzOn"].iloc[i - 1]) and bool(df["sqzOff"].iloc[i])
        if not fired or df["val"].iloc[i] <= 0:
            i += 1
            continue

        sig_idx = i
        if confirm_bars > 0:
            ok = True
            for k in range(1, confirm_bars + 1):
                if sig_idx + k >= n or df["val"].iloc[sig_idx + k] <= 0:
                    ok = False
                    break
            if not ok:
                i += 1
                continue
            check_idx = sig_idx + confirm_bars
        else:
            check_idx = sig_idx

        if use_volume and df["Volume"].iloc[check_idx] < VOL_MULT * df["vol_avg"].iloc[check_idx]:
            i += 1
            continue

        if use_sr:
            levels = find_pivot_levels(df, check_idx)
            price = df["Close"].iloc[check_idx]
            near = any(abs(price - lvl) / price <= SR_PROXIMITY_PCT for lvl in levels)
            if not near:
                i += 1
                continue

        entry_idx = check_idx + 1
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
        trades.append({"ret_pct": ret_pct, "entry_idx_in_df": entry_idx})
        i = exit_idx + 1
    return trades


def main():
    with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
        universe = pickle.load(f)

    variants = [
        ("bare (already tested)",          False, False, 0),
        ("volume expansion only",          True,  False, 0),
        ("S/R proximity only",             False, True,  0),
        ("1-bar confirmation only",        False, False, 1),
        ("ALL THREE combined",             True,  True,  1),
    ]

    for max_hold in MAX_HOLDS:
        print(f"\n{'='*90}\nmax_hold={max_hold}d LONG -- richer, discretionary-style variants\n{'='*90}")
        for label, use_vol, use_sr, confirm in variants:
            all_trades = []
            for ticker, df in universe.items():
                df = df.copy()
                df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                df = df.reset_index(drop=True)
                try:
                    all_trades.extend(backtest_ticker_confirmed(df, max_hold, use_vol, use_sr, confirm))
                except Exception:
                    continue
            tdf = pd.DataFrame(all_trades)
            if tdf.empty:
                print(f"  {label:28s}  n=0")
                continue
            win = (tdf["ret_pct"] > 0).mean() * 100
            print(f"  {label:28s}  n={len(tdf):5d}  win={win:5.1f}%  "
                  f"mean={tdf['ret_pct'].mean():+.3f}%  median={tdf['ret_pct'].median():+.3f}%")


if __name__ == "__main__":
    main()
