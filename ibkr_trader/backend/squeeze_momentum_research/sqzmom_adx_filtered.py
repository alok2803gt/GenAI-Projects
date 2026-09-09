"""
Tests the pairing LazyBear himself recommended -- squeeze-fire + ADX (trend
strength), not a filter this research invented. Wilder's ADX, standard
14-period, computed properly via Wilder smoothing (equivalent to an EWM
with alpha=1/period, the standard correct way to replicate it in pandas --
not a plain rolling average, which is a different, less accurate
approximation).

Tests both the classic threshold read (ADX > 20, ADX > 25 = "trend strong
enough to trust a breakout") and "ADX rising" (trend strengthening, not
just already elevated) since both readings are common in how ADX is
actually taught. Same baseline-comparison discipline as every other test
in this research -- a filter only counts as a real finding if it beats the
unconditional baseline, not just if it's "better than the bare rule."
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sqzmom_backtest import compute_sqzmom, true_range, MAX_HOLDS

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
ADX_PERIOD = 14


def compute_adx(df, period=ADX_PERIOD):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    # Wilder smoothing == EWM with alpha=1/period (adjust=False) -- the
    # standard, correct way to replicate Wilder's original recursive
    # smoothing formula, not a plain rolling mean.
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm_s / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def backtest_ticker_adx(df, max_hold, adx_min, require_rising):
    df = compute_sqzmom(df)
    df["adx"] = compute_adx(df)
    df = df.dropna(subset=["val", "sqzOn", "sqzOff", "adx"]).reset_index(drop=True)
    trades = []
    i, n = 1, len(df)
    while i < n - 1:
        fired = bool(df["sqzOn"].iloc[i - 1]) and bool(df["sqzOff"].iloc[i])
        if not fired or df["val"].iloc[i] <= 0:
            i += 1
            continue
        adx_now = df["adx"].iloc[i]
        if adx_now < adx_min:
            i += 1
            continue
        if require_rising and not (adx_now > df["adx"].iloc[i - 1]):
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
        ("no ADX filter (bare)",     0,  False),
        ("ADX > 20",                20,  False),
        ("ADX > 25",                25,  False),
        ("ADX > 20 AND rising",     20,  True),
        ("ADX > 25 AND rising",     25,  True),
    ]

    for max_hold in MAX_HOLDS:
        print(f"\n{'='*90}\nmax_hold={max_hold}d LONG -- squeeze-fire + ADX (LazyBear's own recommended pairing)\n{'='*90}")
        for label, adx_min, rising in variants:
            all_trades = []
            for ticker, df in universe.items():
                df = df.copy()
                df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                df = df.reset_index(drop=True)
                try:
                    all_trades.extend(backtest_ticker_adx(df, max_hold, adx_min, rising))
                except Exception:
                    continue
            tdf = pd.DataFrame(all_trades)
            if tdf.empty:
                print(f"  {label:24s}  n=0")
                continue
            win = (tdf["ret_pct"] > 0).mean() * 100
            print(f"  {label:24s}  n={len(tdf):5d}  win={win:5.1f}%  "
                  f"mean={tdf['ret_pct'].mean():+.3f}%  median={tdf['ret_pct'].median():+.3f}%")

        baseline = []
        for ticker, df in universe.items():
            close = df["Close"]
            fwd = (close.shift(-max_hold) / close - 1) * 100
            baseline.append(fwd.dropna())
        base = pd.concat(baseline)
        print(f"  {'BASELINE (any random day)':24s}  n={len(base):5d}  win={(base>0).mean()*100:5.1f}%  "
              f"mean={base.mean():+.3f}%  median={base.median():+.3f}%")


if __name__ == "__main__":
    main()
