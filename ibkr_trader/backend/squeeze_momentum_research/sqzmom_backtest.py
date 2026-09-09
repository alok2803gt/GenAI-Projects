"""
Faithful Python port of LazyBear's Squeeze Momentum Indicator (Pine Script,
the standard TTM Squeeze community implementation) + a real backtest of the
one standard way it's actually traded: enter in momentum's direction when
the squeeze fires. See RESEARCH_LOG.md for full methodology.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent

BB_LENGTH, BB_MULT = 20, 2.0
KC_LENGTH, KC_MULT = 20, 1.5
MAX_HOLDS = [10, 20]


def true_range(df):
    prev_close = df["Close"].shift(1)
    return pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def linreg_series(series, length):
    """True OLS fit over the trailing `length` bars, evaluated at the
    current bar (Pine's linreg(src, length, 0)) -- computed exactly via
    numpy polyfit per window, not a shortcut approximation."""
    values = series.to_numpy()
    out = np.full(len(values), np.nan)
    x = np.arange(length)
    for i in range(length - 1, len(values)):
        window = values[i - length + 1: i + 1]
        if np.any(np.isnan(window)):
            continue
        slope, intercept = np.polyfit(x, window, 1)
        out[i] = intercept + slope * (length - 1)
    return pd.Series(out, index=series.index)


def compute_sqzmom(df):
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]

    basis = close.rolling(BB_LENGTH).mean()
    dev = BB_MULT * close.rolling(BB_LENGTH).std(ddof=0)
    upperBB, lowerBB = basis + dev, basis - dev

    ma = close.rolling(KC_LENGTH).mean()
    tr = true_range(df)
    rangema = tr.rolling(KC_LENGTH).mean()
    upperKC, lowerKC = ma + rangema * KC_MULT, ma - rangema * KC_MULT

    df["sqzOn"] = (lowerBB > lowerKC) & (upperBB < upperKC)
    df["sqzOff"] = (lowerBB < lowerKC) & (upperBB > upperKC)

    highest_hi = high.rolling(KC_LENGTH).max()
    lowest_lo = low.rolling(KC_LENGTH).min()
    sma_close = close.rolling(KC_LENGTH).mean()
    midline = (((highest_hi + lowest_lo) / 2) + sma_close) / 2
    df["val"] = linreg_series(close - midline, KC_LENGTH)

    return df


def backtest_ticker(df, ticker, max_hold):
    df = compute_sqzmom(df)
    df = df.dropna(subset=["val", "sqzOn", "sqzOff"]).reset_index(drop=True)
    trades = []
    i = 1
    n = len(df)
    while i < n - 1:
        fired = bool(df["sqzOn"].iloc[i - 1]) and bool(df["sqzOff"].iloc[i])
        if not fired:
            i += 1
            continue
        direction = 1 if df["val"].iloc[i] > 0 else (-1 if df["val"].iloc[i] < 0 else 0)
        if direction == 0:
            i += 1
            continue
        entry_idx = i + 1  # next day's open, no lookahead
        if entry_idx >= n:
            break
        entry_price = df["Open"].iloc[entry_idx]
        entry_date = df["Date"].iloc[entry_idx] if "Date" in df.columns else entry_idx

        exit_idx = None
        for j in range(entry_idx + 1, min(entry_idx + max_hold + 1, n)):
            prev_val, cur_val = df["val"].iloc[j - 1], df["val"].iloc[j]
            if (direction == 1 and prev_val > 0 and cur_val <= 0) or \
               (direction == -1 and prev_val < 0 and cur_val >= 0):
                exit_idx = j
                break
        if exit_idx is None:
            exit_idx = min(entry_idx + max_hold, n - 1)

        exit_price = df["Close"].iloc[exit_idx]
        ret_pct = direction * (exit_price / entry_price - 1) * 100
        trades.append({
            "ticker": ticker, "direction": "long" if direction == 1 else "short",
            "entry_date": entry_date, "hold_days": exit_idx - entry_idx,
            "ret_pct": ret_pct, "max_hold_used": exit_idx == entry_idx + max_hold,
        })
        i = exit_idx + 1
    return trades


def main():
    with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
        universe = pickle.load(f)
    print(f"Loaded {len(universe)} tickers")

    for max_hold in MAX_HOLDS:
        all_trades = []
        for ticker, df in universe.items():
            df = df.copy()
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            df["Date"] = df.index
            df = df.reset_index(drop=True)
            try:
                all_trades.extend(backtest_ticker(df, ticker, max_hold))
            except Exception as exc:
                print(f"  {ticker}: error {exc}")

        tdf = pd.DataFrame(all_trades)
        out_path = HERE / f"sqzmom_trades_maxhold{max_hold}.csv"
        tdf.to_csv(out_path, index=False)
        print(f"\n{'='*70}\nmax_hold={max_hold} trading days -- {len(tdf)} real trades across {tdf['ticker'].nunique() if len(tdf) else 0} tickers")
        for direction in ["long", "short"]:
            sub = tdf[tdf["direction"] == direction]
            if sub.empty:
                print(f"  {direction}: no trades")
                continue
            win_rate = (sub["ret_pct"] > 0).mean() * 100
            print(f"  {direction:6s} n={len(sub):5d}  win={win_rate:5.1f}%  "
                  f"mean={sub['ret_pct'].mean():+.3f}%  median={sub['ret_pct'].median():+.3f}%  "
                  f"avg_hold={sub['hold_days'].mean():.1f}d  "
                  f"pct_maxhold_exit={sub['max_hold_used'].mean()*100:.1f}%")
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
