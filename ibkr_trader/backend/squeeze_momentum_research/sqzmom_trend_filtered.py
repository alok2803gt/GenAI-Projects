"""
Trend-filtered variant of the squeeze-fire rule: only take a LONG signal
when price is already above its 200-day SMA (primary uptrend), only take a
SHORT when price is below it. Same entry/exit mechanics as
sqzmom_backtest.py (next-open entry, momentum-reversal or max-hold exit).

Critically, the comparison baseline here is also trend-filtered -- "any
random day where price was already above its SMA200" -- not the plain
unconditional baseline from the first pass. Comparing a trend-filtered
signal against an un-filtered baseline would just show "being in an
uptrend helps," which isn't the question; the real question is whether
the SQUEEZE-FIRE TIMING adds anything on top of already being in a trend.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sqzmom_backtest import compute_sqzmom, MAX_HOLDS

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
SMA_TREND = 200


def backtest_ticker_trend(df, ticker, max_hold):
    df = compute_sqzmom(df)
    df["sma200"] = df["Close"].rolling(SMA_TREND).mean()
    df = df.dropna(subset=["val", "sqzOn", "sqzOff", "sma200"]).reset_index(drop=True)
    trades = []
    i = 1
    n = len(df)
    while i < n - 1:
        fired = bool(df["sqzOn"].iloc[i - 1]) and bool(df["sqzOff"].iloc[i])
        if not fired:
            i += 1
            continue
        val_i = df["val"].iloc[i]
        direction = 1 if val_i > 0 else (-1 if val_i < 0 else 0)
        if direction == 0:
            i += 1
            continue
        above_trend = df["Close"].iloc[i] > df["sma200"].iloc[i]
        if (direction == 1 and not above_trend) or (direction == -1 and above_trend):
            i += 1
            continue  # squeeze fired AGAINST the primary trend -- skip

        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry_price = df["Open"].iloc[entry_idx]
        entry_date = df["Date"].iloc[entry_idx]

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
            "entry_date": entry_date, "hold_days": exit_idx - entry_idx, "ret_pct": ret_pct,
        })
        i = exit_idx + 1
    return trades


def trend_matched_baseline(universe, max_hold):
    """Unconditional forward return, but only on days price was already
    above (for long) / below (for short) its own SMA200 -- the fair
    comparison point for the trend-filtered signal above."""
    long_rets, short_rets = [], []
    for ticker, df in universe.items():
        close = df["Close"]
        sma200 = close.rolling(SMA_TREND).mean()
        fwd = (close.shift(-max_hold) / close - 1) * 100
        above = (close > sma200)
        long_rets.append(fwd[above].dropna())
        short_rets.append((-fwd[~above & sma200.notna()]).dropna())
    return pd.concat(long_rets), pd.concat(short_rets)


def main():
    with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
        universe = pickle.load(f)

    for max_hold in MAX_HOLDS:
        all_trades = []
        for ticker, df in universe.items():
            df = df.copy()
            df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
            df["Date"] = df.index
            df = df.reset_index(drop=True)
            try:
                all_trades.extend(backtest_ticker_trend(df, ticker, max_hold))
            except Exception as exc:
                print(f"  {ticker}: error {exc}")
        tdf = pd.DataFrame(all_trades)
        base_long, base_short = trend_matched_baseline(universe, max_hold)

        print(f"\n{'='*78}\nmax_hold={max_hold}d, TREND-FILTERED (only trade with the SMA200 trend) -- {len(tdf)} trades")
        for direction, base in [("long", base_long), ("short", base_short)]:
            sub = tdf[tdf["direction"] == direction] if len(tdf) else pd.DataFrame()
            print(f"  SIGNAL {direction:6s} n={len(sub):5d}  "
                  f"win={(sub['ret_pct']>0).mean()*100 if len(sub) else float('nan'):5.1f}%  "
                  f"mean={sub['ret_pct'].mean() if len(sub) else float('nan'):+.3f}%  "
                  f"median={sub['ret_pct'].median() if len(sub) else float('nan'):+.3f}%")
            print(f"  BASE   {direction:6s} n={len(base):6d}  "
                  f"win={(base>0).mean()*100:5.1f}%  mean={base.mean():+.3f}%  median={base.median():+.3f}%  "
                  f"(any day already {'above' if direction=='long' else 'below'} SMA200, same {max_hold}d hold)")
        tdf.to_csv(HERE / f"sqzmom_trendfiltered_maxhold{max_hold}.csv", index=False)


if __name__ == "__main__":
    main()
