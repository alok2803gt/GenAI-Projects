"""
Sequence version of the harami+breakout combined signal, per CEO request
2026-09-16: test BOTH orderings (harami-then-breakout, breakout-then-
harami) across a real window grid (1/3/5/10 bars apart), same 4
timeframes and real per-timeframe data windows as
harami_breakout_multitimeframe.py.

Entry is taken at the CONFIRMING (second) event's bar in each ordering --
e.g. for harami-then-breakout, the harami is the setup, the breakout bar
within the next N bars is the actual entry trigger. Forward returns
measured from that confirming bar, same +1/+5 bar convention as the
same-bar version.
"""
import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE = ["GOOG", "AAPL", "AMZN", "UNH", "SPXC", "MU", "NFLX", "ASML", "META", "MSFT",
            "LRCX", "NVDA", "CVX", "JPM", "COST", "JNJ", "GS", "TSLA", "LLY", "WMT", "MS", "XOM", "HD"]

BODY_LOOKBACK = 60
BB_PERIOD = 20
PCT_B_MIN = 95.0
VOL_RATIO_MIN = 1.5
WINDOWS = [1, 3, 5, 10]

TIMEFRAMES = {
    "weekly": {"interval": "1wk", "period": "5y"},
    "daily":  {"interval": "1d",  "period": "5y"},
    "4h":     {"interval": "1h",  "period": "2y"},
    "15min":  {"interval": "15m", "period": "60d"},
}


def fetch(ticker, interval, period):
    df = yf.Ticker(ticker).history(interval=interval, period=period, auto_adjust=True)
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def to_4h(df):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return df.resample("4h", origin="start_day").agg(agg).dropna()


def compute_signals(df):
    df = df.copy()
    closes, vols = df["Close"], df["Volume"]
    sma20 = closes.rolling(BB_PERIOD).mean()
    std20 = closes.rolling(BB_PERIOD).std()
    upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
    band_w = (upper - lower).replace(0, np.nan)
    pct_b = (closes - lower) / band_w * 100
    avg_vol20 = vols.rolling(20).mean().shift(1)
    vol_ratio = vols / avg_vol20
    df["breakout"] = (pct_b >= PCT_B_MIN) & (vol_ratio >= VOL_RATIO_MIN)

    body = closes - df["Open"]
    abs_body = body.abs()
    prior_open, prior_close = df["Open"].shift(1), closes.shift(1)
    prior_abs_body = abs_body.shift(1)
    body_median = abs_body.rolling(BODY_LOOKBACK).median()
    prior_bearish = prior_close < prior_open
    current_bullish = closes > df["Open"]
    contained = (df["Open"] > prior_close) & (closes < prior_open)
    large_prior_body = prior_abs_body >= body_median.shift(1)
    df["harami"] = prior_bearish & current_bullish & contained & large_prior_body
    return df.reset_index(drop=True)


def find_sequence_trades(df, first_col, second_col, window):
    """First event at bar i; if second event fires at any bar j in
    (i, i+window], entry is at bar j (the confirming bar). Each first-event
    bar triggers at most once (the first confirming occurrence)."""
    n = len(df)
    trades_1, trades_5 = [], []
    first_idx = df.index[df[first_col]].tolist()
    for i in first_idx:
        if i < BODY_LOOKBACK + 1:
            continue
        for j in range(i + 1, min(i + window + 1, n)):
            if df[second_col].iloc[j]:
                if j + 5 >= n:
                    break
                entry = df["Close"].iloc[j]
                trades_1.append((df["Close"].iloc[j + 1] / entry - 1) * 100)
                trades_5.append((df["Close"].iloc[j + 5] / entry - 1) * 100)
                break
    return trades_1, trades_5


def run(name, cfg):
    print(f"=== {name} (real window: {cfg['period']}) ===")
    cached = {}
    for t in UNIVERSE:
        try:
            df = fetch(t, cfg["interval"], cfg["period"])
            if name == "4h":
                df = to_4h(df)
            if len(df) < BODY_LOOKBACK + 30:
                continue
            cached[t] = compute_signals(df)
        except Exception as e:
            print(f"  {t}: error {e}, skipping")

    for order_label, first_col, second_col in [("harami -> breakout", "harami", "breakout"),
                                                  ("breakout -> harami", "breakout", "harami")]:
        print(f"  --- {order_label} ---")
        for w in WINDOWS:
            all_1, all_5, n_tickers = [], [], 0
            for t, df in cached.items():
                tr1, tr5 = find_sequence_trades(df, first_col, second_col, w)
                if tr1:
                    all_1 += tr1; all_5 += tr5; n_tickers += 1
            if all_1:
                win1 = sum(1 for x in all_1 if x > 0) / len(all_1) * 100
                win5 = sum(1 for x in all_5 if x > 0) / len(all_5) * 100
                print(f"    window={w:>2} bars: n={len(all_1):>3} ({n_tickers} tickers)  "
                      f"+1bar avg {sum(all_1)/len(all_1):+.3f}% win {win1:5.1f}%   "
                      f"+5bar avg {sum(all_5)/len(all_5):+.3f}% win {win5:5.1f}%")
            else:
                print(f"    window={w:>2} bars: n=0")
    print()


def main():
    for name, cfg in TIMEFRAMES.items():
        run(name, cfg)


if __name__ == "__main__":
    main()
