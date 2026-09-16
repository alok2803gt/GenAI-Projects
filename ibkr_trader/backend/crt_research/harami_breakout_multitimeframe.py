"""
Harami-bullish + BREAKOUT combined signal, backtested across 4 timeframes,
per CEO request 2026-09-16. Real windows per timeframe (confirmed live via
yfinance's own hard API limits, CEO-approved 2026-09-16 to proceed on this
basis rather than force all 4 to match):
  - Weekly: full real 5 years
  - Daily:  full real 5 years
  - 4-hour (built from real 1h bars): ~2 years (Yahoo's hard 730-day cap on 1h)
  - 15-min: ~60 real days (Yahoo's hard cap on 15m -- cannot be extended)

BREAKOUT definition -- exact match to breakout_scanner.py's own live
criteria (not a new/approximate one): Bollinger %B(20,2) >= 95, real
volume >= 1.5x the trailing 20-period average (the same 1.5 fallback
breakout_scanner.py itself uses when there's insufficient history for its
live percentile threshold -- a legitimate, precedented simplification,
not an invented one). No time-of-day volume projection here since these
are completed historical bars, not an in-progress live one.

Harami definition -- exact match to candlestick_pattern_research/
harami_backtest.py: prior bar bearish, current bar bullish, current body
strictly inside prior body, prior body >= its own 60-period trailing
median.

Universe: breakout_scanner.py's own curated, per-ticker-tuned core list
(the VWAP_RULES/per-ticker signal config), not the full ~147-ticker
incidental alert-history universe -- this is the set the scanner is
actually deliberately built around.
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

TIMEFRAMES = {
    "weekly": {"interval": "1wk", "period": "5y"},
    "daily":  {"interval": "1d",  "period": "5y"},
    "4h":     {"interval": "1h",  "period": "2y"},   # aggregated to 4h after fetch
    "15min":  {"interval": "15m", "period": "60d"},
}


def fetch(ticker, interval, period):
    df = yf.Ticker(ticker).history(interval=interval, period=period, auto_adjust=True)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df


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

    df["combined"] = df["breakout"] & df["harami"]
    return df


def backtest_ticker(df):
    df = compute_signals(df).reset_index(drop=True)
    trades_1, trades_5 = [], []
    n = len(df)
    for i in range(BODY_LOOKBACK + 1, n - 5):
        if not bool(df["combined"].iloc[i]):
            continue
        entry = df["Close"].iloc[i]
        trades_1.append((df["Close"].iloc[i + 1] / entry - 1) * 100)
        trades_5.append((df["Close"].iloc[i + 5] / entry - 1) * 100)
    return trades_1, trades_5


def run_timeframe(name, cfg):
    print(f"--- {name} (real window: {cfg['period']}) ---")
    all_1, all_5 = [], []
    per_ticker = {}
    for t in UNIVERSE:
        try:
            df = fetch(t, cfg["interval"], cfg["period"])
            if name == "4h":
                df = to_4h(df)
            if len(df) < BODY_LOOKBACK + 30:
                print(f"  {t}: insufficient real bars ({len(df)}), skipping")
                continue
            tr1, tr5 = backtest_ticker(df)
            if tr1:
                per_ticker[t] = {"n": len(tr1), "avg_1bar": round(sum(tr1) / len(tr1), 3),
                                  "avg_5bar": round(sum(tr5) / len(tr5), 3)}
                all_1 += tr1
                all_5 += tr5
        except Exception as e:
            print(f"  {t}: error {e}, skipping")
    if all_1:
        win1 = sum(1 for x in all_1 if x > 0) / len(all_1) * 100
        win5 = sum(1 for x in all_5 if x > 0) / len(all_5) * 100
        print(f"  POOLED: n={len(all_1)} real combined signals across {len(per_ticker)} tickers")
        print(f"    +1 bar forward: avg {sum(all_1)/len(all_1):+.3f}%, win rate {win1:.1f}%")
        print(f"    +5 bar forward: avg {sum(all_5)/len(all_5):+.3f}%, win rate {win5:.1f}%")
    else:
        print("  No real combined (breakout + harami) signals found at this timeframe.")
    print()
    return per_ticker, all_1, all_5


def main():
    results = {}
    for name, cfg in TIMEFRAMES.items():
        results[name] = run_timeframe(name, cfg)


if __name__ == "__main__":
    main()
