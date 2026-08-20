"""
Backtest two specific assumptions behind breakout_scanner.py's volume gate
(classify_signal(), ~line 989):
  1. The 90th-percentile choice for each ticker's own vol_90pct threshold --
     is 90th actually better than other percentiles?
  2. The PRE-BREAKOUT vol_threshold_pct=0.75 multiplier -- is a single flat
     75% applied uniformly to every ticker's own threshold actually a good
     choice, or would a different flat multiplier do better?

Real universe: watchlist.json (80 tickers, confirmed live scanner universe --
matches every ticker seen in tape_data.db's real alert_history).

Methodology (daily bars, NOT identical to the live intraday system --
approximations stated explicitly):
  - Signal for ticker/day T is evaluated using price/volume data THROUGH
    day T's close (same convention as the live scanner using today's
    volume), which is what's available from yfinance daily bars.
  - vol_ratio for day T = T's own volume / T's trailing-20-day average
    volume (as of T-1, no lookahead on the average). NOTE: unlike the live
    scanner (which projects a PARTIAL day's volume forward), this uses T's
    REAL, complete volume -- less noisy but not identical.
  - Each ticker's vol_90pct (or vol_Npct for other percentiles tested) is
    computed from that ticker's OWN trailing ratio history strictly BEFORE
    day T (up to 252 prior days) -- shifted, no lookahead.
  - pct_b (20,2 Bollinger %B), RSI-14 (Wilder), above_sma20/50 all computed
    identically to breakout_scanner.py's real formulas.
  - Outcome = NEXT trading day's close-to-close return (T -> T+1). This is
    the real approximation vs. the live system (which fires intraday and
    resolves same-day EOD) -- daily bars can't replicate that, so this
    tests "if this fired at T's close, what happened the next full day"
    instead. Stated plainly, not hidden.

Universe/period: 80 watchlist tickers, ~2.5 years of daily history (yfinance),
tested over the most recent ~1.5 years once the 252-day ratio warmup is
satisfied per ticker.
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

with open("watchlist.json") as f:
    TICKERS = sorted(json.load(f).keys())

print(f"Universe: {len(TICKERS)} tickers")
print("Downloading ~2.5y daily history...")
raw = yf.download(TICKERS, period="30mo", interval="1d", group_by="ticker",
                   auto_adjust=True, progress=False, threads=True)

PCT_B_BREAKOUT_MIN = 95
PCT_B_PRE_MIN = 65
RSI_PRE_MIN = 60
PERCENTILES_TO_TEST = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
MULTIPLIERS_TO_TEST = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]

records = []  # one row per ticker/day with all computed fields

for ticker in TICKERS:
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                continue
            df = raw[ticker].dropna(subset=["Close"])
        else:
            df = raw.dropna(subset=["Close"])
        if len(df) < 300:
            continue

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        band_w = (upper - lower).replace(0, np.nan)
        pct_b = (close - lower) / band_w * 100

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.clip(lower=1e-9)
        rsi = 100 - 100 / (1 + rs)

        sma50 = close.rolling(50).mean()
        above_sma20 = close > sma20
        above_sma50 = close > sma50

        roll_avg = vol.rolling(20).mean().shift(1)
        vol_ratio = vol / roll_avg.replace(0, np.nan)

        # trailing percentile thresholds computed FRESH each day from prior
        # history only (shift(1) so today's own ratio never leaks into its
        # own threshold) -- one rolling-quantile series per percentile tested
        pctl_thresholds = {}
        for p in PERCENTILES_TO_TEST:
            pctl_thresholds[p] = vol_ratio.shift(1).rolling(252, min_periods=20).quantile(p)

        next_ret = close.pct_change().shift(-1) * 100  # T -> T+1 close-to-close

        n = len(df)
        for i in range(260, n - 1):  # warmup + need a next-day outcome
            vr = vol_ratio.iloc[i]
            pb = pct_b.iloc[i]
            r  = rsi.iloc[i]
            bullish = bool(above_sma20.iloc[i]) and bool(above_sma50.iloc[i])
            ret = next_ret.iloc[i]
            if pd.isna(vr) or pd.isna(pb) or pd.isna(r) or pd.isna(ret):
                continue
            row = {"ticker": ticker, "vol_ratio": float(vr), "pct_b": float(pb),
                   "rsi": float(r), "bullish": bullish, "next_ret": float(ret)}
            for p in PERCENTILES_TO_TEST:
                th = pctl_thresholds[p].iloc[i]
                row[f"th_{p}"] = float(th) if not pd.isna(th) else None
            records.append(row)
    except Exception as exc:
        print(f"  {ticker}: skipped ({exc})")

print(f"Total ticker-days with full data: {len(records)}")
recs = pd.DataFrame(records)
recs.to_csv("breakout_volume_threshold_backtest_rows.csv", index=False)


def classify_and_stats(df, percentile, multiplier):
    th_col = f"th_{percentile}"
    valid = df[df[th_col].notna()].copy()
    th = valid[th_col]

    breakout = valid[(valid["pct_b"] > PCT_B_BREAKOUT_MIN) & (valid["vol_ratio"] >= th)]
    pre = valid[
        (valid["pct_b"] >= PCT_B_PRE_MIN) & (valid["pct_b"] <= PCT_B_BREAKOUT_MIN) &
        (valid["rsi"] >= RSI_PRE_MIN) & (valid["bullish"]) &
        (valid["vol_ratio"] >= th * multiplier)
    ]
    out = {}
    for label, sub in (("BREAKOUT", breakout), ("PRE-BREAKOUT", pre), ("BOTH", pd.concat([breakout, pre]))):
        n = len(sub)
        if n == 0:
            out[label] = (0, None, None)
            continue
        win_rate = float((sub["next_ret"] > 0).mean() * 100)
        avg_ret = float(sub["next_ret"].mean())
        out[label] = (n, win_rate, avg_ret)
    return out


print("\n" + "="*70)
print("ASSUMPTION 1: does the 90th-percentile choice matter?")
print("(multiplier held at production value 0.75 for PRE-BREAKOUT)")
print("="*70)
for p in PERCENTILES_TO_TEST:
    stats = classify_and_stats(recs, p, 0.75)
    for label in ("BREAKOUT", "PRE-BREAKOUT", "BOTH"):
        n, wr, ar = stats[label]
        if n:
            print(f"  pctl={p:.2f}  {label:<13} n={n:<5} win_rate={wr:5.1f}%  avg_next_day_ret={ar:+.3f}%")
        else:
            print(f"  pctl={p:.2f}  {label:<13} n=0")
    print()

print("="*70)
print("ASSUMPTION 2: does the 0.75 PRE-BREAKOUT multiplier matter?")
print("(percentile held at production value 0.90)")
print("="*70)
for m in MULTIPLIERS_TO_TEST:
    stats = classify_and_stats(recs, 0.90, m)
    n, wr, ar = stats["PRE-BREAKOUT"]
    if n:
        print(f"  mult={m:.2f}  PRE-BREAKOUT  n={n:<5} win_rate={wr:5.1f}%  avg_next_day_ret={ar:+.3f}%")
    else:
        print(f"  mult={m:.2f}  PRE-BREAKOUT  n=0")

print("\nDone. Full row-level data saved to breakout_volume_threshold_backtest_rows.csv")
