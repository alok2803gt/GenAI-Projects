"""
New filter on BREAKOUT alerts, per CEO request 2026-09-16: does the
underlying ticker's own DAILY candle, on the alert's session_date, also
form a real bullish harami (the exact same, already-validated definition
from candlestick_pattern_research/harami_backtest.py -- not a new/looser
definition):
  - prior day bearish (close < open)
  - alert day bullish (close > open)
  - alert day's real body strictly inside the prior day's real body
    (open > prior_close AND close < prior_open)
  - prior day's body size >= its own 60-day trailing median (a real, non-
    trivial candle, not noise)

Splits BREAKOUT's real alerts into harami-confirmed vs. not, and re-runs
both the same-day and 5-day-hold capital models on each subset -- the
harami pattern is specifically a reversal signal, and the unfiltered
5-day-hold result already showed BREAKOUT alerts tend to fade/reverse
over 5 days, so this tests whether a real, independently-confirmed
reversal candle on the same day changes that.
"""
import json
import sqlite3
from collections import defaultdict

import pandas as pd
import yfinance as yf

BODY_LOOKBACK = 60


def fetch_ohlc(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, interval="1d",
                        progress=False, group_by="ticker", auto_adjust=True, threads=True)
    out = {}
    for t in tickers:
        try:
            df = data[t][["Open", "Close"]].dropna()
            out[t] = df
        except Exception:
            out[t] = None
    return out


def harami_flags(df):
    body = df["Close"] - df["Open"]
    abs_body = body.abs()
    prior_open, prior_close = df["Open"].shift(1), df["Close"].shift(1)
    prior_abs_body = abs_body.shift(1)
    body_median = abs_body.rolling(BODY_LOOKBACK).median()

    prior_bearish = prior_close < prior_open
    current_bullish = df["Close"] > df["Open"]
    contained = (df["Open"] > prior_close) & (df["Close"] < prior_open)
    large_prior_body = prior_abs_body >= body_median.shift(1)

    return prior_bearish & current_bullish & contained & large_prior_body


def main():
    con = sqlite3.connect(r"C:\Projects\GenAI-Projects\ibkr_trader\backend\tape_data.db")
    cur = con.cursor()
    rows = cur.execute("""
        SELECT session_date, ticker, alert_price, eod_return_pct FROM alert_performance
        WHERE signal_type='BREAKOUT' AND eod_return_pct IS NOT NULL AND alert_price > 0
    """).fetchall()
    tickers = sorted(set(r[1] for r in rows))
    print(f"{len(rows)} real BREAKOUT alerts, {len(tickers)} tickers -- downloading real daily OHLC "
          f"(needs {BODY_LOOKBACK}d of history before the earliest alert for a real trailing median)...")
    ohlc = fetch_ohlc(tickers, start="2026-01-01", end="2026-09-25")

    with open("breakout_5d_returns.json") as f:
        fwd5 = {(r["date"], r["ticker"]): r for r in json.load(f)}

    harami_alerts, non_harami_alerts = [], []
    harami_alerts_5d, non_harami_alerts_5d = [], []
    skipped = 0
    for date, ticker, price, ret_eod in rows:
        df = ohlc.get(ticker)
        if df is None or len(df) == 0:
            skipped += 1
            continue
        flags = harami_flags(df)
        ts = pd.Timestamp(date)
        if ts not in flags.index:
            skipped += 1
            continue
        is_harami = bool(flags.loc[ts]) if not pd.isna(flags.loc[ts]) else False
        rec = {"date": date, "ticker": ticker, "price": price, "ret": ret_eod}
        (harami_alerts if is_harami else non_harami_alerts).append(rec)
        fwd = fwd5.get((date, ticker))
        if fwd:
            rec5 = {"date": date, "ticker": ticker, "price": price, "ret_5d": fwd["ret_5d"], "exit_date": fwd["exit_date"]}
            (harami_alerts_5d if is_harami else non_harami_alerts_5d).append(rec5)

    print(f"Real split: {len(harami_alerts)} harami-confirmed / {len(non_harami_alerts)} not "
          f"({skipped} skipped -- insufficient history)\n")

    def summarize_eod(label, recs):
        if not recs:
            print(f"{label}: 0 alerts\n")
            return
        s = sum(r["ret"] for r in recs)
        print(f"{label}: n={len(recs)}, naive sum (same-day) = {s:+.2f}%, avg = {s/len(recs):+.3f}%")

    def summarize_5d(label, recs):
        if not recs:
            print(f"{label}: 0 alerts with a real 5d outcome\n")
            return
        s = sum(r["ret_5d"] for r in recs)
        print(f"{label}: n={len(recs)}, naive sum (5-day hold) = {s:+.2f}%, avg = {s/len(recs):+.3f}%")

    print("--- Same-day EOD return ---")
    summarize_eod("Harami-confirmed BREAKOUT", harami_alerts)
    summarize_eod("Non-harami BREAKOUT", non_harami_alerts)
    print()
    print("--- 5-day-hold return ---")
    summarize_5d("Harami-confirmed BREAKOUT", harami_alerts_5d)
    summarize_5d("Non-harami BREAKOUT", non_harami_alerts_5d)

    with open("breakout_harami_split.json", "w") as f:
        json.dump({"harami": harami_alerts, "non_harami": non_harami_alerts,
                   "harami_5d": harami_alerts_5d, "non_harami_5d": non_harami_alerts_5d}, f)
    print("\nSaved: breakout_harami_split.json")


if __name__ == "__main__":
    main()
