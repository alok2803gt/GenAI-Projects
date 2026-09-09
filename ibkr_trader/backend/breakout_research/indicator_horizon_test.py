"""
Extends breakout_horizon_analysis.py's real, already-validated methodology
(real yfinance forward returns at 1d/3d/5d) but joins it against the actual
per-alert indicator columns already captured in alert_performance (rsi,
pct_b, vol_ratio, tape_label, mins_in_pre_breakout) so Iteration-1's
correlational leads (found only against same-day EOD return) can be
re-tested against real multi-day forward returns -- a longer, less noisy
horizon. Read-only against tape_data.db; writes only to this directory.
"""
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf

con = sqlite3.connect(r"C:\Projects\GenAI-Projects\ibkr_trader\backend\tape_data.db")
alerts = pd.read_sql_query("""
    SELECT session_date, ticker, signal_type, alert_price, eod_return_pct, is_win,
           pct_b, rsi, vol_ratio, tape_label, prev_state, mins_in_pre_breakout
    FROM alert_performance WHERE alert_price IS NOT NULL
""", con)
print(f"{len(alerts)} alerts, {alerts['ticker'].nunique()} unique tickers, "
      f"{alerts['session_date'].min()} to {alerts['session_date'].max()}")

tickers = sorted(alerts['ticker'].unique())
data = yf.download(tickers, start='2026-06-20', end='2026-08-26',
                    group_by='ticker', progress=False, threads=True, auto_adjust=True)


def fwd_return(ticker, alert_date, n_days):
    try:
        closes = data[ticker]['Close'].dropna()
    except Exception:
        return None
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    ad = pd.Timestamp(alert_date)
    future = closes[closes.index > ad]
    if len(future) < n_days:
        return None
    return float(future.iloc[n_days - 1])


rows = []
for _, r in alerts.iterrows():
    ticker, adate, aprice = r['ticker'], r['session_date'], r['alert_price']
    if aprice is None or aprice <= 0:
        continue
    rec = dict(r)
    for n in (1, 3, 5):
        fp = fwd_return(ticker, adate, n)
        rec[f'ret_{n}d'] = (fp - aprice) / aprice * 100 if fp is not None else None
    rows.append(rec)

df = pd.DataFrame(rows)
df.to_csv("indicator_horizon_joined.csv", index=False)
print(f"Saved {len(df)} rows to indicator_horizon_joined.csv")


def report(label, mask, horizon_col):
    sub = df[mask]
    valid = sub[horizon_col].dropna()
    if len(valid) < 10:
        return
    wr = (valid > 0).mean() * 100
    print(f"  {label:32} n={len(valid):4} win_rate={wr:5.1f}% avg={valid.mean():+.3f}% median={valid.median():+.3f}%")


for horizon_col, hname in [('eod_return_pct', 'EOD'), ('ret_1d', '+1d'), ('ret_3d', '+3d'), ('ret_5d', '+5d')]:
    print(f"\n=== horizon: {hname} ===")
    report("ALL (baseline)", df.index == df.index, horizon_col)

    print(" -- RSI --")
    report("RSI < 60", df['rsi'] < 60, horizon_col)
    report("RSI 60-70", (df['rsi'] >= 60) & (df['rsi'] < 70), horizon_col)
    report("RSI >= 70 (overbought)", df['rsi'] >= 70, horizon_col)

    print(" -- pct_b --")
    report("pct_b 50-95 (not extended)", (df['pct_b'] >= 50) & (df['pct_b'] < 95), horizon_col)
    report("pct_b 95-100 (fresh breakout)", (df['pct_b'] >= 95) & (df['pct_b'] < 100), horizon_col)
    report("pct_b >= 100 (extended)", df['pct_b'] >= 100, horizon_col)

    print(" -- vol_ratio --")
    report("vol_ratio < 0.5", df['vol_ratio'] < 0.5, horizon_col)
    report("vol_ratio 0.5-1.5", (df['vol_ratio'] >= 0.5) & (df['vol_ratio'] < 1.5), horizon_col)
    report("vol_ratio >= 1.5 (confirmed)", df['vol_ratio'] >= 1.5, horizon_col)

    print(" -- combined lead: RSI<70 AND pct_b<100 AND vol_ratio>=0.5 --")
    combo = (df['rsi'] < 70) & (df['pct_b'] < 100) & (df['vol_ratio'] >= 0.5)
    report("combo filter passes", combo, horizon_col)
    report("combo filter fails (rest)", ~combo, horizon_col)
