import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np

con = sqlite3.connect('tape_data.db')
alerts = pd.read_sql_query(
    "SELECT session_date, ticker, signal_type, alert_price, eod_return_pct, is_win FROM alert_performance WHERE alert_price IS NOT NULL",
    con
)
tickers = sorted(alerts['ticker'].unique())
print(f"{len(alerts)} alerts, {len(tickers)} unique tickers")

data = yf.download(tickers, start='2026-06-20', end='2026-08-19', group_by='ticker', progress=False, threads=True, auto_adjust=True)

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
    return future.iloc[n_days - 1]

results = []
for _, row in alerts.iterrows():
    ticker, adate, aprice = row['ticker'], row['session_date'], row['alert_price']
    if aprice is None or aprice <= 0:
        continue
    rec = {'ticker': ticker, 'signal_type': row['signal_type'], 'eod_return_pct': row['eod_return_pct']}
    for n in (1, 3, 5):
        fp = fwd_return(ticker, adate, n)
        rec[f'ret_{n}d'] = (fp - aprice) / aprice * 100 if fp is not None else None
    results.append(rec)

df = pd.DataFrame(results)
df.to_csv('breakout_horizon_results.csv', index=False)
print(f"Computed for {len(df)} alerts")

for st in ('BREAKOUT', 'PRE-BREAKOUT'):
    sub = df[df['signal_type'] == st]
    print(f"\n=== {st} ===")
    print(f"same-day (eod): n={sub['eod_return_pct'].notna().sum()} win_rate={ (sub['eod_return_pct']>0).mean()*100:.1f}% avg={sub['eod_return_pct'].mean():.3f}%")
    for n in (1, 3, 5):
        col = f'ret_{n}d'
        valid = sub[col].dropna()
        if len(valid) > 0:
            wr = (valid > 0).mean() * 100
            print(f"+{n}d fwd:     n={len(valid)} win_rate={wr:.1f}% avg={valid.mean():.3f}% median={valid.median():.3f}%")
