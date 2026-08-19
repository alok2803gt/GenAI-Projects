"""
Real backtest: would a dedicated PRE-BREAKOUT strategy (entry on alert,
exit via stop/target/max-hold tuned to the 3-5 day window where the raw
forward-return data showed a real edge) actually work, and at what params?

Data: 510 real PRE-BREAKOUT alerts, alert_performance table in tape_data.db
(2026-06-25 to 2026-08-17). Uses real daily OHLC (not just close) via
yfinance so stop-loss/profit-target hits are simulated against real
intraday high/low, not just close-to-close.

Caveats stated up front:
  - Daily bars only, not intraday -- a stop/target "hit" checks if that
    day's High/Low crossed the level, not the actual intraday path/order,
    so same-day whipsaws (hit target AND stop same day) are resolved
    conservatively (stop takes priority) but timing within the day isn't modeled.
  - No slippage/commission modeled -- real fills would be somewhat worse.
  - ~510 alerts over 8 weeks, clustered on correlated days/sectors -- NOT
    500 independent trials. Wide result variance across configs should be
    read as noise, not a config-quality signal, unless the pattern is
    large and consistent.
"""
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf

con = sqlite3.connect('tape_data.db')
alerts = pd.read_sql_query(
    "SELECT session_date, ticker, alert_price FROM alert_performance "
    "WHERE signal_type='PRE-BREAKOUT' AND alert_price IS NOT NULL",
    con
)
tickers = sorted(alerts['ticker'].unique())
print(f"{len(alerts)} PRE-BREAKOUT alerts, {len(tickers)} tickers")

data = yf.download(tickers, start='2026-06-20', end='2026-08-19', group_by='ticker',
                    progress=False, threads=True, auto_adjust=True)

def get_path(ticker, alert_date, max_days):
    try:
        df = data[ticker][['High', 'Low', 'Close']].dropna()
    except Exception:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    ad = pd.Timestamp(alert_date)
    future = df[df.index > ad].iloc[:max_days]
    return future if len(future) > 0 else None

def simulate(entry, path, stop_pct, target_pct, max_days):
    """Returns (exit_return_pct, exit_reason, days_held)."""
    stop_px   = entry * (1 - stop_pct / 100)
    target_px = entry * (1 + target_pct / 100)
    for i, (_, row) in enumerate(path.iterrows(), start=1):
        if row['Low'] <= stop_px:
            return (-stop_pct, "stop", i)
        if row['High'] >= target_px:
            return (target_pct, "target", i)
        if i >= max_days:
            ret = (row['Close'] - entry) / entry * 100
            return (ret, "max_hold", i)
    # ran out of available data before max_days
    last_close = path.iloc[-1]['Close']
    ret = (last_close - entry) / entry * 100
    return (ret, "data_end", len(path))

grid = []
for stop_pct in (2, 3, 4):
    for target_pct in (2, 3, 4, 5):
        for max_days in (3, 5, 7):
            trades = []
            for _, row in alerts.iterrows():
                path = get_path(row['ticker'], row['session_date'], max_days)
                if path is None or row['alert_price'] is None or row['alert_price'] <= 0:
                    continue
                ret, reason, days = simulate(row['alert_price'], path, stop_pct, target_pct, max_days)
                trades.append((ret, reason))
            if not trades:
                continue
            rets = [t[0] for t in trades]
            wins = [r for r in rets if r > 0]
            win_rate = len(wins) / len(rets) * 100
            avg_ret = np.mean(rets)
            total_ret = sum(rets)
            reasons = {}
            for _, r in trades:
                reasons[r] = reasons.get(r, 0) + 1
            grid.append({
                "stop_pct": stop_pct, "target_pct": target_pct, "max_days": max_days,
                "n": len(rets), "win_rate": round(win_rate, 1), "avg_ret": round(avg_ret, 3),
                "total_ret": round(total_ret, 1), "reasons": reasons,
            })

df_grid = pd.DataFrame(grid).sort_values("avg_ret", ascending=False)
pd.set_option('display.width', 160)
pd.set_option('display.max_rows', 50)
print(df_grid.to_string(index=False))
df_grid.to_csv("pre_breakout_backtest_results.csv", index=False)

print("\n=== Wider/no-stop comparison at target=5%, max_days=7 ===")
for stop_pct in (4, 6, 8, 100):  # 100 = effectively no stop
    trades = []
    for _, row in alerts.iterrows():
        path = get_path(row['ticker'], row['session_date'], 7)
        if path is None or row['alert_price'] is None or row['alert_price'] <= 0:
            continue
        ret, reason, days = simulate(row['alert_price'], path, stop_pct, 5, 7)
        trades.append(ret)
    if trades:
        wr = sum(1 for r in trades if r > 0) / len(trades) * 100
        label = f"stop={stop_pct}%" if stop_pct < 100 else "no stop"
        print(f"{label:12s} n={len(trades)} win_rate={wr:.1f}% avg_ret={np.mean(trades):+.3f}%")
