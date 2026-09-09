"""
Multi-regime validation of the single strongest candidate found in the
1,023-combo screen (2026-09-01): VolRatio_gt1.2 & CCI_gt100 &
WilliamsR_gt-50 & EMA9_gt_EMA21 -- largest real sample among the top
performers (n=1,661 on the 1-year window, 58.2% win / +0.203% avg with
the account's live 0.3% trailing-stop exit), and robust to the
near-duplicate supersets (adding RSI/ADX/ROC on top changed nothing
material, meaning those extra indicators added no real information).

Per the account's own opportunity-evaluation checklist (portfolio-oversight
skill): a single 1-year backtest is not enough to call an edge real --
this extends to the FULL available history (~5 years, 2021-08 to 2026-08)
and breaks results out BY CALENDAR YEAR so a real stress regime (2022
bear market) shows up on its own, not blended into an average.

Real, stated data-floor limitation: Alpaca's minute-bar history (needed
for the real intraday trailing-stop path) does not go back before
2021-09-02 on this account's plan (confirmed 2026-08-31 across the full
112-ticker universe, including newer/smaller names). That means 2020's
COVID crash is NOT reachable with the real trailing-stop mechanic -- this
script does not attempt to fake it, and says so plainly in the output
rather than silently only covering the years that happen to be available.

Same entry/exit methodology as indicator_combo_backtest_trailstop.py
(same 10 indicator definitions, same t-1-close-only signals, same real
Alpaca-sourced entry/exit prices after the 2026-09-01 yfinance-scale bug
fix, same 0.3% trailing stop walked bar-by-bar through the real intraday
path) -- the only change is the evaluation window (full history instead
of the most recent 252 days) and restricting to this one target combo
instead of sweeping all 1,023.
"""
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
ET = ZoneInfo("America/New_York")
TRAIL_STOP_PCT = 0.003
POSITION_SIZE_USD = 150.0  # matches the account's real live Day Trader position size

with open(BACKEND_DIR / "scanner_config.json") as f:
    CFG = json.load(f)
with open(HERE.parent / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
    daily_data = pickle.load(f)
print(f"Loaded {len(daily_data)} tickers of daily data")


def compute_adx(high, low, close, period=14):
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


# ---------- Step 1: target-combo boolean, FULL available history (not just last 252d) ----------
per_ticker_eval = {}

for ticker, hist in daily_data.items():
    if len(hist) < 300:
        continue
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]

    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()

    vol_avg20 = v.rolling(20).mean()
    vol_ratio = v / vol_avg20.replace(0, np.nan)

    tp = (h + l + c) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

    willr = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()

    target = ((vol_ratio > 1.2) & (cci > 100) & (willr > -50) & (ema9 > ema21)).shift(1)

    valid = target.notna()
    idx = hist.index[valid]
    if len(idx) == 0:
        continue

    df = pd.DataFrame({"target": target.loc[idx], "open": o.loc[idx], "close": c.loc[idx]})
    df = df[df["target"] == True]  # only keep days the signal actually fires -- much smaller set
    if df.empty:
        continue
    per_ticker_eval[ticker] = df

total_signal_days = sum(len(df) for df in per_ticker_eval.values())
print(f"Tickers with at least one signal day: {len(per_ticker_eval)}")
print(f"Total real signal-days across full history: {total_signal_days}")

all_dates = sorted(set(d for df in per_ticker_eval.values() for d in df.index))
first_date, last_date = all_dates[0].date(), all_dates[-1].date()
print(f"Signal date range: {first_date} to {last_date}")
print("NOTE: Alpaca minute-bar history has a real floor at 2021-09-02 on this account's "
      "plan -- 2020 COVID crash is NOT reachable with the real intraday trailing-stop path "
      "and is not included here.")

minute_start = datetime(max(first_date.year, 2021), 9, 2, tzinfo=ET) if first_date.year <= 2021 else \
    datetime(first_date.year, first_date.month, first_date.day, tzinfo=ET) - timedelta(days=5)
minute_end = datetime(last_date.year, last_date.month, last_date.day, 23, 59, tzinfo=ET) + timedelta(days=2)

# ---------- Step 2: real minute bars, full window ----------
client = StockHistoricalDataClient(CFG["alpaca_api_key"], CFG["alpaca_secret_key"])
print(f"Pulling real 1-min bars for {len(per_ticker_eval)} tickers, "
      f"{minute_start.date()} to {minute_end.date()}...")

minute_bars = {}
for i, tk in enumerate(per_ticker_eval):
    try:
        req = StockBarsRequest(symbol_or_symbols=[tk], timeframe=TimeFrame.Minute,
                                start=minute_start, end=minute_end)
        df = client.get_stock_bars(req).df
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[tk]
        df.index = df.index.tz_convert(ET)
        minute_bars[tk] = df.between_time("09:30", "15:59")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(per_ticker_eval)} minute pulls done", flush=True)
    except Exception as exc:
        print(f"  {tk}: minute pull failed ({exc})", flush=True)

print("Minute bar pulls complete.")


def trailing_stop_exit(day_min, entry_price, day_close, trail_pct=TRAIL_STOP_PCT):
    if day_min is None or day_min.empty:
        return day_close, False
    highs = day_min["high"].values.astype(float)
    lows = day_min["low"].values.astype(float)
    running_peak = np.maximum.accumulate(np.concatenate(([entry_price], highs)))[1:]
    stop_levels = running_peak * (1 - trail_pct)
    triggered = lows <= stop_levels
    if triggered.any():
        idx = int(np.argmax(triggered))
        return float(stop_levels[idx]), True
    return day_close, False


records = []
skipped_no_minute_data = 0
for tk, df in per_ticker_eval.items():
    mbars = minute_bars.get(tk)
    day_groups = {}
    if mbars is not None and not mbars.empty:
        day_groups = {ts.date(): g for ts, g in mbars.groupby(mbars.index.normalize())}
    for day_ts, row in df.iterrows():
        day = day_ts.date()
        day_min = day_groups.get(day)
        if day_min is None or day_min.empty:
            skipped_no_minute_data += 1
            continue
        entry_price = float(day_min["open"].iloc[0])
        day_close = float(day_min["close"].iloc[-1])
        exit_price, stopped_out = trailing_stop_exit(day_min, entry_price, day_close)
        ret_trail = (exit_price - entry_price) / entry_price * 100
        ret_blind = (day_close - entry_price) / entry_price * 100
        records.append({
            "ticker": tk, "day": str(day), "year": day.year,
            "ret_trail": ret_trail, "ret_blind": ret_blind, "stopped_out": stopped_out,
        })

print(f"Skipped {skipped_no_minute_data} signal-days with no real Alpaca minute data available "
      "(these are the pre-2021-09-02 days this account's data plan can't reach)")

recs = pd.DataFrame(records)
recs.to_csv(HERE / "target_combo_multiregime_rows.csv", index=False)
print(f"\nTotal real signal-days with a real intraday exit simulated: {len(recs)}")
print(f"Years covered: {sorted(recs.year.unique())}")

print("\n=== BY YEAR (real regime breakdown) ===")
for yr, g in recs.groupby("year"):
    rt, rb = g["ret_trail"], g["ret_blind"]
    print(f"  {yr}: n={len(g):4d}  trail: win={(rt>0).mean()*100:5.1f}% avg={rt.mean():+.3f}%  "
          f"|  blind: win={(rb>0).mean()*100:5.1f}% avg={rb.mean():+.3f}%")

print("\n=== FULL PERIOD ===")
rt, rb = recs["ret_trail"], recs["ret_blind"]
print(f"  trail:  n={len(recs)}  win={(rt>0).mean()*100:.2f}%  avg={rt.mean():+.4f}%  "
      f"median={rt.median():+.4f}%  std={rt.std():.4f}%")
print(f"  blind:  n={len(recs)}  win={(rb>0).mean()*100:.2f}%  avg={rb.mean():+.4f}%  "
      f"median={rb.median():+.4f}%  std={rb.std():.4f}%")

# ---------- Real-dollar downside view at the account's actual live position size ----------
recs_sorted = recs.sort_values("day").reset_index(drop=True)
recs_sorted["pnl_usd"] = recs_sorted["ret_trail"] / 100 * POSITION_SIZE_USD
recs_sorted["cum_pnl"] = recs_sorted["pnl_usd"].cumsum()
running_max = recs_sorted["cum_pnl"].cummax()
drawdown = recs_sorted["cum_pnl"] - running_max
max_dd = drawdown.min()
max_dd_idx = drawdown.idxmin()

# longest losing streak (consecutive losing trades, by day order pooled across tickers)
loss_flags = (recs_sorted["ret_trail"] <= 0).astype(int)
streak = (loss_flags.groupby((loss_flags != loss_flags.shift()).cumsum()).cumsum() * loss_flags)
longest_losing_streak = int(streak.max())

worst_trade = recs_sorted.loc[recs_sorted["ret_trail"].idxmin()]
best_trade = recs_sorted.loc[recs_sorted["ret_trail"].idxmax()]

print(f"\n=== REAL-DOLLAR VIEW at ${POSITION_SIZE_USD:.0f}/trade (Day Trader's real live size), "
      "one signal-day trade per ticker per day (no position-limit/overlap modeled) ===")
print(f"  Total P&L if every signal-day were traded: ${recs_sorted['pnl_usd'].sum():+.2f} "
      f"over {len(recs_sorted)} trades")
print(f"  Max drawdown (peak-to-trough, pooled sequence): ${max_dd:.2f} "
      f"(around {recs_sorted.loc[max_dd_idx, 'day']}, {recs_sorted.loc[max_dd_idx, 'ticker']})")
print(f"  Longest losing streak: {longest_losing_streak} consecutive losing trades")
print(f"  Worst single trade: {worst_trade['ticker']} {worst_trade['day']} {worst_trade['ret_trail']:+.2f}% "
      f"(${worst_trade['pnl_usd']:+.2f})")
print(f"  Best single trade: {best_trade['ticker']} {best_trade['day']} {best_trade['ret_trail']:+.2f}% "
      f"(${best_trade['pnl_usd']:+.2f})")
