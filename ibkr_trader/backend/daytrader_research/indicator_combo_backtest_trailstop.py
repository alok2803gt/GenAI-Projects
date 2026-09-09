"""
Re-runs indicator_combo_backtest.py's 1,023-combo same-day screen with a
REAL managed exit instead of blind buy-open/sell-close -- CEO request
2026-09-01, following up on "chasing just technicals gives 50-50, what
are we missing": the account's own Day Trader research already proved
this exact mechanism matters (blind-entry baseline = 67.3% win rate but
NEGATIVE avg return; adding a 0.3% trailing stop = 50.5% win rate but
POSITIVE avg return, and that config is this account's live strategy
today). This script applies the SAME 0.3% trailing stop to every one of
the 1,023 indicator combinations, to see whether their ceiling is really
the entry signal or the (previously blind) exit.

Critical methodology note, learned the hard way earlier in this account's
own research (daytrader_sizing_backtest.py, 2026-08-25): a trailing stop
CANNOT be faithfully simulated from daily OHLC alone -- daily bars don't
reveal whether the high or the low happened first, which made an earlier
version of this exact question "unresolvable" (26.7% or 92.7% win rate
depending on assumed intraday order). daytrader_intraday_backtest.py
fixed that by dropping to real 1-minute bars, where the ambiguity is
small enough to accept a standard, stated convention. This script does
the same: real Alpaca 1-minute bars for the full evaluation window,
walked bar-by-bar in chronological order, peak-then-stop-check per bar
(a standard trailing-stop convention, stated here rather than left
implicit).

Entry signal, universe, evaluation window, and all 10 indicator
definitions are UNCHANGED from indicator_combo_backtest.py (same
breakout_research/universe_5y_ohlcv.pkl daily bars, same t-1-close-only
indicators, same most-recent-252-trading-day window) -- the only change
is the exit: real intraday path + 0.3% trailing stop instead of a blind
hold to the close. Read-only against Alpaca's API and the existing
pickle; never touches daytrader_scanner.py, day_trader live code, or any
config.
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
from itertools import combinations

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
ET = ZoneInfo("America/New_York")
TRAIL_STOP_PCT = 0.003  # 0.3%, same as the account's live Day Trader config

with open(BACKEND_DIR / "scanner_config.json") as f:
    CFG = json.load(f)
with open(HERE.parent / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
    daily_data = pickle.load(f)
print(f"Loaded {len(daily_data)} tickers of daily data")

IND_NAMES = [
    "RSI_50_70", "MACD_bull", "BB_pctB_0.8-1.0", "ADX_gt25", "Stoch_bull",
    "VolRatio_gt1.2", "CCI_gt100", "WilliamsR_gt-50", "EMA9_gt_EMA21", "ROC10_pos",
]
N_IND = len(IND_NAMES)


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


# ---------- Step 1: same indicator computation as indicator_combo_backtest.py ----------
per_ticker_eval = {}  # ticker -> DataFrame indexed by date with bools + eval flag

for ticker, hist in daily_data.items():
    if len(hist) < 300:
        continue
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_w = (upper - lower).replace(0, np.nan)
    pct_b = (c - lower) / band_w

    adx = compute_adx(h.values, l.values, c.values, 14)
    adx.index = c.index

    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    vol_avg20 = v.rolling(20).mean()
    vol_ratio = v / vol_avg20.replace(0, np.nan)

    tp = (h + l + c) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

    willr = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()

    roc10 = (c / c.shift(10) - 1) * 100

    b1 = ((rsi >= 50) & (rsi < 70)).shift(1)
    b2 = (macd_line > macd_signal).shift(1)
    b3 = ((pct_b >= 0.8) & (pct_b <= 1.0)).shift(1)
    b4 = (adx > 25).shift(1)
    b5 = ((stoch_k > stoch_d) & (stoch_k < 80)).shift(1)
    b6 = (vol_ratio > 1.2).shift(1)
    b7 = (cci > 100).shift(1)
    b8 = (willr > -50).shift(1)
    b9 = (ema9 > ema21).shift(1)
    b10 = (roc10 > 0).shift(1)

    bools = pd.concat([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10], axis=1)
    bools.columns = IND_NAMES

    valid = bools.notna().all(axis=1)
    idx = hist.index[valid]
    if len(idx) == 0:
        continue
    eval_idx = idx[-252:]

    df = bools.loc[eval_idx].copy()
    df["open"] = o.loc[eval_idx]
    df["close"] = c.loc[eval_idx]
    per_ticker_eval[ticker] = df

print(f"Tickers with a valid evaluation window: {len(per_ticker_eval)}")
all_dates = sorted(set(d for df in per_ticker_eval.values() for d in df.index))
first_date, last_date = all_dates[0].date(), all_dates[-1].date()
print(f"Evaluation date range: {first_date} to {last_date}")

minute_start = datetime(first_date.year, first_date.month, first_date.day, tzinfo=ET) - timedelta(days=5)
minute_end = datetime(last_date.year, last_date.month, last_date.day, 23, 59, tzinfo=ET) + timedelta(days=2)

# ---------- Step 2: real 1-minute bars for the exit simulation ----------
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


# ---------- Step 3: real trailing-stop exit simulation, bar by bar ----------
def trailing_stop_exit(day_min, entry_price, day_close, trail_pct=TRAIL_STOP_PCT):
    """Real intraday walk, vectorized: peak trails the running max-so-far
    high (starting from entry), stop trails 0.3% below it, checked against
    each bar's real low. Update-peak-then-check-stop per bar is a standard
    trailing-stop convention -- a new high in a bar is allowed to move the
    stop before that same bar's low is checked against it. Falls through
    to the real daily close if never triggered."""
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
n_days_processed = 0
skipped_no_minute_data = 0
for tk, df in per_ticker_eval.items():
    mbars = minute_bars.get(tk)
    # Perf fix 2026-09-01: the first attempt re-scanned mbars (a full year,
    # ~98k rows) with a boolean-mask comparison for EVERY evaluation day
    # (~252/ticker) -- ~2.8 billion comparisons total, on pace for 6+ hours.
    # Group once per ticker into a dict keyed by calendar day (same pattern
    # already proven in faithful_backtest_5y_timeofday.py) for O(1) lookups.
    day_groups = {}
    if mbars is not None and not mbars.empty:
        day_groups = {ts.date(): g for ts, g in mbars.groupby(mbars.index.normalize())}
    for day_ts, row in df.iterrows():
        day = day_ts.date()
        day_min = day_groups.get(day)
        # Real bug found 2026-09-01: yfinance's daily Open/Close for several
        # tickers in universe_5y_ohlcv.pkl (BKNG, CRWD, KLAC, NOW, XLE, XLK,
        # NFLX, ...) carries a spurious per-ticker back-adjustment factor
        # that yfinance's OWN daily bars are internally consistent with
        # (same-day open-to-close % returns are unaffected -- confirmed
        # against BKNG 2026-02-19: yfinance's own -1.99% intraday return
        # matches Alpaca's real -2.00% move that day), but which does NOT
        # match Alpaca's real, unadjusted intraday price levels -- mixing
        # the two sources for entry vs. intraday-path made the trailing
        # stop compare a ~$164 entry against a ~$4,094 real intraday path.
        # Fix: source BOTH entry and exit from Alpaca alone (the same
        # provider as the intraday path), never from the yfinance pickle's
        # dollar levels. Skip the (rare) day if Alpaca has no minute data.
        if day_min is None or day_min.empty:
            skipped_no_minute_data += 1
            continue
        entry_price = float(day_min["open"].iloc[0])
        day_close = float(day_min["close"].iloc[-1])
        exit_price, stopped_out = trailing_stop_exit(day_min, entry_price, day_close)
        ret = (exit_price - entry_price) / entry_price * 100
        rec = {"ticker": tk, "day": str(day), "ret_trail": ret, "stopped_out": stopped_out,
               "ret_blind": (day_close - entry_price) / entry_price * 100}
        for name in IND_NAMES:
            rec[name] = bool(row[name])
        records.append(rec)
        n_days_processed += 1
    if n_days_processed % 5000 < len(df):
        print(f"  processed ~{n_days_processed} ticker-days so far...", flush=True)

print(f"Skipped {skipped_no_minute_data} ticker-days with no real Alpaca minute data available")

print(f"\nTotal ticker-days simulated with real intraday trailing-stop exit: {len(records)}")
recs = pd.DataFrame(records)
recs.to_csv(HERE / "trailstop_rows.csv", index=False)
print("Saved raw rows to trailstop_rows.csv")

print(f"\nStopped out intraday: {recs['stopped_out'].sum()} / {len(recs)} "
      f"({recs['stopped_out'].mean()*100:.1f}%)")
print(f"\nUNCONDITIONAL comparison (n={len(recs)}):")
for col in ("ret_blind", "ret_trail"):
    v = recs[col]
    print(f"  {col}: win_rate={(v>0).mean()*100:.2f}% avg={v.mean():+.4f}% median={v.median():+.4f}%")

# ---------- Step 4: recompute all 1,023 combos for BOTH exit mechanics ----------
weights = (1 << np.arange(N_IND))
bool_matrix = recs[IND_NAMES].values.astype(int)
day_mask = (bool_matrix * weights).sum(axis=1)
ret_blind = recs["ret_blind"].values
ret_trail = recs["ret_trail"].values

results = []
for size in range(1, N_IND + 1):
    for combo in combinations(range(N_IND), size):
        sub_mask = sum(1 << i for i in combo)
        match = (day_mask & sub_mask) == sub_mask
        n = int(match.sum())
        if n == 0:
            continue
        rb = ret_blind[match]
        rt = ret_trail[match]
        results.append({
            "indicators": " & ".join(IND_NAMES[i] for i in combo),
            "n_indicators": size,
            "n_trades": n,
            "win_rate_blind": float((rb > 0).mean() * 100),
            "avg_return_blind": float(rb.mean()),
            "median_return_blind": float(np.median(rb)),
            "win_rate_trail": float((rt > 0).mean() * 100),
            "avg_return_trail": float(rt.mean()),
            "median_return_trail": float(np.median(rt)),
            "std_return_trail": float(rt.std()) if n > 1 else 0.0,
        })

df_out = pd.DataFrame(results)
df_out = df_out.sort_values(["avg_return_trail", "win_rate_trail"], ascending=[False, False]).reset_index(drop=True)
df_out.insert(0, "rank", np.arange(1, len(df_out) + 1))
out_path = HERE / "indicator_combo_trailstop_results.csv"
df_out.to_csv(out_path, index=False)
print(f"\nSaved {len(df_out)} combo rows to {out_path}")

print("\n=== TOP 20 by TRAIL-STOP avg return (n>=30) ===")
d30 = df_out[df_out.n_trades >= 30]
print(d30.head(20)[["rank", "indicators", "n_trades", "win_rate_trail", "avg_return_trail",
                     "win_rate_blind", "avg_return_blind"]].to_string(index=False))

print("\n=== single-indicator results, blind vs trail-stop ===")
singles = df_out[df_out.n_indicators == 1].sort_values("avg_return_trail", ascending=False)
print(singles[["indicators", "n_trades", "win_rate_blind", "avg_return_blind",
                "win_rate_trail", "avg_return_trail"]].to_string(index=False))
