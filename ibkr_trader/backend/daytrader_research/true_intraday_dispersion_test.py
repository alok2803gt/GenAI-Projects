"""
The real test: does dispersion computed from ONLY what you'd actually
know by 10:30am ET (prior close -> 10:30am move, real Alpaca 1-min bars,
no full-day lookahead) still differentiate the EOD-gate-skipped signal
days the way intraday_dispersion_test.py's optimistic full-day proxy
suggested (23.7% of skipped days recoverable, +0.119% vs +0.068% avg
return)?

Method: for each ticker/day, take the last real 1-min bar at or before
10:30 ET (am_window_cache/, fetch_am_window_bars.py) as "the 10:30am
price," compute its return from the PRIOR trading day's real daily close
(universe_5y_ohlcv.pkl), then the cross-sectional std of that return
across the full 112-ticker universe = today's TRUE partial-day
dispersion reading. Ranked against its own trailing 252-day window (same
structure as the EOD/full-day versions, for a fair three-way comparison),
joined onto the same real target_combo_multiregime_rows.csv outcomes.

This has no lookahead: every input existed and was knowable by 10:30am
ET on the day in question.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
AM_CACHE = HERE / "am_window_cache"

DISP_RANK_WINDOW = 252
DISP_PCTILE_MIN = 0.75

with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
    daily_data = pickle.load(f)
print(f"Loaded {len(daily_data)} tickers of daily data")

recs = pd.read_csv(HERE / "target_combo_multiregime_rows.csv", parse_dates=["day"])
print(f"Loaded {len(recs)} real signal-day outcomes")

# ---------- Prior-day close per ticker (naive dates, matching the CSV) ----------
prior_close = {}
for ticker, hist in daily_data.items():
    c = hist["Close"]
    if c.index.tz is not None:
        c = c.tz_localize(None)
    prior_close[ticker] = c

# ---------- 10:30am price per ticker/day from the real AM-window bars ----------
price_1030 = {}
n_loaded = 0
for ticker in sorted(daily_data.keys()):
    path = AM_CACHE / f"{ticker}.pkl"
    if not path.exists():
        continue
    df = pd.read_pickle(path)
    if df.empty:
        continue
    df = df.tz_convert(None) if df.index.tz is not None else df
    df["date"] = df.index.normalize()
    last_by_day = df.groupby("date")["close"].last()  # last real print at/before 10:30 ET each day
    price_1030[ticker] = last_by_day
    n_loaded += 1
print(f"Loaded 10:30am prices for {n_loaded}/{len(daily_data)} tickers")

# ---------- Cross-sectional TRUE partial-day return (prior close -> 10:30am) ----------
ret_1030 = {}
for ticker, p1030 in price_1030.items():
    pc = prior_close.get(ticker)
    if pc is None:
        continue
    pc_shifted = pc.shift(1)  # prior trading day's close, aligned to today's date
    common = p1030.index.intersection(pc_shifted.index)
    ret_1030[ticker] = (p1030.loc[common] / pc_shifted.loc[common] - 1)

wide_1030 = pd.DataFrame(ret_1030).sort_index()
cross_std_1030 = wide_1030.std(axis=1, skipna=True)
true_disp_pctile = cross_std_1030.rolling(DISP_RANK_WINDOW, min_periods=60).rank(pct=True)
print(f"True 10:30am dispersion series: {true_disp_pctile.notna().sum()} valid days")

# ---------- Also recompute the EOD (current live) series for the same 3-way comparison ----------
closes = {}
for ticker, hist in daily_data.items():
    s = hist["Close"]
    if s.index.tz is not None:
        s = s.tz_localize(None)
    closes[ticker] = s
wide_daily = pd.DataFrame(closes).sort_index()
daily_ret = wide_daily.pct_change()
cross_std_daily = daily_ret.std(axis=1, skipna=True)
disp_ma20 = cross_std_daily.rolling(20, min_periods=15).mean()
eod_disp_pctile = disp_ma20.rolling(DISP_RANK_WINDOW, min_periods=60).rank(pct=True).shift(1)

# ---------- Join onto real outcomes ----------
recs["eod_disp_pctile"] = recs["day"].map(eod_disp_pctile)
recs["true1030_disp_pctile"] = recs["day"].map(true_disp_pctile)
before = len(recs)
recs = recs.dropna(subset=["eod_disp_pctile", "true1030_disp_pctile"])
print(f"Matched {len(recs)}/{before} signal-days to both series")

recs["eod_fires"] = recs["eod_disp_pctile"] >= DISP_PCTILE_MIN
recs["true1030_high"] = recs["true1030_disp_pctile"] >= DISP_PCTILE_MIN


def _agg(df, label):
    if df.empty:
        print(f"{label:50s}  n=0")
        return
    rt = df["ret_trail"]
    print(f"{label:50s}  n={len(df):5d}  win={(rt>0).mean()*100:5.1f}%  "
          f"avg={rt.mean():+.4f}%  median={rt.median():+.4f}%  std={rt.std():.4f}%")


print("\n=== Baseline: all real signal-days matched to both series ===")
_agg(recs, "ALL")

print("\n=== Current live behavior: EOD gate fires vs skips ===")
_agg(recs[recs["eod_fires"]], "EOD gate FIRES (what's live today)")
_agg(recs[~recs["eod_fires"]], "EOD gate SKIPS")

print("\n=== THE REAL TEST: among EOD-skipped days, does TRUE 10:30am dispersion (no lookahead) differentiate? ===")
skipped = recs[~recs["eod_fires"]]
_agg(skipped[skipped["true1030_high"]], "EOD skips, TRUE 10:30am dispersion was high")
_agg(skipped[~skipped["true1030_high"]], "EOD skips, 10:30am dispersion also low")

print("\n=== By year (EOD-skip-but-1030-high bucket, checking regime consistency) ===")
sub = skipped[skipped["true1030_high"]]
for yr, g in sub.groupby("year"):
    rt = g["ret_trail"]
    print(f"  {yr}: n={len(g):4d}  win={(rt>0).mean()*100 if len(g) else float('nan'):5.1f}%  avg={rt.mean():+.4f}%")

n_recoverable = len(skipped[skipped["true1030_high"]])
pct_recoverable = n_recoverable / len(skipped) * 100 if len(skipped) else 0
print(f"\nOf {len(skipped)} EOD-skipped signal-days, {n_recoverable} ({pct_recoverable:.1f}%) "
      f"had high TRUE 10:30am dispersion -- these are the ones a real 10:30am intraday "
      f"re-check could actually have caught, with no lookahead.")
