"""
CEO question (2026-09-08): would re-checking the dispersion-combo gate
intraday (not just once at EOD using yesterday's close) catch real,
profitable signal days the current system currently skips?

Context: daytrader_scanner.py's dispersion gate (DISPERSION_COMBO_DISP_PCTILE_MIN
= 0.75) is entirely EOD-anchored -- cross-sectional std of daily returns,
smoothed by a 20-day MA, ranked against a trailing 252-day window, using
data only through YESTERDAY's close (today's still-forming bar excluded,
see daytrader_scanner.py:1115). The code's own comment (2026-09-01) already
argues re-running this same check later in the day would recompute the
IDENTICAL result, since none of its inputs use today's price action at
all. This script tests that claim empirically instead of taking it on
faith: does TODAY's own realized dispersion carry information the
EOD-anchored measure misses?

Method: reuses target_combo_multiregime_rows.csv's real signal-day
outcomes (7,735 real Alpaca-minute-bar entries, real 0.3% trailing stop,
2021-09 to 2026-08 -- generated WITHOUT the dispersion filter applied, so
every signal day is present regardless of what dispersion would have
said). For each day:
  1. Recompute the CURRENT EOD dispersion percentile exactly as
     daytrader_scanner.py does (252-day rank of a 20-day MA of
     cross-sectional daily-return std, using data through yesterday).
  2. Compute a SAME-DAY REALIZED dispersion percentile: cross-sectional
     std of that day's own (close/prior_close - 1) returns, ranked
     against the same trailing 252-day window of daily (not smoothed)
     cross-sectional dispersion. This is an honest proxy for "what
     dispersion looked like by the close of the day in question" -- it is
     NOT a true partial-day intraday read (that would need real intraday
     data this account doesn't have cached for the full 112-ticker
     universe over 5 years), so it is a best-case upper bound on what an
     intraday re-check could see, not an exact simulation of one. Stated
     plainly, not glossed over.
  3. Bucket real signal days by (EOD gate say fire?, same-day dispersion
     high?) and compare real ret_trail outcomes across buckets -- the
     question that actually matters: among days the EOD gate currently
     SKIPS, do the ones where same-day dispersion was ALSO high look
     meaningfully different (better) than the ones where it was low too?
     If not, there's nothing for an intraday check to find.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent

DISP_MA_WINDOW = 20
DISP_RANK_WINDOW = 252
DISP_PCTILE_MIN = 0.75

with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
    daily_data = pickle.load(f)
print(f"Loaded {len(daily_data)} tickers of daily data")

recs = pd.read_csv(HERE / "target_combo_multiregime_rows.csv", parse_dates=["day"])
print(f"Loaded {len(recs)} real signal-day outcomes")

# ---------- Build the cross-sectional daily-return dispersion series (once, shared) ----------
closes = {}
for ticker, hist in daily_data.items():
    s = hist["Close"]
    if s.index.tz is not None:
        s = s.tz_localize(None)  # match target_combo_multiregime_rows.csv's naive "day" column
    closes[ticker] = s
wide = pd.DataFrame(closes).sort_index()
daily_ret = wide.pct_change()
cross_std = daily_ret.std(axis=1, skipna=True)  # one value per real trading day: today's cross-sectional dispersion

# CURRENT (EOD-anchored) methodology: smooth first, then rank, using T-1 data for day T
disp_ma20 = cross_std.rolling(DISP_MA_WINDOW, min_periods=15).mean()
eod_disp_pctile = disp_ma20.rolling(DISP_RANK_WINDOW, min_periods=60).rank(pct=True).shift(1)  # shift(1): as of yesterday's close, for "today"

# SAME-DAY REALIZED methodology: no smoothing, rank today's own cross_std directly (no shift -- uses today's own close)
sameday_disp_pctile = cross_std.rolling(DISP_RANK_WINDOW, min_periods=60).rank(pct=True)

print(f"EOD dispersion series: {eod_disp_pctile.notna().sum()} valid days")
print(f"Same-day dispersion series: {sameday_disp_pctile.notna().sum()} valid days")

# ---------- Join onto the real signal-day outcomes ----------
recs["eod_disp_pctile"] = recs["day"].map(eod_disp_pctile)
recs["sameday_disp_pctile"] = recs["day"].map(sameday_disp_pctile)
before = len(recs)
recs = recs.dropna(subset=["eod_disp_pctile", "sameday_disp_pctile"])
print(f"Matched {len(recs)}/{before} signal-days to both dispersion series")

recs["eod_fires"] = recs["eod_disp_pctile"] >= DISP_PCTILE_MIN
recs["sameday_high"] = recs["sameday_disp_pctile"] >= DISP_PCTILE_MIN


def _agg(df, label):
    if df.empty:
        print(f"{label:45s}  n=0")
        return
    rt = df["ret_trail"]
    print(f"{label:45s}  n={len(df):5d}  win={(rt>0).mean()*100:5.1f}%  "
          f"avg={rt.mean():+.4f}%  median={rt.median():+.4f}%  std={rt.std():.4f}%")


print("\n=== Baseline: all real signal-days, no dispersion filter at all ===")
_agg(recs, "ALL")

print("\n=== Current live behavior: EOD gate fires vs skips ===")
_agg(recs[recs["eod_fires"]], "EOD gate FIRES (what's live today)")
_agg(recs[~recs["eod_fires"]], "EOD gate SKIPS (what's being asked about)")

print("\n=== The real question: among days EOD gate SKIPS, does same-day dispersion differentiate? ===")
skipped = recs[~recs["eod_fires"]]
_agg(skipped[skipped["sameday_high"]], "EOD skips, but SAME-DAY dispersion was high")
_agg(skipped[~skipped["sameday_high"]], "EOD skips, same-day dispersion also low")

print("\n=== By year (EOD-skip-but-sameday-high bucket only, checking regime consistency) ===")
sub = skipped[skipped["sameday_high"]]
for yr, g in sub.groupby("year"):
    rt = g["ret_trail"]
    print(f"  {yr}: n={len(g):4d}  win={(rt>0).mean()*100 if len(g) else float('nan'):5.1f}%  avg={rt.mean():+.4f}%")

n_recoverable = len(skipped[skipped["sameday_high"]])
pct_recoverable = n_recoverable / len(skipped) * 100 if len(skipped) else 0
print(f"\nOf {len(skipped)} EOD-skipped signal-days, {n_recoverable} ({pct_recoverable:.1f}%) "
      f"had high same-day dispersion -- these are the ones an intraday re-check could theoretically have caught.")
