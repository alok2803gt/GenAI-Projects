"""
CEO question (2026-09-06): would adding a 20-day moving average check
benefit the dispersion-combo strategy (VolRatio>1.2 & CCI>100 &
Williams%R>-50 & EMA9>EMA21, gated by the 252-day dispersion percentile
just restored in daytrader_scanner.py)?

Reuses the REAL, already-computed outcomes from
target_combo_multiregime_backtest.py's own output
(target_combo_multiregime_rows.csv -- 7,735 real signal-days, each with a
real Alpaca-minute-bar entry/exit and the account's real 0.3% trailing
stop already simulated, 2021-08 to 2026-08) instead of re-pulling minute
data. A 20DMA check (close > SMA20) can only ever SHRINK the existing
signal set -- it's a pure additional AND condition, never adds a new
signal day -- so this is a clean subset comparison: recompute close>SMA20
per ticker/day from the same universe_5y_ohlcv.pkl used to build the
original combo, join it onto the real outcomes, and compare the days that
would survive the filter against the days that would be dropped.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent

with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
    daily_data = pickle.load(f)

recs = pd.read_csv(HERE / "target_combo_multiregime_rows.csv", parse_dates=["day"])
print(f"Loaded {len(recs)} real signal-day outcomes from the existing multi-regime backtest.")

# Recompute close>SMA20 (as of the signal day itself, matching the account's own
# t-1-close-only signal convention -- SMA20 uses closes through and including
# the prior real trading day, same lag structure as the other combo indicators)
above_20dma = {}
for ticker, hist in daily_data.items():
    if ticker not in recs["ticker"].unique():
        continue
    c = hist["Close"]
    sma20 = c.rolling(20).mean()
    flag = (c.shift(1) > sma20.shift(1))  # shift(1): using the same prior-close basis as the combo signal itself
    above_20dma[ticker] = flag

def _lookup(row):
    flag_series = above_20dma.get(row["ticker"])
    if flag_series is None:
        return None
    day = pd.Timestamp(row["day"]).tz_localize(flag_series.index.tz)
    if day not in flag_series.index:
        return None
    val = flag_series.loc[day]
    return bool(val) if pd.notna(val) else None

recs["above_20dma"] = recs.apply(_lookup, axis=1)
n_unresolved = recs["above_20dma"].isna().sum()
if n_unresolved:
    print(f"WARNING: {n_unresolved}/{len(recs)} rows could not be matched to a real SMA20 value (excluded).")
recs = recs.dropna(subset=["above_20dma"])


def _agg(df, label):
    rt = df["ret_trail"]
    print(f"{label:30s}  n={len(df):5d}  win={(rt>0).mean()*100:5.1f}%  avg={rt.mean():+.4f}%  "
          f"median={rt.median():+.4f}%  std={rt.std():.4f}%")


print("\n=== FULL PERIOD (2021-08 to 2026-08, real trailing-stop outcomes) ===")
_agg(recs, "ALL (no 20DMA filter, baseline)")
_agg(recs[recs["above_20dma"] == True], "WITH 20DMA filter (close>SMA20)")
_agg(recs[recs["above_20dma"] == False], "Filtered OUT (close<=SMA20)")

print(f"\nSignal retention if 20DMA filter added: {(recs['above_20dma']==True).sum()}/{len(recs)} "
      f"({(recs['above_20dma']==True).mean()*100:.1f}%)")

print("\n=== BY YEAR ===")
for yr, g in recs.groupby("year"):
    with_f = g[g["above_20dma"] == True]["ret_trail"]
    without_f = g[g["above_20dma"] == False]["ret_trail"]
    print(f"  {yr}: WITH filter  n={len(with_f):4d} win={(with_f>0).mean()*100 if len(with_f) else float('nan'):5.1f}% "
          f"avg={with_f.mean() if len(with_f) else float('nan'):+.3f}%   |   "
          f"FILTERED OUT n={len(without_f):4d} win={(without_f>0).mean()*100 if len(without_f) else float('nan'):5.1f}% "
          f"avg={without_f.mean() if len(without_f) else float('nan'):+.3f}%")

out_path = HERE / "combo_20dma_filter_results.csv"
recs.to_csv(out_path, index=False)
print(f"\nWrote {out_path}")
