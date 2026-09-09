"""
Follow-up to combo_20dma_filter_test.py: CEO asked about a 20-period EMA
check instead of a 20-day SMA, since EMA20 tracks price more tightly/faster
than SMA20 (EMA20's smoothing constant, alpha=2/21, is barely different
from the combo's own EMA21 -- so this could plausibly be LESS redundant
with the existing EMA9>EMA21 condition than SMA20 turned out to be, or it
could turn out just as redundant since EMA20 and EMA21 are so close to
each other). Testing empirically rather than assuming either way, same
real-outcome-reuse method as the SMA20 test: recompute close>EMA20 from
the same universe_5y_ohlcv.pkl history and join onto the real, already
Alpaca-minute-bar-simulated 7,735 signal-day outcomes.
"""
import pickle
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent

with open(BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl", "rb") as f:
    daily_data = pickle.load(f)

recs = pd.read_csv(HERE / "target_combo_multiregime_rows.csv", parse_dates=["day"])
print(f"Loaded {len(recs)} real signal-day outcomes from the existing multi-regime backtest.")

above_20ema = {}
for ticker, hist in daily_data.items():
    if ticker not in recs["ticker"].unique():
        continue
    c = hist["Close"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    flag = (c.shift(1) > ema20.shift(1))  # same t-1-close-only basis as every other combo indicator
    above_20ema[ticker] = flag

def _lookup(row):
    flag_series = above_20ema.get(row["ticker"])
    if flag_series is None:
        return None
    day = pd.Timestamp(row["day"]).tz_localize(flag_series.index.tz)
    if day not in flag_series.index:
        return None
    val = flag_series.loc[day]
    return bool(val) if pd.notna(val) else None

recs["above_20ema"] = recs.apply(_lookup, axis=1)
n_unresolved = recs["above_20ema"].isna().sum()
if n_unresolved:
    print(f"WARNING: {n_unresolved}/{len(recs)} rows could not be matched to a real EMA20 value (excluded).")
recs = recs.dropna(subset=["above_20ema"])


def _agg(df, label):
    if df.empty:
        print(f"{label:30s}  n=    0")
        return
    rt = df["ret_trail"]
    print(f"{label:30s}  n={len(df):5d}  win={(rt>0).mean()*100:5.1f}%  avg={rt.mean():+.4f}%  "
          f"median={rt.median():+.4f}%  std={rt.std():.4f}%")


print("\n=== FULL PERIOD (2021-08 to 2026-08, real trailing-stop outcomes) ===")
_agg(recs, "ALL (no 20EMA filter, baseline)")
_agg(recs[recs["above_20ema"] == True], "WITH 20EMA filter (close>EMA20)")
_agg(recs[recs["above_20ema"] == False], "Filtered OUT (close<=EMA20)")

print(f"\nSignal retention if 20EMA filter added: {(recs['above_20ema']==True).sum()}/{len(recs)} "
      f"({(recs['above_20ema']==True).mean()*100:.1f}%)")

print("\n=== BY YEAR ===")
for yr, g in recs.groupby("year"):
    with_f = g[g["above_20ema"] == True]["ret_trail"]
    without_f = g[g["above_20ema"] == False]["ret_trail"]
    def _fmt(s):
        return (f"n={len(s):4d} win={(s>0).mean()*100:5.1f}% avg={s.mean():+.3f}%" if len(s)
                else f"n={len(s):4d} win=  n/a  avg=  n/a")
    print(f"  {yr}: WITH filter  {_fmt(with_f)}   |   FILTERED OUT {_fmt(without_f)}")

out_path = HERE / "combo_20ema_filter_results.csv"
recs.to_csv(out_path, index=False)
print(f"\nWrote {out_path}")
