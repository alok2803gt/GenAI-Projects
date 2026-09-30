"""Download the 410 S&P 500 names NOT in the explored 112-ticker panel.

Holdout for PREREG_daytrader_revival.md (sha256 3071a656...). Downloaded AFTER
the pre-registration was written and hashed.
"""
import json, pickle, time
from pathlib import Path
import yfinance as yf

HERE = Path(__file__).parent
BACK = HERE.parent
OUT = HERE / "holdout_universe_5y.pkl"

seen = set(pickle.load(open(BACK / "breakout_research" / "universe_5y_ohlcv.pkl", "rb")))
uni = json.load(open(BACK / "sp500_universe_cache.json"))
# The cache is a dict {tickers: [...], fetched_at, sectors} -- list(dict) would
# yield its KEYS ('tickers', 'fetched_at', 'sectors'), which is what the first
# run actually tried to download.
if isinstance(uni, dict):
    uni = uni.get("tickers") or uni.get("universe") or []
todo = sorted(set(uni) - seen)
print(f"downloading {len(todo)} unseen tickers (5y daily)")

out, failed = {}, []
for i in range(0, len(todo), 40):
    batch = todo[i:i + 40]
    try:
        data = yf.download(batch, period="5y", interval="1d", group_by="ticker",
                           auto_adjust=False, progress=False, threads=True)
    except Exception as exc:
        print(f"  batch {i//40} failed: {exc}")
        failed += batch
        continue
    for t in batch:
        try:
            df = data[t].dropna(subset=["Open", "High", "Low", "Close"])
            if len(df) >= 300:
                out[t] = df[["Open", "High", "Low", "Close", "Volume"]]
            else:
                failed.append(t)
        except Exception:
            failed.append(t)
    print(f"  {i+len(batch)}/{len(todo)} -> kept {len(out)}")
    time.sleep(2)

pickle.dump(out, open(OUT, "wb"))
print(f"\nsaved {len(out)} tickers to {OUT.name}; {len(failed)} unusable")
lens = [len(v) for v in out.values()]
if lens:
    print(f"bars/ticker: min {min(lens)} median {sorted(lens)[len(lens)//2]} max {max(lens)}")
    print(f"total ticker-days: {sum(lens):,}")
