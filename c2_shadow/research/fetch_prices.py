"""Download daily OHLCV (Yahoo chart API, adjusted) for every symbol AELong traded, plus SPY.
Cached in data/prices/<SYM>.json; re-running only fetches missing symbols."""
import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "prices"
OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}
YAHOO_ALIAS = {"EADSY": "EADSY", "BRK.B": "BRK-B"}


def fetch(sym, start="2021-06-01"):
    p1 = int(time.mktime(time.strptime(start, "%Y-%m-%d")))
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{YAHOO_ALIAS.get(sym, sym)}"
           f"?period1={p1}&period2={int(time.time())}&interval=1d&events=div%2Csplit&includeAdjustedClose=true")
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    bars = []
    for i, t in enumerate(res["timestamp"]):
        if q["close"][i] is None or adj[i] is None:
            continue
        f = adj[i] / q["close"][i]  # scale OHLC to adjusted
        bars.append({"d": time.strftime("%Y-%m-%d", time.gmtime(t)), "o": q["open"][i] * f, "h": q["high"][i] * f,
                     "l": q["low"][i] * f, "c": adj[i], "raw_c": q["close"][i], "v": q["volume"][i] or 0})
    return bars


if __name__ == "__main__":
    syms = sorted({p["sym"] for p in json.load(open(HERE / "data" / "aelong_positions.json"))} | {"SPY"} | set(sys.argv[1:]))
    failed = []
    for s in syms:
        f = OUT / f"{s}.json"
        if f.exists():
            continue
        try:
            bars = fetch(s)
            f.write_text(json.dumps(bars))
            print(f"{s}: {len(bars)} bars")
        except Exception as e:
            failed.append(s)
            print(f"{s}: FAILED {e}")
        time.sleep(0.7)
    print("failed:", failed)
