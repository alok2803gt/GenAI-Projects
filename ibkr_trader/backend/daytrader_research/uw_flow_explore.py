"""EXPLORATION phase for PREREG_uw_flow_direction.md (sha256 67c5d03a...).

Runs ONLY on sessions before 2025-09-16. Picks the flow signal's functional
form (which field, which window). The holdout (2025-09-16 onward) is not
touched here -- uw_flow_holdout.py does that once, afterwards.

Entry is the CLOSE of the opening window, taken from the 5m candles, which is
a real tradeable price. Exit is the session close. Long only.

    ../../venv/bin/python daytrader_research/uw_flow_explore.py
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "uw_flow_cache"
ET = ZoneInfo("America/New_York")
SPLIT = "2025-09-16"
FEE_PP = 0.70 / 148.0 * 100

WINDOWS = {"9:30-9:35": 5, "9:30-9:45": 15, "9:30-10:00": 30}
FIELDS = {
    "net_delta":      lambda t: float(t.get("net_delta") or 0),
    "net_prem":       lambda t: float(t.get("net_call_premium") or 0) - float(t.get("net_put_premium") or 0),
    "ask_bid_imbal":  lambda t: ((t.get("call_volume_ask_side") or 0) - (t.get("call_volume_bid_side") or 0))
                                - ((t.get("put_volume_ask_side") or 0) - (t.get("put_volume_bid_side") or 0)),
}


def _et(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ET)


def load() -> pd.DataFrame:
    rows = []
    for f in CACHE.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("error") or not d.get("ticks") or not d.get("ohlc"):
            continue
        ticks = d["ticks"]
        # per-minute flow, ascending by tape_time, restricted to RTH
        tf = []
        for t in ticks:
            tt = t.get("tape_time")
            if not tt:
                continue
            e = _et(tt)
            mins = (e.hour - 9) * 60 + (e.minute - 30)
            if 0 <= mins < 390:
                tf.append((mins, t))
        if not tf:
            continue

        # 5m candles -> entry price at window close, and the session close
        cands = []
        for c in d["ohlc"]:
            st = c.get("start_time")
            if not st:
                continue
            e = _et(st)
            mins = (e.hour - 9) * 60 + (e.minute - 30)
            if 0 <= mins < 390:
                cands.append((mins, float(c["close"]), float(c["open"])))
        if len(cands) < 20:
            continue
        cands.sort()
        close_px = cands[-1][1]

        rec = {"ticker": d["ticker"], "date": pd.Timestamp(d["date"])}
        ok = True
        for wname, wmin in WINDOWS.items():
            # entry = close of the last 5m candle that ENDS at or before wmin
            elig = [c for c in cands if c[0] + 5 <= wmin]
            if not elig:
                ok = False
                break
            rec[f"entry_{wname}"] = elig[-1][1]
            rec[f"ret_{wname}"] = (close_px - elig[-1][1]) / elig[-1][1] * 100
            for fname, fn in FIELDS.items():
                rec[f"{fname}_{wname}"] = sum(fn(t) for m, t in tf if m < wmin)
        if ok:
            rows.append(rec)
    return pd.DataFrame(rows)


def clustered(s: pd.DataFrame, col: str) -> tuple:
    if len(s) < 2:
        return float("nan"), float("nan"), 0
    daily = s.groupby("date")[col].mean()
    if len(daily) < 2:
        return daily.mean(), float("nan"), len(daily)
    return daily.mean(), daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))), len(daily)


def main() -> None:
    r = load()
    if r.empty:
        print("no usable cached records yet")
        return
    r = r.sort_values("date")
    train = r[r["date"] < SPLIT]
    print(f"cache: {len(r):,} ticker-days, {r.date.nunique()} sessions "
          f"({r.date.min().date()} .. {r.date.max().date()})")
    print(f"EXPLORATION set (< {SPLIT}): {len(train):,} rows, {train.date.nunique()} sessions")
    print(f"holdout reserved      (>= {SPLIT}): {len(r) - len(train):,} rows "
          f"-- NOT touched here\n")
    if train.empty:
        return

    print("Entry = close of the opening window (real 5m candle). Exit = session close.")
    print("excess = vs the equal-weighted mean of that session's candidates.")
    print(f"fee hurdle at $148 = {FEE_PP:.3f}pp\n")
    print(f"  {'signal':<18}{'window':<13}{'rows':>7}{'days':>6}"
          f"{'Q5-Q1':>9}{'t':>7}{'Q5 excess':>11}{'t(Q5)':>8}")
    best = None
    for wname in WINDOWS:
        sub = train.dropna(subset=[f"ret_{wname}"]).copy()
        if sub.empty:
            continue
        sub["_ex"] = sub[f"ret_{wname}"] - sub.groupby("date")[f"ret_{wname}"].transform("mean")
        for fname in FIELDS:
            col = f"{fname}_{wname}"
            s = sub.dropna(subset=[col]).copy()
            if len(s) < 200:
                continue
            s["q"] = s.groupby("date")[col].transform(
                lambda x: pd.qcut(x, 3, labels=False, duplicates="drop") if x.notna().sum() >= 3 else np.nan)
            s = s.dropna(subset=["q"])
            hi, lo = s[s.q == s.q.max()], s[s.q == 0]
            a = hi.groupby("date")["_ex"].mean()
            b = lo.groupby("date")["_ex"].mean()
            pair = pd.concat([a.rename("hi"), b.rename("lo")], axis=1, sort=False).dropna()
            if len(pair) < 20:
                continue
            dd = pair["hi"] - pair["lo"]
            td = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd)))
            m5, t5, _ = clustered(hi, "_ex")
            print(f"  {fname:<18}{wname:<13}{len(s):>7}{s.date.nunique():>6}"
                  f"{dd.mean():>9.3f}{td:>7.2f}{m5:>11.3f}{t5:>8.2f}")
            if best is None or abs(td) > abs(best[2]):
                best = (fname, wname, td, dd.mean(), m5, t5)
    if best:
        print(f"\n  strongest on EXPLORATION: {best[0]} over {best[1]} "
              f"(Q5-Q1 {best[3]:+.3f}pp, t={best[2]:+.2f}; top-tercile excess "
              f"{best[4]:+.3f}pp, t={best[5]:+.2f})")
        print("  -> freeze this form, write it to the research log, THEN run the holdout once.")


if __name__ == "__main__":
    main()
