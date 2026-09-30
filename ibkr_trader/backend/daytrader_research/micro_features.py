"""Phase 1 of the exhaustive direction search: INTRADAY MICROSTRUCTURE.

Everything tested so far was either known before the open (daily bars) or
aggregated options flow, and both failed. This tests the one category that is
genuinely NEW information after the bell: how price itself behaved in the first
minutes of the session.

Built from the 5-minute bars already cached in uw_flow_cache (no new API calls).
Three bars exist before 09:45, which is enough for opening-range, direction-
persistence and volume-tilt features.

DISCIPLINE (unchanged from PREREG_uw_flow_direction.md, sha256 67c5d03a...):
  * exploration set = sessions BEFORE 2025-09-16. All searching happens here.
  * holdout = 2025-09-16 onward, still SEALED. Only a survivor gets tested there,
    once.
  * entry is the CLOSE of the 09:45 bar (a real tradeable price), exit is the
    session close, long only.
  * market-adjusted (vs the same session's candidate mean) and date-clustered.
  * every feature's result is printed, including the failures.

    ../../venv/bin/python daytrader_research/micro_features.py
"""
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "uw_flow_cache"
ET = ZoneInfo("America/New_York")
SPLIT = "2025-09-16"
FEE_PP = 0.70 / 148.0 * 100
WIN = 15          # minutes of opening behaviour used to build features


def _et(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ET)


def load() -> pd.DataFrame:
    rows = []
    for f in CACHE.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("error") or not d.get("ohlc"):
            continue
        bars = []
        for c in d["ohlc"]:
            st = c.get("start_time")
            if not st:
                continue
            e = _et(st)
            m = (e.hour - 9) * 60 + (e.minute - 30)
            if 0 <= m < 390:
                bars.append((m, float(c["open"]), float(c["high"]),
                             float(c["low"]), float(c["close"]), float(c.get("volume") or 0)))
        bars.sort()
        if len(bars) < 20:
            continue
        opening = [b for b in bars if b[0] < WIN]
        if len(opening) < 2:
            continue
        entry_bar = opening[-1]
        entry = entry_bar[4]
        close_px = bars[-1][4]
        if not entry:
            continue

        o0 = opening[0][1]
        hi = max(b[2] for b in opening)
        lo = min(b[3] for b in opening)
        rng = hi - lo
        vols = [b[5] for b in opening]
        up_vol = sum(b[5] for b in opening if b[4] >= b[1])
        dn_vol = sum(b[5] for b in opening if b[4] < b[1])
        vwap = (sum((b[2] + b[3] + b[4]) / 3 * b[5] for b in opening) / sum(vols)
                if sum(vols) else np.nan)
        rest_vol = sum(b[5] for b in bars if b[0] >= WIN)

        rec = {
            "ticker": d["ticker"], "date": pd.Timestamp(d["date"]),
            "entry": entry,
            "fwd": (close_px - entry) / entry * 100,        # what we are predicting
            # --- microstructure features, all knowable at 09:45 ---
            "or_pos": (entry - lo) / rng * 100 if rng else np.nan,   # position in opening range
            "or_width_pct": rng / o0 * 100 if o0 else np.nan,
            "first5_ret": (opening[0][4] - opening[0][1]) / opening[0][1] * 100,
            "open15_ret": (entry - o0) / o0 * 100,
            "accel": ((entry - opening[0][4]) / opening[0][4] * 100
                      if len(opening) > 1 and opening[0][4] else np.nan),
            "up_bar_frac": sum(1 for b in opening if b[4] >= b[1]) / len(opening) * 100,
            "vol_tilt": ((up_vol - dn_vol) / (up_vol + dn_vol) * 100
                         if (up_vol + dn_vol) else np.nan),
            "vs_vwap": (entry - vwap) / vwap * 100 if vwap == vwap and vwap else np.nan,
            "close_in_bar": ((entry_bar[4] - entry_bar[3]) / (entry_bar[2] - entry_bar[3]) * 100
                             if entry_bar[2] > entry_bar[3] else np.nan),
            "hh_seq": sum(1 for i in range(1, len(opening))
                          if opening[i][2] > opening[i - 1][2]) / max(len(opening) - 1, 1) * 100,
            "vol_front_load": (sum(vols) / rest_vol * 100 if rest_vol else np.nan),
        }
        rows.append(rec)
    r = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    r["excess"] = r["fwd"] - r.groupby("date")["fwd"].transform("mean")
    return r


FEATURES = ["or_pos", "or_width_pct", "first5_ret", "open15_ret", "accel",
            "up_bar_frac", "vol_tilt", "vs_vwap", "close_in_bar", "hh_seq",
            "vol_front_load"]


def clustered(s: pd.DataFrame, col: str = "excess") -> tuple:
    if len(s) < 2:
        return float("nan"), float("nan"), 0
    d = s.groupby("date")[col].mean()
    if len(d) < 2:
        return d.mean(), float("nan"), len(d)
    return d.mean(), d.mean() / (d.std(ddof=1) / math.sqrt(len(d))), len(d)


def main() -> None:
    r = load()
    train = r[r["date"] < SPLIT]
    print(f"cache: {len(r):,} ticker-days  |  EXPLORATION (<{SPLIT}): {len(train):,} rows, "
          f"{train.date.nunique()} sessions  |  holdout sealed: {len(r) - len(train):,}")
    print(f"entry = close of the 09:{30 + WIN} bar, exit = session close, long only")
    print(f"fee hurdle at $148 (Tiered) = {FEE_PP:.3f}pp\n")

    print(f"  {'feature':<18}{'Q1(low)':>10}{'Q3(high)':>10}{'Q3-Q1':>9}"
          f"{'t':>7}{'Q3 excess':>11}{'t(Q3)':>8}{'net':>9}")
    res = {}
    for f in FEATURES:
        s = train.dropna(subset=[f]).copy()
        if len(s) < 200:
            print(f"  {f:<18}  too few rows ({len(s)})")
            continue
        s["q"] = s.groupby("date")[f].transform(
            lambda x: pd.qcut(x, 3, labels=False, duplicates="drop")
            if x.notna().sum() >= 3 else np.nan)
        s = s.dropna(subset=["q"])
        hi, lo = s[s.q == s.q.max()], s[s.q == 0]
        a = hi.groupby("date")["excess"].mean()
        b = lo.groupby("date")["excess"].mean()
        pair = pd.concat([a.rename("hi"), b.rename("lo")], axis=1, sort=False).dropna()
        if len(pair) < 20:
            continue
        dd = pair["hi"] - pair["lo"]
        td = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd)))
        m3, t3, _ = clustered(hi)
        m1, _, _ = clustered(lo)
        res[f] = (dd.mean(), td, m3, t3)
        print(f"  {f:<18}{m1:>10.3f}{m3:>10.3f}{dd.mean():>9.3f}{td:>7.2f}"
              f"{m3:>11.3f}{t3:>8.2f}{m3 - FEE_PP:>9.3f}")

    if res:
        best = max(res.items(), key=lambda kv: abs(kv[1][1]))
        print(f"\n  strongest |t| on the spread: {best[0]} "
              f"({best[1][0]:+.3f}pp, t={best[1][1]:+.2f})")
        passing = {k: v for k, v in res.items() if v[2] > FEE_PP and v[3] > 2}
        print(f"  features whose TOP TERCILE clears the fee with t>2: "
              f"{list(passing) if passing else 'NONE'}")
        print(f"\n  Bonferroni bar for {len(res)} features: |t| > "
              f"{__import__('statistics').NormalDist().inv_cdf(1 - 0.05 / (2 * len(res))):.2f}")


if __name__ == "__main__":
    main()
