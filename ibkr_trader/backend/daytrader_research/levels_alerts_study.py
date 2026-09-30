"""Phase 2d/2e: the LAST two untested directional sources.

PRICE-LEVEL POSITIONING (/api/stock/{t}/option/stock-price-levels): call and put
volume at every price level. The directional read is where the options market is
positioned RELATIVE TO SPOT -- calls stacked above and puts below is a different
configuration from the reverse, and it is not visible in any aggregate.

UNUSUAL OPTIONS ACTIVITY (/api/option-trades/flow-alerts): UW's flagship feed of
large, unusual single trades, each with delta, premium, type and whether it was
opening. This is the retail-famous "whale following" signal and the most directly
directional product UW sells. Fetched per day (a narrow window returns HTTP 500)
and filtered to the opening 15 minutes client-side, which also sidesteps the
EST/EDT offset.

Same protocol throughout: opening window 09:30-09:45, entry at the close of the
09:45 bar, exit at the session close, long only, market-adjusted, date-clustered,
EXPLORATION SET ONLY (< 2025-09-16).

    ../../venv/bin/python daytrader_research/levels_alerts_study.py
"""
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
LEVELS = HERE / "uw_levels_cache"
LVLPREV = HERE / "uw_levels_prev_cache"
ALERTS = HERE / "uw_alerts_cache"
ET = ZoneInfo("America/New_York")
SPLIT = "2025-09-16"
FEE_PP = 0.70 / 148.0 * 100
WIN = 15


def _f(x, k, dflt=0.0):
    try:
        v = x.get(k)
        return float(v) if v is not None else dflt
    except Exception:
        return dflt


def load_levels(spot: dict, prev: bool = False) -> pd.DataFrame:
    """prev=True reads the PRIOR session's positioning, which is the only valid
    version: the same-day file covers the WHOLE session, so splitting it at an
    intraday price encodes where price subsequently went (corr +0.67 with the
    forward return, and lvl_call_above / lvl_put_below came out at -0.93 to each
    other -- an identity, not a signal). Rejected 2026-09-30."""
    rows = []
    for f in (LVLPREV if prev else LEVELS).glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("error") or not d.get("rows"):
            continue
        key = (d["ticker"], d["date"])
        s = spot.get(key)
        if not s:
            continue
        ca = cb = pa = pb = 0.0
        for x in d["rows"]:
            px = _f(x, "price")
            if not px:
                continue
            cv, pv = _f(x, "call_volume"), _f(x, "put_volume")
            if px > s:
                ca += cv
                pa += pv
            else:
                cb += cv
                pb += pv
        tot = ca + cb + pa + pb
        if tot < 100:
            continue
        rows.append({
            "ticker": d["ticker"], "date": pd.Timestamp(d["date"]),
            # calls above spot are upside bets; puts below are downside bets
            "lvl_call_above": (ca - cb) / (ca + cb) * 100 if (ca + cb) else np.nan,
            "lvl_put_below": (pb - pa) / (pa + pb) * 100 if (pa + pb) else np.nan,
            "lvl_cp_above": (ca - pa) / (ca + pa) * 100 if (ca + pa) else np.nan,
            "lvl_skew": ((ca + pb) - (cb + pa)) / tot * 100,
        })
    return pd.DataFrame(rows)


def load_alerts() -> pd.DataFrame:
    rows = []
    for f in ALERTS.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("error") or not d.get("rows"):
            continue
        cp = pp = 0.0
        dl = 0.0
        n = nc = 0
        for x in d["rows"]:
            ts = x.get("created_at")
            if not ts:
                continue
            try:
                e = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(ET)
            except Exception:
                continue
            m = (e.hour - 9) * 60 + (e.minute - 30)
            if not (0 <= m < WIN):
                continue
            prem = _f(x, "total_premium") or _f(x, "premium")
            typ = str(x.get("type") or x.get("option_type") or "").lower()
            delta = _f(x, "delta")
            if "call" in typ or typ == "c":
                cp += prem
                nc += 1
            elif "put" in typ or typ == "p":
                pp += prem
            dl += delta * prem
            n += 1
        if n < 2:
            continue
        rows.append({
            "ticker": d["ticker"], "date": pd.Timestamp(d["date"]),
            "ua_cp_prem": (cp - pp) / (cp + pp) * 100 if (cp + pp) else np.nan,
            "ua_delta_wt": dl / (cp + pp) if (cp + pp) else np.nan,
            "ua_count": n,
            "ua_call_frac": nc / n * 100,
        })
    return pd.DataFrame(rows)


def clustered(s: pd.DataFrame, col="excess") -> tuple:
    if len(s) < 2:
        return float("nan"), float("nan")
    dd = s.groupby("date")[col].mean()
    if len(dd) < 2:
        return dd.mean(), float("nan")
    return dd.mean(), dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd)))


def screen(r: pd.DataFrame, feats: list, label: str) -> dict:
    tr = r[r["date"] < SPLIT]
    print(f"\n----- {label} -----")
    print(f"exploration rows: {len(tr):,}, sessions: {tr.date.nunique()}")
    print(f"  {'feature':<20}{'Q1':>9}{'Q3':>9}{'Q3-Q1':>9}{'t':>7}"
          f"{'Q3 excess':>11}{'t(Q3)':>8}{'net':>9}")
    out = {}
    for f in feats:
        s = tr.dropna(subset=[f, "excess"]).copy()
        if len(s) < 150:
            print(f"  {f:<20} too few rows ({len(s)})")
            continue
        s["q"] = s.groupby("date")[f].transform(
            lambda x: pd.qcut(x, 3, labels=False, duplicates="drop")
            if x.notna().sum() >= 3 else np.nan)
        s = s.dropna(subset=["q"])
        hi, lo = s[s.q == s.q.max()], s[s.q == 0]
        a = hi.groupby("date")["excess"].mean()
        b = lo.groupby("date")["excess"].mean()
        pair = pd.concat([a.rename("h"), b.rename("l")], axis=1, sort=False).dropna()
        if len(pair) < 20:
            print(f"  {f:<20} too few paired days ({len(pair)})")
            continue
        dd = pair["h"] - pair["l"]
        td = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd)))
        m3, t3 = clustered(hi)
        m1, _ = clustered(lo)
        out[f] = (dd.mean(), td, m3, t3)
        print(f"  {f:<20}{m1:>9.3f}{m3:>9.3f}{dd.mean():>9.3f}{td:>7.2f}"
              f"{m3:>11.3f}{t3:>8.2f}{m3 - FEE_PP:>9.3f}")
    return out


def main() -> None:
    from micro_features import load
    micro = load()
    tgt = micro[["ticker", "date", "excess"]]
    # spot for the levels split = the 09:45 entry price
    spot = {(t, d.strftime("%Y-%m-%d")): e
            for t, d, e in micro[["ticker", "date", "entry"]].itertuples(index=False)}

    print(f"fee hurdle at $148 = {FEE_PP:.3f}pp   (window 09:30-09:{30 + WIN})")
    res = {}
    lv = load_levels(spot, prev=True)
    if not lv.empty:
        res.update(screen(lv.merge(tgt, on=["ticker", "date"]),
                          ["lvl_call_above", "lvl_put_below", "lvl_cp_above", "lvl_skew"],
                          "PRIOR-SESSION option volume by price level vs today's 09:45"))
    else:
        print("\nprior-session levels cache empty")
    al = load_alerts()
    if not al.empty:
        res.update(screen(al.merge(tgt, on=["ticker", "date"]),
                          ["ua_cp_prem", "ua_delta_wt", "ua_count", "ua_call_frac"],
                          "UNUSUAL OPTIONS ACTIVITY (whale alerts)"))
    else:
        print("\nalerts cache empty")

    if res:
        from statistics import NormalDist
        bar = NormalDist().inv_cdf(1 - 0.05 / (2 * len(res)))
        print(f"\n=== VERDICT ===")
        print(f"  {len(res)} features, Bonferroni bar |t| > {bar:.2f}")
        best = max(res.items(), key=lambda kv: abs(kv[1][1]))
        print(f"  strongest: {best[0]} {best[1][0]:+.3f}pp (t {best[1][1]:+.2f})")
        win = [k for k, v in res.items() if v[2] > FEE_PP and v[3] > 2]
        print(f"  clears the {FEE_PP:.2f}pp fee with t>2: {win or 'NONE'}")


if __name__ == "__main__":
    main()
