"""Phase 2a: does DEALER POSITIONING predict same-day direction?

The theoretical case is the best of anything tested so far, because it is not a
forecast. When dealers are short gamma they must buy into strength and sell into
weakness, which amplifies moves; when long gamma they do the opposite and damp
them. Net delta exposure says which way they are forced to hedge. If any
observable carries mechanical, non-predictive directional information, this is it.

DATA: /api/stock/{t}/greek-exposure, one call per ticker, ~250 sessions each,
call_/put_ gamma, delta, vanna, charm. Crucially the history only reaches back to
~2025-10-01 (a 2024-11-15 request returns zero rows), so it CANNOT be explored on
the pre-2025-09-16 window used for the options-flow work.

SPLIT, therefore its own, stated before looking:
  * EXPLORATION: sessions before 2026-04-01
  * CONFIRMATION: 2026-04-01 onward
Disclosure: these dates overlap the window sealed for the options-flow
hypothesis. That holdout was never spent (nothing survived exploration to test),
and these are different features, so nothing is being laundered -- but the flow
holdout can no longer be treated as pristine for a forward-return question.

Features, all from the PRIOR session's close so they are knowable at the open:
  net_gamma      call_gamma + put_gamma   (sign = amplify vs damp)
  net_delta      call_delta + put_delta   (direction of forced hedging)
  net_vanna      vol-of-vol sensitivity
  net_charm      decay-driven hedging drift into expiry
  gamma_chg      1-day change in net gamma (positioning shifting)
  delta_chg      1-day change in net delta
Each is scaled by the ticker's own trailing 60-session mean absolute value, so a
mega-cap and a mid-cap are comparable.

Target: same-day OPEN->CLOSE excess return, market-adjusted against the session's
candidate mean, date-clustered. Long only.

    ../../venv/bin/python daytrader_research/gex_direction_study.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
GREEK = HERE / "uw_greek_cache"
FLOW = HERE / "uw_flow_cache"
SPLIT = "2026-04-01"
FEE_PP = 0.70 / 148.0 * 100

FEATURES = ["net_gamma", "net_delta", "net_vanna", "net_charm",
            "gamma_chg", "delta_chg"]


def load_greeks() -> pd.DataFrame:
    rows = []
    for f in GREEK.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("error") or not d.get("rows"):
            continue
        t = d["ticker"]
        for x in d["rows"]:
            try:
                rows.append({
                    "ticker": t, "date": pd.Timestamp(x["date"]),
                    "net_gamma": float(x.get("call_gamma") or 0) + float(x.get("put_gamma") or 0),
                    "net_delta": float(x.get("call_delta") or 0) + float(x.get("put_delta") or 0),
                    "net_vanna": float(x.get("call_vanna") or 0) + float(x.get("put_vanna") or 0),
                    "net_charm": float(x.get("call_charm") or 0) + float(x.get("put_charm") or 0),
                })
            except Exception:
                pass
    g = pd.DataFrame(rows).sort_values(["ticker", "date"])
    # scale by each ticker's own trailing 60-session mean |value|, then SHIFT so
    # day t uses only data through t-1
    for c in ("net_gamma", "net_delta", "net_vanna", "net_charm"):
        sc = g.groupby("ticker")[c].transform(
            lambda s: s.abs().rolling(60, min_periods=20).mean())
        g[c] = (g[c] / sc.replace(0, np.nan))
    g["gamma_chg"] = g.groupby("ticker")["net_gamma"].diff()
    g["delta_chg"] = g.groupby("ticker")["net_delta"].diff()
    for c in FEATURES:
        g[c] = g.groupby("ticker")[c].shift(1)     # prior session only
    return g


def load_returns() -> pd.DataFrame:
    """Same-day open->close from the cached 5m bars."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    rows = []
    for f in FLOW.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("error") or not d.get("ohlc"):
            continue
        bars = []
        for c in d["ohlc"]:
            st = c.get("start_time")
            if not st:
                continue
            e = datetime.fromisoformat(st.replace("Z", "+00:00")).astimezone(ET)
            m = (e.hour - 9) * 60 + (e.minute - 30)
            if 0 <= m < 390:
                bars.append((m, float(c["open"]), float(c["close"])))
        if len(bars) < 20:
            continue
        bars.sort()
        o, cl = bars[0][1], bars[-1][2]
        if o:
            rows.append({"ticker": d["ticker"], "date": pd.Timestamp(d["date"]),
                         "ret": (cl - o) / o * 100})
    r = pd.DataFrame(rows)
    r["excess"] = r["ret"] - r.groupby("date")["ret"].transform("mean")
    return r


def clustered(s: pd.DataFrame, col="excess") -> tuple:
    if len(s) < 2:
        return float("nan"), float("nan"), 0
    d = s.groupby("date")[col].mean()
    if len(d) < 2:
        return d.mean(), float("nan"), len(d)
    return d.mean(), d.mean() / (d.std(ddof=1) / math.sqrt(len(d))), len(d)


def screen(r: pd.DataFrame, label: str) -> dict:
    print(f"\n----- {label}: {len(r):,} rows, {r.date.nunique()} sessions -----")
    print(f"  {'feature':<14}{'Q1':>9}{'Q3':>9}{'Q3-Q1':>9}{'t':>7}"
          f"{'Q3 excess':>11}{'t(Q3)':>8}{'net':>9}")
    out = {}
    for f in FEATURES:
        s = r.dropna(subset=[f, "excess"]).copy()
        if len(s) < 150:
            print(f"  {f:<14} too few rows ({len(s)})")
            continue
        s["q"] = s.groupby("date")[f].transform(
            lambda x: pd.qcut(x, 3, labels=False, duplicates="drop")
            if x.notna().sum() >= 3 else np.nan)
        s = s.dropna(subset=["q"])
        hi, lo = s[s.q == s.q.max()], s[s.q == 0]
        a = hi.groupby("date")["excess"].mean()
        b = lo.groupby("date")["excess"].mean()
        pair = pd.concat([a.rename("h"), b.rename("l")], axis=1, sort=False).dropna()
        if len(pair) < 15:
            continue
        dd = pair["h"] - pair["l"]
        td = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd)))
        m3, t3, _ = clustered(hi)
        m1, _, _ = clustered(lo)
        out[f] = (dd.mean(), td, m3, t3)
        print(f"  {f:<14}{m1:>9.3f}{m3:>9.3f}{dd.mean():>9.3f}{td:>7.2f}"
              f"{m3:>11.3f}{t3:>8.2f}{m3 - FEE_PP:>9.3f}")
    return out


def main() -> None:
    g = load_greeks()
    rt = load_returns()
    r = g.merge(rt, on=["ticker", "date"], how="inner")
    print(f"merged: {len(r):,} ticker-days, {r.ticker.nunique()} tickers, "
          f"{r.date.min().date()} .. {r.date.max().date()}")
    print(f"fee hurdle at $148 = {FEE_PP:.3f}pp")

    tr = r[r["date"] < SPLIT]
    te = r[r["date"] >= SPLIT]
    a = screen(tr, f"EXPLORATION (< {SPLIT})")
    if not a:
        print("\nno feature had enough data to screen")
        return
    best = max(a.items(), key=lambda kv: abs(kv[1][1]))
    from statistics import NormalDist
    bar = NormalDist().inv_cdf(1 - 0.05 / (2 * len(a)))
    print(f"\n  Bonferroni bar for {len(a)} features: |t| > {bar:.2f}")
    print(f"  strongest: {best[0]} ({best[1][0]:+.3f}pp, t={best[1][1]:+.2f})")
    worth = [k for k, v in a.items() if abs(v[1]) > bar]
    print(f"  clears the bar on exploration: {worth or 'NONE'}")
    if worth:
        b = screen(te, f"CONFIRMATION (>= {SPLIT})")
        print("\n  confirmation for the survivors only:")
        for k in worth:
            if k in b:
                held = (a[k][0] > 0) == (b[k][0] > 0) and abs(b[k][1]) > 2
                print(f"    {k}: exploration {a[k][0]:+.3f} (t{a[k][1]:+.2f}) -> "
                      f"confirmation {b[k][0]:+.3f} (t{b[k][1]:+.2f})  "
                      f"{'HOLDS' if held else 'FAILS'}")
    else:
        print("  -> confirmation set NOT touched; nothing earned a test.")


if __name__ == "__main__":
    main()
