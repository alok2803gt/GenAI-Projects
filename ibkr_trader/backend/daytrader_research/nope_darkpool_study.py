"""Phase 2b/2c: NOPE and DARK POOL as directional signals.

NOPE (Net Options Pricing Effect) is the strongest remaining candidate on
theory. It is options delta-hedging pressure normalised by stock volume:

    nope = (call_delta + put_delta) / stock_volume

The premise is mechanical rather than predictive -- if options positioning
implies dealers must buy more stock than the tape can absorb, price is pushed.
Unlike aggregate premium flow (which we showed is coincident, corr +0.29 with
the move already made and -0.013 with the move to come), NOPE is scaled by the
volume available to absorb the hedging, which is what makes it a pressure
measure rather than an activity measure.

DARK POOL prints are the other candidate: off-exchange institutional volume,
with each print's price against the prevailing NBBO mid, so a buy/sell lean can
be inferred. Institutional accumulation is directional by nature and is not
visible in daily bars.

Both are read over the same opening window the rest of this work uses (09:30 to
09:45), entry at the close of the 09:45 bar, exit at the session close, long
only, market-adjusted and date-clustered. EXPLORATION SET ONLY (< 2025-09-16).

    ../../venv/bin/python daytrader_research/nope_darkpool_study.py
"""
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
NOPE = HERE / "uw_nope_cache"
DARK = HERE / "uw_darkpool_cache"
ET = ZoneInfo("America/New_York")
SPLIT = "2025-09-16"
FEE_PP = 0.70 / 148.0 * 100
WIN = 15


def _et(ts: str) -> datetime:
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(ET)


def _mins(ts: str) -> int:
    e = _et(ts)
    return (e.hour - 9) * 60 + (e.minute - 30)


def load_nope() -> pd.DataFrame:
    rows = []
    for f in NOPE.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("error") or not d.get("ticks"):
            continue
        win = []
        for x in d["ticks"]:
            ts = x.get("timestamp")
            if not ts:
                continue
            m = _mins(ts)
            if 0 <= m < WIN:
                win.append(x)
        if not win:
            continue

        def fl(x, k):
            try:
                return float(x.get(k) or 0)
            except Exception:
                return 0.0

        nope_vals = [fl(x, "nope") for x in win]
        cd = sum(fl(x, "call_delta") for x in win)
        pd_ = sum(fl(x, "put_delta") for x in win)
        sv = sum(fl(x, "stock_vol") for x in win)
        cv = sum(fl(x, "call_vol") for x in win)
        pv = sum(fl(x, "put_vol") for x in win)
        rows.append({
            "ticker": d["ticker"], "date": pd.Timestamp(d["date"]),
            "nope_mean": float(np.mean(nope_vals)) if nope_vals else np.nan,
            "nope_last": nope_vals[-1] if nope_vals else np.nan,
            "nope_trend": (nope_vals[-1] - nope_vals[0]) if len(nope_vals) > 1 else np.nan,
            "net_delta_pressure": (cd + pd_) / sv if sv else np.nan,
            "cp_vol_ratio": (cv - pv) / (cv + pv) if (cv + pv) else np.nan,
        })
    return pd.DataFrame(rows)


def load_dark() -> pd.DataFrame:
    rows = []
    for f in DARK.glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("error") or not d.get("prints"):
            continue
        buy = sell = 0.0
        prem = 0.0
        n = 0
        for x in d["prints"]:
            ts = x.get("executed_at")
            if not ts:
                continue
            try:
                m = _mins(ts)
            except Exception:
                continue
            if not (0 <= m < WIN):
                continue
            try:
                px = float(x.get("price") or 0)
                bid = float(x.get("nbbo_bid") or 0)
                ask = float(x.get("nbbo_ask") or 0)
                p = float(x.get("premium") or 0)
            except Exception:
                continue
            if not (px and bid and ask and ask > bid):
                continue
            mid = (bid + ask) / 2
            # a print above the mid leans buyer-initiated, below leans seller
            if px > mid:
                buy += p
            elif px < mid:
                sell += p
            prem += p
            n += 1
        if n < 3:
            continue
        rows.append({
            "ticker": d["ticker"], "date": pd.Timestamp(d["date"]),
            "dp_lean": (buy - sell) / (buy + sell) * 100 if (buy + sell) else np.nan,
            "dp_premium": prem,
            "dp_prints": n,
        })
    return pd.DataFrame(rows)


def load_target() -> pd.DataFrame:
    from micro_features import load
    r = load()
    return r[["ticker", "date", "fwd", "excess"]]


def clustered(s: pd.DataFrame, col="excess") -> tuple:
    if len(s) < 2:
        return float("nan"), float("nan")
    d = s.groupby("date")[col].mean()
    if len(d) < 2:
        return d.mean(), float("nan")
    return d.mean(), d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))


def screen(r: pd.DataFrame, feats: list, label: str) -> dict:
    print(f"\n----- {label} -----")
    tr = r[r["date"] < SPLIT]
    print(f"exploration rows: {len(tr):,}, sessions: {tr.date.nunique()}")
    print(f"  {'feature':<22}{'Q1':>9}{'Q3':>9}{'Q3-Q1':>9}{'t':>7}"
          f"{'Q3 excess':>11}{'t(Q3)':>8}{'net':>9}")
    out = {}
    for f in feats:
        s = tr.dropna(subset=[f, "excess"]).copy()
        if len(s) < 150:
            print(f"  {f:<22} too few rows ({len(s)})")
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
            continue
        dd = pair["h"] - pair["l"]
        td = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd)))
        m3, t3 = clustered(hi)
        m1, _ = clustered(lo)
        out[f] = (dd.mean(), td, m3, t3)
        print(f"  {f:<22}{m1:>9.3f}{m3:>9.3f}{dd.mean():>9.3f}{td:>7.2f}"
              f"{m3:>11.3f}{t3:>8.2f}{m3 - FEE_PP:>9.3f}")
    return out


def main() -> None:
    tgt = load_target()
    print(f"fee hurdle at $148 = {FEE_PP:.3f}pp   (window 09:30-09:{30 + WIN})")
    res = {}
    n = load_nope()
    if not n.empty:
        res.update(screen(n.merge(tgt, on=["ticker", "date"]),
                          ["nope_mean", "nope_last", "nope_trend",
                           "net_delta_pressure", "cp_vol_ratio"],
                          "NOPE (options hedging pressure / stock volume)"))
    else:
        print("\nNOPE cache empty")
    dk = load_dark()
    if not dk.empty:
        res.update(screen(dk.merge(tgt, on=["ticker", "date"]),
                          ["dp_lean", "dp_premium", "dp_prints"],
                          "DARK POOL (off-exchange print lean vs NBBO mid)"))
    else:
        print("\ndark pool cache empty")

    if res:
        from statistics import NormalDist
        bar = NormalDist().inv_cdf(1 - 0.05 / (2 * len(res)))
        print(f"\n=== VERDICT ===")
        print(f"  {len(res)} features tested, Bonferroni bar |t| > {bar:.2f}")
        best = max(res.items(), key=lambda kv: abs(kv[1][1]))
        print(f"  strongest spread: {best[0]} {best[1][0]:+.3f}pp (t {best[1][1]:+.2f})")
        win = [k for k, v in res.items() if v[2] > FEE_PP and v[3] > 2]
        print(f"  top tercile clears the {FEE_PP:.2f}pp fee with t>2: {win or 'NONE'}")


if __name__ == "__main__":
    main()
