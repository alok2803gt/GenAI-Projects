"""
Runs PREREG_condor_replication.md (QQQ, IWM) and part A of PREREG_condor_gex_spx.md.
Verifies both pre-registrations' hashes, audits data alignment first (put-call
parity lag test), then applies the frozen SPY pipeline with only the root
symbol and the per-session wing width changed.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import explore_round4 as E4
import option_returns as R
import spy0dte_framework as F

ASSETS = ("QQQ", "IWM")
T_BAR = 2.24


def verify(prereg):
    line = Path(prereg.replace(".md", ".sha256")).read_text().split()
    if hashlib.sha256(Path(prereg).read_bytes()).hexdigest() != line[0]:
        raise SystemExit(f"{prereg} changed since it was frozen -- refusing to run")


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 2 else np.nan


def parity_audit(root, bars, files):
    """median |C - P + K - S(lag)| on a sample of sessions; minimum must sit at lag 0."""
    closes = {s: g.set_index("tau")["close"] for s, g in bars.groupby("session")}
    errs = {lag: [] for lag in (-2, -1, 0, 1, 2)}
    for p in files[::10]:
        s = pd.Timestamp(p.stem, tz=F.NY)
        if s not in closes:
            continue
        n = int(bars.loc[bars.session == s, "session_len"].iloc[0])
        ch = R.DayChain(pd.read_parquet(p), s, n)
        if not ch.px:
            continue
        cl = closes[s].reindex(range(n)).values
        for m in (30, 120, 240, 330):
            ks = sorted([k for k in ch.strikes.get("C", []) if ("P", k) in ch.px], key=lambda k: abs(k - cl[m]))[:1]
            for k in ks:
                c_, p_ = ch.px[("C", k)][m], ch.px[("P", k)][m]
                if np.isfinite(c_) and np.isfinite(p_):
                    for lag in errs:
                        errs[lag].append(abs(c_ - p_ + k - cl[m + lag]))
    med = {lag: float(np.median(v)) for lag, v in errs.items()}
    ok = min(med, key=med.get) == 0
    print(f"   {root} parity alignment audit: median error by lag {({k: round(v, 3) for k, v in med.items()})} -> "
          f"{'ALIGNED' if ok else 'MISALIGNED'}")
    return ok


def gex_prior_day(root, key):
    r = requests.get(f"https://api.unusualwhales.com/api/stock/{root}/greek-exposure",
                     headers={"Authorization": f"Bearer {key}"}, params={"timeframe": "5y"}, timeout=60)
    if r.status_code != 200:
        return None
    g = pd.DataFrame(r.json()["data"])
    g["date"] = pd.to_datetime(g["date"])
    g = g.set_index("date").sort_index()
    net = g["call_gamma"].astype(float) + g["put_gamma"].astype(float)
    med = net.rolling(252, min_periods=120).median().shift(1)     # trailing, strictly before the day
    high = (net > med).where(med.notna())
    return high.shift(1)                                           # use the PRIOR day's reading


def condors(root, spy_open):
    bars = F.prepare_bars(pd.read_parquet(f"data/{root}_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
    feats = R.add_underlying_features(F.FeatureBuilder().transform(bars))
    close = bars.groupby("session")["close"].last()
    opens = bars[bars.tau == 0].set_index("session")["open"]
    by_sess = {s: g.set_index("tau", drop=False).rename_axis("tau_idx") for s, g in feats.groupby("session")}
    files = sorted(Path(f"data/{root}_0dte_option_bars").glob("*.parquet"))
    aligned = parity_audit(root, bars, files)
    rows, wings = [], {}
    lo, hi = pd.Timestamp("2024-02-01", tz=F.NY), pd.Timestamp("2026-09-21", tz=F.NY)
    for p in files:
        s = pd.Timestamp(p.stem, tz=F.NY)
        if not (lo <= s <= hi) or s not in by_sess or s not in spy_open or s not in opens:
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue                                               # no same-day expiry that session
        wing = max(1.0, float(round(3.0 * opens[s] / spy_open[s])))
        R.SPREAD_WIDTH = wing
        day = by_sess[s]
        chain = R.DayChain(df, s, int(day["session_len"].iloc[0]))
        if chain.px:
            rows += R.build_day(chain, day, F.QuoteModel(lam=R.LAMBDA))
            wings[s] = wing
    R.SPREAD_WIDTH = 3.0
    c = pd.DataFrame(rows)
    sp = E4.spread_variants(c, close)
    sp["pnl_exp"] = (sp["credit"] - sp["bbexp"]) * 100 - 2 * R.COMMISSION
    cols = ["pnl_exp", "leg1_entry", "leg2_entry", "T_entry", "bbexp"]
    P = sp[sp.struct == "spread_P_d1.0"].set_index(["session", "tau"])[cols]
    C = sp[sp.struct == "spread_C_d1.0"].set_index(["session", "tau"])[cols]
    j = P.join(C, rsuffix="_c", how="inner").reset_index()
    j["condor"] = j["pnl_exp"] + j["pnl_exp_c"]
    return j, aligned, pd.Series(wings)


def main():
    pd.set_option("display.width", 220)
    verify("PREREG_condor_replication.md")
    verify("PREREG_condor_gex_spx.md")
    print("pre-registrations verified (replication + GEX)")
    key = json.loads(Path(sys.argv[1]).read_text())["unusual_whales_api_key"]
    spy = F.prepare_bars(pd.read_parquet("data/SPY_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
    spy_open = spy[spy.tau == 0].set_index("session")["open"]
    results, log = {}, {}
    for root in ASSETS:
        print(f"\n===== {root} =====")
        j, aligned, wings = condors(root, spy_open)
        day = j.groupby("session")["condor"].mean()
        t = tstat(day)
        passed = bool(day.mean() > 0 and t >= T_BAR)
        results[root] = (day, t)
        print(f"   sessions with a same-day expiry: {len(day)} ({day.index.min().date()} .. {day.index.max().date()}), "
              f"{len(j):,} condors, wing ${wings.min():g}-{wings.max():g} (median {wings.median():g})")
        print(f"   REPLICATION: mean ${day.mean():.2f}/condor per session, t = {t:.2f} -> "
              f"{'PASS' if passed else 'fail'} (bar t >= {T_BAR})")
        side = {s: j.groupby('session')[c].mean() for s, c in (('put', 'pnl_exp'), ('call', 'pnl_exp_c'))}
        mon = day.groupby(day.index.tz_localize(None).to_period("M")).sum()
        print(f"   put side ${side['put'].mean():.2f} (t {tstat(side['put']):.2f}); call side ${side['call'].mean():.2f} "
              f"(t {tstat(side['call']):.2f}); winning days {(day > 0).mean():.0%}; positive months {(mon > 0).sum()}/{len(mon)}")
        g = gex_prior_day(root, key)
        if g is None:
            print("   GEX (A): no Unusual Whales series -> asset dropped from test A")
            gex_res = None
        else:
            d = day.copy()
            d.index = d.index.tz_localize(None).normalize()
            flag = g.reindex(d.index)
            hiP, loP = d[flag == 1.0], d[flag == 0.0]
            diff = hiP.mean() - loP.mean()
            tw = diff / math.sqrt(hiP.var(ddof=1) / len(hiP) + loP.var(ddof=1) / len(loP))
            gex_res = dict(high_n=len(hiP), high=hiP.mean(), low_n=len(loP), low=loP.mean(), diff=diff, t=tw)
            print(f"   GEX (A): HIGH-gamma days n={len(hiP)} ${hiP.mean():.2f} vs LOW n={len(loP)} ${loP.mean():.2f}; "
                  f"diff ${diff:.2f}, Welch t {tw:.2f} -> {'PASS' if diff > 0 and tw >= T_BAR else 'fail'}")
        log[root] = dict(aligned=aligned, sessions=len(day), mean=day.mean(), t=t, passed=passed, gex=gex_res)
        j.to_parquet(f"data/replication_{root}_condors.parquet")

    rep_pass = [r for r in ASSETS if log[r]["passed"]]
    rep_neg = [r for r in ASSETS if log[r]["t"] <= -T_BAR]
    verdict = "CONFIRMED" if rep_pass and not rep_neg else "FAILED"
    print(f"\nREPLICATION VERDICT: {verdict} (passed: {rep_pass or 'none'}; significantly negative: {rep_neg or 'none'})")
    gx = {r: log[r]["gex"] for r in ASSETS if log[r]["gex"]}
    gpass = [r for r, v in gx.items() if v["diff"] > 0 and v["t"] >= T_BAR]
    gneg = [r for r, v in gx.items() if v["t"] <= -T_BAR]
    print(f"GEX CONDITION VERDICT: {'CONFIRMED' if gpass and not gneg else 'NOT CONFIRMED'} "
          f"(passed: {gpass or 'none'}; significantly negative: {gneg or 'none'})")
    pooled = pd.concat([results[r][0] for r in ASSETS])
    print(f"pooled (descriptive): mean ${pooled.mean():.2f}, t {tstat(pooled):.2f}")
    with open("holdout_log.jsonl", "a") as f:
        f.write(json.dumps(dict(test="PREREG_condor_replication v1 + GEX A", verdict=verdict,
                                results={k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                                             for kk, vv in v.items() if kk != "gex"} for k, v in log.items()},
                                gex=gx), default=float) + "\n")


if __name__ == "__main__":
    main()
