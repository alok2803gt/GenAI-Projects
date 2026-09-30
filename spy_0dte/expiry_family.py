"""
Hold-to-expiry structure family: discovery on SPY, confirmation on QQQ.

Per session and decision bar (every 15 min, 10:00-15:30): expected-move unit u
from ATM IV at the signal minute (same as the frozen pipeline), then for
right in {P, C} and distance d in {0.5, 1.0, 1.5} u the nearest printed short
strike; entry at the next minute's VWAP (lambda 0.25 modeled cost, $0.65 per
contract), settlement at intrinsic vs the underlying's final close.
    naked short option     (right, d)
    credit spread          (right, d, wing w in {1, 3, 5} x wing-scale)
Structures assembled per (session, tau): spreads, iron condors (both sides,
same d and w), short strangles (both nakeds, same d), short straddle (ATM).
Windows: midday (10:00-14:30) and late (14:45-15:30).

    ./.venv/bin/python expiry_family.py build SPY      # cache legs
    ./.venv/bin/python expiry_family.py discover       # SPY ranking + selection rule
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import condor_lean as L
import option_returns as R
import spy0dte_framework as F

TAUS = range(30, 361, 15)
DISTS = (0.5, 1.0, 1.5)
WINGS = (1.0, 3.0, 5.0)
Q = L.Q


def legs_for_day(chain, S, n, session, settle, wing_scale):
    eod = n - R.EOD_BEFORE_CLOSE
    rows = []
    for tau in TAUS:
        if tau + 1 >= n - 2 or not np.isfinite(S[tau]):
            continue
        s0, T = float(S[tau]), n - tau - 1
        iv0 = chain.atm_iv(tau, s0, T)
        if not (np.isfinite(iv0) and iv0 > 0):
            continue
        u = s0 * iv0 * math.sqrt(T / F.MIN_PER_YEAR)
        te, T_e = tau + 1, n - tau - 1
        win = "midday" if tau <= 300 else "late"
        for right, sgn in (("P", -1), ("C", +1)):
            intr = (lambda k: max(k - settle, 0.0)) if right == "P" else (lambda k: max(settle - k, 0.0))
            for d in (0.0,) + DISTS:
                ks = chain.nearest_printed(right, s0 + sgn * d * u, tau)
                if ks is None:
                    continue
                es, _ = chain.price(right, ks, te)
                if not np.isfinite(es):
                    continue
                naked = (Q.sell(es, T_e) - intr(ks)) * 100 - R.COMMISSION
                rows.append(dict(session=session, tau=tau, window=win, right=right, d=d, w=0.0, ks=ks,
                                 pnl=naked, credit=Q.sell(es, T_e) * 100, u=u))
                if d == 0.0:
                    continue
                for w in WINGS:
                    wd = max(1.0, round(w * wing_scale))
                    kw = ks + sgn * wd
                    ew, _ = chain.price(right, kw, te)
                    if not np.isfinite(ew):
                        continue
                    credit = Q.sell(es, T_e) - Q.buy(ew, T_e)
                    if credit <= 0:
                        continue
                    pnl = (credit - (intr(ks) - intr(kw))) * 100 - 2 * R.COMMISSION
                    rows.append(dict(session=session, tau=tau, window=win, right=right, d=d, w=w, ks=ks,
                                     pnl=pnl, credit=credit * 100, u=u))
    return rows


def after_hours_settle(root, bars):
    """Last price within 90 min after the regular close (holders' exercise cutoff ~17:30)."""
    raw = pd.read_parquet(f"data/{root}_1min_sip.parquet")[["close"]]
    t = raw.index.tz_convert(F.NY)
    raw = raw.set_axis(t)
    out = {}
    ends = bars.groupby("session").apply(lambda g: g.index.max())
    for s, last_bar in ends.items():
        end = last_bar + pd.Timedelta(minutes=1)
        w = raw.loc[(raw.index >= end) & (raw.index < end + pd.Timedelta(minutes=89)), "close"]
        if len(w):
            out[s] = float(w.iloc[-1])
    return pd.Series(out)


def build(root, settle_mode="close"):
    bars = F.prepare_bars(pd.read_parquet(f"data/{root}_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
    close = bars.groupby("session")["close"].last()
    if settle_mode == "after_hours":
        ah = after_hours_settle(root, bars)
        close = ah.reindex(close.index).fillna(close)
    spy = F.prepare_bars(pd.read_parquet("data/SPY_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
    spy_open = spy[spy.tau == 0].set_index("session")["open"]
    opens = bars[bars.tau == 0].set_index("session")["open"]
    rows = []
    for p in sorted(Path(f"data/{root}_0dte_option_bars").glob("*.parquet")):
        s = pd.Timestamp(p.stem, tz=F.NY)
        if s not in close.index or s not in spy_open.index:
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        day = bars[bars.session == s].set_index("tau")
        n = int(day["session_len"].iloc[0])
        chain = R.DayChain(df, s, n)
        if chain.px:
            rows += legs_for_day(chain, day["close"].reindex(range(n)).values, n, s, float(close[s]),
                                 opens[s] / spy_open[s])
    out = pd.DataFrame(rows)
    suffix = "" if settle_mode == "close" else "_" + settle_mode
    out.to_parquet(f"data/expiry_family_legs_{root}{suffix}.parquet")
    print(f"{root}: {len(out):,} legs over {out.session.nunique()} sessions")


def structures(legs: pd.DataFrame) -> pd.DataFrame:
    """Assemble structure P&L per (session, tau)."""
    k = ["session", "tau", "window"]
    out = []
    sp = legs[legs.w > 0]
    for (right, d, w), g in sp.groupby(["right", "d", "w"]):
        out.append(g[k + ["pnl"]].assign(struct=f"{'put' if right == 'P' else 'call'} spread d{d} w{w:g}"))
    for (d, w), g in sp.groupby(["d", "w"]):
        p, c = g[g.right == "P"].set_index(k)["pnl"], g[g.right == "C"].set_index(k)["pnl"]
        both = (p + c).dropna().reset_index()
        out.append(both.assign(struct=f"iron condor d{d} w{w:g}"))
    nk = legs[legs.w == 0]
    for d, g in nk.groupby("d"):
        p, c = g[g.right == "P"].set_index(k)["pnl"], g[g.right == "C"].set_index(k)["pnl"]
        both = (p + c).dropna().reset_index()
        name = "short straddle (ATM)" if d == 0 else f"short strangle d{d}"
        out.append(both.assign(struct=name))
    return pd.concat(out, ignore_index=True)


def session_t(x: pd.DataFrame):
    m = x.groupby("session")["pnl"].mean()
    return m.mean(), m.mean() / (m.std(ddof=1) / math.sqrt(len(m))), len(m)


def discover():
    from scipy.stats import norm
    pd.set_option("display.width", 220)
    st = structures(pd.read_parquet("data/expiry_family_legs_SPY.parquet"))
    split = pd.Timestamp(F.HOLDOUT_START, tz=F.NY)
    rows = []
    for (s, win), g in st.groupby(["struct", "window"]):
        m, t, n = session_t(g)
        m1, t1, _ = session_t(g[g.session < split])
        m2, t2, _ = session_t(g[g.session >= split])
        rows.append(dict(struct=s, window=win, sessions=n, mean=m, t=t, mean_dev=m1, t_dev=t1, mean_hold=m2, t_hold=t2))
    r = pd.DataFrame(rows).sort_values("t", ascending=False)
    bar = norm.ppf(1 - 0.025 / len(r))
    print(f"SPY discovery: {len(r)} structure x window tests; Bonferroni bar t >= {bar:.2f}")
    print(r.head(20).round(2).to_string(index=False))
    print("\nbottom 5:")
    print(r.tail(5).round(2).to_string(index=False))
    r.to_csv("data/expiry_family_SPY_ranking.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "close")
    else:
        discover()
