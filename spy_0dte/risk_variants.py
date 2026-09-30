"""
Risk-reduction variants for the hold-to-expiry iron condor.

Goal: cut the maximum DAILY loss without giving up the edge.
For every condor we record, in one pass:
    credit, P&L held to expiry,
    P&L with an intraday stop at 1.0 / 1.5 / 2.0 x credit (structure marked
        every minute from its fill to 15:45; exit that minute at lambda 0.25),
    how far price had already moved from the open at entry (trend filter),
    entry minute and the minute the stop fired.
Variants (wings, distance, stop, circuit breaker, trend filter, cadence) are
then evaluated offline from this file.

    ./.venv/bin/python risk_variants.py build SPY
    ./.venv/bin/python risk_variants.py evaluate SPY
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import condor_lean as L
import option_returns as R
import spy0dte_framework as F

COMBOS = ((1.0, 3.0), (0.5, 5.0), (1.0, 1.0), (0.5, 1.0), (1.5, 3.0))
STOPS = (0.5, 0.75, 1.0, 1.5, 2.0)
TAUS = range(30, 301, 15)          # midday entries only (the validated window)
Q = L.Q


def leg_series(chain, right, k, n):
    a = chain.px.get((right, k))
    if a is None:
        return None
    return pd.Series(a).ffill(limit=R.MAX_STALE).values


def build(root, settle_mode="close"):
    import expiry_family as XF
    is_spx = root == "SPX"
    src = "SPY" if is_spx else root
    bars = F.prepare_bars(pd.read_parquet(f"data/{src}_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
    close = bars.groupby("session")["close"].last()
    if settle_mode == "after_hours" and not root == "SPX":
        ah = XF.after_hours_settle(root, bars)
        close = ah.reindex(close.index).fillna(close)
    if is_spx:
        spx_close = pd.read_csv("data/SPX_History.csv", parse_dates=["DATE"]).set_index("DATE")["SPX"]
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
        if not chain.px:
            continue
        if is_spx:
            d0 = s.tz_localize(None)
            if d0 not in spx_close.index:
                continue
            S = L.parity_spot(chain, n)
            first = pd.Series(S).bfill()
            if not np.isfinite(first.iloc[0]):
                continue
            scale = float(first.iloc[0]) / spy_open[s]
            settle = float(spx_close[d0])
        else:
            S = day["close"].reindex(range(n)).values
            scale = opens[s] / spy_open[s]
            settle = float(close[s])
        eod = n - R.EOD_BEFORE_CLOSE
        for tau in TAUS:
            if tau + 1 >= eod - 5 or not np.isfinite(S[tau]):
                continue
            s0, T = float(S[tau]), n - tau - 1
            iv0 = chain.atm_iv(tau, s0, T)
            if not (np.isfinite(iv0) and iv0 > 0):
                continue
            u = s0 * iv0 * math.sqrt(T / F.MIN_PER_YEAR)
            te = tau + 1
            for d, w in COMBOS:
                wd = max(5.0, 5.0 * round(w * scale / 5.0)) if is_spx else max(1.0, round(w * scale))
                legs, ok = {}, True
                for right, sgn in (("P", -1), ("C", +1)):
                    ks = chain.nearest_printed(right, s0 + sgn * d * u, tau)
                    if ks is None:
                        ok = False
                        break
                    kw = ks + sgn * wd
                    es, _ = chain.price(right, ks, te)
                    ew, _ = chain.price(right, kw, te)
                    if not (np.isfinite(es) and np.isfinite(ew)):
                        ok = False
                        break
                    legs[right] = (ks, kw, es, ew)
                if not ok:
                    continue
                credit = sum(Q.sell(legs[r][2], T) - Q.buy(legs[r][3], T) for r in ("P", "C"))
                if credit <= 0:
                    continue
                intr = lambda k, r: max(k - settle, 0.0) if r == "P" else max(settle - k, 0.0)
                pnl_exp = (credit - sum(intr(legs[r][0], r) - intr(legs[r][1], r) for r in ("P", "C"))) * 100 - 4 * R.COMMISSION
                # ---- intraday marks for the stops -------------------------------
                ser = {(r, i): leg_series(chain, r, legs[r][i], n) for r in ("P", "C") for i in (0, 1)}
                rec = dict(session=s, tau=tau, d=d, w=w, wing=wd, credit=credit * 100, pnl_exp=pnl_exp,
                           dist_u=abs(s0 - S[0]) / u if np.isfinite(S[0]) else np.nan, u=u)
                for t2 in TAUS:                      # mark-to-market at each later decision time
                    rec[f"mtm{t2}"] = np.nan
                if any(v is None for v in ser.values()):
                    for k in STOPS:
                        rec[f"pnl_stop{k}"], rec[f"stop_min{k}"] = pnl_exp, np.nan
                else:
                    mids = np.full(n, np.nan)
                    idx = np.arange(te, eod + 1)
                    val = np.zeros(len(idx))
                    for r in ("P", "C"):
                        val += ser[(r, 0)][idx] - ser[(r, 1)][idx]        # cost to close (short - long)
                    loss = (val - credit) * 100                            # + = losing
                    for k in STOPS:
                        hit = np.where(np.isfinite(loss) & (loss >= k * credit * 100))[0]
                        if len(hit):
                            m = idx[hit[0]]
                            Tm = n - m
                            close_cost = sum(Q.buy(ser[(r, 0)][m], Tm) - Q.sell(ser[(r, 1)][m], Tm) for r in ("P", "C"))
                            rec[f"pnl_stop{k}"] = (credit - close_cost) * 100 - 8 * R.COMMISSION
                            rec[f"stop_min{k}"] = m
                        else:
                            rec[f"pnl_stop{k}"], rec[f"stop_min{k}"] = pnl_exp, np.nan
                    v_eod = sum(ser[(r, 0)][eod] - ser[(r, 1)][eod] for r in ("P", "C"))
                    if np.isfinite(v_eod):           # Alpaca-compatible: close at 15:45 instead of expiry
                        cost = sum(Q.buy(ser[(r, 0)][eod], n - eod) - Q.sell(ser[(r, 1)][eod], n - eod)
                                   for r in ("P", "C"))
                        rec["pnl_1545"] = (credit - cost) * 100 - 8 * R.COMMISSION
                        for k in STOPS:              # stop first, else 15:45 close
                            rec[f"pnl_stop{k}_1545"] = (rec[f"pnl_stop{k}"] if np.isfinite(rec.get(f"stop_min{k}", np.nan))
                                                        else rec["pnl_1545"])
                    else:
                        rec["pnl_1545"] = np.nan
                        for k in STOPS:
                            rec[f"pnl_stop{k}_1545"] = np.nan
                    for t2 in TAUS:                  # open-position P&L a later entry can actually see
                        if t2 > tau and t2 < n:
                            v = sum(ser[(r, 0)][t2] - ser[(r, 1)][t2] for r in ("P", "C"))
                            rec[f"mtm{t2}"] = (credit - v) * 100 if np.isfinite(v) else np.nan
                rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_parquet(f"data/risk_variants_{root}{'' if settle_mode == 'close' else '_' + settle_mode}.parquet")
    print(f"{root}: {len(out):,} condors, {out.session.nunique()} sessions")


def daily(df, col, breaker=None):
    """Daily P&L per unit. `breaker` stops NEW entries once the OPEN positions'
    mark-to-market at that decision time is below -breaker (no look-ahead: the
    mark uses only prices up to that minute)."""
    if breaker is None:
        return df.groupby("session")[col].sum()
    out = {}
    for s, g in df.sort_values("tau").groupby("session"):
        taken, tot = [], 0.0
        for _, r in g.iterrows():
            mtm = sum(t[f"mtm{int(r.tau)}"] for t in taken if np.isfinite(t.get(f"mtm{int(r.tau)}", np.nan)))
            if taken and mtm <= -breaker:
                break
            taken.append(r)
            tot += r[col]
        out[s] = tot
    return pd.Series(out)


def evaluate(root):
    pd.set_option("display.width", 240)
    df = pd.read_parquet(f"data/risk_variants_{root}.parquet")
    rows = []
    for (d, w), g0 in df.groupby(["d", "w"]):
        for cadence, taus in (("15min (19/day)", list(TAUS)), ("30min (10/day)", list(range(30, 301, 30))),
                              ("60min (5/day)", list(range(30, 301, 60)))):
            for trend in (None, 1.0, 1.5):
                g = g0[g0.tau.isin(taus)]
                if trend is not None:
                    g = g[g.dist_u <= trend]
                for col in ["pnl_exp"] + [f"pnl_stop{k}" for k in STOPS]:
                    for breaker in (None, 1000.0):
                        dly = daily(g, col, breaker)
                        if len(dly) < 100:
                            continue
                        t = dly.mean() / (dly.std(ddof=1) / math.sqrt(len(dly)))
                        eq = dly.cumsum()
                        rows.append(dict(d=d, w=w, cadence=cadence, trend=trend or "-",
                                         exit="expiry" if col == "pnl_exp" else f"stop {col[8:]}x",
                                         breaker=breaker or "-", trades=len(g), day_mean=dly.mean(), t=t,
                                         worst=dly.min(), p01=dly.quantile(0.01), max_dd=(eq - eq.cummax()).min(),
                                         ret_per_worst=dly.mean() * 252 / abs(dly.min())))
    r = pd.DataFrame(rows)
    r.to_csv(f"data/risk_variants_{root}_grid.csv", index=False)
    base = r[(r.d == 1.0) & (r.w == 3.0) & (r.cadence == "15min (19/day)") & (r.trend == "-") &
             (r.exit == "expiry") & (r.breaker == "-")].iloc[0]
    print(f"BASELINE (validated): day mean ${base.day_mean:.2f}, t {base.t:.2f}, worst day ${base.worst:.0f}, "
          f"max DD ${base.max_dd:.0f}, annual/|worst| {base.ret_per_worst:.1f}")
    keep = r[(r.day_mean > 0.5 * base.day_mean) & (r.worst > 0.5 * base.worst)]
    print(f"\nvariants keeping >50% of the edge AND cutting the worst day by >50%: {len(keep)}")
    cols = ["d", "w", "cadence", "trend", "exit", "breaker", "day_mean", "t", "worst", "p01", "max_dd", "ret_per_worst"]
    print(keep.sort_values("ret_per_worst", ascending=False).head(12)[cols].round(2).to_string(index=False))
    print("\nbest by return-per-worst-day overall:")
    print(r.sort_values("ret_per_worst", ascending=False).head(12)[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "close")
    else:
        evaluate(sys.argv[2])
