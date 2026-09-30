"""
SPY 0DTE last-hour test -- implements PREREG_last_hour.md exactly.

The pre-registration's SHA-256 is checked at start-up; the run refuses to
proceed if the document changed. Development sessions only; the holdout is
never loaded. Unconditional test: no model, no features.

    ./.venv/bin/python last_hour.py
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

import execution_sensitivity as ES
import option_returns as R
import spy0dte_framework as F

PREREG = Path(__file__).with_name("PREREG_last_hour.md")
PREREG_SHA256 = "72a2d28ef8774ef939aba864d68397a56a70f1719367ef66f5aa7344ca2f61c9"

FULL_SESSION = 390
SIGNAL_TAUS = (329, 344, 359)           # signal bars 14:59 / 15:14 / 15:29 -> fills 15:00 / 15:15 / 15:30
EXIT_TAU = 385                          # 15:55
EXIT_FORWARD = 4                        # first later print up to 15:59
WIDTH = 2.0
SHORT_U = 0.5
LAM_PRIMARY = 0.25
T_BONFERRONI = 2.50                     # 4 structures, two-sided alpha 0.05
LAMBDA_STAR_MIN = 0.40
MAX_MISSING = 0.02
STRUCTS = ("L_C", "L_P", "SPS", "SCS")


def build_day(chain: R.DayChain, close: np.ndarray, session) -> list:
    """Candidates for one full session. `close[tau]` = SPY close of minute tau."""
    rows, n = [], FULL_SESSION
    q = F.QuoteModel(lam=LAM_PRIMARY)
    for tau in SIGNAL_TAUS:
        S, T = float(close[tau]), n - tau - 1
        iv0 = chain.atm_iv(tau, S, T)                          # prints <= tau only
        if not (np.isfinite(iv0) and iv0 > 0):
            continue
        u = S * iv0 * math.sqrt(T / F.MIN_PER_YEAR)
        te, T_e, T_x = tau + 1, n - tau - 1, n - EXIT_TAU
        base = dict(session=session, tau=tau, entry_time=f"{(570 + te) // 60}:{(570 + te) % 60:02d}", S=S, u=u,
                    atm_iv=iv0, T_entry=float(T_e), T_exit=float(T_x))
        for right, sid in (("C", "L_C"), ("P", "L_P")):
            k = chain.nearest_printed(right, S, tau)
            if k is None:
                continue
            e, est = chain.price(right, k, te)
            if not np.isfinite(e):
                continue
            x, xst = chain.exit_price(right, k, EXIT_TAU, forward=EXIT_FORWARD)
            rows.append(dict(base, struct=sid, family="long", strikes=f"{k:g}{right}", width=np.nan,
                             leg1_entry=e, leg1_exit=x, leg2_entry=np.nan, leg2_exit=np.nan,
                             entry_stale=est, exit_stale=xst))
        for right, sid, sgn in (("P", "SPS", -1), ("C", "SCS", +1)):
            ks = chain.nearest_printed(right, S + sgn * SHORT_U * u, tau)
            if ks is None:
                continue
            kw = ks + sgn * WIDTH
            if not np.isfinite(chain.price(right, kw, tau)[0]):
                continue
            es, ess = chain.price(right, ks, te)
            ew, ews = chain.price(right, kw, te)
            if not (np.isfinite(es) and np.isfinite(ew)) or q.sell(es, T_e) - q.buy(ew, T_e) <= 0:
                continue
            xs, xss = chain.exit_price(right, ks, EXIT_TAU, forward=EXIT_FORWARD)
            xw, xws = chain.exit_price(right, kw, EXIT_TAU, forward=EXIT_FORWARD)
            rows.append(dict(base, struct=sid, family="spread", strikes=f"{ks:g}/{kw:g}{right}", width=WIDTH,
                             leg1_entry=es, leg1_exit=xs, leg2_entry=ew, leg2_exit=xw,
                             entry_stale=max(ess, ews),
                             exit_stale=np.nan if not (np.isfinite(xss) and np.isfinite(xws)) else
                             (min(xss, xws) if min(xss, xws) < 0 else max(xss, xws))))
    return rows


def booked_pnl(c: pd.DataFrame, lam: float) -> np.ndarray:
    """P&L per candidate at execution level lam; missing exits booked at worst case."""
    long_ = (c["family"] == "long").values
    Te, Tx = c["T_entry"].values, c["T_exit"].values
    l1e, l1x, l2e, l2x = (c[k].values for k in ("leg1_entry", "leg1_exit", "leg2_entry", "leg2_exit"))
    cost = ES._buy(l1e, Te, lam)
    long_pnl = (ES._sell(l1x, Tx, lam) - cost) * 100 - 2 * R.COMMISSION
    long_worst = -cost * 100 - 2 * R.COMMISSION
    credit = ES._sell(l1e, Te, lam) - ES._buy(l2e, Te, lam)
    spr_pnl = (credit - (ES._buy(l1x, Tx, lam) - ES._sell(l2x, Tx, lam))) * 100 - 4 * R.COMMISSION
    spr_worst = (credit - c["width"].values) * 100 - 4 * R.COMMISSION
    pnl = np.where(long_, long_pnl, spr_pnl)
    return np.where(np.isfinite(pnl), pnl, np.where(long_, long_worst, spr_worst))


def build(bars_path="data/SPY_1min_sip.parquet", option_dir="data/SPY_0dte_option_bars") -> pd.DataFrame:
    hold = pd.Timestamp(F.HOLDOUT_START, tz=F.NY)
    bars = F.prepare_bars(pd.read_parquet(bars_path)[["open", "high", "low", "close", "volume"]])
    bars = bars[bars["session"] < hold]
    closes = {s: g.set_index("tau")["close"] for s, g in bars.groupby("session")}
    lens = bars.groupby("session")["session_len"].first()
    rows = []
    for p in sorted(Path(option_dir).glob("*.parquet")):
        session = pd.Timestamp(p.stem, tz=F.NY)
        if session >= hold or session not in closes or lens[session] != FULL_SESSION:
            continue
        cl = closes[session].reindex(range(FULL_SESSION)).values
        chain = R.DayChain(pd.read_parquet(p), session, FULL_SESSION)
        if chain.px:
            rows += build_day(chain, cl, session)
    c = pd.DataFrame(rows)
    assert (c["session"] < hold).all(), "holdout session in last-hour candidates"
    return c


def evaluate(c: pd.DataFrame):
    rng = np.random.default_rng(20260922)
    sessions = np.array(sorted(c["session"].unique()))
    median_session = sessions[len(sessions) // 2]
    grid, verdict = [], []
    for sid in STRUCTS:
        g = c[c["struct"] == sid]
        means = []
        for lam in ES.LAMBDAS:
            p = pd.Series(booked_pnl(g, lam), index=g.index)
            st = ES.session_stats(p, g["session"], rng)
            means.append(st["mean"])
            grid.append(dict(struct=sid, lam=lam, **st))
        p25 = pd.Series(booked_pnl(g, LAM_PRIMARY), index=g.index)
        st25 = next(r for r in grid if r["struct"] == sid and r["lam"] == LAM_PRIMARY)
        sm = p25.groupby(g["session"]).mean()
        h1, h2 = sm[sm.index < median_session].mean(), sm[sm.index >= median_session].mean()
        missing = g["leg1_exit"].isna() | (g["family"].eq("spread") & g["leg2_exit"].isna())
        lstar = ES.break_even(ES.LAMBDAS, means)
        try:
            lstar_ok = float(lstar.split()[-1]) >= LAMBDA_STAR_MIN if not lstar.startswith("none") else False
        except ValueError:
            lstar_ok = False
        checks = {"1 mean>0 & t>=2.50": st25["mean"] > 0 and st25["t"] >= T_BONFERRONI,
                  "2 CI above 0": st25["ci_lo"] > 0,
                  "3 lambda*>=0.40": lstar_ok,
                  "4 both halves >0": h1 > 0 and h2 > 0,
                  "5 missing<=2%": missing.mean() <= MAX_MISSING}
        verdict.append(dict(struct=sid, candidates=len(g), sessions=g["session"].nunique(),
                            mean_025=st25["mean"], t_025=st25["t"], ci=f"[{st25['ci_lo']:.2f}, {st25['ci_hi']:.2f}]",
                            lambda_star=lstar, half1=h1, half2=h2, missing=missing.mean(),
                            **{k: "PASS" if v else "fail" for k, v in checks.items()}, PASSED=all(checks.values())))
    return pd.DataFrame(grid), pd.DataFrame(verdict)


def main():
    pd.set_option("display.width", 240)
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if sha != PREREG_SHA256:
        raise SystemExit(f"pre-registration changed (sha {sha}) -- test invalid, refusing to run")
    print(f"pre-registration {PREREG.name} sha256 {sha}: MATCHES the frozen version")
    c = build()
    print(f"DEVELOPMENT ONLY: {c.session.nunique()} full sessions {c.session.min().date()} .. {c.session.max().date()}; "
          f"holdout ({F.HOLDOUT_START} on) not loaded; {len(c):,} candidates\n")

    grid, verdict = evaluate(c)
    print("PRE-REGISTERED PASS RULE (lambda = 0.25, session-level)")
    print(verdict.round(3).to_string(index=False))
    passed = verdict[verdict["PASSED"]]
    if passed.empty:
        decision = "NO structure passes -> last-hour hypothesis REJECTED; no variants; holdout stays unspent."
    else:
        best = passed.sort_values("t_025", ascending=False).iloc[0]["struct"]
        decision = (f"{len(passed)} structure(s) pass -> '{best}' is the single frozen holdout candidate. "
                    "Opening the holdout remains a separate explicit decision.")
    print(f"\nDECISION: {decision}")

    print("\nDESCRIPTIVE ONLY (not part of the pass rule)")
    print("\n  lambda grid, session-level:")
    print(grid.pivot(index="struct", columns="lam", values="mean").round(2).to_string())
    print("\n  by entry time at lambda 0.25 (session-level mean / t):")
    c["pnl025"] = booked_pnl(c, LAM_PRIMARY)
    rng = np.random.default_rng(1)
    rows = []
    for (sid, et), g in c.groupby(["struct", "entry_time"]):
        st = ES.session_stats(g["pnl025"], g["session"], rng)
        rows.append(dict(struct=sid, entry=et, n=len(g), mean=st["mean"], t=st["t"], win=(g["pnl025"] > 0).mean()))
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print(f"\n  exit fills after 15:55 (forward print): "
          f"{(c['exit_stale'] < 0).mean():.2%}; entry prints older than the fill minute: {(c['entry_stale'] > 0).mean():.2%}")
    c.to_parquet("data/last_hour_candidates.parquet")


if __name__ == "__main__":
    main()
