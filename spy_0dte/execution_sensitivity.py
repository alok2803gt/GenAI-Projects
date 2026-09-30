"""
Execution-sensitivity report for the frozen option-return experiment.
Diagnostic only -- NOT another optimization round.

Frozen: candidates, features, structures, decision times, models, thresholds
and trade selection are exactly those of option_returns.py (predictions are
reproduced and checked against the saved frozen copy). Only the execution
cost lambda changes, applied to the SAME selected trades:
    lambda in {0, .05, .10, .15, .20, .25, .50}      (0 = frictionless upper bound)

Pre-registered reading (written before the results were seen):
    lambda=0 spread mean ~0 or negative               -> stop; quote data cannot rescue it
    lambda=0 small positive but not significant       -> stop strategy development
    monotonic EV deciles AND model beats constant     -> execution worth investigating
    constant spread positive, ML no better            -> unconditional premium, not ML edge
    positive at 0, gone by small lambda (lambda* low) -> execution-bound, hard to capture
Inference is session-level: overlapping candidates in one session are one
observation (session means, session-clustered SE, whole-session bootstrap).
The holdout is never loaded.

    ./.venv/bin/python execution_sensitivity.py
"""
import math

import numpy as np
import pandas as pd

import option_returns as R
import spy0dte_framework as F

LAMBDAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50)
Q = F.QuoteModel()                      # width model only; lambda is varied here
N_BOOT = 2000


def _width(mid, T):
    w = np.maximum(Q.min_width, Q.width_pct * np.maximum(mid, 0.0))
    return w * np.where(T <= Q.late_minutes, Q.late_mult, 1.0)


def _buy(mid, T, lam):
    return mid + lam * _width(mid, T)


def _sell(mid, T, lam):
    return np.maximum(0.0, mid - lam * _width(mid, T))


def pnl_at(c: pd.DataFrame, lam: float):
    """(pnl, worst) per candidate at execution level lam -- same formulas as option_returns."""
    long_ = (c["family"] == "long").values
    Te, Tx = c["T_entry"].values, c["T_exit"].values
    l1e, l1x, l2e, l2x = (c[k].values for k in ("leg1_entry", "leg1_exit", "leg2_entry", "leg2_exit"))
    cost = _buy(l1e, Te, lam)
    long_pnl = (_sell(l1x, Tx, lam) - cost) * 100 - 2 * R.COMMISSION
    long_worst = -cost * 100 - 2 * R.COMMISSION
    credit = _sell(l1e, Te, lam) - _buy(l2e, Te, lam)
    spr_pnl = (credit - (_buy(l1x, Tx, lam) - _sell(l2x, Tx, lam))) * 100 - 4 * R.COMMISSION
    spr_worst = (credit - R.SPREAD_WIDTH) * 100 - 4 * R.COMMISSION
    pnl = np.where(long_, long_pnl, spr_pnl)
    worst = np.where(long_, long_worst, spr_worst)
    return pnl, np.where(np.isfinite(pnl), pnl, worst)


def session_stats(x: pd.Series, sessions: pd.Series, rng) -> dict:
    """Session means -> mean, SE, t, % positive sessions, whole-session bootstrap 95% CI."""
    m = x.groupby(sessions.values).mean().dropna()
    n = len(m)
    se = m.std(ddof=1) / math.sqrt(n)
    boot = m.values[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    return dict(sessions=n, mean=m.mean(), se=se, t=m.mean() / se, pos_sessions=(m > 0).mean(),
                ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5))


def break_even(lams, means):
    """lambda* where mean P&L crosses zero (linear interpolation on the grid)."""
    if means[0] <= 0:
        return "none (<= 0 at lambda 0)"
    for (l0, m0), (l1, m1) in zip(zip(lams, means), zip(lams[1:], means[1:])):
        if m1 <= 0 < m0:
            return f"{l0 + (l1 - l0) * m0 / (m0 - m1):.3f}"
    return f"> {lams[-1]}"


def selections(oos: pd.DataFrame) -> dict:
    """Frozen trade selections (index sets), independent of lambda."""
    out = {}
    for model in ("constant", "ridge", "lgbm"):
        g = oos[oos[f"ev_{model}"].notna()]
        best = g.loc[g.groupby(["session", "tau"])[f"ev_{model}"].idxmax()]
        for th in (0.0, 10.0):
            b = best[best[f"ev_{model}"] > th]
            out[(model, th, "every bar")] = b.index
            out[(model, th, "1/day")] = b.sort_values("tau").groupby("session").head(1).index
    return out


def main():
    pd.set_option("display.width", 220)
    rng = np.random.default_rng(20260922)
    c = pd.read_parquet("data/optret_candidates.parquet")
    frozen_c = pd.read_parquet("data/optret_candidates_frozen_v1.parquet")
    assert (c["session"] < pd.Timestamp(F.HOLDOUT_START, tz=F.NY)).all(), "holdout in candidates"
    assert len(c) == len(frozen_c) and np.allclose(c["pnl"], frozen_c["pnl"], equal_nan=True), \
        "rebuilt candidates differ from the frozen set"

    pnl25, _ = pnl_at(c, 0.25)
    assert np.allclose(pnl25, c["pnl"], equal_nan=True), "lambda recomputation does not reproduce stored P&L"

    oos, _ = R.walk_forward(c)
    oos = oos[oos["ev_constant"].notna()]
    frozen = pd.read_parquet("data/optret_oos_predictions_frozen_v1.parquet")
    for m in ("constant", "ridge", "lgbm"):
        assert np.allclose(oos[f"ev_{m}"].values, frozen[f"ev_{m}"].values), f"{m} predictions not reproduced"
    print("FROZEN CHECK: candidates, lambda=0.25 P&L and all OOS predictions reproduce the frozen run exactly.")
    print(f"development sessions {c.session.min().date()} .. {c.session.max().date()}; holdout "
          f"({F.HOLDOUT_START} on) never loaded. Inference unit = session.\n")

    # A -------------------------------------------------------------------------
    print("A) UNCONDITIONAL, every candidate of a structure taken (all development sessions)")
    print("   mean = average of per-session mean P&L ($/contract); t and CI are session-level")
    rows, curves = [], {}
    groups = [(s, c["struct"] == s) for s in sorted(c.loc[c.family == "spread", "struct"].unique())]
    groups += [("ALL spreads", c["family"] == "spread"), ("ALL longs", c["family"] == "long")]
    for lam in LAMBDAS:
        pnl, _ = pnl_at(c, lam)
        s = pd.Series(pnl, index=c.index)
        for name, mask in groups:
            st = session_stats(s[mask], c.loc[mask, "session"], rng)
            rows.append(dict(structure=name, lam=lam, **st))
            curves.setdefault(name, []).append(st["mean"])
    a = pd.DataFrame(rows)
    for name, _ in groups:
        print(f"\n   {name}   break-even lambda* = {break_even(LAMBDAS, curves[name])}")
        print(a[a.structure == name].drop(columns="structure").round(3).to_string(index=False))

    # B -------------------------------------------------------------------------
    print("\nB) FROZEN MODEL TRADE RULES on OOS sessions -- same trades at every lambda")
    print("   avg $/trade by lambda; t = daily-P&L t-stat (all OOS sessions, no-trade days = 0); missing exits at worst case")
    sel = selections(oos)
    all_sess = pd.Index(oos["session"].unique())
    rows = []
    for (model, th, rule), idx in sel.items():
        r = dict(model=model, threshold=th, rule=rule, trades=len(idx))
        for lam in LAMBDAS:
            _, booked = pnl_at(oos.loc[idx], lam)
            s = pd.Series(booked, index=idx)
            daily = s.groupby(oos.loc[idx, "session"]).sum().reindex(all_sess, fill_value=0.0)
            r[f"avg@{lam:g}"] = s.mean() if len(s) else np.nan
            if lam in (0.0, 0.25):
                r[f"t@{lam:g}"] = daily.mean() / (daily.std() / math.sqrt(len(daily))) if daily.std() > 0 else np.nan
        rows.append(r)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # C -------------------------------------------------------------------------
    print("\nC) PREDICTED-EV DECILES vs REALIZED at lambda=0 (predictions frozen; session-clustered se)")
    pnl0, _ = pnl_at(oos, 0.0)
    o0 = oos.assign(pnl=pnl0)
    for m in ("ridge", "lgbm"):
        t = R.decile_table(o0, m)
        for fam, g in t.groupby("family"):
            rho = g["decile"].corr(g["realized"], method="spearman")
            lo, hi = g.iloc[0], g.iloc[-1]
            print(f"   {m:<5} {fam:<6} rank corr {rho:+.2f} | decile 1: {lo.realized:+.2f} (se {lo.se:.2f}) | "
                  f"decile 10: {hi.realized:+.2f} (se {hi.se:.2f})")
    print("   (beating the constant model at lambda=0 would need retraining on lambda=0 P&L -- not done: frozen)")

    # D -------------------------------------------------------------------------
    print("\nD) TAIL ABLATION at lambda=0.25 (as trained): winsorized 1/99 target (frozen) vs RAW target")
    print("   robustness only -- neither version is selected afterwards")
    raw, _ = R.walk_forward(c, winsor=False)
    raw = raw[raw["ev_constant"].notna()]
    for label, o in (("winsorized", oos), ("raw", raw)):
        sc = R.scorecard(o)
        sc = sc[sc.model != "constant"][["family", "model", "spearman", "r2_vs_constant", "t_vs_constant"]]
        print(f"\n   {label} target")
        print(sc.round(4).to_string(index=False))
        rules = [R.trade_rule(o, m, th, one) for m in ("ridge", "lgbm") for th in (0.0, 10.0) for one in (False, True)]
        print(pd.DataFrame(rules)[["model", "threshold", "rule", "trades", "avg_usd", "total_usd", "t_daily"]]
              .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
