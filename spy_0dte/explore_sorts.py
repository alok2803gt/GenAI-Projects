"""
Exploration (hypothesis GENERATION, development data only).

Does real option P&L vary with any single observable variable? For each
structure group x feature:
    quintile edges are set on the FIRST half of development sessions only,
    then applied to BOTH halves; we report Q5 - Q1 mean P&L in each half with
    a session-clustered t. A candidate pattern must have the same sign in both
    halves and |t| >= 2 in the second (out-of-sample) half.
With ~100 tests, about 100 * 5% / 2 = 2.5 such "hits" are expected by chance
alone -- printed alongside, so noise is not mistaken for signal. Anything
found here is only a hypothesis: it must be pre-registered and tested on data
it was not found in (the holdout) before it means anything.
"""
import numpy as np
import pandas as pd

import option_returns as R

FOMC = pd.to_datetime(["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18",
                       "2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
                       "2025-07-30", "2025-09-17"])
GROUPS = {"short put spread": lambda c: c["struct"].str.startswith("spread_P"),
          "short call spread": lambda c: c["struct"].str.startswith("spread_C"),
          "long call": lambda c: c["struct"].str.startswith("long_C"),
          "long put": lambda c: c["struct"].str.startswith("long_P")}


def clustered_t(x: pd.Series, sessions: pd.Series) -> float:
    m, se = R.cluster_mean_se(x, sessions)
    return m / se if se and np.isfinite(se) and se > 0 else np.nan


def diff_t(g: pd.DataFrame) -> tuple:
    """Q5 - Q1 mean difference and a session-clustered t (sessions resampled as units)."""
    hi, lo = g[g.q == 4], g[g.q == 0]
    if len(hi) < 50 or len(lo) < 50:
        return np.nan, np.nan
    d = hi["pnl"].mean() - lo["pnl"].mean()
    _, se_h = R.cluster_mean_se(hi["pnl"], hi["session"])
    _, se_l = R.cluster_mean_se(lo["pnl"], lo["session"])
    return d, d / np.sqrt(se_h ** 2 + se_l ** 2)


def main():
    pd.set_option("display.width", 220)
    c = pd.read_parquet("data/optret_candidates.parquet")
    c = c[c["pnl"].notna()].copy()
    day = c["session"].dt.tz_localize(None).dt.normalize()
    c["dow"] = c["session"].dt.dayofweek.astype(float)
    c["fomc_day"] = day.isin(FOMC).astype(float)
    feats = R.FEATURES + ["dow"]
    sessions = np.array(sorted(c["session"].unique()))
    mid = sessions[len(sessions) // 2]
    c["half"] = np.where(c["session"] < mid, 1, 2)

    rows = []
    for gname, sel in GROUPS.items():
        g = c[sel(c)]
        for f in feats:
            x = g[f]
            h1 = g[(g.half == 1) & x.notna()]
            if h1[f].nunique() < 5:
                continue
            edges = np.unique(np.quantile(h1[f], [0.2, 0.4, 0.6, 0.8]))
            gg = g[x.notna()].assign(q=lambda d: np.searchsorted(edges, d[f], side="right"))
            if gg["q"].max() < 4:
                continue
            d1, t1 = diff_t(gg[gg.half == 1])
            d2, t2 = diff_t(gg[gg.half == 2])
            rows.append(dict(group=gname, feature=f, q5_minus_q1_H1=d1, t_H1=t1, q5_minus_q1_H2=d2, t_H2=t2,
                             hit=bool(np.sign(d1) == np.sign(d2) and abs(t2) >= 2)))
    r = pd.DataFrame(rows)
    n = len(r)
    print(f"{n} tests; expected chance hits (same sign & |t_H2|>=2) ~ {n * 0.05 / 2:.1f}; "
          f"observed hits: {int(r.hit.sum())}\n")
    print("strongest second-half (out-of-sample) effects:")
    print(r.reindex(r.t_H2.abs().sort_values(ascending=False).index).head(15).round(2).to_string(index=False))

    print("\nFOMC days vs other days (mean $/contract, session-clustered t of the difference):")
    for gname, sel in GROUPS.items():
        g = c[sel(c)]
        a, b = g[g.fomc_day == 1], g[g.fomc_day == 0]
        _, sa = R.cluster_mean_se(a["pnl"], a["session"])
        _, sb = R.cluster_mean_se(b["pnl"], b["session"])
        d = a["pnl"].mean() - b["pnl"].mean()
        print(f"   {gname:<18} FOMC {a['pnl'].mean():7.2f} ({a.session.nunique()} days)  other {b['pnl'].mean():6.2f}  "
              f"t {d / np.sqrt(sa ** 2 + sb ** 2):5.2f}")


if __name__ == "__main__":
    main()
