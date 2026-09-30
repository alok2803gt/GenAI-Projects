"""
Exploration loop -- hypothesis GENERATION on development data only.

Every test is written to a ledger. A "rule" = (structure group, condition),
traded at lambda = 0.25. Each rule is measured separately in the first and
second half of the development sessions; condition cut-offs (quintiles) are
set on the FIRST half only, so the second half is out-of-sample for them.

    breakthrough : mean > 0 in both halves, t_H1 >= 1.5, and second-half
                   t >= Bonferroni bar over ALL tests run so far (two-sided 5%)
    watchlist    : mean > 0 in both halves and t_H2 >= 2 (not significant
                   after correction -- noted, never acted on)
A breakthrough is still only a hypothesis: it needs one pre-registered test
on the untouched holdout. Inference unit = session.

Rounds
    1  opening window: entries 09:35-09:55 (all 16 single structures)
    2  volatility structures: long/short straddle, short iron fly, iron condor
       (built from the same real legs, midday and opening windows)
    3  VIX regime: prior-day VIX level, VIX9D/VIX, VIX/VIX3M, VIX change
       as conditions on every structure group
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import execution_sensitivity as ES
import option_returns as R
import spy0dte_framework as F

OPEN_TAUS = (5, 10, 15, 20, 25)
LEDGER = Path("data/explore_ledger.csv")
LAM = 0.25
ledger: list = []


# ------------------------------------------------------------------ statistics
def half_stats(pnl: pd.Series, sessions: pd.Series) -> tuple:
    m = pnl.groupby(sessions.values).mean().dropna()
    if len(m) < 20:
        return np.nan, np.nan, len(m)
    return m.mean(), m.mean() / (m.std(ddof=1) / math.sqrt(len(m))), len(m)


def bar(n_tests: int) -> float:
    return float(norm.ppf(1 - 0.025 / max(n_tests, 1)))


def test_rule(rnd, group, cond, df, mask=None):
    d = df if mask is None else df[mask]
    r = dict(round=rnd, group=group, condition=cond)
    for h in (1, 2):
        x = d[d["half"] == h]
        r[f"mean_H{h}"], r[f"t_H{h}"], r[f"sessions_H{h}"] = half_stats(x["pnl"], x["session"])
    ledger.append(r)


def quintile_rules(rnd, group, df, features):
    h1 = df[df["half"] == 1]
    for f in features:
        v = h1[f].dropna()
        if v.nunique() < 5:
            continue
        lo, hi = np.quantile(v, [0.2, 0.8])
        test_rule(rnd, group, f"{f} <= {lo:.4g} (Q1)", df, df[f] <= lo)
        test_rule(rnd, group, f"{f} >= {hi:.4g} (Q5)", df, df[f] >= hi)


def report(rnd):
    L = pd.DataFrame(ledger)
    n = len(L)
    b = bar(n)
    cur = L[L["round"] == rnd]
    both = cur[(cur.mean_H1 > 0) & (cur.mean_H2 > 0)]
    brk = both[(both.t_H1 >= 1.5) & (both.t_H2 >= b)]
    watch = both[both.t_H2 >= 2]
    print(f"\n=== ROUND {rnd}: {len(cur)} tests this round, {n} cumulative -> breakthrough bar t_H2 >= {b:.2f} ===")
    print(f"    positive in both halves: {len(both)}; watchlist (t_H2>=2): {len(watch)}; BREAKTHROUGHS: {len(brk)}")
    cols = ["group", "condition", "mean_H1", "t_H1", "mean_H2", "t_H2", "sessions_H2"]
    top = both.sort_values("t_H2", ascending=False).head(8)
    if len(top):
        print(top[cols].round(2).to_string(index=False))
    return brk


# ------------------------------------------------------------------ data
def add_halves_and_vix(c: pd.DataFrame, split) -> pd.DataFrame:
    c = c.copy()
    c["half"] = np.where(c["session"] < split, 1, 2)
    vix = {}
    for s in ("VIX", "VIX9D", "VIX3M"):
        v = pd.read_csv(f"data/{s}_History.csv", parse_dates=["DATE"]).set_index("DATE")["CLOSE"]
        vix[s] = v
    v = pd.DataFrame(vix).dropna().sort_index()
    prev = v.shift(1)                                         # prior trading day's close only (causal)
    feat = pd.DataFrame({"vix_prev": prev["VIX"], "vix9d_ratio": prev["VIX9D"] / prev["VIX"],
                         "vix_term": prev["VIX"] / prev["VIX3M"], "vix_chg": np.log(prev["VIX"] / v["VIX"].shift(2))})
    day = c["session"].dt.tz_localize(None).dt.normalize()
    return c.join(feat.reindex(day.values).reset_index(drop=True).set_index(c.index))


def combos(c: pd.DataFrame, window: str) -> pd.DataFrame:
    """Straddles / iron flies / iron condors from the same real legs, lambda 0.25."""
    L = c[c["family"] == "long"].copy()
    L["H"] = L["struct"].str.extract(r"_H(\w+)$")[0]
    L["leg"] = L["struct"].str.extract(r"^long_([CP]_d[\d.]+)")[0]
    key = ["session", "tau", "H"]
    wide = L.pivot_table(index=key, columns="leg",
                         values=["leg1_entry", "leg1_exit", "T_entry", "T_exit", "strikes"], aggfunc="first")
    base_cols = ["half", "atm_iv", "iv_rv", "skew", "atm_iv_chg15", "vix_prev", "vix9d_ratio", "vix_term", "vix_chg"] + \
        [f for f in R.UNDERLYING_FEATURES if f != "tau"]
    base = L[L["leg"] == "C_d0.0"].drop_duplicates(key).set_index(key)[base_cols]
    out = []

    def val(leg, qty):
        e, x = wide[("leg1_entry", leg)].astype(float).values, wide[("leg1_exit", leg)].astype(float).values
        Te, Tx = wide[("T_entry", leg)].astype(float).values, wide[("T_exit", leg)].astype(float).values
        if qty > 0:
            return ES._sell(x, Tx, LAM) - ES._buy(e, Te, LAM)
        return ES._sell(e, Te, LAM) - ES._buy(x, Tx, LAM)

    comm = R.COMMISSION
    same_k = (wide[("strikes", "C_d0.0")].str[:-1] == wide[("strikes", "P_d0.0")].str[:-1]).values
    wings_ok = ((wide[("strikes", "C_d0.5")].str[:-1].astype(float) > wide[("strikes", "C_d0.0")].str[:-1].astype(float)) &
                (wide[("strikes", "P_d0.5")].str[:-1].astype(float) < wide[("strikes", "P_d0.0")].str[:-1].astype(float))).values
    specs = {"long straddle": ((val("C_d0.0", 1) + val("P_d0.0", 1)) * 100 - 4 * comm, same_k),
             "short straddle": ((val("C_d0.0", -1) + val("P_d0.0", -1)) * 100 - 4 * comm, same_k),
             "short iron fly": ((val("C_d0.0", -1) + val("P_d0.0", -1) + val("C_d0.5", 1) + val("P_d0.5", 1)) * 100
                                - 8 * comm, same_k & wings_ok)}
    for name, (pnl, ok) in specs.items():
        d = base.reindex(wide.index).copy()
        d["pnl"], d["group"] = pnl, f"{window} {name}"
        d = d[ok & np.isfinite(pnl)].reset_index()
        d["group"] = d["group"] + " H" + d["H"]
        out.append(d)
    S = c[c["family"] == "spread"].copy()
    for dist in ("0.5", "1.0"):
        p = S[S["struct"] == f"spread_P_d{dist}"].set_index(["session", "tau"])
        k = S[S["struct"] == f"spread_C_d{dist}"].set_index(["session", "tau"])
        j = p.join(k[["pnl"]], rsuffix="_c", how="inner")
        j = j[j["pnl"].notna() & j["pnl_c"].notna()]
        d = j[base_cols].copy()
        d["pnl"], d["group"] = j["pnl"] + j["pnl_c"], f"{window} iron condor d{dist}"
        out.append(d.reset_index())
    return pd.concat(out, ignore_index=True)


def single_groups(c: pd.DataFrame, window: str) -> dict:
    g = {f"{window} {s}": c[c["struct"] == s] for s in sorted(c["struct"].unique())}
    for name, pre in (("short put spread", "spread_P"), ("short call spread", "spread_C"),
                      ("long call", "long_C"), ("long put", "long_P")):
        g[f"{window} {name} (all)"] = c[c["struct"].str.startswith(pre)]
    return g


def main():
    pd.set_option("display.width", 220)
    hold = pd.Timestamp(F.HOLDOUT_START, tz=F.NY)
    mid = R  # noqa
    midday = pd.read_parquet("data/optret_candidates.parquet")
    cache = Path("data/explore_open_candidates.parquet")
    if cache.exists():
        opening = pd.read_parquet(cache)
    else:
        bars = F.prepare_bars(pd.read_parquet("data/SPY_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
        feats = R.add_underlying_features(F.FeatureBuilder().transform(bars))
        opening = R.build_candidates(feats, Path("data/SPY_0dte_option_bars"), F.HOLDOUT_START, taus=OPEN_TAUS)
        opening.to_parquet(cache)
    for d in (midday, opening):
        assert (d["session"] < hold).all(), "holdout in exploration data"
    sessions = np.array(sorted(midday["session"].unique()))
    split = sessions[len(sessions) // 2]
    midday = add_halves_and_vix(midday[midday["pnl"].notna()], split)
    opening = add_halves_and_vix(opening[opening["pnl"].notna()], split)
    print(f"development only; halves split at {split.date()}; holdout ({F.HOLDOUT_START} on) never loaded")
    print(f"opening-window candidates: {len(opening):,} over {opening.session.nunique()} sessions "
          f"(entries {', '.join(f'09:{30 + t + 1:02d}' for t in OPEN_TAUS)})")

    opt_feats = ["atm_iv", "atm_iv_chg15", "iv_rv", "skew", "leg_iv_rel", "prem_rel", "leg_vol15", "leg_mom15"]
    open_feats = R.UNDERLYING_FEATURES + opt_feats
    breakthroughs = []

    # ROUND 1 -- opening window
    for gname, g in single_groups(opening, "open").items():
        test_rule(1, gname, "all", g)
        if gname.endswith("(all)"):
            quintile_rules(1, gname, g, open_feats)
    breakthroughs.append(report(1))

    # ROUND 2 -- volatility structures
    combo_sets = pd.concat([combos(midday, "midday"), combos(opening, "open")], ignore_index=True)
    combo_feats = ["atm_iv", "iv_rv", "skew", "atm_iv_chg15"] + R.UNDERLYING_FEATURES
    for gname, g in combo_sets.groupby("group"):
        test_rule(2, gname, "all", g)
        quintile_rules(2, gname, g, combo_feats)
    breakthroughs.append(report(2))

    # ROUND 3 -- VIX regime on every group
    vix_feats = ["vix_prev", "vix9d_ratio", "vix_term", "vix_chg"]
    groups = {**single_groups(midday, "midday"), **single_groups(opening, "open")}
    groups = {k: v for k, v in groups.items() if k.endswith("(all)")}
    groups.update({k: v for k, v in combo_sets.groupby("group")})
    for gname, g in groups.items():
        quintile_rules(3, gname, g, vix_feats)
    breakthroughs.append(report(3))

    L = pd.DataFrame(ledger)
    LEDGER.parent.mkdir(exist_ok=True)
    L.to_csv(LEDGER, index=False)
    final_bar = bar(len(L))
    both = L[(L.mean_H1 > 0) & (L.mean_H2 > 0)]
    final = both[(both.t_H1 >= 1.5) & (both.t_H2 >= final_bar)]
    exp_watch = len(L) * norm.sf(2) / 2
    print(f"\n=== FINAL: {len(L)} tests; bar t_H2 >= {final_bar:.2f}; breakthroughs at the final bar: {len(final)}; "
          f"watchlist {int((both.t_H2 >= 2).sum())} (chance alone ~{exp_watch:.0f}) ===")
    if len(final):
        print(final.round(2).to_string(index=False))
    print(f"ledger: {LEDGER}")


if __name__ == "__main__":
    main()
