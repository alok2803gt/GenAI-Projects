"""
Exploration round 4 -- assumption audit for short-premium structures.
Appends to data/explore_ledger.csv (the Bonferroni bar counts every test so far).

Varies two of MY assumptions, not the strategy:
    exit        15:45 buy-back (as tested)   vs   hold to 16:00 expiry, settled at
                intrinsic value against SPY's real final close (no exit spread/commission)
    commission  $0.65 / contract              vs   $0
Execution lambda 0.25 at entry (and at a 15:45 exit). Conditions: all, plus the
two calm-regime themes from rounds 2-3 (VIX/VIX3M Q1, seasonal ATR Q1), with
quintile edges from the first half only.

Caveat for "expiry": SPY options are American and physically settled; a short
leg near the money at 16:00 carries after-hours assignment (pin) risk that
intrinsic settlement ignores.
"""
import numpy as np
import pandas as pd

import execution_sensitivity as ES
import explore_loop as X
import spy0dte_framework as F

LAM = 0.25


def intrinsic(right, k, S):
    return np.where(right == "C", np.maximum(S - k, 0.0), np.maximum(k - S, 0.0))


def spread_variants(c, close):
    s = c[c["family"] == "spread"].copy()
    right = s["strikes"].str[-1].values
    ks = s["strikes"].str[:-1].str.split("/").str[0].astype(float).values
    kw = s["strikes"].str[:-1].str.split("/").str[1].astype(float).values
    S = s["session"].map(close).values
    credit = ES._sell(s["leg1_entry"].values, s["T_entry"].values, LAM) - ES._buy(s["leg2_entry"].values, s["T_entry"].values, LAM)
    bb1545 = ES._buy(s["leg1_exit"].values, s["T_exit"].values, LAM) - ES._sell(s["leg2_exit"].values, s["T_exit"].values, LAM)
    bbexp = intrinsic(right, ks, S) - intrinsic(right, kw, S)
    s["credit"], s["bb1545"], s["bbexp"] = credit, bb1545, bbexp
    return s


def straddle_variants(c, close):
    L = c[c["struct"].isin(["long_C_d0.0_Heod", "long_P_d0.0_Heod"])].copy()
    L["right"] = L["struct"].str[5]
    key = ["session", "tau"]
    w = L.pivot_table(index=key, columns="right", values=["leg1_entry", "leg1_exit", "T_entry", "T_exit", "strikes"],
                      aggfunc="first")
    kc, kp = w[("strikes", "C")].str[:-1].astype(float), w[("strikes", "P")].str[:-1].astype(float)
    w = w[(kc == kp).values]
    k = w[("strikes", "C")].str[:-1].astype(float).values
    S = w.index.get_level_values("session").map(close).values
    f = lambda col, r: w[(col, r)].astype(float).values
    credit = ES._sell(f("leg1_entry", "C"), f("T_entry", "C"), LAM) + ES._sell(f("leg1_entry", "P"), f("T_entry", "P"), LAM)
    bb1545 = ES._buy(f("leg1_exit", "C"), f("T_exit", "C"), LAM) + ES._buy(f("leg1_exit", "P"), f("T_exit", "P"), LAM)
    base = L[L["right"] == "C"].set_index(key).reindex(w.index)
    out = base[["half", "vix_term", "f5_atr_seas"]].copy()
    out["credit"], out["bb1545"], out["bbexp"] = credit, bb1545, np.abs(S - k)
    return out.reset_index()


def pnl(d, exit_, comm, legs):
    bb = d["bb1545"] if exit_ == "15:45" else d["bbexp"]
    n_comm = legs * (2 if exit_ == "15:45" else 1)
    return (d["credit"] - bb) * 100 - n_comm * comm


def main():
    pd.set_option("display.width", 220)
    X.ledger[:] = pd.read_csv(X.LEDGER).to_dict("records")
    n0 = len(X.ledger)
    hold = pd.Timestamp(F.HOLDOUT_START, tz=F.NY)
    bars = F.prepare_bars(pd.read_parquet("data/SPY_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
    bars = bars[bars["session"] < hold]
    close = bars.groupby("session")["close"].last()           # SPY final close = settlement reference

    midday = pd.read_parquet("data/optret_candidates.parquet")
    opening = pd.read_parquet("data/explore_open_candidates.parquet")
    sessions = np.array(sorted(midday["session"].unique()))
    split = sessions[len(sessions) // 2]
    sets = {}
    for win, c in (("midday", midday), ("open", opening)):
        c = X.add_halves_and_vix(c, split)
        sp = spread_variants(c, close)
        for st in sorted(sp["struct"].unique()):
            sets[f"{win} {st}"] = (sp[sp["struct"] == st], 2)
        for dist in ("0.5", "1.0"):
            p = sp[sp["struct"] == f"spread_P_d{dist}"].set_index(["session", "tau"])
            k = sp[sp["struct"] == f"spread_C_d{dist}"].set_index(["session", "tau"])
            j = p.join(k[["credit", "bb1545", "bbexp"]], rsuffix="_c", how="inner")
            j["credit"], j["bb1545"], j["bbexp"] = j["credit"] + j["credit_c"], j["bb1545"] + j["bb1545_c"], j["bbexp"] + j["bbexp_c"]
            sets[f"{win} iron condor d{dist}"] = (j.reset_index(), 4)
        sets[f"{win} short straddle"] = (straddle_variants(c, close), 2)

    for gname, (d, legs) in sets.items():
        for exit_ in ("15:45", "expiry"):
            for comm in (0.65, 0.0):
                v = d.assign(pnl=pnl(d, exit_, comm, legs)).dropna(subset=["pnl"])
                tag = f"{gname} | exit {exit_} | comm {comm}"
                X.test_rule(4, tag, "all", v)
                X.quintile_rules(4, tag, v, ["vix_term", "f5_atr_seas"])
    X.ledger[:] = [r for r in X.ledger if not (r["round"] == 4 and "Q5" in r["condition"])]   # calm side only

    L = pd.DataFrame(X.ledger)
    cur = L.iloc[n0:]
    b = X.bar(len(L))
    print(f"round 4: {len(cur)} tests, {len(L)} cumulative -> breakthrough bar t_H2 >= {b:.2f}\n")
    uncond = cur[cur.condition == "all"].copy()
    uncond[["struct", "exit", "comm"]] = uncond["group"].str.split(r" \| ", expand=True)
    print("UNCONDITIONAL, lambda 0.25 -- mean $/contract per session (t) by half")
    uncond["H1"] = uncond.apply(lambda r: f"{r.mean_H1:6.2f} ({r.t_H1:5.2f})", axis=1)
    uncond["H2"] = uncond.apply(lambda r: f"{r.mean_H2:6.2f} ({r.t_H2:5.2f})", axis=1)
    print(uncond.pivot_table(index="struct", columns=["exit", "comm"], values="H2", aggfunc="first").to_string())
    print("\n(first half, same layout)")
    print(uncond.pivot_table(index="struct", columns=["exit", "comm"], values="H1", aggfunc="first").to_string())
    both = cur[(cur.mean_H1 > 0) & (cur.mean_H2 > 0)]
    brk = both[(both.t_H1 >= 1.5) & (both.t_H2 >= b)]
    print(f"\npositive both halves: {len(both)}; watchlist t_H2>=2: {int((both.t_H2 >= 2).sum())}; BREAKTHROUGHS: {len(brk)}")
    print(both.sort_values("t_H2", ascending=False).head(10)[["group", "condition", "mean_H1", "t_H1", "mean_H2", "t_H2",
                                                              "sessions_H2"]].round(2).to_string(index=False))
    L.to_csv(X.LEDGER, index=False)


if __name__ == "__main__":
    main()
