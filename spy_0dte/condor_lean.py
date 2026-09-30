"""
Lean builder for the frozen expiry iron condor -- needs only per-minute
underlying prices and the option chain (no volume/feature bars), so it also
works for SPX, whose index bars we cannot download.

Mirrors option_returns.build_day's spread logic exactly (same decision bars,
expected-move unit, strike selection, the 0.5u-then-1.0u de-duplication,
wing print check, fill minute, credit filter); validated against the frozen
SPY candidates by validate_on_spy().
"""
import math

import numpy as np
import pandas as pd

import option_returns as R
import spy0dte_framework as F

Q = F.QuoteModel(lam=R.LAMBDA)


def parity_spot(chain: R.DayChain, n: int, stale: int = 2) -> np.ndarray:
    """Per-minute spot from put-call parity: median of C - P + K over the 3
    strikes with the smallest |C - P| that printed in that minute; carried at
    most `stale` minutes."""
    ks = sorted(k for (r, k) in chain.px if r == "C" and ("P", k) in chain.px)
    if not ks:
        return np.full(n, np.nan)
    C = np.vstack([chain.px[("C", k)] for k in ks])
    P = np.vstack([chain.px[("P", k)] for k in ks])
    K = np.array(ks)[:, None]
    impl = C - P + K
    gap = np.abs(C - P)
    out = np.full(n, np.nan)
    for m in range(n):
        ok = np.isfinite(impl[:, m])
        if ok.any():
            idx = np.where(ok)[0][np.argsort(gap[ok, m])[:3]]
            out[m] = float(np.median(impl[idx, m]))
    s = pd.Series(out).ffill(limit=stale)
    return s.values


def build_condors(chain: R.DayChain, S: np.ndarray, n: int, session, settle: float, wing: float,
                  ready=None) -> list:
    """Condors (spread_P_d1.0 + spread_C_d1.0) for one session. `ready[tau]` optional gate."""
    eod = n - R.EOD_BEFORE_CLOSE
    out = []
    for tau in R.DECISION_TAUS:
        if tau + 1 >= eod - 5 or not np.isfinite(S[tau]) or (ready is not None and not ready.get(tau, False)):
            continue
        s0, T = float(S[tau]), n - tau - 1
        iv0 = chain.atm_iv(tau, s0, T)
        if not (np.isfinite(iv0) and iv0 > 0):
            continue
        u = s0 * iv0 * math.sqrt(T / F.MIN_PER_YEAR)
        te, T_e = tau + 1, n - tau - 1
        legs = {}
        for right, sgn in (("P", -1), ("C", +1)):
            seen = set()
            for d in R.SPREAD_TARGETS:                        # 0.5 first, exactly like build_day
                ks = chain.nearest_printed(right, s0 + sgn * d * u, tau)
                if ks is None:
                    continue
                kw = ks + sgn * wing
                if ks in seen or not np.isfinite(chain.price(right, kw, tau)[0]):
                    continue
                seen.add(ks)
                es, _ = chain.price(right, ks, te)
                ew, _ = chain.price(right, kw, te)
                if not (np.isfinite(es) and np.isfinite(ew)):
                    continue
                credit = Q.sell(es, T_e) - Q.buy(ew, T_e)
                if credit <= 0:
                    continue
                if d == 1.0:
                    intr = (lambda k: max(k - settle, 0.0)) if right == "P" else (lambda k: max(settle - k, 0.0))
                    legs[right] = dict(ks=ks, kw=kw, credit=credit, bb=intr(ks) - intr(kw))
        if "P" in legs and "C" in legs:
            pnl = (legs["P"]["credit"] - legs["P"]["bb"] + legs["C"]["credit"] - legs["C"]["bb"]) * 100 - 4 * R.COMMISSION
            out.append(dict(session=session, tau=tau, strikes=f"{legs['P']['ks']:g}/{legs['P']['kw']:g}P "
                                                                f"{legs['C']['ks']:g}/{legs['C']['kw']:g}C",
                            condor=pnl, put=(legs["P"]["credit"] - legs["P"]["bb"]) * 100 - 2 * R.COMMISSION,
                            call=(legs["C"]["credit"] - legs["C"]["bb"]) * 100 - 2 * R.COMMISSION, wing=wing))
    return out


def validate_on_spy(n_sessions: int = 60) -> None:
    """Lean builder vs frozen SPY condors on a sample of development sessions."""
    import explore_round4 as E4
    from pathlib import Path
    hold = pd.Timestamp(F.HOLDOUT_START, tz=F.NY)
    bars = F.prepare_bars(pd.read_parquet("data/SPY_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
    bars = bars[bars.session < hold]
    close = bars.groupby("session")["close"].last()
    c = pd.read_parquet("data/optret_candidates.parquet")
    sp = E4.spread_variants(c, close)
    sp["pnl_exp"] = (sp["credit"] - sp["bbexp"]) * 100 - 2 * R.COMMISSION
    P = sp[sp.struct == "spread_P_d1.0"].set_index(["session", "tau"])[["pnl_exp"]]
    C = sp[sp.struct == "spread_C_d1.0"].set_index(["session", "tau"])[["pnl_exp"]]
    frozen = P.join(C, rsuffix="_c", how="inner")
    frozen["condor"] = frozen["pnl_exp"] + frozen["pnl_exp_c"]
    feats = F.FeatureBuilder().transform(bars)
    sessions = sorted(frozen.index.get_level_values(0).unique())[:: max(1, len(frozen.index.get_level_values(0).unique()) // n_sessions)]
    rows = []
    for s in sessions:
        day = bars[bars.session == s].set_index("tau")
        n = int(day["session_len"].iloc[0])
        chain = R.DayChain(pd.read_parquet(Path("data/SPY_0dte_option_bars") / f"{s:%Y-%m-%d}.parquet"), s, n)
        ready = feats[feats.session == s].set_index("tau")["ready"].to_dict()
        rows += build_condors(chain, day["close"].reindex(range(n)).values, n, s, float(close[s]), 3.0, ready)
    lean = pd.DataFrame(rows).set_index(["session", "tau"])
    f = frozen.loc[frozen.index.get_level_values(0).isin(sessions)]
    both = lean.join(f[["condor"]], rsuffix="_frozen", how="outer")
    match = np.isclose(both["condor"], both["condor_frozen"], atol=0.01)
    print(f"lean vs frozen SPY condors on {len(sessions)} sessions: {int(match.sum())}/{len(both)} identical "
          f"(lean-only {both['condor_frozen'].isna().sum()}, frozen-only {both['condor'].isna().sum()})")
    if not match.all():
        print(both[~match].head(10))


if __name__ == "__main__":
    validate_on_spy()
