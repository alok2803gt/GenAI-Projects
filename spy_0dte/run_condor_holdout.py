"""
ONE-TIME holdout test of PREREG_condor_expiry.md. Verifies the pre-registration
and every frozen source file against PREREG_condor_expiry.sha256 before
touching holdout data, then applies the pre-registered PASS rule. The run is
appended to holdout_log.jsonl.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import explore_round4 as E4
import option_returns as R
import spy0dte_framework as F


def verify_frozen():
    for line in Path("PREREG_condor_expiry.sha256").read_text().splitlines():
        h, name = line.split()[:2]
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if actual != h:
            raise SystemExit(f"{name} changed since pre-registration -- test invalid, refusing to run")
    print("pre-registration and all frozen source files match their locked hashes")


def main():
    pd.set_option("display.width", 220)
    verify_frozen()
    hold = pd.Timestamp(F.HOLDOUT_START, tz=F.NY)
    bars = F.prepare_bars(pd.read_parquet("data/SPY_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
    feats = R.add_underlying_features(F.FeatureBuilder().transform(bars))
    close = bars.groupby("session")["close"].last()
    by_sess = {s: g.set_index("tau", drop=False).rename_axis("tau_idx") for s, g in feats.groupby("session")}
    q = F.QuoteModel(lam=R.LAMBDA)
    rows = []
    for p in sorted(Path("data/SPY_0dte_option_bars").glob("*.parquet")):
        s = pd.Timestamp(p.stem, tz=F.NY)
        if s < hold or s not in by_sess:
            continue
        day = by_sess[s]
        chain = R.DayChain(pd.read_parquet(p), s, int(day["session_len"].iloc[0]))
        if chain.px:
            rows += R.build_day(chain, day, q)
    c = pd.DataFrame(rows)
    assert (c["session"] >= hold).all()
    sp = E4.spread_variants(c, close)
    sp["pnl_exp"] = (sp["credit"] - sp["bbexp"]) * 100 - 2 * R.COMMISSION
    P = sp[sp.struct == "spread_P_d1.0"].set_index(["session", "tau"])
    C = sp[sp.struct == "spread_C_d1.0"].set_index(["session", "tau"])
    j = P[["pnl_exp", "leg1_entry", "leg2_entry", "T_entry", "bbexp"]].join(
        C[["pnl_exp", "leg1_entry", "leg2_entry", "T_entry", "bbexp"]], rsuffix="_c", how="inner")
    j["condor"] = j["pnl_exp"] + j["pnl_exp_c"]
    j = j.reset_index()

    day = j.groupby("session")["condor"].mean()
    n = len(day)
    t = day.mean() / (day.std(ddof=1) / np.sqrt(n))
    passed = bool(day.mean() > 0 and t >= 2.0)
    print(f"\nHOLDOUT: {n} sessions {day.index.min().date()} .. {day.index.max().date()}, {len(j):,} condors")
    print(f"PRIMARY (pre-registered): mean ${day.mean():.2f}/condor per session, t = {t:.2f}  ->  "
          f"{'PASS' if passed else 'FAIL'} (rule: mean > 0 and t >= 2.00)")

    print("\nDESCRIPTIVE ONLY")
    import execution_sensitivity as ES
    legs = [("leg1_entry", -1), ("leg2_entry", +1), ("leg1_entry_c", -1), ("leg2_entry_c", +1)]
    credit0 = sum(j[k].values * (1 if s < 0 else -1) for k, s in legs)
    widths = sum(ES._width(j[k].values, j["T_entry"].values) for k, _ in legs)
    bb = (j["bbexp"] + j["bbexp_c"]).values
    for lam in (0.0, 0.5, 1.0):
        d = pd.Series((credit0 - lam * widths - bb) * 100 - 4 * R.COMMISSION).groupby(j["session"].values).mean()
        print(f"   entry lambda {lam}: ${d.mean():.2f}, t {d.mean() / (d.std() / np.sqrt(len(d))):.2f}")
    side = lambda col: j.groupby("session")[col].mean()
    for name, col in (("put side", "pnl_exp"), ("call side", "pnl_exp_c")):
        d = side(col)
        print(f"   {name}: ${d.mean():.2f}, t {d.mean() / (d.std() / np.sqrt(len(d))):.2f}")
    mon = day.groupby(day.index.tz_localize(None).to_period("M")).sum()
    eq = day.cumsum()
    print(f"   winning days {(day > 0).mean():.0%}, positive months {(mon > 0).sum()}/{len(mon)}, "
          f"total ${eq.iloc[-1]:.0f} (1 condor/day avg), max drawdown ${(eq - eq.cummax()).min():.0f}")
    print(f"   worst days: {day.nsmallest(5).round(0).to_dict()}")
    print("   by month:", mon.round(0).to_dict())

    with open("holdout_log.jsonl", "a") as f:
        f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), test="PREREG_condor_expiry v1",
                                prereg_sha256=Path("PREREG_condor_expiry.sha256").read_text().split()[0],
                                sessions=n, mean=round(day.mean(), 4), t=round(t, 4), passed=passed)) + "\n")
    j.to_parquet("data/holdout_condor_results.parquet")
    print("\nlogged to holdout_log.jsonl; the holdout is now spent")


if __name__ == "__main__":
    main()
