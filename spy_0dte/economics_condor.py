"""Economics of the expiry iron condor (descriptive; development + spent holdout)."""
import numpy as np
import pandas as pd

import execution_sensitivity as ES
import explore_round4 as E4
import spy0dte_framework as F

COMM = 0.65
WIDTH = 3.0


def dev_condors():
    hold = pd.Timestamp(F.HOLDOUT_START, tz=F.NY)
    bars = F.prepare_bars(pd.read_parquet("data/SPY_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
    close = bars[bars.session < hold].groupby("session")["close"].last()
    sp = E4.spread_variants(pd.read_parquet("data/optret_candidates.parquet"), close)
    cols = ["leg1_entry", "leg2_entry", "T_entry", "bbexp"]
    P = sp[sp.struct == "spread_P_d1.0"].set_index(["session", "tau"])[cols]
    C = sp[sp.struct == "spread_C_d1.0"].set_index(["session", "tau"])[cols]
    return P.join(C, rsuffix="_c", how="inner").reset_index()


def economics(j, lam):
    T = j["T_entry"].values
    cr = (ES._sell(j.leg1_entry.values, T, lam) - ES._buy(j.leg2_entry.values, T, lam)
          + ES._sell(j.leg1_entry_c.values, T, lam) - ES._buy(j.leg2_entry_c.values, T, lam))
    d = j[["session", "tau"]].copy()
    d["credit"] = cr * 100
    d["pnl"] = (cr - j.bbexp.values - j.bbexp_c.values) * 100 - 4 * COMM
    d["margin"] = WIDTH * 100 - d["credit"]                  # defined risk: one side can lose
    return d


def summarize(d, label):
    daily = d.groupby("session")["pnl"].sum()
    margin_day = d.groupby("session")["margin"].sum()
    eq = daily.cumsum()
    dd = (eq - eq.cummax()).min()
    win, loss = d.pnl[d.pnl > 0], d.pnl[d.pnl <= 0]
    capital = margin_day.max()
    yrs = len(daily) / 252
    return dict(rule=label, condors=len(d), avg_credit=d.credit.mean(), avg_pnl=d.pnl.mean(),
                win=(d.pnl > 0).mean(), avg_win=win.mean(), avg_loss=loss.mean(),
                profit_factor=win.sum() / -loss.sum(), comm_per_condor=4 * COMM,
                day_mean=daily.mean(), worst_day=daily.min(), max_dd=dd,
                sharpe=daily.mean() / daily.std() * np.sqrt(252), total=daily.sum(),
                capital_margin=capital, annual_return_on_margin=daily.sum() / yrs / capital)


def main():
    pd.set_option("display.width", 250)
    sets = {"development": dev_condors(), "HOLDOUT": pd.read_parquet("data/holdout_condor_results.parquet")}
    for period, j in sets.items():
        rows = []
        for lam in (0.25, 0.5):
            d = economics(j, lam)
            for label, sel in (("10:00 only", d.tau == 30), ("12:00 only", d.tau == 150), ("all 19 entries", d.tau >= 0)):
                rows.append(dict(lam=lam, **summarize(d[sel], label)))
        r = pd.DataFrame(rows)
        print(f"\n===== {period}: {j.session.nunique()} sessions ({j.session.min().date()} .. {j.session.max().date()}), 1 contract per condor =====")
        print(r.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
