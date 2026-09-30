"""
Data audit 1: are option prints and SPY bars aligned in time?

Put-call parity for a 0DTE pair at the same strike K and minute:
    C - P + K  ~=  S          (r ~ 0 intraday, no dividend inside the day)
If the option clock were shifted (timezone / DST / bar-start vs bar-end),
the parity-implied S would match SPY at some lag other than 0. We measure the
median |S_parity - S_spy(lag)| for lags -5..+5 minutes, overall and split by
DST (EDT vs EST) sessions. Development sessions only.
"""
import numpy as np
import pandas as pd

import option_returns as R
import spy0dte_framework as F

CHECK_MINUTES = (15, 60, 120, 180, 240, 300, 360)
LAGS = range(-5, 6)


def main():
    hold = pd.Timestamp(F.HOLDOUT_START, tz=F.NY)
    bars = F.prepare_bars(pd.read_parquet("data/SPY_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
    bars = bars[bars["session"] < hold]
    closes = {s: g.set_index("tau")["close"] for s, g in bars.groupby("session")}
    rows = []
    for p in sorted(__import__("pathlib").Path("data/SPY_0dte_option_bars").glob("*.parquet")):
        s = pd.Timestamp(p.stem, tz=F.NY)
        if s >= hold or s not in closes:
            continue
        n = int(bars.loc[bars["session"] == s, "session_len"].iloc[0])
        ch = R.DayChain(pd.read_parquet(p), s, n)
        if not ch.px:
            continue
        cl = closes[s].reindex(range(n)).values
        for m in CHECK_MINUTES:
            if m >= n or not np.isfinite(cl[m]):
                continue
            ks = [k for k in ch.strikes.get("C", []) if ("P", k) in ch.px]
            ks = sorted(ks, key=lambda k: abs(k - cl[m]))[:3]
            for k in ks:
                c, p_ = ch.px[("C", k)][m], ch.px[("P", k)][m]
                if np.isfinite(c) and np.isfinite(p_):
                    sp = c - p_ + k
                    rec = dict(session=s, minute=m, edt=bool(s.utcoffset() == pd.Timedelta(hours=-4)))
                    for lag in LAGS:
                        j = m + lag
                        rec[lag] = abs(sp - cl[j]) if 0 <= j < n and np.isfinite(cl[j]) else np.nan
                    rows.append(rec)
                    break
    d = pd.DataFrame(rows)
    print(f"parity checks: {len(d):,} (sessions {d.session.nunique()})")
    out = pd.DataFrame({"all": d[list(LAGS)].median(), "EDT": d[d.edt][list(LAGS)].median(),
                        "EST": d[~d.edt][list(LAGS)].median()})
    out.index.name = "lag_min"
    print("median |C - P + K - SPY(lag)| in $ -- the minimum should be at lag 0")
    print(out.round(4).to_string())
    worst = d.groupby("session")[0].median().sort_values(ascending=False).head(5)
    print("\nsessions with the largest lag-0 parity error ($):")
    print(worst.round(3).to_string())


if __name__ == "__main__":
    main()
