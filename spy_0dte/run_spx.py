"""
SPX (SPXW 0DTE, cash-settled at the official close):
  B  -- pre-registered test (PREREG_condor_gex_spx.md part B + recorded date deviation):
        frozen midday condor, 1.0u, wing round($3 x SPX/SPY) to $5, lambda 0.25.
  EXP -- exploratory: the hold-to-expiry family on SPX (descriptive; not a verdict),
        to see whether the late-day premium survives without after-hours exercise.
Spot per minute from put-call parity (validated on SPY: $0.05 median error);
settlement = official SPX close (Cboe history). Data: Polygon SPXW 1-min bars.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

import condor_lean as L
import expiry_family as X
import option_returns as R
import spy0dte_framework as F


def main():
    pd.set_option("display.width", 220)
    h = Path("PREREG_condor_gex_spx.sha256").read_text().split()[0]
    assert hashlib.sha256(Path("PREREG_condor_gex_spx.md").read_bytes()).hexdigest() == h, "pre-registration changed"
    spx = pd.read_csv("data/SPX_History.csv", parse_dates=["DATE"]).set_index("DATE")["SPX"]
    spy = F.prepare_bars(pd.read_parquet("data/SPY_1min_sip.parquet")[["open", "high", "low", "close", "volume"]])
    slen = spy.groupby("session")["session_len"].first()
    spy_open = spy[spy.tau == 0].set_index("session")["open"]
    condors, legs, spot_err = [], [], []
    for p in sorted(Path("data/SPX_0dte_option_bars").glob("*.parquet")):
        s = pd.Timestamp(p.stem, tz=F.NY)
        d = s.tz_localize(None)
        if s not in slen.index or d not in spx.index:
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        n = int(slen[s])
        chain = R.DayChain(df, s, n)
        if not chain.px:
            continue
        S = L.parity_spot(chain, n)
        if not np.isfinite(S[0]):
            S0 = pd.Series(S).bfill().iloc[0]
        else:
            S0 = S[0]
        settle = float(spx[d])
        spot_err.append(abs(pd.Series(S).dropna().iloc[-1] - settle))
        ratio = S0 / spy_open[s]
        wing = max(5.0, 5.0 * round(3.0 * ratio / 5.0))
        condors += L.build_condors(chain, S, n, s, settle, wing)
        legs += X.legs_for_day(chain, S, n, s, settle, ratio)
    c = pd.DataFrame(condors)
    print(f"SPX sessions {c.session.nunique()} ({c.session.min().date()} .. {c.session.max().date()}); "
          f"parity spot at last minute vs official close: median ${np.median(spot_err):.2f}; wings {sorted(c.wing.unique())}")
    day = c.groupby("session")["condor"].mean()
    t = day.mean() / (day.std(ddof=1) / math.sqrt(len(day)))
    print(f"\nB (PRE-REGISTERED) frozen midday condor on SPX: mean ${day.mean():.2f}/condor per session, t {t:.2f} -> "
          f"{'PASS' if day.mean() > 0 and t >= 2.0 else 'fail'} (bar t >= 2.00)")
    for side in ("put", "call"):
        m = c.groupby("session")[side].mean()
        print(f"   {side} side ${m.mean():.2f} (t {m.mean() / (m.std() / math.sqrt(len(m))):.2f})")
    print(f"   winning days {(day > 0).mean():.0%}, worst day ${day.min():.0f}")

    st = X.structures(pd.DataFrame(legs))
    print("\nEXPLORATORY: hold-to-expiry family on SPX (cash-settled; no after-hours exercise), $/structure per session")
    rows = []
    for (s, w), g in st.groupby(["struct", "window"]):
        m = g.groupby("session")["pnl"].mean()
        rows.append(dict(struct=s, window=w, sessions=len(m), mean=m.mean(), t=m.mean() / (m.std() / math.sqrt(len(m)))))
    r = pd.DataFrame(rows).sort_values("t", ascending=False)
    print(r.head(12).round(2).to_string(index=False))
    print("...")
    print(r.tail(4).round(2).to_string(index=False))
    c.to_parquet("data/spx_condors.parquet")
    pd.DataFrame(legs).to_parquet("data/expiry_family_legs_SPX.parquet")


if __name__ == "__main__":
    main()
