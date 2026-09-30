"""Out-of-sample replication of the DCA-vs-single result on the 407-ticker panel.

The 2026-09-29 in-sample finding (112-ticker panel): paired on identical
signals, market-adjusted and date-clustered, dca_5 beat single by +0.493pp
(t=9.90) and dca_5_10 by +0.629pp (t=9.83). A t near 10 demands replication
before it is used to size real positions.

The 407-ticker panel (daytrader_research/holdout_universe_5y.pkl) has never been
touched by ANY harami/IDR/DCA work -- it was built for the Day Trader direction
question, a different hypothesis on a different horizon. For THIS hypothesis it
is genuinely fresh data.

Identical construction to dca_sizing_study.py, with one efficiency change: the
equal-weighted market index is accumulated once and window returns are read off
it, rather than re-slicing a 407-column frame per trade. Same numbers, faster.

    ../../venv/bin/python candlestick_pattern_research/dca_replication_407.py
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from harami_backtest import BODY_LOOKBACK, find_harami_signals  # noqa: E402

PANELS = {
    "in-sample 112": HERE.parent / "breakout_research" / "universe_5y_ohlcv.pkl",
    "REPLICATION 407": HERE.parent / "daytrader_research" / "holdout_universe_5y.pkl",
}
CAP_DAYS = 10
VARIANTS = {"single": [], "dca_5": [-5.0], "dca_5_10": [-5.0, -10.0]}


def simulate(closes, entry_px, adds):
    """(return on capital deployed %, units, days). Exit: first close above
    average cost, 10-day cap."""
    units = [entry_px]
    pending = list(adds)
    for day in range(1, min(len(closes), CAP_DAYS + 1)):
        px = closes[day]
        while pending and (px / entry_px - 1) * 100 <= pending[0]:
            units.append(px)
            pending.pop(0)
        avg = sum(units) / len(units)
        if px > avg:
            cost = sum(units)
            return (px * len(units) - cost) / cost * 100, len(units), day
    px = closes[min(len(closes) - 1, CAP_DAYS)]
    cost = sum(units)
    d = min(len(closes) - 1, CAP_DAYS)
    return (px * len(units) - cost) / cost * 100, len(units), d


def run(label: str, panel_path: Path) -> None:
    panel = pickle.load(open(panel_path, "rb"))
    # equal-weighted market index, accumulated once
    closes = {}
    for t, df in panel.items():
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        closes[t] = s
    px_all = pd.DataFrame(closes).sort_index()
    mkt_ret = px_all.pct_change().mean(axis=1)
    mkt_cum = (1 + mkt_ret.fillna(0)).cumprod()
    pos = {d: i for i, d in enumerate(mkt_cum.index)}
    cum = mkt_cum.values

    rows = []
    for t, df in panel.items():
        if len(df) < BODY_LOOKBACK + CAP_DAYS + 10:
            continue
        d = find_harami_signals(df).reset_index(drop=True)
        dates = pd.to_datetime(df.index).tz_localize(None).normalize()
        o, c = d["Open"].values, d["Close"].values
        for i in range(BODY_LOOKBACK + 1, len(d) - CAP_DAYS - 2):
            if not (bool(d["harami"].iloc[i]) and bool(d["downtrend_ctx"].iloc[i])):
                continue
            e = i + 1
            seq = c[e:e + CAP_DAYS + 1]
            if len(seq) < 2:
                continue
            d0 = dates[e]
            if d0 not in pos:
                continue
            i0 = pos[d0]
            for name, adds in VARIANTS.items():
                pct, units, days = simulate(seq, o[e], adds)
                i1 = min(i0 + days, len(cum) - 1)
                mkt = (cum[i1] / cum[i0] - 1) * 100 if cum[i0] else np.nan
                rows.append({"date": d0, "ticker": t, "variant": name,
                             "pct": pct, "excess": pct - mkt, "units": units,
                             "days": days})
    r = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["excess"])

    n_sig = r[r.variant == "single"].shape[0]
    print(f"\n===== {label} =====")
    print(f"{r.ticker.nunique()} tickers, {n_sig:,} IDR signals, {r.date.nunique()} distinct dates")
    print(f"  {'variant':<12}{'excess%':>10}{'t':>8}{'units':>7}{'days':>6}"
          f"{'win%':>7}{'p1%':>8}{'worst%':>9}")
    for v, g in r.groupby("variant"):
        daily = g.groupby("date")["excess"].mean()
        t = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily)))
        print(f"  {v:<12}{daily.mean():>10.3f}{t:>8.2f}{g.units.mean():>7.2f}"
              f"{g.days.mean():>6.1f}{(g.pct > 0).mean() * 100:>7.1f}"
              f"{g.excess.quantile(0.01):>8.2f}{g.excess.min():>9.2f}")

    piv = r.pivot_table(index=["ticker", "date"], columns="variant",
                        values="excess").reset_index()
    print(f"\n  PAIRED on identical signals (market cancels):")
    print(f"  {'comparison':<24}{'mean diff':>11}{'t':>8}{'days':>7}")
    out = {}
    for v in ("dca_5", "dca_5_10"):
        if v not in piv:
            continue
        tmp = pd.DataFrame({"date": piv["date"], "d": piv[v] - piv["single"]}).dropna()
        daily = tmp.groupby("date")["d"].mean()
        t = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily)))
        out[v] = (daily.mean(), t)
        print(f"  {v + ' minus single':<24}{daily.mean():>+11.3f}{t:>8.2f}{len(daily):>7}")
    return out


def main() -> None:
    res = {}
    for label, p in PANELS.items():
        if p.exists():
            res[label] = run(label, p)
    if len(res) == 2:
        a, b = res.values()
        na, nb = list(res)
        print(f"\n===== REPLICATION VERDICT =====")
        print(f"  {'comparison':<24}{na:>18}{nb:>20}{'holds?':>10}")
        for v in ("dca_5", "dca_5_10"):
            if v in a and v in b:
                ok = (a[v][0] > 0) == (b[v][0] > 0) and abs(b[v][1]) > 3
                print(f"  {v + ' minus single':<24}{a[v][0]:>+10.3f} (t{a[v][1]:>5.2f})"
                      f"{b[v][0]:>+12.3f} (t{b[v][1]:>5.2f}){'YES' if ok else 'NO':>10}")
        print("\n  'holds' = same sign AND |t| > 3 on the fresh panel.")


if __name__ == "__main__":
    main()
