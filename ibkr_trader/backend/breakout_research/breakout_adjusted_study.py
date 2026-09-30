"""Does the BREAKOUT signal survive market-adjustment and date-clustering?

WHY (2026-09-29)
----------------
breakout_research/RESEARCH_LOG.md reports BREAKOUT at +1.75%/+1.83% (3d/5d)
with a 53-54% win rate. Grepping that log for "market-adjust", "date-clust",
"excess return" and "benchmark" returns ZERO hits: those are raw, unadjusted,
long-only returns measured over a period in which the market rose.

That is the exact shape the Inside Day Reversal result had at t=7.37 before the
same two corrections collapsed it to t=0.48:
  * MARKET-ADJUSTED -- a long-only signal in a rising market earns beta, not
    alpha. Benchmark = the equal-weighted universe over the SAME window.
  * DATE-CLUSTERED -- breakouts cluster on the same days (one market-wide
    thrust fires dozens of names). Those are ONE event, not N independent bets.

SIGNAL, reproduced from breakout_scanner.py:
  pct_b = (close - lower) / (upper - lower) * 100 on Bollinger(20, 2)
  BREAKOUT = pct_b > 95 AND vol_ratio >= per-ticker 95th percentile of the
             trailing-252-day distribution of (volume / 20-day avg volume)
Live uses an intraday-projected volume; on daily bars the completed volume is
used, which is the same quantity the projection is estimating.

ENTRY is the NEXT open (the alert fires on today's close/intraday state, so
today's close is not tradeable on that information), exit at the close 3 or 5
sessions later. No look-ahead: every input uses bars up to and including the
signal day, and entry is strictly after it.

    ../../venv/bin/python breakout_research/breakout_adjusted_study.py
"""
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PANEL = HERE / "universe_5y_ohlcv.pkl"
HOLDOUT = HERE.parent / "daytrader_research" / "holdout_universe_5y.pkl"

PCT_B_MIN = 95.0
VOL_PCTL = 0.95
HORIZONS = (1, 3, 5)


def build(panel_path: Path) -> pd.DataFrame:
    panel = pickle.load(open(panel_path, "rb"))
    frames = []
    for tkr, df in panel.items():
        d = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        if len(d) < 300:
            continue
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        c, v, o = d.Close, d.Volume, d.Open

        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        upper, lower = sma20 + 2 * std20, sma20 - 2 * std20
        d["pct_b"] = (c - lower) / (upper - lower) * 100

        avg_vol = v.rolling(20).mean().shift(1)          # prior 20 sessions, excl. today
        ratio = v / avg_vol
        # per-ticker 95th pct of the trailing year of ratios, shifted so the
        # threshold itself never uses today's value
        d["vol_ratio"] = ratio
        d["vol_thr"] = ratio.rolling(252, min_periods=60).quantile(VOL_PCTL).shift(1)

        d["breakout"] = (d.pct_b > PCT_B_MIN) & (d.vol_ratio >= d.vol_thr)
        # entry next open, exit close +n sessions
        nxt_open = o.shift(-1)
        for n in HORIZONS:
            d[f"fwd{n}"] = (c.shift(-n) - nxt_open) / nxt_open * 100
        d["ticker"] = tkr
        frames.append(d.reset_index().rename(columns={"index": "date", "Date": "date"}))

    r = pd.concat(frames, ignore_index=True)
    return r.replace([np.inf, -np.inf], np.nan)


def clustered(daily: pd.Series) -> tuple:
    n = len(daily)
    if n < 2:
        return daily.mean() if n else float("nan"), float("nan"), n
    return daily.mean(), daily.mean() / (daily.std(ddof=1) / math.sqrt(n)), n


def report(r: pd.DataFrame, label: str) -> None:
    print(f"\n===== {label} =====")
    print(f"{r.ticker.nunique()} tickers x {r.date.nunique()} sessions, {len(r):,} ticker-days")
    sig = r[r.breakout.fillna(False)]
    print(f"BREAKOUT fires: {len(sig):,} times on {sig.date.nunique()} distinct sessions "
          f"({sig.date.nunique()/r.date.nunique()*100:.1f}% of days, "
          f"{len(sig)/max(sig.date.nunique(),1):.1f} names per firing day)\n")

    print(f"  {'horizon':<10}{'n':>7}{'RAW mean%':>12}{'win%':>8}"
          f"{'EXCESS%':>10}{'t (naive)':>11}{'t (clustered)':>15}")
    for n in HORIZONS:
        col = f"fwd{n}"
        s = sig.dropna(subset=[col])
        if s.empty:
            continue
        # market benchmark: equal-weighted universe over the same window/day
        bench = r.dropna(subset=[col]).groupby("date")[col].mean()
        ex = s[col] - s["date"].map(bench)
        naive_t = ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex)))
        m, t, nd = clustered(ex.groupby(s["date"]).mean())
        print(f"  {n}-day{'':<5}{len(s):>7}{s[col].mean():>12.3f}"
              f"{(s[col] > 0).mean()*100:>8.1f}{m:>10.3f}{naive_t:>11.2f}{t:>15.2f}")

    print("\n  the two corrections, isolated (5-day horizon):")
    col = "fwd5"
    s = sig.dropna(subset=[col])
    bench = r.dropna(subset=[col]).groupby("date")[col].mean()
    ex = s[col] - s["date"].map(bench)
    raw_t = s[col].mean() / (s[col].std(ddof=1) / math.sqrt(len(s)))
    print(f"    raw, per-trade t                  : {raw_t:>6.2f}   (mean {s[col].mean():+.3f}%)")
    print(f"    market-adjusted, per-trade t      : "
          f"{ex.mean()/(ex.std(ddof=1)/math.sqrt(len(ex))):>6.2f}   (mean {ex.mean():+.3f}pp)")
    m, t, nd = clustered(s.groupby("date")[col].mean())
    print(f"    date-clustered only, t            : {t:>6.2f}   ({nd} days)")
    m2, t2, nd2 = clustered(ex.groupby(s["date"]).mean())
    print(f"    BOTH (market-adj + clustered), t  : {t2:>6.2f}   (mean {m2:+.3f}pp, {nd2} days)")


def main() -> None:
    report(build(PANEL), "EXPLORED 112-ticker panel")
    if HOLDOUT.exists():
        report(build(HOLDOUT), "407-ticker panel (independent replication)")


if __name__ == "__main__":
    main()
