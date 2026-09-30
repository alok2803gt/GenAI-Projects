"""Can BREAKOUT supply the direction the Day Trader's magnitude score lacks?

THE INTENT BEING TESTED (CEO, 2026-09-29)
-----------------------------------------
Breakout scanner's purpose is 3d/5d swing. Day Trader's purpose is different:
catch momentum, scalp quickly, be OUT the same day at breakeven or better.
So this does NOT reuse the 3d/5d horizon -- it tests whether yesterday's
BREAKOUT state adds SIGN to today's high-magnitude candidates, on a same-day
scalp with a profit target.

CONSTRUCTION
------------
  direction  : BREAKOUT fired on session t-1 (pct_b > 95 on Bollinger(20,2)
               AND volume ratio >= the ticker's trailing-252d 95th percentile)
  magnitude  : the Day Trader's own candidate test on session t --
               composite_score >= 75 and atr_pct >= 2.5
  entry      : open of session t          (both inputs known before it)
  exit       : profit target if the day's high reaches it, else the close
               -- "out same day", which is the stated intent

No look-ahead: the breakout state uses bars through t-1, the score uses bars
through t-1 plus open[t], and entry is at open[t].

Everything is market-adjusted (excess vs the universe's same-day mean for the
SAME exit rule) and date-clustered (one event per session). The fee hurdle is
the Tiered round trip at the live $148 position = 0.47pp.

    ../../venv/bin/python daytrader_research/breakout_direction_study.py
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PANELS = {
    "explored 112": HERE.parent / "breakout_research" / "universe_5y_ohlcv.pkl",
    "independent 407": HERE / "holdout_universe_5y.pkl",
}
FEE_PP = 0.70 / 148.0 * 100
TARGETS = (0.5, 1.0)


def build(panel_path: Path) -> pd.DataFrame:
    panel = pickle.load(open(panel_path, "rb"))
    frames = []
    for tkr, df in panel.items():
        d = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        if len(d) < 300:
            continue
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        o, h, l, c, v = d.Open, d.High, d.Low, d.Close, d.Volume

        # --- direction: BREAKOUT state as of the PREVIOUS session ---
        sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
        pct_b = (c - (sma20 - 2 * std20)) / ((sma20 + 2 * std20) - (sma20 - 2 * std20)) * 100
        ratio = v / v.rolling(20).mean().shift(1)
        thr = ratio.rolling(252, min_periods=60).quantile(0.95).shift(1)
        # astype(bool): shift+fillna yields OBJECT dtype, and `~` on that does
        # integer negation (-1), not boolean NOT -- which raised a KeyError.
        d["bo_prev"] = ((pct_b > 95) & (ratio >= thr)).shift(1).fillna(False).astype(bool)
        d["pctb_prev"] = pct_b.shift(1)

        # --- magnitude: the Day Trader's own candidate features ---
        prev_c = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().shift(1)
        d["atr_pct"] = atr14 / prev_c * 100
        d["atr_mult"] = (h - l).shift(1) / atr14
        d["gap_pct"] = (o - prev_c) / prev_c * 100
        d["prior_day_ret_pct"] = ((c - o) / o * 100).shift(1)
        d["ret5d_prior"] = (c.shift(1) / c.shift(6) - 1) * 100

        # --- same-day scalp outcomes from the open ---
        d["ret_close"] = (c - o) / o * 100
        for t in TARGETS:
            hit = h >= o * (1 + t / 100)
            d[f"scalp{t}"] = np.where(hit, t, (c - o) / o * 100)
        d["ticker"] = tkr
        frames.append(d.reset_index().rename(columns={"index": "date", "Date": "date"}))

    r = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    return r.dropna(subset=["atr_pct", "atr_mult", "gap_pct", "prior_day_ret_pct",
                            "ret5d_prior", "ret_close"])


def add_scores(r: pd.DataFrame) -> pd.DataFrame:
    g = r.groupby("date")
    r["composite_score"] = (
        0.55 * g["atr_pct"].rank(pct=True) * 100
        + 0.25 * g["gap_pct"].transform(lambda s: s.abs().rank(pct=True)) * 100
        + 0.12 * g["prior_day_ret_pct"].transform(lambda s: s.abs().rank(pct=True)) * 100
        + 0.08 * g["ret5d_prior"].transform(lambda s: s.abs().rank(pct=True)) * 100).round(1)
    return r


def clustered(sub: pd.DataFrame, col: str) -> tuple:
    if sub.empty:
        return float("nan"), float("nan"), 0, 0
    daily = sub.groupby("date")[col].mean()
    if len(daily) < 2:
        return daily.mean(), float("nan"), len(daily), len(sub)
    return daily.mean(), daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))), len(daily), len(sub)


def run(name: str, path: Path) -> None:
    r = add_scores(build(path))
    print(f"\n===== {name} =====")
    print(f"{r.ticker.nunique()} tickers x {r.date.nunique()} sessions, {len(r):,} ticker-days")

    cand = r[(r.composite_score >= 75) & (r.atr_pct >= 2.5)]
    print(f"DT candidates: {len(cand):,}   of which BREAKOUT fired yesterday: "
          f"{int(cand.bo_prev.sum()):,} ({cand.bo_prev.mean()*100:.1f}%)\n")

    for exit_name, col in [("ride to close", "ret_close")] + \
                          [(f"scalp target +{t}%", f"scalp{t}") for t in TARGETS]:
        # market benchmark uses the SAME exit rule across the whole universe
        r["_ex"] = r[col] - r.groupby("date")[col].transform("mean")
        c2 = r[(r.composite_score >= 75) & (r.atr_pct >= 2.5)]
        print(f"  --- exit: {exit_name} ---")
        print(f"  {'population':<34}{'rows':>8}{'days':>6}{'raw%':>9}"
              f"{'excess%':>10}{'t':>7}{'net of fee':>12}")
        for lbl, s in (("DT candidates (all)", c2),
                       ("  + BREAKOUT yesterday", c2[c2.bo_prev]),
                       ("  + NO breakout yesterday", c2[~c2.bo_prev]),
                       ("  + pct_b(t-1) > 80", c2[c2.pctb_prev > 80]),
                       ("  + pct_b(t-1) < 20", c2[c2.pctb_prev < 20])):
            m, t, nd, nr = clustered(s, "_ex")
            raw, _, _, _ = clustered(s, col)
            print(f"  {lbl:<34}{nr:>8}{nd:>6}{raw:>9.3f}{m:>10.3f}{t:>7.2f}"
                  f"{m - FEE_PP:>12.3f}")
        # paired: does breakout-yes beat breakout-no on the SAME day?
        y = c2[c2.bo_prev].groupby("date")["_ex"].mean()
        n_ = c2[~c2.bo_prev].groupby("date")["_ex"].mean()
        pair = pd.concat([y.rename("y"), n_.rename("n")], axis=1, sort=False).dropna()
        if len(pair) > 3:
            d = pair["y"] - pair["n"]
            t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
            print(f"  paired same-day, breakout minus no-breakout: "
                  f"{d.mean():+.3f}pp  t={t:+.2f}  ({len(pair)} days)\n")


def main() -> None:
    print(f"fee hurdle at the live $148 position (Tiered): {FEE_PP:.2f}pp")
    for name, path in PANELS.items():
        if path.exists():
            run(name, path)


if __name__ == "__main__":
    main()
