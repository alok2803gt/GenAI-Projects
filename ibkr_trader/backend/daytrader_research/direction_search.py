"""SEARCH: what directional signal does the Day Trader actually need?

GOAL (CEO, 2026-09-29): enter and exit the SAME DAY, quick scalp. The
composite score already predicts HOW FAR a name moves (validated: >=0.5% move
rate 36.5% -> 42.7%, replicated out of sample). It predicts nothing about
WHICH WAY (corr with signed excess +0.009). This screens candidate signed
features for that missing sign.

WHAT THIS IS AND IS NOT
-----------------------
This is a SCREEN, not a validation. Twelve features x 2 panels is 24 looks;
something will clear |t|=2 by chance. The protections:
  * The feature list is fixed before running, and EVERY result is printed --
    no quiet dropping of the ones that failed.
  * Every feature must be computable BEFORE the entry (bars through t-1, plus
    open[t] for gap-based ones). Entry is open[t], exit is close[t].
  * Market-adjusted (excess vs the universe's same-day mean) and
    date-clustered (one event per session), as everything here must be.
  * The primary filter is CONSISTENCY ACROSS TWO INDEPENDENT PANELS, which is
    much harder to fake than a single t-stat. A feature that flips sign
    between the 112-name and 407-name universes is noise, full stop.
Anything that survives needs a fresh pre-registered holdout before trading --
this screen cannot license a live change by itself.

Population: the Day Trader's own candidates (composite_score >= 75,
atr_pct >= 2.5), because the question is how to direct THAT flow.

    ../../venv/bin/python daytrader_research/direction_search.py
"""
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PANELS = {
    "explored 112": HERE.parent / "breakout_research" / "universe_5y_ohlcv.pkl",
    "independent 407": HERE / "holdout_universe_5y.pkl",
}
FEE_PP = 0.70 / 148.0 * 100

# Fixed before running. Each is SIGNED and known before the entry.
FEATURES = {
    "pct_b(t-1)":          "position in Bollinger(20,2) band, 0-100",
    "close_qual(t-1)":     "(close-low)/(high-low) of the prior session",
    "gap_pct":             "open[t] vs close[t-1], signed",
    "prior_day_ret":       "open->close of t-1, signed",
    "rs5":                 "5-day return minus the universe's 5-day return",
    "rs20":                "20-day relative strength",
    "dist_52w_high":       "close[t-1] vs its own 252-day high, %",
    "sma20_dist":          "close[t-1] vs its own SMA20, %",
    "gap_align":           "+1 if gap and prior-day move agree, else -1",
    "vol_ratio(t-1)":      "prior session volume / its own 20-day average",
    "range_pos_20d":       "close[t-1] within its own 20-day high-low range",
    "overnight_streak":    "count of consecutive up closes entering t",
}


def build(panel_path: Path) -> pd.DataFrame:
    panel = pickle.load(open(panel_path, "rb"))
    frames = []
    for tkr, df in panel.items():
        d = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        if len(d) < 300:
            continue
        d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
        o, h, l, c, v = d.Open, d.High, d.Low, d.Close, d.Volume
        prev_c = c.shift(1)

        sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
        d["pct_b(t-1)"] = ((c - (sma20 - 2 * std20)) /
                           ((sma20 + 2 * std20) - (sma20 - 2 * std20)) * 100).shift(1)
        rng = (h - l).replace(0, np.nan)
        d["close_qual(t-1)"] = ((c - l) / rng).shift(1) * 100
        d["gap_pct"] = (o - prev_c) / prev_c * 100
        d["prior_day_ret"] = ((c - o) / o * 100).shift(1)
        d["_r5"] = (c.shift(1) / c.shift(6) - 1) * 100
        d["_r20"] = (c.shift(1) / c.shift(21) - 1) * 100
        d["dist_52w_high"] = (c.shift(1) / c.rolling(252).max().shift(1) - 1) * 100
        d["sma20_dist"] = (c.shift(1) / sma20.shift(1) - 1) * 100
        d["vol_ratio(t-1)"] = (v / v.rolling(20).mean().shift(1)).shift(1)
        lo20, hi20 = l.rolling(20).min().shift(1), h.rolling(20).max().shift(1)
        d["range_pos_20d"] = ((c.shift(1) - lo20) / (hi20 - lo20).replace(0, np.nan)) * 100
        up = (c > prev_c).astype(int)
        d["overnight_streak"] = (up.groupby((up != up.shift()).cumsum()).cumcount() + 1).shift(1) * \
                                np.where(up.shift(1) == 1, 1, -1)

        # DT candidate features
        tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean().shift(1)
        d["atr_pct"] = atr14 / prev_c * 100
        d["ret5d_prior"] = d["_r5"]
        d["target"] = (c - o) / o * 100          # same-day open->close, the scalp
        d["ticker"] = tkr
        frames.append(d.reset_index().rename(columns={"index": "date", "Date": "date"}))

    r = pd.concat(frames, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    g = r.groupby("date")
    r["composite_score"] = (
        0.55 * g["atr_pct"].rank(pct=True) * 100
        + 0.25 * g["gap_pct"].transform(lambda s: s.abs().rank(pct=True)) * 100
        + 0.12 * g["prior_day_ret"].transform(lambda s: s.abs().rank(pct=True)) * 100
        + 0.08 * g["ret5d_prior"].transform(lambda s: s.abs().rank(pct=True)) * 100).round(1)
    # relative strength needs the cross-section
    r["rs5"] = r["_r5"] - g["_r5"].transform("mean")
    r["rs20"] = r["_r20"] - g["_r20"].transform("mean")
    r["gap_align"] = np.where(np.sign(r.gap_pct) == np.sign(r.prior_day_ret), 1.0, -1.0)
    r["excess"] = r["target"] - g["target"].transform("mean")
    return r.dropna(subset=["atr_pct", "composite_score", "excess"])


def clustered(s: pd.DataFrame, col: str = "excess") -> tuple:
    if len(s) < 2:
        return float("nan"), float("nan"), 0
    daily = s.groupby("date")[col].mean()
    if len(daily) < 2:
        return daily.mean(), float("nan"), len(daily)
    return daily.mean(), daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily))), len(daily)


def screen(panel_name: str, path: Path) -> dict:
    r = build(path)
    cand = r[(r.composite_score >= 75) & (r.atr_pct >= 2.5)].copy()
    print(f"\n===== {panel_name} =====")
    print(f"{r.ticker.nunique()} tickers, {r.date.nunique()} sessions, "
          f"{len(cand):,} DT candidates\n")
    print(f"  {'feature':<20}{'Q1 (low)':>10}{'Q5 (high)':>11}{'Q5-Q1':>9}"
          f"{'t(Q5-Q1)':>10}{'Q5 excess':>11}{'t(Q5)':>8}")
    out = {}
    for f in FEATURES:
        s = cand.dropna(subset=[f]).copy()
        if len(s) < 500:
            print(f"  {f:<20}  insufficient data")
            continue
        try:
            s["q"] = s.groupby("date")[f].transform(
                lambda x: pd.qcut(x, 5, labels=False, duplicates="drop") if x.notna().sum() >= 5 else np.nan)
        except Exception:
            print(f"  {f:<20}  could not bucket")
            continue
        s = s.dropna(subset=["q"])
        q1, q5 = s[s.q == 0], s[s.q == 4]
        m1, _, _ = clustered(q1)
        m5, t5, _ = clustered(q5)
        # paired same-day spread: Q5 minus Q1 on the same session
        a = q5.groupby("date")["excess"].mean()
        b = q1.groupby("date")["excess"].mean()
        pair = pd.concat([a.rename("hi"), b.rename("lo")], axis=1, sort=False).dropna()
        if len(pair) < 10:
            continue
        dd = pair["hi"] - pair["lo"]
        td = dd.mean() / (dd.std(ddof=1) / math.sqrt(len(dd)))
        out[f] = (dd.mean(), td, m5, t5)
        print(f"  {f:<20}{m1:>10.3f}{m5:>11.3f}{dd.mean():>9.3f}{td:>10.2f}"
              f"{m5:>11.3f}{t5:>8.2f}")
    return out


def main() -> None:
    print("Same-day OPEN->CLOSE excess, by quintile of each candidate directional feature.")
    print("Q5-Q1 is the paired same-day spread: the long-short information in the feature.")
    print(f"For a LONG-only scalp, what matters is Q5 excess vs the {FEE_PP:.2f}pp fee.")
    results = {name: screen(name, p) for name, p in PANELS.items() if p.exists()}

    if len(results) == 2:
        a, b = results.values()
        na, nb = list(results)
        print(f"\n===== CONSISTENCY: does the feature agree across both panels? =====")
        print(f"  {'feature':<20}{'Q5-Q1 ' + na:>16}{'Q5-Q1 ' + nb:>20}{'same sign?':>12}{'both |t|>2':>12}")
        keep = []
        for f in FEATURES:
            if f not in a or f not in b:
                continue
            sa, ta = a[f][0], a[f][1]
            sb, tb = b[f][0], b[f][1]
            same = (sa > 0) == (sb > 0)
            strong = abs(ta) > 2 and abs(tb) > 2
            if same and strong:
                keep.append(f)
            print(f"  {f:<20}{sa:>+11.3f} (t{ta:>+5.2f}){sb:>+13.3f} (t{tb:>+5.2f})"
                  f"{'YES' if same else 'no':>12}{'YES' if strong else 'no':>12}")
        print("\n===== VERDICT =====")
        if keep:
            print(f"  Survived both panels with |t|>2: {', '.join(keep)}")
            print("  This is a SCREEN result, not a validated edge. Next step is a")
            print("  pre-registered test on data not used here.")
        else:
            print("  NOTHING survives consistency + |t|>2 across both panels.")
            print("  No directional feature in this set supplies the missing sign.")


if __name__ == "__main__":
    main()
