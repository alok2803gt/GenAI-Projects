"""
Which EXIT makes the Inside Day Reversal (bullish harami + downtrend) trade
worth taking -- measured honestly.

Entry is fixed and unchanged: signal day, buy the NEXT open. Only the exit
varies. Every rule is scored the same way the 2026-09-24 audit scored the
live rule, because the original "+1.08% / p=0.00007" was an artefact:

  * MARKET-ADJUSTED: subtract what the whole universe did over the SAME
    number of days from the SAME date. Haramis cluster after selloffs, and
    the days after a selloff are good for everything, so an unadjusted
    number mostly measures the market.
  * DATE-CLUSTERED: up to 29 signals fire on one day across the universe --
    that is one market event, not 29 independent bets. Averaging within a
    date first, then testing across dates, is what dropped the live rule's
    t from 7.37 to 0.48.

MULTIPLE TESTING: this searches ~20 exits on one dataset, so ~1 will clear
t=2 by chance. A Bonferroni bar is printed and nothing should be adopted on
a single-variant result that only just clears it.

    ../../venv/bin/python harami_exit_search.py
"""
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from harami_backtest import BODY_LOOKBACK, find_harami_signals  # noqa: E402
from harami_exit_signal_backtest import add_bearish_harami      # noqa: E402

MAX_LOOK = 40


def exits_for_trade(o, h, l, c, bear, up):
    """All exit rules for one trade. Arrays start at the ENTRY bar (index 0).
    Returns {rule: (pct_return, days_held)}."""
    entry = o[0]
    out = {}
    for n in (1, 2, 3, 5, 10, 20):
        if len(c) > n:
            out[f"fixed_{n}d"] = ((c[n] / entry - 1) * 100, n)
    # profit target, checked on the close, capped at 20 days
    for tgt in (1.0, 2.0, 3.0):
        res = None
        for j in range(1, min(len(c), 21)):
            if (c[j] / entry - 1) * 100 >= tgt:
                res = ((c[j] / entry - 1) * 100, j)
                break
        out[f"target_{tgt:g}%"] = res or (((c[min(20, len(c) - 1)] / entry - 1) * 100), min(20, len(c) - 1))
    # stop loss on the close, otherwise out at day 5
    for stop in (2.0, 3.0, 5.0):
        res = None
        for j in range(1, min(len(c), 6)):
            if (c[j] / entry - 1) * 100 <= -stop:
                res = ((c[j] / entry - 1) * 100, j)
                break
        out[f"stop_{stop:g}%_5d"] = res or ((c[5] / entry - 1) * 100, 5) if len(c) > 5 else None
    # trailing stop from the running close high, capped at 20 days
    for tr in (2.0, 3.0, 5.0):
        peak, res = entry, None
        for j in range(1, min(len(c), 21)):
            peak = max(peak, c[j])
            if (c[j] / peak - 1) * 100 <= -tr:
                res = ((c[j] / entry - 1) * 100, j)
                break
        out[f"trail_{tr:g}%"] = res or (((c[min(20, len(c) - 1)] / entry - 1) * 100), min(20, len(c) - 1))
    # first close above the entry price ("take the first green close"), cap 10
    res = None
    for j in range(1, min(len(c), 11)):
        if c[j] > entry:
            res = ((c[j] / entry - 1) * 100, j)
            break
    out["first_green_10d"] = res or (((c[min(10, len(c) - 1)] / entry - 1) * 100), min(10, len(c) - 1))
    # bearish harami signal, capped at 20 days
    res = None
    for j in range(1, min(len(c), 21)):
        if bear[j]:
            res = ((c[j] / entry - 1) * 100, j)
            break
    out["bear_harami_20d"] = res or (((c[min(20, len(c) - 1)] / entry - 1) * 100), min(20, len(c) - 1))
    return {k: v for k, v in out.items() if v}


def main():
    universe = pickle.load(open(HERE.parent / "breakout_research" / "universe_5y_ohlcv.pkl", "rb"))
    # market benchmark: mean h-day forward return across the universe, per date
    fwd = {}
    for h in range(1, MAX_LOOK + 1):
        parts = []
        for t, df in universe.items():
            c = df["Close"]
            parts.append(pd.DataFrame({"date": pd.to_datetime(c.index), "f": (c.shift(-h) / c - 1).values * 100}))
        fwd[h] = pd.concat(parts).dropna().groupby("date")["f"].mean()

    rows = []
    for t, df in universe.items():
        d = add_bearish_harami(find_harami_signals(df)).reset_index(drop=True)
        dates = pd.to_datetime(df.index)
        n = len(d)
        o, hi, lo, c = d["Open"].values, d["High"].values, d["Low"].values, d["Close"].values
        bear, up = d["bear_harami"].values, d["uptrend_ctx"].values
        for i in range(BODY_LOOKBACK + 1, n - MAX_LOOK - 2):
            if not (bool(d["harami"].iloc[i]) and bool(d["downtrend_ctx"].iloc[i])):
                continue
            e = i + 1
            sl = slice(e, e + MAX_LOOK + 1)
            res = exits_for_trade(o[sl], hi[sl], lo[sl], c[sl], bear[sl], up[sl])
            for rule, (ret, hold) in res.items():
                mkt = fwd.get(hold, pd.Series(dtype=float)).get(dates[e], np.nan)
                rows.append({"date": dates[e], "ticker": t, "rule": rule,
                             "ret": ret, "hold": hold, "excess": ret - mkt})
    tr = pd.DataFrame(rows).dropna(subset=["excess"])
    n_rules = tr.rule.nunique()
    bar = norm.ppf(1 - 0.025 / n_rules)
    print(f"signals: {tr.groupby('rule').size().max()} | exit rules tested: {n_rules} | "
          f"Bonferroni bar t >= {bar:.2f}\n")
    print(f"{'exit rule':<18}{'n':>6}{'raw%':>8}{'excess pp':>11}{'win':>7}{'hold':>7}"
          f"{'t(naive)':>10}{'t(clustered)':>14}{'pp/day':>9}")
    out = []
    for rule, g in tr.groupby("rule"):
        daily = g.groupby("date")["excess"].mean()
        t_cl = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily)))
        t_naive = g.excess.mean() / (g.excess.std(ddof=1) / math.sqrt(len(g)))
        out.append((t_cl, rule, len(g), g.ret.mean(), g.excess.mean(), (g.excess > 0).mean(),
                    g.hold.mean(), t_naive, g.excess.mean() / g.hold.mean()))
    for t_cl, rule, n, raw, exc, win, hold, t_n, ppd in sorted(out, reverse=True):
        flag = "  <- clears Bonferroni" if t_cl >= bar else ("  *" if t_cl >= 2 else "")
        print(f"{rule:<18}{n:>6}{raw:>8.2f}{exc:>11.3f}{win:>7.1%}{hold:>7.1f}"
              f"{t_n:>10.2f}{t_cl:>14.2f}{ppd:>9.4f}{flag}")
    print("\n* = clears t=2 on its own but NOT after correcting for testing 20 rules")


if __name__ == "__main__":
    main()
