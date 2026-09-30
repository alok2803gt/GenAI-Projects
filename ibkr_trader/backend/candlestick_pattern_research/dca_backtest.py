"""
Does averaging down into Inside Day Reversal positions actually help?

The appeal is obvious: add lower, drop the average cost, and the exit "above
average cost" arrives sooner. The danger is equally obvious: it puts MORE money
into the trades that are going wrong, and it is measured on a strategy whose
market-adjusted edge is already about zero (clustered t = 0.48).

So the metric here is NOT win rate or return per trade -- DCA flatters both by
construction. It is RETURN ON CAPITAL DEPLOYED, plus the tail:

    profit / (dollars committed x days committed)   -- capital efficiency
    worst single trade, and the share of capital at risk when wrong

Variants (all on the SAME signals, entry unchanged: buy next open):
    single        1 unit, exit on first close above entry, 10-day cap
    dca_3         add a 2nd unit if the close is <= -3% from entry
    dca_3_6       add at -3% and again at -6% (3 units max)
    dca_5         add a 2nd unit at -5%
    single_2x     1 unit of DOUBLE size -- the honest control for dca_3, which
                  also ends up holding 2 units on losers
Exit for every DCA variant: first close above the AVERAGE cost, 10-day cap.

    ../../venv/bin/python dca_backtest.py
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

CAP_DAYS = 10
VARIANTS = {
    "single":     {"adds": [], "size": 1},
    "single_2x":  {"adds": [], "size": 2},
    "dca_3":      {"adds": [-3.0], "size": 1},
    "dca_3_6":    {"adds": [-3.0, -6.0], "size": 1},
    "dca_5":      {"adds": [-5.0], "size": 1},
}


def simulate(closes, entry_px, adds, size):
    """Returns (pnl_per_share_of_first_unit, units, dollar_days, days, exited_green)."""
    units = [(entry_px, size)]          # (price, size)
    pending = list(adds)
    dollar_days = 0.0
    for day in range(1, min(len(closes), CAP_DAYS + 1)):
        px = closes[day]
        cost = sum(p * s for p, s in units)
        qty = sum(s for _, s in units)
        dollar_days += cost                      # capital tied up for this day
        avg = cost / qty
        # add lower first, then test the exit on the new average
        while pending and (px / entry_px - 1) * 100 <= pending[0]:
            units.append((px, size))
            pending.pop(0)
            cost = sum(p * s for p, s in units)
            qty = sum(s for _, s in units)
            avg = cost / qty
        if px > avg:
            return (px - avg) * qty, qty, dollar_days, day, True
    px = closes[min(len(closes) - 1, CAP_DAYS)]
    qty = sum(s for _, s in units)
    avg = sum(p * s for p, s in units) / qty
    return (px - avg) * qty, qty, dollar_days, min(len(closes) - 1, CAP_DAYS), False


def main():
    universe = pickle.load(open(HERE.parent / "breakout_research" / "universe_5y_ohlcv.pkl", "rb"))
    rows = []
    for t, df in universe.items():
        d = find_harami_signals(df).reset_index(drop=True)
        dates = pd.to_datetime(df.index)
        o, c = d["Open"].values, d["Close"].values
        n = len(d)
        for i in range(BODY_LOOKBACK + 1, n - CAP_DAYS - 2):
            if not (bool(d["harami"].iloc[i]) and bool(d["downtrend_ctx"].iloc[i])):
                continue
            e = i + 1
            entry = o[e]
            seq = c[e:e + CAP_DAYS + 1]
            for name, cfg in VARIANTS.items():
                pnl, qty, ddays, days, green = simulate(seq, entry, cfg["adds"], cfg["size"])
                rows.append({"date": dates[e], "ticker": t, "variant": name, "pnl_per_share": pnl,
                             "units": qty, "dollar_days": ddays, "days": days, "green": green,
                             "entry": entry, "pct": pnl / entry * 100})
    r = pd.DataFrame(rows)
    print(f"signals: {r[r.variant=='single'].shape[0]}\n")
    print(f"{'variant':<12}{'n':>6}{'ret%':>8}{'win':>7}{'avg units':>11}{'days':>6}"
          f"{'$-days/1$':>11}{'ret per $-day':>15}{'worst%':>9}{'t (clustered)':>15}")
    for v, g in r.groupby("variant"):
        daily = g.groupby("date")["pct"].mean()
        t_cl = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily)))
        # capital efficiency: profit per dollar-day of capital committed
        eff = (g.pnl_per_share / g.dollar_days * 10000).mean()
        print(f"{v:<12}{len(g):>6}{g.pct.mean():>8.3f}{(g.pnl_per_share>0).mean():>7.1%}"
              f"{g.units.mean():>11.2f}{g.days.mean():>6.1f}{(g.dollar_days/g.entry).mean():>11.2f}"
              f"{eff:>15.3f}{g.pct.min():>9.1f}{t_cl:>15.2f}")
    print("\n'ret per $-day' = profit per $10,000 of capital committed per day -- the number that")
    print("matters when capital, not signal count, is the binding constraint.\n")
    print("tail check -- the worst outcomes, by variant:")
    for v, g in r.groupby("variant"):
        q = g.pct.quantile([0.01, 0.05]).round(2).tolist()
        losers = g[g.pnl_per_share <= 0]
        print(f"  {v:<12} 1st pct {q[0]:>7.2f}%  5th pct {q[1]:>7.2f}%  "
              f"mean loss when wrong {losers.pct.mean():>6.2f}%  "
              f"avg units when wrong {losers.units.mean():.2f}")
    print("\nsame-signal paired comparison (dca_3 minus single, per trade):")
    piv = r.pivot_table(index=["ticker", "date"], columns="variant", values="pct")
    for v in ("dca_3", "dca_3_6", "dca_5", "single_2x"):
        d = (piv[v] - piv["single"]).dropna()
        t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
        print(f"  {v:<12} mean {d.mean():+.3f}pp  t={t:+.2f}")


if __name__ == "__main__":
    main()
