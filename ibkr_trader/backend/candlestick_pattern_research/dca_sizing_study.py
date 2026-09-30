"""What position size does the DATA support for IDR accumulation?

WHY (2026-09-29): the $420/name target was a judgement call, not a measurement.
The CEO asked for the backtest to decide the quantity instead.

WHAT A BACKTEST CAN AND CANNOT DECIDE -- read before trusting any number here.

  CANNOT: tell you a dollar amount directly. Position size does not change
  return PER DOLLAR; it scales P&L and risk together. So there is no "optimal
  $X" hiding in the data waiting to be found.

  CAN: measure (a) the honest per-dollar edge, and (b) the LOSS DISTRIBUTION.
  Size then follows from those two plus a stated risk budget -- which is a
  decision, not a measurement. This script supplies (a) and (b) and then
  inverts a risk budget into a size, so the judgement is explicit instead of
  buried.

TWO CORRECTIONS dca_backtest.py never applied, both of which matter here:
  * MARKET-ADJUSTED. Its +1.10% for dca_3 is raw. The IDR signal's own
    market-adjusted, date-clustered edge is +0.088pp at t=0.48. Sizing off a raw
    return is sizing off beta.
  * DATE-CLUSTERED for significance, since IDR signals cluster on down days.

LIMITATION, stated rather than hidden: this covers the TECHNICAL IDR signal, not
the valuation-gated subset that actually accumulates. Backtesting the gate needs
POINT-IN-TIME fundamentals -- using today's financials to judge a 2023 trade is
look-ahead, and would flatter the result. So these numbers describe the
population the gate selects FROM. The gate can only help by excluding names, and
how much it helps is not measurable here.

    ../../venv/bin/python candlestick_pattern_research/dca_sizing_study.py
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

PANEL = HERE.parent / "breakout_research" / "universe_5y_ohlcv.pkl"
CAP_DAYS = 10
NET_LIQ = 1406.0            # live 2026-09-29
COMMISSION_RT = 0.70        # IBKR Tiered, $0.35 each way
# The structures actually available: 1 unit, or add at -5% (what ADD_TRIGGERS uses)
VARIANTS = {"single": [], "dca_5": [-5.0], "dca_5_10": [-5.0, -10.0]}


def simulate(closes, entry_px, adds):
    """-> (pct return on CAPITAL DEPLOYED, units, dollar_days, days).

    METRIC CORRECTION 2026-09-29: the first version of this returned
    (px - avg)/avg, i.e. return on AVERAGE COST. That is the metric
    dca_backtest.py's own docstring warns "DCA flatters by construction" --
    averaging down lowers the denominator while silently deploying more capital,
    so it credited DCA for capital it did not charge it for. It produced
    +0.669pp at t=5.39 for dca_5 versus +0.175pp for single, which contradicted
    every other measurement of this strategy. Return is now dollar P&L over
    TOTAL CAPITAL DEPLOYED, which charges each extra tranche honestly.
    """
    units = [entry_px]
    pending = list(adds)
    dollar_days = 0.0
    for day in range(1, min(len(closes), CAP_DAYS + 1)):
        px = closes[day]
        dollar_days += sum(units)
        while pending and (px / entry_px - 1) * 100 <= pending[0]:
            units.append(px)
            pending.pop(0)
        avg = sum(units) / len(units)
        if px > avg:
            cost = sum(units)
            return (px * len(units) - cost) / cost * 100, len(units), dollar_days, day
    px = closes[min(len(closes) - 1, CAP_DAYS)]
    cost = sum(units)
    return ((px * len(units) - cost) / cost * 100, len(units), dollar_days,
            min(len(closes) - 1, CAP_DAYS))


def build() -> pd.DataFrame:
    universe = pickle.load(open(PANEL, "rb"))
    # benchmark: equal-weighted universe close-to-close, per date
    closes = {}
    for t, df in universe.items():
        s = df["Close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        closes[t] = s
    px = pd.DataFrame(closes).sort_index()
    rets = px.pct_change()

    rows = []
    for t, df in universe.items():
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
            for name, adds in VARIANTS.items():
                pct, units, ddays, days = simulate(seq, o[e], adds)
                # market benchmark over the SAME holding window
                try:
                    win = rets.loc[dates[e]:dates[e + days]]
                    mkt = (win.mean(axis=1) + 1).prod() - 1
                except Exception:
                    mkt = np.nan
                rows.append({"date": dates[e], "ticker": t, "variant": name,
                             "pct": pct, "mkt_pct": mkt * 100 if mkt == mkt else np.nan,
                             "units": units, "dollar_days": ddays, "days": days,
                             "entry": o[e]})
    r = pd.DataFrame(rows)
    r["excess"] = r["pct"] - r["mkt_pct"]
    return r.dropna(subset=["excess"])


def clustered(s: pd.DataFrame, col: str) -> tuple:
    d = s.groupby("date")[col].mean()
    if len(d) < 2:
        return d.mean(), float("nan"), len(d)
    return d.mean(), d.mean() / (d.std(ddof=1) / math.sqrt(len(d))), len(d)


def main() -> None:
    r = build()
    print(f"{r[r.variant == 'single'].shape[0]:,} IDR signals, "
          f"{r.date.nunique()} distinct dates, {r.ticker.nunique()} tickers\n")

    print("(a) THE HONEST PER-DOLLAR EDGE -- raw vs market-adjusted, date-clustered")
    print(f"  {'variant':<12}{'raw%':>9}{'t raw':>8}{'EXCESS%':>10}{'t excess':>10}"
          f"{'units':>7}{'days':>6}{'win%':>7}")
    for v, g in r.groupby("variant"):
        m_raw, t_raw, _ = clustered(g, "pct")
        m_ex, t_ex, nd = clustered(g, "excess")
        print(f"  {v:<12}{m_raw:>9.3f}{t_raw:>8.2f}{m_ex:>10.3f}{t_ex:>10.2f}"
              f"{g.units.mean():>7.2f}{g.days.mean():>6.1f}{(g.pct > 0).mean() * 100:>7.1f}")

    print("\n(b) THE LOSS DISTRIBUTION (market-adjusted excess, per trade)")
    print(f"  {'variant':<12}{'mean':>8}{'p50':>8}{'p5':>8}{'p1':>8}{'worst':>9}"
          f"{'mean loss':>11}")
    for v, g in r.groupby("variant"):
        q = g.excess.quantile([0.5, 0.05, 0.01])
        print(f"  {v:<12}{g.excess.mean():>8.2f}{q[0.5]:>8.2f}{q[0.05]:>8.2f}"
              f"{q[0.01]:>8.2f}{g.excess.min():>9.2f}"
              f"{g[g.excess < 0].excess.mean():>11.2f}")

    print("\n(c) SIZE IMPLIED BY A RISK BUDGET -- the judgement made explicit")
    print(f"     account ${NET_LIQ:,.0f}. 'p1 loss' = the 1-in-100 trade outcome.")
    print(f"     A size is acceptable only if that outcome is inside the budget.\n")
    print(f"  {'budget':<28}{'variant':<11}{'max $/name':>12}{'% of net liq':>14}")
    for budget_pct, label in ((1.0, "1% of account (tight)"),
                              (2.0, "2% of account (moderate)"),
                              (5.0, "5% of account (aggressive)")):
        budget_dollars = NET_LIQ * budget_pct / 100
        for v, g in r.groupby("variant"):
            p1 = g.excess.quantile(0.01)
            if p1 >= 0:
                continue
            size = budget_dollars / (abs(p1) / 100)
            print(f"  {label:<28}{v:<11}{size:>12,.0f}{size / NET_LIQ * 100:>13.0f}%")

    print("\n(d) DOES THE EDGE EVEN PAY THE COMMISSION AT THESE SIZES?")
    print(f"  {'size':<10}{'fee %':>8}", end="")
    for v in VARIANTS:
        print(f"{v + ' net':>14}", end="")
    print()
    for size in (140, 187, 420, 800, 2000):
        fee_pct = COMMISSION_RT / size * 100
        print(f"  ${size:<9,}{fee_pct:>7.2f}%", end="")
        for v in VARIANTS:
            g = r[r.variant == v]
            m_ex, _, _ = clustered(g, "excess")
            print(f"{m_ex - fee_pct:>13.3f}pp", end="")
        print()

    print("\nNOTE: a positive 'net' column is necessary but NOT sufficient -- check the")
    print("t on the excess column in (a). An insignificant edge sized up is just a")
    print("bigger bet on noise, and the p1 column in (b) is what that costs.")


if __name__ == "__main__":
    main()
