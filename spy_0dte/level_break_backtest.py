"""
Backtest of the SPY 0DTE level-break day trade (CEO plan, 2026-09-24),
replicating ibkr_trader/backend/level_break_trader.py rule for rule:

  UPSIDE    first cross above (pre-market high + 0.10) after 09:30
            -> buy the floor(pre-market high) CALL, target the prior RTH close
  DOWNSIDE  first cross below (overnight low - 0.10)
            -> buy the floor(overnight low) PUT, target prior close - ATR14
  exits     target, premium stop at -50%, or hard close 15:45
  limits    one trade per side, none after 14:30, premium cap $250/contract

Real data only: SPY 1-min SIP bars including extended hours (so the overnight
and pre-market levels are the real ones), and real 0DTE option 1-min bars for
entry/exit prices, with the same modeled bid/ask (QuoteModel, lambda 0.25) and
$0.65/contract commission used everywhere else in this project.

    ./.venv/bin/python level_break_backtest.py
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

import option_returns as R
import spy0dte_framework as F

BREAK_BUFFER = 0.10
# 2026-09-24 fix: the first version fired on gap-up opens where SPY was ALREADY
# above the pre-market high at 09:30 and ALREADY past the prior-close target --
# it entered at the open and exited one minute later (median hold: 1 minute,
# 95.8% "target" exits). That is a 1-minute scalp, not the reclaim trade the
# plan describes. A real reclaim now requires price to be BELOW the level after
# the open first, and the target must still be a meaningful distance away.
REQUIRE_RECLAIM_FROM_BELOW = True
MIN_ROOM_TO_TARGET = 0.20          # $ of SPY move still available when entering
STOP_PCT = 0.50
MAX_PREMIUM = 250.0
HARD_CLOSE_TAU = 375          # 15:45 on a full session
NO_ENTRY_AFTER_TAU = 300      # 14:30
LAM = 0.25
Q = F.QuoteModel(lam=LAM)


def sessions_with_levels(raw: pd.DataFrame) -> dict:
    """Per session: prior RTH close/high/low, overnight range, pre-market range, ATR14."""
    t = raw.index
    rth = raw.between_time("09:30", "15:59")
    daily = rth.groupby(rth.index.date).agg(o=("open", "first"), h=("high", "max"),
                                            l=("low", "min"), c=("close", "last"))
    tr = pd.concat([daily.h - daily.l, (daily.h - daily.c.shift()).abs(),
                    (daily.l - daily.c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().shift(1)          # prior sessions only
    days = list(daily.index)
    out = {}
    for i, d in enumerate(days[1:], start=1):
        prev = days[i - 1]
        if not np.isfinite(atr14.loc[d]):
            continue
        day_bars = raw[raw.index.date == d]
        pre = day_bars.between_time("04:00", "09:29")
        prev_close_ts = pd.Timestamp(f"{prev} 16:00", tz=F.NY)
        overnight = raw[(raw.index > prev_close_ts) & (raw.index < pd.Timestamp(f"{d} 09:30", tz=F.NY))]
        if pre.empty or overnight.empty:
            continue
        out[d] = dict(prior_close=float(daily.c.loc[prev]), atr14=float(atr14.loc[d]),
                      premarket_high=float(pre.high.max()), overnight_low=float(overnight.low.min()))
    return out


def run():
    raw = pd.read_parquet("data/SPY_1min_sip.parquet")
    raw.index = raw.index.tz_convert(F.NY)
    lv = sessions_with_levels(raw)
    files = sorted(Path("data/SPY_0dte_option_bars").glob("*.parquet"))
    trades = []
    for p in files:
        d = pd.Timestamp(p.stem).date()
        if d not in lv:
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        sess = pd.Timestamp(p.stem, tz=F.NY)
        chain = R.DayChain(df, sess, 390)
        if not chain.px:
            continue
        L = lv[d]
        rth = raw[(raw.index.date == d)].between_time("09:30", "15:59")
        if len(rth) < 100:
            continue
        closes = rth["close"].values
        up_trig, dn_trig = L["premarket_high"] + BREAK_BUFFER, L["overnight_low"] - BREAK_BUFFER
        up_strike, dn_strike = math.floor(L["premarket_high"]), math.floor(L["overnight_low"])
        targets = dict(upside=L["prior_close"], downside=L["prior_close"] - L["atr14"])
        done, open_pos, armed = set(), None, not REQUIRE_RECLAIM_FROM_BELOW
        for m in range(len(closes)):
            px = closes[m]
            if open_pos:
                right, strike, entry_px, side = open_pos
                mid = chain.price(right, strike, m)[0]
                if not np.isfinite(mid):
                    continue
                mins_left = 390 - m
                exit_now, reason = False, None
                if Q.sell(mid, mins_left) <= entry_px * (1 - STOP_PCT):
                    exit_now, reason = True, "stop"
                elif side == "upside" and px >= targets["upside"]:
                    exit_now, reason = True, "target"
                elif side == "downside" and px <= targets["downside"]:
                    exit_now, reason = True, "target"
                elif m >= HARD_CLOSE_TAU:
                    exit_now, reason = True, "hard_close"
                if exit_now:
                    pnl = (Q.sell(mid, mins_left) - entry_px) * 100 - 2 * R.COMMISSION
                    trades[-1].update(exit_min=m, exit_reason=reason, pnl=pnl)
                    open_pos = None
                continue
            if not armed and px < L["premarket_high"]:
                armed = True                 # traded below the level -> a reclaim is now possible
            if m > NO_ENTRY_AFTER_TAU or len(done) >= 2:
                continue
            side = ("upside" if (armed and px > up_trig and "upside" not in done)
                    else "downside" if px < dn_trig and "downside" not in done else None)
            if not side:
                continue
            room = (targets[side] - px) if side == "upside" else (px - targets[side])
            if room < MIN_ROOM_TO_TARGET:
                done.add(side)               # target already reached -> nothing to trade
                continue
            right = "C" if side == "upside" else "P"
            strike = up_strike if side == "upside" else dn_strike
            mid = chain.price(right, strike, m)[0]
            done.add(side)
            if not np.isfinite(mid):
                continue
            cost = Q.buy(mid, 390 - m)
            if cost * 100 > MAX_PREMIUM:
                continue
            trades.append(dict(date=d, side=side, entry_min=m, spot=px, strike=strike, right=right,
                               entry_px=cost, target=targets[side], exit_min=None, exit_reason=None, pnl=None))
            open_pos = (right, strike, cost, side)
        if open_pos:            # never priced again -> worst case, expires worthless
            trades[-1].update(exit_min=389, exit_reason="no_exit_quote",
                              pnl=-open_pos[2] * 100 - 2 * R.COMMISSION)
    t = pd.DataFrame(trades)
    t.to_csv("data/level_break_trades.csv", index=False)
    return t


def report(t: pd.DataFrame):
    pd.set_option("display.width", 200)
    print(f"sessions traded: {t.date.nunique()} | trades: {len(t)} "
          f"({t.date.min()} .. {t.date.max()})\n")
    def block(g, label):
        if not len(g):
            print(f"{label:<12} no trades"); return
        n = len(g)
        m = g.pnl.mean()
        se = g.pnl.std(ddof=1) / math.sqrt(n)
        print(f"{label:<12}{n:>5}{m:>9.2f}{g.pnl.median():>9.2f}{(g.pnl > 0).mean():>8.1%}"
              f"{g.pnl.sum():>10.0f}{m/se:>7.2f}{g.pnl.min():>9.0f}")
    print(f"{'group':<12}{'n':>5}{'mean':>9}{'median':>9}{'win':>8}{'total':>10}{'t':>7}{'worst':>9}")
    block(t, "ALL")
    for s in ("upside", "downside"):
        block(t[t.side == s], s)
    print("\nexit reasons:", t.exit_reason.value_counts().to_dict())
    print("\nby period (all trades):")
    t = t.copy()
    t["half"] = pd.to_datetime(t.date).dt.to_period("Q").astype(str)
    for q, g in t.groupby("half"):
        block(g, q)
    print("\nper day (both sides combined):")
    daily = t.groupby("date").pnl.sum()
    tstat = daily.mean() / (daily.std(ddof=1) / math.sqrt(len(daily)))
    print(f"  days {len(daily)} | mean ${daily.mean():+.2f}/day | t {tstat:+.2f} | "
          f"win days {(daily > 0).mean():.1%} | worst ${daily.min():.0f} | total ${daily.sum():.0f}")


if __name__ == "__main__":
    report(run())
