"""
VWAP-deviation mean-reversion scalp engine. Built fresh for this strategy
-- no shared code with signals_research/.

Methodology (every approximation stated):

- **RTH only (9:30-16:00 ET), not extended hours.** VWAP mean reversion
  is a regular-session structure -- pre/post-market volume is thin and
  genuine price discovery is weaker there, so VWAP itself is less
  meaningful and noisier. This is a deliberate scope choice, unlike
  signals_research's model which intentionally matched main.py's
  useRTH=False live behavior -- this strategy is being designed from
  scratch, not replicating an existing config.
- **Session-anchored VWAP**: resets every trading day, computed from
  typical price (high+low+close)/3 weighted by volume, cumulative from
  that day's first RTH bar.
- **Deviation** = close - VWAP, in the day's own price units. **Deviation
  std** = rolling standard deviation of that deviation, computed within
  the same trading day only (so day 2's std isn't contaminated by day
  1's, matching VWAP's own daily reset), using a lookback window.
- **Entry**: deviation <= -K*std -> LONG (price stretched below VWAP,
  expect reversion up). deviation >= +K*std -> SHORT. Only when flat --
  no pyramiding, one position at a time per ticker.
- **Exit** (whichever comes first): reversion to within 0.3*std of VWAP
  (profit target); deviation growing past STOP_MULT*K*std in the entry
  direction (stop-loss -- protects against fading a real trend, not
  noise); MAX_HOLD_BARS minutes elapsed (time exit); or the trading day's
  last RTH bar (forced flat -- no overnight scalp risk).
- **Cost**: 6bps round-trip, same assumption as signals_research/, for
  direct comparability.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

ROUND_TRIP_COST_PCT = 0.0006
STD_LOOKBACK_BARS = 30  # rolling window (minutes) for the deviation std, within-day only


@dataclass
class ScalpConfig:
    entry_k: float          # entry threshold, in std units
    stop_mult: float        # stop-loss at stop_mult * entry_k std
    max_hold_bars: int      # minutes


@dataclass
class ScalpTrade:
    ticker: str
    side: str               # LONG / SHORT
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str        # target / stop / time / eod
    raw_ret_pct: float
    net_ret_pct: float
    regime: str = ""


def _rth_only(df: pd.DataFrame) -> pd.DataFrame:
    idx_time = df.index.time
    mask = (idx_time >= pd.Timestamp("09:30").time()) & (idx_time < pd.Timestamp("16:00").time())
    return df[mask]


def compute_vwap_deviation(df: pd.DataFrame) -> pd.DataFrame:
    df = _rth_only(df).copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3.0

    out_frames = []
    for _, day_df in df.groupby(df.index.date):
        day_typical = (day_df["high"] + day_df["low"] + day_df["close"]) / 3.0
        cum_pv = (day_typical * day_df["volume"]).cumsum()
        cum_v = day_df["volume"].cumsum().replace(0, np.nan)
        vwap = cum_pv / cum_v
        deviation = day_df["close"] - vwap
        dev_std = deviation.rolling(STD_LOOKBACK_BARS, min_periods=10).std()
        d = day_df.copy()
        d["vwap"] = vwap
        d["deviation"] = deviation
        d["dev_std"] = dev_std
        out_frames.append(d)

    return pd.concat(out_frames).sort_index()


def simulate(df_dev: pd.DataFrame, ticker: str, config: ScalpConfig) -> list[ScalpTrade]:
    """
    Includes a re-entry cooldown, added after the first version of this
    engine was found (live, on a smoke test) to re-buy the exact same
    falling knife on the bar immediately after every stop-out: a stop
    fires because the deviation kept growing in the adverse direction,
    which means the naive entry condition ("still >= K std from VWAP")
    is usually still true on the very next bar too, so an ungated engine
    just re-enters and immediately re-stops, over and over, through an
    entire trend. That produced a 6.6% win rate with 96% of exits being
    stops on a 20-day/1-ticker smoke test -- a strategy-design flaw, not
    a finding about mean reversion. Fix: after a stop in a given
    direction, that side is "cooled down" until the deviation first
    recovers to within half the entry threshold (a real sign the
    stretch is easing), not just re-armed on the next bar.
    """
    trades: list[ScalpTrade] = []
    position = None  # None | "LONG" | "SHORT"
    entry_time = entry_price = entry_std = None
    hold_bars = 0
    cooldown_side = None  # "LONG" | "SHORT" | None -- side currently blocked from re-entry

    dates = df_dev.index.date
    for i in range(len(df_dev)):
        t = df_dev.index[i]
        row = df_dev.iloc[i]
        dev, std, px = row["deviation"], row["dev_std"], row["close"]
        is_last_bar_of_day = (i == len(df_dev) - 1) or (dates[i] != dates[i + 1])

        if is_last_bar_of_day:
            cooldown_side = None  # never carry a cooldown across a session boundary

        if position is not None:
            hold_bars += 1
            exit_reason = None
            if abs(dev) <= 0.3 * entry_std:
                exit_reason = "target"
            elif position == "LONG" and dev <= -config.stop_mult * config.entry_k * entry_std:
                exit_reason = "stop"
            elif position == "SHORT" and dev >= config.stop_mult * config.entry_k * entry_std:
                exit_reason = "stop"
            elif hold_bars >= config.max_hold_bars:
                exit_reason = "time"
            elif is_last_bar_of_day:
                exit_reason = "eod"

            if exit_reason:
                raw = (px - entry_price) / entry_price if position == "LONG" else (entry_price - px) / entry_price
                trades.append(ScalpTrade(
                    ticker, position, entry_time, t, entry_price, px, exit_reason,
                    raw * 100, (raw - ROUND_TRIP_COST_PCT) * 100,
                ))
                if exit_reason == "stop":
                    cooldown_side = position
                position = None
            continue

        if pd.isna(std) or std <= 0 or is_last_bar_of_day:
            continue

        # Clear a cooldown once the stretch has genuinely eased, not just on a timer.
        if cooldown_side == "LONG" and dev > -0.5 * config.entry_k * std:
            cooldown_side = None
        elif cooldown_side == "SHORT" and dev < 0.5 * config.entry_k * std:
            cooldown_side = None

        if dev <= -config.entry_k * std and cooldown_side != "LONG":
            position, entry_time, entry_price, entry_std, hold_bars = "LONG", t, px, std, 0
        elif dev >= config.entry_k * std and cooldown_side != "SHORT":
            position, entry_time, entry_price, entry_std, hold_bars = "SHORT", t, px, std, 0

    return trades


def summarize(trades: list[ScalpTrade]) -> dict:
    if not trades:
        return {"n_trades": 0}
    rets = np.array([t.net_ret_pct for t in trades])
    wins = (rets > 0).sum()
    equity = np.cumsum(rets)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    return {
        "n_trades": len(trades),
        "win_rate_pct": round(100 * wins / len(trades), 2),
        "avg_ret_pct": round(float(rets.mean()), 4),
        "total_ret_sum_pct": round(float(rets.sum()), 2),
        "best_pct": round(float(rets.max()), 4),
        "worst_pct": round(float(rets.min()), 4),
        "max_drawdown_pct": round(float(drawdown.min()), 2),
        "exit_reasons": exit_reasons,
    }
