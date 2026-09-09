"""
Liquidity-sweep reversal engine. Built fresh -- no shared code with
signals_research/ or scalp_research/.

Methodology (every approximation stated):

- **RTH only (9:30-16:00 ET)**: same reasoning as scalp_research's VWAP
  engine -- this is a regular-session liquidity/stop-hunt concept, thin
  extended-hours volume doesn't have the same resting-order depth.
- **Swing level**: rolling LOOKBACK-minute high/low, shifted by 1 bar so
  the current bar is never included in its own reference level (no
  lookahead).
- **Volatility-scaled buffer**: a rolling 20-bar average bar-range
  (high-low) as a rough intraday ATR proxy, scaled by BUFFER_MULT --
  keeps the "is this a real sweep or just noise poking the level" test
  comparable across tickers with very different price/volatility scales,
  same reasoning as the VWAP engine's std-based entry threshold.
- **Sweep**: a bar's high breaks swing_high + buffer (upside sweep,
  candidate SHORT) or low breaks swing_low - buffer (downside sweep,
  candidate LONG) while flat and not already watching a pending setup.
- **State machine, not a continuously-re-checked condition**: entering
  WAITING_CONFIRM on a sweep, expiring after CONFIRM_WINDOW bars if no
  reversal closes back inside the level. This is a deliberate design
  choice learned from scalp_research/vwap_scalp_engine.py's first draft,
  which re-entered the same failing trade on every bar a continuous
  threshold stayed crossed. A discrete sweep-then-confirm-or-expire event
  can't repeatedly re-trigger off a single extended condition the way a
  bare threshold check can.
- **Entry**: on reversal confirmation (close back on the safe side of the
  swept level) within the confirm window.
- **Exit**: target = TARGET_FRACTION of the (swing_high - swing_low)
  range from entry, in the trade's favor; stop = price re-breaking the
  sweep bar's own extreme (the reversal thesis failing); max hold time;
  forced flat at session close.
- **Cost**: 6bps round-trip, same assumption as the other two strategies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

ROUND_TRIP_COST_PCT = 0.0006
ATR_WINDOW = 20


@dataclass
class SweepConfig:
    lookback: int          # minutes, swing high/low window
    buffer_mult: float      # x ATR proxy
    confirm_window: int     # bars to wait for reversal confirmation
    target_fraction: float  # fraction of swing range as profit target


@dataclass
class SweepTrade:
    ticker: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str
    raw_ret_pct: float
    net_ret_pct: float
    regime: str = ""


def _rth_only(df: pd.DataFrame) -> pd.DataFrame:
    idx_time = df.index.time
    mask = (idx_time >= pd.Timestamp("09:30").time()) & (idx_time < pd.Timestamp("16:00").time())
    return df[mask]


def compute_levels(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df = _rth_only(df).copy()
    out_frames = []
    for _, day_df in df.groupby(df.index.date):
        d = day_df.copy()
        d["swing_high"] = d["high"].rolling(lookback, min_periods=5).max().shift(1)
        d["swing_low"] = d["low"].rolling(lookback, min_periods=5).min().shift(1)
        d["atr"] = (d["high"] - d["low"]).rolling(ATR_WINDOW, min_periods=5).mean()
        out_frames.append(d)
    return pd.concat(out_frames).sort_index()


def simulate(df_lvl: pd.DataFrame, ticker: str, config: SweepConfig) -> list[SweepTrade]:
    trades: list[SweepTrade] = []
    position = None
    entry_time = entry_price = stop_level = target_level = None
    hold_bars = 0

    state = "NONE"  # NONE | WAITING_CONFIRM
    pending_dir = None       # "UP_SWEEP" | "DOWN_SWEEP"
    pending_extreme = None
    pending_swing_high = pending_swing_low = None
    pending_bars_left = 0

    dates = df_lvl.index.date
    n = len(df_lvl)
    for i in range(n):
        t = df_lvl.index[i]
        row = df_lvl.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        sh, sl, atr = row["swing_high"], row["swing_low"], row["atr"]
        is_last_bar_of_day = (i == n - 1) or (dates[i] != dates[i + 1])

        if is_last_bar_of_day:
            state, pending_dir = "NONE", None  # never carry a pending setup across sessions

        if position is not None:
            hold_bars += 1
            exit_reason = None
            if position == "LONG":
                if c >= target_level:
                    exit_reason = "target"
                elif c <= stop_level:
                    exit_reason = "stop"
            else:
                if c <= target_level:
                    exit_reason = "target"
                elif c >= stop_level:
                    exit_reason = "stop"
            if not exit_reason and hold_bars >= 30:  # fixed sane cap regardless of config, safety net
                exit_reason = "time"
            if not exit_reason and is_last_bar_of_day:
                exit_reason = "eod"

            if exit_reason:
                raw = (c - entry_price) / entry_price if position == "LONG" else (entry_price - c) / entry_price
                trades.append(SweepTrade(
                    ticker, position, entry_time, t, entry_price, c, exit_reason,
                    raw * 100, (raw - ROUND_TRIP_COST_PCT) * 100,
                ))
                position = None
            continue

        if state == "WAITING_CONFIRM":
            pending_bars_left -= 1
            if pending_dir == "UP_SWEEP" and c < pending_swing_high:
                rng = pending_swing_high - pending_swing_low
                entry_price, entry_time, hold_bars = c, t, 0
                stop_level = pending_extreme
                target_level = entry_price - config.target_fraction * rng
                position = "SHORT"
                state, pending_dir = "NONE", None
                continue
            if pending_dir == "DOWN_SWEEP" and c > pending_swing_low:
                rng = pending_swing_high - pending_swing_low
                entry_price, entry_time, hold_bars = c, t, 0
                stop_level = pending_extreme
                target_level = entry_price + config.target_fraction * rng
                position = "LONG"
                state, pending_dir = "NONE", None
                continue
            if pending_bars_left <= 0:
                state, pending_dir = "NONE", None
            continue

        if pd.isna(sh) or pd.isna(sl) or pd.isna(atr) or atr <= 0 or is_last_bar_of_day:
            continue
        buffer = config.buffer_mult * atr
        if h > sh + buffer:
            state, pending_dir = "WAITING_CONFIRM", "UP_SWEEP"
            pending_extreme, pending_swing_high, pending_swing_low = h, sh, sl
            pending_bars_left = config.confirm_window
        elif l < sl - buffer:
            state, pending_dir = "WAITING_CONFIRM", "DOWN_SWEEP"
            pending_extreme, pending_swing_high, pending_swing_low = l, sh, sl
            pending_bars_left = config.confirm_window

    return trades


def summarize(trades: list[SweepTrade]) -> dict:
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
