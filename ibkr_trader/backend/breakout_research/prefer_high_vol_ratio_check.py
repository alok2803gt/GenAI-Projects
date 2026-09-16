"""
Checks whether vol_ratio predicts outcome WITHIN the PREFER-HIGH BREAKOUT
tier (BREAKOUT + ADX<30) -- the question raised by a 20-alert live sample
that showed a strong-looking but statistically fragile split (vol_ratio<0.5:
38.5% win/-0.446% avg vs vol_ratio>=0.5: 80%/+0.70%+). That sample disagreed
with the original large-sample research (RESEARCH_LOG.md Iteration 1,
n=1107), which found the OPPOSITE shape for vol_ratio in general (0.5-1x
was the WORST bucket). The original faithful-backtest scripts that could
settle this need Alpaca minute-bar credentials this account doesn't have.

This is a real, disclosed approximation instead, NOT a re-run of the
original study:

- Universe: AAPL, MSFT, NVDA, SPY only (4 tickers, vs the original's ~80)
  -- reuses ict_research/bars_1min_cache_ibkr/*.pkl, already fetched via
  IBKR for the unrelated POC-pullback research, ~6 months of real 1-min RTH
  bars (2026-03-17 -> 2026-09-14). No new data pulled.
- History: ~6 months, vs the original's 2.5 years.
- Formulas replicated exactly from breakout_scanner.py: Bollinger %B(20,2)
  and RSI(14, Wilder) on daily closes, ADX(14, Wilder) on daily H/L/C,
  vol_ratio = (today's volume, projected from a fixed intraday checkpoint
  to a full-day equivalent) / (trailing 20-day average daily volume,
  excluding today) -- same formula as compute_indicators() in
  breakout_scanner.py.
- Real methodology difference: the live scanner's vol_ratio uses whatever
  wall-clock time each scan cycle actually ran (it re-evaluates every 3
  min). This script instead freezes the "alert time" at a fixed 45 minutes
  after the open (10:15 ET) for every ticker/day, chosen because it's in
  the middle of the 09:48-10:22 ET range real BREAKOUT alerts fired at on
  the two live sample days. A real day could alert earlier or later than
  that -- disclosed, not hidden.
- BREAKOUT is defined here as pct_b > 95 alone (the live scanner also
  requires vol_ratio to clear a per-ticker rolling-percentile threshold
  before it counts as a fireable alert -- that gate is intentionally left
  OUT here, since it's the very relationship being tested. This means this
  script's "BREAKOUT" population is a superset of what would actually have
  fired live.)
- Return: (day's close / price-at-checkpoint - 1) * 100, matching
  alert_performance.eod_return_pct's definition exactly.

Run: python prefer_high_vol_ratio_check.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "..", "ict_research", "bars_1min_cache_ibkr")
TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]
CHECKPOINT_MINUTES_AFTER_OPEN = 45   # ~10:15 ET, matches real alert-firing times seen live
ADX_PERIOD = 14
RSI_PERIOD = 14


def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) < period:
        return out
    start = valid[0]
    if start + period > len(arr):
        return out
    out[start + period - 1] = np.nanmean(arr[start:start + period])
    for i in range(start + period, len(arr)):
        if not np.isnan(arr[i]):
            out[i] = out[i - 1] * (period - 1) / period + arr[i] / period
    return out


def _compute_adx_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    prev_high = np.roll(high, 1); prev_high[0] = high[0]
    prev_low = np.roll(low, 1); prev_low[0] = low[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = minus_dm[0] = 0.0
    atr = _wilder_smooth(tr, period)
    plus_di = _wilder_smooth(plus_dm, period)
    minus_di = _wilder_smooth(minus_dm, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(atr > 0, 100 * plus_di / atr, np.nan)
        mdi = np.where(atr > 0, 100 * minus_di / atr, np.nan)
        di_sum = pdi + mdi
        dx = np.where(di_sum > 0, 100 * np.abs(pdi - mdi) / di_sum, 0.0)
    return _wilder_smooth(dx, period)


def build_daily(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLCV + the intraday checkpoint volume snapshot, from 1-min bars."""
    df = df_1m.copy()
    df.index = df.index.tz_convert("America/New_York")
    rows = []
    for day, g in df.groupby(df.index.date):
        g = g.between_time("09:30", "16:00")
        if g.empty:
            continue
        o, h, l, c, v = g["open"].iloc[0], g["high"].max(), g["low"].min(), g["close"].iloc[-1], g["volume"].sum()
        checkpoint_time = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30 + CHECKPOINT_MINUTES_AFTER_OPEN)
        upto = g[g.index <= checkpoint_time]
        checkpoint_vol = float(upto["volume"].sum()) if not upto.empty else float("nan")
        checkpoint_price = float(upto["close"].iloc[-1]) if not upto.empty else float("nan")
        rows.append(dict(date=day, open=o, high=h, low=l, close=c, volume=v,
                          checkpoint_vol=checkpoint_vol, checkpoint_price=checkpoint_price))
    d = pd.DataFrame(rows).set_index("date").sort_index()
    return d


def compute_signals(d: pd.DataFrame) -> pd.DataFrame:
    closes = d["close"]
    sma20 = closes.rolling(20).mean()
    std20 = closes.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_w = upper - lower
    pct_b = (closes - lower) / band_w.replace(0, np.nan) * 100

    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs)

    adx = _compute_adx_series(d["high"].to_numpy(float), d["low"].to_numpy(float), closes.to_numpy(float), ADX_PERIOD)

    scale = min(3.0, 390 / CHECKPOINT_MINUTES_AFTER_OPEN)
    avg_vol_20d = d["volume"].rolling(20).mean().shift(1)
    proj_today = d["checkpoint_vol"] * scale
    vol_ratio = proj_today / avg_vol_20d.replace(0, np.nan)

    next_close = closes  # same-day close, matching eod_return_pct's own-day definition
    ret_pct = (next_close / d["checkpoint_price"] - 1) * 100

    out = pd.DataFrame({
        "pct_b": pct_b, "rsi": rsi, "adx": adx, "vol_ratio": vol_ratio,
        "checkpoint_price": d["checkpoint_price"], "close": closes, "ret_pct": ret_pct,
    })
    return out


def main():
    all_rows = []
    for tk in TICKERS:
        path = os.path.join(CACHE_DIR, f"{tk}.pkl")
        df_1m = pd.read_pickle(path)
        daily = build_daily(df_1m)
        sig = compute_signals(daily)
        sig["ticker"] = tk
        all_rows.append(sig)
    full = pd.concat(all_rows).dropna(subset=["pct_b", "rsi", "adx", "vol_ratio", "ret_pct"])
    print(f"Total ticker-days with complete indicators: {len(full)}")

    breakout = full[full["pct_b"] > 95].copy()
    print(f"BREAKOUT-qualifying days (pct_b > 95): {len(breakout)}")

    prefer_high = breakout[breakout["adx"] < 30].copy()
    print(f"PREFER-HIGH-equivalent (BREAKOUT + ADX<30): {len(prefer_high)}")
    print()

    def stats(sub):
        n = len(sub)
        if n == 0:
            return "n=0"
        wins = (sub["ret_pct"] > 0).sum()
        return f"n={n:<4} win_rate={wins/n*100:5.1f}%  avg_ret={sub['ret_pct'].mean():+.3f}%"

    print("=== Within PREFER-HIGH-equivalent: vol_ratio buckets ===")
    for lo, hi, label in [(0, 0.5, "<0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, 2.0, "1.0-2.0"),
                            (2.0, 3.0, "2.0-3.0"), (3.0, np.inf, ">=3.0")]:
        sub = prefer_high[(prefer_high["vol_ratio"] >= lo) & (prefer_high["vol_ratio"] < hi)]
        print(f"  {label:10s}", stats(sub))

    print()
    corr = prefer_high[["vol_ratio", "ret_pct"]].corr().iloc[0, 1]
    print(f"corr(vol_ratio, ret_pct) within PREFER-HIGH-equivalent: {corr:+.3f}  (n={len(prefer_high)})")

    print()
    print("=== Sanity check against RESEARCH_LOG Iteration 1: vol_ratio buckets across ALL BREAKOUT days (no ADX filter) ===")
    for lo, hi, label in [(0, 0.5, "<0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, 2.0, "1.0-2.0"),
                            (2.0, 3.0, "2.0-3.0"), (3.0, np.inf, ">=3.0")]:
        sub = breakout[(breakout["vol_ratio"] >= lo) & (breakout["vol_ratio"] < hi)]
        print(f"  {label:10s}", stats(sub))

    full.to_csv(os.path.join(HERE, "prefer_high_vol_ratio_check_rows.csv"))
    print("\nSaved full row data: prefer_high_vol_ratio_check_rows.csv")


if __name__ == "__main__":
    main()
