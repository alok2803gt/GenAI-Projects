"""
Reconstructs breakout_scanner.py's real 7-state Bollinger %B classification
(same formulas: 20,2 Bollinger; RSI-14 Wilder; ADX-14) over the full 5-year
universe_5y_ohlcv.pkl dataset, identifies BREAKOUT/PRE-BREAKOUT transition
days as a daily-bar proxy for "would have alerted" (same class of stated
approximation breakout_volume_threshold_backtest.py already used -- NOT a
faithful intraday replay, that's what breakout_intraday_faithful_backtest.py
is for), and adds THREE genuinely new candidate features not currently
captured anywhere in this account's alert data:
  - dist_52w_high: % below the trailing 252-day high at entry
  - adx: ADX(14) trend strength (already computed live for F8, but never
    tested here as its own standalone filter against forward returns)
  - gap_pct: today's open vs yesterday's close, as a same-day-gap flag

Read-only against universe_5y_ohlcv.pkl (built by fetch_5y_data.py in this
same directory). Never touches breakout_scanner.py or any live file.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
with open(HERE / "universe_5y_ohlcv.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} tickers")


def compute_adx(high, low, close, period=14):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def classify_state(pct_b):
    if pct_b > 100: return "EXTENDED"
    if pct_b >= 95: return "BREAKOUT"
    if pct_b >= 75: return "PRE-BREAKOUT"
    if pct_b >= 40: return "NEUTRAL"
    if pct_b >= 25: return "WEAKENING"
    if pct_b >= 0: return "PRE-BREAKDOWN"
    return "BREAKDOWN"


events = []
for ticker, hist in data.items():
    if len(hist) < 260:
        continue
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    open_ = hist["Open"]
    vol = hist["Volume"]

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_w = (upper - lower).replace(0, np.nan)
    pct_b = (close - lower) / band_w * 100

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs)

    vol_avg20 = vol.rolling(20).mean().shift(1)
    vol_ratio = vol / vol_avg20.replace(0, np.nan)

    adx = compute_adx(high.values, low.values, close.values, 14)
    adx.index = close.index

    roll_high_252 = close.rolling(252, min_periods=100).max()
    dist_52w_high = (close - roll_high_252) / roll_high_252 * 100

    gap_pct = (open_ - close.shift(1)) / close.shift(1) * 100

    state = pct_b.apply(lambda x: classify_state(x) if pd.notna(x) else None)
    prev_state = state.shift(1)

    fwd_1d = close.shift(-1) / close - 1
    fwd_3d = close.shift(-3) / close - 1
    fwd_5d = close.shift(-5) / close - 1

    is_event = state.isin(["BREAKOUT", "PRE-BREAKOUT"]) & ~prev_state.isin(["BREAKOUT", "PRE-BREAKOUT", "EXTENDED"])

    for idx in close.index[is_event.fillna(False)]:
        events.append({
            "ticker": ticker, "date": idx, "state": state.loc[idx],
            "pct_b": pct_b.loc[idx], "rsi": rsi.loc[idx], "vol_ratio": vol_ratio.loc[idx],
            "adx": adx.loc[idx], "dist_52w_high": dist_52w_high.loc[idx], "gap_pct": gap_pct.loc[idx],
            "ret_1d": fwd_1d.loc[idx] * 100, "ret_3d": fwd_3d.loc[idx] * 100, "ret_5d": fwd_5d.loc[idx] * 100,
        })

df = pd.DataFrame(events)
df = df.dropna(subset=["pct_b", "rsi"])
df.to_csv(HERE / "events_5y.csv", index=False)
print(f"\n{len(df)} real historical BREAKOUT/PRE-BREAKOUT transition events, "
      f"{df['ticker'].nunique()} tickers, {df['date'].min()} to {df['date'].max()}")
print(f"by state: {df['state'].value_counts().to_dict()}")

print("\n=== BASELINE (all events) ===")
for h in ("ret_1d", "ret_3d", "ret_5d"):
    v = df[h].dropna()
    print(f"  {h}: n={len(v)} win_rate={(v>0).mean()*100:.1f}% avg={v.mean():+.3f}% median={v.median():+.3f}%")
