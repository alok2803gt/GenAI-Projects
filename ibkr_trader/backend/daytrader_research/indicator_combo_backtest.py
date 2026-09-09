"""
Backtests all 1,023 non-empty AND-combinations of 10 standard technical
indicators, on real daily OHLCV, for a same-day (buy-the-open, sell-the-
close) long day trade -- CEO request 2026-09-01.

Universe: same 112-ticker curated universe as breakout_research's
5-year reconstruction (../breakout_research/universe_5y_ohlcv.pkl, real
yfinance daily bars, fetched 2026-08-25, through 2026-08-24 -- reused
read-only rather than re-fetched, ~1 week stale which is immaterial
against a ~252-trading-day evaluation window).

Evaluation window: the most recent ~252 trading days per ticker (~1 year),
using the ticker's full 5-year history only as warmup for the rolling
indicators (SMA/EMA/RSI/etc. all need real lookback, not zero-padded).

No lookahead: every indicator is computed as of day t-1's close; the
"trade" simulated is buy at day t's real open, sell at day t's real
close. This mirrors the account's standing no-lookahead discipline (same
principle as the ADX point-in-time fix in shadow_filter_monitor.py,
2026-08-31).

The 10 indicators (a defensible standard "top 10" list -- not user-
specified, stated explicitly here since the request didn't name them),
each reduced to a single bullish/long boolean at day t-1's close:
  1. RSI(14) in 50-70 (bullish momentum, not overbought)
  2. MACD(12,26,9) line > signal line (bullish crossover state)
  3. Bollinger %B(20,2) in 0.8-1.0 (near/at upper band)
  4. ADX(14) > 25 (trend strength present)
  5. Stochastic %K(14,3) > %D and %K < 80 (bullish, not overbought)
  6. Volume(t-1) > 1.2x 20-day average volume (interest confirmation)
  7. CCI(20) > 100 (bullish breakout zone)
  8. Williams %R(14) > -50 (bullish zone)
  9. EMA(9) > EMA(21) (short-term bullish trend)
 10. ROC(10) > 0 (positive 10-day momentum)

For each of the 1,023 possible non-empty subsets (AND logic -- every
indicator in the subset must be true), computes n, win rate, avg return,
median return of the resulting same-day open->close trades, across the
full universe/evaluation window. All 1,023 rows are saved, sorted by avg
return then win rate -- ranking is NOT restricted to a minimum sample
size, but n is reported for every row so tiny-sample extremes are visible
rather than hidden.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

HERE = Path(__file__).parent
PKL = HERE.parent / "breakout_research" / "universe_5y_ohlcv.pkl"

with open(PKL, "rb") as f:
    data = pickle.load(f)
print(f"Loaded {len(data)} tickers from {PKL}")

IND_NAMES = [
    "RSI_50_70", "MACD_bull", "BB_pctB_0.8-1.0", "ADX_gt25", "Stoch_bull",
    "VolRatio_gt1.2", "CCI_gt100", "WilliamsR_gt-50", "EMA9_gt_EMA21", "ROC10_pos",
]
N_IND = len(IND_NAMES)


def compute_adx(high, low, close, period=14):
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


all_masks = []
all_rets = []
all_meta = []  # (ticker, date) for traceability

for ticker, hist in data.items():
    if len(hist) < 300:
        continue
    o, h, l, c, v = hist["Open"], hist["High"], hist["Low"], hist["Close"], hist["Volume"]

    # 1. RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, 1e-9))

    # 2. MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()

    # 3. Bollinger %B (20,2)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_w = (upper - lower).replace(0, np.nan)
    pct_b = (c - lower) / band_w

    # 4. ADX(14)
    adx = compute_adx(h.values, l.values, c.values, 14)
    adx.index = c.index

    # 5. Stochastic (14,3)
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    stoch_k = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    stoch_d = stoch_k.rolling(3).mean()

    # 6. Volume ratio
    vol_avg20 = v.rolling(20).mean()
    vol_ratio = v / vol_avg20.replace(0, np.nan)

    # 7. CCI(20)
    tp = (h + l + c) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

    # 8. Williams %R(14)
    willr = -100 * (high14 - c) / (high14 - low14).replace(0, np.nan)

    # 9. EMA9 vs EMA21
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()

    # 10. ROC(10)
    roc10 = (c / c.shift(10) - 1) * 100

    # Boolean bullish flags, ALL as of t-1 (shift(1)) -- no lookahead into day t's own bar
    b1 = ((rsi >= 50) & (rsi < 70)).shift(1)
    b2 = (macd_line > macd_signal).shift(1)
    b3 = ((pct_b >= 0.8) & (pct_b <= 1.0)).shift(1)
    b4 = (adx > 25).shift(1)
    b5 = ((stoch_k > stoch_d) & (stoch_k < 80)).shift(1)
    b6 = (vol_ratio > 1.2).shift(1)
    b7 = (cci > 100).shift(1)
    b8 = (willr > -50).shift(1)
    b9 = (ema9 > ema21).shift(1)
    b10 = (roc10 > 0).shift(1)

    bools = pd.concat([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10], axis=1)
    bools.columns = IND_NAMES

    # Same-day open->close return (the actual simulated trade)
    ret = (c - o) / o * 100

    # Evaluation window: most recent ~252 trading days
    valid = bools.notna().all(axis=1) & ret.notna()
    idx = hist.index[valid]
    if len(idx) == 0:
        continue
    eval_idx = idx[-252:]

    sub_bools = bools.loc[eval_idx].values.astype(bool)
    sub_ret = ret.loc[eval_idx].values.astype(float)

    # bitmask per day: bit i set if indicator i is True
    weights = (1 << np.arange(N_IND))
    day_mask = (sub_bools.astype(int) * weights).sum(axis=1)

    all_masks.append(day_mask)
    all_rets.append(sub_ret)
    all_meta.extend([(ticker, str(d.date())) for d in eval_idx])

masks = np.concatenate(all_masks)
rets = np.concatenate(all_rets)
print(f"\nTotal ticker-days in evaluation window: {len(masks)} across {len(data)} tickers")
print(f"Unconditional baseline: n={len(rets)} win_rate={(rets>0).mean()*100:.2f}% avg={rets.mean():+.4f}% median={np.median(rets):+.4f}%")

# All 1023 non-empty subsets, via bitmask
results = []
for size in range(1, N_IND + 1):
    for combo in combinations(range(N_IND), size):
        sub_mask = sum(1 << i for i in combo)
        match = (masks & sub_mask) == sub_mask
        n = int(match.sum())
        if n == 0:
            continue
        r = rets[match]
        results.append({
            "indicators": " & ".join(IND_NAMES[i] for i in combo),
            "n_indicators": size,
            "n_trades": n,
            "win_rate_pct": float((r > 0).mean() * 100),
            "avg_return_pct": float(r.mean()),
            "median_return_pct": float(np.median(r)),
            "std_return_pct": float(r.std()) if n > 1 else 0.0,
        })

print(f"Non-empty combos (out of 1023 possible): {len(results)}")

df = pd.DataFrame(results)
df = df.sort_values(["avg_return_pct", "win_rate_pct"], ascending=[False, False]).reset_index(drop=True)
df.insert(0, "rank", np.arange(1, len(df) + 1))
out_path = HERE / "indicator_combo_results.csv"
df.to_csv(out_path, index=False)
print(f"Saved {len(df)} combo rows to {out_path}")

print("\n=== TOP 20 by avg return (any sample size) ===")
print(df.head(20)[["rank", "indicators", "n_trades", "win_rate_pct", "avg_return_pct", "median_return_pct"]].to_string(index=False))

print("\n=== TOP 20 by avg return, n_trades >= 30 ===")
df30 = df[df.n_trades >= 30].reset_index(drop=True)
print(df30.head(20)[["rank", "indicators", "n_trades", "win_rate_pct", "avg_return_pct", "median_return_pct"]].to_string(index=False))

print("\n=== single-indicator results (for reference) ===")
singles = df[df.n_indicators == 1].sort_values("avg_return_pct", ascending=False)
print(singles[["indicators", "n_trades", "win_rate_pct", "avg_return_pct"]].to_string(index=False))
