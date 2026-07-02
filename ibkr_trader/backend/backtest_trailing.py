"""
Backtest: 5d hold then trailing stop vs fixed exits
Standalone script — no project imports, no code changes.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

TICKERS = sorted(set([
    "SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","XLI","GLD","TLT","ARKK",
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","NFLX",
    "AMD","INTC","QCOM","AVGO","TXN","MU","AMAT","LRCX","KLAC","MRVL","SMCI",
    "CRM","NOW","ADBE","ORCL","SNOW","PANW","CRWD","ZS","DDOG","NET",
    "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","V","MA","AXP","TFC",
    "JNJ","UNH","LLY","PFE","ABBV","MRK","TMO","DHR","ISRG","VRTX","GILD","BMY",
    "HD","MCD","SBUX","NKE","LOW","TGT","COST","BKNG","LULU",
    "PG","KO","PEP","WMT",
    "XOM","CVX","COP","SLB","MPC","VLO","OXY",
    "BA","GE","CAT","HON","RTX","LMT","FDX","UPS","DE","UAL",
    "DIS","CMCSA","VZ","T",
    "COIN","PLTR","UBER","RIVN","ROKU","HOOD","SOFI","PYPL","IBM",
    "RBLX","RCL","ABNB",
]))

BB_PERIOD = 20
BB_STD    = 2.0
RSI_PERIOD = 14
MAX_HOLD   = 30
BULLISH    = {"BREAKOUT", "EXTENDED", "PRE-BREAKOUT"}
TRAIL_LEVELS = [3, 5, 7, 10]


def pct_b_series(c):
    sma = c.rolling(BB_PERIOD).mean()
    std = c.rolling(BB_PERIOD).std(ddof=1)
    ub  = sma + BB_STD * std
    lb  = sma - BB_STD * std
    bw  = ub - lb
    return ((c - lb) / bw * 100).where(bw > 0)


def rsi_series(c, n=RSI_PERIOD):
    d    = c.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def vol_90pct(v, window=90):
    return v.rolling(window, min_periods=30).quantile(0.90)


def classify(pctb):
    if pctb > 100: return "EXTENDED"
    if pctb >= 95: return "BREAKOUT"
    if pctb >= 75: return "PRE-BREAKOUT"
    if pctb >= 40: return "NEUTRAL"
    if pctb >= 25: return "WEAKENING"
    if pctb >= 0:  return "PRE-BREAKDOWN"
    return "BREAKDOWN"


print("Downloading 4 years ...")
raw       = yf.download(TICKERS, period="4y", auto_adjust=True, progress=False, threads=True)
close_all = raw["Close"]
high_all  = raw["High"]
low_all   = raw["Low"]
vol_all   = raw["Volume"]
available = [t for t in TICKERS if t in close_all.columns]
print(f"Downloaded: {len(available)} tickers")

results = []
for tk in available:
    c  = close_all[tk].dropna()
    h  = high_all[tk].reindex(c.index)
    lo = low_all[tk].reindex(c.index)
    v  = vol_all[tk].reindex(c.index)
    if len(c) < 120:
        continue

    pb    = pct_b_series(c)
    rsi   = rsi_series(c)
    v90   = vol_90pct(v)
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    states = pb.apply(lambda x: classify(x) if pd.notna(x) else "NEUTRAL")
    n   = len(c)
    idx = c.index

    for i in range(BB_PERIOD + RSI_PERIOD, n - MAX_HOLD - 2):
        if states.iloc[i] != "BREAKOUT" or states.iloc[i - 1] != "PRE-BREAKOUT":
            continue
        rv = rsi.iloc[i]
        if pd.isna(rv) or rv < 55:
            continue
        vi   = v.iloc[i]
        v90i = v90.iloc[i]
        if pd.isna(vi) or pd.isna(v90i) or vi < v90i:
            continue
        if not (c.iloc[i] > sma20.iloc[i] and c.iloc[i] > sma50.iloc[i]):
            continue

        entry = c.iloc[i]

        # Collect future daily OHLC for MAX_HOLD days
        closes = [c.iloc[min(i + d, n - 1)]  for d in range(1, MAX_HOLD + 1)]
        lows   = [lo.iloc[min(i + d, n - 1)] for d in range(1, MAX_HOLD + 1)]
        highs  = [h.iloc[min(i + d, n - 1)]  for d in range(1, MAX_HOLD + 1)]

        ret_5d  = (closes[4]  - entry) / entry * 100
        ret_10d = (closes[9]  - entry) / entry * 100
        ret_20d = (closes[19] - entry) / entry * 100
        ret_30d = (closes[29] - entry) / entry * 100

        row = {
            "ticker":    tk,
            "date":      idx[i],
            "entry":     entry,
            "ret_5d":    ret_5d,
            "ret_10d":   ret_10d,
            "ret_20d":   ret_20d,
            "ret_30d":   ret_30d,
            "pos_at_5d": ret_5d,
        }

        # 7% fixed stop + 5d hold (best single-phase strategy from prior run)
        stop7   = entry * 0.93
        ret_sl7 = ret_5d
        days_sl7 = 5
        for d, (fc, fl) in enumerate(zip(closes[:5], lows[:5]), 1):
            if fl <= stop7:
                ret_sl7 = -7.0
                days_sl7 = d
                break
        row["ret_sl7"]  = ret_sl7
        row["days_sl7"] = days_sl7

        # 5d hold + trailing stop from day 6, cap at MAX_HOLD
        for ts in TRAIL_LEVELS:
            # Peak tracked from entry through day 5 (using daily highs)
            peak  = max([entry] + highs[:5])
            ret_ts  = ret_5d
            days_ts = 5
            extended = False
            for d in range(5, MAX_HOLD):
                dh  = highs[d]
                dl  = lows[d]
                dc  = closes[d]
                peak = max(peak, dh)
                trail_stop = peak * (1 - ts / 100)
                if dl <= trail_stop:
                    # Exit at trail stop price
                    ret_ts  = (trail_stop - entry) / entry * 100
                    days_ts = d + 1
                    extended = True
                    break
            if not extended:
                # MAX_HOLD reached without trail triggering
                ret_ts  = (closes[MAX_HOLD - 1] - entry) / entry * 100
                days_ts = MAX_HOLD

            row[f"ret_ts{ts}"]  = ret_ts
            row[f"days_ts{ts}"] = days_ts
            row[f"ext_ts{ts}"]  = days_ts > 5   # did trail extend past day 5?

        results.append(row)

df = pd.DataFrame(results)
print(f"\nSignals: {len(df)}\n")

strategies = [
    ("5d fixed hold (baseline)",          "ret_5d",    None,        "—"),
    ("7% stop + 5d hold",                 "ret_sl7",   "days_sl7",  "—"),
    ("5d hold then 3% trail (->30d)",      "ret_ts3",   "days_ts3",  "ext_ts3"),
    ("5d hold then 5% trail (->30d)",      "ret_ts5",   "days_ts5",  "ext_ts5"),
    ("5d hold then 7% trail (->30d)",      "ret_ts7",   "days_ts7",  "ext_ts7"),
    ("5d hold then 10% trail (->30d)",     "ret_ts10",  "days_ts10", "ext_ts10"),
    ("10d fixed hold",                    "ret_10d",   None,        "—"),
    ("20d fixed hold",                    "ret_20d",   None,        "—"),
    ("30d fixed hold",                    "ret_30d",   None,        "—"),
]

print(f"{'Strategy':<38} {'WR':>6}  {'Avg':>7}  {'Median':>7}  {'Worst':>8}  {'Best':>8}  {'Std':>6}  {'AvgDays':>8}")
print("-" * 112)
for label, rcol, dcol, hcol in strategies:
    r    = df[rcol]
    wr   = (r > 0).mean() * 100
    avg  = r.mean()
    med  = r.median()
    worst = r.min()
    best  = r.max()
    std   = r.std()
    days  = f"{df[dcol].mean():.1f}d" if dcol and dcol in df.columns else "  5d"
    print(f"{label:<38} {wr:>5.1f}%  {avg:>+6.3f}%  {med:>+6.3f}%  {worst:>+7.3f}%  "
          f"{best:>+7.3f}%  {std:>5.3f}%  {days:>8}")

# Trail extension stats
print("\n  How often trail extends past day 5 and adds value:")
for ts in TRAIL_LEVELS:
    ext_col  = f"ext_ts{ts}"
    ret_col  = f"ret_ts{ts}"
    days_col = f"days_ts{ts}"
    ext_mask = df[ext_col]
    n_ext    = ext_mask.sum()
    n_tot    = len(df)
    # Return for extended trades
    ext_ret  = df.loc[ext_mask, ret_col].mean()
    ext_5d   = df.loc[ext_mask, "ret_5d"].mean()   # what 5d would have given them
    ext_days = df.loc[ext_mask, days_col].mean()
    # Return for non-extended (stayed at day 5)
    flat_ret = df.loc[~ext_mask, ret_col].mean()
    print(f"    Trail {ts:2d}%: extends on {n_ext:3d}/{n_tot} ({n_ext/n_tot*100:4.0f}%) trades  "
          f"avg days={ext_days:.1f}  avg_return={ext_ret:+.3f}%  "
          f"(vs 5d alone: {ext_5d:+.3f}%  delta={ext_ret-ext_5d:+.3f}%)")

# Day-5 position breakdown
win5  = df[df["pos_at_5d"] > 0]
loss5 = df[df["pos_at_5d"] <= 0]
print(f"\n  At day 5: {len(win5)} trades in profit, {len(loss5)} trades at a loss")
print(f"  {'Strategy':<38} {'WR(win@d5)':>11} {'Avg(win@d5)':>12} {'WR(loss@d5)':>12} {'Avg(loss@d5)':>13}")
for label, rcol, dcol, hcol in strategies[:7]:
    wr_w = (win5[rcol]  > 0).mean() * 100
    av_w = win5[rcol].mean()
    wr_l = (loss5[rcol] > 0).mean() * 100
    av_l = loss5[rcol].mean()
    print(f"  {label:<38} {wr_w:>10.1f}%  {av_w:>+11.3f}%  {wr_l:>11.1f}%  {av_l:>+12.3f}%")

# Expectancy table
print("\n  Expectancy  (WR * avg_win + (1-WR) * avg_loss):")
for label, rcol, dcol, hcol in strategies:
    r      = df[rcol]
    wins   = r[r > 0]
    losses = r[r <= 0]
    if len(wins) > 0 and len(losses) > 0:
        exp = (len(wins) / len(r)) * wins.mean() + (len(losses) / len(r)) * losses.mean()
        rr  = wins.mean() / abs(losses.mean()) if losses.mean() != 0 else 0
        print(f"    {label:<38}  expectancy={exp:+.3f}%  W/L ratio={rr:.2f}x  "
              f"avg_win={wins.mean():+.3f}%  avg_loss={losses.mean():+.3f}%")

# Best trades: how much extra does trail capture vs 5d?
print("\n  Top 15 trades where 7% trail captured the most extra return vs 5d:")
df["extra_ts7"] = df["ret_ts7"] - df["ret_5d"]
top = df.nlargest(15, "extra_ts7")[["date","ticker","ret_5d","ret_ts7","extra_ts7","days_ts7"]]
top.columns = ["date","ticker","5d_ret%","trail_ret%","extra%","days_held"]
print(top.to_string(index=False))
