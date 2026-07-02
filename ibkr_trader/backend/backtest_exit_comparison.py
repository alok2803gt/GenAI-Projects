"""
Backtest: state-transition exit vs fixed 5-day hold
Standalone script — no project imports needed, no code changes made.
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

BB_PERIOD  = 20
BB_STD     = 2.0
RSI_PERIOD = 14
MAX_HOLD   = 20   # cap for state-exit strategy

# ── Helpers ───────────────────────────────────────────────────────────────────
def pct_b_series(close):
    sma = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD).std(ddof=1)
    ub  = sma + BB_STD * std
    lb  = sma - BB_STD * std
    bw  = ub - lb
    return ((close - lb) / bw * 100).where(bw > 0)

def rsi_series(close, n=RSI_PERIOD):
    d    = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def vol_90pct(vol, window=90):
    return vol.rolling(window, min_periods=30).quantile(0.90)

def classify(pctb):
    if pctb > 100:  return "EXTENDED"
    if pctb >= 95:  return "BREAKOUT"
    if pctb >= 75:  return "PRE-BREAKOUT"
    if pctb >= 40:  return "NEUTRAL"
    if pctb >= 25:  return "WEAKENING"
    if pctb >= 0:   return "PRE-BREAKDOWN"
    return "BREAKDOWN"

# ── Download ──────────────────────────────────────────────────────────────────
print("Downloading 4 years of daily data ...")
raw = yf.download(TICKERS, period="4y", auto_adjust=True, progress=False, threads=True)

close_all = raw["Close"]
high_all  = raw["High"]
low_all   = raw["Low"]
vol_all   = raw["Volume"]

available = [t for t in TICKERS if t in close_all.columns]
print(f"Downloaded: {len(available)} tickers, {len(close_all)} trading days")

# ── Per-ticker backtest ───────────────────────────────────────────────────────
BULLISH = {"BREAKOUT", "EXTENDED", "PRE-BREAKOUT"}
results = []

for tk in available:
    c = close_all[tk].dropna()
    h = high_all[tk].reindex(c.index)
    l = low_all[tk].reindex(c.index)
    v = vol_all[tk].reindex(c.index)

    if len(c) < 120:
        continue

    pb  = pct_b_series(c)
    rsi = rsi_series(c)
    v90 = vol_90pct(v)
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()

    states = pb.apply(lambda x: classify(x) if pd.notna(x) else "NEUTRAL")
    n = len(c)
    idx = c.index

    for i in range(BB_PERIOD + RSI_PERIOD, n - MAX_HOLD - 1):
        today_state = states.iloc[i]
        prev_state  = states.iloc[i - 1]

        # Signal: BREAKOUT preceded by PRE-BREAKOUT (F9 warm-only)
        if today_state != "BREAKOUT":
            continue
        if prev_state != "PRE-BREAKOUT":
            continue

        # RSI gate
        rsi_val = rsi.iloc[i]
        if pd.isna(rsi_val) or rsi_val < 55:
            continue

        # Volume gate: today >= 90th percentile
        vi   = v.iloc[i]
        v90i = v90.iloc[i]
        if pd.isna(vi) or pd.isna(v90i) or vi < v90i:
            continue

        # SMA gates
        if not (c.iloc[i] > sma20.iloc[i] and c.iloc[i] > sma50.iloc[i]):
            continue

        entry = c.iloc[i]

        # Exit A: fixed 5-day hold
        j5        = min(i + 5, n - 1)
        ret_5d    = (c.iloc[j5] - entry) / entry * 100

        # Exit B: state-transition (first day %B < 75, i.e. not bullish)
        exit_days = MAX_HOLD
        ret_state = (c.iloc[min(i + MAX_HOLD, n - 1)] - entry) / entry * 100
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if states.iloc[j] not in BULLISH:
                exit_days = j - i
                ret_state = (c.iloc[j] - entry) / entry * 100
                break

        results.append({
            "ticker":          tk,
            "date":            idx[i],
            "entry":           round(entry, 2),
            "ret_5d":          round(ret_5d, 3),
            "ret_state":       round(ret_state, 3),
            "days_state":      exit_days,
            "win_5d":          ret_5d > 0,
            "win_state":       ret_state > 0,
        })

df = pd.DataFrame(results)
print(f"\nSignals found: {len(df)}")
print(f"Date range   : {df['date'].min().date()} to {df['date'].max().date()}")

# ── Summary stats ─────────────────────────────────────────────────────────────
def show_stats(ret_col, win_col, label):
    r  = df[ret_col]
    wr = df[win_col].mean() * 100
    print(f"\n  {label}")
    print(f"    Win rate : {wr:.1f}%")
    print(f"    Avg ret  : {r.mean():+.3f}%")
    print(f"    Median   : {r.median():+.3f}%")
    print(f"    25p/75p  : {r.quantile(.25):+.3f}% / {r.quantile(.75):+.3f}%")
    print(f"    Worst    : {r.min():+.3f}%")
    print(f"    Best     : {r.max():+.3f}%")
    print(f"    Std dev  : {r.std():.3f}%")

print("\n================================================================")
print("  BREAKOUT signals (F9 warm + RSI>=55 + VOL>=90p + above SMA)")
print("================================================================")
show_stats("ret_5d",    "win_5d",    "EXIT A — 5-day fixed hold")
show_stats("ret_state", "win_state", "EXIT B — state-transition (hold until %B<75)")
print(f"\n  Avg days held (state exit): {df['days_state'].mean():.1f}  "
      f"  Median: {df['days_state'].median():.0f}d")

dist = df["days_state"].value_counts().sort_index()
print("  Days-held distribution (state exit):")
for d, cnt in dist.items():
    bar = "#" * (cnt // 5)
    print(f"    {d:3d}d: {cnt:4d}  {bar}")

# ── Head-to-head ──────────────────────────────────────────────────────────────
df["winner"] = "tie"
df.loc[df["ret_state"] > df["ret_5d"], "winner"] = "state"
df.loc[df["ret_5d"] > df["ret_state"], "winner"] = "5d"

hth = df["winner"].value_counts()
print("\n  Head-to-head (which exit beat the other per trade):")
for k in ["state", "5d", "tie"]:
    ct  = hth.get(k, 0)
    pct = ct / len(df) * 100
    print(f"    {k:6s}: {ct:4d}  ({pct:.1f}%)")

# ── By year ───────────────────────────────────────────────────────────────────
df["year"] = df["date"].dt.year
print("\n  By year:")
print(f"  {'Year':>5} {'n':>5} {'WR_5d':>7} {'Avg_5d':>8} {'WR_st':>7} {'Avg_st':>8} {'AvgDays':>8}")
for yr, g in df.groupby("year"):
    print(f"  {yr:>5} {len(g):>5} "
          f"{g['win_5d'].mean()*100:>6.1f}% "
          f"{g['ret_5d'].mean():>+7.3f}% "
          f"{g['win_state'].mean()*100:>6.1f}% "
          f"{g['ret_state'].mean():>+7.3f}% "
          f"{g['days_state'].mean():>7.1f}d")

# ── Worst 10 trades comparison ────────────────────────────────────────────────
print("\n  Worst 10 signals by 5d return (did state-exit save you?):")
cols = ["date","ticker","ret_5d","ret_state","days_state"]
worst10 = df.nsmallest(10, "ret_5d")[cols]
worst10["saved?"] = worst10.apply(lambda r: "YES" if r["ret_state"] > r["ret_5d"] else "no", axis=1)
print(worst10.to_string(index=False))

# ── Extended / never-exited trades ───────────────────────────────────────────
max_hold_rows = df[df["days_state"] == MAX_HOLD]
print(f"\n  Trades that never exited within {MAX_HOLD} days (held full period):")
print(f"    Count: {len(max_hold_rows)} ({len(max_hold_rows)/len(df)*100:.1f}%)")
print(f"    Avg 5d return on these: {max_hold_rows['ret_5d'].mean():+.3f}%")
print(f"    Avg 20d return on these: {max_hold_rows['ret_state'].mean():+.3f}%")
