"""
Stress backtest: 7% hard stop (days 1-5) + 5% trailing stop (days 5-30)
across all market regimes including financial crisis, COVID crash, 2022 bear.
Standalone script — no project imports, no code changes.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Tickers with long history — IPO-era tickers like COIN/RIVN/HOOD excluded for pre-2020 periods
# but included where data exists
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

START_DATE = "2008-01-01"
BB_PERIOD  = 20
BB_STD     = 2.0
RSI_PERIOD = 14
MAX_HOLD   = 30
HARD_STOP  = 0.07   # 7% hard stop in phase 1
TRAIL_PCT  = 0.05   # 5% trailing stop in phase 2
BULLISH    = {"BREAKOUT", "EXTENDED", "PRE-BREAKOUT"}


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


# ── Download ──────────────────────────────────────────────────────────────────
print(f"Downloading all tickers from {START_DATE} ...")
raw       = yf.download(TICKERS, start=START_DATE, auto_adjust=True, progress=False, threads=True)
close_all = raw["Close"]
high_all  = raw["High"]
low_all   = raw["Low"]
vol_all   = raw["Volume"]
available = [t for t in TICKERS if t in close_all.columns]
print(f"Downloaded: {len(available)} tickers, {len(close_all)} trading days "
      f"({close_all.index[0].date()} to {close_all.index[-1].date()})")

# SPY for regime classification
spy_close = close_all["SPY"].dropna()
spy_sma200 = spy_close.rolling(200).mean()
spy_ann_ret = {}   # calendar year -> annual return
for yr in spy_close.index.year.unique():
    yr_data = spy_close[spy_close.index.year == yr]
    if len(yr_data) > 20:
        spy_ann_ret[yr] = (yr_data.iloc[-1] / yr_data.iloc[0] - 1) * 100

print("\nSPY annual returns:")
for yr in sorted(spy_ann_ret):
    tag = ""
    if spy_ann_ret[yr] < -15: tag = " << BEAR"
    elif spy_ann_ret[yr] < 0: tag = " < down year"
    elif spy_ann_ret[yr] > 20: tag = " > strong bull"
    print(f"  {yr}: {spy_ann_ret[yr]:+.1f}%{tag}")

# Named stress periods
STRESS_PERIODS = {
    "2008 Financial Crisis":  ("2008-01-01", "2009-03-31"),
    "2011 Debt Ceiling":      ("2011-05-01", "2011-12-31"),
    "2015-2016 China Scare":  ("2015-08-01", "2016-02-28"),
    "2018 Q4 Selloff":        ("2018-10-01", "2018-12-31"),
    "2020 COVID Crash":       ("2020-02-01", "2020-04-30"),
    "2022 Bear Market":       ("2022-01-01", "2022-12-31"),
    "2025 Tariff Shock":      ("2025-02-01", "2025-05-31"),
    "All Bull Years (SPY>10%)": None,   # computed dynamically
    "All Bear Years (SPY<0%)":  None,
}

# ── Per-ticker backtest ───────────────────────────────────────────────────────
results = []
for tk in available:
    c  = close_all[tk].dropna()
    h  = high_all[tk].reindex(c.index)
    lo = low_all[tk].reindex(c.index)
    v  = vol_all[tk].reindex(c.index)
    if len(c) < 200:
        continue

    pb     = pct_b_series(c)
    rsi    = rsi_series(c)
    v90    = vol_90pct(v)
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    states = pb.apply(lambda x: classify(x) if pd.notna(x) else "NEUTRAL")
    n   = len(c)
    idx = c.index

    for i in range(BB_PERIOD + RSI_PERIOD + 10, n - MAX_HOLD - 2):
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

        entry     = c.iloc[i]
        entry_date = idx[i]
        yr        = entry_date.year

        # SPY context on entry day
        spy_idx = spy_close.index.get_loc(entry_date) if entry_date in spy_close.index else None
        spy_above_200 = (spy_close.iloc[spy_idx] > spy_sma200.iloc[spy_idx]) if spy_idx else True
        spy_yr_ret    = spy_ann_ret.get(yr, 0)

        # Future OHLC
        closes = [c.iloc[min(i + d, n - 1)]  for d in range(1, MAX_HOLD + 1)]
        lows   = [lo.iloc[min(i + d, n - 1)] for d in range(1, MAX_HOLD + 1)]
        highs  = [h.iloc[min(i + d, n - 1)]  for d in range(1, MAX_HOLD + 1)]

        # ── Phase 1: days 1-5 with 7% hard stop ───────────────────────────
        hard_stop_price = entry * (1 - HARD_STOP)
        stopped_early   = False
        phase1_ret      = (closes[4] - entry) / entry * 100
        phase1_days     = 5
        for d, (fc, fl) in enumerate(zip(closes[:5], lows[:5]), 1):
            if fl <= hard_stop_price:
                phase1_ret    = -HARD_STOP * 100
                phase1_days   = d
                stopped_early = True
                break

        # ── Phase 2: trailing stop from day 5, cap at day 30 ─────────────
        if stopped_early:
            final_ret  = phase1_ret
            final_days = phase1_days
            exit_type  = "hard_stop"
        else:
            # Peak = highest high seen from entry through day 5
            peak       = max([entry] + highs[:5])
            final_ret  = (closes[MAX_HOLD - 1] - entry) / entry * 100
            final_days = MAX_HOLD
            exit_type  = "max_hold"
            for d in range(5, MAX_HOLD):
                dh   = highs[d]
                dl   = lows[d]
                peak = max(peak, dh)
                trail_stop = peak * (1 - TRAIL_PCT)
                if dl <= trail_stop:
                    final_ret  = (trail_stop - entry) / entry * 100
                    final_days = d + 1
                    exit_type  = "trail_stop"
                    break

        # Baseline for comparison
        ret_5d  = (closes[4]  - entry) / entry * 100
        ret_20d = (closes[19] - entry) / entry * 100

        results.append({
            "ticker":         tk,
            "date":           entry_date,
            "year":           yr,
            "spy_yr_ret":     spy_yr_ret,
            "spy_above_200":  spy_above_200,
            "entry":          entry,
            "ret_combined":   round(final_ret,  3),
            "ret_5d":         round(ret_5d,     3),
            "ret_20d":        round(ret_20d,    3),
            "days_held":      final_days,
            "exit_type":      exit_type,
            "stopped_early":  stopped_early,
            "win":            final_ret > 0,
        })

df = pd.DataFrame(results)
print(f"\nTotal signals: {len(df)}  "
      f"({df['date'].min().date()} to {df['date'].max().date()})\n")


def show(label, mask, df):
    g = df[mask]
    if len(g) == 0:
        print(f"  {label:<35}  n=0  (no signals)")
        return
    r   = g["ret_combined"]
    r5  = g["ret_5d"]
    wr  = g["win"].mean() * 100
    avg = r.mean(); med = r.median(); worst = r.min(); best = r.max(); std = r.std()
    avg5 = r5.mean(); wr5 = (r5 > 0).mean() * 100
    stopped = g["stopped_early"].sum()
    trail   = (g["exit_type"] == "trail_stop").sum()
    maxh    = (g["exit_type"] == "max_hold").sum()
    days_avg = g["days_held"].mean()
    print(f"  {label:<35}  n={len(g):4d}  WR={wr:5.1f}%  avg={avg:+6.3f}%  "
          f"med={med:+6.3f}%  worst={worst:+7.3f}%  best={best:+7.3f}%  "
          f"std={std:5.2f}%  days={days_avg:4.1f}  "
          f"[stop={stopped} trail={trail} maxhold={maxh}]  "
          f"(5d baseline: WR={wr5:.0f}% avg={avg5:+.3f}%)")


# ── Overall summary ───────────────────────────────────────────────────────────
print("=" * 155)
print("  COMBINED STRATEGY: 7% hard stop (d1-5) + 5% trailing stop (d5-30)")
print("=" * 155)
show("ALL SIGNALS (all years)", df.index.isin(df.index), df)

# ── By year ───────────────────────────────────────────────────────────────────
print("\n  --- By calendar year ---")
for yr in sorted(df["year"].unique()):
    spy_ret = spy_ann_ret.get(yr, 0)
    tag = f"SPY {spy_ret:+.0f}%"
    show(f"{yr}  ({tag})", df["year"] == yr, df)

# ── SPY regime ────────────────────────────────────────────────────────────────
print("\n  --- By SPY regime ---")
show("SPY above SMA-200 (bull trend)",   df["spy_above_200"] == True,  df)
show("SPY below SMA-200 (bear trend)",   df["spy_above_200"] == False, df)
show("Bull years (SPY yr ret > +10%)",   df["spy_yr_ret"] > 10,        df)
show("Neutral years (SPY 0-10%)",        df["spy_yr_ret"].between(0,10), df)
show("Down years (SPY yr ret < 0%)",     df["spy_yr_ret"] < 0,         df)

# ── Named stress periods ──────────────────────────────────────────────────────
print("\n  --- Named stress / crash periods ---")
stress_results = {}
for period_name, dates in STRESS_PERIODS.items():
    if dates is None:
        continue
    start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])
    mask = (df["date"] >= start) & (df["date"] <= end)
    show(period_name, mask, df)
    stress_results[period_name] = df[mask]

# ── Worst individual trades ───────────────────────────────────────────────────
print("\n  Worst 15 individual trades (combined strategy):")
cols = ["date", "ticker", "year", "ret_combined", "ret_5d", "days_held", "exit_type"]
print(df.nsmallest(15, "ret_combined")[cols].to_string(index=False))

# ── Distribution of exit types ────────────────────────────────────────────────
print("\n  Exit type breakdown (all years):")
et = df["exit_type"].value_counts()
for k, v in et.items():
    avg_r = df[df["exit_type"] == k]["ret_combined"].mean()
    wr_e  = (df[df["exit_type"] == k]["win"]).mean() * 100
    print(f"    {k:<15}  {v:4d} trades ({v/len(df)*100:4.1f}%)  "
          f"avg_ret={avg_r:+.3f}%  WR={wr_e:.1f}%")

# ── Crash-specific: did hard stop save us? ───────────────────────────────────
print("\n  Crash-period hard-stop analysis (when hard stop triggered):")
crashed = df[df["stopped_early"]]
if len(crashed):
    print(f"    {len(crashed)} trades stopped early at -7%")
    print(f"    Their 5d return (what would have happened without stop): "
          f"avg={crashed['ret_5d'].mean():+.3f}%  worst={crashed['ret_5d'].min():+.3f}%")
    print(f"    Saved on average: {crashed['ret_5d'].mean() - (-7.0):+.3f}% per stopped trade")
    by_yr = crashed.groupby("year").agg(
        n=("ret_5d","count"),
        avg_5d=("ret_5d","mean"),
        worst_5d=("ret_5d","min")
    )
    print(f"    By year:")
    for yr, row in by_yr.iterrows():
        spy_ret = spy_ann_ret.get(yr, 0)
        print(f"      {yr} (SPY {spy_ret:+.0f}%): {row['n']:2d} stops  "
              f"avg_5d={row['avg_5d']:+.3f}%  worst_5d={row['worst_5d']:+.3f}%")
