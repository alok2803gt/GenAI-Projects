"""
backtest_trail.py — Trailing stop width impact on breakout signal outcomes
Tests trail widths 5% through 20% (+ hold-to-expiry) on the same F9-qualified
breakout signals as the live strategy. Hard stop (phase 1, days 1-5) stays fixed at 7%.

Run:  python backtest_trail.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ── Universe ──────────────────────────────────────────────────────────────────
TICKERS = sorted({
    "SPY","QQQ","IWM","DIA","XLK","XLF","XLE","XLV","XLI","GLD","TLT","ARKK",
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","NFLX",
    "AMD","INTC","QCOM","AVGO","TXN","MU","AMAT","LRCX","KLAC","MRVL",
    "CRM","NOW","ADBE","ORCL","SNOW","PANW","CRWD","ZS","DDOG","NET",
    "JPM","BAC","WFC","GS","MS","C","BLK","SCHW","V","MA","AXP","TFC",
    "JNJ","UNH","LLY","PFE","ABBV","MRK","TMO","DHR","ISRG","VRTX","GILD","BMY",
    "HD","MCD","SBUX","NKE","LOW","TGT","COST","BKNG","LULU",
    "PG","KO","PEP","WMT",
    "XOM","CVX","COP","SLB","MPC","VLO","OXY",
    "BA","GE","CAT","HON","RTX","LMT","FDX","UPS","DE","UAL",
    "DIS","CMCSA","VZ","T",
    "COIN","PLTR","UBER","RIVN","ROKU","HOOD","SOFI","PYPL","IBM",
    "RBLX","RCL","ABNB","NKE",
})

# ── Constants ─────────────────────────────────────────────────────────────────
BB_PERIOD     = 20
BB_STD        = 2.0
RSI_PERIOD    = 14
HARD_STOP_PCT = 7.0    # phase 1 stop — fixed, not varied in this test
PHASE1_DAYS   = 5
MAX_HOLD_DAYS = 30

TRAIL_WIDTHS  = [5, 7, 8, 10, 12, 15, 20, 999]  # 999 = hold to 30d (no trail)


# ── Indicator helpers ─────────────────────────────────────────────────────────

def pct_b(close):
    sma = close.rolling(BB_PERIOD).mean()
    std = close.rolling(BB_PERIOD).std(ddof=1)
    ub  = sma + BB_STD * std
    lb  = sma - BB_STD * std
    bw  = ub - lb
    return ((close - lb) / bw * 100).where(bw > 0)

def rsi(close, n=RSI_PERIOD):
    d    = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def vol_90pct(vol, window=90):
    return vol.rolling(window, min_periods=30).quantile(0.90)

def classify(v):
    if pd.isna(v):  return "NEUTRAL"
    if v > 100:     return "EXTENDED"
    if v >= 95:     return "BREAKOUT"
    if v >= 75:     return "PRE-BREAKOUT"
    if v >= 40:     return "NEUTRAL"
    if v >= 25:     return "WEAKENING"
    if v >= 0:      return "PRE-BREAKDOWN"
    return "BREAKDOWN"


# ── Exit simulation ───────────────────────────────────────────────────────────

def simulate_exit(closes, highs, lows, entry_idx, entry_price, trail_pct):
    """
    Phase 1 (days 1-PHASE1_DAYS): 7% hard stop.
    Phase 2 (days PHASE1_DAYS+1 to MAX_HOLD_DAYS): trail_pct% trailing stop.
    trail_pct=999 means no trail — force-close at MAX_HOLD_DAYS.
    Returns (exit_price, days_held, exit_type).
    """
    n = len(closes)
    hard_stop = entry_price * (1 - HARD_STOP_PCT / 100)
    trail_high = entry_price
    phase = 1

    for d in range(1, MAX_HOLD_DAYS + 1):
        idx = entry_idx + d
        if idx >= n:
            break
        day_low  = lows[idx]
        day_high = highs[idx]
        day_close = closes[idx]

        if phase == 1:
            if day_low <= hard_stop:
                return hard_stop, d, "hard_stop"
            if d >= PHASE1_DAYS:
                phase = 2
                trail_high = day_close

        if phase == 2:
            if trail_pct < 999:
                trail_high = max(trail_high, day_high)
                trail_stop = trail_high * (1 - trail_pct / 100)
                if day_low <= trail_stop:
                    return trail_stop, d, "trail_stop"

        if d == MAX_HOLD_DAYS:
            return day_close, d, "max_hold"

    return closes[min(entry_idx + MAX_HOLD_DAYS, n - 1)], MAX_HOLD_DAYS, "max_hold"


# ── Download ──────────────────────────────────────────────────────────────────

print("Downloading 4 years of daily data...")
raw = yf.download(TICKERS, period="4y", auto_adjust=True, progress=False, threads=True)

close_all = raw["Close"]
high_all  = raw["High"]
low_all   = raw["Low"]
vol_all   = raw["Volume"]

available = [t for t in TICKERS
             if t in close_all.columns and close_all[t].notna().sum() > 200]
print(f"Available: {len(available)} tickers, {len(close_all)} trading days\n")


# ── Per-ticker signal scan ────────────────────────────────────────────────────

signals = []
warmup  = BB_PERIOD + RSI_PERIOD + 90 + 10

for tk in available:
    c = close_all[tk].dropna()
    h = high_all[tk].reindex(c.index).ffill()
    l = low_all[tk].reindex(c.index).ffill()
    v = vol_all[tk].reindex(c.index).fillna(0)

    if len(c) < warmup + MAX_HOLD_DAYS + 2:
        continue

    pb_s   = pct_b(c)
    rsi_s  = rsi(c)
    v90_s  = vol_90pct(v)
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    states = pb_s.map(classify)

    c_arr = c.values
    h_arr = h.values
    l_arr = l.values
    n     = len(c_arr)

    for i in range(warmup, n - MAX_HOLD_DAYS - 2):
        if states.iloc[i] != "BREAKOUT":       continue
        if states.iloc[i-1] != "PRE-BREAKOUT": continue   # F9

        rsi_v = rsi_s.iloc[i]
        if pd.isna(rsi_v) or rsi_v < 55:       continue

        vi = v.iloc[i]; v90i = v90_s.iloc[i]
        if pd.isna(vi) or pd.isna(v90i) or vi < v90i: continue

        if not (c.iloc[i] > sma20.iloc[i] > 0 and c.iloc[i] > sma50.iloc[i] > 0): continue

        entry = c_arr[i]
        row   = {
            "ticker": tk,
            "date":   c.index[i],
            "entry":  entry,
            "entry_idx": i,
            "c_arr":  c_arr,
            "h_arr":  h_arr,
            "l_arr":  l_arr,
        }
        signals.append(row)

print(f"Total signals: {len(signals)}")
print(f"Date range   : {min(s['date'] for s in signals).date()} "
      f"to {max(s['date'] for s in signals).date()}\n")


# ── Run exit simulation for each trail width ──────────────────────────────────

results = {tw: [] for tw in TRAIL_WIDTHS}

for sig in signals:
    for tw in TRAIL_WIDTHS:
        exit_px, days, etype = simulate_exit(
            sig["c_arr"], sig["h_arr"], sig["l_arr"],
            sig["entry_idx"], sig["entry"], tw
        )
        ret = (exit_px - sig["entry"]) / sig["entry"] * 100
        results[tw].append({
            "ret":       round(ret, 3),
            "days":      days,
            "exit_type": etype,
            "year":      sig["date"].year,
        })


# ── Summary stats ─────────────────────────────────────────────────────────────

def summarize(rows):
    r = pd.Series([x["ret"] for x in rows])
    d = pd.Series([x["days"] for x in rows])
    return {
        "n":          len(r),
        "win_rate":   (r > 0).mean() * 100,
        "avg_ret":    r.mean(),
        "median_ret": r.median(),
        "sharpe":     r.mean() / r.std() if r.std() > 0 else 0,
        "worst":      r.min(),
        "best":       r.max(),
        "avg_days":   d.mean(),
        "hard_stop":  sum(1 for x in rows if x["exit_type"] == "hard_stop"),
        "trail_stop": sum(1 for x in rows if x["exit_type"] == "trail_stop"),
        "max_hold":   sum(1 for x in rows if x["exit_type"] == "max_hold"),
    }

print("=" * 78)
print(f"  {'Trail':>6}  {'N':>4}  {'WinRate':>7}  {'AvgRet':>7}  "
      f"{'Median':>7}  {'Sharpe':>7}  {'AvgDays':>7}  Exits (H/T/M)")
print("=" * 78)

best_sharpe = -999
best_wr     = -999

for tw in TRAIL_WIDTHS:
    s   = summarize(results[tw])
    lbl = f"{tw}%" if tw < 999 else "HOLD"
    flag = ""
    if s["sharpe"] > best_sharpe:
        best_sharpe = s["sharpe"]
        best_trail_sharpe = tw
    if s["win_rate"] > best_wr:
        best_wr = s["win_rate"]
        best_trail_wr = tw
    print(f"  {lbl:>6}  {s['n']:>4}  {s['win_rate']:>6.1f}%  "
          f"{s['avg_ret']:>+6.2f}%  {s['median_ret']:>+6.2f}%  "
          f"{s['sharpe']:>7.3f}  {s['avg_days']:>6.1f}d  "
          f"{s['hard_stop']}/{s['trail_stop']}/{s['max_hold']}")

print("=" * 78)
print(f"  H=hard_stop  T=trail_stop  M=max_hold\n")


# ── Year-by-year for top 3 trail widths ───────────────────────────────────────

# Pick the three most interesting widths + current 5%
sharpes = {tw: summarize(results[tw])["sharpe"] for tw in TRAIL_WIDTHS}
top3 = sorted(TRAIL_WIDTHS, key=lambda tw: sharpes[tw], reverse=True)[:3]
show = sorted(set([5] + top3))

print("=" * 78)
print(f"  YEAR-BY-YEAR  (trail widths: {[str(tw)+'%' if tw<999 else 'HOLD' for tw in show]})")
print("=" * 78)

by_year = {}
for tw in show:
    for row in results[tw]:
        yr = row["year"]
        if yr not in by_year:
            by_year[yr] = {}
        if tw not in by_year[yr]:
            by_year[yr][tw] = []
        by_year[yr][tw].append(row["ret"])

for yr in sorted(by_year.keys()):
    parts = []
    for tw in show:
        rets = by_year[yr].get(tw, [])
        if rets:
            wr  = sum(1 for r in rets if r > 0) / len(rets) * 100
            avg = sum(rets) / len(rets)
            lbl = f"{tw}%" if tw < 999 else "HOLD"
            parts.append(f"{lbl}: {wr:.0f}% wr {avg:+.2f}% avg (n={len(rets)})")
    print(f"  {yr}: " + "  |  ".join(parts))

print()


# ── Exit-type return breakdown for current vs best ────────────────────────────

print("=" * 78)
print("  EXIT TYPE BREAKDOWN")
print("=" * 78)
for tw in [5, best_trail_sharpe]:
    lbl = f"{tw}% trail" if tw < 999 else "HOLD"
    rows = results[tw]
    print(f"\n  {lbl}:")
    for etype in ["hard_stop", "trail_stop", "max_hold"]:
        sub = [r["ret"] for r in rows if r["exit_type"] == etype]
        if not sub: continue
        wr  = sum(1 for r in sub if r > 0) / len(sub) * 100
        avg = sum(sub) / len(sub)
        print(f"    {etype:12s}: {len(sub):3d} exits  WR={wr:.0f}%  avg={avg:+.2f}%")


# ── Distribution of returns (5% vs best) ─────────────────────────────────────

print()
print("=" * 78)
print("  RETURN DISTRIBUTION (buckets)")
print("=" * 78)
buckets = [(-999,-7),(-7,-5),(-5,-2),(-2,0),(0,2),(2,5),(5,10),(10,999)]
labels  = ["<-7%","-7to-5","-5to-2","-2to0","0to2","2to5","5to10",">10%"]

for tw in [5, best_trail_sharpe]:
    lbl  = f"{tw}% trail" if tw < 999 else "HOLD"
    rets = [r["ret"] for r in results[tw]]
    n    = len(rets)
    line = "  ".join(
        f"{labels[i]:>8s}:{sum(1 for r in rets if lo<=r<hi):3d} ({sum(1 for r in rets if lo<=r<hi)/n*100:.0f}%)"
        for i,(lo,hi) in enumerate(buckets)
    )
    print(f"  {lbl}: {line}")

print("\nDone.")
