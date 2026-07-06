"""
backtest_sr.py — Support/Resistance filter impact on breakout signals
Standalone script, no project imports.

Tests three S/R enhancements on top of the current F9-quality breakout signal:
  A) Resistance proximity filter  — skip if nearest swing resistance < N% above entry
  B) Support-anchored stop        — stop just below nearest swing support vs fixed 7%
  C) Combined A + B

Baseline reproduces the current live strategy:
  Entry  : BREAKOUT state (F9 path gate: PRE-BREAKOUT → BREAKOUT)
  Filters: RSI >= 55, Vol >= 90th-pct, above SMA20 & SMA50
  Exit   : Phase 1 — 7% hard stop OR day 5, Phase 2 — 5% trail until day 30

Run:
  python backtest_sr.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ── Universe (matches live breakout scanner) ───────────────────────────────────
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
    "RBLX","RCL","ABNB","XYZ","NKE",
})

# ── Strategy constants (matches live config) ───────────────────────────────────
BB_PERIOD      = 20
BB_STD         = 2.0
RSI_PERIOD     = 14
HARD_STOP_PCT  = 7.0     # phase 1 stop %
TRAIL_PCT      = 5.0     # phase 2 trailing %
PHASE1_DAYS    = 5       # days before switching to trail
MAX_HOLD_DAYS  = 30      # force close
SR_LOOKBACK    = 60      # bars of history for swing S/R detection
SR_CONFIRM     = 5       # bars each side for swing confirmation
SR_CLUSTER     = 0.003   # 0.3% cluster tolerance

# ── Resistance proximity thresholds to test ────────────────────────────────────
RES_THRESHOLDS = [2.0, 3.0, 5.0]   # skip if nearest resistance within X% above entry


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
    if pd.isna(v):    return "NEUTRAL"
    if v > 100:       return "EXTENDED"
    if v >= 95:       return "BREAKOUT"
    if v >= 75:       return "PRE-BREAKOUT"
    if v >= 40:       return "NEUTRAL"
    if v >= 25:       return "WEAKENING"
    if v >= 0:        return "PRE-BREAKDOWN"
    return "BREAKDOWN"


# ── S/R helpers ───────────────────────────────────────────────────────────────

def _find_swings(arr, mode="high", n_confirm=SR_CONFIRM):
    """Return array of swing high (mode='high') or swing low (mode='low') values."""
    n   = len(arr)
    out = []
    for i in range(n_confirm, n - n_confirm):
        if mode == "high":
            if all(arr[i] >= arr[i - j] for j in range(1, n_confirm + 1)) and \
               all(arr[i] >= arr[i + j] for j in range(1, n_confirm + 1)):
                out.append(arr[i])
        else:
            if all(arr[i] <= arr[i - j] for j in range(1, n_confirm + 1)) and \
               all(arr[i] <= arr[i + j] for j in range(1, n_confirm + 1)):
                out.append(arr[i])
    return out

def _cluster(levels, tol=SR_CLUSTER):
    if not levels:
        return []
    out, grp = [], [sorted(levels)[0]]
    for v in sorted(levels)[1:]:
        if (v - grp[0]) / grp[0] <= tol:
            grp.append(v)
        else:
            out.append(sum(grp) / len(grp))
            grp = [v]
    out.append(sum(grp) / len(grp))
    return out

def nearest_resistance(highs_window, price):
    """Nearest clustered swing high above price. Returns None if not found."""
    raw = [v for v in _find_swings(highs_window, "high") if v > price]
    lvls = [v for v in _cluster(raw) if v > price]
    return min(lvls) if lvls else None

def nearest_support(lows_window, price):
    """Nearest clustered swing low below price. Returns None if not found."""
    raw = [v for v in _find_swings(lows_window, "low") if v < price]
    lvls = [v for v in _cluster(raw) if v < price]
    return max(lvls) if lvls else None

def pivot_r1_s1(prev_high, prev_low, prev_close):
    """Prior-day pivot R1 and S1 as quick intraday S/R."""
    pp = (prev_high + prev_low + prev_close) / 3
    return 2 * pp - prev_low, 2 * pp - prev_high   # R1, S1


# ── Exit simulation (matches live phase 1→2 logic) ───────────────────────────

def simulate_exit(closes, highs, lows, entry_idx, entry_price, stop_px):
    """
    Simulate the two-phase exit:
      Phase 1 (days 1-5): hard stop at stop_px (GTC)
      Phase 2 (days 5-30): 5% trailing stop from highest price seen
    Returns (exit_price, days_held, exit_type)
    """
    n = len(closes)
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
            # Check hard stop (simplified: triggered if day low <= stop_px)
            if day_low <= stop_px:
                return stop_px, d, "hard_stop"
            if d >= PHASE1_DAYS:
                phase = 2
                trail_high = day_close

        if phase == 2:
            trail_high = max(trail_high, day_high)
            trail_stop = trail_high * (1 - TRAIL_PCT / 100)
            if day_low <= trail_stop:
                return trail_stop, d, "trail_stop"

        # Force close at MAX_HOLD_DAYS
        if d == MAX_HOLD_DAYS:
            return day_close, d, "max_hold"

    return closes[min(entry_idx + MAX_HOLD_DAYS, n - 1)], MAX_HOLD_DAYS, "max_hold"


# ── Download data ─────────────────────────────────────────────────────────────

print("Downloading 4 years of daily data (this takes ~60s)...")
raw = yf.download(TICKERS, period="4y", auto_adjust=True, progress=False, threads=True)

close_all  = raw["Close"]
high_all   = raw["High"]
low_all    = raw["Low"]
vol_all    = raw["Volume"]

available = [t for t in TICKERS if t in close_all.columns and close_all[t].notna().sum() > 200]
print(f"Available: {len(available)} tickers, {len(close_all)} trading days\n")


# ── Per-ticker backtest ───────────────────────────────────────────────────────

results = []
skipped_tickers = 0

for tk in available:
    c = close_all[tk].dropna()
    h = high_all[tk].reindex(c.index).fillna(method="ffill")
    l = low_all[tk].reindex(c.index).fillna(method="ffill")
    v = vol_all[tk].reindex(c.index).fillna(0)

    if len(c) < BB_PERIOD + RSI_PERIOD + SR_LOOKBACK + MAX_HOLD_DAYS + 10:
        skipped_tickers += 1
        continue

    pb_s   = pct_b(c)
    rsi_s  = rsi(c)
    v90_s  = vol_90pct(v)
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    states = pb_s.map(classify)

    c_arr  = c.values
    h_arr  = h.values
    l_arr  = l.values
    n      = len(c_arr)
    warmup = BB_PERIOD + RSI_PERIOD + SR_LOOKBACK

    for i in range(warmup, n - MAX_HOLD_DAYS - 2):

        # ── Signal gate (identical to live scanner) ────────────────────────────
        if states.iloc[i] != "BREAKOUT":
            continue
        if states.iloc[i - 1] != "PRE-BREAKOUT":   # F9
            continue

        rsi_val = rsi_s.iloc[i]
        if pd.isna(rsi_val) or rsi_val < 55:
            continue

        vi = v.iloc[i]; v90i = v90_s.iloc[i]
        if pd.isna(vi) or pd.isna(v90i) or vi < v90i:
            continue

        if not (c.iloc[i] > sma20.iloc[i] > 0 and c.iloc[i] > sma50.iloc[i] > 0):
            continue

        entry = c_arr[i]

        # ── Baseline exit (7% hard stop → 5% trail) ──────────────────────────
        baseline_stop = entry * (1 - HARD_STOP_PCT / 100)
        exit_px, days, etype = simulate_exit(c_arr, h_arr, l_arr, i, entry, baseline_stop)
        ret_base = (exit_px - entry) / entry * 100

        # ── S/R computation (using only bars BEFORE signal — no lookahead) ───
        h_window = h_arr[i - SR_LOOKBACK: i]
        l_window = l_arr[i - SR_LOOKBACK: i]

        res_near = nearest_resistance(h_window, entry)
        sup_near = nearest_support(l_window, entry)

        # Resistance proximity: pct distance to nearest resistance above
        res_pct = ((res_near / entry) - 1) * 100 if res_near else None

        # Support-anchored stop: stop just below nearest support (floor = 10%, cap = 7%)
        if sup_near and sup_near < entry:
            sup_dist_pct = (entry - sup_near) / entry * 100
            # Place stop 0.5% below support, but cap at HARD_STOP_PCT
            anchored_stop_pct = min(sup_dist_pct + 0.5, HARD_STOP_PCT)
        else:
            anchored_stop_pct = HARD_STOP_PCT  # no support found → use default

        anchored_stop_px = entry * (1 - anchored_stop_pct / 100)
        exit_px_anc, days_anc, etype_anc = simulate_exit(
            c_arr, h_arr, l_arr, i, entry, anchored_stop_px
        )
        ret_anchored = (exit_px_anc - entry) / entry * 100

        # Pivot S1/R1 from prior day
        pv_r1, pv_s1 = pivot_r1_s1(h_arr[i-1], l_arr[i-1], c_arr[i-1])
        pv_res_pct = ((pv_r1 / entry) - 1) * 100 if pv_r1 > entry else None

        results.append({
            "ticker":           tk,
            "date":             c.index[i],
            "entry":            round(entry, 2),

            # Baseline
            "ret_base":         round(ret_base, 3),
            "days_base":        days,
            "exit_base":        etype,

            # S/R data
            "res_near":         round(res_near, 2) if res_near else None,
            "res_pct":          round(res_pct, 2)  if res_pct  else None,
            "sup_near":         round(sup_near, 2) if sup_near else None,
            "anchored_stop_pct": round(anchored_stop_pct, 2),
            "pv_r1":            round(pv_r1, 2),
            "pv_s1":            round(pv_s1, 2),
            "pv_res_pct":       round(pv_res_pct, 2) if pv_res_pct else None,

            # Anchored stop result
            "ret_anchored":     round(ret_anchored, 3),
            "days_anchored":    days_anc,
            "exit_anchored":    etype_anc,
        })

df = pd.DataFrame(results)
print(f"Total signals: {len(df)}")
print(f"Date range   : {df['date'].min().date()} to {df['date'].max().date()}\n")


# ── Stats helper ──────────────────────────────────────────────────────────────

def stats(sub, ret_col, label, n_total):
    if len(sub) == 0:
        print(f"  {label}: no signals")
        return
    r   = sub[ret_col]
    wr  = (r > 0).mean() * 100
    kept_pct = len(sub) / n_total * 100
    print(f"  {label}")
    print(f"    Signals  : {len(sub):,}  ({kept_pct:.0f}% of baseline)")
    print(f"    Win rate : {wr:.1f}%")
    print(f"    Avg ret  : {r.mean():+.3f}%")
    print(f"    Median   : {r.median():+.3f}%")
    print(f"    Sharpe*  : {r.mean()/r.std():.3f}  (* avg/std, not annualized)")
    print(f"    Worst    : {r.min():+.3f}%   Best: {r.max():+.3f}%")
    print()


n_total = len(df)

# ── SECTION 1: Baseline ───────────────────────────────────────────────────────
print("=" * 70)
print("  BASELINE  --  current live strategy (7% hard stop -> 5% trail, 30d)")
print("=" * 70)
stats(df, "ret_base", "All signals", n_total)

# ── SECTION 2: Resistance proximity filter ────────────────────────────────────
print("=" * 70)
print("  FILTER A  —  Skip if nearest swing resistance within X% of entry")
print("=" * 70)
for thresh in RES_THRESHOLDS:
    # Keep signals where resistance is far away OR not found (open air breakout)
    mask = df["res_pct"].isna() | (df["res_pct"] >= thresh)
    label = f"Skip if resistance < {thresh:.0f}% above (keep {mask.sum():,} signals)"
    stats(df[mask], "ret_base", label, n_total)

# Signals where entry is ABOVE prior resistance (resistance flipped to support)
above_res = df["res_near"].isna()   # no resistance found = clean breakout above all prior highs
print(f"  Clean break (no prior resistance found above entry): {above_res.sum():,} signals")
stats(df[above_res], "ret_base", "Clean breakout — above all prior swing highs", n_total)
print()

# ── SECTION 3: Support-anchored stop ─────────────────────────────────────────
print("=" * 70)
print("  FILTER B  —  Support-anchored stop vs fixed 7% stop")
print("=" * 70)
stats(df, "ret_base",     "Baseline (7% hard stop -> 5% trail)", n_total)
stats(df, "ret_anchored", "Anchored stop (below nearest support, max 7%) -> 5% trail", n_total)

# Break down by whether support was found
sup_found    = df["sup_near"].notna()
sup_notfound = ~sup_found
print(f"  Support level found  : {sup_found.sum():,} signals ({sup_found.mean()*100:.0f}%)")
print(f"  No support found     : {sup_notfound.sum():,} signals (uses default 7% stop)")
print()
stats(df[sup_found],    "ret_base",     "  Support found — baseline", sup_found.sum())
stats(df[sup_found],    "ret_anchored", "  Support found — anchored stop", sup_found.sum())

# Distribution of anchored stop distances
print("  Anchored stop distribution (when support found):")
for bucket in [2, 4, 5, 6, 7]:
    cnt = ((df.loc[sup_found, "anchored_stop_pct"] <= bucket) &
           (df.loc[sup_found, "anchored_stop_pct"] > bucket - 1)).sum()
    print(f"    {bucket-1:.0f}–{bucket:.0f}%: {cnt:,}")

print()

# ── SECTION 4: Pivot R1 proximity filter ─────────────────────────────────────
print("=" * 70)
print("  FILTER A2  —  Skip if prior-day Pivot R1 within X% of entry")
print("=" * 70)
for thresh in RES_THRESHOLDS:
    mask = df["pv_res_pct"].isna() | (df["pv_res_pct"] >= thresh)
    label = f"Skip if Pivot R1 < {thresh:.0f}% above entry (keep {mask.sum():,})"
    stats(df[mask], "ret_base", label, n_total)

# ── SECTION 5: Combined A + B ─────────────────────────────────────────────────
print("=" * 70)
print("  COMBINED  —  Filter A (resistance >= 3%) + anchored stop")
print("=" * 70)
mask_comb = df["res_pct"].isna() | (df["res_pct"] >= 3.0)
df_comb = df[mask_comb]
stats(df_comb, "ret_base",     "Combined filter — baseline stop", n_total)
stats(df_comb, "ret_anchored", "Combined filter — anchored stop", n_total)

# ── SECTION 6: Exit type breakdown ───────────────────────────────────────────
print("=" * 70)
print("  EXIT TYPE BREAKDOWN")
print("=" * 70)
print("\n  Baseline:")
for etype, grp in df.groupby("exit_base"):
    r = grp["ret_base"]
    print(f"    {etype:12s}: {len(grp):4d} signals  wr={( r>0).mean()*100:.0f}%  avg={r.mean():+.2f}%")

print("\n  Anchored stop:")
for etype, grp in df.groupby("exit_anchored"):
    r = grp["ret_anchored"]
    print(f"    {etype:12s}: {len(grp):4d} signals  wr={(r>0).mean()*100:.0f}%  avg={r.mean():+.2f}%")

# ── SECTION 7: Year-by-year to check consistency ─────────────────────────────
print()
print("=" * 70)
print("  YEAR-BY-YEAR COMPARISON  (baseline vs best combined strategy)")
print("=" * 70)
df["year"] = df["date"].dt.year
mask_best = df["res_pct"].isna() | (df["res_pct"] >= 3.0)
for yr, grp in df.groupby("year"):
    base = grp["ret_base"]
    best = grp.loc[mask_best, "ret_anchored"]
    print(f"  {yr}: baseline wr={( base>0).mean()*100:.0f}% avg={base.mean():+.2f}% n={len(base):3d}  "
          f"| combined wr={(best>0).mean()*100 if len(best)>0 else 0:.0f}% avg={best.mean():+.2f}% n={len(best):3d}")

print("\nDone.")
