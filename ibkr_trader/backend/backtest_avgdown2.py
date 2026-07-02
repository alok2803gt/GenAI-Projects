"""
Diagnostic: why does "prior at loss + new BREAKOUT" never occur?
Also checks without regime gate.
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
    "PG","KO","PEP","WMT","XOM","CVX","COP","SLB","MPC","VLO","OXY",
    "BA","GE","CAT","HON","RTX","LMT","FDX","UPS","DE","UAL",
    "DIS","CMCSA","VZ","T","COIN","PLTR","UBER","RIVN","ROKU","HOOD","SOFI","PYPL","IBM",
    "RBLX","RCL","ABNB",
]))

BB_PERIOD=20; BB_STD=2.0; RSI_PERIOD=14; MAX_HOLD=30
HARD_STOP=0.07; TRAIL_PCT=0.05


def pct_b_series(c):
    sma=c.rolling(BB_PERIOD).mean(); std=c.rolling(BB_PERIOD).std(ddof=1)
    ub=sma+BB_STD*std; lb=sma-BB_STD*std; bw=ub-lb
    return ((c-lb)/bw*100).where(bw>0)


def rsi_series(c, n=RSI_PERIOD):
    d=c.diff(); gain=d.clip(lower=0).rolling(n).mean()
    loss=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+gain/loss.replace(0,np.nan))


def vol_90pct(v, w=90):
    return v.rolling(w, min_periods=30).quantile(0.90)


def classify(p):
    if p > 100: return "EXTENDED"
    if p >= 95: return "BREAKOUT"
    if p >= 75: return "PRE-BREAKOUT"
    if p >= 40: return "NEUTRAL"
    if p >= 25: return "WEAKENING"
    if p >= 0:  return "PRE-BREAKDOWN"
    return "BREAKDOWN"


def sim_exit(entry, closes, lows, highs):
    hsp = entry * (1 - HARD_STOP)
    for d, (fc, fl) in enumerate(zip(closes[:5], lows[:5]), 1):
        if fl <= hsp:
            return -HARD_STOP * 100, d, "hard_stop"
    peak = max([entry] + highs[:5])
    for d in range(5, MAX_HOLD):
        peak = max(peak, highs[d])
        ts   = peak * (1 - TRAIL_PCT)
        if lows[d] <= ts:
            return (ts - entry) / entry * 100, d + 1, "trail_stop"
    return (closes[MAX_HOLD - 1] - entry) / entry * 100, MAX_HOLD, "max_hold"


def build_signals(close_all, high_all, low_all, vol_all, available, regime_set=None):
    sigs = []
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
        states = pb.apply(lambda x: classify(x) if pd.notna(x) else "NEUTRAL")
        n = len(c); idx = c.index
        for i in range(BB_PERIOD + RSI_PERIOD + 10, n - MAX_HOLD - 2):
            if states.iloc[i] != "BREAKOUT" or states.iloc[i-1] != "PRE-BREAKOUT":
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
            if regime_set is not None and idx[i] not in regime_set:
                continue
            entry  = c.iloc[i]
            closes = [c.iloc[min(i+d, n-1)]  for d in range(1, MAX_HOLD+1)]
            lows   = [lo.iloc[min(i+d, n-1)] for d in range(1, MAX_HOLD+1)]
            highs  = [h.iloc[min(i+d, n-1)]  for d in range(1, MAX_HOLD+1)]
            ret, days, etype = sim_exit(entry, closes, lows, highs)
            sigs.append({
                "ticker":     tk,
                "entry_date": idx[i],
                "exit_date":  idx[min(i + days, n-1)],
                "entry":      entry,
                "ret":        ret,
                "days":       days,
                "exit_type":  etype,
            })
    return pd.DataFrame(sigs)


def find_dups(sig, loss_only=False):
    rows = []
    for _, new in sig.iterrows():
        tk = new["ticker"]
        nd = new["entry_date"]
        ne = new["entry"]
        priors = sig[
            (sig["ticker"] == tk) &
            (sig["entry_date"] < nd) &
            (sig["exit_date"] > nd)
        ]
        for _, pr in priors.iterrows():
            ret_now = (ne - pr["entry"]) / pr["entry"] * 100
            if loss_only and (ret_now >= 0 or ret_now <= -HARD_STOP * 100):
                continue
            rows.append({
                "ticker":           tk,
                "prior_entry":      pr["entry_date"],
                "new_date":         nd,
                "days_into":        (nd - pr["entry_date"]).days,
                "prior_ret_at_new": round(ret_now, 2),
                "new_ret":          round(new["ret"], 2),
                "prior_final_ret":  round(pr["ret"], 2),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["ticker", "prior_entry", "new_date"])


print("Downloading ...")
raw       = yf.download(TICKERS, start="2008-01-01", auto_adjust=True, progress=False, threads=True)
close_all = raw["Close"]
high_all  = raw["High"]
low_all   = raw["Low"]
vol_all   = raw["Volume"]
available = [t for t in TICKERS if t in close_all.columns]

spy_close  = close_all["SPY"].dropna()
spy_sma200 = spy_close.rolling(200).mean()
regime_ok  = set(d for d in close_all.index
                 if d in spy_close.index
                 and pd.notna(spy_sma200.get(d))
                 and spy_close[d] > spy_sma200[d])

# Build both signal sets
print("Building signals WITH regime gate ...")
sig_r  = build_signals(close_all, high_all, low_all, vol_all, available, regime_ok)
print(f"  Signals: {len(sig_r)}")

print("Building signals WITHOUT regime gate ...")
sig_nr = build_signals(close_all, high_all, low_all, vol_all, available, None)
print(f"  Signals: {len(sig_nr)}")

# ── All duplicates: P&L distribution ─────────────────────────────────────────
print("\n=== ALL duplicates (prior still open, any P&L) ===")
for label, sig in [("WITH regime gate", sig_r), ("NO regime gate", sig_nr)]:
    dups = find_dups(sig, loss_only=False)
    print(f"\n  {label}: {len(dups)} duplicates")
    if len(dups):
        bins = [(-100,-7),(-7,-3),(-3,0),(0,3),(3,10),(10,100)]
        for lo_b, hi_b in bins:
            mask = (dups["prior_ret_at_new"] >= lo_b) & (dups["prior_ret_at_new"] < hi_b)
            ct   = mask.sum()
            if ct:
                avg_new = dups.loc[mask, "new_ret"].mean()
                avg_pr  = dups.loc[mask, "prior_final_ret"].mean()
                print(f"    {lo_b:+4.0f}% to {hi_b:+3.0f}%: {ct:3d} cases  "
                      f"avg_new={avg_new:+.2f}%  avg_prior_final={avg_pr:+.2f}%")

# ── Losing-position duplicates ────────────────────────────────────────────────
print("\n=== LOSING POSITION duplicates (prior at -0.1% to -6.9%) ===")
for label, sig in [("WITH regime gate", sig_r), ("NO regime gate", sig_nr)]:
    ld = find_dups(sig, loss_only=True)
    print(f"\n  {label}: {len(ld)} scenarios")
    if len(ld):
        print(f"    Loss at re-alert: avg={ld['prior_ret_at_new'].mean():.2f}%")
        blend = (ld["prior_final_ret"] + ld["new_ret"]) / 2
        print(f"    A ignore:  avg={ld['prior_final_ret'].mean():+.3f}%  WR={(ld['prior_final_ret']>0).mean()*100:.0f}%")
        print(f"    B avgdown: avg={blend.mean():+.3f}%                WR={(blend>0).mean()*100:.0f}%")
        print(f"    new leg:   avg={ld['new_ret'].mean():+.3f}%         WR={(ld['new_ret']>0).mean()*100:.0f}%")
        print(f"\n    Full table:")
        print(ld.sort_values("prior_ret_at_new").to_string(index=False))
