"""
Supplemental analysis:
1. Kelly position sizing from actual backtest win/loss distribution
2. Max concurrent open positions (what's the peak overlap?)
3. Duplicate-signal scenarios (how often does a ticker re-alert while position open?)
Standalone — no project imports, no code changes.
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
def vol_90pct(v, window=90): return v.rolling(window,min_periods=30).quantile(0.90)
def classify(p):
    if p>100: return "EXTENDED"
    if p>=95: return "BREAKOUT"
    if p>=75: return "PRE-BREAKOUT"
    if p>=40: return "NEUTRAL"
    if p>=25: return "WEAKENING"
    if p>=0:  return "PRE-BREAKDOWN"
    return "BREAKDOWN"

print("Downloading data ...")
raw=yf.download(TICKERS,start="2008-01-01",auto_adjust=True,progress=False,threads=True)
close_all=raw["Close"]; high_all=raw["High"]; low_all=raw["Low"]; vol_all=raw["Volume"]
available=[t for t in TICKERS if t in close_all.columns]
spy_close=close_all["SPY"].dropna()
spy_sma200=spy_close.rolling(200).mean()
regime_ok=set(d for d in close_all.index
               if d in spy_close.index and pd.notna(spy_sma200.get(d))
               and spy_close[d] > spy_sma200[d])

results=[]
for tk in available:
    c=close_all[tk].dropna(); h=high_all[tk].reindex(c.index)
    lo=low_all[tk].reindex(c.index); v=vol_all[tk].reindex(c.index)
    if len(c)<200: continue
    pb=pct_b_series(c); rsi=rsi_series(c); v90=vol_90pct(v)
    sma20=c.rolling(20).mean(); sma50=c.rolling(50).mean()
    states=pb.apply(lambda x: classify(x) if pd.notna(x) else "NEUTRAL")
    n=len(c); idx=c.index
    for i in range(BB_PERIOD+RSI_PERIOD+10, n-MAX_HOLD-2):
        if states.iloc[i]!="BREAKOUT" or states.iloc[i-1]!="PRE-BREAKOUT": continue
        rv=rsi.iloc[i]
        if pd.isna(rv) or rv<55: continue
        vi=v.iloc[i]; v90i=v90.iloc[i]
        if pd.isna(vi) or pd.isna(v90i) or vi<v90i: continue
        if not (c.iloc[i]>sma20.iloc[i] and c.iloc[i]>sma50.iloc[i]): continue
        if idx[i] not in regime_ok: continue   # regime gate

        entry=c.iloc[i]
        closes=[c.iloc[min(i+d,n-1)] for d in range(1,MAX_HOLD+1)]
        lows  =[lo.iloc[min(i+d,n-1)] for d in range(1,MAX_HOLD+1)]
        highs =[h.iloc[min(i+d,n-1)] for d in range(1,MAX_HOLD+1)]

        hard_stop_price=entry*(1-HARD_STOP)
        stopped=False; final_ret=(closes[MAX_HOLD-1]-entry)/entry*100
        exit_day=MAX_HOLD; exit_type="max_hold"
        for d2,(fc,fl) in enumerate(zip(closes[:5],lows[:5]),1):
            if fl<=hard_stop_price:
                final_ret=-7.0; exit_day=d2; exit_type="hard_stop"; stopped=True; break
        if not stopped:
            peak=max([entry]+highs[:5])
            for d2 in range(5,MAX_HOLD):
                peak=max(peak,highs[d2]); ts=peak*(1-TRAIL_PCT)
                if lows[d2]<=ts:
                    final_ret=(ts-entry)/entry*100; exit_day=d2+1; exit_type="trail_stop"; break

        # Track entry and exit dates for concurrency analysis
        entry_date = idx[i]
        exit_date  = idx[min(i+exit_day, n-1)]

        results.append({
            "ticker":     tk,
            "entry_date": entry_date,
            "exit_date":  exit_date,
            "exit_day":   exit_day,
            "ret":        round(final_ret,3),
            "win":        final_ret > 0,
            "exit_type":  exit_type,
        })

df=pd.DataFrame(results)
print(f"Signals (regime-gated): {len(df)}\n")

# ═══════════════════════════════════════════════════════
# 1. KELLY SIZING
# ═══════════════════════════════════════════════════════
wins   = df[df["win"]]["ret"]
losses = df[~df["win"]]["ret"]
wr     = df["win"].mean()
avg_win  = wins.mean()
avg_loss = losses.mean()   # negative number
wl_ratio = avg_win / abs(avg_loss)

# Full Kelly: f* = (p*b - q) / b  where b = avg_win/avg_loss_abs
b = wl_ratio
q = 1 - wr
kelly_full   = (wr * b - q) / b
kelly_half   = kelly_full / 2
kelly_qtr    = kelly_full / 4

print("=" * 60)
print("  POSITION SIZING")
print("=" * 60)
print(f"\n  Empirical stats (regime-gated, 18yr, n={len(df)}):")
print(f"    Win rate      : {wr*100:.1f}%")
print(f"    Avg win       : +{avg_win:.2f}%")
print(f"    Avg loss      : {avg_loss:.2f}%")
print(f"    Win/Loss ratio: {wl_ratio:.2f}x")
print(f"    Expectancy    : {wr*avg_win + (1-wr)*avg_loss:+.3f}% per trade")

print(f"\n  Kelly fractions:")
print(f"    Full Kelly  : {kelly_full*100:.1f}% of capital per trade")
print(f"    Half Kelly  : {kelly_half*100:.1f}% of capital per trade")
print(f"    Quarter Kelly: {kelly_qtr*100:.1f}% of capital per trade")

# Show what these mean in dollar terms at various capital sizes
print(f"\n  Dollar sizing per trade:")
print(f"  {'Capital':>10}  {'Full-K':>8}  {'Half-K':>8}  {'Qtr-K':>8}  {'Fixed-2%':>10}  {'Fixed-1%':>10}")
for cap in [25_000, 50_000, 100_000]:
    fk  = cap * kelly_full
    hk  = cap * kelly_half
    qk  = cap * kelly_qtr
    f2  = cap * 0.02
    f1  = cap * 0.01
    print(f"  ${cap:>9,}  ${fk:>7,.0f}  ${hk:>7,.0f}  ${qk:>7,.0f}  ${f2:>9,.0f}  ${f1:>9,.0f}")

# Risk of ruin approximation
bad_yr_trades = 40
bad_yr_wins   = int(bad_yr_trades * 0.32)
bad_yr_losses = bad_yr_trades - bad_yr_wins
net_pnl = (bad_yr_wins*(kelly_half*avg_win) + bad_yr_losses*(kelly_half*avg_loss))*100
print(f"\n  Worst-year simulation (~{bad_yr_trades} trades, 32% WR like 2018):")
print(f"    Half-Kelly {kelly_half*100:.1f}%: {bad_yr_wins} wins avg+{avg_win:.1f}%, {bad_yr_losses} losses avg{avg_loss:.1f}%")
print(f"    Net P&L on capital: {net_pnl:+.1f}%")

# ═══════════════════════════════════════════════════════
# 2. CONCURRENT POSITION ANALYSIS
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  CONCURRENT POSITIONS")
print("=" * 60)

# For each trading day, count how many positions are open
all_dates = pd.bdate_range(df["entry_date"].min(), df["exit_date"].max())
daily_open = []
for d in all_dates:
    open_pos = df[(df["entry_date"] <= d) & (df["exit_date"] > d)]
    daily_open.append({"date": d, "open": len(open_pos),
                        "tickers": ",".join(open_pos["ticker"].tolist())})

conc = pd.DataFrame(daily_open)

print(f"\n  Concurrent open positions over {len(all_dates)} trading days:")
print(f"    Average  : {conc['open'].mean():.1f}")
print(f"    Median   : {conc['open'].median():.0f}")
print(f"    75th pct : {conc['open'].quantile(0.75):.0f}")
print(f"    90th pct : {conc['open'].quantile(0.90):.0f}")
print(f"    95th pct : {conc['open'].quantile(0.95):.0f}")
print(f"    Max ever : {conc['open'].max()}")

# Find peak days
peak_days = conc.nlargest(5, "open")
print(f"\n  Peak concurrent days (top 5):")
for _, row in peak_days.iterrows():
    print(f"    {row['date'].date()}  {int(row['open'])} open: {row['tickers']}")

# Distribution
print(f"\n  Distribution of concurrent open count:")
dist = conc["open"].value_counts().sort_index()
for k, v in dist.items():
    pct = v / len(conc) * 100
    bar = "#" * max(1, int(pct / 2))
    print(f"    {k:2d} open: {v:5d} days ({pct:5.1f}%)  {bar}")

# ═══════════════════════════════════════════════════════
# 3. DUPLICATE / RE-ALERT ANALYSIS
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  DUPLICATE SIGNAL ANALYSIS")
print("=" * 60)

# Find signals for a ticker while a prior position is still open
df_sorted = df.sort_values("entry_date").reset_index(drop=True)

duplicate_scenarios = []
for i, row in df_sorted.iterrows():
    tk = row["ticker"]
    entry = row["entry_date"]
    # Any earlier signal for same ticker still open?
    prior = df_sorted[
        (df_sorted["ticker"] == tk) &
        (df_sorted["entry_date"] < entry) &
        (df_sorted["exit_date"] > entry)
    ]
    if len(prior) > 0:
        pr = prior.iloc[-1]   # most recent prior
        days_into_prior = (entry - pr["entry_date"]).days
        phase = "phase1" if days_into_prior <= 7 else "phase2"
        duplicate_scenarios.append({
            "ticker":          tk,
            "new_signal_date": entry,
            "prior_entry":     pr["entry_date"],
            "days_into_prior": days_into_prior,
            "phase":           phase,
            "prior_ret_sofar": round((close_all[tk].get(entry, np.nan) /
                                      close_all[tk].get(pr["entry_date"], np.nan) - 1) * 100, 2)
                               if tk in close_all.columns else np.nan,
            "new_signal_ret":  row["ret"],
            "prior_final_ret": pr["ret"],
        })

dups = pd.DataFrame(duplicate_scenarios) if duplicate_scenarios else pd.DataFrame()
if len(dups):
    print(f"\n  Re-alerts while position still open: {len(dups)} occurrences")
    print(f"  Tickers most prone to re-alerting:")
    print(dups["ticker"].value_counts().head(10).to_string())

    print(f"\n  Phase breakdown:")
    for ph, g in dups.groupby("phase"):
        print(f"    {ph}: {len(g)} re-alerts  "
              f"avg days_into_prior={g['days_into_prior'].mean():.1f}  "
              f"avg prior_ret_sofar={g['prior_ret_sofar'].mean():+.2f}%  "
              f"avg new_signal_ret={g['new_signal_ret'].mean():+.3f}%  "
              f"avg prior_final_ret={g['prior_final_ret'].mean():+.3f}%")

    # What was the new signal return vs the existing position final return?
    dups["new_better"] = dups["new_signal_ret"] > dups["prior_final_ret"]
    print(f"\n  If we had REPLACED prior position with new signal:")
    print(f"    New signal better : {dups['new_better'].sum()} ({dups['new_better'].mean()*100:.0f}%)")
    print(f"    Prior pos better  : {(~dups['new_better']).sum()} ({(~dups['new_better']).mean()*100:.0f}%)")
    print(f"    Avg new signal ret: {dups['new_signal_ret'].mean():+.3f}%")
    print(f"    Avg prior final ret:{dups['prior_final_ret'].mean():+.3f}%")

    print(f"\n  If we had ADDED (pyramided) into the position:")
    dups["blended"] = (dups["prior_ret_sofar"] + dups["new_signal_ret"]) / 2
    print(f"    Avg blended return: {dups['blended'].mean():+.3f}%")
    print(f"    vs just ignoring (keeping prior): {dups['prior_final_ret'].mean():+.3f}%")

    print(f"\n  Sample re-alert scenarios (first 15):")
    sample_cols = ["ticker","prior_entry","new_signal_date","days_into_prior",
                   "phase","prior_ret_sofar","new_signal_ret","prior_final_ret"]
    print(dups[sample_cols].head(15).to_string(index=False))
else:
    print("  No duplicate signals found (each ticker signal is unique in this run)")
