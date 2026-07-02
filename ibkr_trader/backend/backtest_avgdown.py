"""
Targeted backtest: what to do when a new BREAKOUT alert fires
while existing position is at a LOSS (between 0% and -7% hard stop).
Options: A) ignore, B) average down, C) replace (close + reopen).
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
BULLISH={"BREAKOUT","EXTENDED","PRE-BREAKOUT"}

def pct_b_series(c):
    sma=c.rolling(BB_PERIOD).mean(); std=c.rolling(BB_PERIOD).std(ddof=1)
    ub=sma+BB_STD*std; lb=sma-BB_STD*std; bw=ub-lb
    return ((c-lb)/bw*100).where(bw>0)
def rsi_series(c, n=RSI_PERIOD):
    d=c.diff(); gain=d.clip(lower=0).rolling(n).mean()
    loss=(-d.clip(upper=0)).rolling(n).mean()
    return 100-100/(1+gain/loss.replace(0,np.nan))
def vol_90pct(v, w=90): return v.rolling(w,min_periods=30).quantile(0.90)
def classify(p):
    if p>100: return "EXTENDED"
    if p>=95: return "BREAKOUT"
    if p>=75: return "PRE-BREAKOUT"
    if p>=40: return "NEUTRAL"
    if p>=25: return "WEAKENING"
    if p>=0:  return "PRE-BREAKDOWN"
    return "BREAKDOWN"


def sim_exit(entry, closes, lows, highs, hard_stop=HARD_STOP, trail=TRAIL_PCT, max_hold=MAX_HOLD):
    """Run phase1+phase2 exit logic from a given entry. Returns (ret%, days, exit_type)."""
    n = len(closes)
    hsp = entry * (1 - hard_stop)
    # Phase 1: days 1-5 hard stop
    for d, (fc, fl) in enumerate(zip(closes[:5], lows[:5]), 1):
        if fl <= hsp:
            return -hard_stop * 100, d, "hard_stop"
    # Phase 2: trailing from day 5
    peak = max([entry] + highs[:5])
    for d in range(5, min(max_hold, n)):
        peak = max(peak, highs[d])
        ts   = peak * (1 - trail)
        if lows[d] <= ts:
            return (ts - entry) / entry * 100, d + 1, "trail_stop"
    return (closes[min(max_hold-1, n-1)] - entry) / entry * 100, max_hold, "max_hold"


print("Downloading data ...")
raw       = yf.download(TICKERS, start="2008-01-01", auto_adjust=True, progress=False, threads=True)
close_all = raw["Close"]
high_all  = raw["High"]
low_all   = raw["Low"]
vol_all   = raw["Volume"]
available = [t for t in TICKERS if t in close_all.columns]

spy_close  = close_all["SPY"].dropna()
spy_sma200 = spy_close.rolling(200).mean()
regime_ok  = set(d for d in close_all.index
                 if d in spy_close.index and pd.notna(spy_sma200.get(d))
                 and spy_close[d] > spy_sma200[d])

# Build full signal list first (same as prior backtests)
all_signals = []
for tk in available:
    c  = close_all[tk].dropna()
    h  = high_all[tk].reindex(c.index)
    lo = low_all[tk].reindex(c.index)
    v  = vol_all[tk].reindex(c.index)
    if len(c) < 200: continue
    pb    = pct_b_series(c)
    rsi   = rsi_series(c)
    v90   = vol_90pct(v)
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    states = pb.apply(lambda x: classify(x) if pd.notna(x) else "NEUTRAL")
    n = len(c); idx = c.index

    for i in range(BB_PERIOD+RSI_PERIOD+10, n-MAX_HOLD-2):
        if states.iloc[i]!="BREAKOUT" or states.iloc[i-1]!="PRE-BREAKOUT": continue
        rv=rsi.iloc[i]
        if pd.isna(rv) or rv<55: continue
        vi=v.iloc[i]; v90i=v90.iloc[i]
        if pd.isna(vi) or pd.isna(v90i) or vi<v90i: continue
        if not (c.iloc[i]>sma20.iloc[i] and c.iloc[i]>sma50.iloc[i]): continue
        if idx[i] not in regime_ok: continue

        entry  = c.iloc[i]
        closes = [c.iloc[min(i+d,n-1)]  for d in range(1, MAX_HOLD+1)]
        lows   = [lo.iloc[min(i+d,n-1)] for d in range(1, MAX_HOLD+1)]
        highs  = [h.iloc[min(i+d,n-1)]  for d in range(1, MAX_HOLD+1)]

        ret, days, exit_type = sim_exit(entry, closes, lows, highs)

        # Precompute per-day close/return from entry for duplicate analysis
        daily_ret = [(c.iloc[min(i+d,n-1)] - entry)/entry*100 for d in range(1, MAX_HOLD+1)]

        # Actual calendar exit date (index position i + days)
        exit_date = idx[min(i + days, n - 1)]

        all_signals.append({
            "ticker":     tk,
            "entry_idx":  i,
            "entry_date": idx[i],
            "exit_date":  exit_date,
            "entry":      entry,
            "ret":        ret,
            "days":       days,
            "exit_type":  exit_type,
            "closes":     closes,
            "lows":       lows,
            "highs":      highs,
            "daily_ret":  daily_ret,
            "c_series":   c,
            "lo_series":  lo,
            "hi_series":  h,
            "idx":        idx,
            "n":          n,
        })

print(f"Total regime-gated signals: {len(all_signals)}")

# ── Find duplicate-while-losing scenarios ─────────────────────────────────────
losing_dup_scenarios = []

sig_df = pd.DataFrame([{
    "ticker":     s["ticker"],
    "entry_date": s["entry_date"],
    "entry":      s["entry"],
    "ret":        s["ret"],
    "days":       s["days"],
} for s in all_signals])

for new_sig in all_signals:
    tk         = new_sig["ticker"]
    new_date   = new_sig["entry_date"]
    new_entry  = new_sig["entry"]

    # Find any prior open signal for same ticker
    prior_sigs = [s for s in all_signals
                  if s["ticker"] == tk
                  and s["entry_date"] < new_date
                  and s["entry_date"] + pd.Timedelta(days=MAX_HOLD*1.5) > new_date]

    for prior in prior_sigs:
        # Days into prior position when new signal fires
        days_into = (new_date - prior["entry_date"]).days

        # KEY CHECK: prior position must still be open on the new signal date
        # (exit_date is when hard stop / trail stop / max_hold actually fired)
        if new_date >= prior["exit_date"]:
            continue   # prior already exited before new signal — not a duplicate

        # What was the prior position's P&L on the new signal date?
        close_on_new_date = new_sig["entry"]   # new signal fires at today's close
        prior_ret_now = (close_on_new_date - prior["entry"]) / prior["entry"] * 100

        # Only care about losing positions (between -0.1% and -6.9%)
        if prior_ret_now >= 0 or prior_ret_now <= -HARD_STOP * 100:
            continue   # in profit or already stopped out — not the scenario we want

        # ── What prior position eventually returned ─────────────────────────
        # We already computed prior's full exit in all_signals
        prior_final_ret = prior["ret"]

        # ── Option A: Ignore new signal, hold original ──────────────────────
        ret_A = prior_final_ret

        # ── Option B: Average down (same $ added at new_entry price) ────────
        # Blended avg cost = (prior_entry + new_entry) / 2
        # Combined position exits when EITHER hits stop or trail triggers on combined
        # Simplify: track both legs separately, use same phase1/phase2 logic
        # For the new leg: sim_exit from new_entry forward
        ni = new_sig["entry_idx"]
        nn = new_sig["n"]
        new_closes = [new_sig["c_series"].iloc[min(ni+d,nn-1)] for d in range(1,MAX_HOLD+1)]
        new_lows   = [new_sig["lo_series"].iloc[min(ni+d,nn-1)] for d in range(1,MAX_HOLD+1)]
        new_highs  = [new_sig["hi_series"].iloc[min(ni+d,nn-1)] for d in range(1,MAX_HOLD+1)]
        new_ret, new_days, new_exit = sim_exit(new_entry, new_closes, new_lows, new_highs)

        # Blended return = simple average of both legs (equal $ in each)
        ret_B_blended = (prior_final_ret + new_ret) / 2

        # ── Option C: Replace (close old at new_entry price, open fresh) ────
        # Close prior at current price = prior_ret_now (realized loss)
        # Open new position with full phase1/phase2 from new_entry
        # Net: realize the existing loss, start fresh
        ret_C_close_loss  = prior_ret_now   # lock in this loss
        ret_C_new_position = new_ret
        # Combined P&L assuming equal $ in each trade:
        # Old trade: realized at prior_ret_now
        # New trade: goes through its exit
        ret_C = (ret_C_close_loss + ret_C_new_position) / 2   # 50/50 allocation

        losing_dup_scenarios.append({
            "ticker":            tk,
            "prior_entry_date":  prior["entry_date"],
            "new_signal_date":   new_date,
            "days_into_prior":   days_into,
            "prior_entry":       round(prior["entry"], 2),
            "new_entry":         round(new_entry, 2),
            "prior_ret_at_new":  round(prior_ret_now, 2),   # loss at time of new signal
            "A_ignore_ret":      round(ret_A, 3),
            "B_avgdown_ret":     round(ret_B_blended, 3),
            "C_replace_ret":     round(ret_C, 3),
            "new_leg_ret":       round(new_ret, 3),
            "prior_final_ret":   round(prior_final_ret, 3),
        })

if not losing_dup_scenarios:
    print("\nNo losing-position duplicate scenarios found.")
else:
    ld = pd.DataFrame(losing_dup_scenarios)
    # deduplicate — same (prior, new) pair might appear once
    ld = ld.drop_duplicates(subset=["ticker","prior_entry_date","new_signal_date"])

    print(f"\nScenarios: new BREAKOUT fires while prior position is at a LOSS: {len(ld)}")
    print(f"Loss range at time of re-alert: "
          f"min={ld['prior_ret_at_new'].min():.2f}%  "
          f"max={ld['prior_ret_at_new'].max():.2f}%  "
          f"avg={ld['prior_ret_at_new'].mean():.2f}%")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'Option':<45} {'WR':>6}  {'Avg':>7}  {'Median':>7}  {'Worst':>8}  {'Best':>8}")
    print("-"*85)
    for label, col in [
        ("A — Ignore (keep original, ride it out)",    "A_ignore_ret"),
        ("B — Average down (add equal $ at new price)","B_avgdown_ret"),
        ("C — Replace (close loss, open fresh trade)", "C_replace_ret"),
        ("  [new leg alone for reference]",            "new_leg_ret"),
    ]:
        r  = ld[col]
        wr = (r>0).mean()*100
        print(f"  {label:<43} {wr:>5.1f}%  {r.mean():>+6.3f}%  "
              f"{r.median():>+6.3f}%  {r.min():>+7.3f}%  {r.max():>+7.3f}%")

    # ── Who wins on each scenario? ───────────────────────────────────────────
    ld["best"] = ld[["A_ignore_ret","B_avgdown_ret","C_replace_ret"]].idxmax(axis=1).map({
        "A_ignore_ret": "A-ignore",
        "B_avgdown_ret": "B-avgdown",
        "C_replace_ret": "C-replace",
    })
    bv = ld["best"].value_counts()
    print(f"\n  Best option per scenario:")
    for k in ["A-ignore","B-avgdown","C-replace"]:
        print(f"    {k}: {bv.get(k,0)} of {len(ld)} ({bv.get(k,0)/len(ld)*100:.0f}%)")

    # ── Bucket by how far underwater at re-alert ─────────────────────────────
    print(f"\n  Breakdown by loss depth at re-alert:")
    bins   = [(-7,  -5,  "deep  (-5% to -7%)"),
              (-5,  -3,  "mid   (-3% to -5%)"),
              (-3,  -0.1,"light (-0.1% to -3%)")]
    print(f"  {'Loss bucket':<25} {'n':>4}  {'A_avg':>8}  {'B_avg':>8}  {'C_avg':>8}  {'best':>12}")
    for lo_b, hi_b, lbl in bins:
        mask = (ld["prior_ret_at_new"] >= lo_b) & (ld["prior_ret_at_new"] < hi_b)
        g = ld[mask]
        if len(g) == 0: continue
        best_opt = g[["A_ignore_ret","B_avgdown_ret","C_replace_ret"]].mean().idxmax()
        best_lbl = {"A_ignore_ret":"A-ignore","B_avgdown_ret":"B-avgdown","C_replace_ret":"C-replace"}[best_opt]
        print(f"  {lbl:<25} {len(g):>4}  "
              f"{g['A_ignore_ret'].mean():>+7.3f}%  "
              f"{g['B_avgdown_ret'].mean():>+7.3f}%  "
              f"{g['C_replace_ret'].mean():>+7.3f}%  "
              f"{best_lbl:>12}")

    # ── Full scenario table ───────────────────────────────────────────────────
    print(f"\n  All scenarios detail:")
    show_cols = ["ticker","prior_entry_date","new_signal_date","days_into_prior",
                 "prior_ret_at_new","A_ignore_ret","B_avgdown_ret","C_replace_ret","best"]
    print(ld[show_cols].sort_values("prior_ret_at_new").to_string(index=False))

    # ── Avg-down risk: what if BOTH legs hit the hard stop? ──────────────────
    both_stopped = ld[(ld["prior_final_ret"] == -7.0) & (ld["new_leg_ret"] == -7.0)]
    print(f"\n  Catastrophic avg-down: both legs stop out at -7%: {len(both_stopped)} cases "
          f"({len(both_stopped)/len(ld)*100:.0f}%)")
    if len(both_stopped):
        print(f"    These cost -7% on each leg = -14% combined on 2x capital")
        print(both_stopped[["ticker","prior_entry_date","new_signal_date","prior_ret_at_new"]].to_string(index=False))
