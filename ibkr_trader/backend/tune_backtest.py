"""
Grid search over CSP backtest parameters to find optimal profit_target_pct × stop_loss_mult.
Runs fully offline — yfinance only, no IBKR needed.
"""

import asyncio, math, itertools, sys
import numpy as np
import yfinance as yf

# ── Black-Scholes helpers ────────────────────────────────────────────────────
def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))

def bs_put(S, K, T, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S/K) + 0.5*sigma**2*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    return K*_norm_cdf(-d2) - S*_norm_cdf(-d1)

# ── Single-ticker backtest ────────────────────────────────────────────────────
def backtest_ticker(closes, dates, weeks, profit_target_pct, stop_loss_mult):
    trades = []
    i = 20
    while i < len(closes) - 6 and len(trades) < weeks:
        rv    = closes[i-20:i]
        lr    = np.log(rv[1:] / rv[:-1])
        sigma = float(np.std(lr) * np.sqrt(252))
        if sigma <= 0:
            i += 5; continue
        S    = float(closes[i])
        T    = 5 / 252
        K    = round(S * np.exp(-0.842 * sigma * math.sqrt(T)), 0)
        prem = bs_put(S, K, T, sigma)
        if prem < 0.05:
            i += 5; continue
        tgt  = prem * (1 - profit_target_pct)
        stp  = prem * (1 + stop_loss_mult) if stop_loss_mult > 0 else float('inf')
        exit_pnl = 0.0; win = False
        for j in range(i+1, min(i+6, len(closes))):
            Sj  = float(closes[j])
            Tj  = max((min(i+5, len(closes)-1) - j) / 252, 0.001)
            pj  = bs_put(Sj, K, Tj, sigma)
            gain = (prem - pj) * 100
            if pj <= tgt:
                exit_pnl, win = gain, True; break
            if pj >= stp:
                exit_pnl = gain; break
        else:
            Se       = float(closes[min(i+5, len(closes)-1)])
            exit_pnl = (prem - max(0, K - Se)) * 100
            win      = bool(Se >= K)
        trades.append({"exit_pnl": round(float(exit_pnl), 2), "win": bool(win), "prem": round(float(prem),2)})
        i += 5

    if not trades:
        return None
    pnls    = np.array([t["exit_pnl"] for t in trades])
    wins    = [t for t in trades if t["win"]]
    losses  = [t for t in trades if not t["win"]]
    avg_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls))
    cum     = np.cumsum(pnls)
    dd      = cum - np.maximum.accumulate(cum)
    return {
        "total_pnl":    round(float(np.sum(pnls)), 0),
        "win_rate":     round(len(wins)/len(trades)*100, 1),
        "sharpe":       round(avg_pnl/std_pnl*math.sqrt(52), 2) if std_pnl > 0 else 0,
        "max_dd":       round(float(np.min(dd)), 0),
        "avg_win":      round(float(np.mean([t["exit_pnl"] for t in wins])), 0) if wins else 0,
        "avg_loss":     round(float(np.mean([t["exit_pnl"] for t in losses])), 0) if losses else 0,
        "n_trades":     len(trades),
    }

# ── Grid search ──────────────────────────────────────────────────────────────
TICKERS  = ["AMD", "AAPL", "NVDA", "MSFT", "SPY", "QQQ", "LLY", "META"]
WEEKS    = 52          # 1 year of weekly trades for statistical significance

PROFIT_TARGETS = [0.25, 0.35, 0.50, 0.65, 0.75, 0.90]  # % of premium to capture
STOP_MULTS     = [1.0, 1.5, 2.0, 3.0, 5.0, 0]           # 0 = hold to expiry (no stop)

def run():
    print("Downloading price data …")
    data = {}
    for t in TICKERS:
        try:
            hist   = yf.Ticker(t).history(period="2y", interval="1d")
            closes = hist["Close"].values.astype(float)
            dates  = [str(d.date()) for d in hist.index]
            if len(closes) >= 60:
                data[t] = (closes, dates)
                print(f"  {t}: {len(closes)} bars")
        except Exception as e:
            print(f"  {t}: SKIP ({e})")

    print(f"\nRunning {len(PROFIT_TARGETS) * len(STOP_MULTS)} parameter combos × {len(data)} tickers …\n")

    results = []
    for pt, sl in itertools.product(PROFIT_TARGETS, STOP_MULTS):
        combo_pnl    = []
        combo_sharpe = []
        combo_wr     = []
        combo_dd     = []
        for ticker, (closes, dates) in data.items():
            r = backtest_ticker(closes, dates, WEEKS, pt, sl)
            if r:
                combo_pnl.append(r["total_pnl"])
                combo_sharpe.append(r["sharpe"])
                combo_wr.append(r["win_rate"])
                combo_dd.append(r["max_dd"])

        if not combo_pnl:
            continue

        results.append({
            "pt":        pt,
            "sl":        sl,
            "avg_pnl":   round(sum(combo_pnl) / len(combo_pnl), 0),
            "total_pnl": round(sum(combo_pnl), 0),
            "avg_sharpe":round(sum(combo_sharpe) / len(combo_sharpe), 2),
            "avg_wr":    round(sum(combo_wr) / len(combo_wr), 1),
            "avg_dd":    round(sum(combo_dd) / len(combo_dd), 0),
        })

    # ── Sort by Sharpe (risk-adjusted), then total P&L ──────────────────────
    by_sharpe = sorted(results, key=lambda x: (x["avg_sharpe"], x["total_pnl"]), reverse=True)
    by_pnl    = sorted(results, key=lambda x: x["total_pnl"], reverse=True)

    # ── Print full grid ──────────────────────────────────────────────────────
    print(f"{'PT%':>5} {'SL×':>5}  {'AvgP&L':>8} {'TotP&L':>8} {'Sharpe':>7} {'WinRate':>8} {'MaxDD':>8}")
    print("-" * 62)
    for r in sorted(results, key=lambda x: (x["pt"], x["sl"])):
        sl_label = "NONE" if r["sl"] == 0 else f"{r['sl']}×"
        pnl_flag = " << BEST PNL"    if r == by_pnl[0]    else ""
        shr_flag = " << BEST SHARPE" if r == by_sharpe[0] else ""
        print(f"{int(r['pt']*100):>4}%  {sl_label:>5}  "
              f"${r['avg_pnl']:>7,.0f}  ${r['total_pnl']:>7,.0f}  "
              f"{r['avg_sharpe']:>6.2f}  {r['avg_wr']:>6.1f}%  "
              f"${r['avg_dd']:>7,.0f}{pnl_flag}{shr_flag}")

    # ── Top 5 by Sharpe ─────────────────────────────────────────────────────
    print("\n── Top 5 by Sharpe (risk-adjusted) ──")
    for r in by_sharpe[:5]:
        sl_label = "hold-to-expiry" if r["sl"] == 0 else f"{r['sl']}× stop"
        print(f"  Target {int(r['pt']*100)}% profit, {sl_label}: "
              f"Sharpe {r['avg_sharpe']:+.2f}, P&L ${r['total_pnl']:+,.0f}, "
              f"WR {r['avg_wr']}%, MaxDD ${r['avg_dd']:,.0f}")

    # ── Top 5 by total P&L ──────────────────────────────────────────────────
    print("\n── Top 5 by Total P&L ──")
    for r in by_pnl[:5]:
        sl_label = "hold-to-expiry" if r["sl"] == 0 else f"{r['sl']}× stop"
        print(f"  Target {int(r['pt']*100)}% profit, {sl_label}: "
              f"P&L ${r['total_pnl']:+,.0f}, Sharpe {r['avg_sharpe']:+.2f}, "
              f"WR {r['avg_wr']}%, MaxDD ${r['avg_dd']:,.0f}")

    # ── Recommendation ──────────────────────────────────────────────────────
    best = by_sharpe[0]
    sl_best = "NONE (hold-to-expiry)" if best['sl'] == 0 else f"{best['sl']}x stop"
    print(f"""
+======================================================+
|  RECOMMENDED SETTINGS                               |
|  profit_target_pct : {int(best['pt']*100)}%                            |
|  stop_loss_mult    : {sl_best:<30} |
|  Avg Sharpe        : {best['avg_sharpe']:+.2f}                          |
|  Total P&L         : ${best['total_pnl']:+,.0f} ({len(data)} tickers x {WEEKS}wks)  |
|  Win Rate          : {best['avg_wr']}%                           |
+======================================================+""")

    return by_sharpe[0]

if __name__ == "__main__":
    run()
