"""
Real-price-action backtest for a weekly iron condor premium-collection
strategy on GOOG/AAPL/COST -- these tickers only have Friday-cadence
weekly expiries (confirmed live 2026-08-19, no true 0DTE), so this tests
entering Monday, holding through Friday's close, on REAL historical daily
price data.

APPROXIMATIONS STATED UP FRONT:
  1. Premium/credit is MODELED via Black-Scholes using REALIZED volatility
     (20-day historical, annualized) as an implied-vol proxy at entry --
     no historical options premium data exists for these individual names
     (same limitation noted in this account's original spy_0dte_backtest.py
     before the real-price version was built for SPY specifically via
     Massive/Polygon flat files, which aren't available here).
  2. Win/loss determination uses REAL historical daily closes -- a
     candidate week's outcome is real: did the actual price stay inside
     the short strikes through Friday's close, not modeled.
  3. Entry Monday's open, exit Friday's close -- a full 5-trading-day
     hold, no early profit-target/stop-loss simulated (that would need
     intraday data this backtest doesn't have -- stated as a real gap,
     not modeled around).
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

RISK_FREE = 0.045

def bs_put_price(S, K, T, sigma, r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_call_price(S, K, T, sigma, r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def run(ticker):
    hist = yf.download(ticker, period="3y", interval="1d", auto_adjust=True, progress=False)
    closes = hist["Close"].squeeze()
    closes.index = pd.to_datetime(closes.index)

    # 20-day realized vol (annualized), as of each Monday
    log_ret = np.log(closes / closes.shift(1))
    realized_vol = log_ret.rolling(20).std() * np.sqrt(252)

    mondays = closes.index[closes.index.weekday == 0]
    results = []
    for otm_pct in (0.03, 0.05, 0.07, 0.10):
        for width_pct in (0.02, 0.03, 0.05):
            trades = []
            for mon in mondays:
                if mon not in closes.index or mon not in realized_vol.index:
                    continue
                sigma = realized_vol.loc[mon]
                if pd.isna(sigma) or sigma <= 0:
                    continue
                S0 = closes.loc[mon]
                fri_candidates = closes.index[(closes.index > mon) & (closes.index <= mon + pd.Timedelta(days=5))]
                if len(fri_candidates) == 0:
                    continue
                fri = fri_candidates[-1]
                S_fri = closes.loc[fri]
                T = 5/252

                short_put_k  = S0*(1-otm_pct)
                long_put_k   = S0*(1-otm_pct-width_pct)
                short_call_k = S0*(1+otm_pct)
                long_call_k  = S0*(1+otm_pct+width_pct)

                credit = (bs_put_price(S0, short_put_k, T, sigma) - bs_put_price(S0, long_put_k, T, sigma)) + \
                         (bs_call_price(S0, short_call_k, T, sigma) - bs_call_price(S0, long_call_k, T, sigma))
                max_risk = S0*width_pct

                if S_fri <= long_put_k:
                    pnl = credit - (short_put_k - S_fri)
                elif S_fri >= long_call_k:
                    pnl = credit - (S_fri - short_call_k)
                elif S_fri < short_put_k:
                    pnl = credit - (short_put_k - S_fri)
                elif S_fri > short_call_k:
                    pnl = credit - (S_fri - short_call_k)
                else:
                    pnl = credit
                pnl = max(pnl, -max_risk)
                trades.append({"pnl": pnl, "pnl_pct_risk": pnl/max_risk*100 if max_risk > 0 else 0, "credit": credit, "max_risk": max_risk})

            if not trades:
                continue
            df = pd.DataFrame(trades)
            win_rate = (df["pnl"] > 0).mean() * 100
            avg_pnl_pct = df["pnl_pct_risk"].mean()
            total_pnl = df["pnl"].sum()
            avg_credit_pct_of_risk = (df["credit"]*100/df["max_risk"]).mean() if len(df) else 0
            results.append({
                "ticker": ticker, "otm_pct": otm_pct, "width_pct": width_pct,
                "n": len(df), "win_rate": round(win_rate,1), "avg_pnl_pct_risk": round(avg_pnl_pct,2),
                "total_pnl_per_contract_notional": round(total_pnl,2),
                "avg_credit_pct_of_width": round(avg_credit_pct_of_risk,1),
            })
    return results

all_results = []
for t in ("GOOG", "AAPL", "COST"):
    print(f"=== {t} ===")
    r = run(t)
    for row in sorted(r, key=lambda x: -x["avg_pnl_pct_risk"])[:6]:
        print(row)
    all_results.extend(r)
    print()

pd.DataFrame(all_results).to_csv("weekly_condor_backtest_results.csv", index=False)
print("Saved weekly_condor_backtest_results.csv")
