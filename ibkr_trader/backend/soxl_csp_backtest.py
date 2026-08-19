"""
SOXL CSP / put-spread backtest -- CIO (portfolio-oversight) + CRO (risk-manager)
joint strategy design, per CEO request 2026-08-08.

METHODOLOGY (read before trusting the numbers):
  - Real historical SOXL daily closes (yfinance, full history back to 2010
    inception -- captures 2015-16 selloff, 2018 Q4, 2020 COVID crash, 2022
    bear market, 2023-2025 AI rally: multiple vol regimes, not just a calm
    stretch).
  - No historical SOXL options chain data exists to backtest against (this
    account's other backtests hit the same wall -- see trade-history-playbook's
    note on GEX/VEX). Premiums are therefore estimated via Black-Scholes,
    using each entry date's trailing 30-day realized volatility x a 1.3
    vol-risk-premium multiplier as the IV proxy (options typically trade
    above realized vol; 1.3x is a reasonable, documented, but approximate
    multiplier -- not observed IV). Risk-free rate fixed at 4.5% (matches
    the gex-vex-calculator skill's existing assumption for consistency).
  - Settlement is evaluated at the expiry-date close (European-style
    approximation) -- same simplification the existing safe-income-screener
    cushion backtest uses. Real weekly/monthly American-style puts can be
    assigned early or managed before expiry; this backtest does not model
    early management, so it is a "worst-case, hold-to-expiry" view, not a
    prediction of how an actively-managed position would actually perform.
  - Entries are rolled weekly (every 5 trading days) and monthly (every 21
    trading days) across the whole history -- NOT independent trades, so
    win-rate stats are correlated within a given selloff (a crash fails many
    overlapping entries at once). Treat the stats as regime-level evidence,
    not i.i.d. probability.

Output: results/soxl_csp_backtest_results.json
"""
import json
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

RISK_FREE_RATE = 0.045
VOL_RISK_PREMIUM = 1.3
SPREAD_WIDTH_PCT = 0.10   # defined-risk long put strike = short strike * (1 - 10%)

CUSHIONS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
EXPIRIES = {"weekly": 7, "monthly": 30}   # calendar days to expiry
ENTRY_STEP = {"weekly": 5, "monthly": 21}  # trading days between rolled entries


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes European put price. T in years."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def main():
    print("Pulling SOXL full history...")
    hist = yf.Ticker("SOXL").history(period="max", auto_adjust=True)
    hist = hist[["Close"]].copy()
    hist["logret"] = np.log(hist["Close"] / hist["Close"].shift(1))
    hist["rv30"] = hist["logret"].rolling(30).std() * np.sqrt(252)
    hist = hist.dropna(subset=["rv30"])
    closes = hist["Close"]
    dates = hist.index
    n = len(hist)
    print(f"{n} trading days from {dates[0].date()} to {dates[-1].date()}")

    results = {}
    for exp_name, cal_days in EXPIRIES.items():
        step = ENTRY_STEP[exp_name]
        entry_idxs = list(range(0, n, step))
        for cushion in CUSHIONS:
            naked_trades = []
            spread_trades = []
            for i in entry_idxs:
                entry_date = dates[i]
                S0 = closes.iloc[i]
                sigma = hist["rv30"].iloc[i] * VOL_RISK_PREMIUM
                # find expiry index: first trading day >= entry_date + cal_days
                target = entry_date + pd.Timedelta(days=cal_days)
                future = dates[dates >= target]
                if len(future) == 0:
                    continue
                exp_date = future[0]
                j = dates.get_loc(exp_date)
                if isinstance(j, slice):
                    j = j.start
                S_exp = closes.iloc[j]
                T = cal_days / 365.0

                K_short = S0 * (1 - cushion)
                premium_short = bs_put_price(S0, K_short, T, RISK_FREE_RATE, sigma)

                # Naked CSP: capital at risk = K_short (cash-secured)
                loss_naked = max(K_short - S_exp, 0.0)
                pnl_naked = premium_short - loss_naked
                ret_naked = pnl_naked / K_short
                naked_trades.append({"entry": str(entry_date.date()), "ret": ret_naked,
                                      "assigned": bool(S_exp < K_short)})

                # Defined-risk put credit spread: long put further OTM
                K_long = K_short * (1 - SPREAD_WIDTH_PCT)
                premium_long = bs_put_price(S0, K_long, T, RISK_FREE_RATE, sigma)
                net_credit = premium_short - premium_long
                width = K_short - K_long
                loss_spread = min(max(K_short - S_exp, 0.0), width)
                pnl_spread = net_credit - loss_spread
                capital_at_risk_spread = width  # max loss basis (defined-risk margin)
                ret_spread = pnl_spread / capital_at_risk_spread
                spread_trades.append({"entry": str(entry_date.date()), "ret": ret_spread,
                                       "breached": bool(S_exp < K_short)})

            def summarize(trades, key):
                rets = np.array([t["ret"] for t in trades])
                wins = rets > 0
                breach_key = "assigned" if key == "naked" else "breached"
                breaches = [t for t in trades if t[breach_key]]
                return {
                    "n_trades": len(trades),
                    "win_rate": round(float(wins.mean()) * 100, 1) if len(trades) else None,
                    "avg_return_pct": round(float(rets.mean()) * 100, 2) if len(trades) else None,
                    "worst_return_pct": round(float(rets.min()) * 100, 2) if len(trades) else None,
                    "best_return_pct": round(float(rets.max()) * 100, 2) if len(trades) else None,
                    "n_breached": len(breaches),
                    "breach_rate_pct": round(len(breaches) / len(trades) * 100, 1) if trades else None,
                    "worst_5_trades": sorted(trades, key=lambda t: t["ret"])[:5],
                }

            key = f"{exp_name}_{int(cushion*100)}pct"
            results[key] = {
                "expiry": exp_name, "cushion_pct": cushion * 100,
                "naked_csp": summarize(naked_trades, "naked"),
                "put_spread": summarize(spread_trades, "spread"),
            }
            print(f"{key}: naked win={results[key]['naked_csp']['win_rate']}% "
                  f"avg={results[key]['naked_csp']['avg_return_pct']}%  |  "
                  f"spread win={results[key]['put_spread']['win_rate']}% "
                  f"avg={results[key]['put_spread']['avg_return_pct']}%")

    with open("soxl_csp_backtest_results.json", "w") as f:
        json.dump({
            "methodology": {
                "vol_risk_premium": VOL_RISK_PREMIUM, "risk_free_rate": RISK_FREE_RATE,
                "spread_width_pct": SPREAD_WIDTH_PCT, "history_start": str(dates[0].date()),
                "history_end": str(dates[-1].date()), "n_trading_days": n,
            },
            "results": results,
        }, f, indent=2)
    print("\nSaved -> soxl_csp_backtest_results.json")


if __name__ == "__main__":
    main()
