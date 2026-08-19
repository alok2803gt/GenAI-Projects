"""
SPY 0DTE iron condor backtest -- real 1-min IBKR bars, real trading days,
honest results (no massaging toward a target win rate).

APPROXIMATIONS, STATED UP FRONT (per this account's own backtest discipline --
see trade-history-playbook, gex-vex-calculator):
  1. No historical 0DTE options-chain data exists anywhere accessible to this
     account (same wall hit by the SOXL backtest, task 2026-08-08-003) --
     option premiums are MODELED via Black-Scholes, not observed. Real fills
     would differ from modeled Black-Scholes prices, especially near the
     open/close when 0DTE gamma is extreme.
  2. Implied vol input uses a realized-vol proxy (first 30 min of each day's
     actual price action, annualized) rather than a true observed IV term
     structure. This tends to UNDERSTATE real 0DTE IV (which typically runs
     above realized vol, especially early in the session), meaning modeled
     premiums here are likely conservative (smaller credits than real ones)
     -- if anything this should bias the backtest pessimistic, not optimistic.
  3. Real historical depth available: ~2 months of real 1-min IBKR bars
     (tested and confirmed; yfinance caps at 8 days, IBKR goes deeper).
     This is a real trading-day sample, not synthetic, but it is NOT a
     multi-year, multi-regime sample -- it reflects whatever volatility
     regime the last ~2 months actually were, nothing more.
  4. Assumes fills at exactly the modeled mid-price with no slippage/spread
     cost and no commission. Real fills will be worse, per every other
     execution-quality lesson learned this session (COHR needed real
     repricing to fill even with decent liquidity).

Structure: iron condor, short strikes at otm_pct from the 09:45 ET entry
price, protective long strikes at otm_pct + width. Hard close at 15:45 ET
(mandatory -- SPY is physically settled, unlike cash-settled SPX, so this
strategy must never hold into the close/settlement, see the SPY-vs-SPX risk
discussion this session). Profit target / stop loss checked every minute
using re-priced Black-Scholes values, so target-vs-stop-first sequencing is
resolved from the real intraday path, not assumed.

Usage:
  python spy_0dte_backtest.py --fetch   # pull real bars for the last N trading days (rate-limited, slow)
  python spy_0dte_backtest.py --run     # simulate across the parameter grid using cached bars
"""
import argparse
import json
import math
from datetime import date, datetime, timedelta

import requests

BACKEND = "http://localhost:8000"
N_DAYS = 40                 # real trading days to pull (~2 months)
ENTRY_TIME = "09:45"
HARD_CLOSE = "15:45"
RISK_FREE = 0.045


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, sigma, right):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max((S - K) if right == "C" else (K - S), 0.0)
    d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if right == "C":
        return S * _norm_cdf(d1) - K * math.exp(-RISK_FREE * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-RISK_FREE * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def trading_days_back(n):
    """Last n real weekdays ending yesterday (today's session may be incomplete)."""
    days = []
    d = date.today() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d.isoformat())
        d -= timedelta(days=1)
    return list(reversed(days))


def fetch_bars():
    days = trading_days_back(N_DAYS)
    print(f"Fetching {len(days)} real trading days of SPY 1-min bars via IBKR "
          f"(paced ~5/15s per backend limits, this will take a few minutes)...")
    all_bars = {}
    # Backend paces internally per its own docstring, but batch conservatively
    # from this side too -- one date per request, small batches.
    for i, d in enumerate(days):
        try:
            r = requests.post(f"{BACKEND}/market/history/minute",
                               json=[{"ticker": "SPY", "date": d}], timeout=30)
            r.raise_for_status()
            data = r.json()["data"]
            key = f"SPY:{d}"
            if key in data and data[key]:
                all_bars[d] = data[key]
                print(f"  [{i+1}/{len(days)}] {d}: {len(data[key])} bars")
            else:
                print(f"  [{i+1}/{len(days)}] {d}: no data (holiday or unavailable)")
        except Exception as e:
            print(f"  [{i+1}/{len(days)}] {d}: ERROR {e}")
    with open("spy_0dte_bars.json", "w") as f:
        json.dump(all_bars, f)
    print(f"\nSaved {len(all_bars)} days of real bars to spy_0dte_bars.json")


def simulate_day(bars, otm_pct, width, profit_target_pct, stop_loss_mult, entry_time=ENTRY_TIME):
    """Returns dict with pnl_pct (of max risk) for one day under one parameter set.
    stop_loss_mult=None means no stop loss -- hold to profit target or hard close only."""
    by_time = {b["time"][11:16]: b for b in bars}
    if entry_time not in by_time:
        return None
    entry_bar = by_time[entry_time]
    S0 = entry_bar["close"]

    # realized vol proxy: stdev of 1-min log returns over the 30 min BEFORE entry, annualized
    idx0 = next(i for i, b in enumerate(bars) if b["time"][11:16] == entry_time)
    window = bars[max(0, idx0 - 30):idx0]
    if len(window) < 10:
        return None
    rets = [math.log(window[i]["close"] / window[i - 1]["close"])
            for i in range(1, len(window)) if window[i - 1]["close"] > 0]
    if not rets:
        return None
    minute_std = (sum(r ** 2 for r in rets) / len(rets)) ** 0.5
    sigma = minute_std * math.sqrt(390 * 252)   # annualize from 1-min stdev
    sigma = max(sigma, 0.05)

    short_put_k  = round(S0 * (1 - otm_pct))
    long_put_k   = round(S0 * (1 - otm_pct - width))
    short_call_k = round(S0 * (1 + otm_pct))
    long_call_k  = round(S0 * (1 + otm_pct + width))

    entry_dt = datetime.strptime(f"{bars[0]['time'][:10]} {entry_time}", "%Y-%m-%d %H:%M")
    close_dt = datetime.strptime(f"{bars[0]['time'][:10]} 16:00", "%Y-%m-%d %H:%M")
    hard_close_dt = datetime.strptime(f"{bars[0]['time'][:10]} {HARD_CLOSE}", "%Y-%m-%d %H:%M")
    total_seconds = (close_dt - entry_dt).total_seconds()

    def condor_value(S, t_frac):
        T = max(t_frac, 1e-6) / (252 * 6.5)   # remaining trading hours today / hours in a trading year
        sp = bs_price(S, short_put_k, T, sigma, "P")
        lp = bs_price(S, long_put_k, T, sigma, "P")
        sc = bs_price(S, short_call_k, T, sigma, "C")
        lc = bs_price(S, long_call_k, T, sigma, "C")
        return (sp - lp) + (sc - lc)   # value of the SHORT condor (what we owe to close)

    entry_credit = condor_value(S0, 1.0)
    if entry_credit <= 0:
        return None
    max_risk = round(short_put_k - long_put_k, 2) * 100   # $ per contract (put side width == call side width)
    if max_risk <= 0:
        return None

    profit_target_val = entry_credit * (1 - profit_target_pct)
    stop_loss_val = entry_credit * (1 + stop_loss_mult) if stop_loss_mult is not None else None

    for b in bars[idx0:]:
        t = datetime.strptime(f"{b['time'][:10]} {b['time'][11:16]}", "%Y-%m-%d %H:%M")
        if t > hard_close_dt:
            break
        t_frac = max((close_dt - t).total_seconds() / total_seconds, 0.0001)
        cur_val = condor_value(b["close"], t_frac)
        if cur_val <= profit_target_val:
            pnl = (entry_credit - cur_val) * 100
            return {"exit": "profit_target", "pnl": pnl, "pnl_pct_of_risk": pnl / max_risk * 100}
        if stop_loss_val is not None and cur_val >= stop_loss_val:
            pnl = (entry_credit - cur_val) * 100
            return {"exit": "stop_loss", "pnl": pnl, "pnl_pct_of_risk": pnl / max_risk * 100}

    # hard close -- mark at whatever the condor is worth at 15:45
    final_bar = min(bars, key=lambda b: abs(
        datetime.strptime(f"{b['time'][:10]} {b['time'][11:16]}", "%Y-%m-%d %H:%M") - hard_close_dt))
    t_frac = max((close_dt - hard_close_dt).total_seconds() / total_seconds, 0.0001)
    final_val = condor_value(final_bar["close"], t_frac)
    pnl = (entry_credit - final_val) * 100
    return {"exit": "hard_close", "pnl": pnl, "pnl_pct_of_risk": pnl / max_risk * 100}


def run(bars_file, results_file):
    with open(bars_file) as f:
        all_bars = json.load(f)
    print(f"Loaded {len(all_bars)} real trading days.\n")
    print("=== ROUND 2: wider strikes (real cushion), looser/no stop, two entry times ===")
    print("Testing a genuinely different hypothesis, not tuning round 1's failed tight range.\n")

    # Real cushion this time -- SPY's typical daily range is ~1%, so 0.3-0.8% OTM
    # (round 1) put strikes inside routine daily noise. 1-3% is actual distance.
    otm_pcts = [0.010, 0.015, 0.020, 0.030]
    widths   = [0.004, 0.008]
    profit_target_pct = 0.50
    stop_configs = [("stop_1.0x", 1.0), ("stop_2.0x", 2.0), ("no_stop", None)]
    entry_times = [("09:45", "09:45"), ("13:30", "13:30")]

    grid_results = []
    for entry_label, entry_time in entry_times:
        for otm in otm_pcts:
            for w in widths:
                for stop_label, stop_mult in stop_configs:
                    trades = []
                    for day, bars in all_bars.items():
                        r = simulate_day(bars, otm, w, profit_target_pct, stop_mult, entry_time)
                        if r:
                            trades.append(r)
                    if not trades:
                        continue
                    wins = [t for t in trades if t["pnl"] > 0]
                    win_rate = len(wins) / len(trades) * 100
                    avg_pnl = sum(t["pnl"] for t in trades) / len(trades)
                    total_pnl = sum(t["pnl"] for t in trades)
                    max_loss = min((t["pnl"] for t in trades), default=0)
                    exit_breakdown = {}
                    for t in trades:
                        exit_breakdown[t["exit"]] = exit_breakdown.get(t["exit"], 0) + 1
                    row = {
                        "entry_time": entry_label, "otm_pct": otm, "width_pct": w,
                        "stop": stop_label, "n_trades": len(trades),
                        "win_rate_pct": round(win_rate, 1), "avg_pnl": round(avg_pnl, 2),
                        "total_pnl": round(total_pnl, 2), "worst_trade": round(max_loss, 2),
                        "exit_breakdown": exit_breakdown,
                    }
                    grid_results.append(row)
                    print(f"entry={entry_label} OTM={otm*100:.1f}% width={w*100:.1f}% {stop_label:9s} "
                          f"n={len(trades)}  win_rate={win_rate:5.1f}%  avg_pnl=${avg_pnl:+7.2f}  "
                          f"total_pnl=${total_pnl:+9.2f}  worst=${max_loss:8.2f}  {exit_breakdown}")

    with open(results_file, "w") as f:
        json.dump(grid_results, f, indent=2)
    print(f"\nSaved {len(grid_results)} configurations to {results_file}")

    profitable = [r for r in grid_results if r["total_pnl"] > 0]
    print(f"\n{len(profitable)}/{len(grid_results)} configurations were net profitable over the real 40-day sample.")
    if profitable:
        best = max(profitable, key=lambda r: r["total_pnl"])
        print(f"Best: entry={best['entry_time']} OTM={best['otm_pct']*100:.1f}% "
              f"width={best['width_pct']*100:.1f}% {best['stop']} -> "
              f"win_rate={best['win_rate_pct']}% total_pnl=${best['total_pnl']:+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--bars", default="spy_0dte_bars.json")
    ap.add_argument("--results", default="spy_0dte_backtest_results.json")
    args = ap.parse_args()
    if args.fetch:
        fetch_bars()
    elif args.run:
        run(args.bars, args.results)
    else:
        print("Pass --fetch or --run")
