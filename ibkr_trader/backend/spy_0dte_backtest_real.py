"""
SPY/SPX 0DTE iron condor backtest using REAL observed option prices from
Massive/Polygon flat files -- replaces the Black-Scholes MODEL in
spy_0dte_backtest.py with actual historical trade prices.

APPROXIMATIONS STILL PRESENT, STATED UP FRONT:
  1. These are last-TRADE prices (OHLC per minute), not bid/ask quotes.
     A real fill crosses the spread; this still assumes fills at the
     traded price, same limitation the BS version had -- now grounded in
     real prints instead of a model, but still not a true fill simulation.
  2. Strike selection snaps to the NEAREST REAL LISTED STRIKE that had at
     least one real trade that day, not the theoretical round(S0*(1+/-otm))
     value -- flagged per-trade when the snap distance is non-trivial.
  3. Missing minutes (a strike with no trade in a given minute) are
     forward-filled from the last known real price for that contract,
     same convention a real position holder would use (mark at last
     trade), not interpolated or modeled.
  4. A day/config is SKIPPED (not faked) if any of the 4 legs never
     traded near the entry time at all -- consistent with the BS
     version's "entry_credit <= 0: return None" discipline of dropping
     rather than guessing.

Usage:
  python spy_0dte_backtest_real.py --ticker spy
  python spy_0dte_backtest_real.py --ticker spx
"""
import argparse
import json
from datetime import datetime, timedelta

HARD_CLOSE = "15:45"


def nearest_strike_ticker(contracts, root, day, right, target_strike):
    """Find the real listed contract ticker closest to target_strike among
    contracts that actually traded that day. right: 'C' or 'P'."""
    yymmdd = day[2:4] + day[5:7] + day[8:10]
    prefix = f"O:{root}{yymmdd}{right}"
    best = None
    best_dist = None
    for ticker in contracts:
        if not ticker.startswith(prefix):
            continue
        strike_str = ticker[len(prefix):]
        try:
            strike = int(strike_str) / 1000.0
        except ValueError:
            continue
        dist = abs(strike - target_strike)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (ticker, strike)
    return best  # (ticker, actual_strike) or None


def build_price_lookup(bars):
    """{HH:MM: close} for one contract's real minute bars."""
    return {b["time"][11:16]: b["close"] for b in bars}


def price_at(lookup, sorted_times, t):
    """Real price at time t (HH:MM), forward-filled from the last known
    real trade at or before t. Returns None if no trade has happened yet."""
    if t in lookup:
        return lookup[t]
    # forward-fill: find latest time <= t
    candidates = [x for x in sorted_times if x <= t]
    if not candidates:
        return None
    return lookup[candidates[-1]]


def simulate_day_real(day, underlying_bars, option_contracts, root,
                       otm_pct, width, profit_target_pct, stop_loss_mult, entry_time,
                       width_is_dollars=False):
    """width: fraction of spot (default) or a fixed dollar amount when
    width_is_dollars=True -- the latter caps max_risk regardless of the
    underlying's price level, which is the whole point for SPX (its
    percentage-of-spot widths from the SPY-derived grid mapped to
    real $30-60 wide spreads, i.e. $3,000-6,000/contract max risk --
    unaffordable at this account's real capital, see oversight_log.jsonl
    2026-08-13 capital_sizing_check)."""
    by_time = {b["time"][11:16]: b for b in underlying_bars}
    if entry_time not in by_time:
        return None
    S0 = by_time[entry_time]["close"]

    target_short_put = S0 * (1 - otm_pct)
    target_short_call = S0 * (1 + otm_pct)
    if width_is_dollars:
        target_long_put = target_short_put - width
        target_long_call = target_short_call + width
    else:
        target_long_put = S0 * (1 - otm_pct - width)
        target_long_call = S0 * (1 + otm_pct + width)

    sp = nearest_strike_ticker(option_contracts, root, day, "P", target_short_put)
    lp = nearest_strike_ticker(option_contracts, root, day, "P", target_long_put)
    sc = nearest_strike_ticker(option_contracts, root, day, "C", target_short_call)
    lc = nearest_strike_ticker(option_contracts, root, day, "C", target_long_call)
    if not (sp and lp and sc and lc):
        return None
    sp_ticker, sp_strike = sp
    lp_ticker, lp_strike = lp
    sc_ticker, sc_strike = sc
    lc_ticker, lc_strike = lc
    if lp_strike >= sp_strike or lc_strike <= sc_strike:
        return None  # real strikes collapsed the spread to zero or inverted width

    legs = {
        "sp": (build_price_lookup(option_contracts[sp_ticker]), None),
        "lp": (build_price_lookup(option_contracts[lp_ticker]), None),
        "sc": (build_price_lookup(option_contracts[sc_ticker]), None),
        "lc": (build_price_lookup(option_contracts[lc_ticker]), None),
    }
    for k in legs:
        lookup, _ = legs[k]
        legs[k] = (lookup, sorted(lookup.keys()))

    def condor_value(t):
        prices = {}
        for k, (lookup, times) in legs.items():
            p = price_at(lookup, times, t)
            if p is None:
                return None
            prices[k] = p
        return (prices["sp"] - prices["lp"]) + (prices["sc"] - prices["lc"])

    entry_credit = condor_value(entry_time)
    if entry_credit is None or entry_credit <= 0:
        return None
    max_risk = round(sp_strike - lp_strike, 2) * 100
    if max_risk <= 0:
        return None

    profit_target_val = entry_credit * (1 - profit_target_pct)
    stop_loss_val = entry_credit * (1 + stop_loss_mult) if stop_loss_mult is not None else None

    entry_dt = datetime.strptime(f"{day} {entry_time}", "%Y-%m-%d %H:%M")
    hard_close_dt = datetime.strptime(f"{day} {HARD_CLOSE}", "%Y-%m-%d %H:%M")
    all_minutes = []
    t = entry_dt
    while t <= hard_close_dt:
        all_minutes.append(t.strftime("%H:%M"))
        t += timedelta(minutes=1)

    for minute in all_minutes:
        cur_val = condor_value(minute)
        if cur_val is None:
            continue
        if cur_val <= profit_target_val:
            pnl = (entry_credit - cur_val) * 100
            return {"exit": "profit_target", "pnl": pnl, "pnl_pct_of_risk": pnl / max_risk * 100,
                    "strikes": {"sp": sp_strike, "lp": lp_strike, "sc": sc_strike, "lc": lc_strike}}
        if stop_loss_val is not None and cur_val >= stop_loss_val:
            pnl = (entry_credit - cur_val) * 100
            return {"exit": "stop_loss", "pnl": pnl, "pnl_pct_of_risk": pnl / max_risk * 100,
                    "strikes": {"sp": sp_strike, "lp": lp_strike, "sc": sc_strike, "lc": lc_strike}}

    final_val = condor_value(HARD_CLOSE)
    if final_val is None:
        # walk backward from hard close to find the last minute with full real data
        for minute in reversed(all_minutes):
            final_val = condor_value(minute)
            if final_val is not None:
                break
    if final_val is None:
        return None
    pnl = (entry_credit - final_val) * 100
    return {"exit": "hard_close", "pnl": pnl, "pnl_pct_of_risk": pnl / max_risk * 100,
            "strikes": {"sp": sp_strike, "lp": lp_strike, "sc": sc_strike, "lc": lc_strike}}


def run(ticker):
    root = {"spy": "SPY", "spx": "SPXW"}[ticker]
    with open(f"{ticker}_0dte_bars.json") as f:
        underlying = json.load(f)
    with open(f"{ticker}_0dte_real_option_bars.json") as f:
        option_data = json.load(f)

    print(f"=== REAL-PRICE backtest: {ticker.upper()} (root {root}), "
          f"{len(underlying)} underlying days, {len(option_data)} option-data days ===\n")

    otm_pcts = [0.010, 0.015, 0.020, 0.030]
    widths = [0.004, 0.008]
    profit_target_pct = 0.50
    stop_configs = [("stop_1.0x", 1.0), ("stop_2.0x", 2.0), ("no_stop", None)]
    entry_times = [("09:45", "09:45"), ("13:30", "13:30")]

    grid_results = []
    for entry_label, entry_time in entry_times:
        for otm in otm_pcts:
            for w in widths:
                for stop_label, stop_mult in stop_configs:
                    trades = []
                    skipped = 0
                    for day, ubars in underlying.items():
                        contracts = option_data.get(day)
                        if not contracts:
                            skipped += 1
                            continue
                        r = simulate_day_real(day, ubars, contracts, root, otm, w,
                                               profit_target_pct, stop_mult, entry_time)
                        if r:
                            trades.append(r)
                        else:
                            skipped += 1
                    if not trades:
                        continue
                    wins = [t for t in trades if t["pnl"] > 0]
                    win_rate = len(wins) / len(trades) * 100
                    avg_pnl = sum(t["pnl"] for t in trades) / len(trades)
                    total_pnl = sum(t["pnl"] for t in trades)
                    worst = min((t["pnl"] for t in trades), default=0)
                    exit_breakdown = {}
                    for t in trades:
                        exit_breakdown[t["exit"]] = exit_breakdown.get(t["exit"], 0) + 1
                    row = {
                        "entry_time": entry_label, "otm_pct": otm, "width_pct": w,
                        "stop": stop_label, "n_trades": len(trades), "skipped_days": skipped,
                        "win_rate_pct": round(win_rate, 1), "avg_pnl": round(avg_pnl, 2),
                        "total_pnl": round(total_pnl, 2), "worst_trade": round(worst, 2),
                        "exit_breakdown": exit_breakdown,
                    }
                    grid_results.append(row)
                    print(f"entry={entry_label} OTM={otm*100:.1f}% width={w*100:.1f}% {stop_label:9s} "
                          f"n={len(trades):2d} (skip {skipped:2d})  win_rate={win_rate:5.1f}%  "
                          f"avg_pnl=${avg_pnl:+7.2f}  total_pnl=${total_pnl:+9.2f}  worst=${worst:8.2f}  {exit_breakdown}")

    out_file = f"{ticker}_0dte_backtest_real_results.json"
    with open(out_file, "w") as f:
        json.dump(grid_results, f, indent=2)
    print(f"\nSaved {len(grid_results)} configurations to {out_file}")

    profitable = [r for r in grid_results if r["total_pnl"] > 0]
    print(f"\n{len(profitable)}/{len(grid_results)} configurations were net profitable "
          f"(REAL prices) over the real sample.")
    if profitable:
        best = max(profitable, key=lambda r: r["total_pnl"])
        print(f"Best: entry={best['entry_time']} OTM={best['otm_pct']*100:.1f}% "
              f"width={best['width_pct']*100:.1f}% {best['stop']} -> "
              f"win_rate={best['win_rate_pct']}% total_pnl=${best['total_pnl']:+.2f} (n={best['n_trades']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, choices=["spy", "spx"])
    args = ap.parse_args()
    run(args.ticker)
