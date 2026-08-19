"""
SPX 0DTE narrow-width grid -- follows up the 2026-08-13 real-price backtest
finding that SPX condors at SPY-derived percentage widths (0.4-0.8% of
spot) carry $3,000-6,000/contract max risk, unaffordable against this
account's real ~$3,400 capital.

Real SPXW strikes near the money trade in clean $5 increments (confirmed
2026-08-13 from the actual flat-file data). This tests FIXED DOLLAR
widths ($5/$10/$15/$20 = 1x-4x the real strike increment) instead of a
percentage of spot, to see whether a structure that fits this account's
capital still holds a real edge, or whether the earlier positive SPX
result was specific to the wider (and unaffordable) spreads.

Usage: python spx_narrow_width_backtest.py
"""
import json

from spy_0dte_backtest_real import simulate_day_real

ROOT = "SPXW"


def run():
    with open("spx_0dte_bars.json") as f:
        underlying = json.load(f)
    with open("spx_0dte_real_option_bars.json") as f:
        option_data = json.load(f)

    print("=== SPX narrow-width grid (real prices, fixed-$ widths) === \n")

    otm_pcts = [0.010, 0.015, 0.020]
    widths_dollars = [5, 10, 15, 20]
    profit_target_pct = 0.50
    stop_configs = [("stop_1.0x", 1.0), ("stop_2.0x", 2.0), ("no_stop", None)]
    entry_times = [("09:45", "09:45"), ("13:30", "13:30")]

    grid_results = []
    for entry_label, entry_time in entry_times:
        for otm in otm_pcts:
            for w in widths_dollars:
                for stop_label, stop_mult in stop_configs:
                    trades = []
                    skipped = 0
                    for day, ubars in underlying.items():
                        contracts = option_data.get(day)
                        if not contracts:
                            skipped += 1
                            continue
                        r = simulate_day_real(day, ubars, contracts, ROOT, otm, w,
                                               profit_target_pct, stop_mult, entry_time,
                                               width_is_dollars=True)
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
                    max_risk = w * 100  # $ per contract, fixed by construction
                    exit_breakdown = {}
                    for t in trades:
                        exit_breakdown[t["exit"]] = exit_breakdown.get(t["exit"], 0) + 1
                    row = {
                        "entry_time": entry_label, "otm_pct": otm, "width_usd": w,
                        "max_risk_usd": max_risk, "stop": stop_label,
                        "n_trades": len(trades), "skipped_days": skipped,
                        "win_rate_pct": round(win_rate, 1), "avg_pnl": round(avg_pnl, 2),
                        "total_pnl": round(total_pnl, 2), "worst_trade": round(worst, 2),
                        "return_on_risk_pct": round(avg_pnl / max_risk * 100, 2),
                        "exit_breakdown": exit_breakdown,
                    }
                    grid_results.append(row)
                    print(f"entry={entry_label} OTM={otm*100:.1f}% width=${w} (max_risk=${max_risk:,.0f}) {stop_label:9s} "
                          f"n={len(trades):2d} (skip {skipped:2d})  win_rate={win_rate:5.1f}%  "
                          f"avg_pnl=${avg_pnl:+7.2f}  total_pnl=${total_pnl:+9.2f}  worst=${worst:8.2f}  "
                          f"RoR={row['return_on_risk_pct']:+.2f}%  {exit_breakdown}")

    out_file = "spx_narrow_width_results.json"
    with open(out_file, "w") as f:
        json.dump(grid_results, f, indent=2)
    print(f"\nSaved {len(grid_results)} configurations to {out_file}")

    profitable = [r for r in grid_results if r["total_pnl"] > 0]
    print(f"\n{len(profitable)}/{len(grid_results)} configurations were net profitable.")
    if profitable:
        best = max(profitable, key=lambda r: r["return_on_risk_pct"])
        print(f"Best by return-on-risk: entry={best['entry_time']} OTM={best['otm_pct']*100:.1f}% "
              f"width=${best['width_usd']} {best['stop']} -> win_rate={best['win_rate_pct']}% "
              f"total_pnl=${best['total_pnl']:+.2f} RoR={best['return_on_risk_pct']:+.2f}% (n={best['n_trades']})")


if __name__ == "__main__":
    run()
