"""
Extension of spy_0dte_backtest_real.py's grid to tighter OTM% (0.3-1.0%),
motivated by a real live finding 2026-08-17: at 1.0%+ OTM, live conservative
credit (worst-fill bid/ask) went negative (-$0.01/contract) on SPY -- the
same failure SPX hit 2026-08-13 -- because far-OTM 0DTE premium ($0.01-0.03)
is smaller than the market's flat ~$0.01/leg minimum-tick spread. Live quotes
the same morning showed strikes at 0.3-0.5% OTM trading at $0.06-0.16 with
the SAME flat $0.01 spread, so the spread eats a much smaller fraction of
the credit there. This reuses the exact same methodology and same real
last-trade-price data files as the original backtest -- only the otm_pcts
grid is extended tighter. Same limitations apply (last-trade prices, not
bid/ask -- see spy_0dte_backtest_real.py's docstring).
"""
import json
from spy_0dte_backtest_real import simulate_day_real

def run():
    with open("spy_0dte_bars.json") as f:
        underlying = json.load(f)
    with open("spy_0dte_real_option_bars.json") as f:
        option_data = json.load(f)

    print(f"=== TIGHT-OTM real-price backtest: SPY, {len(underlying)} underlying days ===\n")

    otm_pcts = [0.003, 0.004, 0.005, 0.007, 0.010]
    widths = [0.002, 0.003, 0.004]
    profit_target_pct = 0.50
    stop_configs = [("stop_1.0x", 1.0), ("stop_2.0x", 2.0), ("no_stop", None)]
    entry_time = "09:45"

    grid_results = []
    for otm in otm_pcts:
        for w in widths:
            if w >= otm:
                continue  # long leg would cross or invert the short strike
            for stop_label, stop_mult in stop_configs:
                trades = []
                skipped = 0
                for day, ubars in underlying.items():
                    contracts = option_data.get(day)
                    if not contracts:
                        skipped += 1
                        continue
                    r = simulate_day_real(day, ubars, contracts, "SPY", otm, w,
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
                    "otm_pct": otm, "width_pct": w, "stop": stop_label,
                    "n_trades": len(trades), "skipped_days": skipped,
                    "win_rate_pct": round(win_rate, 1), "avg_pnl": round(avg_pnl, 2),
                    "total_pnl": round(total_pnl, 2), "worst_trade": round(worst, 2),
                    "exit_breakdown": exit_breakdown,
                }
                grid_results.append(row)
                print(f"OTM={otm*100:.1f}% width={w*100:.1f}% {stop_label:9s} "
                      f"n={len(trades):2d} (skip {skipped:2d})  win_rate={win_rate:5.1f}%  "
                      f"avg_pnl=${avg_pnl:+7.2f}  total_pnl=${total_pnl:+9.2f}  worst=${worst:8.2f}  {exit_breakdown}")

    with open("spy_0dte_tight_otm_backtest_results.json", "w") as f:
        json.dump(grid_results, f, indent=2)
    print(f"\nSaved {len(grid_results)} configurations.")

if __name__ == "__main__":
    run()
