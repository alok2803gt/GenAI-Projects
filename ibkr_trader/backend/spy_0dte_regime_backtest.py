"""
Regime-selectivity test: same real 40-day bars, same strategy structure, but
now skip days where VIX (real historical closes) is above a threshold --
testing the actual claimed mechanism behind GEX/regime-aware 0DTE trading
(selectivity, not blind daily entry), using VIX as an honest, available
proxy for true dealer gamma positioning (which isn't retained historically
anywhere accessible -- see oversight_log.jsonl for that limitation).

Caveat stated up front: real VIX for this window ranged 14.55-22.22 -- a
calm-to-moderate period with no genuine stress day. A VIX filter here can
only differentiate "calm" from "moderately calm," not "calm" from "crisis."
Results should be read with that ceiling in mind, not as a full test of
whether regime-filtering helps during real volatility events.

Usage: python spy_0dte_regime_backtest.py --ticker spy|spx
"""
import argparse
import json

from spy_0dte_backtest import simulate_day


def run(ticker):
    bars_file = f"{ticker}_0dte_bars.json"
    with open(bars_file) as f:
        all_bars = json.load(f)
    with open("vix_history.json") as f:
        vix_by_date = json.load(f)

    print(f"=== Regime-selectivity test: {ticker.upper()}, real VIX filter ===\n")

    configs = [
        ("09:45", 0.010, 0.004, 1.0),
        ("09:45", 0.010, 0.004, 2.0),
        ("09:45", 0.010, 0.004, None),
        ("13:30", 0.010, 0.004, 1.0),
        ("13:30", 0.010, 0.004, 2.0),
        ("13:30", 0.010, 0.004, None),
    ]
    vix_thresholds = [16.0, 17.0, 18.0, 999.0]   # 999 = no filter (baseline)

    results = []
    for entry_time, otm, width, stop_mult in configs:
        for vix_max in vix_thresholds:
            trades = []
            skipped_days = 0
            for day, bars in all_bars.items():
                vix_today = vix_by_date.get(day)
                if vix_today is None:
                    continue   # no real VIX data for this day, exclude rather than guess
                if vix_today > vix_max:
                    skipped_days += 1
                    continue
                r = simulate_day(bars, otm, width, 0.50, stop_mult, entry_time)
                if r:
                    trades.append(r)
            if not trades:
                continue
            wins = [t for t in trades if t["pnl"] > 0]
            win_rate = len(wins) / len(trades) * 100
            total_pnl = sum(t["pnl"] for t in trades)
            avg_pnl = total_pnl / len(trades)
            worst = min((t["pnl"] for t in trades), default=0)
            stop_label = f"stop_{stop_mult}x" if stop_mult else "no_stop"
            filt_label = f"VIX<={vix_max:.0f}" if vix_max < 999 else "no_filter"
            row = {
                "ticker": ticker, "entry_time": entry_time, "otm_pct": otm, "width_pct": width,
                "stop": stop_label, "vix_filter": filt_label, "skipped_days": skipped_days,
                "n_trades": len(trades), "win_rate_pct": round(win_rate, 1),
                "avg_pnl": round(avg_pnl, 2), "total_pnl": round(total_pnl, 2), "worst": round(worst, 2),
            }
            results.append(row)
            print(f"entry={entry_time} OTM={otm*100:.1f}% {stop_label:9s} {filt_label:10s} "
                  f"(skipped {skipped_days:2d} days)  n={len(trades):2d}  win_rate={win_rate:5.1f}%  "
                  f"avg_pnl=${avg_pnl:+7.2f}  total_pnl=${total_pnl:+9.2f}  worst=${worst:8.2f}")
        print()

    out_file = f"{ticker}_0dte_regime_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, choices=["spy", "spx"])
    args = ap.parse_args()
    run(args.ticker)
