"""
Real entry/exit simulation (same 0.35%+RVOL>=1.2x confirm, 0.2% real live
trailing stop mechanics used throughout this session) for the 7 individual
S/R/SMA/EMA factor variants from build_candidates_sma_sr.py, each tested
against baseline_repro2 in isolation (one variable at a time, matching
this session's established discipline).
"""
from variant_comparison_backtest import load_all_bars, load_avg_daily_vol, run_variant
from pathlib import Path

HERE = Path(__file__).parent

FACTORS = ["range_pos_20d", "pivot_dist_pct", "sma20_dist_pct",
           "sma50_dist_pct", "sma200_dist_pct", "ema9_dist_pct", "ema21_dist_pct"]


def main():
    bar_data = load_all_bars()
    print(f"Total real minute-bar keys available: {len(bar_data)}")

    import csv
    all_tickers = set()
    for name in ["candidates_baseline_repro2.csv"] + [f"candidates_{f}.csv" for f in FACTORS]:
        rows = list(csv.DictReader(open(HERE / name)))
        all_tickers.update(r["ticker"] for r in rows)
    avgvol = load_avg_daily_vol(sorted(all_tickers))

    print("\n" + "=" * 70)
    agg_base = run_variant(HERE / "candidates_baseline_repro2.csv", bar_data, avgvol, "BASELINE (repro2)")

    results = {"baseline": agg_base}
    for factor in FACTORS:
        agg = run_variant(HERE / f"candidates_{factor}.csv", bar_data, avgvol, f"+ {factor}")
        results[factor] = agg

    import json
    (HERE / "sma_sr_comparison_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote sma_sr_comparison_results.json")

    print("\n" + "=" * 70)
    print("SUMMARY (sorted by win rate):")
    rows = [(k, v.get("win_rate_pct", 0), v.get("avg_ret_pct", 0), v.get("n", 0)) for k, v in results.items()]
    rows.sort(key=lambda r: -r[1])
    for name, win, avg, n in rows:
        print(f"  {name:20s}  n={n:4d}  win={win:5.1f}%  avg={avg:+.4f}%")


if __name__ == "__main__":
    main()
