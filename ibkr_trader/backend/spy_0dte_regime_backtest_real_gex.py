"""
Real-GEX regime-selectivity test for SPY 0DTE -- replaces the earlier VIX
PROXY (spy_0dte_regime_backtest.py, 2026-08-12) with genuine historical
dealer gamma exposure, now that CBOE DataShop has confirmed real
historical OI (2026-08-13). This is the actual test the account has been
working toward since the "why do GEX-aware 0DTE traders make money"
conversation -- a real answer, not an inferred one.

Uses PRIOR TRADING DAY's real GEX (spy_real_gex_history.json, computed
from CBOE's real EOD open interest + gamma) to filter which of the 40
real backtest days to trade -- mirrors exactly how real GEX-aware traders
actually use it (yesterday's close GEX informs today's regime), and
exactly the same "yesterday informs today" structure the VIX-proxy
version used.

Runs on REAL PRICES (spy_0dte_backtest_real.py's simulate_day_real), not
the Black-Scholes model -- the most rigorous combination built this
session: real option prices x real dealer positioning.

Usage: python spy_0dte_regime_backtest_real_gex.py
"""
import json
from datetime import datetime, timedelta

from spy_0dte_backtest_real import simulate_day_real


def prior_trading_day(date_str, gex_history):
    """Real prior trading day present in gex_history, walking back from date_str."""
    d = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    for _ in range(5):   # handle weekends/holidays
        ds = d.strftime("%Y-%m-%d")
        if ds in gex_history:
            return ds
        d -= timedelta(days=1)
    return None


def main():
    with open("spy_0dte_bars.json") as f:
        underlying = json.load(f)
    with open("spy_0dte_real_option_bars.json") as f:
        option_data = json.load(f)
    with open("spy_real_gex_history.json") as f:
        gex_history = json.load(f)

    print("=== SPY 0DTE real-price backtest, filtered by REAL prior-day GEX regime ===\n")

    # validated best real-price configs from 2026-08-13 (see oversight_log.jsonl)
    configs = [
        ("09:45", 0.010, 0.004, None),   # best no-stop config
        ("09:45", 0.010, 0.008, None),
        ("09:45", 0.015, 0.004, None),
        ("13:30", 0.010, 0.004, None),
    ]
    regime_filters = ["all", "positive_gamma", "negative_gamma"]

    day_regime = {}
    for day in underlying:
        prior = prior_trading_day(day, gex_history)
        if prior:
            day_regime[day] = gex_history[prior]["regime"]
            day_regime[day + "_gex"] = gex_history[prior]["net_gex"]

    print("Prior-day real GEX regime assigned to each backtest day:")
    for day in sorted(underlying):
        r = day_regime.get(day, "NO_DATA")
        g = day_regime.get(day + "_gex")
        print(f"  {day}: regime={r}  prior_net_gex={g:+,.0f}" if g is not None else f"  {day}: regime={r}")
    print()

    results = []
    for entry_time, otm, width, stop_mult in configs:
        for filt in regime_filters:
            trades = []
            skipped_no_data = 0
            skipped_regime = 0
            for day, ubars in underlying.items():
                regime = day_regime.get(day)
                if regime is None:
                    skipped_no_data += 1
                    continue
                if filt != "all" and regime != filt:
                    skipped_regime += 1
                    continue
                contracts = option_data.get(day)
                if not contracts:
                    continue
                r = simulate_day_real(day, ubars, contracts, "SPY", otm, width, 0.50, stop_mult, entry_time)
                if r:
                    trades.append(r)
            if not trades:
                continue
            wins = [t for t in trades if t["pnl"] > 0]
            win_rate = len(wins) / len(trades) * 100
            avg_pnl = sum(t["pnl"] for t in trades) / len(trades)
            total_pnl = sum(t["pnl"] for t in trades)
            worst = min((t["pnl"] for t in trades), default=0)
            row = {
                "entry_time": entry_time, "otm_pct": otm, "width_pct": width,
                "stop": "no_stop" if stop_mult is None else f"stop_{stop_mult}x",
                "gex_filter": filt, "n_trades": len(trades),
                "skipped_no_gex_data": skipped_no_data, "skipped_regime_mismatch": skipped_regime,
                "win_rate_pct": round(win_rate, 1), "avg_pnl": round(avg_pnl, 2),
                "total_pnl": round(total_pnl, 2), "worst_trade": round(worst, 2),
            }
            results.append(row)
            print(f"entry={entry_time} OTM={otm*100:.1f}% width={width*100:.1f}% filter={filt:16s} "
                  f"n={len(trades):2d} (no_data={skipped_no_data} regime_skip={skipped_regime})  "
                  f"win_rate={win_rate:5.1f}%  avg_pnl=${avg_pnl:+7.2f}  total_pnl=${total_pnl:+9.2f}  "
                  f"worst=${worst:8.2f}")
        print()

    with open("spy_0dte_regime_real_gex_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} rows to spy_0dte_regime_real_gex_results.json")


if __name__ == "__main__":
    main()
