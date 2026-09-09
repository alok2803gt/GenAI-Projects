"""
Consolidated view across all 270 real 0DTE butterfly backtest rows:
entry/exit detail, the S/R (near prior-day high/low) flag, and P&L under
four exit rules -- real close (4pm), hard close at 15:45, hard close at
15:30, and the 25%-trailing-stop -- all computed from the SAME real
intraday data already validated in this session (real 9:45 ET entry via
Unusual Whales' intraday tape, real Polygon underlying bars, real
close-of-day intrinsic value). The 15:45/15:30 marks are forward-filled
from the most recent real trade at or before that time, same approximation
already disclosed for the trailing-stop analysis -- real option trades are
sparse (~10-15/day), so an intraday mark between prints is a stale price,
not a guaranteed executable quote.

Output: butterfly_full_comparison.csv (feeds the HTML artifact table).
"""
import sys
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd

import butterfly_0dte_backtest_v2 as v2
from butterfly_0dte_trailing_sl_backtest import get_full_intraday_tape, et_minutes_to_utc, forward_fill_price

TRAIL_PCT = 0.25
MINUTE_1530 = 15 * 60 + 30
MINUTE_1545 = 15 * 60 + 45
ENTRY_MINUTE = 9 * 60 + 45
PATH_CHECKPOINTS = list(range(10 * 60, 15 * 60 + 46, 15))  # 10:00 to 15:45 ET, every 15min


def process_row(ticker, wing_step, entry_date, k1, k2, k3, entry_debit, close_payoff):
    d = date.fromisoformat(entry_date)
    call_map = v2.get_contract_grid_0dte(ticker, d)
    occ_map = {"k1": call_map.get(k1), "k2": call_map.get(k2), "k3": call_map.get(k3)}
    if any(v is None for v in occ_map.values()):
        return None

    tapes = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {name: ex.submit(get_full_intraday_tape, occ, d) for name, occ in occ_map.items()}
        for name, fut in futs.items():
            tapes[name] = fut.result()

    entry_target = et_minutes_to_utc(d, ENTRY_MINUTE)

    def leg_entry_price(name):
        best, best_diff = None, None
        for ts, p in tapes[name]:
            diff = abs((ts - entry_target).total_seconds())
            if diff <= v2.ENTRY_SEARCH_WINDOW_MIN * 60 and (best_diff is None or diff < best_diff):
                best, best_diff = p, diff
        return best

    leg_entry = {name: leg_entry_price(name) for name in ("k1", "k2", "k3")}
    if any(v is None for v in leg_entry.values()):
        return None

    path = []
    for minute in PATH_CHECKPOINTS:
        at_time = et_minutes_to_utc(d, minute)
        p1 = forward_fill_price(tapes["k1"], at_time, leg_entry["k1"])
        p2 = forward_fill_price(tapes["k2"], at_time, leg_entry["k2"])
        p3 = forward_fill_price(tapes["k3"], at_time, leg_entry["k3"])
        path.append((minute, p1 + p3 - 2 * p2))

    value_1530 = next((v for m, v in path if m == MINUTE_1530), None)
    value_1545 = next((v for m, v in path if m == MINUTE_1545), None)

    triggered_value = None
    for minute, value in path:
        running_peak = max([v for m, v in path if m <= minute] + [entry_debit])
        if value <= running_peak * (1 - TRAIL_PCT):
            triggered_value = value
            break

    return {
        "value_1530": value_1530,
        "value_1545": value_1545,
        "trail25_exit_value": triggered_value,
        "trail25_triggered": triggered_value is not None,
    }


def main():
    base = pd.read_csv("butterfly_0dte_v2_rows.csv")
    base = base[base.status == "ok"].copy()
    sr = pd.read_csv("butterfly_gex_sr_enriched_v2.csv")[
        ["ticker", "wing_step", "entry_date", "near_prior_day_range_edge", "gamma_oi"]
    ]
    base = base.merge(sr, on=["ticker", "wing_step", "entry_date"], how="left")

    print(f"Processing {len(base)} rows for real 15:30/15:45/trailing-stop marks...")
    results = []
    for i, row in enumerate(base.itertuples(), 1):
        if i % 20 == 0:
            print(f"  {i}/{len(base)}")
        r = process_row(row.ticker, row.wing_step, row.entry_date, row.k1, row.k2, row.k3,
                         row.entry_debit, row.long_payoff)
        results.append(r if r else {})

    extra = pd.DataFrame(results)
    out = pd.concat([base.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)

    out["payoff_close_4pm"] = out.long_payoff * 100
    out["payoff_hardclose_1545"] = (out.value_1545 - out.entry_debit) * 100
    out["payoff_hardclose_1530"] = (out.value_1530 - out.entry_debit) * 100
    out["payoff_trail25"] = out.apply(
        lambda r: (r.trail25_exit_value - r.entry_debit) * 100 if r.trail25_triggered else r.payoff_close_4pm,
        axis=1,
    )
    out["near_sr"] = out.near_prior_day_range_edge.fillna(False)

    cols = ["ticker", "wing_step", "entry_date", "morning_spot", "close_spot", "k1", "k2", "k3",
            "entry_debit", "near_sr", "payoff_close_4pm", "payoff_hardclose_1545",
            "payoff_hardclose_1530", "payoff_trail25", "trail25_triggered"]
    out = out[cols].sort_values(["entry_date", "ticker", "wing_step"])
    out.to_csv("butterfly_full_comparison.csv", index=False)
    print(f"\nSaved {len(out)} rows to butterfly_full_comparison.csv")

    print("\n=== SUMMARY: all 270, no S/R filter ===")
    for col in ["payoff_close_4pm", "payoff_hardclose_1545", "payoff_hardclose_1530", "payoff_trail25"]:
        s = out[col].dropna()
        print(f"  {col}: n={len(s)} mean=${s.mean():+.2f} median=${s.median():+.2f} win_rate={(s>0).mean():.1%}")

    print("\n=== SUMMARY: near S/R only ===")
    near = out[out.near_sr]
    for col in ["payoff_close_4pm", "payoff_hardclose_1545", "payoff_hardclose_1530", "payoff_trail25"]:
        s = near[col].dropna()
        print(f"  {col}: n={len(s)} mean=${s.mean():+.2f} median=${s.median():+.2f} win_rate={(s>0).mean():.1%}")

    print("\n=== SUMMARY: NOT near S/R ===")
    far = out[~out.near_sr]
    for col in ["payoff_close_4pm", "payoff_hardclose_1545", "payoff_hardclose_1530", "payoff_trail25"]:
        s = far[col].dropna()
        print(f"  {col}: n={len(s)} mean=${s.mean():+.2f} median=${s.median():+.2f} win_rate={(s>0).mean():.1%}")


if __name__ == "__main__":
    main()
