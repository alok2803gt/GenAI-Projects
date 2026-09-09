"""
Recalibrates the 9:35-10:00 entry-window backtest (butterfly_dynamic_trigger_rows.csv,
116 real triggered rows) under the REAL platform constraint found live
2026-09-02: Alpaca auto-closes 0DTE option positions at 15:45 ET, not the
theoretical real-4pm-close the original backtest assumed. Fetches a real
15:45 ET mark (forward-filled from the most recent real trade at or
before that time, same sparse-data caveat already disclosed for every
intraday mark in this series) for each of the 116 rows, so this entry
design can be compared apples-to-apples against the fixed-9:45 design's
own realistic-exit numbers (near-S/R + wing_step=4: mean +$7.98, median
+$5.00, win 50.8%).
"""
import sys
import io
from datetime import date, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd

import butterfly_0dte_backtest_v2 as v2
from butterfly_0dte_trailing_sl_backtest import get_full_intraday_tape, et_minutes_to_utc, forward_fill_price

EXIT_MINUTE_1545 = 15 * 60 + 45


def main():
    df = pd.read_csv("butterfly_dynamic_trigger_rows.csv")
    print(f"Base dataset: {len(df)} real triggered rows")

    rows_out = []
    for i, row in enumerate(df.itertuples(), 1):
        if i % 20 == 0:
            print(f"  {i}/{len(df)}")
        ticker = row.ticker
        d = date.fromisoformat(row.entry_date)
        trigger_hh, trigger_mm = map(int, row.trigger_time_et.split(":"))
        trigger_utc = et_minutes_to_utc(d, trigger_hh * 60 + trigger_mm)

        call_map = v2.get_contract_grid_0dte(ticker, d)
        strikes_sorted = sorted(call_map.keys())
        k2 = min(strikes_sorted, key=lambda s: abs(s - row.trigger_spot))
        k2_pos = strikes_sorted.index(k2)
        wing_step = 4  # only width still in live use
        lo_pos, hi_pos = k2_pos - wing_step, k2_pos + wing_step
        if lo_pos < 0 or hi_pos >= len(strikes_sorted):
            continue
        k1, k3 = strikes_sorted[lo_pos], strikes_sorted[hi_pos]
        occ_map = {"k1": call_map[k1], "k2": call_map[k2], "k3": call_map[k3]}
        tapes = {name: get_full_intraday_tape(occ, d) for name, occ in occ_map.items()}

        def price_near(name, target_utc):
            best, best_diff = None, None
            for ts, p in tapes[name]:
                diff = abs((ts - target_utc).total_seconds())
                if diff <= v2.ENTRY_SEARCH_WINDOW_MIN * 60 and (best_diff is None or diff < best_diff):
                    best, best_diff = p, diff
            return best

        p1 = price_near("k1", trigger_utc)
        p2 = price_near("k2", trigger_utc)
        p3 = price_near("k3", trigger_utc)
        if p1 is None or p2 is None or p3 is None:
            continue
        entry_debit = p1 + p3 - 2 * p2
        if entry_debit <= 0.005:
            continue

        exit_1545_utc = et_minutes_to_utc(d, EXIT_MINUTE_1545)
        e1 = forward_fill_price(tapes["k1"], exit_1545_utc, p1)
        e2 = forward_fill_price(tapes["k2"], exit_1545_utc, p2)
        e3 = forward_fill_price(tapes["k3"], exit_1545_utc, p3)
        exit_value_1545 = e1 + e3 - 2 * e2
        payoff_1545 = (exit_value_1545 - entry_debit) * 100

        rows_out.append(dict(ticker=ticker, entry_date=row.entry_date, trigger_time_et=row.trigger_time_et,
                              entry_debit=entry_debit, payoff_1545_dollar=payoff_1545, win=payoff_1545 > 0))

    out = pd.DataFrame(rows_out)
    out.to_csv("butterfly_window_realistic_exit_rows.csv", index=False)
    print(f"\nSaved {len(out)} rows")
    if len(out) == 0:
        return
    print(f"\n=== 9:35-10:00 window entry + REAL 15:45 exit, wing_step=4 ===")
    print(f"  n={len(out)} mean=${out.payoff_1545_dollar.mean():+.2f} "
          f"median=${out.payoff_1545_dollar.median():+.2f} win_rate={out.win.mean():.1%}")
    print("\n=== By ticker ===")
    for ticker in v2.ETF_UNIVERSE:
        g = out[out.ticker == ticker]
        if len(g) == 0:
            continue
        print(f"  {ticker}: n={len(g)} mean=${g.payoff_1545_dollar.mean():+.2f} win_rate={g.win.mean():.1%}")


if __name__ == "__main__":
    main()
