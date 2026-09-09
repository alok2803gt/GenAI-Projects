"""
Trigger-based entry (near prior-day S/R, whenever that first fires between
09:30-11:00 ET -- same real methodology as butterfly_dynamic_trigger_backtest.py,
which already showed this entry design underperforms the fixed-9:45 version
when held to the real close: mean -$7.53 vs +$18.70) PAIRED WITH a 3:55pm ET
exit instead of holding to the close, to test directly whether a later-but-
not-quite-close exit changes that conclusion rather than assuming either way.

Real data throughout: Polygon 1-min underlying bars find the trigger
minute, Unusual Whales' real intraday option tape supplies BOTH the entry
price (at the trigger minute) and the 3:55pm exit price (forward-filled
from the most recent real trade at or before 15:55 -- same sparse-data
caveat already disclosed for every intraday mark in this series).

Output: butterfly_trigger_exit355pm_rows.csv
"""
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

import butterfly_0dte_backtest_v2 as v2
from butterfly_dynamic_trigger_backtest import get_minute_bars, find_trigger, WING_STEP
from butterfly_0dte_trailing_sl_backtest import get_full_intraday_tape, et_minutes_to_utc, forward_fill_price

EXIT_MINUTE = 15 * 60 + 55  # 3:55pm ET


def main():
    earliest = v2.uw_earliest_available_date()
    rows = []

    for ticker in v2.ETF_UNIVERSE:
        print(f"\n=== {ticker} ===", flush=True)
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=False, actions=False)
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
        trading_days = [d for d in hist.index.date if earliest <= d < date.today()]

        for i, d in enumerate(trading_days, 1):
            if i % 15 == 0:
                print(f"  [{i}/{len(trading_days)}] {d}", flush=True)
            prior = hist[hist.index.date < d]
            if prior.empty:
                continue
            prior_high, prior_low = float(prior.iloc[-1]["High"]), float(prior.iloc[-1]["Low"])

            bars = get_minute_bars(ticker, d)
            if not bars:
                continue
            trigger_ts, trigger_px = find_trigger(bars, d, prior_high, prior_low)
            if trigger_ts is None:
                continue

            call_map = v2.get_contract_grid_0dte(ticker, d)
            if len(call_map) < 2 * WING_STEP + 1:
                continue
            strikes_sorted = sorted(call_map.keys())
            k2 = min(strikes_sorted, key=lambda s: abs(s - trigger_px))
            k2_pos = strikes_sorted.index(k2)
            lo_pos, hi_pos = k2_pos - WING_STEP, k2_pos + WING_STEP
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

            p1 = price_near("k1", trigger_ts)
            p2 = price_near("k2", trigger_ts)
            p3 = price_near("k3", trigger_ts)
            if p1 is None or p2 is None or p3 is None:
                continue
            entry_debit = p1 + p3 - 2 * p2
            if entry_debit <= 0.005:
                continue

            exit_target = et_minutes_to_utc(d, EXIT_MINUTE)
            e1 = forward_fill_price(tapes["k1"], exit_target, p1)
            e2 = forward_fill_price(tapes["k2"], exit_target, p2)
            e3 = forward_fill_price(tapes["k3"], exit_target, p3)
            exit_value_355pm = e1 + e3 - 2 * e2
            payoff_355pm = exit_value_355pm - entry_debit

            trigger_et = trigger_ts - timedelta(hours=4)
            rows.append(dict(
                ticker=ticker, entry_date=d.isoformat(), trigger_time_et=trigger_et.strftime("%H:%M"),
                trigger_spot=trigger_px, entry_debit=entry_debit,
                exit_value_355pm=exit_value_355pm, payoff_355pm_dollar=payoff_355pm * 100, win=payoff_355pm > 0,
            ))

    df = pd.DataFrame(rows)
    df.to_csv("butterfly_trigger_exit355pm_rows.csv", index=False)
    print(f"\nSaved {len(df)} rows")
    if len(df) == 0:
        return

    print("\n=== Trigger entry + 3:55pm exit, ALL ===")
    print(f"  n={len(df)} mean=${df.payoff_355pm_dollar.mean():+.2f} "
          f"median=${df.payoff_355pm_dollar.median():+.2f} win_rate={df.win.mean():.1%}")
    print("\n=== By ticker ===")
    for ticker in v2.ETF_UNIVERSE:
        g = df[df.ticker == ticker]
        if len(g) == 0:
            continue
        print(f"  {ticker}: n={len(g)} mean=${g.payoff_355pm_dollar.mean():+.2f} win_rate={g.win.mean():.1%}")

    print("\n=== Compare vs the same trigger design held to the real close (butterfly_dynamic_trigger_rows.csv) ===")
    try:
        close_df = pd.read_csv("butterfly_dynamic_trigger_rows.csv")
        merged = df.merge(close_df[["ticker", "entry_date", "payoff_dollar"]], on=["ticker", "entry_date"], how="inner")
        print(f"  n={len(merged)} (matched rows)")
        print(f"  3:55pm exit : mean=${merged.payoff_355pm_dollar.mean():+.2f}  win_rate={(merged.payoff_355pm_dollar>0).mean():.1%}")
        print(f"  real close  : mean=${merged.payoff_dollar.mean():+.2f}  win_rate={(merged.payoff_dollar>0).mean():.1%}")
    except FileNotFoundError:
        print("  (butterfly_dynamic_trigger_rows.csv not found -- skipping comparison)")


if __name__ == "__main__":
    main()
