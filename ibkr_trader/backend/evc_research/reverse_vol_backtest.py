"""
Backtest for the OPPOSITE side of EVC's own trade: when a real, ATM-implied
move ahead of earnings already exceeds EVC's own MAX_MOVE_BOUND (12%) --
exactly the events EVC's own gate skips as "too risky to sell premium" --
does BUYING premium instead (straddle / OTM strangle / a defined-risk
"reverse iron condor") pay off? Motivated by the real 2026-09-03 SNOW
event: EVC correctly skipped it at a 13.2-13.3% implied move, and the real
move (+16.6% close-to-close, +25.7% intraday) would have been a strong
winner for a long-premium structure on the other side of that same signal.

Real data, no Black-Scholes: reuses cushion_symmetry_test.py's real
Polygon-option-price pipeline verbatim (same universe-fetch, same real ATM
straddle price = expected_move, same real 4-strike condor wing prices) --
only the move-filter direction is flipped (> MAX_MOVE_BOUND instead of
< MAX_MOVE_BOUND) and the payoff math is for the 3 long-premium structures
below instead of the short condor.

Three structures tested, all priced off the SAME real entry-day option
closes already fetched for EVC's own condor at this event:
  1. STRADDLE   -- buy the real ATM call + ATM put (cost = expected_move,
                   the real ATM straddle price). Payoff = intrinsic value
                   at the real outcome-date close.
  2. STRANGLE   -- buy EVC's own real short_put/short_call strikes (the
                   condor's inner, moderately-OTM legs) instead of selling
                   them. Cheaper than the straddle, same real strikes/prices
                   EVC already validated as liquid.
  3. REVERSE IRON CONDOR -- the exact mechanical mirror of EVC's real condor
                   at the SAME 4 real strikes: buy what EVC sells (short_put,
                   short_call), sell what EVC buys (long_put, long_call).
                   Cost = -net_credit (a debit). Payoff at any price is
                   exactly -payoff_at_price(...) -- flipping every leg's
                   side flips the whole structure's P&L sign, so this needs
                   no new pricing at all, just the negation of EVC's own
                   already-computed real payoff function.

Real outcome price: yfinance daily close on outcome_date (same convention
historical_move_screen_test.py already established for a close-based test
that doesn't need a live IBKR connection).

Caveat stated explicitly: an intrinsic-value-only payoff assumes the
position is held to a point where extrinsic value has decayed away (matches
this account's own established next-morning EVC exit convention) -- it is
NOT a same-day, still-has-time-value mark. Real fills would also incur
bid-ask slippage on both entry and exit, not modeled here (same
already-disclosed limitation as every other real-data EVC backtest in this
codebase).
"""
import json
import re
import sys

import pandas as pd
import yfinance as yf

from datetime import date as _date, timedelta as _timedelta

from cushion_symmetry_test import (
    round_strike, make_unadjust_fn, fetch_ticker_data,
    get_contract_grid, get_closes_concurrent, payoff_at_price,
    PUT_CUSHION_MULT, CALL_CUSHION_MULT, WING_MULT, MAX_MOVE_BOUND,
    PLAN_AGGS_LOOKBACK_DAYS, PLAN_BOUNDARY_SAFETY_DAYS,
)

PLAN_CUTOFF = _date.today() - _timedelta(days=PLAN_AGGS_LOOKBACK_DAYS + PLAN_BOUNDARY_SAFETY_DAYS)

if hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

MIN_ELEVATED_MOVE = MAX_MOVE_BOUND  # 0.12 -- the exact real threshold EVC itself uses to skip
EARNINGS_LOOKBACK = 20               # deeper than cushion_symmetry_test's 8 -- elevated-move events are rarer


def build_elevated_events(tickers: list[str]) -> tuple[list[dict], dict]:
    events = []
    stats = {"total_events": 0, "outside_plan_window": 0, "no_contracts_listed": 0, "no_atm_contract": 0,
              "no_trade_atm_leg": 0, "not_elevated": 0, "missing_wing_contract": 0,
              "no_trade_wing_leg": 0, "no_real_outcome": 0, "ok": 0}

    for ti, ticker in enumerate(tickers):
        hist, edf, splits = fetch_ticker_data(ticker, retries=2)
        if hist is None:
            continue
        # re-fetch with the deeper lookback this script wants (fetch_ticker_data hardcodes limit=8)
        try:
            edf = yf.Ticker(ticker).get_earnings_dates(limit=EARNINGS_LOOKBACK)
        except Exception:
            pass
        if edf is None or edf.empty:
            continue
        unadjust = make_unadjust_fn(splits)
        trading_days = list(hist.index.date)
        td_index = {d: i for i, d in enumerate(trading_days)}
        closes = hist["Close"].values

        def next_trading_day(d):
            for td in trading_days:
                if td > d:
                    return td
            return None

        def prev_trading_day(d):
            prev = None
            for td in trading_days:
                if td >= d:
                    break
                prev = td
            return prev

        def trading_day_on_or_after(d):
            for td in trading_days:
                if td >= d:
                    return td
            return None

        from datetime import date as _date
        today = _date.today()
        for ts, erow in edf.iterrows():
            edate = pd.Timestamp(ts)
            edate_local = edate.date()
            if edate_local >= today:
                continue
            if pd.isna(erow.get("Reported EPS", float("nan"))):
                continue
            timing = "BMO" if edate.hour < 12 else "AMC"
            if timing == "BMO":
                base_day = edate_local if edate_local in td_index else trading_day_on_or_after(edate_local)
                if base_day is None:
                    continue
                entry_date = prev_trading_day(base_day)
            else:
                entry_date = edate_local if edate_local in td_index else prev_trading_day(
                    trading_day_on_or_after(edate_local) or edate_local)
            if entry_date is None or entry_date not in td_index:
                continue
            outcome_date = next_trading_day(entry_date)
            if outcome_date is None:
                continue
            if entry_date < PLAN_CUTOFF:
                stats["outside_plan_window"] += 1
                continue

            entry_idx, outcome_idx = td_index[entry_date], td_index[outcome_date]
            spot_adj, outcome_adj = float(closes[entry_idx]), float(closes[outcome_idx])
            if spot_adj <= 0 or outcome_adj <= 0:
                continue
            spot = spot_adj * unadjust(entry_date)
            real_outcome_price = outcome_adj * unadjust(entry_date)
            atm_strike = round_strike(spot)
            stats["total_events"] += 1

            expiry_used, call_map, put_map = get_contract_grid(ticker, entry_date)
            if expiry_used is None:
                stats["no_contracts_listed"] += 1
                continue
            atm_call_occ, atm_put_occ = call_map.get(atm_strike), put_map.get(atm_strike)
            if atm_call_occ is None or atm_put_occ is None:
                stats["no_atm_contract"] += 1
                continue
            entry_date_str = entry_date.isoformat()
            atm_closes = get_closes_concurrent({"call": atm_call_occ, "put": atm_put_occ}, entry_date_str)
            atm_call_price, atm_put_price = atm_closes["call"], atm_closes["put"]
            if atm_call_price is None or atm_put_price is None:
                stats["no_trade_atm_leg"] += 1
                continue

            expected_move = atm_call_price + atm_put_price
            im_pct = expected_move / spot if spot > 0 else float("nan")
            if not (im_pct == im_pct) or im_pct <= MIN_ELEVATED_MOVE:
                stats["not_elevated"] += 1
                continue

            short_put = round_strike(spot - PUT_CUSHION_MULT * expected_move)
            short_call = round_strike(spot + CALL_CUSHION_MULT * expected_move)
            long_put = round_strike(spot - WING_MULT * expected_move)
            long_call = round_strike(spot + WING_MULT * expected_move)
            if not (long_put < short_put < short_call < long_call):
                continue

            missing = [name for name, strike, m in [
                ("short_put", short_put, put_map), ("long_put", long_put, put_map),
                ("short_call", short_call, call_map), ("long_call", long_call, call_map),
            ] if strike not in m]
            if missing:
                stats["missing_wing_contract"] += 1
                continue

            wing_occ = {
                "short_put": put_map[short_put], "long_put": put_map[long_put],
                "short_call": call_map[short_call], "long_call": call_map[long_call],
            }
            wing_closes = get_closes_concurrent(wing_occ, entry_date_str)
            if any(v is None for v in wing_closes.values()):
                stats["no_trade_wing_leg"] += 1
                continue

            net_credit = (wing_closes["short_put"] + wing_closes["short_call"]) - \
                         (wing_closes["long_put"] + wing_closes["long_call"])
            put_width, call_width = short_put - long_put, long_call - short_call

            # ── 1. STRADDLE: buy real ATM call + put ──────────────────────
            straddle_cost = expected_move
            straddle_payoff = abs(real_outcome_price - atm_strike)
            straddle_pnl = round((straddle_payoff - straddle_cost) * 100, 2)

            # ── 2. STRANGLE: buy EVC's real short_put + short_call ────────
            strangle_cost = wing_closes["short_put"] + wing_closes["short_call"]
            strangle_payoff = max(short_put - real_outcome_price, 0) + max(real_outcome_price - short_call, 0)
            strangle_pnl = round((strangle_payoff - strangle_cost) * 100, 2)

            # ── 3. REVERSE IRON CONDOR: exact mirror of EVC's real condor ─
            condor_payoff = payoff_at_price(real_outcome_price, short_put, long_put,
                                             short_call, long_call, net_credit, put_width, call_width)
            reverse_pnl = round(-condor_payoff * 100, 2)

            stats["ok"] += 1
            events.append({
                "ticker": ticker, "earnings_date": str(edate_local), "timing": timing,
                "entry_date": str(entry_date), "outcome_date": str(outcome_date),
                "spot": round(spot, 2), "outcome_price": round(real_outcome_price, 2),
                "real_move_pct": round((real_outcome_price - spot) / spot * 100, 2),
                "implied_move_pct": round(im_pct * 100, 2),
                "atm_strike": atm_strike, "short_put": short_put, "short_call": short_call,
                "straddle_cost": round(straddle_cost * 100, 2), "straddle_pnl": straddle_pnl,
                "strangle_cost": round(strangle_cost * 100, 2), "strangle_pnl": strangle_pnl,
                "reverse_condor_cost": round(-net_credit * 100, 2), "reverse_condor_pnl": reverse_pnl,
            })
        if (ti + 1) % 10 == 0:
            print(f"  ... {ti + 1}/{len(tickers)} tickers processed, {len(events)} elevated-move events so far")
    return events, stats


def summarize(events: list[dict]) -> None:
    df = pd.DataFrame(events)
    for col in ("straddle_pnl", "strangle_pnl", "reverse_condor_pnl"):
        vals = df[col]
        wins = (vals > 0).sum()
        print(f"\n=== {col} ===")
        print(f"  n={len(vals)}  win_rate={wins/len(vals)*100:.1f}%  "
              f"mean=${vals.mean():+.2f}  median=${vals.median():+.2f}  "
              f"total=${vals.sum():+.2f}  worst=${vals.min():+.2f}  best=${vals.max():+.2f}")


def main():
    main_src = open("../main.py", encoding="utf-8").read()
    m = re.search(r'CANDIDATE_POOL: List\[str\] = \[(.*?)\]\n', main_src, re.DOTALL)
    all_tickers = re.findall(r'"([A-Z\.]+)"', m.group(1))
    tickers = all_tickers[::2]  # same every-2nd sampling as cushion_symmetry_test.py
    print(f"Universe: {len(all_tickers)} total, sampling {len(tickers)} (every 2nd)")
    print(f"Filtering for events with real ATM-implied move > {MIN_ELEVATED_MOVE:.0%} "
          f"(EVC's own MAX_MOVE_BOUND -- exactly what EVC itself skips)...")

    events, stats = build_elevated_events(tickers)
    print(f"\nPipeline funnel: {json.dumps(stats, indent=2)}")
    print(f"\n{len(events)} real elevated-implied-move events found")

    if not events:
        print("No events -- nothing to backtest.")
        return

    pd.DataFrame(events).to_csv("reverse_vol_backtest_events.csv", index=False)
    summarize(events)
    print("\nSaved reverse_vol_backtest_events.csv")


if __name__ == "__main__":
    main()
