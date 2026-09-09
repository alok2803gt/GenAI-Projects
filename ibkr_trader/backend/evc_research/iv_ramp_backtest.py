"""
Does implied vol ramp up predictably ahead of a KNOWN earnings date, and can
that ramp itself be captured -- buy the ATM straddle N trading days before
earnings, sell it the day before the reaction (T-1), never holding through
the actual event/IV-crush at all? This is a different bet than
reverse_vol_backtest.py: that one bet the REALIZED move would beat the
ALREADY-elevated implied move (it lost on average). This one doesn't bet on
the outcome at all -- it only bets that IV itself rises as a known catalyst
approaches, which is a real, mechanical vega effect independent of which
way (or how far) the stock eventually moves.

Reuses the exact same 41 real elevated-move events from
reverse_vol_backtest_events.csv (real Polygon option prices, same tickers/
dates/strikes/expiry already found and vetted) -- for each, re-derives the
same real ATM call/put OCC symbols and pulls REAL historical closing prices
for that SAME contract at several real trading-day snapshots before the
already-known entry_date (which is T-1, the last close before the earnings
reaction). This traces the real, actual price path of the same straddle as
the event approached -- no modeling, real quoted closes throughout.

Caveats stated explicitly:
  - A short-dated option's own theta (time decay) is a real, constant drag
    working against a long holder over the SAME days IV may be rising --
    this test measures net P&L (which already nets both effects), not IV
    in isolation, since IV alone isn't independently observable from daily
    closes without a pricing model this account has deliberately avoided
    using here.
  - Contract liquidity/quotes further from the event can be thinner --
    missing data at a given lookback is skipped for that event rather than
    guessed.
  - Real bid-ask slippage on entry and exit is not modeled (same
    already-disclosed limitation as every other real-data backtest here).
"""
import io
import sys

import pandas as pd

from cushion_symmetry_test import fetch_ticker_data, get_contract_grid, get_closes_concurrent, round_strike

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LOOKBACKS_TD = [1, 2, 3, 5, 7, 10, 15, 20]  # trading days before entry_date (entry_date itself = T-1 reaction eve)


def main():
    events_df = pd.read_csv("reverse_vol_backtest_events.csv")
    print(f"Re-deriving real ATM contracts + pulling real historical closes for "
          f"{len(events_df)} events, {len(LOOKBACKS_TD)} lookback snapshots each...")

    rows = []
    tickers_cache = {}
    for i, row in enumerate(events_df.itertuples(), 1):
        ticker = row.ticker
        entry_date = pd.Timestamp(row.entry_date).date()
        atm_strike = row.atm_strike

        if ticker not in tickers_cache:
            hist, edf, splits = fetch_ticker_data(ticker, retries=2)
            tickers_cache[ticker] = hist
        hist = tickers_cache[ticker]
        if hist is None:
            continue
        trading_days = sorted(hist.index.date)
        if entry_date not in trading_days:
            continue
        entry_idx = trading_days.index(entry_date)

        expiry_used, call_map, put_map = get_contract_grid(ticker, entry_date)
        if expiry_used is None:
            continue
        call_occ, put_occ = call_map.get(atm_strike), put_map.get(atm_strike)
        if call_occ is None or put_occ is None:
            continue

        # Real T-1 (entry_date itself) straddle price -- the reference point
        base_closes = get_closes_concurrent({"call": call_occ, "put": put_occ}, entry_date.isoformat())
        if base_closes["call"] is None or base_closes["put"] is None:
            continue
        base_straddle = base_closes["call"] + base_closes["put"]

        for lb in LOOKBACKS_TD:
            snap_idx = entry_idx - lb
            if snap_idx < 0:
                continue
            snap_date = trading_days[snap_idx]
            snap_closes = get_closes_concurrent({"call": call_occ, "put": put_occ}, snap_date.isoformat())
            if snap_closes["call"] is None or snap_closes["put"] is None:
                continue
            snap_straddle = snap_closes["call"] + snap_closes["put"]
            pnl = round((base_straddle - snap_straddle) * 100, 2)
            rows.append({
                "ticker": ticker, "entry_date": str(entry_date), "lookback_td": lb,
                "snap_date": str(snap_date), "snap_straddle": round(snap_straddle, 2),
                "t_minus_1_straddle": round(base_straddle, 2), "pnl": pnl,
                "pct_gain": round((base_straddle / snap_straddle - 1) * 100, 1) if snap_straddle else None,
            })
        if i % 10 == 0:
            print(f"  ... {i}/{len(events_df)} events processed")

    if not rows:
        print("No real snapshot data found -- nothing to report.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("iv_ramp_backtest_rows.csv", index=False)
    print(f"\nSaved {len(df)} real snapshot rows to iv_ramp_backtest_rows.csv\n")

    print("=== Real straddle price path by lookback (buy at T-minus-N, mark at T-1) ===")
    g = df.groupby("lookback_td").agg(
        n=("pnl", "count"),
        win_rate=("pnl", lambda x: (x > 0).mean() * 100),
        mean_pnl=("pnl", "mean"),
        median_pnl=("pnl", "median"),
        mean_pct_gain=("pct_gain", "mean"),
    ).round(2)
    print(g.to_string())


if __name__ == "__main__":
    main()
