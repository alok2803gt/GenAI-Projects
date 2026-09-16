"""
Candle Range Theory (CRT) backtest -- use case 1 of 3, per CEO request
2026-09-11: "increase win probability of SPY 0DTE."

Precise, falsifiable definition used here (ICT/SMC "CRT" as commonly
described, stated explicitly rather than assumed understood):
  - Reference range = PRIOR real trading day's RTH high/low (from real
    IBKR daily bars).
  - Sweep = within the first SWEEP_WINDOW_MIN minutes of today's RTH
    session, price trades beyond the reference high (bullish sweep setup)
    or reference low (bearish sweep setup) by at least SWEEP_MIN_TICKS.
  - Reclaim = within RECLAIM_WINDOW_MIN minutes of the sweep, a 5-min bar
    CLOSES back inside the reference range (below the reference high after
    an upside sweep, or above the reference low after a downside sweep).
  - "CRT day" = a sweep followed by a real reclaim, confirmed by the first
    RECLAIM_WINDOW_MIN minutes of the session.

What this tests for SPY 0DTE specifically: 0DTE butterflies (SPY_custom
box centered near the open) profit from the underlying staying RANGE-BOUND
the rest of the day. The real question isn't "does CRT predict direction"
(that's use cases 2/3) -- it's "does a confirmed CRT reclaim predict a
TIGHTER realized range for the remainder of the session," which would mean
entering only on CRT-reclaim days should raise the butterfly's real win
probability. Measures, for CRT days vs. all other days:
  - Realized range of the REST of the session (first bar after the
    reclaim window through the close), as a % of the day's open.
  - How often that rest-of-day range stays within typical 0DTE butterfly
    wing widths (tested at a few realistic wing-width percentages).

Data: real IBKR 5-min bars (reqHistoricalData) -- yfinance's intraday
history only covers ~60 days, not enough for a real multi-regime
backtest; IBKR gives a genuine 1-2 year lookback for SPY 5-min bars.
"""
import argparse
import json
from datetime import datetime, timedelta

import numpy as np
from ib_insync import IB, Stock, util

TICKER = "SPY"  # overridden via --ticker; module-level default kept for direct-import callers
SWEEP_WINDOW_MIN = 30      # first 30 min of RTH to look for a sweep
RECLAIM_WINDOW_MIN = 30    # minutes after the sweep bar to confirm a reclaim
SWEEP_MIN_TICKS = 0.05     # min $ beyond the reference level to count as a real sweep, not noise
WING_WIDTHS_TESTED = [0.003, 0.005, 0.008]  # realistic 0DTE butterfly wing % (0.3%/0.5%/0.8% of spot)


def fetch_daily_bars(ib: IB, years: int = 2):
    contract = ib.qualifyContracts(Stock(TICKER, "SMART", "USD"))[0]
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr=f"{years} Y",
        barSizeSetting="1 day", whatToShow="TRADES", useRTH=True, formatDate=1)
    return util.df(bars)


def fetch_5min_bars(ib: IB, years: int = 2):
    """IBKR times out / rejects a multi-year request for sub-daily bars in
    one call -- pull in 1-month chunks, walking backward via endDateTime,
    same pagination pattern IBKR's own docs recommend for intraday history."""
    contract = ib.qualifyContracts(Stock(TICKER, "SMART", "USD"))[0]
    all_bars = []
    end_dt = ""
    months_needed = years * 12
    for _ in range(months_needed):
        bars = ib.reqHistoricalData(
            contract, endDateTime=end_dt, durationStr="1 M",
            barSizeSetting="5 mins", whatToShow="TRADES", useRTH=True, formatDate=1)
        if not bars:
            break
        all_bars = bars + all_bars
        end_dt = bars[0].date.strftime("%Y%m%d %H:%M:%S") if hasattr(bars[0].date, "strftime") else str(bars[0].date)
        ib.sleep(1)  # stay well under IBKR's pacing limits
    return util.df(all_bars)


def analyze(daily_df, intraday_df):
    import pandas as pd
    # ib_insync returns daily-bar dates as plain date objects (no time
    # component) but intraday-bar dates as full datetimes -- normalize both
    # through pd.to_datetime rather than assuming a consistent dtype.
    daily_df["date_only"] = pd.to_datetime(daily_df["date"], utc=True).dt.date
    intraday_df["date_only"] = pd.to_datetime(intraday_df["date"], utc=True).dt.date
    day_ref = {row["date_only"]: (row["high"], row["low"]) for _, row in daily_df.iterrows()}
    trading_days = sorted(intraday_df["date_only"].unique())

    results = []
    for idx in range(1, len(trading_days)):
        today = trading_days[idx]
        prior = trading_days[idx - 1]
        if prior not in day_ref:
            continue
        ref_high, ref_low = day_ref[prior]

        day_bars = intraday_df[intraday_df["date_only"] == today].reset_index(drop=True)
        if len(day_bars) < 10:
            continue
        session_open = day_bars.iloc[0]["open"]
        bar_minutes = 5
        sweep_bars = max(1, SWEEP_WINDOW_MIN // bar_minutes)
        reclaim_bars = max(1, RECLAIM_WINDOW_MIN // bar_minutes)

        early = day_bars.iloc[:sweep_bars]
        swept_up = (early["high"] >= ref_high + SWEEP_MIN_TICKS).any()
        swept_down = (early["low"] <= ref_low - SWEEP_MIN_TICKS).any()

        crt_day = False
        direction = None
        reclaim_end_idx = sweep_bars
        if swept_up:
            sweep_bar_idx = early[early["high"] >= ref_high + SWEEP_MIN_TICKS].index[0]
            window = day_bars.iloc[sweep_bar_idx:sweep_bar_idx + reclaim_bars]
            if (window["close"] < ref_high).any():
                crt_day = True
                direction = "bearish_reversal_expected"
                reclaim_end_idx = sweep_bar_idx + reclaim_bars
        elif swept_down:
            sweep_bar_idx = early[early["low"] <= ref_low - SWEEP_MIN_TICKS].index[0]
            window = day_bars.iloc[sweep_bar_idx:sweep_bar_idx + reclaim_bars]
            if (window["close"] > ref_low).any():
                crt_day = True
                direction = "bullish_reversal_expected"
                reclaim_end_idx = sweep_bar_idx + reclaim_bars

        rest_of_day = day_bars.iloc[reclaim_end_idx:]
        if len(rest_of_day) < 5:
            continue
        rest_high = rest_of_day["high"].max()
        rest_low = rest_of_day["low"].min()
        rest_range_pct = (rest_high - rest_low) / session_open

        results.append({
            "date": str(today), "crt_day": crt_day, "direction": direction,
            "session_open": round(session_open, 2), "rest_range_pct": round(rest_range_pct, 5),
        })

    return results


def summarize(results):
    crt_days = [r for r in results if r["crt_day"]]
    non_crt_days = [r for r in results if not r["crt_day"]]

    def summarize_group(group, label):
        if not group:
            return {"label": label, "n": 0}
        ranges = [r["rest_range_pct"] for r in group]
        out = {
            "label": label, "n": len(group),
            "avg_rest_of_day_range_pct": round(float(np.mean(ranges)) * 100, 3),
            "median_rest_of_day_range_pct": round(float(np.median(ranges)) * 100, 3),
        }
        for w in WING_WIDTHS_TESTED:
            pct_within = sum(1 for r in ranges if r <= w) / len(ranges)
            out[f"pct_within_{w*100:.1f}pct_wing"] = round(pct_within, 4)
        return out

    return {
        "total_days": len(results),
        "crt_days": len(crt_days),
        "crt_day_rate": round(len(crt_days) / len(results), 4) if results else None,
        "crt_days_summary": summarize_group(crt_days, "CRT reclaim days"),
        "non_crt_days_summary": summarize_group(non_crt_days, "all other days"),
    }


def main():
    ap = argparse.ArgumentParser()
    global SWEEP_WINDOW_MIN, RECLAIM_WINDOW_MIN, TICKER
    ap.add_argument("--years", type=int, default=2)
    ap.add_argument("--ticker", type=str, default=TICKER)
    ap.add_argument("--use-cache", action="store_true", help="reuse cached bars from a prior run instead of re-fetching from IBKR")
    ap.add_argument("--sweep-window-min", type=int, default=SWEEP_WINDOW_MIN)
    ap.add_argument("--reclaim-window-min", type=int, default=RECLAIM_WINDOW_MIN)
    args = ap.parse_args()
    SWEEP_WINDOW_MIN = args.sweep_window_min
    RECLAIM_WINDOW_MIN = args.reclaim_window_min
    TICKER = args.ticker.upper()

    import pandas as pd
    daily_cache = f"{TICKER.lower()}_daily_bars_cache.csv"
    intraday_cache = f"{TICKER.lower()}_5min_bars_cache.csv"
    import os
    if args.use_cache and os.path.exists(daily_cache) and os.path.exists(intraday_cache):
        print("Using cached bars from a prior fetch.")
        daily_df = pd.read_csv(daily_cache, parse_dates=["date"])
        intraday_df = pd.read_csv(intraday_cache, parse_dates=["date"])
    else:
        ib = IB()
        ib.errorEvent += lambda *a: None
        ib.connect("127.0.0.1", 7496, clientId=1953, timeout=30)
        try:
            print(f"Fetching real {TICKER} daily bars...")
            daily_df = fetch_daily_bars(ib, args.years)
            print(f"  {len(daily_df)} daily bars")
            print(f"Fetching real {TICKER} 5-min bars (this can take a minute)...")
            intraday_df = fetch_5min_bars(ib, args.years)
            print(f"  {len(intraday_df)} 5-min bars")
        finally:
            ib.disconnect()
        daily_df.to_csv(daily_cache, index=False)
        intraday_df.to_csv(intraday_cache, index=False)

    results = analyze(daily_df, intraday_df)
    summary = summarize(results)
    summary["params"] = {"sweep_window_min": SWEEP_WINDOW_MIN, "reclaim_window_min": RECLAIM_WINDOW_MIN}
    print(json.dumps(summary, indent=2))

    out_name = f"crt_{TICKER.lower()}_0dte_results_sweep{SWEEP_WINDOW_MIN}_reclaim{RECLAIM_WINDOW_MIN}.json"
    with open(out_name, "w") as f:
        json.dump({"summary": summary, "daily_results": results}, f, indent=2)
    print(f"Saved: {out_name}")


if __name__ == "__main__":
    main()
