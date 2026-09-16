"""
CRT backtest -- use case 2 of 3, per CEO request 2026-09-11/12: "Day Trader
signals and score enrichment."

Different target metric from the SPY 0DTE test (crt_spy_0dte_backtest.py):
Day Trader is a momentum/breakout strategy that wants CONTINUATION, not
range compression. So the real question here is directional: does a CRT
reclaim predict a same-day continuation move in the reclaim's direction
(bullish reclaim after a downside sweep -> positive rest-of-day return;
bearish reclaim after an upside sweep -> negative rest-of-day return)?

Universe: Day Trader's own REAL historically-traded tickers (from
trade_journal.db, strategy_type='DAY_BREAKOUT', is_paper=0) rather than a
synthetic universe -- most directly relevant, even though testing across
every real trading day for these names (not just the 33 actual trade
instances, which is far too small a sample on its own) gives the
statistical power a real backtest needs.

Same CRT definition as the validated SPY test: 15-min sweep window,
30-min reclaim window, prior real day's RTH high/low as the reference
range. Data: real IBKR 5-min bars, paginated (see crt_spy_0dte_backtest.py
for why -- IBKR rejects multi-year single-shot intraday requests).
"""
import argparse
import json

import numpy as np
import pandas as pd
from ib_insync import IB, Stock, util
from scipy.stats import ttest_ind

SWEEP_WINDOW_MIN = 15
RECLAIM_WINDOW_MIN = 30
SWEEP_MIN_TICKS_PCT = 0.0005  # 0.05% of spot -- tickers here span $10s to $100s, a flat $ tick doesn't scale


def fetch_daily_bars(ib: IB, ticker: str, years: int):
    contract = ib.qualifyContracts(Stock(ticker, "SMART", "USD"))[0]
    bars = ib.reqHistoricalData(contract, endDateTime="", durationStr=f"{years} Y",
                                 barSizeSetting="1 day", whatToShow="TRADES", useRTH=True, formatDate=1)
    return util.df(bars) if bars else None


def fetch_5min_bars(ib: IB, ticker: str, years: int):
    contract = ib.qualifyContracts(Stock(ticker, "SMART", "USD"))[0]
    all_bars = []
    end_dt = ""
    for _ in range(years * 12):
        bars = ib.reqHistoricalData(contract, endDateTime=end_dt, durationStr="1 M",
                                     barSizeSetting="5 mins", whatToShow="TRADES", useRTH=True, formatDate=1)
        if not bars:
            break
        all_bars = bars + all_bars
        end_dt = bars[0].date.strftime("%Y%m%d %H:%M:%S") if hasattr(bars[0].date, "strftime") else str(bars[0].date)
        ib.sleep(1)
    return util.df(all_bars) if all_bars else None


def analyze_ticker(daily_df, intraday_df):
    daily_df = daily_df.copy(); intraday_df = intraday_df.copy()
    daily_df["date_only"] = pd.to_datetime(daily_df["date"], utc=True).dt.date
    intraday_df["date_only"] = pd.to_datetime(intraday_df["date"], utc=True).dt.date
    day_ref = {row["date_only"]: (row["high"], row["low"]) for _, row in daily_df.iterrows()}
    trading_days = sorted(intraday_df["date_only"].unique())

    results = []
    for idx in range(1, len(trading_days)):
        today, prior = trading_days[idx], trading_days[idx - 1]
        if prior not in day_ref:
            continue
        ref_high, ref_low = day_ref[prior]

        day_bars = intraday_df[intraday_df["date_only"] == today].reset_index(drop=True)
        if len(day_bars) < 10:
            continue
        session_open = day_bars.iloc[0]["open"]
        tick = session_open * SWEEP_MIN_TICKS_PCT
        sweep_bars = max(1, SWEEP_WINDOW_MIN // 5)
        reclaim_bars = max(1, RECLAIM_WINDOW_MIN // 5)
        early = day_bars.iloc[:sweep_bars]

        swept_up = (early["high"] >= ref_high + tick).any()
        swept_down = (early["low"] <= ref_low - tick).any()

        crt_day, direction, reclaim_end_idx = False, None, sweep_bars
        if swept_up:
            sweep_idx = early[early["high"] >= ref_high + tick].index[0]
            window = day_bars.iloc[sweep_idx:sweep_idx + reclaim_bars]
            if (window["close"] < ref_high).any():
                crt_day, direction, reclaim_end_idx = True, "bearish", sweep_idx + reclaim_bars
        elif swept_down:
            sweep_idx = early[early["low"] <= ref_low - tick].index[0]
            window = day_bars.iloc[sweep_idx:sweep_idx + reclaim_bars]
            if (window["close"] > ref_low).any():
                crt_day, direction, reclaim_end_idx = True, "bullish", sweep_idx + reclaim_bars

        rest_of_day = day_bars.iloc[reclaim_end_idx:]
        if len(rest_of_day) < 5:
            continue
        anchor_price = day_bars.iloc[min(reclaim_end_idx, len(day_bars) - 1)]["close"]
        close_price = rest_of_day.iloc[-1]["close"]
        continuation_ret_pct = (close_price - anchor_price) / anchor_price

        results.append({"date": str(today), "crt_day": crt_day, "direction": direction,
                         "continuation_ret_pct": round(continuation_ret_pct, 5)})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", required=True, help="comma-separated")
    ap.add_argument("--years", type=int, default=2)
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    ib = IB()
    ib.errorEvent += lambda *a: None
    ib.connect("127.0.0.1", 7496, clientId=1954, timeout=30)

    all_bullish, all_bearish, all_non = [], [], []
    per_ticker = {}
    try:
        for t in tickers:
            print(f"Fetching {t}...")
            try:
                daily_df = fetch_daily_bars(ib, t, args.years)
                intraday_df = fetch_5min_bars(ib, t, args.years)
                if daily_df is None or intraday_df is None:
                    print(f"  {t}: no data, skipping")
                    continue
                results = analyze_ticker(daily_df, intraday_df)
            except Exception as e:
                print(f"  {t}: error {e}, skipping")
                continue
            bullish = [r["continuation_ret_pct"] for r in results if r["crt_day"] and r["direction"] == "bullish"]
            bearish = [r["continuation_ret_pct"] for r in results if r["crt_day"] and r["direction"] == "bearish"]
            non_crt = [r["continuation_ret_pct"] for r in results if not r["crt_day"]]
            all_bullish += bullish; all_bearish += bearish; all_non += non_crt
            per_ticker[t] = {"n_days": len(results), "n_bullish_crt": len(bullish), "n_bearish_crt": len(bearish),
                              "avg_bullish_ret_pct": round(float(np.mean(bullish)) * 100, 3) if bullish else None,
                              "avg_bearish_ret_pct": round(float(np.mean(bearish)) * 100, 3) if bearish else None,
                              "avg_non_crt_ret_pct": round(float(np.mean(non_crt)) * 100, 3) if non_crt else None}
            if len(bullish) > 5 and len(non_crt) > 5:
                _, p = ttest_ind(bullish, non_crt, equal_var=False)
                per_ticker[t]["bullish_vs_non_pvalue"] = round(p, 4)
            if len(bearish) > 5 and len(non_crt) > 5:
                _, p = ttest_ind(bearish, non_crt, equal_var=False)
                per_ticker[t]["bearish_vs_non_pvalue"] = round(p, 4)
            print(f"  {t}: {json.dumps(per_ticker[t])}")
    finally:
        ib.disconnect()

    summary = {
        "n_tickers": len(per_ticker), "per_ticker": per_ticker,
        "pooled": {
            "n_bullish_crt": len(all_bullish), "n_bearish_crt": len(all_bearish), "n_non_crt": len(all_non),
            "avg_bullish_crt_ret_pct": round(float(np.mean(all_bullish)) * 100, 3) if all_bullish else None,
            "avg_bearish_crt_ret_pct": round(float(np.mean(all_bearish)) * 100, 3) if all_bearish else None,
            "avg_non_crt_ret_pct": round(float(np.mean(all_non)) * 100, 3) if all_non else None,
        }
    }
    if all_bullish and all_non:
        _, p_bull = ttest_ind(all_bullish, all_non, equal_var=False)
        summary["pooled"]["bullish_vs_non_pvalue"] = round(p_bull, 4)
    if all_bearish and all_non:
        _, p_bear = ttest_ind(all_bearish, all_non, equal_var=False)
        summary["pooled"]["bearish_vs_non_pvalue"] = round(p_bear, 4)

    print(json.dumps(summary["pooled"], indent=2))
    with open("crt_research/crt_daytrader_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved: crt_research/crt_daytrader_results.json")


if __name__ == "__main__":
    main()
