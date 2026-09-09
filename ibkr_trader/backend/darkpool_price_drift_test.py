"""
Dark-pool print-sequence price-drift test -- CEO question 2026-08-30:
"if price is increasing with every [dark pool] print can that be a bullish
signal, if price is dropping with every print can that be sell-side activity."

This is exactly the same class of question as the F11 hypothesis already
tested in this codebase (same-day aggressive dark-pool print predicting
breakout continuation, found NO edge 2026-08-23) -- a plausible-sounding
heuristic that needs a real backtest before being treated as a signal, not
after. Same standard applied here.

Method: for each ticker/day in the real sample (darkpool_price_sequence_sample.json,
15 tickers x 5 real trading days, real Unusual Whales prints ordered by
execution time), compute whether the SEQUENCE of dark-pool print prices within
that session drifted up or down (two independent measures: simple first-vs-last,
and a proper linear regression slope of price vs. print order, more robust to a
single noisy first/last print). Compare the SIGN of that drift against the
REAL same-day price action (yfinance daily open->close) for that ticker/day --
this directly tests "does the direction implied by the dark-pool print sequence
match what the stock actually did that day."

Honest note: this correlates dark-pool print price drift against SAME-DAY
price action, not a forward-looking prediction -- since dark-pool prints
happen throughout the trading day, a same-day match mostly confirms dark-pool
prints get executed at prices that track the day's real move (not surprising
on its own), while a MISMATCH would be the more informative result (dark pool
flow moving opposite to the tape). Report the real hit rate either way.
"""
import json

import numpy as np
import pandas as pd
import yfinance as yf


def drift_metrics(prices: list[float]) -> dict:
    n = len(prices)
    if n < 3:
        return {"n": n, "first_last_pct": None, "slope_sign": None}
    first_last_pct = (prices[-1] - prices[0]) / prices[0] * 100
    x = np.arange(n)
    slope, _intercept = np.polyfit(x, prices, 1)
    slope_pct_per_print = slope / prices[0] * 100
    return {
        "n": n,
        "first_price": prices[0], "last_price": prices[-1],
        "first_last_pct": round(first_last_pct, 3),
        "slope_pct_per_print": round(slope_pct_per_print, 5),
    }


def main():
    with open("darkpool_price_sequence_sample.json") as f:
        prints = json.load(f)
    df = pd.DataFrame(prints)
    df = df[df["price"] > 0]
    print(f"Loaded {len(df)} real dark-pool prints, "
          f"{df['ticker'].nunique()} tickers x up to 5 days")

    rows = []
    for (ticker, day), sub in df.groupby(["ticker", "date"]):
        sub = sub.sort_values("executed_at")
        m = drift_metrics(sub["price"].tolist())
        if m["n"] < 10:
            continue  # too few real prints that day to trust a drift estimate

        try:
            hist = yf.Ticker(ticker).history(start=day, end=pd.Timestamp(day) + pd.Timedelta(days=1))
            if hist.empty:
                continue
            real_open = float(hist["Open"].iloc[0])
            real_close = float(hist["Close"].iloc[0])
            real_pct = round((real_close - real_open) / real_open * 100, 3)
        except Exception as exc:
            print(f"  [SKIP] {ticker} {day}: {exc}")
            continue

        dp_dir = "up" if m["first_last_pct"] > 0 else "down" if m["first_last_pct"] < 0 else "flat"
        real_dir = "up" if real_pct > 0 else "down" if real_pct < 0 else "flat"
        rows.append({
            "ticker": ticker, "date": day, "n_prints": m["n"],
            "dp_first_last_pct": m["first_last_pct"], "dp_slope_pct_per_print": m["slope_pct_per_print"],
            "dp_direction": dp_dir, "real_day_pct": real_pct, "real_direction": real_dir,
            "match": dp_dir == real_dir,
        })

    res = pd.DataFrame(rows)
    res.to_csv("darkpool_price_drift_test_results.csv", index=False)
    print(f"\n{len(res)} real ticker-days with a usable dark-pool print sequence (n>=10 prints)")
    print(res.to_string(index=False))

    usable = res[res["dp_direction"] != "flat"]
    hit_rate = usable["match"].mean() * 100 if len(usable) else None
    print(f"\n=== RESULT: dark-pool print-sequence direction vs. real same-day price direction ===")
    print(f"n = {len(usable)} real ticker-days")
    print(f"hit rate (dark-pool drift direction matched real day direction): {hit_rate:.1f}%" if hit_rate is not None else "n/a")
    print("(50% = coin flip / no relationship; meaningfully above 50% -- worth a closer look; "
          "meaningfully at/below 50% -- same conclusion as the F11 test: no real edge)")


if __name__ == "__main__":
    main()
