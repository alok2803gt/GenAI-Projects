"""
Dark-pool weekly review -- CEO request 2026-08-30: "run dark pool activity
last week and check price action ... run correlation with price action."

One-off historical pull, NOT a scheduled job (unlike darkpool_activity_monitor.py,
which only watches live/recent prints, and darkpool-levels-calculator, which only
looks at the single most recent trading day). This uses
UnusualWhalesClient.ticker_trades(ticker, date=...) -- real per-ticker, per-day
dark-pool prints -- across the SAME curated 112-ticker universe already used by
darkpool-levels-calculator/darkpool_activity_monitor.py, for every real trading
day last week (2026-08-24 through 2026-08-28), then correlates each ticker's
weekly dark-pool premium against its REAL price action over the same week
(yfinance daily OHLC).

Epistemic status -- same as every other dark-pool tool in this codebase, stated
explicitly here too so this doesn't get treated as more than it is:
  - Real, observed trade data (premium/size/price), NOT modeled.
  - No buy/sell aggressor side in this data -- a big print could be
    accumulation, distribution, or neutral block-facilitation crossing.
  - NOT validated as predictive. The one related hypothesis actually backtested
    in this codebase (F11: same-day aggressive dark-pool print predicts
    breakout continuation, tested against breakout_scanner.py's real alert
    history 2026-08-23) came back with NO edge (n=41-392, well-sampled).
  - This script reports a correlation for the CEO to eyeball, not a signal --
    any apparent relationship here is descriptive, not a validated trading
    edge, and should not be wired into any live strategy without its own
    real backtest.
"""
import json
import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from unusual_whales_client import UnusualWhalesClient

TRADING_DAYS_LAST_WEEK = ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28"]
PACE_S = 0.3


def load_universe() -> dict:
    import sys
    sys.path.insert(0, r"C:\Users\AlokD\.claude\skills\darkpool-levels-calculator")
    import calc_darkpool_levels as m
    return dict(m.UNIVERSE)


def main():
    universe = load_universe()
    tickers = sorted(universe.keys())
    print(f"Universe: {len(tickers)} tickers, {len(TRADING_DAYS_LAST_WEEK)} trading days "
          f"({TRADING_DAYS_LAST_WEEK[0]} to {TRADING_DAYS_LAST_WEEK[-1]})")

    client = UnusualWhalesClient()
    per_ticker = {t: {"total_premium": 0.0, "n_prints": 0, "biggest": None, "by_day": {}} for t in tickers}

    for ti, ticker in enumerate(tickers):
        for day in TRADING_DAYS_LAST_WEEK:
            try:
                trades = client.ticker_trades(ticker, date=day, limit=500,
                                               order_by="premium", order="desc")
            except Exception as exc:
                print(f"  [SKIP] {ticker} {day}: {exc}")
                time.sleep(PACE_S)
                continue
            if not trades:
                time.sleep(PACE_S)
                continue
            day_premium = 0.0
            for tr in trades:
                if tr.get("canceled"):
                    continue  # same "non-canceled print" rule darkpool_activity_monitor.py uses
                prem = float(tr.get("premium") or 0)
                day_premium += prem
                cur_biggest = per_ticker[ticker]["biggest"]
                if cur_biggest is None or prem > cur_biggest["premium"]:
                    per_ticker[ticker]["biggest"] = {
                        "premium": prem, "size": tr.get("size"), "price": tr.get("price"),
                        "executed_at": tr.get("executed_at"), "date": day,
                    }
            per_ticker[ticker]["total_premium"] += day_premium
            per_ticker[ticker]["n_prints"] += len(trades)
            per_ticker[ticker]["by_day"][day] = round(day_premium, 2)
            time.sleep(PACE_S)
        if (ti + 1) % 20 == 0:
            print(f"  ... {ti+1}/{len(tickers)} tickers processed")

    # Relative ranking too -- flat premium trivially favors mega-caps (SPY/QQQ
    # move $100M+ prints routinely). This account's own darkpool_activity_monitor.py
    # already learned this lesson (2026-08-24, real ADV check found a flat $5M
    # print is 0.0145% of MU/SPY's ADV but 1.64% of LULU's -- a ~113x gap) --
    # applying the same real ADV cache here surfaces UNUSUALLY concentrated
    # names, not just the biggest tickers.
    adv = {}
    try:
        with open("darkpool_adv_cache.json") as f:
            adv = json.load(f).get("adv", {})
    except Exception as exc:
        print(f"[WARN] could not load darkpool_adv_cache.json, skipping relative ranking: {exc}")

    for t, v in per_ticker.items():
        a = adv.get(t)
        v["weekly_adv_ratio_pct"] = round(v["total_premium"] / (5 * a) * 100, 3) if a else None

    ranked_abs = sorted(per_ticker.items(), key=lambda kv: kv[1]["total_premium"], reverse=True)
    ranked_rel = sorted(
        [(t, v) for t, v in per_ticker.items() if v["weekly_adv_ratio_pct"] is not None],
        key=lambda kv: kv[1]["weekly_adv_ratio_pct"], reverse=True)

    with open("darkpool_weekly_review.json", "w") as f:
        json.dump({"week": TRADING_DAYS_LAST_WEEK, "universe_size": len(tickers),
                   "tickers": {k: v for k, v in ranked_abs}}, f, indent=2)

    top_n = [t for t, v in ranked_abs if v["total_premium"] > 0][:20]
    print(f"\n=== Top {len(top_n)} tickers by real weekly dark-pool premium (absolute $) ===")
    for t in top_n:
        v = per_ticker[t]
        print(f"{t:6} sector={universe[t]:10} total_premium=${v['total_premium']:,.0f}  "
              f"n_prints={v['n_prints']:4}  {v['weekly_adv_ratio_pct']}% of 5-day ADV  "
              f"biggest=${v['biggest']['premium']:,.0f} @ ${v['biggest']['price']} on {v['biggest']['date']}")

    print(f"\n=== Top 20 tickers by dark-pool premium RELATIVE to own 5-day ADV "
          f"(unusually concentrated, not just biggest) ===")
    for t, v in ranked_rel[:20]:
        print(f"{t:6} sector={universe[t]:10} {v['weekly_adv_ratio_pct']:6.3f}% of 5-day ADV  "
              f"total_premium=${v['total_premium']:,.0f}  n_prints={v['n_prints']:4}")

    # Union of both rankings (top-15 absolute + top-15 relative) gets the
    # yfinance price-action pull -- the interesting names for correlation
    # aren't necessarily the same ones in each view.
    top_n = list(dict.fromkeys(
        [t for t, v in ranked_abs if v["total_premium"] > 0][:15]
        + [t for t, _v in ranked_rel[:15]]
    ))

    print("\n=== Pulling real weekly price action (yfinance) for top tickers ===")
    price_rows = []
    for t in top_n:
        try:
            hist = yf.Ticker(t).history(
                start=TRADING_DAYS_LAST_WEEK[0],
                end=str(date.fromisoformat(TRADING_DAYS_LAST_WEEK[-1]) + timedelta(days=1)),
            )
            if hist.empty:
                continue
            week_open = float(hist["Open"].iloc[0])
            week_close = float(hist["Close"].iloc[-1])
            week_high = float(hist["High"].max())
            week_low = float(hist["Low"].min())
            pct = round((week_close - week_open) / week_open * 100, 2)
            price_rows.append({
                "ticker": t, "week_open": round(week_open, 2), "week_close": round(week_close, 2),
                "week_high": round(week_high, 2), "week_low": round(week_low, 2),
                "week_pct_change": pct, "dp_total_premium": round(per_ticker[t]["total_premium"], 0),
                "dp_n_prints": per_ticker[t]["n_prints"],
            })
        except Exception as exc:
            print(f"  [SKIP PRICE] {t}: {exc}")

    df = pd.DataFrame(price_rows)
    df.to_csv("darkpool_weekly_review_price_action.csv", index=False)
    print("\n=== Dark-pool activity vs. real weekly price action ===")
    print(df.to_string(index=False))
    print("\nSaved darkpool_weekly_review.json and darkpool_weekly_review_price_action.csv")


if __name__ == "__main__":
    main()
