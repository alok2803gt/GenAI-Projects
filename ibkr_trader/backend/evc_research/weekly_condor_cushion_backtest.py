"""
Real weekly iron-condor cushion-vs-containment backtest for SPY and QQQ --
the "build it around what's already proven" answer from tonight's PCR/GEX/
VRP work, all three of which failed or came back too weak to trust.

Method, directly adapted from safe-income-screener's own validated
backtest_cushion.py methodology (real 2-year, ~15,900-entry backtest,
95.4%/89.8% containment at 15% cushion for CSPs) -- but for a WEEKLY iron
condor: real price only, no options data needed at all, so this sidesteps
tonight's CBOE quota problem entirely and can use a much longer, real,
multi-regime history (SPY/QQQ back to 2015 -- includes 2018 vol spike,
2020 COVID crash, 2022 bear market, and 2026 as real stress tests, not
just one calm recent stretch, unlike tonight's earlier PCR pilot mistake).

Week definition matches this account's own real, already-live GOOG weekly
condor (IBKR-GOOGCondorMonday): Monday entry (or first trading day of the
week), Friday expiry (or last trading day of the week) -- one real trading
week, symmetric cushion tested on both sides (put cushion == call cushion
here; the account's own EVC work tonight showed asymmetric cushions can
matter, but symmetric is the honest baseline before adding that complexity).

Real caveat stated up front: this tests whether the underlying's REALIZED
weekly move stayed inside a hypothetical cushion band -- it does not model
real credit received, real bid-ask, or real fill mechanics (no historical
options data used, unlike the CBOE-based backtests tonight). Containment
rate alone tells you whether the STRUCTURE would have survived, not
whether it was profitable after costs -- a real, deliberate scope
narrowing given tonight's data-budget lesson.
"""
import json
from datetime import date

import numpy as np
import yfinance as yf

TICKERS = ["SPY", "QQQ"]
CUSHIONS_PCT = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
YEARS_BACK = "10y"


def real_weekly_returns(ticker: str) -> list[dict]:
    hist = yf.Ticker(ticker).history(period=YEARS_BACK, interval="1d")
    hist.index = hist.index.tz_localize(None)
    hist["iso_year"] = hist.index.isocalendar().year
    hist["iso_week"] = hist.index.isocalendar().week

    weeks = []
    for (yr, wk), grp in hist.groupby(["iso_year", "iso_week"]):
        grp = grp.sort_index()
        if len(grp) < 2:
            continue  # holiday-shortened week with only one session -- no real weekly move to test
        entry_px = grp["Close"].iloc[0]
        exit_px = grp["Close"].iloc[-1]
        # Real intraweek extremes too -- a condor breaches if price EVER
        # touches the strike intraweek, not just at Friday's close.
        week_high = grp["High"].max()
        week_low = grp["Low"].min()
        weeks.append({
            "ticker": ticker,
            "year": int(yr), "week": int(wk),
            "entry_date": grp.index[0].date().isoformat(),
            "exit_date": grp.index[-1].date().isoformat(),
            "entry_px": round(float(entry_px), 2),
            "exit_px": round(float(exit_px), 2),
            "week_high": round(float(week_high), 2),
            "week_low": round(float(week_low), 2),
            "close_to_close_pct": round((exit_px / entry_px - 1) * 100, 3),
            "max_up_pct": round((week_high / entry_px - 1) * 100, 3),
            "max_down_pct": round((week_low / entry_px - 1) * 100, 3),
        })
    return weeks


def containment_rate(weeks: list[dict], cushion_pct: float, use_intraweek: bool) -> dict:
    n = len(weeks)
    if use_intraweek:
        # Real breach = price EVER traded beyond the strike during the week
        contained = sum(1 for w in weeks if w["max_up_pct"] <= cushion_pct and w["max_down_pct"] >= -cushion_pct)
    else:
        # Close-to-close only (matches backtest_cushion.py's own convention)
        contained = sum(1 for w in weeks if abs(w["close_to_close_pct"]) <= cushion_pct)
    return {"n": n, "contained": contained, "rate": round(contained / n, 4) if n else None}


def main():
    all_results = {}
    for ticker in TICKERS:
        print(f"\n=== {ticker}: fetching real {YEARS_BACK} daily history ===")
        weeks = real_weekly_returns(ticker)
        print(f"{len(weeks)} real trading weeks, {weeks[0]['entry_date']} to {weeks[-1]['exit_date']}")

        ticker_result = {"n_weeks": len(weeks), "date_range": [weeks[0]["entry_date"], weeks[-1]["exit_date"]],
                          "by_cushion": {}}
        print(f"{'Cushion':>8} | {'Close-to-close':>16} | {'Intraweek (real)':>16}")
        for c in CUSHIONS_PCT:
            cc = containment_rate(weeks, c, use_intraweek=False)
            iw = containment_rate(weeks, c, use_intraweek=True)
            ticker_result["by_cushion"][str(c)] = {"close_to_close": cc, "intraweek": iw}
            print(f"{c:>7.1f}% | {cc['contained']:>5}/{cc['n']:<5}={cc['rate']:.1%} | {iw['contained']:>5}/{iw['n']:<5}={iw['rate']:.1%}")

        # Per-year breakdown at a representative cushion (2.5%) for regime honesty
        print(f"\n  Per-year containment at 2.5% cushion (intraweek, real):")
        years = sorted(set(w["year"] for w in weeks))
        by_year = {}
        for yr in years:
            yr_weeks = [w for w in weeks if w["year"] == yr]
            r = containment_rate(yr_weeks, 2.5, use_intraweek=True)
            by_year[yr] = r
            print(f"    {yr}: {r['contained']}/{r['n']} = {r['rate']:.1%}" if r['rate'] is not None else f"    {yr}: n=0")
        ticker_result["by_year_2.5pct_intraweek"] = by_year

        all_results[ticker] = ticker_result
        with open("evc_research/weekly_condor_weeks_" + ticker.lower() + ".json", "w") as f:
            json.dump(weeks, f, indent=2)

    with open("evc_research/weekly_condor_cushion_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved evc_research/weekly_condor_cushion_results.json")


if __name__ == "__main__":
    main()
