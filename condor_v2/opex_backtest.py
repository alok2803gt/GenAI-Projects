#!/usr/bin/env python3
"""Condor V2 on monthly-OPEX days only (user ask 2026-10-16): the full 10-entry setup (10:01 .. 14:31)
replayed with replay_backtest.py's engine on every third Friday our data covers, under three stops.

Data: Massive/Polygon one-minute bars. The plan keeps a rolling 2 years (2024-10-17 is the oldest day
it serves; 2024-09-20 and older return 403), so "5 years" isn't reachable -- this covers
2024-10 .. 2026-09. When a third Friday was a market holiday the Thursday before is used.

    python opex_backtest.py            # fetch what's missing (slow: ~5 requests/min), replay, report
"""
import json
import sys
from datetime import date, timedelta

import replay_backtest as R

SLOTS10 = ["10:01", "10:31", "11:01", "11:31", "12:01", "12:31", "13:01", "13:31", "14:01", "14:31"]
VARIANTS = {"0.5x stop (current)": 0.5, "1x stop": 1.0, "no stop": 1e9}
FIRST, LAST = date(2024, 10, 1), date(2026, 9, 30)

_orig_bars = R.bars


def bars(ticker, day):
    """Cached day files hold only the strikes the 12-month study needed; fetch any other ticker
    rather than letting the replay fall back to Black-Scholes for it."""
    f = R.CACHE / f"day_{day}.json"
    if f.exists():
        got = _orig_bars(ticker, day)
        if got:
            return got
        f.rename(f.with_suffix(".json.hold"))  # route this one call to the REST path
        try:
            return _orig_bars(ticker, day)
        finally:
            f.with_suffix(".json.hold").rename(f)
    return _orig_bars(ticker, day)


R.bars = bars


def third_friday(y, m):
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def opex_days():
    y, m = FIRST.year, FIRST.month
    while date(y, m, 1) <= LAST:
        yield third_friday(y, m)
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def main():
    results = {v: {} for v in VARIANTS}
    for fri in opex_days():
        day = fri.isoformat()
        R.CFG = dict(R.C.DEFAULTS, entry_times=SLOTS10)
        if len(R.bars("SPY", day)) < 300:  # market holiday (e.g. Good Friday): OPEX moved to Thursday
            day = (fri - timedelta(days=1)).isoformat()
        for name, mult in VARIANTS.items():
            R.CFG = dict(R.C.DEFAULTS, entry_times=SLOTS10, stop_mult=mult)
            rows = R.replay_day(day)
            if rows:
                results[name][day] = rows
        n = sum(1 for r in results["0.5x stop (current)"].get(day, []) if "pnl" in r)
        print(f"{day}: {n} condors replayed", flush=True)
    (R.HERE / "opex_results.json").write_text(json.dumps(results, indent=1))
    for name in VARIANTS:
        print(f"\n================ {name} ================")
        R.summary(results[name])


if __name__ == "__main__":
    main()
