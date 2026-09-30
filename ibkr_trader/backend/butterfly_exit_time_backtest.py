"""
Does moving the butterfly hard close from 15:45 to 15:55 pay?

The 2026-09-23 paired test on the 65-trade UW sample said holding to the 16:00
close beats 15:45 by +$20/trade (t=2.51) -- but expiry is unusable here (a
winning fly finishes ITM and the body is SHORT 2 contracts, i.e. ~200 shares
of assignment this account cannot carry). 15:55 is the compromise: most of the
last half hour's decay, still flat before the bell.

Same real-data methodology as butterfly_entry_time_backtest.py:
  entry     09:45 ET, wing_step=4, near-S/R gate (prior day's high/low within
            0.3%), whole-dollar strikes around the 09:45 spot
  prices    Polygon 1-minute OPTION bars (UW's tape 401s since 2026-09-23),
            so the window is ~2 years rather than UW's ~90 days
  exits     15:45 / 15:50 / 15:55 / 15:59 marks, plus close-of-day intrinsic
            as the (unreachable) upper bound
Every exit is priced from the SAME trade, so the comparison is perfectly
paired -- no day-mix artefact like the entry-time study had.

Output: butterfly_exit_time_rows.csv
"""
import io
import sys
from datetime import date, datetime, timedelta, timezone

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import math
import pandas as pd

import butterfly_0dte_backtest_v2 as v2
import butterfly_entry_time_backtest as et

WING_STEP = 4
ENTRY = (9, 45)
EXIT_TIMES = {"15:45": (15, 45), "15:50": (15, 50), "15:55": (15, 55), "15:59": (15, 59)}
NEAR_SR_PCT = 0.003
LOOKBACK_DAYS = 730


def option_price_at(occ, d, hh, mm, tol_min=3):
    """Close of the 1-min bar at hh:mm (or the nearest within tol_min)."""
    key = (occ, d)
    if key not in et._OPT_BAR_CACHE:
        data = v2._polygon_get(f"/v2/aggs/ticker/{occ}/range/1/minute/{d.isoformat()}/{d.isoformat()}",
                               {"adjusted": "true", "sort": "asc", "limit": 50000})
        et._OPT_BAR_CACHE[key] = (data or {}).get("results") or []
    bars = et._OPT_BAR_CACHE[key]
    if not bars:
        return None
    target = (datetime(d.year, d.month, d.day, hh, mm, tzinfo=timezone.utc) + timedelta(hours=4)).timestamp() * 1000
    best, best_diff = None, None
    for b in bars:
        diff = abs(b["t"] - target) / 1000.0
        if diff > tol_min * 60:
            continue
        if best_diff is None or diff < best_diff:
            best, best_diff = b, diff
    return float(best["c"]) if best else None


def main():
    earliest = date.today() - timedelta(days=LOOKBACK_DAYS)
    rows = []
    for ticker in v2.ETF_UNIVERSE:
        print(f"\n=== {ticker} ===", flush=True)
        hist = et.ibkr_daily_history(ticker, days=LOOKBACK_DAYS + 30)
        hist.index = pd.to_datetime(hist.index).normalize()
        close_map = dict(zip(hist.index.date, hist["Close"].values))
        days = [d for d in hist.index.date if earliest <= d < date.today()]
        for i, d in enumerate(days, 1):
            if i % 40 == 0:
                print(f"  [{i}/{len(days)}] {d}  rows={len(rows)}", flush=True)
            prior = hist[hist.index.date < d]
            if prior.empty:
                continue
            prior_high, prior_low = float(prior.iloc[-1]["High"]), float(prior.iloc[-1]["Low"])
            spot = et.get_morning_underlying_price_at(ticker, d, *ENTRY)
            if spot is None:
                continue
            if not (abs(prior_high - spot) / spot < NEAR_SR_PCT or abs(spot - prior_low) / spot < NEAR_SR_PCT):
                continue                                   # near-S/R gate, same as live
            grid = et.contract_grid_polygon(ticker, d, spot)
            ks = sorted(grid)
            k2 = min(ks, key=lambda s: abs(s - spot))
            i2 = ks.index(k2)
            if i2 - WING_STEP < 0 or i2 + WING_STEP >= len(ks):
                continue
            k1, k3 = ks[i2 - WING_STEP], ks[i2 + WING_STEP]
            legs = [grid[k1], grid[k2], grid[k3]]
            entry = [et.get_morning_option_price_at(c, d, *ENTRY) for c in legs]
            if any(p is None for p in entry):
                continue
            debit = entry[0] + entry[2] - 2 * entry[1]
            if debit <= 0.005:
                continue
            row = dict(ticker=ticker, entry_date=d, spot=spot, k1=k1, k2=k2, k3=k3, debit=debit)
            for label, (hh, mm) in EXIT_TIMES.items():
                px = [option_price_at(c, d, hh, mm) for c in legs]
                row[f"exit_{label}"] = ((px[0] + px[2] - 2 * px[1]) - debit) * 100 if all(p is not None for p in px) else None
            cs = float(close_map[d])
            intr = max(0.0, cs - k1) - 2 * max(0.0, cs - k2) + max(0.0, cs - k3)
            row["exit_expiry"] = (intr - debit) * 100
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("butterfly_exit_time_rows.csv", index=False)
    print(f"\nSaved {len(df)} near-S/R trades ({df.entry_date.min()} -> {df.entry_date.max()})")

    cols = [f"exit_{k}" for k in EXIT_TIMES] + ["exit_expiry"]
    print(f"\n{'exit':<12}{'n':>5}{'mean':>9}{'median':>9}{'win':>8}{'worst':>9}")
    for c in cols:
        x = df[c].dropna()
        if len(x) == 0:
            print(f"{c[5:]:<12}{0:>5}   no data"); continue
        print(f"{c[5:]:<12}{len(x):>5}{x.mean():>9.2f}{x.median():>9.2f}{(x > 0).mean():>8.1%}{x.min():>9.0f}")

    print("\npaired vs 15:45 (same trades):")
    for c in cols[1:]:
        p = df[["exit_15:45", c]].dropna()
        diff = p[c] - p["exit_15:45"]
        t = diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 2 else float("nan")
        print(f"  {c[5:]:<10} n={len(diff):>4} mean_diff=${diff.mean():+7.2f}  t={t:+5.2f}"
              f"{'  <- significant' if abs(t) >= 2 else ''}")

    print("\nper ticker, 15:55 minus 15:45:")
    for tk in v2.ETF_UNIVERSE:
        g = df[df.ticker == tk][["exit_15:45", "exit_15:55"]].dropna()
        if len(g) < 3:
            print(f"  {tk}: n={len(g)} too few"); continue
        diff = g["exit_15:55"] - g["exit_15:45"]
        t = diff.mean() / (diff.std(ddof=1) / math.sqrt(len(diff)))
        print(f"  {tk}: n={len(diff)} mean_diff=${diff.mean():+7.2f} t={t:+5.2f}")


if __name__ == "__main__":
    main()
