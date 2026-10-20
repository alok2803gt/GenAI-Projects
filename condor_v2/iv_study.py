#!/usr/bin/env python3
"""Condor V2 P&L vs implied vol at entry (user ask 2026-10-16). Replays the 10-entry setup at the
0.5x stop over every cached day (replay_cache/day_*.json: the last ~12 months, every trading day)
plus the older OPEX days in opex_results.json, then buckets each condor by its entry IV.

    python iv_study.py
"""
import json
import math
from collections import defaultdict
from datetime import date, timedelta

import opex_backtest as O
import replay_backtest as R

R.bars = O._orig_bars  # cached data only -- no slow REST fetches for a whole year


def is_opex(day):
    d = date.fromisoformat(day)
    tf = O.third_friday(d.year, d.month)
    return d == tf or d == tf - timedelta(days=1) and len(R.bars("SPY", tf.isoformat())) < 300


def main():
    R.CFG = dict(R.C.DEFAULTS, entry_times=O.SLOTS10, stop_mult=0.5)
    trades = []
    for f in sorted(R.CACHE.glob("day_*.json")):
        day = f.stem[4:]
        for r in R.replay_day(day):
            if "pnl" in r:
                trades.append(dict(r, day=day, net=R.net(r), opex=is_opex(day)))
    seen = {t["day"] for t in trades}
    for day, rows in json.loads((R.HERE / "opex_results.json").read_text())["0.5x stop (current)"].items():
        if day not in seen:
            trades += [dict(r, day=day, net=R.net(r), opex=True) for r in rows if "pnl" in r]
    (R.HERE / "iv_study.json").write_text(json.dumps(trades))
    days = sorted({t["day"] for t in trades})
    print(f"{len(trades)} condors on {len(days)} days ({days[0]} .. {days[-1]})")

    def table(rows, title):
        ivs = sorted(t["iv"] for t in rows)
        cuts = [ivs[int(len(ivs) * q / 5)] for q in range(1, 5)]
        b = defaultdict(list)
        for t in rows:
            b[sum(t["iv"] >= c for c in cuts)].append(t)
        print(f"\n{title}  (IV quintiles)")
        print(f"{'IV band':<16}{'condors':>8}{'avg credit':>11}{'stop rate':>10}{'avg net':>9}{'total net':>10}{'win%':>6}")
        for q in range(5):
            rs = b[q]
            lo = min(t["iv"] for t in rs) * 100
            hi = max(t["iv"] for t in rs) * 100
            print(f"{lo:5.1f}-{hi:5.1f}%     {len(rs):>8}{sum(t['credit'] for t in rs) / len(rs) * 100:>10.0f}$"
                  f"{sum(t['stopped'] for t in rs) / len(rs):>10.0%}{sum(t['net'] for t in rs) / len(rs):>+9.1f}"
                  f"{sum(t['net'] for t in rs):>+10.0f}{sum(t['net'] > 0 for t in rs) / len(rs):>6.0%}")
        n = len(rows)
        x = [t["iv"] for t in rows]
        y = [t["net"] for t in rows]
        mx, my = sum(x) / n, sum(y) / n
        r = sum((a - mx) * (c - my) for a, c in zip(x, y)) / math.sqrt(
            sum((a - mx) ** 2 for a in x) * sum((c - my) ** 2 for c in y))
        print(f"correlation (Pearson) entry IV vs net P&L: {r:+.3f}  (n={n})")

    table(trades, "ALL DAYS")
    table([t for t in trades if t["opex"]], "OPEX DAYS ONLY")
    table([t for t in trades if not t["opex"]], "NON-OPEX DAYS")

    by_day = defaultdict(list)
    for t in trades:
        by_day[t["day"]].append(t)
    first = [(min(ts, key=lambda t: t["slot"])["iv"], sum(t["net"] for t in ts)) for ts in by_day.values()]
    first.sort()
    k = len(first) // 3
    print("\nper DAY, by the first entry's IV (what you know at ~10:01):")
    for name, part in (("low third", first[:k]), ("middle third", first[k:2 * k]), ("high third", first[2 * k:])):
        print(f"  {name:<13} IV {part[0][0] * 100:4.1f}-{part[-1][0] * 100:4.1f}%  days {len(part):>3}  "
              f"avg day net {sum(v for _, v in part) / len(part):+7.1f}  days up {sum(v > 0 for _, v in part) / len(part):.0%}")


if __name__ == "__main__":
    main()
