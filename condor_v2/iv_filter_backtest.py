#!/usr/bin/env python3
"""Condor V2 with an IV gate on every entry (user ask 2026-10-16): place a condor only when its entry
IV is inside a band. Uses iv_study.json (10 entries/day, 0.5x stop, net of measured costs).

In-sample bands were found on this same data, so they flatter themselves. The walk-forward section
re-picks the best 1.5-point band using ONLY earlier data and trades it on the next block.

    python iv_filter_backtest.py
"""
import json
from collections import defaultdict

import replay_backtest as R

T = json.loads((R.HERE / "iv_study.json").read_text())
DAYS = sorted({t["day"] for t in T})


def stats(rows, label):
    by_day = defaultdict(float)
    for t in rows:
        by_day[t["day"]] += t["net"]
    traded = [by_day[d] for d in DAYS if d in by_day]
    peak = cum = dd = 0.0
    for d in DAYS:
        cum += by_day.get(d, 0.0)
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    n = len(rows)
    tot = sum(t["net"] for t in rows)
    return (f"{label:<26}{n:>6}{len(traded):>6}{tot:>+9.0f}{tot / n if n else 0:>+8.2f}"
            f"{tot / len(DAYS):>+8.1f}{(sum(v > 0 for v in traded) / len(traded) if traded else 0):>7.0%}"
            f"{(min(traded) if traded else 0):>+8.0f}{dd:>+9.0f}{(sum(t['stopped'] for t in rows) / n if n else 0):>7.0%}")


HDR = (f"{'rule':<26}{'trades':>6}{'days':>6}{'net $':>9}{'$/trade':>8}{'$/day*':>8}{'days+':>7}"
       f"{'worst':>8}{'maxDD':>9}{'stops':>7}")


def gate(lo, hi):
    return [t for t in T if lo <= t["iv"] < hi]


def best_band(rows, width=0.015, min_n=60):
    """Best 1.5-point IV window by avg net/condor on `rows` (what a trader could have picked then)."""
    best = None
    for lo in [x / 1000 for x in range(40, 160, 5)]:
        w = [t for t in rows if lo <= t["iv"] < lo + width]
        if len(w) >= min_n:
            a = sum(t["net"] for t in w) / len(w)
            if best is None or a > best[0]:
                best = (a, lo)
    return best[1] if best else None


def main():
    global DAYS
    print(f"{len(T)} condors, {len(DAYS)} days ({DAYS[0]} .. {DAYS[-1]}), 0.5x stop, net of measured costs")
    print("* $/day is over ALL days in the data, so skipped days count as $0\n")
    print("IN-SAMPLE (band found on this same data -- optimistic)")
    print(HDR)
    print(stats(T, "no gate (current V2)"))
    print(stats(gate(0.072, 0.089), "IV 7.2-8.9% (the band)"))
    print(stats(gate(0.070, 0.090), "IV 7.0-9.0%"))
    print(stats(gate(0.065, 0.095), "IV 6.5-9.5% (wider)"))
    print(stats(gate(0.0, 0.115), "skip IV > 11.5% (H2)"))
    print(stats(gate(0.065, 0.115), "IV 6.5-11.5%"))

    print("\nWALK-FORWARD (pick the band from earlier data only, trade the next block)")
    n = len(DAYS)
    blocks = [DAYS[i * n // 6:(i + 1) * n // 6] for i in range(6)]
    oos = {"chosen band": [], "skip >11.5%": [], "no gate": []}
    print(f"{'trade block':<25}{'band picked from prior data':<30}{'band $/trade':>13}{'no-gate $/trade':>16}")
    for i in range(1, len(blocks)):
        prior = {d for b in blocks[:i] for d in b}
        block = set(blocks[i])
        lo = best_band([t for t in T if t["day"] in prior])
        rows = [t for t in T if t["day"] in block]
        g = [t for t in rows if lo is not None and lo <= t["iv"] < lo + 0.015]
        oos["chosen band"] += g
        oos["skip >11.5%"] += [t for t in rows if t["iv"] < 0.115]
        oos["no gate"] += rows
        f = lambda rs: f"{sum(t['net'] for t in rs) / len(rs):+.2f} (n={len(rs)})" if rs else "no trades"
        print(f"{min(block)}..{max(block)}  {lo:.1%}-{lo + 0.015:.1%}{'':<18}{f(g):>13}{f(rows):>16}")
    print("\nout-of-sample totals (blocks 2..6 only):")
    print(HDR)
    keep = DAYS
    DAYS = [d for b in blocks[1:] for d in b]
    for k, rows in oos.items():
        print(stats(rows, k))
    DAYS = keep


if __name__ == "__main__":
    main()
