#!/usr/bin/env python3
"""Pressure test of iv_study.py's finding that condors entered at 7.2-8.9% IV carry nearly all the
profit (user ask 2026-10-16). Works on iv_study.json, re-replaying only for the stop-rule check.

    python iv_pressure.py
"""
import json
import random
from collections import Counter, defaultdict

import opex_backtest as O
import replay_backtest as R

LO, HI = 0.072, 0.089
T = json.loads((R.HERE / "iv_study.json").read_text())
random.seed(7)


def band(t, lo=LO, hi=HI):
    return lo <= t["iv"] < hi


def tot(rows, k="net"):
    return sum(r[k] for r in rows)


def avg(rows, k="net"):
    return tot(rows, k) / len(rows) if rows else float("nan")


def main():
    B = [t for t in T if band(t)]
    rest = [t for t in T if not band(t)]
    print(f"band {LO:.1%}-{HI:.1%}: {len(B)} condors, total {tot(B):+.0f}, avg {avg(B):+.2f} | "
          f"rest: {len(rest)} condors, avg {avg(rest):+.2f}")

    print("\n1) CONCENTRATION -- band profit by day, biggest days removed")
    by_day = defaultdict(float)
    for t in B:
        by_day[t["day"]] += t["net"]
    days = sorted(by_day.values(), reverse=True)
    print(f"   {len(days)} days carry band trades; best day {days[0]:+.0f}, worst {days[-1]:+.0f}")
    for k in (1, 3, 5, 10):
        print(f"   minus best {k:>2} day(s): total {sum(days[k:]):+.0f}, avg/condor "
              f"{sum(days[k:]) / max(1, len(B) - sum(1 for t in B if by_day[t['day']] in days[:k])):+.2f}")
    print(f"   days with band net > 0: {sum(v > 0 for v in days) / len(days):.0%}")

    print("\n2) TIME STABILITY -- each quarter of the data, band vs the rest")
    alld = sorted({t["day"] for t in T})
    q = len(alld) // 4
    for i in range(4):
        span = set(alld[i * q:(i + 1) * q] if i < 3 else alld[3 * q:])
        b = [t for t in B if t["day"] in span]
        r = [t for t in rest if t["day"] in span]
        lab = f"{min(span)}..{max(span)}"
        print(f"   {lab}: band {len(b):>3} condors avg {avg(b):+6.2f} | rest {len(r):>4} avg {avg(r):+6.2f}")

    print("\n3) EDGE SENSITIVITY -- avg net/condor for neighbouring 1.5-point windows")
    for lo in [x / 1000 for x in range(50, 125, 5)]:
        w = [t for t in T if band(t, lo, lo + 0.015)]
        if len(w) > 60:
            print(f"   {lo:5.1%}-{lo + 0.015:5.1%}: {len(w):>4} condors  avg {avg(w):+6.2f}  stop {avg(w, 'stopped'):.0%}")

    print("\n4) PERMUTATION -- shuffle IV across DAYS (each day keeps its own trades), 2,000 times:")
    print("   how often does the best of 5 IV quintiles look as good as the real band?")
    day_iv = defaultdict(list)
    for t in T:
        day_iv[t["day"]].append(t)
    real_best = tot(B)
    dlist = list(day_iv)
    hits = 0
    for _ in range(2000):
        perm = dlist[:]
        random.shuffle(perm)
        mapping = dict(zip(dlist, perm))  # day d's trades get day mapping[d]'s IVs, slot by slot
        fake = []
        for d in dlist:
            ivs = sorted(t["iv"] for t in day_iv[mapping[d]])
            for t, v in zip(sorted(day_iv[d], key=lambda t: t["slot"]), ivs + ivs[:10]):
                fake.append((v, t["net"]))
        fake.sort()
        n = len(fake) // 5
        best = max(sum(v for _, v in fake[i * n:(i + 1) * n]) for i in range(5))
        hits += best >= real_best
    print(f"   p = {hits / 2000:.3f}  (share of shuffles whose best quintile >= {real_best:+.0f})")

    print("\n5) BOOTSTRAP -- resample days with replacement, 5,000 times: band avg net/condor")
    bd = defaultdict(list)
    for t in B:
        bd[t["day"]].append(t["net"])
    keys = list(bd)
    sims = []
    for _ in range(5000):
        s = [x for k in random.choices(keys, k=len(keys)) for x in bd[k]]
        sims.append(sum(s) / len(s))
    sims.sort()
    print(f"   95% interval {sims[125]:+.2f} .. {sims[4875]:+.2f}; P(avg <= 0) = {sum(x <= 0 for x in sims) / 5000:.3f}")

    print("\n6) COST STRESS -- extra cost per condor on top of the replay's measured costs")
    for extra in (0, 3, 5, 8):
        print(f"   +${extra}: band avg {avg(B) - extra:+.2f}, total {tot(B) - extra * len(B):+.0f} | "
              f"rest avg {avg(rest) - extra:+.2f}")

    print("\n7) CONFOUNDS -- is the band really time of day, or one stretch of the market?")
    sb, sr = Counter(t["slot"] for t in B), Counter(t["slot"] for t in rest)
    print("   slot   share of band  share of rest  band avg net in that slot")
    for s in O.SLOTS10:
        bs = [t for t in B if t["slot"] == s]
        print(f"   {s}  {sb[s] / len(B):>12.0%}  {sr[s] / len(rest):>13.0%}  {avg(bs):>+10.2f}")
    mb = Counter(t["day"][:7] for t in B)
    print("   band condors per month:", " ".join(f"{m[2:]}:{n}" for m, n in sorted(mb.items())))
    by_m = defaultdict(list)
    for t in B:
        by_m[t["day"][:7]].append(t["net"])
    pos = sum(sum(v) > 0 for v in by_m.values())
    print(f"   months where the band made money: {pos}/{len(by_m)}")

    print("\n8) STOP RULE -- does the band still lead under 1x and no stop? (re-replay)")
    for name, mult in (("1x stop", 1.0), ("no stop", 1e9)):
        R.CFG = dict(R.C.DEFAULTS, entry_times=O.SLOTS10, stop_mult=mult)
        R.bars = O._orig_bars
        rows = []
        for d in sorted({t["day"] for t in T}):
            rows += [dict(r, net=R.net(r)) for r in R.replay_day(d) if "pnl" in r]
        b = [t for t in rows if band(t)]
        r = [t for t in rows if not band(t)]
        print(f"   {name:<8}: band avg {avg(b):+6.2f} (n={len(b)}), rest avg {avg(r):+6.2f} (n={len(r)})")


if __name__ == "__main__":
    main()
