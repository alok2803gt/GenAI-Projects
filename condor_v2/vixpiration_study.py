#!/usr/bin/env python3
"""Does SPX move more (or up more) from 09:30 to 11:00 on VIX expiration days? (user thesis 2026-10-21)

Data: IBKR SPX index 30-minute bars (RTH) and VIX daily bars, 5 years, via the read-only paper TWS.
VIX monthly expiration: the Wednesday 30 days before the 3rd Friday of the following month (if that
Friday isn't a trading day, 30 days before the trading day before it; if the Wednesday isn't one, the
trading day before). Compared against every other day and against other Wednesdays.

Morning window = 09:30 open -> 11:00 (close of the 10:30 bar); range = high - low of the three bars.
Because vol regimes differ, every move is also scaled by its VIX-implied 1.5-hour sigma:
    sigma_pts = SPX * VIX/100 * sqrt(1.5 / (6.5 * 252))

    python vixpiration_study.py
"""
import asyncio
import json
import math
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from ib_insync import IB, Index

HERE = Path(__file__).resolve().parent
CACHE = HERE / "replay_cache" / "spx_vix_5y.json"
YEARS = 5


async def fetch():
    ib = IB()
    await ib.connectAsync("127.0.0.1", 7497, clientId=1656, timeout=15, readonly=True)
    spx, vix = Index("SPX", "CBOE"), Index("VIX", "CBOE")
    await ib.qualifyContractsAsync(spx, vix)
    bars30 = {}
    end = ""
    for _ in range(YEARS):
        bs = await ib.reqHistoricalDataAsync(spx, end, "1 Y", "30 mins", "TRADES", useRTH=True, formatDate=2)
        for b in bs:
            t = b.date.astimezone()
            bars30[t.strftime("%Y-%m-%d %H:%M")] = [b.open, b.high, b.low, b.close]
        end = bs[0].date.strftime("%Y%m%d-%H:%M:%S")
        await asyncio.sleep(2)
    vd = await ib.reqHistoricalDataAsync(vix, "", f"{YEARS + 1} Y", "1 day", "TRADES", useRTH=True)
    vixd = {b.date.isoformat(): b.close for b in vd}
    ib.disconnect()
    CACHE.write_text(json.dumps({"spx30": bars30, "vix": vixd}))


def third_friday(y, m):
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def vix_expiries(trading):
    last = max(trading)
    out = set()
    for y in range(2019, 2028):
        for m in range(1, 13):
            ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
            f = third_friday(ny, nm)
            # A Friday is a holiday only if it falls inside the data and is missing from it; a Friday
            # after the data's end (e.g. Nov 20 2026 for today's expiry) is assumed open.
            while f.isoformat() <= last and f.isoformat() not in trading and (third_friday(ny, nm) - f).days < 4:
                f -= timedelta(days=1)
            w = f - timedelta(days=30)
            while w.isoformat() not in trading and (f - timedelta(days=30) - w).days < 5:
                w -= timedelta(days=1)
            out.add(w.isoformat())
    return {d for d in out if d in trading}


def stats(rows, label):
    n = len(rows)
    if not n:
        return f"{label:<28} n=0"
    mean = lambda k: sum(r[k] for r in rows) / n
    up = sum(r["ret"] > 0 for r in rows) / n
    big_up = sum(r["ret_z"] >= 0.63 for r in rows) / n  # ~ +26 pts at today's vol: the user's 7750 trigger
    return (f"{label:<28} n={n:<5} |move| {mean('absret'):.2f}%  range {mean('rng'):.2f}%  "
            f"|move|/sigma {mean('abs_z'):.2f}  range/sigma {mean('rng_z'):.2f}  "
            f"avg move {mean('ret'):+.3f}%  up {up:.0%}  up>=0.63sigma {big_up:.0%}")


def perm_p(a, b, key, n=10000, seed=1):
    """One-sided permutation p-value that group a's mean `key` exceeds group b's."""
    random.seed(seed)
    va, vb = [r[key] for r in a], [r[key] for r in b]
    obs = sum(va) / len(va) - sum(vb) / len(vb)
    pool, k, hits = va + vb, len(va), 0
    for _ in range(n):
        random.shuffle(pool)
        if sum(pool[:k]) / k - sum(pool[k:]) / (len(pool) - k) >= obs:
            hits += 1
    return obs, hits / n


def main():
    if not CACHE.exists():
        asyncio.run(fetch())
    d = json.loads(CACHE.read_text())
    by_day = defaultdict(dict)
    for k, v in d["spx30"].items():
        day, hm = k.split(" ")
        by_day[day][hm] = v
    vix = d["vix"]
    vdays = sorted(vix)
    rows = []
    for day, bars in sorted(by_day.items()):
        need = ["09:30", "10:00", "10:30"]
        if not all(h in bars for h in need):
            continue
        prev = [v for v in vdays if v < day]
        if not prev:
            continue
        vx = vix[prev[-1]]  # prior close: known before the open
        o, c = bars["09:30"][0], bars["10:30"][3]
        hi = max(bars[h][1] for h in need)
        lo = min(bars[h][2] for h in need)
        sig = o * vx / 100 * math.sqrt(1.5 / (6.5 * 252))
        ret = (c / o - 1) * 100
        rows.append({"day": day, "wd": date.fromisoformat(day).weekday(), "ret": ret, "absret": abs(ret),
                     "rng": (hi - lo) / o * 100, "ret_z": (c - o) / sig, "abs_z": abs(c - o) / sig,
                     "rng_z": (hi - lo) / sig, "vix": vx})
    trading = {r["day"] for r in rows}
    exp = vix_expiries(trading)
    E = [r for r in rows if r["day"] in exp]
    O = [r for r in rows if r["day"] not in exp]
    W = [r for r in O if r["wd"] == 2]
    print(f"{rows[0]['day']} .. {rows[-1]['day']}: {len(rows)} days, {len(E)} VIX expirations\n")
    print(stats(E, "VIX expiration days"))
    print(stats(O, "all other days"))
    print(stats(W, "other Wednesdays"))
    print()
    for key, name in (("abs_z", "bigger |move| (vol-scaled)"), ("rng_z", "bigger range (vol-scaled)"),
                      ("ret_z", "more UP (vol-scaled)")):
        obs, p = perm_p(E, O, key)
        print(f"VIX-exp vs other days, {name:<28}: difference {obs:+.3f}  permutation p = {p:.3f}")
    print("\nEvery VIX expiration (09:30 -> 11:00):")
    for r in E:
        print(f"  {r['day']}  VIX {r['vix']:5.1f}  move {r['ret']:+.2f}% ({r['ret_z']:+.2f} sigma)  range {r['rng']:.2f}%")


if __name__ == "__main__":
    main()
