#!/usr/bin/env python3
"""SPX 09:30-11:00 move/range per 15-minute slot, VIX expirations vs other days (user ask 2026-10-21).
Every day is rescaled to TODAY's terms -- points at SPX_NOW and VIX_NOW -- so 5 years of different
price levels and vol regimes read as "SPX points at today's level and volatility":
    pts = pct_move / 100 * SPX_NOW * (VIX_NOW / that day's prior VIX close)

    python vixpiration_15m.py [SPX_NOW VIX_NOW]
"""
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

from ib_insync import IB, Index

import vixpiration_study as V

HERE = Path(__file__).resolve().parent
CACHE = HERE / "replay_cache" / "spx_15m_5y.json"
SLOTS = ["09:30", "09:45", "10:00", "10:15", "10:30", "10:45"]  # slot starts; window ends 11:00
END = {"09:30": "09:45", "09:45": "10:00", "10:00": "10:15", "10:15": "10:30", "10:30": "10:45", "10:45": "11:00"}


async def fetch():
    ib = IB()
    await ib.connectAsync("127.0.0.1", 7497, clientId=1657, timeout=15, readonly=True)
    spx = Index("SPX", "CBOE")
    await ib.qualifyContractsAsync(spx)
    out, end = {}, ""
    for _ in range(60):  # 5 years in ~1-month requests
        bs = await ib.reqHistoricalDataAsync(spx, end, "1 M", "15 mins", "TRADES", useRTH=True, formatDate=2)
        if not bs:
            break
        for b in bs:
            out[b.date.astimezone().strftime("%Y-%m-%d %H:%M")] = [b.open, b.high, b.low, b.close]
        end = bs[0].date.strftime("%Y%m%d-%H:%M:%S")
        if bs[0].date.year < 2021 or (bs[0].date.year == 2021 and bs[0].date.month < 10):
            break
        await asyncio.sleep(1)
    ib.disconnect()
    CACHE.write_text(json.dumps(out))


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    spx_now = float(sys.argv[1]) if len(sys.argv) > 2 else 7724.0
    vix_now = float(sys.argv[2]) if len(sys.argv) > 2 else 17.4
    if not CACHE.exists():
        asyncio.run(fetch())
    bars = json.loads(CACHE.read_text())
    vix = json.loads(V.CACHE.read_text())["vix"]
    vdays = sorted(vix)
    by_day = defaultdict(dict)
    for k, v in bars.items():
        d, hm = k.split(" ")
        by_day[d][hm] = v
    days = {d: b for d, b in by_day.items() if all(s in b for s in SLOTS)}
    exp = V.vix_expiries(set(days))
    groups = {"VIX expiration": [], "other days": []}
    for d, b in sorted(days.items()):
        prev = [x for x in vdays if x < d]
        if not prev or d < "2021-10-20":
            continue
        k = spx_now / 100 * vix_now / vix[prev[-1]]  # pct -> today's points
        o = b["09:30"][0]
        row, hi, lo = {}, o, o
        for s in SLOTS:
            bo, bh, bl, bc = b[s]
            hi, lo = max(hi, bh), min(lo, bl)
            row[s] = {"slot_rng": (bh - bl) / o * 100 * k, "slot_move": (bc - bo) / o * 100 * k,
                      "cum_rng": (hi - lo) / o * 100 * k, "cum_move": (bc - o) / o * 100 * k,
                      "cum_up_ext": (hi - o) / o * 100 * k, "cum_dn_ext": (o - lo) / o * 100 * k}
        groups["VIX expiration" if d in exp else "other days"].append(row)

    print(f"Scaled to today: SPX {spx_now:,.0f}, VIX {vix_now}.  Points = SPX index points.\n")
    for name, rows in groups.items():
        print(f"=== {name} ({len(rows)} days) ===")
        print(f"{'slot':<13}{'slot range':>11}{'|slot move|':>12}  {'range since 09:30 (median / 75th / 90th)':>42}"
              f"{'max up from open (med/90th)':>30}{'max down (med/90th)':>22}{'net since open (med)':>22}")
        for s in SLOTS:
            sr = [r[s]["slot_rng"] for r in rows]
            sm = [abs(r[s]["slot_move"]) for r in rows]
            cr = [r[s]["cum_rng"] for r in rows]
            up = [r[s]["cum_up_ext"] for r in rows]
            dn = [r[s]["cum_dn_ext"] for r in rows]
            cm = [r[s]["cum_move"] for r in rows]
            print(f"{s}-{END[s]}  {sum(sr) / len(sr):>8.1f}  {sum(sm) / len(sm):>10.1f}"
                  f"  {pct(cr, .5):>20.1f} / {pct(cr, .75):>5.1f} / {pct(cr, .9):>5.1f}"
                  f"  {pct(up, .5):>18.1f} / {pct(up, .9):>5.1f}  {pct(dn, .5):>12.1f} / {pct(dn, .9):>5.1f}"
                  f"  {pct(cm, .5):>+16.1f}")
        print()


if __name__ == "__main__":
    main()
