#!/usr/bin/env python3
"""Record one-minute bid/ask for SPY and its same-day options around spot from IBKR's live feed.

Why: replays price fills at one-minute TRADE prices because no quote history is available to us --
Massive's plan returns 403 for historical quotes, and IBKR serves no data for expired options. These
files build our own history from the feed the trader actually uses (user ask 2026-10-11).

Read-only connection on its own client id; never places orders. Streams SPY plus every strike within
BAND dollars of spot (both rights), re-centred each minute as SPY moves, capped at MAX_LINES streams so
the trader and monitor keep their market-data lines. Reconnects if TWS drops. Holds the laptop awake.
Task CondorV2-QuoteRecorder starts it weekdays 09:25; it records 09:30-16:00 and exits.

Output quotes/SPY_<date>.jsonl, one line per minute:
    {"t": "09:31", "spy": [bid, ask, last], "o": {"779C": [bid, ask], ...},
     "spx": last, "x": {"7800C": [bid, ask], ...}}     (SPX/SPXW from 2026-10-14)

    python quote_recorder.py           # the day's recording
    python quote_recorder.py --once    # connect, take one sample, print it (no file)
"""
import json
import sys
import time
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import IB, Index, Option, Stock

import condor_v2 as C

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
OUT = HERE / "quotes"
CLIENT_ID = 1590
BAND = 8              # SPY strikes within +/- $8 of spot
MAX_LINES = 36         # SPY option streams
SPX_BAND = 30          # SPXW strikes within +/- 30 points (5-point grid), added 2026-10-14
SPX_MAX = 28           # SPXW option streams; with SPY + SPX index ~66 lines, under IBKR's 100
START, END = dtime(9, 30), dtime(16, 0)


def now():
    return datetime.now(ET)


def wanted(spot, band=BAND, step=1):
    """(strike, right) keys within `band` of spot on a `step` strike grid."""
    k0 = step * round(spot / step)
    return {(float(k), r) for k in range(int(k0 - band), int(k0 + band) + 1, step) for r in ("C", "P")}


def trim(keys, spot, limit):
    """The `limit` keys nearest spot (SPY's own stream is counted by the caller)."""
    return set(sorted(keys, key=lambda kr: (abs(kr[0] - spot), kr[0], kr[1]))[:limit])


def px(v):
    return C.safe_px(v)


def label(key):
    return f"{key[0]:g}{key[1]}"


class Recorder:
    def __init__(self, expiry):
        self.ib, self.expiry, self.spy, self.opts = IB(), expiry, None, {}
        self.spx, self.xopts = None, {}

    def connect(self):
        self.ib.connect("127.0.0.1", 7497, clientId=CLIENT_ID, readonly=True, timeout=15)
        self.ib.reqMarketDataType(1)
        stock = Stock("SPY", "SMART", "USD")
        self.ib.qualifyContracts(stock)
        self.spy = self.ib.reqMktData(stock, "", False, False)
        self.opts = {}
        index = Index("SPX", "CBOE", "USD")
        self.ib.qualifyContracts(index)
        self.spx = self.ib.reqMktData(index, "", False, False)
        self.xopts = {}

    def spot(self):
        t = self.spy
        b, a = px(t.bid), px(t.ask)
        return (b + a) / 2 if b and a else px(t.last) or px(t.close)

    def spx_spot(self):
        return (px(self.spx.last) or px(self.spx.close)) if self.spx else None

    def _follow(self, store, keys, spot, limit, symbol, tclass):
        keep = trim(keys | set(store), spot, limit)
        for key in [k for k in store if k not in keep]:
            self.ib.cancelMktData(store.pop(key).contract)
        new = [k for k in keep if k not in store]
        if new:
            cons = [Option(symbol, self.expiry, k, r, "SMART", tradingClass=tclass) for k, r in new]
            self.ib.qualifyContracts(*cons)
            for key, c in zip(new, cons):
                if c.conId:
                    store[key] = self.ib.reqMktData(c, "", False, False)

    def follow(self, spot):
        """Stream the strikes around spot; drop the farthest when over the cap."""
        self._follow(self.opts, wanted(spot), spot, MAX_LINES, "SPY", "SPY")
        x = self.spx_spot()
        if x:
            self._follow(self.xopts, wanted(x, SPX_BAND, 5), x, SPX_MAX, "SPX", "SPXW")

    def sample(self, hhmm):
        s = self.spy
        return {"t": hhmm, "spy": [px(s.bid), px(s.ask), px(s.last)],
                "o": {label(k): [px(t.bid), px(t.ask)] for k, t in sorted(self.opts.items())},
                "spx": self.spx_spot(),
                "x": {label(k): [px(t.bid), px(t.ask)] for k, t in sorted(self.xopts.items())}}


def main():
    once = "--once" in sys.argv
    n = now()
    if n.weekday() >= 5 and not once:
        print("weekend -- nothing to record")
        return
    day, expiry = n.date().isoformat(), n.strftime("%Y%m%d")
    OUT.mkdir(exist_ok=True)
    path = OUT / f"SPY_{day}.jsonl"
    rec = Recorder(expiry)
    C.keep_awake(True)
    written = 0
    done = set()  # minutes already written (a restart mid-day appends to the same file)
    if path.exists():
        done = {json.loads(l)["t"] for l in path.read_text().splitlines() if l.strip()}
    try:
        while once or now().time() <= END:
            if not rec.ib.isConnected():
                try:
                    rec.connect()
                    print(f"{now():%H:%M:%S} connected", flush=True)
                except Exception as e:
                    print(f"{now():%H:%M:%S} TWS not reachable ({type(e).__name__}) -- retrying in 60s", flush=True)
                    rec.ib.disconnect()
                    time.sleep(60)
                    continue
            try:
                spot = None
                for _ in range(10):  # first quotes take a moment after (re)connecting
                    spot = rec.spot()
                    if spot:
                        break
                    rec.ib.sleep(1)
                if spot:
                    rec.follow(spot)
                if once:
                    rec.ib.sleep(5)
                    line = rec.sample(f"{now():%H:%M}")
                    print(json.dumps(line)[:600])
                    print(f"spot {spot}, {len(rec.opts)} option streams")
                    return
                # Sample just after the next minute boundary and label it with that boundary.
                # Labelling by wall clock wrote some minutes twice on 2026-10-12: Windows timers
                # can wake a few ms early, before the boundary.
                target = now().replace(second=0, microsecond=0) + timedelta(minutes=1)
                rec.ib.sleep(max((target - now()).total_seconds() + 0.3, 0.3))
                minute = f"{target:%H:%M}"
                if START <= target.time() <= END and minute not in done:
                    with open(path, "a") as f:
                        f.write(json.dumps(rec.sample(minute)) + "\n")
                    done.add(minute)
                    written += 1
            except Exception as e:  # a dropped TWS or a bad contract must not end the day's recording
                print(f"{now():%H:%M:%S} error: {type(e).__name__} {e} -- reconnecting", flush=True)
                rec.ib.disconnect()
                time.sleep(5)
    finally:
        C.keep_awake(False)
        rec.ib.disconnect()
        print(f"{now():%H:%M:%S} done -- {written} samples in {path.name}", flush=True)


if __name__ == "__main__":
    main()
