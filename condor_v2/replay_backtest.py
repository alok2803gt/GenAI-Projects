#!/usr/bin/env python3
"""Replay condor_v2's rules on historical minute data; compare with the trader's actual fills, or
summarize a long stretch with realistic costs.

Each entry slot is rebuilt from SPY and 0DTE option one-minute bars using condor_v2's own math
(ATM implied vol -> u -> shorts at +/-0.5u rounded, $5 wings, $0.10 credit floor). Each condor is
marked every minute until a 0.5x-credit stop or 15:58, where shorts in the money or within $0.25
are closed; anything left settles at intrinsic on the 15:59 close.

Bars are TRADE prices, not quotes (the data plan has no bid/ask), so the replay prices the rules at
a fair mid-ish level. Costs measured on the 2026-10-05..09 paper fills are added in --summary:
entry ~$1 below the replay credit, ~$4 more per stop exit, commissions by IBKR's per-contract
schedule (about $3.30 a condor) or a flat $5 a condor (what the paper account actually showed).

Data: per-day files replay_cache/day_<date>.json (built from the 2026-09-27 study's 12 months of
bars), else Massive/Polygon REST, cached per ticker. An option with no trade yet is priced with
Black-Scholes at the entry's implied vol.

    python replay_backtest.py 2026-10-05 2026-10-09            # replay + compare with actual fills
    python replay_backtest.py 2025-09-29 2026-10-09 --summary  # long replay with costs
"""
import json
import math
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

import condor_v2 as C

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
CACHE = HERE / "replay_cache"
KEY = json.loads((HERE.parent / "ibkr_trader" / "backend" / "scanner_config.json").read_text())["polygon_api_key"]
CFG = C.DEFAULTS
ENTRY_SLIP, STOP_SLIP, FLAT_FEE = 0.01, 0.04, 5.0  # per condor, measured 2026-10-05..09
_day_files = {}


def per_contract_fee(px):
    """IBKR fixed-rate option commission by premium, plus ~$0.03 of exchange/regulatory fees."""
    return (0.65 if px >= 0.10 else 0.50 if px >= 0.05 else 0.25) + 0.03


def bars(ticker, day):
    """{'HH:MM': close} of regular-session one-minute bars."""
    f = CACHE / f"day_{day}.json"
    if f.exists():
        if day not in _day_files:
            _day_files.clear()
            _day_files[day] = json.loads(f.read_text())
        return _day_files[day].get(ticker, {})
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{ticker.replace(':', '_')}_{day}.json"
    if f.exists():
        return json.loads(f.read_text())
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/{day}/{day}"
    r = None
    for attempt in range(6):
        r = requests.get(url, params={"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": KEY}, timeout=30)
        if r.status_code == 429:
            time.sleep(15)
            continue
        r.raise_for_status()
        break
    out = {}
    for b in r.json().get("results") or []:
        t = datetime.fromtimestamp(b["t"] / 1000, ET)
        if (9, 30) <= (t.hour, t.minute) <= (15, 59):
            out[f"{t:%H:%M}"] = b["c"]
    f.write_text(json.dumps(out))
    return out


def occ(day, strike, right):
    return f"O:SPY{day.replace('-', '')[2:]}{right}{int(round(strike * 1000)):08d}"


def minutes(day):
    t, end = datetime.fromisoformat(f"{day} 09:30"), datetime.fromisoformat(f"{day} 15:59")
    while t <= end:
        yield f"{t:%H:%M}"
        t += timedelta(minutes=1)


def mins_to_close(hhmm):
    h, m = map(int, hhmm.split(":"))
    return max(16 * 60 - (h * 60 + m), 1)


class Leg:
    """Forward-filled trade price of one option, with a Black-Scholes fallback before its first trade."""

    def __init__(self, day, strike, right, spy):
        self.k, self.right, self.spy = strike, right, spy
        raw, last, self.ff = bars(occ(day, strike, right), day), None, {}
        for m in minutes(day):
            last = raw.get(m, last)
            self.ff[m] = last

    def price(self, hhmm, iv):
        p = self.ff.get(hhmm)
        if p is not None:
            return p
        return C.bs_price(self.spy[hhmm], self.k, mins_to_close(hhmm), iv, self.right)


def replay_day(day):
    raw = bars("SPY", day)
    if len(raw) < 300:
        return []
    spy, last = {}, None
    for m in minutes(day):
        last = raw.get(m, last)
        spy[m] = last
    legs_cache = {}

    def leg(k, r):
        if (k, r) not in legs_cache:
            legs_cache[(k, r)] = Leg(day, k, r, spy)
        return legs_cache[(k, r)]

    out = []
    for slot in CFG["entry_times"]:
        S = spy.get(slot)
        if not S:
            continue
        mins, k = mins_to_close(slot), float(round(S))
        ivs = []
        for r in ("C", "P"):
            px = leg(k, r).ff.get(slot)
            if px:
                v = C.implied_vol(px, S, k, mins, r)
                if 0.01 < v < 3.0:
                    ivs.append(v)
        if not ivs:
            out.append({"slot": slot, "skip": "no ATM price"})
            continue
        iv = sum(ivs) / len(ivs)
        u = S * iv * math.sqrt(mins / C.MIN_PER_YEAR)
        st = {"short_put": float(round(S - CFG["short_u"] * u)), "short_call": float(round(S + CFG["short_u"] * u))}
        st["long_put"], st["long_call"] = st["short_put"] - CFG["wing"], st["short_call"] + CFG["wing"]
        L = {n: leg(st[n], "P" if "put" in n else "C") for n in st}

        def value(m):
            return sum(L[n].price(m, iv) * (1 if n.startswith("short") else -1) for n in st)

        credit = value(slot)
        if credit < CFG["min_credit"]:
            out.append({"slot": slot, "skip": f"credit {credit:.2f} < floor", "strikes": st})
            continue
        fees = sum(per_contract_fee(L[n].price(slot, iv)) for n in st)
        res = {"slot": slot, "spot": S, "iv": round(iv, 4), "u": round(u, 2), "strikes": st,
               "credit": round(credit, 2)}
        stop_at = None
        for m in minutes(day):
            if slot < m <= "15:58" and (value(m) - credit) >= CFG["stop_mult"] * credit:
                stop_at = m
                break
        if stop_at:
            v = value(stop_at)
            fees += sum(per_contract_fee(L[n].price(stop_at, iv)) for n in st)
            res.update(outcome=f"stop {stop_at}", stopped=True, pnl=round((credit - v) * 100, 2))
        else:
            S58, close = spy["15:58"], spy["15:59"]
            pnl, notes = credit, []
            for side, sk, lk, r in C.SIDES:
                kk = st[sk]
                if (S58 - kk if r == "P" else kk - S58) <= CFG["near_money_dist"]:
                    cost = L[sk].price("15:58", iv) - L[lk].price("15:58", iv)
                    fees += per_contract_fee(L[sk].price("15:58", iv)) + per_contract_fee(L[lk].price("15:58", iv))
                    notes.append(f"{side} closed 15:58")
                else:
                    intr = (lambda K: max(0.0, K - close)) if r == "P" else (lambda K: max(0.0, close - K))
                    cost = intr(kk) - intr(st[lk])
                    if cost:
                        notes.append(f"{side} settles {cost:.2f}")
                pnl -= cost
            res.update(outcome="expiry" + (f" ({'; '.join(notes)})" if notes else ""), stopped=False,
                       pnl=round(pnl * 100, 2))
        res["fees"] = round(fees, 2)
        out.append(res)
    return out


def actual_day(day):
    """What the SPY trader did: entries (strikes, fill credit) and stop closes, from its log."""
    ent, closed, other = {}, {}, []
    for line in (HERE / "condor_v2_log.jsonl").read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not e.get("time", "").startswith(day):
            continue
        a = e.get("action")
        if a == "entered":
            ent[e["id"]] = e
        elif a == "closed":
            closed[e["pos"]] = e
        elif a in ("entry_skipped", "entry_unwound", "entry_blocked", "near_money_close",
                   "short_closed_by_broker", "near_money_close_failed"):
            other.append(e)
    return ent, closed, other


def days(d0, d1):
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            yield d.isoformat()
        d += timedelta(days=1)


def net(r, flat_fee=False):
    """Replay P&L of one condor after the measured execution costs and commissions."""
    cost = ENTRY_SLIP * 100 + (STOP_SLIP * 100 if r["stopped"] else 0)
    return r["pnl"] - cost - (FLAT_FEE if flat_fee else r["fees"])


def summary(results):
    trades = [dict(r, day=d) for d, rows in results.items() for r in rows if "pnl" in r]
    if not trades:
        print("no trades")
        return
    n, stops = len(trades), sum(r["stopped"] for r in trades)
    gross = sum(r["pnl"] for r in trades)
    sched = sum(net(r) for r in trades)
    flat = sum(net(r, True) for r in trades)
    ndays = len([d for d, rows in results.items() if any("pnl" in r for r in rows)])
    print(f"\n{ndays} days, {n} condors, stop rate {stops / n:.0%}, avg credit ${sum(r['credit'] for r in trades) / n * 100:.0f}")
    print(f"gross ${gross:+,.0f} (${gross / n:+.2f}/condor) | after measured slippage + schedule fees ${sched:+,.0f} "
          f"(${sched / n:+.2f}/condor) | with flat $5 fees ${flat:+,.0f} (${flat / n:+.2f}/condor)")
    print("\nmonth     condors  stops  gross    net(sched)  net($5)")
    by_m = defaultdict(list)
    for r in trades:
        by_m[r["day"][:7]].append(r)
    for m in sorted(by_m):
        rs = by_m[m]
        print(f"{m}   {len(rs):6d}  {sum(r['stopped'] for r in rs) / len(rs):4.0%}  {sum(r['pnl'] for r in rs):+7.0f}  "
              f"{sum(net(r) for r in rs):+9.0f}  {sum(net(r, True) for r in rs):+8.0f}")
    print("\nslot   condors  stops  avg gross  avg net(sched)")
    by_s = defaultdict(list)
    for r in trades:
        by_s[r["slot"]].append(r)
    for s in sorted(by_s):
        rs = by_s[s]
        print(f"{s}  {len(rs):6d}  {sum(r['stopped'] for r in rs) / len(rs):4.0%}  {sum(r['pnl'] for r in rs) / len(rs):+8.1f}  "
              f"{sum(net(r) for r in rs) / len(rs):+10.1f}")
    by_d = defaultdict(float)
    for r in trades:
        by_d[r["day"]] += net(r)
    vals = [by_d[d] for d in sorted(by_d)]
    peak = cum = dd = 0.0
    for v in vals:
        cum += v
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    print(f"\ndays up {sum(v > 0 for v in vals) / len(vals):.0%} | best ${max(vals):+.0f} | worst ${min(vals):+.0f} | "
          f"max drawdown ${dd:,.0f} | cumulative ${cum:+,.0f} (schedule fees)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d0 = date.fromisoformat(args[0] if args else "2026-10-05")
    d1 = date.fromisoformat(args[1] if len(args) > 1 else (args[0] if args else "2026-10-09"))
    results = {}
    for day in days(d0, d1):
        rows = replay_day(day)
        if rows:
            results[day] = rows
    if "--summary" in sys.argv:
        (HERE / "replay_results.json").write_text(json.dumps(results, indent=1))
        summary(results)
        return
    report = {}
    for day, bt in results.items():
        ent, closed, other = actual_day(day)
        report[day] = {"backtest": bt, "spy_close": bars("SPY", day).get("15:59"),
                       "actual": {"entered": list(ent.values()), "closed": closed, "events": other}}
        print(f"{day}: replayed {sum(1 for r in bt if 'pnl' in r)} condors, actual entries {len(ent)}", flush=True)
    (HERE / "replay_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {HERE / 'replay_report.json'}")


if __name__ == "__main__":
    main()
