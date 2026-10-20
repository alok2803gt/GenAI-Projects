#!/usr/bin/env python3
"""How did one OPEX day differ from the others? Builds that day's replay file from the quote
recorder (bid/ask midpoints, since same-day bars aren't served), replays the 10-entry setup at the
0.5x stop, and prints its SPY path and condor results next to every OPEX day in opex_results.json.

    python opex_compare.py 2026-10-16
"""
import json
import math
import sys

import opex_backtest as O
import replay_backtest as R


def build_day_from_recorder(day):
    f = R.HERE / "quotes" / f"SPY_{day}.jsonl"
    out = {"SPY": {}}
    for line in f.read_text().splitlines():
        s = json.loads(line)
        b, a = (s.get("spy") or [None, None])[:2]
        if b and a:
            out["SPY"][s["t"]] = round((b + a) / 2, 3)
        for lab, (ob, oa) in (s.get("o") or {}).items():
            if ob is not None and oa:
                k, r = float(lab[:-1]), lab[-1]
                out.setdefault(R.occ(day, k, r), {})[s["t"]] = round((ob + oa) / 2, 3)
    (R.CACHE / f"day_{day}.json").write_text(json.dumps(out))


def day_stats(day, rows):
    spy = R.bars("SPY", day)
    ts = sorted(spy)
    px = [spy[t] for t in ts]
    o, c = spy.get("09:31") or px[0], px[-1]
    after10 = [spy[t] for t in ts if t >= "10:01"]
    rets = [math.log(b / a) for a, b in zip(px, px[1:]) if a and b]
    rv = math.sqrt(sum(r * r for r in rets) * 252) * 100 if rets else 0  # annualized, from 1-min moves
    trades = [r for r in rows if "pnl" in r]
    net = sum(R.net(r) for r in trades)
    s10 = spy.get("10:01")
    return {"day": day, "move%": (c / o - 1) * 100, "range%": (max(px) - min(px)) / o * 100,
            "drift_from_10:01": c - s10 if s10 else None, "rv%": rv,
            "iv%": sum(r["iv"] for r in trades) / len(trades) * 100 if trades else None,
            "credit": sum(r["credit"] for r in trades) / len(trades) * 100 if trades else None,
            "stops": sum(r["stopped"] for r in trades), "n": len(trades), "net": net}


def main():
    day = sys.argv[1]
    if not (R.CACHE / f"day_{day}.json").exists():
        build_day_from_recorder(day)
    R.CFG = dict(R.C.DEFAULTS, entry_times=O.SLOTS10, stop_mult=0.5)
    today_rows = R.replay_day(day)
    hist = json.loads((R.HERE / "opex_results.json").read_text())["0.5x stop (current)"]
    stats = [day_stats(d, rows) for d, rows in sorted(hist.items())]
    t = day_stats(day, today_rows)
    hdr = f"{'day':<11}{'move%':>7}{'range%':>8}{'drift$':>8}{'RV%':>6}{'IV%':>6}{'cred$':>7}{'stops':>7}{'net$':>8}"
    print(hdr)
    for s in stats + [t]:
        mark = "  <-- " + day if s is t else ""
        print(f"{s['day']:<11}{s['move%']:>+7.2f}{s['range%']:>8.2f}{(s['drift_from_10:01'] or 0):>+8.2f}{s['rv%']:>6.1f}"
              f"{(s['iv%'] or 0):>6.1f}{(s['credit'] or 0):>7.0f}{s['stops']:>4}/{s['n']:<2}{s['net']:>+8.0f}{mark}")
    win = [s for s in stats if s["net"] > 0]
    lose = [s for s in stats if s["net"] <= 0]
    avg = lambda xs, k: sum(abs(x[k] or 0) for x in xs) / len(xs)
    print(f"\nwinning OPEX days ({len(win)}): |move| {avg(win, 'move%'):.2f}%  range {avg(win, 'range%'):.2f}%  "
          f"|drift| ${avg(win, 'drift_from_10:01'):.2f}  RV {avg(win, 'rv%'):.1f}%  IV {avg(win, 'iv%'):.1f}%")
    print(f"losing OPEX days  ({len(lose)}): |move| {avg(lose, 'move%'):.2f}%  range {avg(lose, 'range%'):.2f}%  "
          f"|drift| ${avg(lose, 'drift_from_10:01'):.2f}  RV {avg(lose, 'rv%'):.1f}%  IV {avg(lose, 'iv%'):.1f}%")
    print(f"\n{day} slots:")
    for r in today_rows:
        print(" ", r.get("slot"), r.get("strikes"), r.get("credit"), r.get("outcome") or r.get("skip"), r.get("pnl"))


if __name__ == "__main__":
    main()
