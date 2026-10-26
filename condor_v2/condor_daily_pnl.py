#!/usr/bin/env python3
"""Condor V2 daily P&L since inception, rebuilt from condor_v2_log.jsonl (user ask 2026-10-22).

Per day: stops/closes and unwinds as logged; every held condor priced at that day's SPY close
(intrinsic), with a 15:57 near-money buyback charged at that short's last trade price that minute
(replay_cache day files / Polygon minute bars); commissions estimated at IBKR's per-contract schedule.
From 2026-10-12 the shadow ledger's realistic (real bid/ask) column is shown alongside.

    python condor_daily_pnl.py
"""
import json
from collections import OrderedDict, defaultdict

import replay_backtest as R

RULES = [("2026-10-01", "early build"), ("2026-10-06", "v1: 0.5x stop, 10 slots"),
         ("2026-10-13", "1x stop, late slots"), ("2026-10-14", "FINAL: 0.5x stop, late slots")]


def rules_for(day):
    name = RULES[0][1]
    for d, n in RULES:
        if day >= d:
            name = n
    return name


def fee(px):
    return R.per_contract_fee(px or 0.0)


def main():
    log = [json.loads(l) for l in open(R.HERE / "condor_v2_log.jsonl", encoding="utf-8") if l.strip()]
    by_day = OrderedDict()
    for r in log:
        d = r.get("time", "")[:10]
        if r.get("action") in ("entered", "closed", "entry_unwound", "near_money_close", "entry_partial"):
            by_day.setdefault(d, []).append(r)
    ledger = {}
    if (R.HERE / "shadow_ledger.jsonl").exists():
        for l in open(R.HERE / "shadow_ledger.jsonl", encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                rows = [x for x in d["rows"] if x["trade"].startswith("condor") and x["realistic"] is not None]
                if rows:
                    ledger[d["date"]] = round(sum(x["realistic"] for x in rows), 2)
    out, cum_g, cum_n = [], 0.0, 0.0
    for day, rows in by_day.items():
        ents = {r["id"]: r for r in rows if r["action"] == "entered"}
        if not ents and not any(r["action"] == "entry_unwound" for r in rows):
            continue
        closed = {r["pos"]: r for r in rows if r["action"] == "closed"}
        nm = defaultdict(list)
        for r in rows:
            if r["action"] == "near_money_close":
                nm[r["pos"]].append(r)
        if not (R.CACHE / f"day_{day}.json").exists() and (R.HERE / "quotes" / f"SPY_{day}.jsonl").exists():
            import opex_compare  # same-day bars aren't served: build the day from the quote recorder
            opex_compare.build_day_from_recorder(day)
        spy = R.bars("SPY", day)
        close = spy.get("15:59") or (list(spy.values())[-1] if spy else None)
        gross, fees, stops, held = 0.0, 0.0, 0, 0
        for r in rows:
            if r["action"] == "entry_unwound":
                gross += r["pnl"]
                fees += 6 * 0.68  # ~3 legs in and back out
        for pid, e in ents.items():
            k = e["strikes"]
            fees += sum(fee(0.3) for _ in range(4))
            c = closed.get(pid)
            if c:
                gross += c["pnl"]
                stops += str(c.get("reason", "")).startswith("stop")
                fees += 4 * 0.68
                continue
            held += 1
            if close is None:
                continue
            pnl = e["credit"]
            gone = {x["side"]: x for x in nm.get(pid, [])}
            for side, sk, lk, rt in R.C.SIDES:
                ks, kl = k[sk], k[lk]
                intr = (lambda K: max(0.0, K - close)) if rt == "P" else (lambda K: max(0.0, close - K))
                if side in gone:
                    hm = gone[side]["time"][11:16]
                    sb = R.bars(R.occ(day, ks, rt), day)
                    px = next((sb[m] for m in (hm, "15:58", "15:57", "15:56") if m in sb), intr(ks))
                    pnl -= px - intr(kl)  # bought the short back; the wing settles
                    fees += fee(px)
                else:
                    pnl -= intr(ks) - intr(kl)
            gross += pnl * 100 * e.get("qty", 1)
        net = gross - fees
        cum_g += gross
        cum_n += net
        out.append((day, rules_for(day), len(ents), stops, held, gross, ledger.get(day), net, cum_n))
    print(f"{'date':<11}{'rules':<30}{'entered':>8}{'stopped':>8}{'held':>6}{'gross':>9}{'real mkt*':>11}"
          f"{'est. net':>10}{'cum net':>10}")
    for day, rules, n, s, h, g, real, net, cum in out:
        print(f"{day:<11}{rules:<30}{n:>8}{s:>8}{h:>6}{g:>+9.0f}{(f'{real:+.0f}' if real is not None else '-'):>11}"
              f"{net:>+10.0f}{cum:>+10.0f}")
    print(f"\nTOTAL {len(out)} days, {sum(o[2] for o in out)} condors: gross {cum_g:+,.0f}, est. net {cum_n:+,.0f}")
    fin = [o for o in out if o[0] >= "2026-10-14"]
    if fin:
        print(f"FINAL RULES (since 10-14): {len(fin)} days, gross {sum(o[5] for o in fin):+,.0f}, "
              f"est. net {sum(o[7] for o in fin):+,.0f}, real-market gross "
              f"{sum(o[6] for o in fin if o[6] is not None):+,.0f}")


if __name__ == "__main__":
    main()
