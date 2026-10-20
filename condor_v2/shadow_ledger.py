#!/usr/bin/env python3
"""Shadow ledger: re-score each day's PAPER trades against the real recorded quotes, with stricter
fill rules than IBKR's paper engine, to estimate what live trading would have kept (user ask
2026-10-14: "how can we build the environment simulating live trade?").

Data: quotes/SPY_<day>.jsonl (quote_recorder.py, 1-minute bid/ask for SPY + SPXW options) and
spx_<call|put>_quotes.jsonl (the SPX traders log bid/ask every 3 s while a position is open).

Rules (realistic = pessimistic where paper is generous):
  SPX scalps
    entry   paper filled at/below the mid; live may not. Entry = the larger of the paper fill and the
            ask seen when the order went out.
    target  a resting sell limit only fills live when a buyer is there: the BID must reach it. Walk the
            3-second quotes, then the minute quotes, until 15:50.
    stop    bid <= stop -> out at the bid minus one tick.
    time    15:50 -> out at the bid.
  Condors
    entry   shorts sold at the bid, wings bought at the ask, in the entry minute.
    stops   closed at the ask for shorts and the bid for wings, in the stop minute.
    others  (held, IBKR- or near-money-closed) keep the paper result, adjusted for the entry.
Commissions are the traders' own estimates on both sides, so the gap is fills only.

SPX target what-if (2026-10-14): each scalp is also walked with +$100/+$150/+$200 targets from the
real entry, over the same bid path (trader's 3-s log, then the recorder's minute SPXW quotes, which
continue after the paper trade exits), to see which target nets most once fills are real.

    python shadow_ledger.py [YYYY-MM-DD]   # default today; writes shadow_ledger.jsonl, Telegrams
    python shadow_ledger.py --summary      # SPX target what-if over every day so far
"""
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import condor_v2 as C

HERE = Path(__file__).resolve().parent
OUT = HERE / "shadow_ledger.jsonl"
SPX_FEES = 2 * 1.30


def tick(px):
    return 0.05 if px < 3 else 0.10


def jsonl(path, day):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if str(r.get("time", r.get("ts", ""))).startswith(day):
            rows.append(r)
    return rows


def minute_quotes(day):
    """{'HH:MM': sample} from the recorder (last sample per minute)."""
    f = HERE / "quotes" / f"SPY_{day}.jsonl"
    out = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["t"]] = r
    return out


def spx_trade(kind, day, mq):
    log = jsonl(HERE / f"spx_{kind}_log.jsonl", day)
    ent = next((r for r in log if r["action"] == "entered"), None)
    out = next((r for r in log if r["action"] == "closed"), None)
    if not ent or not out:
        return None
    steps = [r for r in log if r["action"] == "buy_step" and r["time"] <= ent["time"]]
    ask0 = steps[-1].get("ask") if steps else None
    fill, strike = ent["fill"], ent["strike"]
    entry = max(fill, ask0 or fill)
    label = f"{strike}{'C' if kind == 'call' else 'P'}"
    path = [(r["ts"][11:19], r.get("bid")) for r in jsonl(HERE / f"spx_{kind}_quotes.jsonl", day)
            if r.get("strike") == strike and r["ts"] > ent["time"]]
    last_t = path[-1][0][:5] if path else ent["time"][11:16]
    path += [(m + ":00", (s.get("x", {}).get(label) or [None])[0]) for m, s in sorted(mq.items())
             if m > last_t and m <= "15:50"]
    paper = round((out["exit"] - fill) * 100 - SPX_FEES, 2)
    # Live, the trader sets its target from its REAL fill (fill + $1.10 nets $100), so the realistic
    # result is the +$100 what-if from the real entry -- not paper's target from paper's entry.
    whatif = {t: walk(path, ent["time"][11:19], entry, t) for t in WHATIF_TARGETS}
    w = whatif[100]
    row = {"trade": f"SPX {kind} {label}", "paper": paper, "realistic": w["pnl"],
           "note": f"entry {fill}->{entry}, paper exit {out['exit']} ({out['reason']}); live +$100: {w['how']}",
           "whatif": {str(t): v for t, v in whatif.items()}}
    return row


WHATIF_TARGETS = (100, 150, 200)  # user ask 2026-10-14: would +$150/+$200 net more once fills are real?


def walk(path, entry_hms, entry, target_usd):
    """Exit the real bid path would have given a 1-lot bought at `entry` with a +target_usd resting
    sell (net of fees), the -$200 bid-triggered stop and the 15:50 time exit. A gap > 2 min in the
    record before the exit leaves it unscored -- it could hide a target or a stop."""
    import spx_call_trader as X
    tgt = X.round_up(entry + (target_usd + SPX_FEES) / 100)
    stp = round(entry - X.STOP_USD / 100, 2)
    exit_px, how = None, "no quotes after entry"
    prev = datetime.strptime(entry_hms, "%H:%M:%S")
    for ts, bid in path:
        if bid is None:
            continue
        cur = datetime.strptime(ts, "%H:%M:%S")
        if (cur - prev).total_seconds() > 120:
            return {"pnl": None, "how": f"quote gap {prev:%H:%M}-{cur:%H:%M}, unscored"}
        prev = cur
        if bid >= tgt:
            exit_px, how = tgt, f"target {tgt} at {ts}"
            break
        if bid <= stp:
            exit_px, how = max(round(bid - tick(bid), 2), 0.05), f"stop at {ts}"
            break
        exit_px, how = bid, f"time exit {ts}"
    if exit_px is None:
        return {"pnl": None, "how": how}
    if how.startswith("time") and ts < "15:49":
        return {"pnl": None, "how": f"record ends {ts}, before 15:50 -- unscored"}
    return {"pnl": round((exit_px - entry) * 100 - SPX_FEES, 2), "how": how}


ASHLEY_TWINS = HERE.parent / "ibkr_trader" / "backend" / "ashleyklieu_spx_twin_trades.jsonl"


def ashley_twins(day, mq):
    """Ashley's SPXW twins (paper, SPX-only since 2026-10-16): bought at the worse ask and sold at the
    worse bid of the recorder samples around each fill -- never better than paper got."""
    rows, open_ = [], {}
    for r in jsonl(ASHLEY_TWINS, day):
        if r["action"] == "entered":
            open_[r["setup"]] = r
            continue
        e = open_.pop(r["setup"], None)
        if not e:
            continue
        label, qty = f"{e['strike']:g}{e['right']}", e["qty"]
        paper = round((r["fill"] - e["fill"]) * 100 * qty, 2)
        # The executor's own quote at the signal (from 2026-10-17) beats 1-minute samples, which put a
        # $0.70 SPXW move between them on 2026-10-16. Older records fall back to the samples.
        asks = [e["ask"]] if e.get("ask") else \
            [q[1] for s in bracket(mq, e["time"]) if (q := s.get("x", {}).get(label)) and q[1]]
        bids = [r["bid"]] if r.get("bid") is not None else \
            [q[0] for s in bracket(mq, r["time"]) if (q := s.get("x", {}).get(label)) and q[0] is not None]
        name = f"Ashley SPX {label}"
        if not asks or not bids:
            rows.append({"trade": name, "paper": paper, "realistic": None,
                         "note": f"{e['fill']}->{r['fill']}; SPXW {label} not in the recorded quotes"})
            continue
        buy, sell = max([e["fill"]] + asks), min([r["fill"]] + bids)
        rows.append({"trade": name, "paper": paper, "realistic": round((sell - buy) * 100 * qty, 2),
                     "note": f"in {e['fill']}->{buy}, out {r['fill']}->{sell} ({e['time'][11:16]}-{r['time'][11:16]})"})
    for k, e in open_.items():
        rows.append({"trade": f"Ashley SPX {e['strike']:g}{e['right']}", "paper": None, "realistic": None,
                     "note": "no exit logged (held to settlement?)"})
    return rows


def ba(sample, label):
    q = (sample or {}).get("o", {}).get(label)
    return (q[0] or 0, q[1]) if q and q[1] is not None else (None, None)


def bracket(mq, iso):
    """The recorder samples on either side of a fill: HH:MM is taken at the minute's start, so a fill
    at HH:MM:40 sits between it and HH:MM+1. Scoring with the worse of the two removes the bias of
    always pricing a stop before the move that triggered it."""
    t = datetime.fromisoformat(iso)
    return [mq[m] for m in (f"{t:%H:%M}", f"{t + timedelta(minutes=1):%H:%M}") if m in mq]


def condor_credit(sample, lab):
    """Credit to OPEN at the real market: shorts sold at the bid, wings bought at the ask."""
    b = {n: ba(sample, l) for n, l in lab.items()}
    if any(v[1] is None for v in b.values()):
        return None
    return b["short_put"][0] + b["short_call"][0] - b["long_put"][1] - b["long_call"][1]


def condor_cost(sample, lab):
    """Cost to CLOSE at the real market: shorts bought at the ask, wings sold at the bid."""
    b = {n: ba(sample, l) for n, l in lab.items()}
    if any(v[1] is None for v in b.values()):
        return None
    return b["short_put"][1] + b["short_call"][1] - b["long_put"][0] - b["long_call"][0]


def condor_trades(day, mq):
    """Condor P&L in the log is gross (no fees), per the trader's close_condor -- so is this."""
    log = jsonl(HERE / "condor_v2_log.jsonl", day)
    ents = {r["id"]: r for r in log if r["action"] == "entered"}
    closed = {r["pos"]: r for r in log if r["action"] == "closed"}
    near, gone_at = {}, {}
    for r in log:
        if r["action"] == "near_money_close":
            near.setdefault(r["pos"], set()).add(r["side"])
            gone_at.setdefault(r["pos"], []).append((r["side"], r["time"]))
    held_cost = {}  # paper's own buyback cost, only knowable from state.json while it is still this day's
    try:
        st = json.loads((HERE / "state.json").read_text())
        if st.get("day", st.get("date")) == day or any(p.get("id") in near for p in st.get("open", [])):
            held_cost = {p["id"]: p["closed_cost"] for p in st.get("open", []) if p.get("closed_cost")}
    except (OSError, ValueError):
        pass
    mids = [(s["spy"][0] + s["spy"][1]) / 2 for m, s in sorted(mq.items()) if s["spy"][0] and s["spy"][1]]
    spy_close = mids[-1] if mids else None
    rows = []
    for pid, e in ents.items():
        k, qty, name = e["strikes"], e.get("qty", 1), f"condor {e['slot']}"
        lab = {"short_put": f"{k['short_put']:g}P", "long_put": f"{k['long_put']:g}P",
               "short_call": f"{k['short_call']:g}C", "long_call": f"{k['long_call']:g}C"}
        credits = [x for x in (condor_credit(s, lab) for s in bracket(mq, e["time"])) if x is not None]
        if not credits:
            rows.append({"trade": name, "paper": None, "realistic": None, "note": "no entry quotes"})
            continue
        real_credit = min([e["credit"]] + credits)  # never better than paper got
        c = closed.get(pid)
        extra, note = 0.0, f"credit {e['credit']:.2f} -> {real_credit:.2f}"
        if c:
            paper = c["pnl"]
        elif spy_close:
            # A held condor can still have had its threatened short bought back by the 15:57
            # near-money pass (2026-10-15: three 772C shorts at $0.44 with SPY 772.05). That side
            # is no longer worth its intrinsic -- it cost what the buyback cost.
            gone = near.get(pid, set())
            paid = held_cost.get(pid)
            intr = 0.0
            if "put" not in gone:
                intr += max(0, k["short_put"] - spy_close) - max(0, k["long_put"] - spy_close)
            else:
                intr -= max(0, k["long_put"] - spy_close)
            if "call" not in gone:
                intr += max(0, spy_close - k["short_call"]) - max(0, spy_close - k["long_call"])
            else:
                intr -= max(0, spy_close - k["long_call"])
            real_paid = 0.0
            for side, t in gone_at.get(pid, []):
                asks = [ba(s, lab[f"short_{side}"])[1] for s in bracket(mq, t)]
                real_paid += max([a for a in asks if a is not None] or [0.0])
            if paid is None:
                paid = real_paid
            extra = max(0.0, real_paid - paid)
            paper = (e["credit"] - paid - intr) * 100 * qty
            if gone:
                note += f", near-money buyback {'/'.join(sorted(gone))} {paid:.2f} -> {max(paid, real_paid):.2f}"
        else:
            rows.append({"trade": name, "paper": None, "realistic": None, "note": "no close quotes"})
            continue
        real = paper + (real_credit - e["credit"] - extra) * 100 * qty
        if c and str(c.get("reason", "")).startswith("stop"):
            paper_cost = e["credit"] - paper / 100 / qty
            costs = [x for x in (condor_cost(s, lab) for s in bracket(mq, c["time"])) if x is not None]
            real_cost = max([paper_cost] + costs)
            real = (real_credit - real_cost) * 100 * qty
            note += f", {c['reason']} cost {paper_cost:.2f} -> {real_cost:.2f}"
        else:
            note += f", {c['reason']}" if c else ", held"
        rows.append({"trade": name, "paper": round(paper, 2), "realistic": round(real, 2), "note": note})
    rows += unwound_entries(log, mq)
    return rows


def unwound_entries(log, mq):
    """Entries that filled only some legs and were sold back (2026-10-16: two cost $29 the ledger
    never saw). Real cost of each filled leg: a long bought at the ask and sold at the bid, a short
    sold at the bid and bought back at the ask -- around the entry and unwind times."""
    plans = {r["slot"]: r for r in log if r["action"] == "entry_plan"}
    partial = {r["slot"]: r for r in log if r["action"] == "entry_partial"}
    rows = []
    for u in (r for r in log if r["action"] == "entry_unwound"):
        slot, paper = u["slot"], u["pnl"]
        plan, part = plans.get(slot), partial.get(slot)
        real, note = paper, "unwound -- no leg quotes, paper cost kept"
        if plan and part:
            k = plan["strikes"]
            total, ok = 0.0, True
            for leg in part.get("filled", {}):
                lab = f"{k[leg]:g}{'P' if 'put' in leg else 'C'}"
                ins = [ba(s, lab) for s in bracket(mq, part["time"])]
                outs = [ba(s, lab) for s in bracket(mq, u["time"])]
                ins, outs = [q for q in ins if q[1] is not None], [q for q in outs if q[1] is not None]
                if not ins or not outs:
                    ok = False
                    break
                if leg.startswith("long"):
                    total += min(b for b, _ in outs) - max(a for _, a in ins)
                else:
                    total += min(b for b, _ in ins) - max(a for _, a in outs)
            if ok:
                real = min(paper, total * 100)
                note = f"unwound ({u.get('status', '')}): filled {', '.join(part.get('filled', {}))}"
        rows.append({"trade": f"condor {slot} (unwound)", "paper": round(paper, 2), "realistic": round(real, 2),
                     "note": note})
    return rows


def whatif_summary(lines):
    """Per target: trades scored, hits, stops, time exits, total and average realistic P&L."""
    out = {}
    for t in WHATIF_TARGETS:
        res = [r["whatif"][str(t)] for d in lines for r in d["rows"] if "whatif" in r and r["whatif"][str(t)]["pnl"] is not None]
        n = len(res)
        out[t] = {"n": n, "hits": sum(x["how"].startswith("target") for x in res),
                  "stops": sum(x["how"].startswith("stop") for x in res),
                  "total": round(sum(x["pnl"] for x in res), 2), "avg": round(sum(x["pnl"] for x in res) / n, 2) if n else None}
    return out


def print_whatif(s, title):
    print(title)
    for t, v in s.items():
        avg = "n/a" if v["avg"] is None else f"{v['avg']:+.0f}"
        print(f"  +${t:<4} {v['n']:>3} scored  {v['hits']:>3} hit  {v['stops']:>3} stopped  "
              f"{v['n'] - v['hits'] - v['stops']:>3} time-exit  total {v['total']:+8.0f}  avg {avg}")


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--summary" in sys.argv:  # every day so far: which SPX target nets the most at real prices
        lines = [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()] if OUT.exists() else []
        print_whatif(whatif_summary(lines), f"SPX scalper targets, realistic fills, {len(lines)} days in the ledger:")
        return
    day = pos[0] if pos else date.today().isoformat()
    mq = minute_quotes(day)
    rows = ([r for r in (spx_trade("call", day, mq), spx_trade("put", day, mq)) if r]
            + ashley_twins(day, mq) + condor_trades(day, mq))
    scored = [r for r in rows if r["paper"] is not None and r["realistic"] is not None]
    paper, real = sum(r["paper"] for r in scored), sum(r["realistic"] for r in scored)
    fmt = lambda v: "n/a" if v is None else f"{v:+.0f}"
    for r in rows:
        print(f"{r['trade']:<20} paper {fmt(r['paper']):>6}  realistic {fmt(r['realistic']):>6}  {r['note']}")
        for t, w in r.get("whatif", {}).items():
            print(f"{'':<20}   what-if +${t}: {fmt(w['pnl'])}  {w['how']}")
    print(f"TOTAL {len(scored)} scored: paper {paper:+.0f}, realistic {real:+.0f}, gap {real - paper:+.0f}")
    keep = [l for l in (OUT.read_text().splitlines() if OUT.exists() else []) if l.strip() and json.loads(l)["date"] != day]
    keep.append(json.dumps({"date": day, "paper": round(paper, 2), "realistic": round(real, 2), "rows": rows}))
    OUT.write_text("\n".join(sorted(keep)) + "\n")  # one line per day, so a re-run replaces it
    lines = [json.loads(l) for l in keep]
    allday = whatif_summary(lines)
    print_whatif(allday, f"SPX targets to date ({len(lines)} days):")
    if "--no-telegram" not in sys.argv and scored:
        cfg = C.load_config()
        today = whatif_summary([lines[[d["date"] for d in lines].index(day)]])
        wi = " | ".join(f"+${t}: {v['hits']}/{v['n']} hit, ${v['total']:+.0f}" for t, v in today.items() if v["n"])
        cum = " | ".join(f"+${t}: avg ${v['avg']:+.0f} over {v['n']}" for t, v in allday.items() if v["n"])
        C.telegram(cfg, f"SHADOW LEDGER {day}: {len(scored)} trades -- paper ${paper:+.0f}, realistic ${real:+.0f} "
                        f"(gap ${real - paper:+.0f}). Realistic = fills at the real bid/ask instead of paper's."
                        + (f"\nSPX target what-if today: {wi}\nTo date: {cum}" if wi else ""))


if __name__ == "__main__":
    main()
