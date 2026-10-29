#!/usr/bin/env python3
"""Condor V2 end-of-day P&L tracker -- deterministic, no LLM (user ask 2026-10-23). Task CondorV2-EODTracker,
weekdays 16:20, after the 16:10 shadow ledger.

Each day, from IBKR itself (read-only, paper account only):
  cash       every SPY option fill by the condor trader's client id (1580): sells +, buys -, x100
  settlement SPY options still held at the close, settled at intrinsic on SPY's 15:59 close
  fees       IBKR's reported commissions; replay_backtest's per-contract schedule if paper reports 0
  gross = cash + settlement, net = gross - fees
plus counts from condor_v2_log.jsonl and the shadow ledger's real-bid/ask column. One row per date in
condor_daily_tracker.csv (re-running a day replaces its row), and a one-line Telegram.

    python condor_eod_tracker.py              # today (after 16:00)
    python condor_eod_tracker.py --backfill   # seed earlier days from condor_daily_pnl.py's rebuild
    python condor_eod_tracker.py --show       # print the table
"""
import asyncio
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
CSV = HERE / "condor_daily_tracker.csv"
ET = ZoneInfo("America/New_York")
CONDOR_CLIENT = 1580
FIELDS = ["date", "rules", "entered", "stopped", "unwound", "held", "gross", "fees", "net", "real_mkt_gross",
          "cum_net", "source"]
RULES = [("2026-10-01", "early build"), ("2026-10-06", "v1 0.5x stop 10 slots"),
         ("2026-10-13", "1x stop late slots"), ("2026-10-14", "FINAL 0.5x stop late slots")]


def rules_for(day):
    name = RULES[0][1]
    for d, n in RULES:
        if day >= d:
            name = n
    return name


def counts(day):
    ents, stops, unw, closed = set(), 0, 0, set()
    p = HERE / "condor_v2_log.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not r.get("time", "").startswith(day):
            continue
        a = r.get("action")
        if a == "entered":
            ents.add(r["id"])
        elif a == "closed":
            closed.add(r["pos"])
            stops += str(r.get("reason", "")).startswith("stop")
        elif a == "entry_unwound":
            unw += 1
    return len(ents), stops, unw, len(ents - closed)


def real_mkt(day):
    p = HERE / "shadow_ledger.jsonl"
    if not p.exists():
        return ""
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            if d["date"] == day:
                rows = [x for x in d["rows"] if x["trade"].startswith("condor") and x["realistic"] is not None]
                return round(sum(x["realistic"] for x in rows), 2) if rows else ""
    return ""


def load():
    if not CSV.exists():
        return {}
    with open(CSV, newline="", encoding="utf-8") as f:
        return {r["date"]: r for r in csv.DictReader(f)}


def save(rows):
    cum = 0.0
    out = []
    for day in sorted(rows):
        r = rows[day]
        cum += float(r["net"])
        r["cum_net"] = round(cum, 2)
        out.append(r)
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    return out


async def from_ibkr(day):
    from ib_insync import IB, ExecutionFilter, Stock
    import replay_backtest as R
    ib = IB()
    await ib.connectAsync("127.0.0.1", 7497, clientId=1659, timeout=20, readonly=True)
    try:
        accts = ib.managedAccounts()
        if not accts or not all(a.startswith("DU") for a in accts):
            raise SystemExit(f"refusing: not a paper account {accts}")
        fills = await ib.reqExecutionsAsync(ExecutionFilter(clientId=CONDOR_CLIENT))
        cash, fees, n = 0.0, 0.0, 0
        for f in fills:
            c, e = f.contract, f.execution
            if c.symbol != "SPY" or c.secType != "OPT" or f.time.astimezone(ET).date().isoformat() != day:
                continue
            sign = 1 if e.side == "SLD" else -1
            cash += sign * e.price * e.shares * 100
            com = f.commissionReport.commission if f.commissionReport else 0.0
            fees += com if com and com < 1e9 else R.per_contract_fee(e.price) * e.shares
            n += 1
        spy = Stock("SPY", "SMART", "USD")
        await ib.qualifyContractsAsync(spy)
        bars = await ib.reqHistoricalDataAsync(spy, "", "1 D", "1 min", "TRADES", useRTH=True, formatDate=2)
        close = bars[-1].close if bars else None
        settle = 0.0
        exp = day.replace("-", "")
        for p in ib.positions():
            c = p.contract
            if c.symbol == "SPY" and c.secType == "OPT" and c.lastTradeDateOrContractMonth == exp and p.position:
                k = float(c.strike)
                intr = max(0.0, close - k) if c.right == "C" else max(0.0, k - close)
                settle += p.position * intr * 100
        return {"cash": cash, "settle": settle, "fees": fees, "fills": n, "spy_close": close}
    finally:
        ib.disconnect()


def telegram(text):
    import condor_v2 as C
    C.telegram(C.load_config(), text)


def run_today():
    day = datetime.now(ET).date().isoformat()
    got = asyncio.run(from_ibkr(day))
    ent, stp, unw, held = counts(day)
    if got["fills"] == 0 and ent == 0:
        print(f"{day}: no condor activity")
        return
    gross = round(got["cash"] + got["settle"], 2)
    fees = round(got["fees"], 2)
    rows = load()
    rows[day] = {"date": day, "rules": rules_for(day), "entered": ent, "stopped": stp, "unwound": unw,
                 "held": held, "gross": gross, "fees": fees, "net": round(gross - fees, 2),
                 "real_mkt_gross": real_mkt(day), "cum_net": "", "source": "ibkr fills"}
    out = save(rows)
    last = out[-1]
    fin = [r for r in out if r["date"] >= "2026-10-14"]
    msg = (f"DAY P&L {day}: {ent} condors, {stp} stopped, {held} held -> gross ${gross:+.0f}, fees ${fees:.0f}, "
           f"net ${gross - fees:+.0f} (real-market gross {last['real_mkt_gross'] or 'n/a'}). "
           f"Cumulative net ${float(last['cum_net']):+.0f}; final rules {len(fin)} days ${sum(float(r['net']) for r in fin):+.0f}. "
           f"SPY close {got['spy_close']}.")
    print(msg)
    telegram(msg)


def backfill():
    import condor_daily_pnl as D
    import io
    import contextlib
    rows = load()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        D.main()
    for line in buf.getvalue().splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("2026-"):
            continue
        day = parts[0]
        if day in rows and rows[day]["source"].startswith("ibkr fills"):
            continue  # never overwrite a fill-based day with the rebuild
        nums = parts[-7:]  # entered stopped held gross real net cum
        ent, stp, held, gross, real, net = nums[0], nums[1], nums[2], float(nums[3]), nums[4], float(nums[5])
        unw = sum(1 for l in (HERE / "condor_v2_log.jsonl").read_text(encoding="utf-8").splitlines()
                  if day in l and '"entry_unwound"' in l)
        rows[day] = {"date": day, "rules": rules_for(day), "entered": ent, "stopped": stp, "unwound": unw,
                     "held": held, "gross": gross, "fees": round(gross - net, 2), "net": net,
                     "real_mkt_gross": "" if real == "-" else float(real), "cum_net": "", "source": "rebuild"}
    save(rows)
    show()


def show():
    rows = load()
    print(f"{'date':<11}{'rules':<27}{'ent':>4}{'stop':>5}{'unw':>4}{'held':>5}{'gross':>9}{'fees':>7}{'net':>9}"
          f"{'real mkt':>10}{'cum net':>10}  source")
    for d in sorted(rows):
        r = rows[d]
        rm = r["real_mkt_gross"]
        real = f"{float(rm):+.0f}" if rm not in ("", None) else "-"
        print(f"{d:<11}{r['rules']:<27}{r['entered']:>4}{r['stopped']:>5}{r['unwound']:>4}{r['held']:>5}"
              f"{float(r['gross']):>+9.0f}{float(r['fees']):>7.0f}{float(r['net']):>+9.0f}"
              f"{real:>10}{float(r['cum_net']):>+10.0f}  {r['source']}")


if __name__ == "__main__":
    if "--show" in sys.argv:
        show()
    elif "--backfill" in sys.argv:
        backfill()
    else:
        run_today()
