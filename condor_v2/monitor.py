#!/usr/bin/env python3
"""Watchdog for the condor_v2 instances (SPY in ./, QQQ in ./qqq).

Task Scheduler runs this every 5 minutes; it does nothing outside 09:50-16:10 ET on weekdays.
Each run:
  - restarts an instance that died for an infrastructure reason (TWS disconnect) or never
    started, at most 3x/day, only on a paper account with no untracked short legs;
  - never restarts after a partial fill / partial close -- alerts with the details instead;
  - alerts on any SPY/QQQ 0DTE option leg that no running trader is tracking;
  - alerts on entry slots (10:01..14:31) that passed without being attempted;
  - 15:50: closes untracked legs that could be assigned/exercised (shorts; longs near/in the money);
  - 16:05: sends a per-instance day summary.
Deterministic, no LLM. Alerts go to the Telegram chat in ./config.json.

    python monitor.py           # normal run
    python monitor.py --check   # print status only: no restarts, orders, or alerts
"""
import json
import random
import re
import subprocess
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from ib_insync import IB, LimitOrder, Option, Stock

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
PY = HERE / ".venv" / "Scripts" / "python.exe"
INSTANCES = {"SPY": HERE, "QQQ": HERE / "qqq"}
SLOTS = ["10:01", "10:31", "11:01", "11:31", "12:01", "12:31", "13:01", "13:31", "14:01", "14:31"]
MON_STATE = HERE / "monitor_state.json"
MON_LOG = HERE / "monitor_log.jsonl"
MAX_RESTARTS = 3
INFRA_ERRORS = ("ConnectionError", "Socket disconnect", "TimeoutError", "ConnectionRefused", "WinError")
ACTIVE = (dtime(9, 50), dtime(16, 10))
RESTART_WINDOW = (dtime(10, 0), dtime(15, 55))
FLATTEN_AT = dtime(15, 50)
SUMMARY_AT = dtime(16, 5)
NEAR_MONEY = 2.0
CHECK_ONLY = "--check" in sys.argv
NOW = datetime.now(ET)
TODAY = NOW.date().isoformat()
EXPIRY = NOW.strftime("%Y%m%d")


# ----------------------------------------------------------------- plumbing
def load_mon():
    if MON_STATE.exists():
        s = json.loads(MON_STATE.read_text())
        if s.get("day") == TODAY:
            return s
    return {"day": TODAY, "alerted": [], "restarts": {k: 0 for k in INSTANCES},
            "start_netliq": None, "flattened": False, "summary_sent": False}


def save_mon(s):
    if not CHECK_ONLY:
        MON_STATE.write_text(json.dumps(s, indent=1))


def log(msg):
    line = f"{datetime.now(ET).isoformat()} {msg}"
    print(line)
    if not CHECK_ONLY:
        with open(MON_LOG, "a") as f:
            f.write(line + "\n")


def telegram(text, high=False):
    if CHECK_ONLY:
        print(f"[would telegram{' HIGH' if high else ''}] {text}")
        return
    cfg = json.loads((HERE / "config.json").read_text())
    if not cfg.get("telegram_token") or not cfg.get("telegram_chat_id"):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"],
                            "text": ("[HIGH] " if high else "") + "CONDOR MONITOR: " + text}, timeout=10)
    except Exception as e:
        log(f"telegram failed: {e}")


def alert_once(mon, key, text, high=False):
    if key in mon["alerted"]:
        return
    mon["alerted"].append(key)
    log(f"ALERT {key}: {text}")
    telegram(text, high)


# ----------------------------------------------------------------- instance state
def running_instances():
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object { $_.CommandLine }"],
        capture_output=True, text=True, timeout=60).stdout
    running = set()
    for line in out.splitlines():
        if "condor_v2.py" not in line:
            continue
        m = re.search(r"--dir\s+\"?([^\"\s]+)", line)
        running.add("QQQ" if m and Path(m.group(1)).name.lower() == "qqq" else "SPY")
    return running


def read_state(base):
    p = base / "state.json"
    if p.exists():
        s = json.loads(p.read_text())
        if s.get("session") == TODAY:
            return s
    return {"open": [], "closed": [], "realized": 0.0}


def events_today(base):
    p = base / "condor_v2_log.jsonl"
    if not p.exists():
        return []
    evs = []
    for line in p.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("time", "").startswith(TODAY):
            evs.append(e)
    return evs


def pause_reason(base):
    p = base / "PAUSED.flag"
    return p.read_text().strip() if p.exists() else None


def expected_legs(tickers):
    """Option legs the given (running) traders believe they hold: {(sym, strike, right): qty}."""
    exp = {}
    for sym in tickers:
        for pos in read_state(INSTANCES[sym])["open"]:
            closed_sides = set(pos.get("legs_closed", []))
            closed_legs = set(pos.get("close_fills", {}))
            for leg, k in pos["strikes"].items():
                right = "P" if "put" in leg else "C"
                if ("put" if right == "P" else "call") in closed_sides or leg in closed_legs:
                    continue
                q = pos["qty"] * (-1 if leg.startswith("short") else 1)
                exp[(sym, float(k), right)] = exp.get((sym, float(k), right), 0) + q
    return exp


def actual_legs(ib, acct):
    act = {}
    for p in ib.positions(acct):
        c = p.contract
        if c.secType == "OPT" and c.symbol in INSTANCES and c.lastTradeDateOrContractMonth == EXPIRY:
            act[(c.symbol, float(c.strike), c.right)] = act.get((c.symbol, float(c.strike), c.right), 0) + p.position
    return act


def untracked(ib, acct, tickers):
    exp, act = expected_legs(tickers), actual_legs(ib, acct)
    diff = {}
    for key in set(exp) | set(act):
        d = act.get(key, 0) - exp.get(key, 0)
        if d:
            diff[key] = d
    return diff


def fmt(legs):
    return ", ".join(f"{q:+g} {s} {k:g}{r}" for (s, k, r), q in sorted(legs.items())) or "none"


# ----------------------------------------------------------------- actions
def restart(mon, sym, why):
    if mon["restarts"][sym] >= MAX_RESTARTS:
        alert_once(mon, f"restart_cap:{sym}",
                   f"{sym} instance down ({why}) but already restarted {MAX_RESTARTS}x today -- needs a human", True)
        return
    mon["restarts"][sym] += 1
    log(f"restarting {sym} ({why}), attempt {mon['restarts'][sym]}")
    if CHECK_ONLY:
        return
    args = [str(PY), "-u", str(HERE / "safe_start.py")] + (["qqq"] if sym == "QQQ" else [])
    subprocess.Popen(args, cwd=str(HERE), creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    telegram(f"{sym} instance was down ({why}) -- restarted (attempt {mon['restarts'][sym]}/{MAX_RESTARTS})")


def close_leg(ib, sym, strike, right, qty):
    """qty>0 means we hold it long (SELL to close); qty<0 short (BUY to close). Returns fill or None."""
    c = Option(sym, EXPIRY, strike, right, "SMART")
    ib.qualifyContracts(c)
    t = ib.reqMktData(c, "", False, False)
    ib.sleep(3)
    bid, ask = t.bid if t.bid == t.bid and t.bid > 0 else 0.0, t.ask if t.ask == t.ask and t.ask > 0 else None
    ib.cancelMktData(c)
    action = "SELL" if qty > 0 else "BUY"
    if action == "SELL":
        if bid <= 0:
            log(f"skip selling {sym} {strike:g}{right}: no bid")
            return None
        ladder = [round((bid + (ask or bid)) / 2, 2), round(bid, 2), round(max(bid - 0.05, 0.01), 2)]
    else:
        if ask is None:
            log(f"cannot buy back {sym} {strike:g}{right}: no ask")
            return None
        ladder = [round((bid + ask) / 2, 2), round(ask, 2), round(ask + 0.05, 2)]
    for px in ladder:
        trade = ib.placeOrder(c, LimitOrder(action, abs(qty), px, tif="DAY"))
        for _ in range(15):
            ib.sleep(1)
            if trade.orderStatus.status == "Filled":
                return trade.orderStatus.avgFillPrice
        ib.cancelOrder(trade.order)
        for _ in range(30):
            ib.sleep(1)
            if trade.orderStatus.status == "Filled":
                return trade.orderStatus.avgFillPrice
            if trade.orderStatus.status in ("Cancelled", "ApiCancelled", "Inactive"):
                break
        else:
            raise RuntimeError(f"cancel never confirmed for {action} {sym} {strike:g}{right}")
    return None


def flatten(ib, acct, mon, running):
    """15:50: close untracked legs that could be assigned/exercised overnight."""
    legs = untracked(ib, acct, running)
    if not legs:
        log("15:50 flatten: no untracked legs")
        return
    spots = {}
    for sym in {k[0] for k in legs}:
        s = Stock(sym, "SMART", "USD")
        ib.qualifyContracts(s)
        t = ib.reqMktData(s, "", False, False)
        ib.sleep(3)
        spots[sym] = t.marketPrice()
        ib.cancelMktData(s)
    done = []
    for (sym, k, r), q in sorted(legs.items()):
        spot = spots.get(sym)
        itm_or_near = spot is not None and (spot - k if r == "C" else k - spot) > -NEAR_MONEY
        if q < 0 or itm_or_near:
            if CHECK_ONLY:
                done.append(f"[would close] {q:+g} {sym} {k:g}{r} (spot {spot})")
                continue
            px = close_leg(ib, sym, k, r, q)
            done.append(f"closed {q:+g} {sym} {k:g}{r} @ {px}" if px else
                        f"COULD NOT close {q:+g} {sym} {k:g}{r} (no bid/ask or no fill) -- check exercise/assignment")
    left = untracked(ib, acct, running) if not CHECK_ONLY else legs
    shorts_left = {k: v for k, v in left.items() if v < 0}
    msg = "15:50 cleanup of untracked legs: " + ("; ".join(done) or "nothing near the money")
    log(msg)
    telegram(msg + (f". STILL SHORT: {fmt(shorts_left)} -- close by hand before 16:00" if shorts_left else ""),
             high=bool(shorts_left))


def summary(ib, acct, mon):
    parts = []
    for sym, base in INSTANCES.items():
        evs = events_today(base)
        acts = [e["action"] for e in evs]
        st = read_state(base)
        parts.append(f"{sym}: {acts.count('entered')} entered, {acts.count('entry_skipped')} skipped, "
                     f"{acts.count('entry_blocked')} blocked, {acts.count('closed')} stopped, "
                     f"{len(st.get('unwound', []))} partial entries unwound, "
                     f"{sum(1 for p in st.get('partial_legs', []) if p.get('legs'))} left open, "
                     f"{acts.count('close_incomplete')} close retries, "
                     f"{len(st['open'])} held to expiry, trader-booked ${st['realized']:+.0f}")
    netliq = next((float(r.value) for r in ib.accountSummary(acct) if r.tag == "NetLiquidation"), None)
    if netliq and mon["start_netliq"]:
        parts.append(f"account NetLiq {netliq:,.0f} ({netliq - mon['start_netliq']:+,.0f} today)")
    parts.append(f"0DTE legs left in account (incl. condors held to expiry): {fmt(actual_legs(ib, acct))}")
    msg = "DAY SUMMARY -- " + " | ".join(parts)
    log(msg)
    telegram(msg)


# ----------------------------------------------------------------- main
def main():
    if NOW.weekday() >= 5 or not (ACTIVE[0] <= NOW.time() <= ACTIVE[1]):
        return
    mon = load_mon()
    t = NOW.time()

    last = mon.get("last_run")
    if last and NOW - datetime.fromisoformat(last) > timedelta(minutes=15):
        alert_once(mon, f"blind:{last[:16]}",
                   f"monitor did not run from {last[11:16]} to {NOW:%H:%M} (PC asleep, off, or tasks blocked) -- "
                   f"positions were unwatched; check stops and P&L", True)
    mon["last_run"] = NOW.isoformat()
    # battery alerts: ibkr_trader/backend/battery_watch.py (IBKR-BatteryWatch, every minute, 24/7)

    ib = IB()
    try:
        ib.connect("127.0.0.1", 7497, clientId=random.randint(9000, 9999), timeout=15)
    except Exception as e:
        alert_once(mon, f"tws_down:{NOW.hour}", f"cannot reach TWS on 7497 ({type(e).__name__}) -- traders are blind", True)
        save_mon(mon)
        return
    try:
        acct = ib.managedAccounts()[0]
        if not acct.startswith("DU"):
            alert_once(mon, f"not_paper:{NOW.hour}",
                       f"TWS on 7497 is logged into {acct}, not paper -- monitor took no action", True)
            return
        if mon["start_netliq"] is None:
            mon["start_netliq"] = next((float(r.value) for r in ib.accountSummary(acct)
                                        if r.tag == "NetLiquidation"), None)

        running = running_instances()
        log(f"running instances: {sorted(running) or 'none'}")

        for sym, base in INSTANCES.items():
            reason = pause_reason(base)
            evs = events_today(base)
            if sym in running:
                if reason:
                    alert_once(mon, f"paused:{sym}:{reason[:40]}",
                               f"{sym} paused mid-session (no new entries): {reason}")
            elif RESTART_WINDOW[0] <= t < RESTART_WINDOW[1]:
                if reason is None:
                    restart(mon, sym, "process not running")
                elif any(e in reason for e in INFRA_ERRORS):
                    shorts = {k: v for k, v in untracked(ib, acct, running).items() if k[0] == sym and v < 0}
                    if shorts:
                        alert_once(mon, f"norestart_shorts:{sym}",
                                   f"{sym} died ({reason}) and the account has untracked SHORT legs "
                                   f"{fmt(shorts)} -- not restarting, check by hand", True)
                    else:
                        if not CHECK_ONLY:
                            (base / "PAUSED.flag").unlink()
                        restart(mon, sym, f"infrastructure error: {reason[33:110]}")
                else:
                    alert_once(mon, f"paused_down:{sym}:{reason[:40]}",
                               f"{sym} stopped and paused, NOT auto-restarting: {reason}. "
                               f"Reconcile positions, then delete {base / 'PAUSED.flag'}", True)

            # every slot that has passed should show up in the trader's log once it was running
            starts = [datetime.fromisoformat(e["time"]) for e in evs if e["action"] == "start"]
            seen = {e.get("slot") for e in evs if e.get("slot")}
            for slot in SLOTS:
                h, m = map(int, slot.split(":"))
                slot_dt = NOW.replace(hour=h, minute=m, second=0, microsecond=0)
                if NOW > slot_dt + timedelta(minutes=12) and slot not in seen and any(s < slot_dt for s in starts):
                    alert_once(mon, f"missed:{sym}:{slot}", f"{sym} {slot} entry slot passed without an attempt")

        # Held-to-expiry condors stay in state.json after the traders exit at 15:58, so judge
        # "untracked" against every instance's records, not just the running ones.
        for (sym, k, r), q in untracked(ib, acct, INSTANCES).items():
            if q < 0:
                alert_once(mon, f"untracked_short:{sym}:{k:g}{r}",
                           f"untracked SHORT {sym} {k:g}{r} x{-q} in the account -- no trader is managing it", True)

        if t >= FLATTEN_AT and not mon["flattened"]:
            mon["flattened"] = True
            flatten(ib, acct, mon, running)
        if t >= SUMMARY_AT and not mon["summary_sent"]:
            mon["summary_sent"] = True
            summary(ib, acct, mon)
    except Exception as e:
        alert_once(mon, f"monitor_error:{type(e).__name__}:{NOW.hour}", f"monitor itself failed: {type(e).__name__}: {e}", True)
        raise
    finally:
        save_mon(mon)
        ib.disconnect()


if __name__ == "__main__":
    main()
