#!/usr/bin/env python3
"""SPX call (or, with --put, put) scalper: buy one same-day at-the-money SPXW option at the open, take +$100
the moment it's there.

Rules (user, 2026-10-13):
  entry   09:31 ET (one minute after the open; the first minute's quotes are too wide), 1 ATM 0DTE SPXW
          call, strike = SPX rounded to the 5-point grid
  target  +$100 after commissions: a resting SELL limit goes in right after the fill, so the exchange
          fills it the instant that price trades
  stop    -$200: a resting SELL STOP triggered by the bid, in a one-cancels-other group with the
          target, so both exits live at IBKR and survive a crash of this script or the laptop
  time    anything still open at 15:50 is sold (SPXW is cash-settled, but we don't hold into the close)
One trade a day. PAPER only: refuses unless the logged-in account starts with DU. Own client id, own
state and log; the condor trader, monitor and recorder ignore SPX.

    python spx_call_trader.py            # today's trade
    python spx_call_trader.py --dry-run  # price everything, place nothing
    python spx_call_trader.py --put      # the put trader (state spx_put_*, client 1601)
"""
import json
import math
import sys
from datetime import datetime, time as dtime
from pathlib import Path

from ib_insync import IB, Index, LimitOrder, Option, StopOrder

import condor_v2 as C

HERE = Path(__file__).resolve().parent
# --put runs the mirror-image PUT trader (user ask 2026-10-13): same rules, its own state, log and
# client id, so the two run side by side.
RIGHT = "P" if "--put" in sys.argv else "C"
KIND = "put" if RIGHT == "P" else "call"
NAME = f"SPX {KIND.upper()}"
STATE = HERE / f"spx_{KIND}_state.json"
LOG = HERE / f"spx_{KIND}_log.jsonl"
CLIENT_ID = 1601 if RIGHT == "P" else 1600
ENTRY, EXIT_BY = dtime(9, 31), dtime(15, 50)
ENTRY_GIVE_UP = dtime(9, 40)  # no TWS by then -> skip the day
TARGET_USD, STOP_USD = 100.0, 200.0
FEE_PER_SIDE = 1.30          # IBKR + CBOE/regulatory fees per SPXW contract, each way (estimate)
POLL_SECONDS = 3


def now():
    return C.now_et()


def log(action, **kw):
    line = {"time": now().isoformat(), "action": action, **kw}
    print(f"[{now():%H:%M:%S}] {action}: {json.dumps(kw)}", flush=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(line) + "\n")


def tick(px):
    """SPXW minimum tick: $0.05 below $3, $0.10 from $3 up."""
    return 0.05 if px < 3 else 0.10


def round_up(px):
    t = tick(px)
    return round(math.ceil(round(px / t, 6)) * t, 2)


def round_down(px):
    t = tick(px)
    return round(math.floor(round(px / t, 6)) * t, 2)


def target_price(fill):
    """Sell price that nets TARGET_USD after both sides' fees, rounded up to a valid tick."""
    return round_up(fill + (TARGET_USD + 2 * FEE_PER_SIDE) / 100)


def stop_price(fill):
    """Bid at or below which the position has lost STOP_USD before fees."""
    return round(fill - STOP_USD / 100, 2)


def load_state():
    if STATE.exists():
        s = json.loads(STATE.read_text())
        if s.get("day") == now().date().isoformat():
            return s
    return {"day": now().date().isoformat(), "status": "waiting"}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))


def quote(ib, c):
    t = ib.reqMktData(c, "", False, False)
    ib.sleep(2)
    b, a = C.safe_px(t.bid), C.safe_px(t.ask)
    ib.cancelMktData(c)
    return b, a


def place(ib, c, action, px, wait):
    """One limit order; returns (filled, avg price, trade). Cancels if not filled within `wait` s."""
    trade = ib.placeOrder(c, LimitOrder(action, 1, px, tif="DAY"))
    for _ in range(wait):
        ib.sleep(1)
        if trade.orderStatus.status == "Filled":
            return True, float(trade.orderStatus.avgFillPrice), trade
        if trade.orderStatus.status in C.IB_DONE_NOT_FILLED:
            return False, None, trade
    ib.cancelOrder(trade.order)
    for _ in range(30):
        ib.sleep(1)
        if trade.orderStatus.status == "Filled":
            return True, float(trade.orderStatus.avgFillPrice), trade
        if trade.orderStatus.status in C.IB_DONE_NOT_FILLED:
            return False, None, trade
    raise RuntimeError(f"{action} @ {px}: cancel never confirmed -- fill state unknown")


def buy(ib, c):
    """Mid, then the ask, then a tick through the ask -- re-quoted each step."""
    for step in range(3):
        b, a = quote(ib, c)
        if not (b and a):
            continue
        mid = (b + a) / 2
        px = [round_down(mid), a, round_up(a + tick(a))][step]
        log("buy_step", step=step + 1, bid=b, ask=a, limit=px)
        ok, fill, _ = place(ib, c, "BUY", px, 15)
        if ok:
            return fill
    return None


def sell_now(ib, c, why):
    """Exit: cross the bid, re-quoted each step, deeper each time."""
    for step in range(4):
        b, a = quote(ib, c)
        ref = b if b else (a or 0.05)
        px = max(round_down(ref - (0, tick(ref), 3 * tick(ref), 0.25 * ref)[step]), 0.05)
        log("sell_step", why=why, step=step + 1, bid=b, ask=a, limit=px)
        ok, fill, _ = place(ib, c, "SELL", px, 6)
        if ok:
            return fill
    return None


def finish(state, cfg, how, exit_px):
    pnl = round((exit_px - state["fill"]) * 100 - 2 * FEE_PER_SIDE, 2)
    state.update(status="done", exit=exit_px, exit_reason=how, pnl=pnl)
    save_state(state)
    log("closed", reason=how, entry=state["fill"], exit=exit_px, pnl_est=pnl)
    C.telegram(cfg, f"{NAME} {state['strike']:g}{RIGHT} closed ({how}): bought ${state['fill']}, sold ${exit_px}, "
                    f"about ${pnl:+.0f} after fees")


def main():
    dry = "--dry-run" in sys.argv
    n = now()
    if n.weekday() >= 5:
        print("weekend -- nothing to do")
        return
    cfg = C.load_config()
    state = load_state()
    if state["status"] == "done":
        print(f"today's trade is done: {state.get('exit_reason')} {state.get('pnl')}")
        return
    if state["status"] == "waiting" and n.time() > ENTRY_GIVE_UP:
        print(f"started after {ENTRY_GIVE_UP:%H:%M} with no entry -- skipping today (late buys aren't this strategy)")
        return
    ib = IB()
    # Restarts mid-trade retry until the 15:50 exit; the morning entry gives up at 09:40, since a
    # later buy is a different trade than the 09:31 one being measured (2026-10-15 outage).
    until = ENTRY_GIVE_UP if state["status"] == "waiting" else EXIT_BY
    if not C.connect_retry(ib, 7497, CLIENT_ID, until, cfg, NAME):
        if state["status"] == "waiting":
            state.update(status="done", exit_reason="TWS unreachable -- no entry")
            save_state(state)
            log("skip", reason=f"TWS unreachable until {until:%H:%M}")
        return
    accts = ib.managedAccounts()
    if not accts or not all(a.startswith("DU") for a in accts):
        log("abort", reason=f"not a paper account: {accts}")
        C.telegram(cfg, f"{NAME}: refusing to trade -- TWS on 7497 is logged into {accts}, not paper", True)
        ib.disconnect()
        return
    C.keep_awake(True)
    try:
        expiry = n.strftime("%Y%m%d")
        if state["status"] == "waiting":
            while now().time() < ENTRY:
                ib.sleep(5)
            idx = Index("SPX", "CBOE", "USD")
            ib.qualifyContracts(idx)
            t = ib.reqMktData(idx, "", False, False)
            ib.sleep(3)
            spx = C.safe_px(t.last) or C.safe_px(t.close)
            ib.cancelMktData(idx)
            if not spx:
                log("skip", reason="no SPX index price")
                return
            strike = 5 * round(spx / 5)
            c = Option("SPX", expiry, strike, RIGHT, "SMART", tradingClass="SPXW")
            ib.qualifyContracts(c)
            log("entry_plan", spx=spx, strike=strike, conId=c.conId)
            if dry:
                b, a = quote(ib, c)
                log("dry_run", bid=b, ask=a, target=target_price(a or 0), stop=stop_price(a or 0))
                return
            fill = buy(ib, c)
            if fill is None:
                state.update(status="done", exit_reason="entry not filled")
                save_state(state)
                log("skip", reason="entry did not fill")
                C.telegram(cfg, f"{NAME} {strike}{RIGHT}: entry did not fill -- no trade today")
                return
            state.update(status="open", strike=strike, conId=c.conId, fill=fill)
            save_state(state)
            log("entered", strike=strike, fill=fill, target=target_price(fill), stop=stop_price(fill))
            C.telegram(cfg, f"{NAME} bought 1x {strike}{RIGHT} 0DTE @ ${fill} -- target ${target_price(fill)} "
                            f"(+$100), stop bid <= ${stop_price(fill)} (-$200), out by 15:50")
        else:
            c = Option("SPX", expiry, state["strike"], RIGHT, "SMART", tradingClass="SPXW")
            ib.qualifyContracts(c)
            held = sum(p.position for p in ib.positions() if p.contract.conId == c.conId)
            if held <= 0:
                log("restart", note="position already gone -- treating as closed at unknown price")
                state.update(status="done", exit_reason="gone on restart")
                save_state(state)
                return
            for tr in ib.reqAllOpenOrders():
                if tr.contract.conId == c.conId:
                    ib.cancelOrder(tr.order)  # replaced by a fresh target below
            ib.sleep(2)

        fill = state["fill"]
        tgt, stp = target_price(fill), stop_price(fill)
        # Both exits rest AT IBKR as a one-cancels-other pair (user ask 2026-10-13), so they still
        # work if this script, TWS's API link or the laptop goes away. The stop triggers on the bid
        # (triggerMethod 4 = bid for a sell) and then sells at market.
        oca = f"spx{KIND}-{state['day']}-{state['conId']}"
        tp = ib.placeOrder(c, LimitOrder("SELL", 1, tgt, tif="DAY", ocaGroup=oca, ocaType=1))
        sl = ib.placeOrder(c, StopOrder("SELL", 1, stp, tif="DAY", ocaGroup=oca, ocaType=1, triggerMethod=4))
        log("exits_placed", target=tgt, stop=stp, oca=oca)
        # Every poll's real bid/ask goes to a quote log (2026-10-14), so paper fills can be checked
        # against what the live market actually offered at that second.
        qt = ib.reqMktData(c, "", False, False)
        qlog = HERE / f"spx_{KIND}_quotes.jsonl"
        while True:
            ib.sleep(POLL_SECONDS)
            with open(qlog, "a") as f:
                f.write(json.dumps({"ts": now().isoformat(), "strike": state["strike"], "bid": C.safe_px(qt.bid),
                                    "ask": C.safe_px(qt.ask), "last": C.safe_px(qt.last)}) + "\n")
            for tr, why in ((tp, "target +$100"), (sl, "stop -$200")):
                if tr.orderStatus.status == "Filled":
                    finish(state, cfg, why, float(tr.orderStatus.avgFillPrice))
                    return
            if now().time() >= EXIT_BY:
                for tr in (tp, sl):
                    if tr.orderStatus.status not in C.IB_DONE_NOT_FILLED:
                        ib.cancelOrder(tr.order)
                for _ in range(15):
                    ib.sleep(1)
                    if all(tr.orderStatus.status in C.IB_DONE_NOT_FILLED | {"Filled"} for tr in (tp, sl)):
                        break
                for tr, why in ((tp, "target +$100"), (sl, "stop -$200")):
                    if tr.orderStatus.status == "Filled":  # an exit won the race with the cancel
                        finish(state, cfg, why, float(tr.orderStatus.avgFillPrice))
                        return
                if any(tr.orderStatus.status not in C.IB_DONE_NOT_FILLED for tr in (tp, sl)):
                    log("exit_failed", why="15:50: resting exits did not confirm cancelled")
                    C.telegram(cfg, f"{NAME} {state['strike']:g}{RIGHT}: could not cancel the resting exits at 15:50 "
                                    f"-- check TWS by hand", True)
                    return
                px = sell_now(ib, c, "time 15:50")
                if px is None:
                    log("exit_failed", why="time 15:50")
                    C.telegram(cfg, f"{NAME} {state['strike']:g}{RIGHT}: 15:50 exit did NOT fill -- close it by hand", True)
                    return
                finish(state, cfg, "time 15:50", px)
                return
    finally:
        C.keep_awake(False)
        ib.disconnect()


if __name__ == "__main__":
    main()
