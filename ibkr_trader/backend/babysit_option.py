"""Babysit one long option position out at BREAKEVEN OR BETTER.

Generalised from babysit_771p.py (2026-09-25, SPY 771P -> filled $1.17, +$1.22)
because this is now a recurring need: an Ashley 0DTE leg that has not hit her
exit signal, where the CEO wants the position off at no loss rather than left
to decay into the 15:55 force-close.

HOW BREAKEVEN IS DERIVED
------------------------
Not from the fill price. IBKR's avgCost for an option position is the total
cost PER CONTRACT and ALREADY INCLUDES the entry commission (766C: 1.46 fill
x100 + 0.7473 comm = 146.7473). Breakeven therefore also has to cover the
EXIT commission, which is not yet known, so SELL_COMM_ASSUMPTION is set to the
worst exit fee actually observed on this account (~$1.05; the cheapest seen is
$0.75). The limit is then rounded UP to the next cent, because rounding down
would rest the order one cent below true breakeven -- the error that actually
matters here.

TRAILING
--------
The floor never moves down. If the bid runs above breakeven the limit follows
it up at TRAIL below the session's high bid, so a real move is not handed back
while the worst case stays "no loss". Re-priced at most once a minute.

COORDINATION WITH THE ASHLEY EXECUTOR
-------------------------------------
The executor still tracks this position and will try to sell it on her exit
signal or at 15:55. If this script's order fills first, that later sell would
be an attempt to SHORT the option, which IBKR rejects on this account under
the $2,000 NLV rule (error 201). So on a fill the executor is restarted, which
makes it reconcile against real IBKR positions and drop the closed leg -- the
same step taken by hand after the 766P and 771P exits.

    ../venv/bin/python babysit_option.py --symbol SPY --expiry 20260928 \
        --strike 766 --right C [--trail 0.05] [--stop-at 15:45]
"""
import argparse
import json
import subprocess
from datetime import datetime
from math import ceil, isnan
from os import getuid
from zoneinfo import ZoneInfo

from ib_insync import IB, LimitOrder, Option, Stock

ET = ZoneInfo("America/New_York")
SELL_COMM_ASSUMPTION = 1.05     # worst exit fee seen on this account
ASHLEY_JOB = "com.ibkrtrader.ashley"


def telegram(text: str) -> None:
    try:
        import requests
        cfg = json.load(open("scanner_config.json"))
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"], "text": "babysitter: " + text},
                      timeout=15)
    except Exception as exc:
        print(f"telegram failed: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--expiry", required=True, help="YYYYMMDD")
    ap.add_argument("--strike", required=True, type=float)
    ap.add_argument("--right", required=True, choices=["C", "P"])
    ap.add_argument("--trail", type=float, default=0.05)
    ap.add_argument("--stop-at", default="15:45", help="ET HH:MM to stop babysitting")
    ap.add_argument("--no-restart-ashley", action="store_true")
    a = ap.parse_args()
    stop_h, stop_m = map(int, a.stop_at.split(":"))

    ib = IB()
    # Surface IBKR's reject reasons: the first 771P attempt printed only
    # "status Cancelled" because no error handler was attached.
    ib.errorEvent += lambda rid, code, msg, c: print(f"IBKR {code}: {msg}", flush=True)
    ib.connect("127.0.0.1", 7496, clientId=1733, timeout=25)

    und = Stock(a.symbol, "SMART", "USD")
    ib.qualifyContracts(und)
    opt = Option(a.symbol, a.expiry, a.strike, a.right, "SMART")
    ib.qualifyContracts(opt)

    pos = next((p for p in ib.positions() if p.contract.conId == opt.conId), None)
    if pos is None or pos.position <= 0:
        print(f"no long {a.strike:g}{a.right} position -- nothing to babysit")
        telegram(f"no {a.strike:g}{a.right} position found; nothing placed")
        ib.disconnect()
        return 0

    qty = int(abs(pos.position))
    cost_per_contract = abs(float(pos.avgCost))          # entry commission included
    total_cost = cost_per_contract * qty
    floor = ceil((total_cost + SELL_COMM_ASSUMPTION) / qty) / 100
    print(f"{a.strike:g}{a.right} qty {qty} | IBKR avgCost ${cost_per_contract:.4f}/contract "
          f"(${cost_per_contract / 100:.4f}/share) | breakeven limit ${floor:.2f}", flush=True)

    limit = floor
    order = LimitOrder("SELL", qty, limit, tif="DAY")
    trade = ib.placeOrder(opt, order)
    ib.sleep(3)
    print(f"placed SELL {qty} @ {limit:.2f} (status {trade.orderStatus.status})", flush=True)
    telegram(f"{a.strike:g}{a.right}: resting SELL {qty} @ ${floor:.2f} = breakeven incl. both "
             f"commissions (avgCost ${cost_per_contract / 100:.4f}). Trails up if the bid runs.")

    high_bid = 0.0
    while True:
        now = datetime.now(ET)
        if (now.hour, now.minute) >= (stop_h, stop_m):
            print(f"{now:%H:%M} stopping -- executor force-closes at 15:55", flush=True)
            telegram(f"{a.strike:g}{a.right}: {a.stop_at} reached, still open at ${limit:.2f}. "
                     f"Executor force-closes at 15:55.")
            break

        if trade.orderStatus.status == "Filled":
            px = trade.orderStatus.avgFillPrice
            pnl = px * 100 * qty - total_cost - SELL_COMM_ASSUMPTION
            print(f"{now:%H:%M:%S} FILLED @ {px:.2f} -> P&L ~${pnl:+.2f}", flush=True)
            telegram(f"{a.strike:g}{a.right} FILLED @ ${px:.2f} -> P&L ~${pnl:+.2f} "
                     f"(approx: exit fee assumed ${SELL_COMM_ASSUMPTION})")
            if not a.no_restart_ashley:
                r = subprocess.run(["launchctl", "kickstart", "-k", f"gui/{getuid()}/{ASHLEY_JOB}"],
                                   capture_output=True, text=True)
                print(f"ashley executor restarted to reconcile (rc={r.returncode})", flush=True)
            break

        if not any(p.contract.conId == opt.conId and p.position for p in ib.positions()):
            print(f"{now:%H:%M:%S} position closed elsewhere -- cancelling our order", flush=True)
            ib.cancelOrder(order)
            telegram(f"{a.strike:g}{a.right}: closed by the executor; our resting order cancelled")
            break

        uq = ib.reqMktData(und, "", False, False)
        oq = ib.reqMktData(opt, "", False, False)
        ib.sleep(4)
        spot = uq.last if uq.last and not isnan(uq.last) else uq.close
        bid = oq.bid
        ib.cancelMktData(und)
        ib.cancelMktData(opt)

        if bid and not isnan(bid) and bid > 0:
            high_bid = max(high_bid, bid)
            want = max(floor, round(high_bid - a.trail, 2))
            if want > limit + 0.004:                       # never lower the limit
                limit = want
                if trade.orderStatus.status in ("Cancelled", "Inactive", "ApiCancelled"):
                    order = LimitOrder("SELL", qty, limit, tif="DAY")   # fresh: ib_insync
                    trade = ib.placeOrder(opt, order)                   # asserts on a done trade
                else:
                    order.lmtPrice = limit
                    trade = ib.placeOrder(opt, order)
                print(f"{now:%H:%M:%S} {a.symbol} {spot or 0:.2f} bid {bid:.2f} "
                      f"-> limit raised to {limit:.2f}", flush=True)
                telegram(f"{a.strike:g}{a.right}: bid {bid:.2f} -- limit raised to ${limit:.2f}")
            else:
                print(f"{now:%H:%M:%S} {a.symbol} {spot or 0:.2f} | bid {bid:.2f} | "
                      f"limit {limit:.2f} | P&L if filled "
                      f"${limit * 100 * qty - total_cost - SELL_COMM_ASSUMPTION:+.0f}", flush=True)
        ib.sleep(45)

    ib.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
