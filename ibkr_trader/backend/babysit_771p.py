"""
Babysit the open SPY 771P (Ashley, avg cost $1.15) to BREAKEVEN OR BETTER.

Resting SELL LIMIT at $1.17 (entry + both commissions). If the bid runs well
past that, the limit TRAILS up -- max(1.17, high_bid - 0.05) -- so a real move
is not handed back while the floor stays at breakeven. Re-prices at most once a
minute, never downward. Stops at 15:45; the Ashley executor force-closes at
15:55 if it is somehow still open. On a fill, the executor is restarted so it
reconciles to the real (flat) position instead of trying to sell it again.
"""
import json
import subprocess
from datetime import datetime
from os import getuid
from zoneinfo import ZoneInfo

from ib_insync import IB, LimitOrder, Option, Stock

ET = ZoneInfo("America/New_York")
ENTRY, FLOOR, TRAIL = 1.15, 1.17, 0.05
STRIKE, EXPIRY = 771.0, "20260925"


def telegram(text):
    try:
        import requests
        cfg = json.load(open("scanner_config.json"))
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"], "text": "771P babysitter: " + text}, timeout=15)
    except Exception as e:
        print("telegram failed:", e)


ib = IB()
ib.errorEvent += lambda rid, code, msg, c: print(f"IBKR {code}: {msg}", flush=True)
ib.connect("127.0.0.1", 7496, clientId=1712, timeout=20)
spy = Stock("SPY", "SMART", "USD"); ib.qualifyContracts(spy)
opt = Option("SPY", EXPIRY, STRIKE, "P", "SMART"); ib.qualifyContracts(opt)

qty = sum(p.position for p in ib.positions() if p.contract.conId == opt.conId)
if qty <= 0:
    print("no 771P position -- nothing to babysit")
    telegram("no position found; not placing anything")
    ib.disconnect(); raise SystemExit(0)

limit = FLOOR
order = LimitOrder("SELL", int(qty), limit, tif="DAY")
trade = ib.placeOrder(opt, order)
ib.sleep(3)
print(f"placed SELL {qty:g} @ {limit:.2f} (status {trade.orderStatus.status})", flush=True)
telegram(f"resting SELL {qty:g} @ ${limit:.2f} (breakeven incl. commissions). Trails up if the bid runs.")

high_bid = 0.0
while True:
    now = datetime.now(ET)
    if (now.hour, now.minute) >= (15, 45):
        print(f"{now:%H:%M} stopping -- executor force-closes at 15:55", flush=True)
        telegram(f"15:45: still open at limit ${limit:.2f}. Executor force-closes at 15:55.")
        break
    if trade.orderStatus.status == "Filled":
        px = trade.orderStatus.avgFillPrice
        pnl = (px - ENTRY) * 100 * qty
        print(f"{now:%H:%M:%S} FILLED @ {px:.2f} -> P&L ${pnl:+.2f}", flush=True)
        telegram(f"FILLED @ ${px:.2f} -> P&L ${pnl:+.2f} (entry ${ENTRY})")
        subprocess.run(["launchctl", "kickstart", "-k", f"gui/{getuid()}/com.ibkrtrader.ashley"],
                       capture_output=True)
        break
    if sum(p.position for p in ib.positions() if p.contract.conId == opt.conId) == 0:
        print(f"{now:%H:%M:%S} position closed elsewhere (Ashley exit) -- cancelling our order", flush=True)
        ib.cancelOrder(order)
        telegram("position closed by the executor; our resting order cancelled")
        break
    sq = ib.reqMktData(spy, "", False, False)
    oq = ib.reqMktData(opt, "", False, False)
    ib.sleep(4)
    spot, bid = sq.last or sq.close, oq.bid
    ib.cancelMktData(spy); ib.cancelMktData(opt)
    if bid and bid == bid and bid > 0:
        high_bid = max(high_bid, bid)
        want = max(FLOOR, round(high_bid - TRAIL, 2))
        if want > limit + 0.004:                      # never lower the limit
            limit = want
            if trade.orderStatus.status in ("Cancelled", "Inactive", "ApiCancelled"):
                order = LimitOrder("SELL", int(qty), limit, tif="DAY")   # fresh: ib_insync
                trade = ib.placeOrder(opt, order)                        # asserts on a done trade
            else:
                order.lmtPrice = limit
                trade = ib.placeOrder(opt, order)
            print(f"{now:%H:%M:%S} SPY {spot:.2f} bid {bid:.2f} -> limit raised to {limit:.2f}", flush=True)
            telegram(f"bid {bid:.2f} (SPY {spot:.2f}) -- limit raised to ${limit:.2f}")
        else:
            print(f"{now:%H:%M:%S} SPY {spot:.2f} | bid {bid:.2f} | limit {limit:.2f} | "
                  f"P&L if filled ${(limit-ENTRY)*100:+.0f}", flush=True)
    ib.sleep(45)
ib.disconnect()
