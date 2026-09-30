"""Monitor the open SPY 771P (Ashley, filled $1.14 / avg cost $1.15) until it is
closed or 15:50 ET. Alerts on: breakeven reachable, -50% premium, SPY crossing
the strike, and a 15:30 warning if still open. Places no orders."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from ib_insync import IB, Option, Stock

ET = ZoneInfo("America/New_York")
ENTRY, BREAKEVEN, STOP = 1.15, 1.17, 0.57
STRIKE, EXPIRY = 771.0, "20260925"


def telegram(text):
    try:
        import requests
        cfg = json.load(open("scanner_config.json"))
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"], "text": "SPY 771P: " + text}, timeout=15)
    except Exception as e:
        print("telegram failed:", e)


ib = IB()
ib.connect("127.0.0.1", 7496, clientId=1710, timeout=20)
spy = Stock("SPY", "SMART", "USD"); ib.qualifyContracts(spy)
opt = Option("SPY", EXPIRY, STRIKE, "P", "SMART"); ib.qualifyContracts(opt)
sent = set()
while True:
    now = datetime.now(ET)
    if (now.hour, now.minute) >= (15, 50):
        print(f"{now:%H:%M} stopping watch (15:50) -- executor force-closes at 15:55", flush=True)
        break
    held = sum(p.position for p in ib.positions() if p.contract.conId == opt.conId)
    if held == 0:
        print(f"{now:%H:%M:%S} position CLOSED (Ashley exit or manual)", flush=True)
        telegram("position is closed")
        break
    sq = ib.reqMktData(spy, "", False, False)
    oq = ib.reqMktData(opt, "", False, False)
    ib.sleep(4)
    spot, bid, ask = sq.last or sq.close, oq.bid, oq.ask
    ib.cancelMktData(spy); ib.cancelMktData(opt)
    if not bid or bid != bid:
        ib.sleep(30); continue
    pnl = (bid - ENTRY) * 100
    line = f"{now:%H:%M:%S} SPY {spot:.2f} | 771P bid {bid:.2f} ask {ask:.2f} | P&L ${pnl:+.0f}"
    print(line, flush=True)
    if bid >= BREAKEVEN and "be" not in sent:
        sent.add("be"); telegram(f"BREAKEVEN reachable: bid {bid:.2f} >= {BREAKEVEN} (SPY {spot:.2f})")
    if bid <= STOP and "stop" not in sent:
        sent.add("stop"); telegram(f"down 50%: bid {bid:.2f} (SPY {spot:.2f}) -- entry was {ENTRY}")
    if spot < STRIKE and "itm" not in sent:
        sent.add("itm"); telegram(f"now ITM: SPY {spot:.2f} below {STRIKE:.0f} strike, bid {bid:.2f}")
    if (now.hour, now.minute) >= (15, 30) and "late" not in sent:
        sent.add("late"); telegram(f"15:30 and still open: bid {bid:.2f}, P&L ${pnl:+.0f}. "
                                   f"Executor force-closes at 15:55.")
    ib.sleep(50)
ib.disconnect()
