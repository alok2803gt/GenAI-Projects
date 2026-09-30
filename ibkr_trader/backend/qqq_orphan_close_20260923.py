"""
ONE-OFF hard close for the orphaned QQQ 0DTE structure of 2026-09-23.

Why this exists: the 09:45 QQQ butterfly entry was REJECTED on its body leg
(IBKR error 201 -- "would result in an uncovered option position ... at least
USD 2000 NLV", account was $1,999.85), leaving both wings on and the trader
process exited. CEO then chose to sell 1x 746C (covered by the 742C), giving
742C +1 / 746C -1 / 750C +1 -- defined risk, $300 net debit, but with NO
babysitter process attached to close it.

These are physically settled 0DTE options: the 742C is ITM, so letting it
expire means a ~$74,200 share assignment this account cannot support. This
script closes all three legs at CLOSE_AT_ET, or earlier if run with --now.

    python qqq_orphan_close_20260923.py            # waits until 15:45 ET, then closes
    python qqq_orphan_close_20260923.py --now      # close immediately
"""
import argparse
import json
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from ib_insync import IB, Option

from alpaca_0dte_common import get_quote
from ibkr_0dte_common import ibkr_place_leg_with_ladder

ET = ZoneInfo("America/New_York")
CLOSE_AT_ET = (15, 45)
EXPIRY = "20260923"
LEGS = [("buy_back_short", 746.0, "BUY"), ("sell_long_742", 742.0, "SELL"), ("sell_long_750", 750.0, "SELL")]
CLIENT_ID = 1609


def telegram(text):
    try:
        import requests
        cfg = json.load(open("scanner_config.json"))
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"], "text": "QQQ orphan close: " + text}, timeout=10)
    except Exception as e:
        print(f"telegram failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", action="store_true")
    args = ap.parse_args()

    while not args.now:
        now = datetime.now(ET)
        if (now.hour, now.minute) >= CLOSE_AT_ET:
            break
        if now.hour >= 16:
            print("past the close -- nothing this script can do")
            telegram("RAN TOO LATE: past 16:00, position may have expired/assigned. CHECK IBKR.")
            sys.exit(1)
        time.sleep(20)

    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=CLIENT_ID, timeout=20)
    held = {p.contract.strike: p.position for p in ib.positions()
            if p.contract.secType == "OPT" and p.contract.symbol == "QQQ"
            and p.contract.lastTradeDateOrContractMonth == EXPIRY}
    print(f"{datetime.now(ET):%H:%M:%S} ET -- held: {held}")
    if not held:
        print("nothing open -- already flat")
        telegram("already flat, nothing to close")
        ib.disconnect()
        return

    results = {}
    for name, strike, action in LEGS:
        if not held.get(strike):
            continue
        c = Option("QQQ", EXPIRY, strike, "C", "SMART")
        ib.qualifyContracts(c)
        q = get_quote(ib, c)
        ok, px = ibkr_place_leg_with_ladder(ib, c, action, name, abs(int(held[strike])),
                                            q["bid"], q["ask"], q["mid"])
        results[name] = px if ok else None
        print(f"  {name}: {'FILLED @ $' + str(px) if ok else 'NOT FILLED'}")

    still = {p.contract.strike: p.position for p in ib.positions()
             if p.contract.secType == "OPT" and p.contract.symbol == "QQQ"
             and p.contract.lastTradeDateOrContractMonth == EXPIRY}
    ib.disconnect()
    msg = f"closed {results}; remaining {still or 'none'}"
    print(msg)
    telegram(("ALL CLOSED. " if not still else "STILL OPEN -- CHECK IBKR NOW. ") + msg)
    sys.exit(0 if not still else 2)


if __name__ == "__main__":
    main()
