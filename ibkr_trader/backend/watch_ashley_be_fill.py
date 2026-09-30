"""Watch the $0.96 breakeven limit on the Ashley SPY 766P until it fills,
is cancelled, or 15:50 ET. On a fill, restart the Ashley executor so it
reconciles against real IBKR positions instead of a stale in-memory one."""
import json, subprocess, time
from datetime import datetime
from zoneinfo import ZoneInfo
from ib_insync import IB, Option

ET = ZoneInfo("America/New_York")


def telegram(text):
    try:
        import requests
        cfg = json.load(open("scanner_config.json"))
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"], "text": "ASHLEY 766P: " + text}, timeout=10)
    except Exception as e:
        print("telegram failed:", e)


ib = IB()
ib.connect("127.0.0.1", 7496, clientId=1660, timeout=20)
c = Option("SPY", "20260924", 766.0, "P", "SMART")
ib.qualifyContracts(c)
last_note = 0
while True:
    now = datetime.now(ET)
    if (now.hour, now.minute) >= (15, 50):
        print(f"{now:%H:%M:%S} stopping watch (15:50); the executor's force-close takes over")
        break
    ib.sleep(20)
    # ib.trades() only shows THIS client's orders -- the limit was placed from
    # another clientId, so ask for every client's orders and watch the position.
    oo = ib.reqAllOpenOrders()
    ib.sleep(2)
    working = [t for t in oo if t.contract.conId == c.conId and t.order.action == "SELL"]
    qty = sum(p.position for p in ib.positions() if p.contract.conId == c.conId)
    if qty == 0:
        px = next((t.orderStatus.avgFillPrice for t in working if t.orderStatus.avgFillPrice), 0.96)
        pnl = (px - 0.94) * 100 - 1.30
        msg = f"FILLED at ${px:.2f} -> P&L ${pnl:+.2f} (entry $0.94, both commissions in)"
        print(f"{now:%H:%M:%S} {msg}", flush=True)
        telegram(msg)
        subprocess.run(["launchctl", "kickstart", "-k", f"gui/{__import__('os').getuid()}/com.ibkrtrader.ashley"],
                       capture_output=True)
        print("ashley executor restarted -- it will reconcile to the real (now flat) position")
        break
    if not working:
        print(f"{now:%H:%M:%S} position still open ({qty}) but NO working sell order -- check IBKR")
        telegram(f"position open ({qty:g}) with no working exit order -- check IBKR")
        break
    if time.time() - last_note > 900:          # progress note every 15 min
        q = ib.reqMktData(c, "", False, False); ib.sleep(3)
        print(f"{now:%H:%M:%S} still working | bid {q.bid} ask {q.ask} (need 0.96)", flush=True)
        ib.cancelMktData(c)
        last_note = time.time()
ib.disconnect()
