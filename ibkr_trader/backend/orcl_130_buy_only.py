"""Single-leg test: BUY ORCL 130C only (close the short leg alone, per CEO request)."""
from ib_insync import IB, Option, LimitOrder
import time

ib = IB()
errors = []
ib.errorEvent += lambda reqId, code, msg, contract: errors.append(f"[{code}] {msg}")
ib.connect("127.0.0.1", 7496, clientId=1420, timeout=20)
print("Connected.")

contract = Option("ORCL", "20260821", 130.0, "C", "SMART")
ib.qualifyContracts(contract)
print(f"Qualified: {contract.localSymbol} conId={contract.conId}")

tkr = ib.reqMktData(contract, "", False, False)
ib.sleep(5)
bid, ask = tkr.bid, tkr.ask
mid = round((bid + ask) / 2, 2) if bid and ask and bid == bid and ask == ask else tkr.last
print(f"130C: bid={bid} ask={ask} mid={mid}")
ib.cancelMktData(contract)

price = round(mid * 1.05, 2)  # small concession above mid to help get filled
errors.clear()
order = LimitOrder("BUY", 1, price, tif="DAY")
order.transmit = True
trade = ib.placeOrder(contract, order)
ib.sleep(3)
print(f"Status: {trade.orderStatus.status}")
if trade.orderStatus.status == "Inactive":
    reason = "; ".join(errors) if errors else "(no error captured)"
    print(f"Inactive. IBKR said: {reason}")
    try:
        ib.cancelOrder(order)
    except Exception:
        pass
else:
    deadline = time.time() + 20
    while time.time() < deadline and trade.orderStatus.filled < 1:
        ib.sleep(1)
    if trade.orderStatus.filled >= 1:
        print(f"FILLED @ {trade.orderStatus.avgFillPrice}")
    else:
        print(f"Not filled, status={trade.orderStatus.status}")
        try:
            ib.cancelOrder(order)
        except Exception:
            pass

ib.disconnect()
print("Done.")
