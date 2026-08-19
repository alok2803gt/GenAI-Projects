"""
Close the ORCL bull call spread (120C long / 130C short, 20260821) as a
single BAG combo order -- sequential single-leg closes were rejected for
margin deficit (legs are individually worth thousands even though the net
position is only ~$588). Mirrors the exact BAG/ComboLeg construction
already proven working in _mt_place_order_coro (main.py) for opening
multi-leg positions, applied here to close: SELL 120C + BUY 130C.
"""
from ib_insync import IB, Option, Contract, ComboLeg, LimitOrder
import time

CLIENT_ID = 1310
TWS_PORT = 7496
TICKER = "ORCL"
EXPIRY = "20260821"
LONG_STRIKE = 120.0   # currently held BUY -- close via SELL
SHORT_STRIKE = 130.0  # currently held SELL -- close via BUY
REPRICE_STEPS = 4
REPRICE_WAIT_S = 15
MAX_CONCESSION_PCT = 0.20


def main():
    ib = IB()
    errors = []
    ib.errorEvent += lambda reqId, code, msg, contract: errors.append(f"[{code}] {msg}")
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    print("Connected.")

    long_leg = Option(TICKER, EXPIRY, LONG_STRIKE, "C", "SMART")
    short_leg = Option(TICKER, EXPIRY, SHORT_STRIKE, "C", "SMART")
    qualified = ib.qualifyContracts(long_leg, short_leg)
    for q in qualified:
        if not q.conId:
            raise SystemExit(f"Could not qualify {q}")
    print(f"Qualified: {long_leg.localSymbol} (conId={long_leg.conId}), "
          f"{short_leg.localSymbol} (conId={short_leg.conId})")

    # Real streaming quotes for both legs
    long_tkr = ib.reqMktData(long_leg, "", False, False)
    short_tkr = ib.reqMktData(short_leg, "", False, False)
    ib.sleep(6)

    def mid(tkr):
        bid = tkr.bid if tkr.bid and tkr.bid == tkr.bid and tkr.bid > 0 else None
        ask = tkr.ask if tkr.ask and tkr.ask == tkr.ask and tkr.ask > 0 else None
        if bid and ask:
            return round((bid + ask) / 2, 2)
        last = tkr.last if tkr.last and tkr.last == tkr.last and tkr.last > 0 else None
        return last

    long_mid = mid(long_tkr)
    short_mid = mid(short_tkr)
    print(f"120C long: bid={long_tkr.bid} ask={long_tkr.ask} mid={long_mid}")
    print(f"130C short: bid={short_tkr.bid} ask={short_tkr.ask} mid={short_mid}")
    ib.cancelMktData(long_leg)
    ib.cancelMktData(short_leg)

    if long_mid is None or short_mid is None:
        raise SystemExit("Could not get real quotes for both legs -- aborting, no order placed.")

    # Closing: SELL the long (120C), BUY back the short (130C).
    # Net credit = what we receive selling 120C - what we pay buying back 130C.
    net_credit = round(long_mid - short_mid, 2)
    print(f"Target net credit (mid-to-mid): {net_credit}")

    # Build BAG combo -- mirrors _mt_place_order_coro's proven construction
    bag = Contract()
    bag.symbol = TICKER
    bag.secType = "BAG"
    bag.currency = "USD"
    bag.exchange = "SMART"

    leg_sell = ComboLeg()
    leg_sell.conId = long_leg.conId
    leg_sell.ratio = 1
    leg_sell.action = "SELL"
    leg_sell.exchange = "SMART"
    leg_sell.openClose = 0
    leg_sell.shortSaleSlot = 0
    leg_sell.designatedLocation = ""
    leg_sell.exemptCode = -1

    leg_buy = ComboLeg()
    leg_buy.conId = short_leg.conId
    leg_buy.ratio = 1
    leg_buy.action = "BUY"
    leg_buy.exchange = "SMART"
    leg_buy.openClose = 0
    leg_buy.shortSaleSlot = 0
    leg_buy.designatedLocation = ""
    leg_buy.exemptCode = -1

    bag.comboLegs = [leg_sell, leg_buy]

    # 1 SELL leg + 1 BUY leg -> tied -> "SELL" per the same tie-break logic
    # already proven correct on this exact position's original entry.
    net_action = "SELL"

    price = net_credit
    per_step = max(round(abs(net_credit) * MAX_CONCESSION_PCT / REPRICE_STEPS, 2), 0.01)
    filled = None

    for step in range(REPRICE_STEPS + 1):
        errors.clear()
        order = LimitOrder(net_action, 1, price)
        order.tif = "DAY"
        order.transmit = True
        trade = ib.placeOrder(bag, order)
        ib.sleep(2)

        status = trade.orderStatus.status
        print(f"Step {step}: placed {net_action} combo @ {price} -> status={status}")
        if status == "Inactive":
            try:
                ib.cancelOrder(order)
            except Exception:
                pass
            reason = "; ".join(errors) if errors else "(no IBKR error captured)"
            print(f"  Inactive. IBKR said: {reason}")
            if step == REPRICE_STEPS:
                raise SystemExit(f"Final attempt Inactive: {reason}")
            price = round(price - per_step, 2)
            continue

        deadline = time.time() + REPRICE_WAIT_S
        while time.time() < deadline:
            ib.sleep(1)
            if trade.orderStatus.filled >= 1:
                filled = trade
                break
            if trade.orderStatus.status in ("Cancelled", "ApiCancelled", "Inactive"):
                break
        if filled:
            break

        try:
            ib.cancelOrder(order)
            ib.sleep(1)
        except Exception:
            pass
        price = round(price - per_step, 2)

    if filled:
        print(f"\nFILLED: {net_action} combo @ avgFillPrice={filled.orderStatus.avgFillPrice}")
        print(f"Real net credit realized: {filled.orderStatus.avgFillPrice}")
    else:
        print("\nNOT FILLED after all reprice steps. No position change -- still holding both legs.")

    ib.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
