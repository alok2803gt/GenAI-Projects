"""
Close all EVC phantom option positions (exp 7/17) at market open 7/15.
Targets ONLY options expiring 20260717 — leaves stocks and LEAPs untouched.
Waits until 9:31 AM ET before placing orders.
Uses ib_insync synchronous API with ib.sleep() to keep connection alive.
"""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from ib_insync import IB, Option, Order

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
TARGET_EXPIRY = "20260717"
CLIENT_ID = 92


def wait_for_market_open(ib: IB):
    """Block (keeping IB heartbeat alive) until 9:31 AM ET."""
    while True:
        now = datetime.now(ET)
        t = now.strftime("%H:%M")
        if t >= "09:31":
            log.info("Market open — proceeding")
            return
        remaining = (9 * 60 + 31) - (now.hour * 60 + now.minute)
        log.info("Waiting for 9:31 AM ET … %dh %02dm remaining  (now %s ET)",
                 remaining // 60, remaining % 60, t)
        ib.sleep(60)   # keeps the IB event loop ticking


def close_all(ib: IB):
    portfolio = list(ib.portfolio())
    targets = [
        p for p in portfolio
        if p.position != 0
        and getattr(p.contract, "secType", "") == "OPT"
        and getattr(p.contract, "lastTradeDateOrContractMonth", "")[:8] == TARGET_EXPIRY
    ]

    if not targets:
        log.info("No %s option positions found — nothing to close.", TARGET_EXPIRY)
        return

    targets.sort(key=lambda x: (x.contract.symbol, x.contract.right, x.contract.strike))
    log.info("Found %d legs to close:", len(targets))
    for p in targets:
        c = p.contract
        action = "SELL" if p.position > 0 else "BUY"
        log.info("  %s %s%s  pos=%+.0f  -> %s",
                 c.symbol, c.right, int(c.strike), p.position, action)

    # Place MKT orders
    submitted = []
    for p in targets:
        c = p.contract
        qty = int(abs(p.position))
        action = "SELL" if p.position > 0 else "BUY"

        contract = Option(
            c.symbol,
            c.lastTradeDateOrContractMonth[:8],
            float(c.strike),
            c.right,
            "SMART", "100", "USD",
        )
        contract.conId = c.conId

        order = Order()
        order.orderType = "MKT"
        order.action = action
        order.totalQuantity = qty
        order.tif = "DAY"

        try:
            trade = ib.placeOrder(contract, order)
            submitted.append((trade, c.symbol, c.right, c.strike, action, qty))
            log.info("Placed %s MKT x%d  %s %s%s  orderId=%s",
                     action, qty, c.symbol, c.right, int(c.strike),
                     trade.order.orderId)
        except Exception as exc:
            log.error("Failed %s %s%s: %s", c.symbol, c.right, c.strike, exc)

    if not submitted:
        log.warning("No orders placed.")
        return

    log.info("Waiting up to 5 min for %d orders to fill…", len(submitted))
    _TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}

    for tick in range(60):
        ib.sleep(5)
        statuses = [t.orderStatus.status for t, *_ in submitted]
        n_done = sum(1 for s in statuses if s in _TERMINAL)
        log.info("  [%ds] %d/%d terminal", (tick + 1) * 5, n_done, len(submitted))
        if all(s in _TERMINAL for s in statuses):
            break

    # Summary
    log.info("\n=== CLOSE SUMMARY ===")
    for trade, sym, right, strike, action, qty in submitted:
        status  = trade.orderStatus.status
        fill_px = float(trade.orderStatus.avgFillPrice or 0)
        filled  = float(trade.orderStatus.filled or 0)
        net = fill_px * filled * 100 * (-1 if action == "BUY" else 1)
        log.info("  %s %s%s: %-12s fill=%.4f  qty=%.0f  net=$%+.2f",
                 sym, right, int(strike), status, fill_px, filled, net)
    log.info("=== DONE ===")


def main():
    ib = IB()
    log.info("Connecting to IBKR paper TWS (port 7497, clientId=%d)…", CLIENT_ID)
    ib.connect("127.0.0.1", 7497, clientId=CLIENT_ID)
    log.info("Connected. Waiting for market open…")
    try:
        log.info("Market already open — proceeding immediately")
        close_all(ib)
    finally:
        ib.disconnect()
        log.info("Disconnected.")


if __name__ == "__main__":
    main()
