"""
META bull put credit spread via Alpaca, placed as two SEQUENTIAL single-leg
orders instead of a combo -- mirrors the proven pattern from
alpaca_spread_test_sequential.py (CSCO) and alpaca_cohr_spread.py (COHR),
used here because IBKR's own BAG-combo mechanism (Manual Trader
/manual-trader/enter) failed to fill this exact spread on 2 consecutive
real attempts today (see oversight_log.jsonl, secretary_tasks.json task
2026-08-13-002) -- trying a genuinely different execution path, not
retrying the same broken one.

Order: BUY the protective long put first (plain purchase, no margin
concern), THEN SELL the short put (now covered by the long).

Usage: python alpaca_meta_spread.py
"""
import json
import sys
import time

from ib_insync import IB, Option, Stock
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest

TICKER      = "META"
EXPIRY_YF   = "2026-09-18"
EXPIRY_IBKR = "20260918"
SHORT_PUT   = 570.0
LONG_PUT    = 565.0
QTY         = 1
TWS_PORT    = 7496
CLIENT_ID   = 1583
FILL_WAIT_S = 20

with open("scanner_config.json") as f:
    cfg = json.load(f)


def safe_px(v):
    if v is None:
        return None
    try:
        f = float(v)
        return f if f > 0 and f == f else None
    except (TypeError, ValueError):
        return None


def get_quote(ib: IB, contract) -> dict:
    ib.qualifyContracts(contract)
    if not contract.conId:
        raise ValueError(f"could not qualify {contract}")
    td = ib.reqMktData(contract, "", False, False)
    ib.sleep(3)
    bid, ask = safe_px(td.bid), safe_px(td.ask)
    ib.cancelMktData(contract)
    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2 if bid and ask else None}


def main():
    print(f"=== Alpaca SEQUENTIAL-leg: META {SHORT_PUT}/{LONG_PUT}P bull put spread, {EXPIRY_YF} ===")

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    print("Connected to IBKR.")

    try:
        spot_q = get_quote(ib, Stock(TICKER, "SMART", "USD"))
        spot = spot_q["mid"] or spot_q["bid"] or spot_q["ask"]
        print(f"spot: ${spot:.2f}" if spot else "spot: unavailable")

        long_q  = get_quote(ib, Option(TICKER, EXPIRY_IBKR, LONG_PUT,  "P", "SMART"))
        short_q = get_quote(ib, Option(TICKER, EXPIRY_IBKR, SHORT_PUT, "P", "SMART"))
        print(f"long put ${LONG_PUT}: bid={long_q['bid']} ask={long_q['ask']}")
        print(f"short put ${SHORT_PUT}: bid={short_q['bid']} ask={short_q['ask']}")
        if not (long_q["bid"] and long_q["ask"] and short_q["bid"] and short_q["ask"]):
            print("ERROR: missing live bid/ask on one or both legs. Aborting.")
            sys.exit(1)

        # Conservative credit (worst realistic fill): sell at bid, buy at ask
        conservative_credit = round(short_q["bid"] - long_q["ask"], 2)
        long_limit  = round(long_q["ask"] - (long_q["ask"] - long_q["mid"]) * 0.40, 2)   # buy: concede toward ask
        short_limit = round(short_q["bid"] + (short_q["mid"] - short_q["bid"]) * 0.40, 2)  # sell: concede toward bid
        target_credit = round(short_limit - long_limit, 2)
        print(f"target credit: ${target_credit:.2f}  (conservative worst-fill: ${conservative_credit:.2f})")
        print(f"long put limit: ${long_limit}   short put limit: ${short_limit}")
        if conservative_credit <= 0:
            print("ERROR: no positive conservative credit -- aborting.")
            sys.exit(1)
    finally:
        ib.disconnect()
        print("Disconnected from IBKR.")

    client = TradingClient(
        cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
        paper=False, url_override=cfg["alpaca_base_url"],
    )
    contracts = client.get_option_contracts(GetOptionContractsRequest(
        underlying_symbols=[TICKER], expiration_date=EXPIRY_YF, type=ContractType.PUT,
        strike_price_gte=str(LONG_PUT - 1), strike_price_lte=str(SHORT_PUT + 1),
    )).option_contracts
    by_strike = {float(c.strike_price): c for c in contracts}
    if SHORT_PUT not in by_strike or LONG_PUT not in by_strike:
        print(f"ERROR: strikes not listed on Alpaca: {[k for k in (SHORT_PUT, LONG_PUT) if k not in by_strike]}")
        sys.exit(1)
    short_sym = by_strike[SHORT_PUT].symbol
    long_sym  = by_strike[LONG_PUT].symbol
    print(f"Alpaca symbols: long={long_sym}  short={short_sym}")

    # ── Leg 1: BUY the protective long put first ──
    print(f"\nSubmitting leg 1/2: BUY {QTY}x {long_sym} @ ${long_limit}...")
    long_order = client.submit_order(LimitOrderRequest(
        symbol=long_sym, qty=QTY, side=OrderSide.BUY,
        type=OrderType.LIMIT, time_in_force=TimeInForce.DAY, limit_price=long_limit,
    ))
    print(f"Order ID: {long_order.id}  status: {long_order.status}")

    filled = False
    for _ in range(FILL_WAIT_S):
        time.sleep(1)
        chk = client.get_order_by_id(long_order.id)
        if str(chk.status) == "OrderStatus.FILLED":
            filled = True
            print(f"Leg 1 FILLED @ ${chk.filled_avg_price}")
            break
        if str(chk.status) in ("OrderStatus.REJECTED", "OrderStatus.CANCELED"):
            print(f"Leg 1 {chk.status} -- aborting, not proceeding to leg 2.")
            sys.exit(1)
    if not filled:
        print("Leg 1 did not fill within wait window -- aborting, not proceeding to leg 2.")
        try:
            client.cancel_order_by_id(long_order.id)
        except Exception:
            pass
        sys.exit(1)

    # ── Leg 2: SELL the short put (now protected by the filled long) ────────
    print(f"\nSubmitting leg 2/2: SELL {QTY}x {short_sym} @ ${short_limit}...")
    short_order = client.submit_order(LimitOrderRequest(
        symbol=short_sym, qty=QTY, side=OrderSide.SELL,
        type=OrderType.LIMIT, time_in_force=TimeInForce.DAY, limit_price=short_limit,
    ))
    print(f"Order ID: {short_order.id}  status: {short_order.status}")

    for _ in range(FILL_WAIT_S):
        time.sleep(1)
        chk = client.get_order_by_id(short_order.id)
        if str(chk.status) == "OrderStatus.FILLED":
            print(f"Leg 2 FILLED @ ${chk.filled_avg_price}")
            break
        if str(chk.status) in ("OrderStatus.REJECTED", "OrderStatus.CANCELED"):
            print(f"Leg 2 {chk.status} -- you are now holding the long put NAKED. Manual attention needed.")
            break
    else:
        print("Leg 2 still not filled after wait window -- check manually. Long leg is filled and held either way.")

    print(json.dumps({
        "ticker": TICKER, "expiry": EXPIRY_YF, "short_strike": SHORT_PUT, "long_strike": LONG_PUT,
        "long_order_id": str(long_order.id), "short_order_id": str(short_order.id),
    }, indent=2))


if __name__ == "__main__":
    main()
