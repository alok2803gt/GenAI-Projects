"""
CSCO put credit spread on Alpaca, placed as two SEQUENTIAL single-leg orders
instead of one MLEG combo order.

Why: 3 straight MLEG order submissions today were rejected instantly (~5-8ms)
with no retrievable reason (see oversight_log.jsonl, task 2026-08-12-003) --
order construction matched Alpaca's own official MLEG example verbatim on the
third attempt and still failed. A plain single-symbol order (1 share F stock)
filled cleanly in seconds, isolating the problem to MLEG order acceptance
specifically, not the account/API/keys.

This mirrors the exact fix already proven on IBKR for this account's own EVC
strategy (_evc_place_condor, main.py): BAG/combo orders didn't fill for
multi-leg options there either, so it switched to sequential single-leg
orders -- BUY the protective long leg first (a plain purchase, no margin
concern), THEN SELL the short leg (now covered by the long, spread margin
instead of naked). Same logic applied here to Alpaca.

Usage: python alpaca_spread_test_sequential.py
Requires regular market hours for clean bid/ask.
"""
import json
import sys
import time

from ib_insync import IB, Option, Stock
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest

TICKER      = "CSCO"
EXPIRY_YF   = "2026-08-14"
EXPIRY_IBKR = "20260814"
WING_MULT   = 1.5
QTY         = 1
TWS_PORT    = 7496
CLIENT_ID   = 1552   # distinct from the other two scripts (1550, 1551)
FILL_WAIT_S = 20      # how long to wait for the first (long) leg to fill before trying the short

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
    print(f"=== Alpaca SEQUENTIAL-leg test: {TICKER} {EXPIRY_YF} put credit spread ===")

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    print("Connected to IBKR.")

    try:
        spot_q = get_quote(ib, Stock(TICKER, "SMART", "USD"))
        spot = spot_q["mid"] or spot_q["bid"] or spot_q["ask"]
        if not spot:
            print("ERROR: no live stock quote. Aborting.")
            sys.exit(1)
        print(f"spot: ${spot:.2f}")

        atm = round(spot)
        atm_call_q = get_quote(ib, Option(TICKER, EXPIRY_IBKR, atm, "C", "SMART"))
        atm_put_q  = get_quote(ib, Option(TICKER, EXPIRY_IBKR, atm, "P", "SMART"))
        if not (atm_call_q["mid"] and atm_put_q["mid"]):
            print("ERROR: no live ATM option quotes. Aborting.")
            sys.exit(1)
        im_pct = (atm_call_q["mid"] + atm_put_q["mid"]) / spot
        short_k = round(spot - im_pct * spot)
        long_k  = round(spot - WING_MULT * im_pct * spot)
        print(f"implied move {im_pct*100:.1f}%  ->  short put ${short_k} / long put ${long_k}")

        long_q  = get_quote(ib, Option(TICKER, EXPIRY_IBKR, long_k,  "P", "SMART"))
        short_q = get_quote(ib, Option(TICKER, EXPIRY_IBKR, short_k, "P", "SMART"))
        if not (long_q["bid"] and long_q["ask"] and short_q["bid"] and short_q["ask"]):
            print("ERROR: no live bid/ask on one or both legs. Aborting.")
            sys.exit(1)
        long_limit  = round(long_q["ask"] - (long_q["ask"] - long_q["mid"]) * 0.40, 2)   # buy: concede toward ask
        short_limit = round(short_q["bid"] + (short_q["mid"] - short_q["bid"]) * 0.40, 2)  # sell: concede toward bid
        print(f"long put ${long_k}: bid={long_q['bid']} ask={long_q['ask']} -> limit ${long_limit}")
        print(f"short put ${short_k}: bid={short_q['bid']} ask={short_q['ask']} -> limit ${short_limit}")
    finally:
        ib.disconnect()
        print("Disconnected from IBKR.")

    client = TradingClient(
        cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
        paper=False, url_override=cfg["alpaca_base_url"],
    )
    contracts = client.get_option_contracts(GetOptionContractsRequest(
        underlying_symbols=[TICKER], expiration_date=EXPIRY_YF, type=ContractType.PUT,
        strike_price_gte=str(long_k - 1), strike_price_lte=str(short_k + 1),
    )).option_contracts
    by_strike = {float(c.strike_price): c for c in contracts}
    if short_k not in by_strike or long_k not in by_strike:
        print(f"ERROR: strikes not listed on Alpaca: {[k for k in (short_k, long_k) if k not in by_strike]}")
        sys.exit(1)
    short_sym = by_strike[short_k].symbol
    long_sym  = by_strike[long_k].symbol
    print(f"Alpaca symbols: long={long_sym}  short={short_sym}")

    # ── Leg 1: BUY the protective long put first (plain purchase, no margin concern) ──
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
        print("Leg 1 did not fill within wait window -- aborting, not proceeding to leg 2 "
              "(never sell a naked short without its protective long already on).")
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
            print(f"Leg 2 {chk.status} -- you are now holding the long put NAKED "
                  f"(leg 1 filled, leg 2 did not). Manual attention needed.")
            break
    else:
        print("Leg 2 still not filled after wait window -- check manually. "
              "Long leg is filled and held either way (defined-risk direction, not naked).")

    print(json.dumps({
        "ticker": TICKER, "expiry": EXPIRY_YF, "short_strike": short_k, "long_strike": long_k,
        "long_order_id": str(long_order.id), "short_order_id": str(short_order.id),
    }, indent=2))


if __name__ == "__main__":
    main()
