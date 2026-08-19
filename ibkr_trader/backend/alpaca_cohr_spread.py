"""
COHR earnings iron condor via Alpaca, placed as 4 sequential single-leg
orders (long legs first, then short legs) -- MLEG combo orders reject on
this account for an unresolved reason (task 2026-08-12-003); sequential
legs are the proven working path (see the CSCO trade earlier today).

Context: COHR reports earnings tonight AH (2026-08-12 16:00 ET). CEO
approved this over CBRS after comparing both on real liquidity/credit
data -- COHR held a consistent ~25-27% credit/width ratio across widths
with real open interest both sides; CBRS was thin on puts and produced a
NEGATIVE credit at tight width (task 2026-08-12-00x, oversight_log.jsonl).
$10-wide chosen: max risk ~$730 against $1,635 real Alpaca options buying
power, leaving a real buffer alongside the existing CSCO position.

Order: BUY long_put, BUY long_call FIRST (defined-risk legs, plain
purchases, no margin concern) -- THEN SELL short_put, SELL short_call
(now each is protected by its long leg). Never sell a short leg before
its long leg has filled. Mirrors _evc_place_condor's proven sequencing
on IBKR, applied here to Alpaca.

Usage: python alpaca_cohr_spread.py
"""
import json
import sys
import time

from ib_insync import IB, Option, Stock
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest

TICKER      = "COHR"
EXPIRY_YF   = "2026-08-14"
EXPIRY_IBKR = "20260814"
SHORT_PUT   = 320.0
LONG_PUT    = 310.0
SHORT_CALL  = 400.0
LONG_CALL   = 410.0
QTY         = 1
TWS_PORT    = 7496
CLIENT_ID   = 1554
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


def place_leg(client, symbol, side, limit_price, label):
    print(f"Submitting {label}: {side.value} 1x {symbol} @ ${limit_price}...")
    order = client.submit_order(LimitOrderRequest(
        symbol=symbol, qty=QTY, side=side,
        type=OrderType.LIMIT, time_in_force=TimeInForce.DAY, limit_price=limit_price,
    ))
    for _ in range(FILL_WAIT_S):
        time.sleep(1)
        chk = client.get_order_by_id(order.id)
        if str(chk.status) == "OrderStatus.FILLED":
            print(f"  FILLED @ ${chk.filled_avg_price}")
            return True, chk.filled_avg_price
        if str(chk.status) in ("OrderStatus.REJECTED", "OrderStatus.CANCELED"):
            print(f"  {chk.status}")
            return False, None
    try:
        client.cancel_order_by_id(order.id)
    except Exception:
        pass
    print("  did not fill in time -- cancelled")
    return False, None


def main():
    print(f"=== COHR iron condor via Alpaca: {SHORT_PUT}/{LONG_PUT}P, {SHORT_CALL}/{LONG_CALL}C, {EXPIRY_YF} ===")

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    try:
        spot_q = get_quote(ib, Stock(TICKER, "SMART", "USD"))
        spot = spot_q["mid"] or spot_q["bid"] or spot_q["ask"]
        print(f"spot: ${spot:.2f}" if spot else "spot: unavailable")

        legs = {
            "long_put":   Option(TICKER, EXPIRY_IBKR, LONG_PUT,   "P", "SMART"),
            "short_put":  Option(TICKER, EXPIRY_IBKR, SHORT_PUT,  "P", "SMART"),
            "long_call":  Option(TICKER, EXPIRY_IBKR, LONG_CALL,  "C", "SMART"),
            "short_call": Option(TICKER, EXPIRY_IBKR, SHORT_CALL, "C", "SMART"),
        }
        quotes = {name: get_quote(ib, c) for name, c in legs.items()}
        for name, q in quotes.items():
            print(f"  {name}: bid={q['bid']} ask={q['ask']}")
        if any(not (q["bid"] and q["ask"]) for q in quotes.values()):
            print("ERROR: missing live bid/ask on one or more legs -- aborting.")
            sys.exit(1)

        conservative_credit = round(
            (quotes["short_put"]["bid"] + quotes["short_call"]["bid"])
            - (quotes["long_put"]["ask"] + quotes["long_call"]["ask"]), 2
        )
        print(f"conservative credit (worst-fill): ${conservative_credit:.2f}")
        if conservative_credit <= 0:
            print("ERROR: no positive conservative credit -- aborting.")
            sys.exit(1)

        def target_px(q, is_short):
            mid = q["mid"]
            return round((mid - (mid - q["bid"]) * 0.40) if is_short else (mid + (q["ask"] - mid) * 0.40), 2)

        limits = {name: target_px(q, name.startswith("short")) for name, q in quotes.items()}
        for name, px in limits.items():
            print(f"  {name} limit: ${px}")
    finally:
        ib.disconnect()
        print("Disconnected from IBKR.")

    client = TradingClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
                            paper=False, url_override=cfg["alpaca_base_url"])

    # Real, verified Alpaca contract symbols -- looked up, not hand-built
    puts = client.get_option_contracts(GetOptionContractsRequest(
        underlying_symbols=[TICKER], expiration_date=EXPIRY_YF, type=ContractType.PUT,
        strike_price_gte=str(LONG_PUT - 1), strike_price_lte=str(SHORT_PUT + 1),
    )).option_contracts
    calls = client.get_option_contracts(GetOptionContractsRequest(
        underlying_symbols=[TICKER], expiration_date=EXPIRY_YF, type=ContractType.CALL,
        strike_price_gte=str(SHORT_CALL - 1), strike_price_lte=str(LONG_CALL + 1),
    )).option_contracts
    put_by_strike = {float(c.strike_price): c for c in puts}
    call_by_strike = {float(c.strike_price): c for c in calls}
    missing = [k for k in (LONG_PUT, SHORT_PUT) if k not in put_by_strike] + \
              [k for k in (SHORT_CALL, LONG_CALL) if k not in call_by_strike]
    if missing:
        print(f"ERROR: strikes not listed on Alpaca: {missing}")
        sys.exit(1)
    sym = {
        "long_put":   put_by_strike[LONG_PUT].symbol,
        "short_put":  put_by_strike[SHORT_PUT].symbol,
        "long_call":  call_by_strike[LONG_CALL].symbol,
        "short_call": call_by_strike[SHORT_CALL].symbol,
    }
    print(f"Alpaca symbols: {sym}")

    print("\n--- Leg 1/4: BUY long put (defined-risk, no margin concern) ---")
    ok, _ = place_leg(client, sym["long_put"], OrderSide.BUY, limits["long_put"], "long put")
    if not ok:
        print("Long put did not fill -- aborting before any short leg goes on.")
        sys.exit(1)

    print("\n--- Leg 2/4: BUY long call (defined-risk, no margin concern) ---")
    ok, _ = place_leg(client, sym["long_call"], OrderSide.BUY, limits["long_call"], "long call")
    if not ok:
        print("Long call did not fill -- long put is now a naked long position. Manual attention needed.")
        sys.exit(1)

    print("\n--- Leg 3/4: SELL short put (now covered by long put) ---")
    ok, _ = place_leg(client, sym["short_put"], OrderSide.SELL, limits["short_put"], "short put")
    if not ok:
        print("Short put did not fill -- holding both longs uncovered on the short side. Manual attention needed.")

    print("\n--- Leg 4/4: SELL short call (now covered by long call) ---")
    ok, _ = place_leg(client, sym["short_call"], OrderSide.SELL, limits["short_call"], "short call")
    if not ok:
        print("Short call did not fill -- check position manually.")

    print("\nDone. Verify final position via Alpaca get_all_positions().")


if __name__ == "__main__":
    main()
