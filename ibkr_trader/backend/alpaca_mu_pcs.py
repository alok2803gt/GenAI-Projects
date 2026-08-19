"""
MU put credit spread via Alpaca, sequential-leg (same proven mechanism as
COHR/CSCO/the LOW earnings condor completed 2026-08-18): BUY the long put
first (plain purchase, no margin concern), THEN SELL the short put (now
covered). Re-prices live right before committing rather than trusting the
scan done a few minutes earlier.

CEO-confirmed 2026-08-18 after being shown: (1) real capital fit -- ~49% of
Alpaca options buying power on one contract, well above this accounts usual
$200-400/trade sizing, and (2) real active event risk -- Netlist patent
litigation seeking a US import/sales ban on MU's DDR5 RDIMM/MRDIMM products,
driving IV to 75-107% across the chain (not ambient high vol).

Usage: python alpaca_mu_pcs.py
"""
import sys

from ib_insync import IB, Option, Stock
from alpaca.trading.enums import OrderSide

from alpaca_0dte_common import load_config, alpaca_client, get_quote, get_alpaca_symbols, place_leg, register_position, now_et

TICKER      = "MU"
EXPIRY_YF   = "2026-08-21"
EXPIRY_IBKR = "20260821"
SHORT_PUT   = 900.0
LONG_PUT    = 890.0
QTY         = 1
TWS_PORT    = 7496
CLIENT_ID   = 1596
MIN_CONSERVATIVE_CREDIT = 0.50  # real floor, not just >0 -- same discipline as spy_0dte_auto.py


def main():
    print(f"=== MU PCS: {SHORT_PUT}/{LONG_PUT}P, exp {EXPIRY_YF} ===")

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

        conservative_credit = round(short_q["bid"] - long_q["ask"], 2)
        long_limit  = round(long_q["ask"] - (long_q["ask"] - long_q["mid"]) * 0.40, 2)
        short_limit = round(short_q["bid"] + (short_q["mid"] - short_q["bid"]) * 0.40, 2)
        target_credit = round(short_limit - long_limit, 2)
        max_risk = round((SHORT_PUT - LONG_PUT) * 100 * QTY, 2)
        print(f"conservative credit (worst-fill): ${conservative_credit:.2f}  (threshold ${MIN_CONSERVATIVE_CREDIT:.2f})")
        print(f"target credit: ${target_credit:.2f}   max_risk: ${max_risk:.0f}")
        if conservative_credit < MIN_CONSERVATIVE_CREDIT:
            print(f"ERROR: conservative credit ${conservative_credit:.2f} < ${MIN_CONSERVATIVE_CREDIT:.2f} -- aborting, no order placed.")
            sys.exit(1)
    finally:
        ib.disconnect()
        print("Disconnected from IBKR (pricing done).")

    cfg = load_config()
    client = alpaca_client(cfg)
    put_by_strike, _ = get_alpaca_symbols(client, TICKER, EXPIRY_YF, LONG_PUT - 1, SHORT_PUT + 1, SHORT_PUT + 1, SHORT_PUT + 1)
    if SHORT_PUT not in put_by_strike or LONG_PUT not in put_by_strike:
        print(f"ERROR: strikes not listed on Alpaca: {[k for k in (SHORT_PUT, LONG_PUT) if k not in put_by_strike]}")
        sys.exit(1)
    short_sym = put_by_strike[SHORT_PUT]
    long_sym  = put_by_strike[LONG_PUT]
    print(f"Alpaca symbols: long={long_sym}  short={short_sym}")

    print(f"\n--- Leg 1/2: BUY long put ---")
    ok1, long_fill = place_leg(client, long_sym, OrderSide.BUY, long_limit, "long put", QTY)
    if not ok1:
        print("Long put did not fill -- aborting, nothing else committed.")
        sys.exit(1)

    print(f"--- Leg 2/2: SELL short put (covered by long put) ---")
    ok2, short_fill = place_leg(client, short_sym, OrderSide.SELL, short_limit, "short put", QTY)
    if not ok2:
        print("Short put did NOT fill -- you are holding the long put NAKED. Manual attention needed.")
        sys.exit(1)

    net_credit = round(short_fill - long_fill, 2)
    entry_time = now_et()
    pos_id = f"MU_pcs_{entry_time.strftime('%Y%m%d_%H%M%S')}"
    register_position(
        pos_id, TICKER, "put_credit_spread", [
            {"leg": "long_put", "strike": LONG_PUT, "symbol": long_sym, "fill": long_fill},
            {"leg": "short_put", "strike": SHORT_PUT, "symbol": short_sym, "fill": short_fill},
        ],
        net_credit, QTY, max_risk, None, None, entry_time.isoformat(),
        notes="MU 4% cushion PCS, CEO-confirmed after being shown real capital-fit and active Netlist litigation event-risk concerns.",
    )
    print(f"\nENTERED. pos_id={pos_id} net_credit=${net_credit:.2f} (${net_credit*QTY*100:.0f} total) max_risk=${max_risk:.0f}")


if __name__ == "__main__":
    main()
