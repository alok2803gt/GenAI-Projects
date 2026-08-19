"""
One-off test: place a single put credit spread on Alpaca, priced off live IBKR
quotes (per the CEO's architecture decision -- data stays on IBKR, orders route
through Alpaca to sidestep the PDT rule IBKR hasn't retired yet).

Connects to IBKR directly (own clientId, not through the backend's HTTP API) --
the backend's ad-hoc option-quote endpoints (/market/options/{ticker}) wrap
_live_options_coro, which is a known open bug (task 2026-08-10-005): it hangs
indefinitely for any contract without an already-active streaming subscription.
Mirrors the same direct-connection pattern already proven working in this
session's other one-off scripts (orcl_combo_close.py, soxl_live_quotes_v2.py).

This validates the execution pathway end-to-end before AutoTrader's own Alpaca
integration (task 2026-08-11-002) gets built: real IBKR pricing -> real Alpaca
contract symbols -> real Alpaca multi-leg order -> real fill.

Usage: python alpaca_spread_test.py
Requires regular market hours for clean bid/ask on both IBKR and Alpaca --
pre-market quotes are thin/stale/zero on both sides.
"""
import json
import sys

from ib_insync import IB, Option, Stock
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    ContractType, OrderClass, OrderSide, OrderType, PositionIntent, TimeInForce,
)
from alpaca.trading.requests import (
    GetOptionContractsRequest, LimitOrderRequest, OptionLegRequest,
)

TICKER      = "CSCO"
EXPIRY_YF   = "2026-08-14"     # yfinance/Alpaca format (YYYY-MM-DD)
EXPIRY_IBKR = "20260814"       # IBKR format (YYYYMMDD)
WING_MULT   = 1.5              # matches EVC's proven wing_mult
QTY         = 1                # one contract -- this is a pipeline test, not a position
TWS_PORT    = 7496             # live account
CLIENT_ID   = 1550             # unused by backend (3) or any other one-off script this session

with open("scanner_config.json") as f:
    cfg = json.load(f)


def safe_px(v):
    if v is None:
        return None
    try:
        f = float(v)
        return f if f > 0 and f == f else None   # f==f rejects NaN
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
    print(f"=== Alpaca spread test: {TICKER} {EXPIRY_YF} put credit spread ===")

    ib = IB()
    errors = []
    ib.errorEvent += lambda reqId, code, msg, contract: errors.append(f"[{code}] {msg}")
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    print("Connected to IBKR.")

    try:
        stk = Stock(TICKER, "SMART", "USD")
        spot_q = get_quote(ib, stk)
        spot = spot_q["mid"] or spot_q["bid"] or spot_q["ask"]
        if not spot:
            print("ERROR: no live stock quote -- market likely closed. Aborting.")
            sys.exit(1)
        print(f"spot: ${spot:.2f}")

        atm = round(spot)
        atm_call_q = get_quote(ib, Option(TICKER, EXPIRY_IBKR, atm, "C", "SMART"))
        atm_put_q  = get_quote(ib, Option(TICKER, EXPIRY_IBKR, atm, "P", "SMART"))
        if not (atm_call_q["mid"] and atm_put_q["mid"]):
            print("ERROR: no live ATM option quotes -- market likely closed. Aborting.")
            sys.exit(1)
        straddle = atm_call_q["mid"] + atm_put_q["mid"]
        im_pct   = straddle / spot
        print(f"implied move: {im_pct*100:.1f}% (straddle ${straddle:.2f})")

        short_k = round(spot - im_pct * spot)
        long_k  = round(spot - WING_MULT * im_pct * spot)
        width   = short_k - long_k
        print(f"short put ${short_k} / long put ${long_k}  (width ${width})")
        if width <= 0:
            print("ERROR: invalid strikes computed. Aborting.")
            sys.exit(1)

        short_q = get_quote(ib, Option(TICKER, EXPIRY_IBKR, short_k, "P", "SMART"))
        long_q  = get_quote(ib, Option(TICKER, EXPIRY_IBKR, long_k, "P", "SMART"))
        if not (short_q["bid"] and short_q["ask"] and long_q["bid"] and long_q["ask"]):
            print("ERROR: no live bid/ask on one or both legs -- aborting.")
            sys.exit(1)

        s_mid = short_q["mid"]
        l_mid = long_q["mid"]
        target_credit = round(
            (s_mid - (s_mid - short_q["bid"]) * 0.40) - (l_mid + (long_q["ask"] - l_mid) * 0.40), 2
        )
        conservative_credit = round(short_q["bid"] - long_q["ask"], 2)
        print(f"target credit: ${target_credit:.2f}  (conservative worst-fill: ${conservative_credit:.2f})")
        if conservative_credit <= 0:
            print("ERROR: no positive conservative credit -- aborting.")
            sys.exit(1)
    finally:
        ib.disconnect()
        print("Disconnected from IBKR.")

    # ── Real Alpaca contract symbols for these exact strikes ───────────────
    client = TradingClient(
        cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
        paper=False, url_override=cfg["alpaca_base_url"],
    )
    contracts = client.get_option_contracts(GetOptionContractsRequest(
        underlying_symbols=[TICKER],
        expiration_date=EXPIRY_YF,
        type=ContractType.PUT,
        strike_price_gte=str(long_k - 1),
        strike_price_lte=str(short_k + 1),
    )).option_contracts
    by_strike = {float(c.strike_price): c for c in contracts}
    if short_k not in by_strike or long_k not in by_strike:
        print(f"ERROR: strikes {short_k}/{long_k} not both listed on Alpaca. "
              f"Available: {sorted(by_strike.keys())}")
        sys.exit(1)
    short_c = by_strike[short_k]
    long_c  = by_strike[long_k]
    print(f"Alpaca symbols: SELL {short_c.symbol} / BUY {long_c.symbol}")

    # ── Place the 2-leg multi-leg (MLEG) order on Alpaca ────────────────────
    order = LimitOrderRequest(
        qty=QTY,
        order_class=OrderClass.MLEG,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        limit_price=target_credit,
        legs=[
            OptionLegRequest(symbol=short_c.symbol, ratio_qty=1, side=OrderSide.SELL),
            OptionLegRequest(symbol=long_c.symbol, ratio_qty=1, side=OrderSide.BUY),
        ],
    )
    print(f"\nSubmitting: SELL {QTY}x {short_c.symbol} / BUY {QTY}x {long_c.symbol} "
          f"@ net credit ${target_credit:.2f}...")
    result = client.submit_order(order)
    print(f"Order ID: {result.id}  status: {result.status}")
    print(json.dumps({
        "ticker": TICKER, "expiry": EXPIRY_YF, "short_strike": short_k, "long_strike": long_k,
        "qty": QTY, "target_credit": target_credit, "conservative_credit": conservative_credit,
        "alpaca_order_id": str(result.id), "status": str(result.status),
    }, indent=2))


if __name__ == "__main__":
    main()
