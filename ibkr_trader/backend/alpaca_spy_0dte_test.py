"""
One-off test: place a single SPY 0DTE iron condor on Alpaca.

Why SPY, not SPX: IBKR's SPX 0DTE strategy is configured for a $500/contract
max risk (5pt width x $100 mult) which itself has never once filled in 35
live attempts (task 2026-08-08-002, root cause still unresolved -- not
confirmed to be PDT). Rather than port an unproven, oversized strategy to a
new broker, the CEO chose SPY: a standard, hyper-liquid equity-style 0DTE
option (unlike SPX/XSP's index-option mechanics), already proven tradable
via Alpaca (see alpaca_spread_test.py's CSCO run), with open interest in the
1,000s-9,000s per strike today -- far deeper than anything else tested this
session.

Same direct-IBKR-for-pricing / Alpaca-for-execution split as the CSCO test.
Sizing is NOT a raw port of SPX's 5pt width -- proportional to SPY's price
level and, more importantly, to this account's real ~$1,500 capital: a $2
wide condor caps max risk at ~$200/contract per side, not $500.

Usage: python alpaca_spy_0dte_test.py
Requires regular market hours for clean bid/ask on both IBKR and Alpaca.
"""
import json
import sys
from datetime import date

from ib_insync import IB, Option, Index
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (
    ContractType, OrderClass, OrderSide, OrderType, PositionIntent, TimeInForce,
)
from alpaca.trading.requests import (
    GetOptionContractsRequest, LimitOrderRequest, OptionLegRequest,
)

TICKER      = "SPY"
OTM_PCT     = 0.005    # 0.5% OTM shorts, matches the existing SPX 0DTE config
WIDTH       = 2.0      # $2 wide -- proportional to SPY price + this account's capital,
                        # NOT a raw port of SPX's 5pt ($500/contract) width
QTY         = 1
TWS_PORT    = 7496
CLIENT_ID   = 1551     # distinct from alpaca_spread_test.py's 1550

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


def nearest_strike(target: float) -> float:
    return round(target)   # SPY has $1-wide strikes -- confirmed via Alpaca contract lookup


def main():
    today_ibkr = date.today().strftime("%Y%m%d")
    today_alp  = date.today().strftime("%Y-%m-%d")
    print(f"=== Alpaca SPY 0DTE iron condor test: expiry {today_alp} ===")

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    print("Connected to IBKR.")

    try:
        from ib_insync import Stock
        spy_stk = Stock(TICKER, "SMART", "USD")
        spot_q = get_quote(ib, spy_stk)
        spot = spot_q["mid"] or spot_q["bid"] or spot_q["ask"]
        if not spot:
            print("ERROR: no live SPY quote -- market likely closed. Aborting.")
            sys.exit(1)
        print(f"SPY spot: ${spot:.2f}")

        short_put_k  = nearest_strike(spot * (1 - OTM_PCT))
        long_put_k   = nearest_strike(short_put_k - WIDTH)
        short_call_k = nearest_strike(spot * (1 + OTM_PCT))
        long_call_k  = nearest_strike(short_call_k + WIDTH)
        print(f"Condor: {long_put_k}P / {short_put_k}P  ...  {short_call_k}C / {long_call_k}C")

        legs = {
            "short_put":  Option(TICKER, today_ibkr, short_put_k,  "P", "SMART"),
            "long_put":   Option(TICKER, today_ibkr, long_put_k,   "P", "SMART"),
            "short_call": Option(TICKER, today_ibkr, short_call_k, "C", "SMART"),
            "long_call":  Option(TICKER, today_ibkr, long_call_k,  "C", "SMART"),
        }
        quotes = {name: get_quote(ib, c) for name, c in legs.items()}
        for name, q in quotes.items():
            print(f"  {name}: bid={q['bid']} ask={q['ask']}")
        if any(not (q["bid"] and q["ask"]) for q in quotes.values()):
            print("ERROR: missing live bid/ask on one or more legs -- aborting.")
            sys.exit(1)

        # Conservative credit (worst realistic fill): sell at bid, buy at ask
        conservative_credit = round(
            (quotes["short_put"]["bid"] + quotes["short_call"]["bid"])
            - (quotes["long_put"]["ask"] + quotes["long_call"]["ask"]), 2
        )
        # Target: 40%-into-mid on each leg, same convention as every other entry this session
        def target_px(q, is_short):
            mid = q["mid"]
            return (mid - (mid - q["bid"]) * 0.40) if is_short else (mid + (q["ask"] - mid) * 0.40)
        target_credit = round(
            (target_px(quotes["short_put"], True) + target_px(quotes["short_call"], True))
            - (target_px(quotes["long_put"], False) + target_px(quotes["long_call"], False)), 2
        )
        print(f"target credit: ${target_credit:.2f}  (conservative worst-fill: ${conservative_credit:.2f})")
        print(f"max risk if breached: ~${WIDTH*100 - target_credit*100:.0f}/contract")
        if conservative_credit <= 0:
            print("ERROR: no positive conservative credit -- aborting.")
            sys.exit(1)
    finally:
        ib.disconnect()
        print("Disconnected from IBKR.")

    # ── Real Alpaca contract symbols for all 4 legs ─────────────────────────
    client = TradingClient(
        cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
        paper=False, url_override=cfg["alpaca_base_url"],
    )
    puts = client.get_option_contracts(GetOptionContractsRequest(
        underlying_symbols=[TICKER], expiration_date=today_alp, type=ContractType.PUT,
        strike_price_gte=str(long_put_k - 1), strike_price_lte=str(short_put_k + 1),
    )).option_contracts
    calls = client.get_option_contracts(GetOptionContractsRequest(
        underlying_symbols=[TICKER], expiration_date=today_alp, type=ContractType.CALL,
        strike_price_gte=str(short_call_k - 1), strike_price_lte=str(long_call_k + 1),
    )).option_contracts
    put_by_strike  = {float(c.strike_price): c for c in puts}
    call_by_strike = {float(c.strike_price): c for c in calls}

    missing = [k for k in (short_put_k, long_put_k) if k not in put_by_strike] + \
              [k for k in (short_call_k, long_call_k) if k not in call_by_strike]
    if missing:
        print(f"ERROR: strikes not listed on Alpaca: {missing}")
        sys.exit(1)

    sp_c, lp_c = put_by_strike[short_put_k], put_by_strike[long_put_k]
    sc_c, lc_c = call_by_strike[short_call_k], call_by_strike[long_call_k]
    print(f"Alpaca symbols: SELL {sp_c.symbol} / BUY {lp_c.symbol} / "
          f"SELL {sc_c.symbol} / BUY {lc_c.symbol}")

    order = LimitOrderRequest(
        qty=QTY,
        order_class=OrderClass.MLEG,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        limit_price=target_credit,
        legs=[
            OptionLegRequest(symbol=sp_c.symbol, ratio_qty=1, side=OrderSide.SELL,
                              position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=lp_c.symbol, ratio_qty=1, side=OrderSide.BUY,
                              position_intent=PositionIntent.BUY_TO_OPEN),
            OptionLegRequest(symbol=sc_c.symbol, ratio_qty=1, side=OrderSide.SELL,
                              position_intent=PositionIntent.SELL_TO_OPEN),
            OptionLegRequest(symbol=lc_c.symbol, ratio_qty=1, side=OrderSide.BUY,
                              position_intent=PositionIntent.BUY_TO_OPEN),
        ],
    )
    print(f"\nSubmitting 4-leg iron condor @ net credit ${target_credit:.2f}...")
    result = client.submit_order(order)
    print(f"Order ID: {result.id}  status: {result.status}")
    print(json.dumps({
        "ticker": TICKER, "expiry": today_alp,
        "short_put": short_put_k, "long_put": long_put_k,
        "short_call": short_call_k, "long_call": long_call_k,
        "qty": QTY, "target_credit": target_credit, "conservative_credit": conservative_credit,
        "alpaca_order_id": str(result.id), "status": str(result.status),
    }, indent=2))


if __name__ == "__main__":
    main()
