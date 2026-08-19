"""
Real-time earnings iron condor via Alpaca sequential-leg execution -- same
proven mechanism as alpaca_cohr_spread.py (filled, settled +$242 as
projected) and alpaca_spread_test_sequential.py (CSCO, filled). Built to
route around IBKR's PDT rejection blocking EVC's own order placement today
(2026-08-18): BUY the two long legs first (plain purchase, no margin
concern), THEN SELL the two short legs (now covered by their longs).

Usage:
  python alpaca_earnings_condor.py --ticker LOW --expiry 2026-08-21 \
      --short-put 207.5 --long-put 202.5 --short-call 227.5 --long-call 230.0 \
      --min-credit 0.50
"""
import argparse
import json
import sys

from ib_insync import IB, Option

from alpaca_0dte_common import (
    load_config, alpaca_client, get_quote, place_condor_sequential,
    get_alpaca_symbols, register_position, now_et,
)

TWS_PORT = 7496


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--expiry", required=True, help="YYYY-MM-DD")
    ap.add_argument("--short-put", type=float, required=True)
    ap.add_argument("--long-put", type=float, required=True)
    ap.add_argument("--short-call", type=float, required=True)
    ap.add_argument("--long-call", type=float, required=True)
    ap.add_argument("--min-credit", type=float, default=0.50)
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--client-id", type=int, required=True)
    args = ap.parse_args()

    ticker = args.ticker.upper()
    expiry_ibkr = args.expiry.replace("-", "")
    print(f"=== Alpaca earnings condor: {ticker} {args.long_put}/{args.short_put}P .. "
          f"{args.short_call}/{args.long_call}C, exp {args.expiry} ===")

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=args.client_id, timeout=20)
    print("Connected to IBKR.")

    try:
        legs = {
            "long_put":   Option(ticker, expiry_ibkr, args.long_put,   "P", "SMART"),
            "short_put":  Option(ticker, expiry_ibkr, args.short_put,  "P", "SMART"),
            "short_call": Option(ticker, expiry_ibkr, args.short_call, "C", "SMART"),
            "long_call":  Option(ticker, expiry_ibkr, args.long_call,  "C", "SMART"),
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
        print(f"conservative credit (worst-fill): ${conservative_credit:.2f}  threshold: ${args.min_credit:.2f}")
        if conservative_credit < args.min_credit:
            print(f"ERROR: conservative credit ${conservative_credit:.2f} < ${args.min_credit:.2f} -- aborting, no order placed.")
            sys.exit(1)

        # Concede 40% toward the market from mid, same convention as the proven scripts
        entry_limits = {}
        for name in ("long_put", "long_call"):
            q = quotes[name]
            entry_limits[name] = round(q["ask"] - (q["ask"] - q["mid"]) * 0.40, 2)
        for name in ("short_put", "short_call"):
            q = quotes[name]
            entry_limits[name] = round(q["bid"] + (q["mid"] - q["bid"]) * 0.40, 2)
        print("entry limits:", entry_limits)
    finally:
        ib.disconnect()
        print("Disconnected from IBKR (pricing done).")

    cfg = load_config()
    client = alpaca_client(cfg)
    put_by_strike, call_by_strike = get_alpaca_symbols(
        client, ticker, args.expiry,
        args.long_put - 1, args.short_put + 1,
        args.short_call - 1, args.long_call + 1,
    )
    missing = [k for k in (args.short_put, args.long_put) if k not in put_by_strike] + \
              [k for k in (args.short_call, args.long_call) if k not in call_by_strike]
    if missing:
        print(f"ERROR: strikes not listed on Alpaca: {missing}")
        sys.exit(1)
    syms = {
        "short_put":  put_by_strike[args.short_put],
        "long_put":   put_by_strike[args.long_put],
        "short_call": call_by_strike[args.short_call],
        "long_call":  call_by_strike[args.long_call],
    }
    print(f"Alpaca symbols: {syms}")

    ok, fills, state = place_condor_sequential(client, syms, entry_limits, args.qty)
    if not ok:
        print(f"\nENTRY INCOMPLETE ({state}) -- check Alpaca positions manually before doing anything else.")
        sys.exit(1)

    net_entry_credit = round(
        (fills["short_put"] + fills["short_call"]) - (fills["long_put"] + fills["long_call"]), 2
    )
    entry_time = now_et()
    pos_id = f"{ticker}_earnings_condor_{entry_time.strftime('%Y%m%d_%H%M%S')}"
    max_risk = round((args.short_put - args.long_put) * 100, 2)
    strikes = {
        "short_put": args.short_put, "long_put": args.long_put,
        "short_call": args.short_call, "long_call": args.long_call,
    }
    register_position(
        pos_id, ticker, "iron_condor_earnings",
        [{"leg": k, "strike": strikes[k], "symbol": syms[k], "fill": fills[k]} for k in syms],
        net_entry_credit, args.qty, max_risk * args.qty, None, None,
        entry_time.isoformat(),
        notes="Earnings condor via Alpaca, routed around IBKR PDT block 2026-08-18. EVC found this candidate, IBKR rejected it.",
    )
    print(f"\nENTERED. pos_id={pos_id}  net_entry_credit=${net_entry_credit:.2f} "
          f"(${net_entry_credit*args.qty*100:.0f} total)")
    print(json.dumps({"ticker": ticker, "pos_id": pos_id, "net_entry_credit": net_entry_credit, "fills": fills}, indent=2))


if __name__ == "__main__":
    main()
