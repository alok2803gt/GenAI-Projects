"""
Close script for the real PDD earnings condor entered 2026-08-21 15:44:15 ET
(EVC, via pre-trade review): short 95C/long 98C, short 82P/long 78P, exp
2026-08-28, net entry credit $0.97. CEO decision 2026-08-24: close now to
lock in the real +$61 unrealized gain rather than hold to expiry.

Order: SHORT legs first (short_call, short_put) -- covers the actual open
liability first, matching this account's established two-phase-close
discipline (same order used in close_goog_condor_20260819.py earlier
today). Long legs (98C, 78P) are both nearly worthless (~$0.04-0.05) --
selling them recovers little, but closing the full position is still the
clean, complete exit. Note: today's GOOG close hit a real Alpaca quirk on
a sell-to-close for an existing LONG position (misrouted as opening a new
cash-secured put) -- this script may hit the same thing on the long legs;
if so, they're cheap enough to just leave to expire rather than fight it,
same call made on the GOOG residual leg.

Usage: python close_pdd_condor.py [--dry-run]
"""
import argparse
import json
import sys

from ib_insync import IB, Option
from alpaca.trading.enums import OrderSide

from alpaca_0dte_common import load_config, alpaca_client, get_quote, place_leg_with_ladder, now_et

TICKER = "PDD"
EXPIRY_IBKR = "20260828"
TWS_PORT = 7496
LEGS = {
    "short_call": {"strike": 95.0, "right": "C", "alpaca_symbol": "PDD260828C00095000", "close_side": OrderSide.BUY},
    "short_put":  {"strike": 82.0, "right": "P", "alpaca_symbol": "PDD260828P00082000", "close_side": OrderSide.BUY},
    "long_call":  {"strike": 98.0, "right": "C", "alpaca_symbol": "PDD260828C00098000", "close_side": OrderSide.SELL},
    "long_put":   {"strike": 78.0, "right": "P", "alpaca_symbol": "PDD260828P00078000", "close_side": OrderSide.SELL},
}
ENTRY_CREDIT_TOTAL = 97.0


def evc_decisions_log(action, detail):
    pass  # PDD is tracked in EVC's own state, not a dedicated decisions file like GOOG


def oversight_log(summary, outcome):
    entry = {
        "time": now_et().isoformat(), "actor": "trader", "category": "position_closed",
        "summary": summary,
        "rationale": "CEO decision 2026-08-24: close the real PDD earnings condor to lock in the +$61 unrealized gain rather than hold to the 2026-08-28 expiry.",
        "outcome": outcome, "pnl_impact": None,
    }
    with open("oversight_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--client-id", type=int, default=1690)
    args = ap.parse_args()

    cfg = load_config()
    client = alpaca_client(cfg)

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=args.client_id, timeout=20)
    print("Connected to IBKR.")

    quotes = {}
    try:
        for name, leg in LEGS.items():
            c = Option(TICKER, EXPIRY_IBKR, leg["strike"], leg["right"], "SMART", "100", "USD")
            quotes[name] = get_quote(ib, c)
            print(f"  {name} {leg['strike']}{leg['right']}: bid={quotes[name]['bid']} ask={quotes[name]['ask']}")
    finally:
        ib.disconnect()
        print("Disconnected from IBKR (pricing done).")

    if any(not (q["bid"] and q["ask"]) for q in quotes.values()):
        print("ERROR: missing live bid/ask on one or more legs -- aborting, nothing closed.")
        oversight_log("PDD condor close aborted: missing live bid/ask on one or more legs", "Aborted, nothing closed.")
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN -- would close short_call, short_put, then long_call, long_put via ladder. No orders submitted.")
        return

    fills = {}
    for name in ("short_call", "short_put", "long_call", "long_put"):
        leg = LEGS[name]
        q = quotes[name]
        print(f"\n--- Closing {name} ({leg['close_side'].value}) ---")
        ok, px = place_leg_with_ladder(client, leg["alpaca_symbol"], leg["close_side"], name, 1, q["bid"], q["ask"], q["mid"])
        if not ok:
            msg = f"PDD condor close: {name} FAILED to fill. Fills so far: {fills}. "
            if name in ("long_call", "long_put"):
                msg += "This leg is a long option worth only a few dollars -- reasonable to leave it to expire rather than force the close further (same call made on the GOOG residual leg today)."
                print(msg)
                oversight_log(msg, f"Partial close: {fills}. Remaining leg(s) left to expire naturally.")
                print(json.dumps({"fills": fills, "note": "remaining leg(s) left to expire"}, indent=2))
                return
            else:
                msg += "This is a SHORT leg -- needs manual review NOW, not left unattended."
                print(msg)
                oversight_log(msg, "PAUSED -- short leg failed to close, needs manual review")
                sys.exit(1)
        fills[name] = px

    close_cost = (fills["short_call"] + fills["short_put"]) - (fills["long_call"] + fills["long_put"])
    realized_pnl = round(ENTRY_CREDIT_TOTAL - close_cost * 100, 2)
    msg = f"PDD earnings condor (2026-08-21 entry) CLOSED. Fills: {fills}. Realized P&L: ${realized_pnl:.2f}"
    print(msg)
    oversight_log(msg, f"Closed via repricing ladder, shorts first. Realized P&L ${realized_pnl:.2f}.")
    print(json.dumps({"fills": fills, "realized_pnl": realized_pnl}, indent=2))


if __name__ == "__main__":
    main()
