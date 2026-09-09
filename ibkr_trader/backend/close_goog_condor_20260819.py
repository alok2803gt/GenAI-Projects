"""
One-time close script for the real GOOG weekly condor entered 2026-08-19
09:43 ET (pos_id GOOG_weekly_condor_20260819_094252), off the validated
Monday-only cycle (entered Wednesday). CEO decision 2026-08-24: close near
breakeven to free up GOOG concentration, then re-enter fresh through
today's Monday-cycle process (now with a real pre-trade review, 15% GOOG-
specific cap).

Real numbers at decision time: entry credit $87, cost to close ~$107 at
last marks (~-$20). Uses the same repricing-ladder execution validated on
the DE close (2026-08-20: favorable step filled at $10.10 vs $8.55 mid) --
real ladder fills are often better than the raw mark, so the actual
realized loss may be smaller than the mark-implied $20.

Order: SHORT legs first (short_call, short_put) -- covers the actual open
liability first, leaving only the defined-risk long legs exposed
mid-close if anything goes wrong, matching this account's established
two-phase-close discipline (cover shorts first). Long legs closed second.

Usage: python close_goog_condor_20260819.py [--dry-run]
"""
import argparse
import json
import sys

from ib_insync import IB, Option
from alpaca.trading.enums import OrderSide

from alpaca_0dte_common import load_config, alpaca_client, get_quote, place_leg_with_ladder, now_et

TICKER = "GOOG"
EXPIRY_IBKR = "20260828"
TWS_PORT = 7496
LEGS = {
    "short_call": {"strike": 355.0, "right": "C", "alpaca_symbol": "GOOG260828C00355000", "close_side": OrderSide.BUY},
    "short_put":  {"strike": 322.5, "right": "P", "alpaca_symbol": "GOOG260828P00322500", "close_side": OrderSide.BUY},
    "long_call":  {"strike": 360.0, "right": "C", "alpaca_symbol": "GOOG260828C00360000", "close_side": OrderSide.SELL},
    "long_put":   {"strike": 317.5, "right": "P", "alpaca_symbol": "GOOG260828P00317500", "close_side": OrderSide.SELL},
}
ENTRY_CREDIT_TOTAL = 87.0


def goog_condor_log(action, detail):
    entry = {"time": now_et().strftime("%Y-%m-%d %H:%M:%S ET"), "action": action, "detail": detail}
    try:
        with open("goog_condor_decisions.json") as f:
            decisions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        decisions = []
    decisions.append(entry)
    with open("goog_condor_decisions.json", "w") as f:
        json.dump(decisions[-200:], f, indent=2)


def oversight_log(summary, outcome):
    entry = {
        "time": now_et().isoformat(), "actor": "trader", "category": "position_closed",
        "summary": summary, "rationale": "CEO decision 2026-08-24: close existing off-cycle GOOG condor near breakeven to free GOOG concentration before re-entering fresh through the Monday-cycle process with a real pre-trade review.",
        "outcome": outcome, "pnl_impact": None,
    }
    with open("oversight_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--client-id", type=int, default=1665)
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
        goog_condor_log("CLOSE_ABORT", "missing live bid/ask on one or more legs")
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN -- would close short_call, short_put, then long_call, long_put via ladder. No orders submitted.")
        return

    fills = {}
    # Shorts first -- covers the real open liability before touching the long legs.
    for name in ("short_call", "short_put", "long_call", "long_put"):
        leg = LEGS[name]
        q = quotes[name]
        print(f"\n--- Closing {name} ({leg['close_side'].value}) ---")
        ok, px = place_leg_with_ladder(client, leg["alpaca_symbol"], leg["close_side"], name, 1, q["bid"], q["ask"], q["mid"])
        if not ok:
            msg = f"GOOG condor close: {name} FAILED to fill. Fills so far: {fills}. Needs manual review NOW."
            print(msg)
            goog_condor_log("CLOSE_INCOMPLETE", msg)
            oversight_log(msg, "PAUSED -- partial close, needs manual review")
            sys.exit(1)
        fills[name] = px

    # Realized P&L: credit received at entry minus net cost to close.
    close_cost = (fills["short_call"] + fills["short_put"]) - (fills["long_call"] + fills["long_put"])
    realized_pnl = round(ENTRY_CREDIT_TOTAL - close_cost * 100, 2)
    msg = f"GOOG weekly condor (2026-08-19 entry) CLOSED. Fills: {fills}. Realized P&L: ${realized_pnl:.2f}"
    print(msg)
    goog_condor_log("CLOSED", msg)
    oversight_log(msg, f"Closed via repricing ladder, shorts first. Realized P&L ${realized_pnl:.2f}.")
    print(json.dumps({"fills": fills, "realized_pnl": realized_pnl}, indent=2))


if __name__ == "__main__":
    main()
