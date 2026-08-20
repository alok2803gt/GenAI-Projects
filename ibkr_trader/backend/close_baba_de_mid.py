"""
Close BABA (4-leg condor) and DE (2-leg naked longs) via Alpaca, both at
computed MID price limits (not market orders) -- CEO explicitly asked for
mid-priced limits given Alpaca marks conservatively (bid for longs, ask for
shorts), so mid captures more realistic value than the reported P&L implied.

BABA: reuses the account's proven close_condor_sequential() (shorts bought
back first, removing risk, then longs sold) -- same function/sequencing
already used for real closes this account has done before.
DE: two independent SELL TO CLOSE orders (both legs already long, no
combo/sequencing risk either order).

Quotes are IBKR live mid as of the pre-flight check baked into this file
(gathered right before writing it) -- re-verify if this is run much later,
since these are wide, thin markets that can move.

Usage:
    python close_baba_de_mid.py            # dry run -- shows plan only
    python close_baba_de_mid.py --fire     # places real limit orders
"""
import argparse
import json
import sys

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from alpaca_0dte_common import place_leg, close_condor_sequential

with open("scanner_config.json") as f:
    cfg = json.load(f)

# Mid prices computed from live IBKR quotes just before this script was written.
DE_LIMITS = {
    "call": 6.60,   # DE 627.5C: bid 4.40 / ask 8.80 -- mid
    "put":  0.02,   # DE 540P: no real bid (dead quote), ask 0.05 -- using a
                     # low, likely-fillable price since mid isn't computable;
                     # this leg is worth at most $20 total either way.
}
BABA_LIMITS = {
    "short_call": 0.11,  # 137C: bid 0.08 / ask 0.13 -- mid (BUY TO CLOSE)
    "short_put":  0.17,  # 120P: bid 0.15 / ask 0.20 -- mid (BUY TO CLOSE)
    "long_call":  0.03,  # 141C: bid 0.02 / ask 0.04 -- mid (SELL TO CLOSE)
    "long_put":   0.03,  # 115P: bid 0.02 / ask 0.03 -- mid (SELL TO CLOSE)
}

DE_SYMBOLS = {"call": "DE260821C00627500", "put": "DE260821P00540000"}
BABA_SYMBOLS = {
    "short_call": "BABA260821C00137000", "long_call": "BABA260821C00141000",
    "short_put": "BABA260821P00120000",  "long_put": "BABA260821P00115000",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true")
    args = ap.parse_args()

    client = TradingClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
                            paper=False, url_override=cfg["alpaca_base_url"])

    print("=== PLAN ===")
    print(f"DE:   SELL 2x {DE_SYMBOLS['call']} @ ${DE_LIMITS['call']}  "
          f"+ SELL 4x {DE_SYMBOLS['put']} @ ${DE_LIMITS['put']}")
    print(f"BABA: BUY TO CLOSE {BABA_SYMBOLS['short_call']} @ ${BABA_LIMITS['short_call']}, "
          f"BUY TO CLOSE {BABA_SYMBOLS['short_put']} @ ${BABA_LIMITS['short_put']} (shorts first)")
    print(f"      SELL TO CLOSE {BABA_SYMBOLS['long_call']} @ ${BABA_LIMITS['long_call']}, "
          f"SELL TO CLOSE {BABA_SYMBOLS['long_put']} @ ${BABA_LIMITS['long_put']} (longs after)")

    if not args.fire:
        print("\nDRY RUN -- no orders placed. Re-run with --fire to execute.")
        return

    print("\n=== FIRING: DE ===")
    ok_c, px_c = place_leg(client, DE_SYMBOLS["call"], OrderSide.SELL, DE_LIMITS["call"], "DE 627.5C", qty=2)
    ok_p, px_p = place_leg(client, DE_SYMBOLS["put"], OrderSide.SELL, DE_LIMITS["put"], "DE 540P", qty=4)
    print(f"DE call: {'FILLED @ $'+str(px_c) if ok_c else 'NOT FILLED'}")
    print(f"DE put:  {'FILLED @ $'+str(px_p) if ok_p else 'NOT FILLED'}")

    print("\n=== FIRING: BABA (shorts first, then longs) ===")
    ok, fills = close_condor_sequential(
        client,
        syms={"short_call": BABA_SYMBOLS["short_call"], "short_put": BABA_SYMBOLS["short_put"],
              "long_call": BABA_SYMBOLS["long_call"], "long_put": BABA_SYMBOLS["long_put"]},
        limits=BABA_LIMITS,
        qty=1,
    )
    print(f"BABA all filled: {ok}")
    print(f"BABA fills: {fills}")

    print("\nDone. Verify final state via GET /alpaca/positions.")


if __name__ == "__main__":
    main()
