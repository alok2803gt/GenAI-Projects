"""
Close BABA (4-leg condor) and DE (2-leg naked longs) via Alpaca, walking each
leg's limit price from a FAVORABLE starting point toward mid, then toward a
guaranteed-fill price if needed -- maximizes realized profit vs a single flat
mid order, while still guaranteeing the position actually closes today.

Ladder per leg (3 steps, ~15s wait each):
  1. Favorable: 75% of the way from mid toward the best side (ask for sells,
     bid for buys) -- tries for meaningfully better than mid first.
  2. Mid: the CEO-approved fallback price.
  3. Aggressive: crosses to the opposite side (ask for sells, bid for buys)
     -- guarantees a fill; only reached if steps 1-2 both time out.

BABA closes shorts first (removes risk), then longs -- same proven
sequencing this account always uses. DE has no combo risk (both legs long).

Live quotes are re-pulled fresh from IBKR right before firing, not reused
from an earlier dry-run, since these are thin/wide markets that move.

Usage:
    python close_baba_de_maxprofit.py            # dry run -- shows live quotes + planned ladder
    python close_baba_de_maxprofit.py --fire      # walks the ladder for real
"""
import argparse
import json
import time

from ib_insync import IB, Option
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import LimitOrderRequest

with open("scanner_config.json") as f:
    cfg = json.load(f)

STEP_WAIT_S = 15

LEGS = {
    # name: (ib_ticker, ib_expiry, strike, right, alpaca_symbol, side, qty)
    "DE_call":       ("DE", "20260821", 627.5, "C", "DE260821C00627500", OrderSide.SELL, 2),
    "DE_put":        ("DE", "20260821", 540.0, "P", "DE260821P00540000", OrderSide.SELL, 4),
    "BABA_short_c":  ("BABA", "20260821", 137.0, "C", "BABA260821C00137000", OrderSide.BUY, 1),
    "BABA_short_p":  ("BABA", "20260821", 120.0, "P", "BABA260821P00120000", OrderSide.BUY, 1),
    "BABA_long_c":   ("BABA", "20260821", 141.0, "C", "BABA260821C00141000", OrderSide.SELL, 1),
    "BABA_long_p":   ("BABA", "20260821", 115.0, "P", "BABA260821P00115000", OrderSide.SELL, 1),
}
# BABA close order: shorts first (remove risk), then longs
BABA_ORDER = ["BABA_short_c", "BABA_short_p", "BABA_long_c", "BABA_long_p"]


def get_live_quotes():
    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", 7496, clientId=1711, timeout=15)
    quotes = {}
    try:
        contracts = {}
        for name, (tk, exp, strike, right, *_rest) in LEGS.items():
            c = Option(tk, exp, strike, right, "SMART")
            contracts[name] = c
        ib.qualifyContracts(*contracts.values())
        tds = {name: ib.reqMktData(c, "", False, False) for name, c in contracts.items()}
        ib.sleep(6)
        for name, td in tds.items():
            b, a = td.bid, td.ask
            b = b if (b and b > 0) else None
            a = a if (a and a > 0) else None
            mid = round((b + a) / 2, 2) if (b and a) else None
            quotes[name] = {"bid": b, "ask": a, "mid": mid, "last": td.last}
            ib.cancelMktData(contracts[name])
    finally:
        ib.disconnect()
    return quotes


def price_ladder(side, q):
    """Return [favorable, mid, aggressive] prices for this leg. side: OrderSide."""
    bid, ask, mid = q["bid"], q["ask"], q["mid"]
    if mid is None:
        # No two-sided market -- use whatever single side exists, no ladder room.
        fallback = ask if side == OrderSide.SELL else bid
        fallback = fallback or (q["last"] or 0.01)
        return [round(fallback, 2)] * 3
    if side == OrderSide.SELL:
        favorable = round(mid + (ask - mid) * 0.75, 2)
        aggressive = bid if bid else round(mid * 0.5, 2)
    else:  # BUY (to close a short)
        favorable = round(mid - (mid - bid) * 0.75, 2) if bid else round(mid * 0.75, 2)
        aggressive = ask
    return [favorable, mid, aggressive]


def place_with_ladder(client, name, symbol, side, qty, ladder):
    for step_i, px in enumerate(ladder, 1):
        label = ["favorable", "mid", "aggressive"][step_i - 1]
        print(f"  [{name}] step {step_i} ({label}): {side.value} {qty}x {symbol} @ ${px}")
        try:
            order = client.submit_order(LimitOrderRequest(
                symbol=symbol, qty=qty, side=side,
                type=OrderType.LIMIT, time_in_force=TimeInForce.DAY, limit_price=px,
            ))
        except Exception as exc:
            print(f"    submit failed: {exc}")
            continue
        filled = False
        for _ in range(STEP_WAIT_S):
            time.sleep(1)
            try:
                chk = client.get_order_by_id(order.id)
            except Exception:
                continue
            if str(chk.status) == "OrderStatus.FILLED":
                print(f"    FILLED @ ${chk.filled_avg_price}")
                return True, float(chk.filled_avg_price)
            if str(chk.status) in ("OrderStatus.REJECTED", "OrderStatus.CANCELED"):
                print(f"    {chk.status}")
                break
        else:
            filled = False
        if not filled:
            try:
                client.cancel_order_by_id(order.id)
            except Exception:
                pass
            print(f"    not filled at step {step_i}, moving to next price" if step_i < len(ladder) else "    not filled at final step")
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true")
    args = ap.parse_args()

    print("Pulling live quotes...")
    quotes = get_live_quotes()
    ladders = {}
    for name, (_tk, _exp, _strike, _right, sym, side, qty) in LEGS.items():
        q = quotes[name]
        ladder = price_ladder(side, q)
        ladders[name] = ladder
        print(f"{name}: bid={q['bid']} ask={q['ask']} mid={q['mid']}  ->  ladder {ladder} ({side.value})")

    if not args.fire:
        print("\nDRY RUN -- no orders placed. Re-run with --fire to execute.")
        return

    client = TradingClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
                            paper=False, url_override=cfg["alpaca_base_url"])

    print("\n=== FIRING: DE (independent legs, no combo risk) ===")
    results = {}
    for name in ("DE_call", "DE_put"):
        _tk, _exp, _strike, _right, sym, side, qty = LEGS[name]
        ok, px = place_with_ladder(client, name, sym, side, qty, ladders[name])
        results[name] = (ok, px)

    print("\n=== FIRING: BABA (shorts first, then longs) ===")
    for name in BABA_ORDER:
        _tk, _exp, _strike, _right, sym, side, qty = LEGS[name]
        ok, px = place_with_ladder(client, name, sym, side, qty, ladders[name])
        results[name] = (ok, px)
        if name in ("BABA_short_c", "BABA_short_p") and not ok:
            print(f"  WARNING: {name} did not fill -- pausing before selling longs, check position manually")
            break

    print("\n=== SUMMARY ===")
    for name, (ok, px) in results.items():
        print(f"  {name}: {'FILLED @ $'+str(px) if ok else 'NOT FILLED'}")
    print("\nVerify final state via GET /alpaca/positions.")


if __name__ == "__main__":
    main()
