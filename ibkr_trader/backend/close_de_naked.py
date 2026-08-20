"""
Close DE's naked long position (the EVC incident, 2026-08-19, task 2026-08-19-005)
via Alpaca. Both legs are LONG (no shorts, no combo/sequencing risk) -- this is a
plain sell-to-close on each leg, done as two independent market orders so it fills
promptly at whatever the market shows right after DE's 8:00am ET earnings release.

Dry-run by default: pulls the REAL live Alpaca position (not hardcoded) and shows
exactly what would be submitted, places nothing. Only fires real orders with --fire,
per this account's standing rule -- never auto-fire a real close without explicit
confirmation at the time.

Usage:
    python close_de_naked.py            # dry run -- shows plan, places nothing
    python close_de_naked.py --fire     # places real market sell-to-close orders
"""
import argparse
import json
import sys
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

FILL_WAIT_S = 30

with open("scanner_config.json") as f:
    cfg = json.load(f)


def close_leg(client, symbol, qty, label):
    print(f"  {label}: SELL {qty}x {symbol} @ MARKET...")
    order = client.submit_order(MarketOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
    ))
    for _ in range(FILL_WAIT_S):
        time.sleep(1)
        try:
            chk = client.get_order_by_id(order.id)
        except Exception as exc:
            print(f"    status check failed: {exc}")
            continue
        if str(chk.status) == "OrderStatus.FILLED":
            print(f"    FILLED @ ${chk.filled_avg_price}")
            return True, float(chk.filled_avg_price)
        if str(chk.status) in ("OrderStatus.REJECTED", "OrderStatus.CANCELED"):
            print(f"    {chk.status}")
            return False, None
    print("    still not filled after wait -- leaving order working, check manually")
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true", help="place real orders (default: dry run)")
    args = ap.parse_args()

    client = TradingClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
                            paper=False, url_override=cfg["alpaca_base_url"])

    print("Fetching live DE positions from Alpaca...")
    positions = client.get_all_positions()
    de_legs = [p for p in positions if p.symbol.startswith("DE2")]

    if not de_legs:
        print("No DE option positions found -- nothing to close (already closed, or symbol changed).")
        sys.exit(0)

    print(f"Found {len(de_legs)} DE leg(s):")
    total_mv = 0.0
    for p in de_legs:
        mv = float(p.market_value)
        total_mv += mv
        print(f"  {p.symbol}  qty={p.qty}  side={p.side}  "
              f"current_price={p.current_price}  market_value=${mv:.2f}  "
              f"unrealized_pl=${float(p.unrealized_pl):+.2f}")
    print(f"Total current market value: ${total_mv:.2f}")

    if not args.fire:
        print("\nDRY RUN -- no orders placed. Re-run with --fire to actually close these legs.")
        return

    print(f"\n=== FIRING: closing {len(de_legs)} leg(s) via market sell-to-close ===")
    results = []
    for p in de_legs:
        qty = abs(int(float(p.qty)))
        if p.side != "PositionSide.LONG":
            print(f"  SKIPPING {p.symbol}: side={p.side}, expected LONG -- unexpected state, check manually")
            continue
        ok, fill_px = close_leg(client, p.symbol, qty, p.symbol)
        results.append((p.symbol, qty, ok, fill_px))

    print("\n=== CLOSE SUMMARY ===")
    total_proceeds = 0.0
    for sym, qty, ok, fill_px in results:
        if ok:
            proceeds = fill_px * qty * 100
            total_proceeds += proceeds
            print(f"  {sym}: FILLED {qty}x @ ${fill_px}  proceeds=${proceeds:+.2f}")
        else:
            print(f"  {sym}: NOT FILLED -- check manually, may still be open")
    print(f"Total proceeds from filled legs: ${total_proceeds:+.2f}")
    print("Verify final state via Alpaca get_all_positions() / GET /alpaca/positions.")


if __name__ == "__main__":
    main()
