"""Pull 7/14 EVC executions directly from IBKR and compute true P&L."""
import asyncio, json, os, sys
from collections import defaultdict
from ib_insync import IB, ExecutionFilter

EVC_TICKERS = {"MS","BLK","C","PNC","PGR","JNJ","ELV","UAL","JBHT"}

async def main():
    ib = IB()
    try:
        await ib.connectAsync("127.0.0.1", 7497, clientId=98)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    fil = ExecutionFilter(time="20260714 00:00:00")
    fills = await ib.reqExecutionsAsync(fil)

    print(f"\nTotal executions on 7/14: {len(fills)}")

    # Separate EVC from non-EVC
    evc_fills  = [f for f in fills if f.contract.symbol in EVC_TICKERS and f.contract.secType == "OPT"]
    other_fills = [f for f in fills if f not in evc_fills]

    # ── EVC breakdown ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"EVC OPTION EXECUTIONS ({len(evc_fills)} legs)")
    print(f"{'='*70}")
    print(f"{'Time':<20} {'Symbol':<6} {'Right':<2} {'Strike':>8} {'Expiry':<10} {'Action':<5} {'Qty':>4} {'Price':>8}")
    print("-"*70)

    # Group by ticker then by pos_id heuristic (batch: first entry vs re-entry)
    by_ticker = defaultdict(list)
    for f in sorted(evc_fills, key=lambda x: x.execution.time):
        sym = f.contract.symbol
        side   = f.execution.side   # "BOT" or "SLD"
        strike = f.contract.strike
        right  = f.contract.right
        qty    = f.execution.shares
        px     = f.execution.avgPrice
        t      = f.execution.time
        by_ticker[sym].append((t, right, strike, side, qty, px))
        print(f"{t:<20} {sym:<6} {right:<2} {strike:>8.1f} {f.contract.lastTradeDateOrContractMonth:<10} {side:<5} {qty:>4} {px:>8.4f}")

    # ── Compute net P&L per ticker per condor instance ─────────────────────────
    print(f"\n{'='*70}")
    print("EVC P&L SUMMARY (from IBKR fills)")
    print(f"{'='*70}")

    total_pnl = 0.0
    for sym in sorted(by_ticker.keys()):
        legs = by_ticker[sym]
        # Condors are 4-leg; group in sets of 4 (entry) + 4 (close) = 8 per condor
        # Entry: SLD short legs, BOT long legs (net = short_credit - long_cost)
        # Close: BOT short legs, SLD long legs (net = long_proceeds - short_cost)
        # Net P&L = entry_credit - close_debit
        entries = [l for l in legs if (l[3] == "SLD" and "short" not in str(l)) or True]
        # Simple approach: sum(SLD × price) - sum(BOT × price) for all legs
        net_flow = 0.0
        for (t, right, strike, side, qty, px) in legs:
            if side == "SLD":
                net_flow += qty * px
            else:
                net_flow -= qty * px
        pnl_dollars = round(net_flow * 100, 2)
        print(f"  {sym:<6}  net_flow={net_flow:+.4f}/share  P&L per share=${net_flow:+.4f}  Total=${pnl_dollars:+.2f}")
        total_pnl += pnl_dollars

    print(f"\n  {'-'*40}")
    print(f"  EVC TOTAL P&L:  ${total_pnl:+.2f}")

    # ── Non-EVC trades ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"NON-EVC TRADES ({len(other_fills)} executions)")
    print(f"{'='*70}")
    print(f"{'Time':<20} {'Symbol':<8} {'Type':<5} {'Action':<5} {'Qty':>6} {'Price':>10}")
    print("-"*70)
    for f in sorted(other_fills, key=lambda x: x.execution.time):
        print(f"{f.execution.time:<20} {f.contract.symbol:<8} {f.contract.secType:<5} "
              f"{f.execution.side:<5} {f.execution.shares:>6} {f.execution.avgPrice:>10.4f}")

    ib.disconnect()

asyncio.run(main())
