"""Check whether EVC positions are actually still open in IBKR."""
from ib_insync import IB
import json

EVC_TICKERS = {'MS','BLK','C','PNC','PGR','JNJ','ELV','UAL','JBHT'}

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=98)

port = ib.portfolio()
evc_port = [p for p in port if p.contract.symbol in EVC_TICKERS
            and p.contract.secType == "OPT"]

print(f"EVC option legs still open in IBKR: {len(evc_port)}")
print()
hdr = f"{'Symbol':<6} {'R':<1} {'Strike':>8} {'Expiry':<10} {'Pos':>4} {'AvgCost':>9} {'MktVal':>10} {'UnrealPnL':>10}"
print(hdr)
print("-" * len(hdr))

total_unrealized = 0.0
by_ticker = {}
for p in sorted(evc_port, key=lambda x: (x.contract.symbol, x.contract.right, x.contract.strike)):
    sym    = p.contract.symbol
    right  = p.contract.right
    strike = p.contract.strike
    expiry = p.contract.lastTradeDateOrContractMonth
    pos    = p.position
    cost   = p.averageCost
    mval   = p.marketValue
    upnl   = p.unrealizedPNL
    print(f"{sym:<6} {right:<1} {strike:>8.1f} {expiry:<10} {pos:>4} {cost:>9.4f} {mval:>10.2f} {upnl:>10.2f}")
    total_unrealized += upnl
    by_ticker.setdefault(sym, []).append(p)

print("-" * len(hdr))
print(f"{'EVC total unrealized P&L:':>50} {total_unrealized:>10.2f}")

print()
with open("earnings_vol_crush_state.json") as f:
    ev = json.load(f)
pos_keys = list(ev["positions"].keys())
print(f"State file ev['positions']: {pos_keys if pos_keys else '(empty - believes all closed)'}")
print(f"State file closed_today:    {len(ev['closed_today'])} records")
print()
print("CONCLUSION:")
if evc_port:
    print("  IBKR has EVC option legs still open.")
    print("  State file incorrectly shows them as closed.")
    print("  The _evc_close_position MKT orders either:")
    print("    a) Were placed but fills returned in 0 in 0.5s window -> code declared 'closed'")
    print("    b) Some filled, some didn't -> mixed state")
    print()
    # Identify which tickers still have open legs
    open_tickers = set(by_ticker.keys())
    print(f"  Tickers with open legs: {sorted(open_tickers)}")
    print(f"  Expected legs per condor: 4 (short_put, long_put, short_call, long_call)")
    for sym, positions in sorted(by_ticker.items()):
        net_pos = sum(p.position for p in positions)
        print(f"    {sym}: {len(positions)} legs, net_pos={net_pos}")
else:
    print("  No EVC option positions open in IBKR. Positions were correctly closed.")

ib.disconnect()
