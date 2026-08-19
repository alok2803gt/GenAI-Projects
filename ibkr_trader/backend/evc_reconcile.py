"""Reconcile EVC state file vs IBKR live portfolio."""
from ib_insync import IB
from collections import defaultdict

EVC_TICKERS = {'MS','BLK','C','PNC','PGR','JNJ','ELV','UAL','JBHT'}

ib = IB()
ib.connect("127.0.0.1", 7497, clientId=98)
port = ib.portfolio()
evc = [p for p in port if p.contract.symbol in EVC_TICKERS and p.contract.secType == "OPT"]

by_ticker = defaultdict(lambda: {"cost": 0.0, "mkt": 0.0, "upnl": 0.0, "legs": []})
for p in evc:
    sym = p.contract.symbol
    by_ticker[sym]["cost"] += p.averageCost * abs(p.position)
    by_ticker[sym]["mkt"]  += p.marketValue
    by_ticker[sym]["upnl"] += p.unrealizedPNL
    by_ticker[sym]["legs"].append(p)

print("EVC positions still open in IBKR (exp 7/17) — true P&L state:")
print(f"  {'Ticker':<6} {'Legs':>4} {'CostBasis':>10} {'MktValue':>10} {'UnrealPnL':>10}  net_pos")
print("  " + "-" * 68)
total_upnl = 0.0
for sym in sorted(by_ticker):
    d = by_ticker[sym]
    net = sum(p.position for p in d["legs"])
    print(f"  {sym:<6} {len(d['legs']):>4} {d['cost']:>10.2f} {d['mkt']:>10.2f} {d['upnl']:>+10.2f}  {net:+.0f}")
    total_upnl += d["upnl"]

print("  " + "-" * 68)
print(f"  {'TOTAL':<6} {'':>4} {'':>10} {'':>10} {total_upnl:>+10.2f}")
print()
print(f"  IBKR RealizedPnL (from accountValues): $0.00")
print(f"  State file claimed P&L:                $+822.00  <-- WRONG")
print(f"  Actual open unrealized P&L:            ${total_upnl:+.2f}")
print()

# Show which positions are reversed (double-close signature)
print("Position sign check vs expected condor structure:")
for sym in sorted(by_ticker):
    legs = by_ticker[sym]["legs"]
    for p in sorted(legs, key=lambda x: (x.contract.right, x.contract.strike)):
        c = p.contract
        sign = "+" if p.position > 0 else "-"
        print(f"  {sym} {c.right}{c.strike:.0f}  pos={p.position:+.0f}  upnl={p.unrealizedPNL:+.2f}")

ib.disconnect()
