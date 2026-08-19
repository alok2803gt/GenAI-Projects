import json

with open('earnings_vol_crush_state.json') as f:
    ev = json.load(f)

closed = ev['closed_today']

# Separate first entries (batch 1) from re-entries (batch 2)
seen = {}
batch1 = []
batch2 = []
for r in closed:
    t = r['ticker']
    if t not in seen:
        seen[t] = 1
        batch1.append(r)
    else:
        seen[t] += 1
        batch2.append(r)

def print_batch(label, rows):
    print(label)
    print(f"  {'Ticker':<6} {'Credit':>7} {'CloseCost':>10} {'P&L':>8} {'P&L%':>7}  Notes")
    print("  " + "-"*65)
    total = 0
    for r in rows:
        note = "(fill=0, credited at $0)" if r['close_cost'] == 0 else ""
        print(f"  {r['ticker']:<6} {r['net_credit']:>7.2f} {r['close_cost']:>10.2f} "
              f"{r['pnl']:>+8.2f} {r['pnl_pct']:>6.1f}%  {note}")
        total += r['pnl']
    print(f"  {'':6} {'':7} {'SUBTOTAL':>10} {total:>+8.2f}")
    return total

print()
print("=" * 70)
print("EVC P&L SUMMARY - 7/14/2026 (from state file fills)")
print("=" * 70)
print()

b1 = print_batch("BATCH 1 - Real entries (3:31-3:40 PM ET, 9 positions)", batch1)
print()
b2 = print_batch("BATCH 2 - Re-entries (bug: 3:45-3:51 PM ET, re-entry guard missing)", batch2)

print()
print(f"  GRAND TOTAL EVC P&L 7/14:  ${b1 + b2:+.2f}")
print()
print("DATA QUALITY NOTES:")
zero_cost = [r['ticker'] for r in closed if r['close_cost'] == 0]
print(f"  1. close_cost=0 for: {zero_cost}")
print(f"     => Fill prices returned $0 in 0.5s window; P&L booked as full net_credit.")
print(f"     => Actual fills likely happened at near-zero (market was pricing them out).")
print(f"  2. close_cost formula bug: sums all 4 leg fills without netting SELL proceeds.")
print(f"     => For positions with close_cost>0, cost is inflated by ~long_leg_value,")
print(f"        making losses appear larger than actual (e.g., JNJ -$299 and MS -$274).")
print(f"  3. All 15 journal rows marked 'orphaned' (exit_price=NULL) — journal UPDATE bug fixed today.")
print()
print("WHAT ACTUALLY HAPPENED:")
print("  - Positions should have been held overnight for vol crush (BMO July 15 earnings).")
print("  - False max_loss_stop fired within 5-20 min of entry because P&L monitor")
print("    only quoted 2 short legs, ignoring long leg offset (4-leg bug fixed today).")
print("  - Batch 1 closed 3:31-3:40 PM instead of next morning.")
print("  - Batch 2 re-entered same tickers because closed_today guard was missing (fixed today).")
