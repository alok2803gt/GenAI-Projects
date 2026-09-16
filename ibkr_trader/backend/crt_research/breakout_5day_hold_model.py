"""
5-day-hold scenario, per CEO request 2026-09-16: same equal-$D-per-alert
sizing model, but now the exit is 5 REAL trading days later (real yfinance
daily closes), not the same-day close.

This changes the capital mechanics fundamentally from every earlier
scenario in this thread: same-day positions never overlap (each closes
before the next trading day), so peak capital was just the busiest single
day's total. A 5-day hold means a position opened Monday is still open
when Tuesday/Wednesday/Thursday/Friday's alerts fire -- positions from up
to 5 different entry days can be open SIMULTANEOUSLY. Peak capital here is
found by a real rolling-window simulation: walk every real calendar day,
sum the $ cost of every position entered in the last 5 trading days that
hasn't exited yet, and track the maximum across the whole real history.
"""
import json
from collections import defaultdict


def load_alerts():
    with open(r"C:\Projects\GenAI-Projects\ibkr_trader\backend\crt_research\breakout_5d_returns.json") as f:
        rows = json.load(f)
    return rows


def evaluate(rows, target_dollars_per_alert):
    # Build per-alert real position record: entry_date, exit_date, price, ret_5d
    positions = []
    for r in rows:
        shares = round(target_dollars_per_alert / r["price"])
        if shares < 1:
            continue
        cost = shares * r["price"]
        pnl = cost * (r["ret_5d"] / 100)
        positions.append({"entry": r["date"], "exit": r["exit_date"], "cost": cost, "pnl": pnl})

    total_pnl = sum(p["pnl"] for p in positions)
    n_captured = len(positions)

    # Rolling concurrent-capital simulation: for every real date that is
    # either an entry or exit date anywhere in the dataset, compute total
    # $ of positions open ON that date (entry <= date < exit).
    all_dates = sorted(set([p["entry"] for p in positions] + [p["exit"] for p in positions]))
    peak_capital = 0.0
    peak_date = None
    for d in all_dates:
        open_cost = sum(p["cost"] for p in positions if p["entry"] <= d < p["exit"])
        if open_cost > peak_capital:
            peak_capital, peak_date = open_cost, d

    return total_pnl, peak_capital, peak_date, n_captured, len(rows)


def report(label, rows):
    naive_sum = sum(r["ret_5d"] for r in rows)
    max_price = max(r["price"] for r in rows)
    print(f"=== {label} (5-day hold) ===")
    print(f"{len(rows)} real alerts with a real 5d-forward outcome, naive sum of 5d returns = {naive_sum:+.2f}%, "
          f"max single price = ${max_price:,.2f}")
    print(f"{'$/alert':>10}  {'captured':>10}  {'peak capital':>16}  {'$ P&L':>10}  {'ratio':>8}  {'peak on'}")
    full_D = round(max_price) + 1
    for D in sorted(set([200, 500, 1000, 2000, full_D])):
        pnl, peak, peak_date, cap, tot = evaluate(rows, D)
        ratio = pnl / peak * 100 if peak else 0
        tag = "  <- full coverage" if D == full_D else ""
        print(f"{'$'+str(D):>10}  {f'{cap}/{tot}':>10}  {'$'+format(peak,',.0f'):>16}  {'$'+format(pnl,'+,.0f'):>10}  "
              f"{ratio:>+7.2f}%{tag}  ({peak_date})")
    print()


def main():
    rows = load_alerts()
    report("BREAKOUT, all alerts", rows)


if __name__ == "__main__":
    main()
