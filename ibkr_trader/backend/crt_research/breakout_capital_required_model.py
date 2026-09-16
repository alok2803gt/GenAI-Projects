"""
Solve the inverse of the sizing question: what real capital and position-
sizing MODEL would be needed to actually realize something proportional to
BREAKOUT's naive +76.83% cumulative sum of eod_return_pct?

Model: equal-DOLLAR weight $D per alert (not equal shares) -- the only
sizing rule that turns "sum of independent % returns" into a single,
well-defined dollar P&L: total $ P&L = D x (sum of eod_return_pct / 100).
Whole shares only (matches how this account actually places real orders).

Capital required is bounded by the busiest REAL day (2026-08-13, 28 real
alerts), since eod_return_pct positions are same-day (alert -> that day's
close) -- every alert on the busiest day needs to be held CONCURRENTLY
until that day's close, not sequentially, so peak capital = that day's
real per-alert dollar need summed, not total-across-all-51-days.
"""
import sqlite3
from collections import defaultdict


def fetch_alerts():
    con = sqlite3.connect(r"C:\Projects\GenAI-Projects\ibkr_trader\backend\tape_data.db")
    cur = con.cursor()
    rows = cur.execute("""
        SELECT session_date, ticker, alert_price, eod_return_pct
        FROM alert_performance
        WHERE signal_type='BREAKOUT' AND eod_return_pct IS NOT NULL AND alert_price > 0
        ORDER BY session_date
    """).fetchall()
    by_day = defaultdict(list)
    for date, ticker, price, ret in rows:
        by_day[date].append({"ticker": ticker, "price": price, "ret": ret})
    return by_day


def evaluate(by_day, target_dollars_per_alert):
    """For a given target $D per alert: real whole-share count per alert =
    round(D/price) (0 if price > D -- can't afford even 1 share within
    target, alert is skipped, a real deviation from pure equal-weighting).
    Returns: total $ P&L, peak same-day concurrent capital actually used,
    the day it occurred on, and how many of the 763 real alerts were
    actually captured at this D."""
    total_pnl = 0.0
    peak_capital = 0.0
    peak_day = None
    n_captured = 0
    n_total = 0
    for date, alerts in by_day.items():
        day_capital = 0.0
        for a in alerts:
            n_total += 1
            shares = round(target_dollars_per_alert / a["price"])
            if shares < 1:
                continue
            cost = shares * a["price"]
            pnl = cost * (a["ret"] / 100)
            total_pnl += pnl
            day_capital += cost
            n_captured += 1
        if day_capital > peak_capital:
            peak_capital = day_capital
            peak_day = date
    return total_pnl, peak_capital, peak_day, n_captured, n_total


def main():
    by_day = fetch_alerts()
    naive_sum_pct = sum(a["ret"] for alerts in by_day.values() for a in alerts)
    n_alerts = sum(len(v) for v in by_day.values())
    max_alert_price = max(a["price"] for alerts in by_day.values() for a in alerts)
    busiest_day = max(by_day.items(), key=lambda kv: len(kv[1]))
    print(f"Real data: {n_alerts} BREAKOUT alerts, naive sum of eod_return_pct = {naive_sum_pct:+.2f}%")
    print(f"Busiest real day: {busiest_day[0]} ({len(busiest_day[1])} alerts)")
    print(f"Most expensive single real alert: ${max_alert_price:,.2f}\n")

    print(f"{'Target $/alert':>16}  {'Alerts captured':>16}  {'Peak capital needed':>20}  {'Real $ P&L':>12}  {'Return on peak capital':>24}")
    for D in (200, 500, 1000, 2000, 3000, 5000, 6829):
        pnl, peak, peak_day, captured, total = evaluate(by_day, D)
        ratio = pnl / peak * 100 if peak > 0 else 0
        print(f"{'$'+str(D):>16}  {f'{captured}/{total}':>16}  {'$'+format(peak, ',.2f'):>20}  {'$'+format(pnl, '+,.2f'):>12}  {ratio:>+22.2f}%   (peak day: {peak_day})")


if __name__ == "__main__":
    main()
