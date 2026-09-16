"""
Same capital-required model as breakout_capital_required_model.py, run
side by side for full BREAKOUT vs. PREFER-HIGH only, per CEO request
2026-09-16: "run a comparison contrasting this with just prefer-high."

Same method: equal-$D-per-alert, whole shares, peak capital = the worst
single real day's concurrent dollar need (same-day positions must be held
simultaneously until that day's close).
"""
import json
import sqlite3
from collections import defaultdict


def fetch_breakout():
    con = sqlite3.connect(r"C:\Projects\GenAI-Projects\ibkr_trader\backend\tape_data.db")
    cur = con.cursor()
    rows = cur.execute("""
        SELECT session_date, alert_price, eod_return_pct FROM alert_performance
        WHERE signal_type='BREAKOUT' AND eod_return_pct IS NOT NULL AND alert_price > 0
    """).fetchall()
    by_day = defaultdict(list)
    for date, price, ret in rows:
        by_day[date].append({"price": price, "ret": ret})
    return by_day


def fetch_prefer_high():
    with open(r"C:\Projects\GenAI-Projects\ibkr_trader\backend\crt_research\prefer_high_alerts.json") as f:
        rows = json.load(f)
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["date"]].append({"price": r["price"], "ret": r["ret"]})
    return by_day


def evaluate(by_day, target_dollars_per_alert):
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
            total_pnl += cost * (a["ret"] / 100)
            day_capital += cost
            n_captured += 1
        if day_capital > peak_capital:
            peak_capital, peak_day = day_capital, date
    return total_pnl, peak_capital, peak_day, n_captured, n_total


def report(label, by_day):
    naive_sum = sum(a["ret"] for alerts in by_day.values() for a in alerts)
    n_alerts = sum(len(v) for v in by_day.values())
    n_days = len(by_day)
    max_price = max(a["price"] for alerts in by_day.values() for a in alerts)
    busiest = max(by_day.items(), key=lambda kv: len(kv[1]))
    print(f"=== {label} ===")
    print(f"{n_alerts} real alerts across {n_days} real days, naive sum = {naive_sum:+.2f}%, "
          f"busiest day = {busiest[0]} ({len(busiest[1])} alerts), max single price = ${max_price:,.2f}")
    print(f"{'$/alert':>10}  {'captured':>10}  {'peak capital':>16}  {'$ P&L':>10}  {'ratio':>8}")
    full_D = round(max_price) + 1
    for D in sorted(set([200, 500, 1000, 2000, full_D])):
        pnl, peak, peak_day, cap, tot = evaluate(by_day, D)
        ratio = pnl / peak * 100 if peak else 0
        tag = "  <- covers every alert" if D == full_D else ""
        print(f"{'$'+str(D):>10}  {f'{cap}/{tot}':>10}  {'$'+format(peak,',.0f'):>16}  {'$'+format(pnl,'+,.0f'):>10}  {ratio:>+7.2f}%{tag}")
    print()
    return naive_sum, n_alerts, n_days, max_price


def main():
    bo = fetch_breakout()
    ph = fetch_prefer_high()
    report("BREAKOUT (all alerts)", bo)
    report("PREFER-HIGH only", ph)


if __name__ == "__main__":
    main()
