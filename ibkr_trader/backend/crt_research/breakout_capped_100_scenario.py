"""
Scenario: cap the universe to alert_price <= $100, per CEO request
2026-09-16. Same equal-$D-per-alert / peak-concurrent-capital model as
breakout_vs_prefer_high_capital.py, run on the price-capped subset for
both BREAKOUT and PREFER-HIGH, alongside the uncapped figures for a full
side-by-side.
"""
import json
import sqlite3
from collections import defaultdict

PRICE_CAP = 100.0


def fetch_breakout(cap=None):
    con = sqlite3.connect(r"C:\Projects\GenAI-Projects\ibkr_trader\backend\tape_data.db")
    cur = con.cursor()
    rows = cur.execute("""
        SELECT session_date, alert_price, eod_return_pct FROM alert_performance
        WHERE signal_type='BREAKOUT' AND eod_return_pct IS NOT NULL AND alert_price > 0
    """).fetchall()
    by_day = defaultdict(list)
    for date, price, ret in rows:
        if cap and price > cap:
            continue
        by_day[date].append({"price": price, "ret": ret})
    return by_day


def fetch_prefer_high(cap=None):
    with open(r"C:\Projects\GenAI-Projects\ibkr_trader\backend\crt_research\prefer_high_alerts.json") as f:
        rows = json.load(f)
    by_day = defaultdict(list)
    for r in rows:
        if cap and r["price"] > cap:
            continue
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
    if not by_day:
        print(f"=== {label} === (no alerts survive this filter)\n")
        return
    naive_sum = sum(a["ret"] for alerts in by_day.values() for a in alerts)
    n_alerts = sum(len(v) for v in by_day.values())
    n_days = len(by_day)
    max_price = max(a["price"] for alerts in by_day.values() for a in alerts)
    busiest = max(by_day.items(), key=lambda kv: len(kv[1]))
    print(f"=== {label} ===")
    print(f"{n_alerts} real alerts / {n_days} real days, naive sum = {naive_sum:+.2f}%, "
          f"busiest day = {busiest[0]} ({len(busiest[1])} alerts), max price = ${max_price:,.2f}")
    print(f"{'$/alert':>10}  {'captured':>10}  {'peak capital':>14}  {'$ P&L':>9}  {'ratio':>8}")
    full_D = round(max_price) + 1
    for D in sorted(set([50, 100, full_D])):
        pnl, peak, peak_day, cap, tot = evaluate(by_day, D)
        ratio = pnl / peak * 100 if peak else 0
        tag = "  <- full coverage" if D == full_D else ""
        print(f"{'$'+str(D):>10}  {f'{cap}/{tot}':>10}  {'$'+format(peak,',.0f'):>14}  {'$'+format(pnl,'+,.0f'):>9}  {ratio:>+7.2f}%{tag}")
    print()


def main():
    print("################ UNCAPPED (for reference) ################\n")
    report("BREAKOUT, uncapped", fetch_breakout())
    report("PREFER-HIGH, uncapped", fetch_prefer_high())

    print(f"################ PRICE CAPPED AT ${PRICE_CAP:.0f} ################\n")
    report(f"BREAKOUT, price <= ${PRICE_CAP:.0f}", fetch_breakout(cap=PRICE_CAP))
    report(f"PREFER-HIGH, price <= ${PRICE_CAP:.0f}", fetch_prefer_high(cap=PRICE_CAP))


if __name__ == "__main__":
    main()
