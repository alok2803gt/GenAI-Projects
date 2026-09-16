"""
Real, capital-constrained position-sizing model for breakout_scanner's
BREAKOUT alerts, per CEO request 2026-09-16: "model correct ratio and
provide how trades should have been sized to align with this cumulative
EOD return."

The problem being fixed: the heatmap's "cumulative sum of EOD return"
(+76.83% for BREAKOUT) implicitly assumes UNLIMITED capital -- one full,
equal-weight position per alert, no matter how many fire (median 15/day,
up to 28/day, real prices from $16.95 to $6,828.62). That's not
executable on this account's real ~$2,027 net liq -- even a single share
of the expensive names exceeds the whole account, and median-priced
alerts ($184) leave room for only 1-2 real positions on a busy day, not 15.

This script replays the REAL alert stream in REAL chronological order
(alert_time_et, no lookahead -- you don't know a later alert's outcome
when deciding on an earlier one) under a few real, capital-honest sizing
rules, and reports the REAL dollar P&L each rule would have produced
against this account's real capital, not a theoretical percentage.
"""
import sqlite3
from collections import defaultdict

NET_LIQ = 2027.27  # real, as of the last live check this session

def fetch_alerts():
    con = sqlite3.connect(r"C:\Projects\GenAI-Projects\ibkr_trader\backend\tape_data.db")
    cur = con.cursor()
    rows = cur.execute("""
        SELECT session_date, alert_time_et, ticker, alert_price, eod_return_pct
        FROM alert_performance
        WHERE signal_type='BREAKOUT' AND eod_return_pct IS NOT NULL AND alert_price > 0
        ORDER BY session_date, alert_time_et
    """).fetchall()
    by_day = defaultdict(list)
    for date, t, ticker, price, ret in rows:
        by_day[date].append({"time": t, "ticker": ticker, "price": price, "ret_pct": ret})
    return by_day


def simulate(by_day, per_trade_budget, max_positions_per_day, label):
    """Walk each day's real alerts in real chronological order. Take an
    alert only if: (a) its real price fits within per_trade_budget for at
    least 1 real share, (b) the day's position count hasn't hit
    max_positions_per_day yet. No lookahead -- decisions use only what
    was knowable at alert time (price), never the outcome (ret_pct)."""
    total_pnl = 0.0
    total_deployed = 0.0
    n_trades = 0
    n_skipped_too_expensive = 0
    daily_pnl = {}
    for date, alerts in sorted(by_day.items()):
        taken = 0
        day_pnl = 0.0
        for a in alerts:
            if taken >= max_positions_per_day:
                break
            shares = int(per_trade_budget // a["price"])
            if shares < 1:
                n_skipped_too_expensive += 1
                continue
            cost = shares * a["price"]
            pnl = cost * (a["ret_pct"] / 100)
            total_pnl += pnl
            total_deployed += cost
            day_pnl += pnl
            n_trades += 1
            taken += 1
        daily_pnl[date] = day_pnl
    print(f"--- {label} ---")
    print(f"  per_trade_budget=${per_trade_budget:.0f}  max_positions_per_day={max_positions_per_day}")
    print(f"  real trades taken: {n_trades}  (skipped as unaffordable: {n_skipped_too_expensive})")
    print(f"  total capital deployed (sum across all entries, not concurrent): ${total_deployed:,.2f}")
    print(f"  REAL total $ P&L: ${total_pnl:+,.2f}")
    print(f"  as % of real net liq (${NET_LIQ:,.2f}): {total_pnl/NET_LIQ*100:+.2f}%")
    pos_days = sum(1 for v in daily_pnl.values() if v > 0)
    neg_days = sum(1 for v in daily_pnl.values() if v < 0)
    print(f"  positive days: {pos_days}  negative days: {neg_days}")
    print()
    return total_pnl, n_trades


def main():
    by_day = fetch_alerts()
    total_days = len(by_day)
    total_alerts = sum(len(v) for v in by_day.values())
    print(f"Real data: {total_alerts} alerts across {total_days} real trading days\n")

    # Reference: the naive, capital-unconstrained "take every alert equally" number
    naive_sum_pct = sum(a["ret_pct"] for alerts in by_day.values() for a in alerts)
    print(f"Naive (unconstrained) sum of eod_return_pct across all alerts: {naive_sum_pct:+.2f}% "
          f"-- this is the heatmap's number, and it's not achievable with real capital.\n")

    cap_pct_default = 0.05      # this account's shared per-trade default (cro_cfo_capital_budget)
    cap_pct_generous = 0.10     # a more generous, still-real-precedent override (GOOG Condor uses 0.15, Day Trader 0.20)

    simulate(by_day, NET_LIQ * cap_pct_default, max_positions_per_day=1,
             label="A) 5% per-trade budget, 1 position/day (most conservative, matches this account's shared default cap)")
    simulate(by_day, NET_LIQ * cap_pct_default, max_positions_per_day=2,
             label="B) 5% per-trade budget, up to 2 positions/day")
    simulate(by_day, NET_LIQ * cap_pct_generous, max_positions_per_day=1,
             label="C) 10% per-trade budget, 1 position/day")
    simulate(by_day, NET_LIQ * cap_pct_generous, max_positions_per_day=2,
             label="D) 10% per-trade budget, up to 2 positions/day")


if __name__ == "__main__":
    main()
