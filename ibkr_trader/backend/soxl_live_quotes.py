"""
One-off: pull REAL live SOXL option quotes from IBKR to validate the
Black-Scholes-modeled premiums in the CIO+CRO SOXL CSP report (2026-08-08).
Read-only market data -- no orders placed. Uses its own clientId to avoid
colliding with the live backend (clientId=3).
"""
import sys
from ib_insync import IB, Stock, Option, util

CLIENT_ID = 812
TWS_PORT = 7496  # live account, per this codebase's standing config

def main():
    ib = IB()
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=30)
    print("Connected.")

    soxl = Stock("SOXL", "SMART", "USD")
    ib.qualifyContracts(soxl)
    [ticker] = ib.reqTickers(soxl)
    spot = ticker.marketPrice()
    if spot != spot:  # NaN check
        spot = ticker.last or ticker.close
    print(f"SOXL spot: {spot:.2f}")

    chains = ib.reqSecDefOptParams(soxl.symbol, "", soxl.secType, soxl.conId)
    chain = next(c for c in chains if c.exchange == "SMART")
    expiries = sorted(chain.expirations)
    strikes_all = sorted(chain.strikes)

    from datetime import datetime
    today = datetime.now()
    def days_out(e):
        return (datetime.strptime(e, "%Y%m%d") - today).days

    monthly_candidates = [e for e in expiries if 21 <= days_out(e) <= 45]
    weekly_candidates = [e for e in expiries if 4 <= days_out(e) <= 10]
    monthly_exp = min(monthly_candidates, key=days_out) if monthly_candidates else None
    weekly_exp = min(weekly_candidates, key=days_out) if weekly_candidates else None
    print(f"Monthly expiry chosen: {monthly_exp} ({days_out(monthly_exp) if monthly_exp else 'n/a'}d out)")
    print(f"Weekly expiry chosen: {weekly_exp} ({days_out(weekly_exp) if weekly_exp else 'n/a'}d out)")

    targets = []
    for cushion in [0.15, 0.20, 0.25]:
        k_short_theoretical = spot * (1 - cushion)
        k_short = min(strikes_all, key=lambda s: abs(s - k_short_theoretical))
        width_theoretical = k_short * 0.10
        k_long_theoretical = k_short - width_theoretical
        k_long = min([s for s in strikes_all if s < k_short], key=lambda s: abs(s - k_long_theoretical), default=None)
        # also a narrow (right-sized) long strike ~$3 below short, snapped to nearest listed strike
        k_long_narrow = min([s for s in strikes_all if s < k_short], key=lambda s: abs(s - (k_short - 3)), default=None)
        targets.append((cushion, k_short, k_long, k_long_narrow))

    results = []
    for exp_label, exp in [("monthly", monthly_exp), ("weekly", weekly_exp)]:
        if not exp:
            continue
        for cushion, k_short, k_long_std, k_long_narrow in targets:
            strikes_needed = {k_short, k_long_std, k_long_narrow} - {None}
            opts = [Option("SOXL", exp, k, "P", "SMART") for k in strikes_needed]
            qualified = ib.qualifyContracts(*opts)
            tickers = ib.reqTickers(*qualified)
            ib.sleep(2)
            quotes = {}
            for t in tickers:
                bid, ask, last = t.bid, t.ask, t.last
                mid = (bid + ask) / 2 if bid and ask and bid > 0 and ask > 0 else None
                quotes[t.contract.strike] = {"bid": bid, "ask": ask, "last": last, "mid": mid}
            short_q = quotes.get(k_short, {})
            long_std_q = quotes.get(k_long_std, {}) if k_long_std else {}
            long_narrow_q = quotes.get(k_long_narrow, {}) if k_long_narrow else {}

            def credit(short_q, long_q):
                if short_q.get("mid") and long_q.get("mid"):
                    return round(short_q["mid"] - long_q["mid"], 3)
                return None

            row = {
                "expiry": exp, "type": exp_label, "cushion_pct": cushion * 100,
                "short_strike": k_short, "short_quote": short_q,
                "long_strike_10pct_width": k_long_std, "long_quote_10pct_width": long_std_q,
                "credit_10pct_width": credit(short_q, long_std_q),
                "long_strike_narrow": k_long_narrow, "long_quote_narrow": long_narrow_q,
                "credit_narrow": credit(short_q, long_narrow_q),
            }
            results.append(row)
            print(f"{exp_label:>7} {exp} cushion={cushion*100:.0f}% short={k_short} "
                  f"bid/ask={short_q.get('bid')}/{short_q.get('ask')} "
                  f"| 10%-wide long={k_long_std} credit={row['credit_10pct_width']} "
                  f"| narrow long={k_long_narrow} credit={row['credit_narrow']}")

    import json
    with open("soxl_live_quotes_results.json", "w") as f:
        json.dump({"spot": spot, "results": results}, f, indent=2, default=str)
    print("\nSaved -> soxl_live_quotes_results.json")
    ib.disconnect()

if __name__ == "__main__":
    main()
