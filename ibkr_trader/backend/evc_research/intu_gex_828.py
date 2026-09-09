"""
Real GEX/VEX + call/put OI ratio for INTU's specific 8/28/2026 expiry --
the exact contract EVC actually quoted tonight's condor against. The
standing gex-vex-calculator skill only covers a 25-45 DTE window (this
expiry is 2-3 DTE, an earnings-specific weekly), so this is a one-off,
same methodology (dollar_gamma = gamma * OI * 100 * spot^2 * 0.01, net GEX
= sum(call_gex) - sum(put_gex), OI-based not volume-based).
Read-only, separate clientId, no live backend/state touched.
"""
from ib_insync import IB, Stock, Option

TICKER = "INTU"
EXPIRY = "20260828"
BAND = 15

ib = IB()
ib.connect("127.0.0.1", 7496, clientId=71)

stk = Stock(TICKER, "SMART", "USD")
ib.qualifyContracts(stk)
tk = ib.reqMktData(stk, "", False, False)
ib.sleep(2)
spot = tk.last or tk.close
ib.cancelMktData(stk)
print(f"spot=${spot:.2f}  expiry={EXPIRY}")

chains = ib.reqSecDefOptParams(TICKER, "", "STK", stk.conId)
chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
all_strikes = sorted(chain.strikes)
atm_k = min(all_strikes, key=lambda s: abs(s - spot))
atm_idx = all_strikes.index(atm_k)
band_strikes = all_strikes[max(0, atm_idx - BAND): atm_idx + BAND + 1]

def safe(v):
    try:
        return float(v) if v is not None else 0.0
    except Exception:
        return 0.0

by_strike = []
total_call_oi = 0
total_put_oi = 0
for k in band_strikes:
    row = {"strike": k, "call_gex": 0.0, "put_gex": 0.0, "call_oi": 0, "put_oi": 0}
    for right in ("C", "P"):
        c = Option(TICKER, EXPIRY, k, right, "SMART", "100", "USD")
        ib.qualifyContracts(c)
        if not c.conId:
            continue
        td = ib.reqMktData(c, "101", False, False)
        ib.sleep(1.5)
        oi = safe(td.callOpenInterest if right == "C" else td.putOpenInterest)
        mg = td.modelGreeks
        ib.cancelMktData(c)
        if oi <= 0 or mg is None or mg.gamma is None:
            continue
        dollar_gamma = mg.gamma * oi * 100 * (spot ** 2) * 0.01
        if right == "C":
            row["call_gex"] = dollar_gamma
            row["call_oi"] = int(oi)
            total_call_oi += int(oi)
        else:
            row["put_gex"] = -dollar_gamma
            row["put_oi"] = int(oi)
            total_put_oi += int(oi)
    row["net_gex"] = round(row["call_gex"] + row["put_gex"], 0)
    if row["call_oi"] or row["put_oi"]:
        by_strike.append(row)
        print(f"  {k:>7.1f}: call_oi={row['call_oi']:>5} put_oi={row['put_oi']:>5}  net_gex={row['net_gex']:>+14,.0f}")

ib.disconnect()

total_gex = sum(r["net_gex"] for r in by_strike)
regime = "positive_gamma" if total_gex >= 0 else "negative_gamma"
pc_ratio = round(total_put_oi / total_call_oi, 3) if total_call_oi else None

print(f"\n=== Summary: INTU {EXPIRY} ===")
print(f"Total call OI: {total_call_oi}   Total put OI: {total_put_oi}   Put/Call OI ratio: {pc_ratio}")
print(f"Net GEX: {total_gex:+,.0f}   Regime: {regime}")

if by_strike:
    top_pos = max(by_strike, key=lambda r: r["net_gex"])
    top_neg = min(by_strike, key=lambda r: r["net_gex"])
    print(f"Largest POSITIVE gamma wall: strike {top_pos['strike']}  net_gex={top_pos['net_gex']:+,.0f}")
    print(f"Largest NEGATIVE gamma wall: strike {top_neg['strike']}  net_gex={top_neg['net_gex']:+,.0f}")
