"""
Real, live check: what if EVC pushed ONLY the put side out to a wider
cushion (e.g. 1.2x/1.3x EM for short/long put) while leaving the call
side at the current 1.0x/1.5x EM -- motivated by tonight's real finding
that the default short put (327.5) sits on top of INTU's largest
negative-gamma wall (-1.75M net GEX at 325). Read-only, separate
clientId, no order placement.
"""
from datetime import date, timedelta
from ib_insync import IB, Stock, Option

TICKER = "INTU"

ib = IB()
ib.connect("127.0.0.1", 7496, clientId=73)

stk = Stock(TICKER, "SMART", "USD")
ib.qualifyContracts(stk)
tk = ib.reqMktData(stk, "", False, False)
ib.sleep(2)
spot = tk.last or tk.close
ib.cancelMktData(stk)

chains = ib.reqSecDefOptParams(TICKER, "", "STK", stk.conId)
chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
tomorrow = (date.today() + timedelta(days=1)).strftime("%Y%m%d")
expiry = sorted(e for e in chain.expirations if e >= tomorrow)[0]
strikes = sorted(chain.strikes)

def nearest(target):
    return min(strikes, key=lambda s: abs(s - target))

def spx_mid(td):
    b, a = td.bid, td.ask
    if b and a and b > 0 and a > 0:
        return (b + a) / 2
    return td.last or 0.0

atm_k = nearest(spot)
atm_c = Option(TICKER, expiry, atm_k, "C", "SMART", "100", "USD")
atm_p = Option(TICKER, expiry, atm_k, "P", "SMART", "100", "USD")
ib.qualifyContracts(atm_c, atm_p)
td_c = ib.reqMktData(atm_c, "", False, False)
td_p = ib.reqMktData(atm_p, "", False, False)
ib.sleep(3)
em = spx_mid(td_c) + spx_mid(td_p)
ib.cancelMktData(atm_c); ib.cancelMktData(atm_p)
print(f"spot=${spot:.2f}  EM=${em:.2f}  expiry={expiry}")

# Call side FIXED at current default (1.0x / 1.5x EM)
short_call_k = nearest(spot + 1.0 * em)
long_call_k  = nearest(spot + 1.5 * em)

def quote_condor(short_put_k, long_put_k, label):
    legs = {
        "lp": Option(TICKER, expiry, long_put_k,   "P", "SMART", "100", "USD"),
        "sp": Option(TICKER, expiry, short_put_k,  "P", "SMART", "100", "USD"),
        "sc": Option(TICKER, expiry, short_call_k, "C", "SMART", "100", "USD"),
        "lc": Option(TICKER, expiry, long_call_k,  "C", "SMART", "100", "USD"),
    }
    ib.qualifyContracts(*legs.values())
    if not all(c.conId for c in legs.values()):
        print(f"  {label}: could not qualify all legs"); return
    tds = {name: ib.reqMktData(c, "", False, False) for name, c in legs.items()}
    ib.sleep(4)
    mids = {name: spx_mid(td) for name, td in tds.items()}
    for c in legs.values():
        ib.cancelMktData(c)
    credit = round(mids["sp"] + mids["sc"] - mids["lp"] - mids["lc"], 2)
    put_width = short_put_k - long_put_k
    call_width = long_call_k - short_call_k
    width = min(put_width, call_width)
    max_risk = round(width * 100 - credit * 100, 2)
    print(f"  {label}: puts {long_put_k}/{short_put_k}  calls {short_call_k}/{long_call_k}  "
          f"credit=${credit:.2f}  width={width}  max_risk=${max_risk:.2f}  "
          f"pct_of_4316={max_risk/4316*100:.1f}%")

print("\n-- Current EVC default (symmetric 1.0x/1.5x both sides) --")
quote_condor(nearest(spot - 1.0*em), nearest(spot - 1.5*em), "put@1.0x/1.5x")

print("\n-- Asymmetric: put cushion widened, call side unchanged --")
for k_short in (1.1, 1.2, 1.3):
    short_put_k = nearest(spot - k_short * em)
    long_put_k  = nearest(spot - 1.5 * em)
    quote_condor(short_put_k, long_put_k, f"put short@{k_short}x (long fixed 1.5x)")

ib.disconnect()
