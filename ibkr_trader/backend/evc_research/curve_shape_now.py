"""
Real, live IV-curve-shape check for tonight's remaining EVC candidates
(INTU, KSS, SJM, LI -- ZM excluded per CEO decision 2026-08-25) using the
EXACT same quadratic-fit-in-log-moneyness logic just added to
_evc_pretrade_review in main.py. Read-only, separate clientId, no order
placement, no live backend state touched.
"""
from datetime import date, timedelta
import numpy as np
from ib_insync import IB, Stock, Option

TICKERS = ["INTU", "KSS", "SJM", "LI"]

ib = IB()
ib.connect("127.0.0.1", 7496, clientId=63)


def nearest(strikes, target):
    return min(strikes, key=lambda s: abs(s - target))


def spx_mid(td):
    b, a = td.bid, td.ask
    if b and a and b > 0 and a > 0:
        return (b + a) / 2
    return td.last or 0.0


def iv_of(td):
    mg = td.modelGreeks
    if mg and mg.impliedVol and mg.impliedVol > 0:
        return float(mg.impliedVol)
    return None


def curve_shape(spot, points):
    pts = [(k, iv) for k, iv in points if iv is not None]
    if len(pts) < 3:
        return {"available": False, "reason": f"only {len(pts)}/5 legs returned usable IV"}
    x = np.array([np.log(k / spot) for k, _ in pts])
    y = np.array([iv for _, iv in pts])
    a, b, c = np.polyfit(x, y, 2)
    shape = "CONCAVE (event-risk signature)" if a < 0 else "convex (typical smile)"
    return {"available": True, "curvature": round(float(a), 4), "shape": shape,
            "iv_by_strike": {str(k): round(iv, 4) for k, iv in pts}, "n_points": len(pts)}


for ticker in TICKERS:
    print(f"\n--- {ticker} ---")
    stk = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(stk)
    tk = ib.reqMktData(stk, "", False, False)
    ib.sleep(2)
    spot = tk.last or tk.close or spx_mid(tk)
    ib.cancelMktData(stk)
    if not spot or spot <= 0:
        print("  no spot -- skipping"); continue

    chains = ib.reqSecDefOptParams(ticker, "", "STK", stk.conId)
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
    tomorrow = (date.today() + timedelta(days=1)).strftime("%Y%m%d")
    future_expiries = sorted(e for e in chain.expirations if e >= tomorrow)
    expiry = future_expiries[0]
    strikes = sorted(chain.strikes)

    atm_k = nearest(strikes, spot)
    atm_c = Option(ticker, expiry, atm_k, "C", "SMART", "100", "USD")
    atm_p = Option(ticker, expiry, atm_k, "P", "SMART", "100", "USD")
    ib.qualifyContracts(atm_c, atm_p)
    td_c = ib.reqMktData(atm_c, "100,101,106", False, False)
    td_p = ib.reqMktData(atm_p, "100,101,106", False, False)
    ib.sleep(4)
    em = spx_mid(td_c) + spx_mid(td_p)
    iv_atm_c, iv_atm_p = iv_of(td_c), iv_of(td_p)
    ib.cancelMktData(atm_c); ib.cancelMktData(atm_p)
    if em <= 0:
        print("  no EM -- skipping"); continue
    print(f"  spot=${spot:.2f}  EM=${em:.2f}  im_pct={em/spot:.2%}  expiry={expiry}")

    long_put_k   = nearest(strikes, spot - 1.5 * em)
    short_put_k  = nearest(strikes, spot - 1.0 * em)
    short_call_k = nearest(strikes, spot + 1.0 * em)
    long_call_k  = nearest(strikes, spot + 1.5 * em)

    legs = {
        "lp": Option(ticker, expiry, long_put_k,   "P", "SMART", "100", "USD"),
        "sp": Option(ticker, expiry, short_put_k,  "P", "SMART", "100", "USD"),
        "sc": Option(ticker, expiry, short_call_k, "C", "SMART", "100", "USD"),
        "lc": Option(ticker, expiry, long_call_k,  "C", "SMART", "100", "USD"),
    }
    ib.qualifyContracts(*legs.values())
    tds = {name: ib.reqMktData(c, "100,101,106", False, False) for name, c in legs.items()}
    ib.sleep(4)
    ivs = {name: iv_of(td) for name, td in tds.items()}
    for c in legs.values():
        ib.cancelMktData(c)

    iv_atm = None
    if iv_atm_c is not None and iv_atm_p is not None:
        iv_atm = (iv_atm_c + iv_atm_p) / 2
    elif iv_atm_c is not None or iv_atm_p is not None:
        iv_atm = iv_atm_c if iv_atm_c is not None else iv_atm_p

    result = curve_shape(spot, [
        (long_put_k, ivs["lp"]), (short_put_k, ivs["sp"]), (atm_k, iv_atm),
        (short_call_k, ivs["sc"]), (long_call_k, ivs["lc"]),
    ])
    if result["available"]:
        print(f"  IV by strike: {result['iv_by_strike']}")
        print(f"  curvature={result['curvature']:+.4f}  -> {result['shape']}  ({result['n_points']}/5 legs)")
    else:
        print(f"  IV curve unavailable: {result['reason']}")

ib.disconnect()
print("\nDone.")
