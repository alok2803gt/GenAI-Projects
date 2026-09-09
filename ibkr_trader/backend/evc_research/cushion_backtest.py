"""
Real, read-only cushion-multiple analysis for tonight's 5 real EVC candidates
(INTU, ZM, KSS, SJM, LI) -- does NOT touch main.py, EVC's live config, or
place any order. Separate IBKR clientId, separate process.

Question: EVC currently places short strikes at exactly 1.0x the live
implied move (ATM straddle price). Would pushing that out to 1.2x/1.5x/2.0x
meaningfully reduce how often a real historical earnings move would have
breached the structure, and what would it actually cost in real credit?

Method (every number here is real, not simulated):
  1. Pull TODAY's REAL live implied move for each ticker via the same ATM
     straddle method _evc_quote_condor uses (options chains are live now,
     hours before EVC's own 15:00 entry-window gate -- that gate is a
     strategy design choice, not a market-data constraint).
  2. Quote REAL live premiums at 1.0x/1.2x/1.5x/2.0x EM strikes on both
     wings, so the credit given up per cushion level is an observed
     price, not a Black-Scholes guess.
  3. Pull each ticker's own REAL historical earnings-day % moves via
     yfinance (same method _evc_pretrade_review already uses).
  4. For each cushion multiple k, containment = fraction of that ticker's
     own past earnings moves that stayed within +/- k * (today's real
     im_pct). This is the SAME formula _evc_pretrade_review already runs
     at k=1 -- just swept across k, using today's real EM as the scale.

Honest limitation stated up front: this uses TODAY's real implied move as
the assumed scale for PAST quarters too (no historical point-in-time IV
exists for these tickers -- same data wall already hit with DKS). That's a
real approximation, not a true multi-quarter implied-vs-realized backtest.
"""
import sys
from datetime import date, timedelta
import numpy as np
import yfinance as yf
from ib_insync import IB, Stock, Option

TICKERS = ["INTU", "ZM", "KSS", "SJM", "LI"]
CUSHION_MULTIPLES = [1.0, 1.2, 1.5, 2.0]

ib = IB()
ib.connect("127.0.0.1", 7496, clientId=61)


def nearest(strikes, target):
    return min(strikes, key=lambda s: abs(s - target))


def spx_mid(td):
    b, a = td.bid, td.ask
    if b and a and b > 0 and a > 0:
        return (b + a) / 2
    return td.last or 0.0


def get_quote_and_em(ticker):
    stk = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(stk)
    if not stk.conId:
        return None
    tk = ib.reqMktData(stk, "", False, False)
    ib.sleep(2)
    spot = tk.last or tk.close or spx_mid(tk)
    ib.cancelMktData(stk)
    if not spot or spot <= 0:
        return None

    chains = ib.reqSecDefOptParams(ticker, "", "STK", stk.conId)
    if not chains:
        return None
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
    tomorrow = (date.today() + timedelta(days=1)).strftime("%Y%m%d")
    future_expiries = sorted(e for e in chain.expirations if e >= tomorrow)
    if not future_expiries:
        return None
    expiry = future_expiries[0]
    strikes = sorted(chain.strikes)

    atm_k = nearest(strikes, spot)
    atm_c = Option(ticker, expiry, atm_k, "C", "SMART", "100", "USD")
    atm_p = Option(ticker, expiry, atm_k, "P", "SMART", "100", "USD")
    ib.qualifyContracts(atm_c, atm_p)
    if not atm_c.conId or not atm_p.conId:
        return None
    td_c = ib.reqMktData(atm_c, "", False, False)
    td_p = ib.reqMktData(atm_p, "", False, False)
    ib.sleep(3)
    em = spx_mid(td_c) + spx_mid(td_p)
    ib.cancelMktData(atm_c)
    ib.cancelMktData(atm_p)
    if em <= 0:
        return None

    return {"ticker": ticker, "spot": spot, "expiry": expiry, "em": round(em, 2),
            "im_pct": round(em / spot, 4), "strikes": strikes}


def quote_wing_credit(ticker, expiry, strikes, spot, em, k):
    """Real live credit for a k*EM short strike (long leg fixed at 1.5x EM,
    same wing_mult EVC already uses, so only the SHORT-strike cushion varies)."""
    short_put_k  = nearest(strikes, spot - k * em)
    long_put_k   = nearest(strikes, spot - 1.5 * em)
    short_call_k = nearest(strikes, spot + k * em)
    long_call_k  = nearest(strikes, spot + 1.5 * em)
    if long_put_k >= short_put_k or short_call_k >= long_call_k:
        return None

    legs = {
        "sp": Option(ticker, expiry, short_put_k,  "P", "SMART", "100", "USD"),
        "lp": Option(ticker, expiry, long_put_k,   "P", "SMART", "100", "USD"),
        "sc": Option(ticker, expiry, short_call_k, "C", "SMART", "100", "USD"),
        "lc": Option(ticker, expiry, long_call_k,  "C", "SMART", "100", "USD"),
    }
    ib.qualifyContracts(*legs.values())
    if not all(c.conId for c in legs.values()):
        return None
    tds = {name: ib.reqMktData(c, "", False, False) for name, c in legs.items()}
    ib.sleep(3)
    mids = {name: spx_mid(td) for name, td in tds.items()}
    for c in legs.values():
        ib.cancelMktData(c)
    credit = round(mids["sp"] + mids["sc"] - mids["lp"] - mids["lc"], 2)
    width = min(short_put_k - long_put_k, long_call_k - short_call_k)
    max_risk = round(width * 100 - credit * 100, 2)
    return {"short_put": short_put_k, "short_call": short_call_k,
            "credit": credit, "max_risk": max_risk}


def historical_moves(ticker):
    t = yf.Ticker(ticker)
    cal = t.get_earnings_dates(limit=9)
    hist = t.history(period="3y", interval="1d")
    hist.index = hist.index.tz_localize(None)
    moves = []
    if cal is not None and len(hist):
        for ed in cal.index[1:]:
            ed_naive = ed.tz_localize(None) if ed.tzinfo else ed
            idx = hist.index.searchsorted(ed_naive)
            if 0 < idx < len(hist):
                prior = hist["Close"].iloc[idx - 1]
                react = hist["Close"].iloc[idx]
                moves.append(react / prior - 1)
    return moves


print(f"{'='*100}")
for ticker in TICKERS:
    print(f"\n--- {ticker} ---")
    q = get_quote_and_em(ticker)
    if not q:
        print("  Could not get a real live quote/EM -- skipping.")
        continue
    print(f"  spot=${q['spot']:.2f}  real live EM=${q['em']:.2f}  im_pct={q['im_pct']:.2%}  expiry={q['expiry']}")

    moves = historical_moves(ticker)
    if not moves:
        print("  No historical earnings-move data available.")
        continue
    print(f"  {len(moves)} real historical earnings moves: "
          f"{[f'{m:+.1%}' for m in moves]}")

    for k in CUSHION_MULTIPLES:
        contained = sum(1 for m in moves if abs(m) <= k * q["im_pct"])
        rate = contained / len(moves)
        wing = quote_wing_credit(ticker, q["expiry"], q["strikes"], q["spot"], q["em"], k)
        if wing:
            print(f"  k={k:.1f}x EM: strikes {wing['short_put']}P/{wing['short_call']}C  "
                  f"real credit=${wing['credit']:.2f}  max_risk=${wing['max_risk']:.2f}  "
                  f"historical containment {contained}/{len(moves)} = {rate:.0%}")
        else:
            print(f"  k={k:.1f}x EM: could not get real wing quote -- "
                  f"historical containment {contained}/{len(moves)} = {rate:.0%} (credit n/a)")

ib.disconnect()
print(f"\n{'='*100}\nDone.")
