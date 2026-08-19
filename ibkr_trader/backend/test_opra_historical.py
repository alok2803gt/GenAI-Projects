"""
One-off research script (CIO/CEO request 2026-08-10): does IBKR's TWS API
actually return HISTORICAL bars for OPTION contracts via reqHistoricalData,
not just live OPRA quotes? Read-only market data, no orders. Uses its own
clientId (701) to avoid colliding with the live backend (clientId=3) or the
earlier soxl_live_quotes.py script (clientId=641).

Pattern for qualifying a real SOXL option contract borrowed from
soxl_live_quotes.py (reqSecDefOptParams -> pick expiry/strike -> qualifyContracts).
"""
import sys
from datetime import datetime
from ib_insync import IB, Stock, Option, util

CLIENT_ID = 733
TWS_PORT = 7496  # live account, per this codebase's standing config


def try_hist(ib, contract, whatToShow, durationStr, barSizeSetting, label):
    print(f"\n--- reqHistoricalData: whatToShow={whatToShow} duration={durationStr} "
          f"barSize={barSizeSetting} ({label}) ---")
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=durationStr,
            barSizeSetting=barSizeSetting,
            whatToShow=whatToShow,
            useRTH=True,
            formatDate=1,
            timeout=30,
        )
        if not bars:
            print("  -> EMPTY result (no error, but zero bars returned)")
        else:
            print(f"  -> {len(bars)} bars returned. First: {bars[0]}  Last: {bars[-1]}")
        return bars
    except Exception as e:
        print(f"  -> EXCEPTION: {type(e).__name__}: {e}")
        return None


def main():
    ib = IB()
    # capture reqHistoricalData / reqMktData error callbacks (IBKR sends
    # "errors" for things like no-permission or no-data that aren't Python exceptions)
    def on_error(reqId, errorCode, errorString, contract):
        print(f"  [errorCallback] reqId={reqId} code={errorCode} msg={errorString}")
    ib.errorEvent += on_error

    print(f"Connecting clientId={CLIENT_ID} port={TWS_PORT} ...")
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=30)
    print("Connected.", "Server time:", ib.reqCurrentTime())

    soxl = Stock("SOXL", "SMART", "USD")
    ib.qualifyContracts(soxl)
    [ticker] = ib.reqTickers(soxl)
    spot = ticker.marketPrice()
    if spot != spot:
        spot = ticker.last or ticker.close
    print(f"SOXL spot: {spot}")

    chains = ib.reqSecDefOptParams(soxl.symbol, "", soxl.secType, soxl.conId)
    chain = next(c for c in chains if c.exchange == "SMART")
    expiries = sorted(chain.expirations)
    strikes_all = sorted(chain.strikes)

    today = datetime.now()
    def days_out(e):
        return (datetime.strptime(e, "%Y%m%d") - today).days

    # pick a monthly-ish expiry ~30-45 days out, currently listed & liquid
    monthly_candidates = [e for e in expiries if 21 <= days_out(e) <= 60]
    exp = min(monthly_candidates, key=days_out) if monthly_candidates else expiries[0]
    print(f"Chosen expiry: {exp} ({days_out(exp)}d out)")

    # ATM-ish put strike
    k = min(strikes_all, key=lambda s: abs(s - spot))
    print(f"Chosen strike: {k} (put)")

    opt = Option("SOXL", exp, k, "P", "SMART")
    [opt] = ib.qualifyContracts(opt)
    print(f"Qualified option contract: {opt}")

    # Sanity: confirm live quote works on this contract (known-good path)
    [otk] = ib.reqTickers(opt)
    ib.sleep(2)
    print(f"Live quote sanity check -> bid={otk.bid} ask={otk.ask} last={otk.last} "
          f"(pre-market/off-hours values may be NaN/stale)")

    # --- Now the actual research questions ---
    # 1. Daily bars, several months, TRADES
    try_hist(ib, opt, "TRADES", "6 M", "1 day", "6mo daily TRADES on live contract")

    # 2. Daily bars, MIDPOINT (often used as proxy for options since TRADES can be sparse)
    try_hist(ib, opt, "MIDPOINT", "6 M", "1 day", "6mo daily MIDPOINT")

    # 3. Daily bars, BID_ASK
    try_hist(ib, opt, "BID_ASK", "6 M", "1 day", "6mo daily BID_ASK")

    # 4. Shorter window, intraday bars (30 min), TRADES -- see if intraday works even if EOD doesn't
    try_hist(ib, opt, "TRADES", "5 D", "30 mins", "5d 30min TRADES")

    # 5. Try 1 year duration to see how far back a currently-listed contract's
    #    history actually extends (should be capped by its own listing date,
    #    likely far less than 1 year given options typically list weeks-months
    #    before expiry, but test the boundary)
    try_hist(ib, opt, "TRADES", "1 Y", "1 day", "1yr daily TRADES (testing listing-date ceiling)")

    # 6. Try an OPTION_IMPLIED_VOLATILITY series
    try_hist(ib, opt, "OPTION_IMPLIED_VOLATILITY", "3 M", "1 day", "3mo daily IV series")

    # 7. Try an EXPIRED/near-expired contract to see if IBKR truly blocks it.
    #    Find the earliest (soonest) expiry, which may already be very close
    #    to/at expiration, and also try a strike far OTM (thinly traded) for contrast.
    near_exp = min(expiries, key=days_out)
    print(f"\nNearest expiry available in chain: {near_exp} ({days_out(near_exp)}d out) -- "
          f"NOT actually expired (IBKR chain won't list expired contracts), "
          f"used only as the closest-to-expiry comparison point.")

    ib.disconnect()
    print("\nDisconnected. Done.")


if __name__ == "__main__":
    main()
