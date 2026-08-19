"""
SOXL real live IBKR quotes, v2 -- uses a proper STREAMING subscription
(reqMktData, snapshot=False) + wait + read + cancel, instead of the
reqTickersAsync-based snapshot approach that was found to hang indefinitely
for non-actively-streamed contracts (see oversight_log.jsonl 2026-08-10,
task 2026-08-10-005). This mirrors the pattern the account's own
subscribe_ticker() uses for its persistent (working) subscriptions.
"""
from ib_insync import IB, Stock, Option
import time

CLIENT_ID = 1455
TWS_PORT = 7496


def main():
    ib = IB()
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    print("Connected.")

    soxl = Stock("SOXL", "SMART", "USD")
    ib.qualifyContracts(soxl)

    # Streaming subscribe (not snapshot) for the stock itself
    stk_ticker = ib.reqMktData(soxl, "", False, False)
    ib.sleep(4)
    spot = stk_ticker.marketPrice()
    if spot != spot:  # NaN
        spot = (stk_ticker.bid + stk_ticker.ask) / 2 if stk_ticker.bid and stk_ticker.ask else stk_ticker.last
    print(f"SOXL spot (streaming): {spot}")
    ib.cancelMktData(soxl)

    chains = ib.reqSecDefOptParams(soxl.symbol, "", soxl.secType, soxl.conId)
    chain = next(c for c in chains if c.exchange == "SMART")
    expiries = sorted(chain.expirations)
    strikes_all = sorted(chain.strikes)

    from datetime import datetime
    today = datetime.now()
    def days_out(e):
        return (datetime.strptime(e, "%Y%m%d") - today).days
    monthly = [e for e in expiries if 21 <= days_out(e) <= 45]
    exp = min(monthly, key=days_out) if monthly else expiries[0]
    print(f"Monthly expiry: {exp} ({days_out(exp)}d out)")

    pairs = []
    for cushion in [0.15, 0.20, 0.25]:
        target_short = spot * (1 - cushion)
        short_k = min(strikes_all, key=lambda s: abs(s - target_short))
        target_long = short_k - 3
        long_k = min([s for s in strikes_all if s < short_k], key=lambda s: abs(s - target_long))
        pairs.append((cushion, short_k, long_k))

    strikes_needed = sorted({k for _, s, l in pairs for k in (s, l)})
    contracts = [Option("SOXL", exp, k, "P", "SMART") for k in strikes_needed]
    qualified = ib.qualifyContracts(*contracts)
    qualified = [c for c in qualified if c.conId]
    print(f"Qualified {len(qualified)} contracts, requesting streaming market data...")

    tickers = {}
    for c in qualified:
        tickers[c.strike] = ib.reqMktData(c, "", False, False)

    # Give the streaming ticks time to arrive
    ib.sleep(6)

    print("\n--- Real streaming quotes ---")
    quotes = {}
    for strike, tk in sorted(tickers.items()):
        bid = tk.bid if tk.bid and tk.bid == tk.bid and tk.bid > 0 else None
        ask = tk.ask if tk.ask and tk.ask == tk.ask and tk.ask > 0 else None
        last = tk.last if tk.last and tk.last == tk.last and tk.last > 0 else None
        mid = round((bid + ask) / 2, 3) if bid and ask else None
        quotes[strike] = {"bid": bid, "ask": ask, "last": last, "mid": mid}
        print(f"  Strike {strike}: bid={bid} ask={ask} last={last} mid={mid}")

    print(f"\nSpot: {spot:.2f}  Expiry: {exp}")
    for cushion, short_k, long_k in pairs:
        sq, lq = quotes.get(short_k, {}), quotes.get(long_k, {})
        smid, lmid = sq.get("mid"), lq.get("mid")
        credit = round(smid - lmid, 3) if smid is not None and lmid is not None else None
        width = short_k - long_k
        max_loss = round(width * 100 - (credit or 0) * 100, 2) if credit is not None else None
        print(f"{int(cushion*100)}% cushion: SELL {short_k}P (mid={smid}) / BUY {long_k}P (mid={lmid}) "
              f"-> credit={credit} width={width} max_loss={max_loss}")

    for c in qualified:
        ib.cancelMktData(c)
    ib.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
