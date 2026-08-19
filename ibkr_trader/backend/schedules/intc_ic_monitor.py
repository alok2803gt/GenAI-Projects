"""
INTC Iron Condor monitor — close when profitable.
Entry credit: $1.77/sh = $177 total
Target: close at >= 50% of max profit ($88+), hard close at 3:45 PM ET.
Legs: LONG 85P / SHORT 90P / SHORT 112C / LONG 117C, expiry 20260724

Run every 10 min via Task Scheduler on Jul 24.
All 4 close orders submitted simultaneously (wings already in account, no margin issue).
"""
import sys, io, time, json, os
from datetime import datetime
from zoneinfo import ZoneInfo
from ib_insync import IB, Option, LimitOrder

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PORT, CLIENT = 7496, 37
TICKER  = "INTC"
EXPIRY  = "20260724"
ET      = ZoneInfo("America/New_York")

ENTRY_CREDIT  = 1.77   # per share (net credit collected)
MAX_PROFIT    = 177.0  # dollars
PROFIT_TARGET = 88.0   # 50% of max — close when P&L >= this
HARD_CLOSE_HR = 15     # 3 PM ET hard close trigger hour
HARD_CLOSE_MI = 45     # 3:45 PM ET

# All 4 legs
LEGS = [
    (85.0,  "P", "LONG",  "SELL"),   # wing — sell to close
    (90.0,  "P", "SHORT", "BUY"),    # short — buy to close
    (112.0, "C", "SHORT", "BUY"),    # short — buy to close
    (117.0, "C", "LONG",  "SELL"),   # wing — sell to close
]

SCANNER_CFG = os.path.join(os.path.dirname(__file__), "..", "scanner_config.json")

def send_telegram(msg):
    try:
        import requests
        with open(SCANNER_CFG) as f:
            cfg = json.load(f)
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception as e:
        print(f"Telegram failed: {e}")

def qualify(ib, strike, right):
    from ib_insync import Option as Opt
    o = Opt(TICKER, EXPIRY, float(strike), right, "SMART", "100", "USD")
    q = ib.qualifyContracts(o)
    return q[0] if q else None

def get_quote(ib, contract):
    [td] = ib.reqTickers(contract)
    ib.sleep(1.2)
    bid = td.bid if (td.bid and td.bid > 0) else None
    ask = td.ask if (td.ask and td.ask > 0) else None
    mid = round((bid + ask) / 2, 2) if (bid and ask) else (td.last or td.close or 0)
    return mid, bid, ask

def wait_fill(ib, trade, label, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        ib.sleep(2)
        st = trade.orderStatus.status
        if st == "Filled":
            print(f"  {label}: FILLED @ ${trade.orderStatus.avgFillPrice:.2f}")
            return trade.orderStatus.avgFillPrice
        if st in ("Cancelled", "ApiCancelled", "Inactive"):
            print(f"  {label}: {st}")
            return None
        print(f"  {label}: {st} ...", end="\r")
    print(f"\n  {label}: TIMEOUT")
    return None

def close_all_legs(ib, reason):
    """Submit all 4 close orders simultaneously, then wait for fills."""
    print(f"\n=== CLOSING ALL LEGS ({reason}) ===")
    trades = []
    contracts = []
    for strike, right, position, action in LEGS:
        c = qualify(ib, strike, right)
        if not c:
            print(f"  Could not qualify {strike}{right} — aborting close")
            return None
        mid, bid, ask = get_quote(ib, c)
        if action == "BUY":
            lmt = round((ask or mid) + 0.01, 2)
        else:
            lmt = max(round((bid or mid) - 0.01, 2), 0.01)
        order = LimitOrder(action, 1, lmt, tif="DAY")
        trade = ib.placeOrder(c, order)
        label = f"{action} {strike}{right}"
        print(f"  Placed {label} @ ${lmt}")
        trades.append((trade, label))
        contracts.append((strike, right, action, lmt))

    fills = {}
    for trade, label in trades:
        fill = wait_fill(ib, trade, label, timeout=90)
        fills[label] = fill

    filled_all = all(v is not None for v in fills.values())
    return fills if filled_all else None

# ─────────────────────────────────────────────────────────────────────────────
now_et = datetime.now(ET)
print(f"INTC IC Monitor — {now_et.strftime('%Y-%m-%d %H:%M ET')}")

# Check if market is open (9:30–16:00 ET)
market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
if not (market_open <= now_et <= market_close):
    print("Market closed — nothing to do.")
    sys.exit(0)

hard_close_time = now_et.replace(hour=HARD_CLOSE_HR, minute=HARD_CLOSE_MI, second=0, microsecond=0)
force_close = now_et >= hard_close_time

ib = IB()
ib.connect("127.0.0.1", PORT, clientId=CLIENT)
ib.reqMarketDataType(1)
print("Connected to IBKR")

# ── Check portfolio — are positions still open? ───────────────────────────────
ib.sleep(2)
portfolio = {
    (getattr(p.contract, 'strike', 0), getattr(p.contract, 'right', '')): p.position
    for p in ib.portfolio()
    if p.contract.symbol == TICKER and getattr(p.contract, 'lastTradeDateOrContractMonth', '') == EXPIRY
}

print(f"INTC positions in account: {portfolio}")

# Check if all legs still open
expected = {(85.0, 'P'): 1, (90.0, 'P'): -1, (112.0, 'C'): -1, (117.0, 'C'): 1}
legs_open = all(portfolio.get(k, 0) == v for k, v in expected.items())

if not legs_open:
    print("Condor already closed or partially filled — exiting.")
    send_telegram("<b>INTC IC Monitor</b>: Condor already closed or no positions found. Skipping.")
    ib.disconnect()
    sys.exit(0)

# ── Get live quotes for P&L calculation ──────────────────────────────────────
print("\nFetching live quotes...")
quotes = {}
for strike, right, position, action in LEGS:
    c = qualify(ib, strike, right)
    mid, bid, ask = get_quote(ib, c)
    quotes[(strike, right)] = (mid, bid, ask)
    print(f"  {strike}{right} ({position}): mid=${mid}  b/a={bid}/{ask}")

# Cost to close = buy back shorts + sell wings
# BUY 90P @ ask, BUY 112C @ ask, SELL 85P @ bid, SELL 117C @ bid
close_cost = round(
    (quotes[(90.0, 'P')][2] or quotes[(90.0, 'P')][0]) +   # buy 90P ask
    (quotes[(112.0, 'C')][2] or quotes[(112.0, 'C')][0]) -  # buy 112C ask
    (quotes[(85.0, 'P')][1] or quotes[(85.0, 'P')][0]) -    # sell 85P bid
    (quotes[(117.0, 'C')][1] or quotes[(117.0, 'C')][0]),   # sell 117C bid
    2
)
current_pnl = round((ENTRY_CREDIT - close_cost) * 100, 2)
pnl_pct     = round(current_pnl / MAX_PROFIT * 100, 1)

print(f"\nClose cost: ${close_cost:.2f}/sh")
print(f"Current P&L: ${current_pnl:.2f}  ({pnl_pct}% of max ${MAX_PROFIT:.0f})")
print(f"Force close at 3:45 PM: {force_close}")

# ── Decision ──────────────────────────────────────────────────────────────────
should_close = current_pnl >= PROFIT_TARGET or force_close

if not should_close:
    # Report status only
    status = "GREEN" if current_pnl > 0 else "RED"
    msg = (
        f"<b>INTC IC Status [{status}]</b> - {now_et.strftime('%H:%M ET')}\n\n"
        f"P&L: <b>${current_pnl:+.2f}</b>  ({pnl_pct}% of max)\n"
        f"Close cost: ${close_cost:.2f}/sh  (entry: ${ENTRY_CREDIT:.2f})\n\n"
        f"85P: ${quotes[(85.0,'P')][0]}  |  90P: ${quotes[(90.0,'P')][0]}\n"
        f"112C: ${quotes[(112.0,'C')][0]}  |  117C: ${quotes[(117.0,'C')][0]}\n\n"
        f"Target: +${PROFIT_TARGET:.0f}  |  Hard close: 3:45 PM\n"
        f"Checking again in ~10 min."
    )
    print("Not yet at target — sending status update.")
    send_telegram(msg)
    ib.disconnect()
    sys.exit(0)

# ── CLOSE ────────────────────────────────────────────────────────────────────
reason = "3:45 PM hard close" if force_close else f"profit target hit (${current_pnl:.0f} >= ${PROFIT_TARGET:.0f})"
fills = close_all_legs(ib, reason)

if fills:
    # Calculate actual realized P&L from fills
    buy_90p  = fills.get("BUY 90.0P") or quotes[(90.0, 'P')][0]
    buy_112c = fills.get("BUY 112.0C") or quotes[(112.0, 'C')][0]
    sell_85p = fills.get("SELL 85.0P") or quotes[(85.0, 'P')][0]
    sell_117c = fills.get("SELL 117.0C") or quotes[(117.0, 'C')][0]

    actual_close_cost = round((buy_90p + buy_112c) - (sell_85p + sell_117c), 2)
    actual_pnl = round((ENTRY_CREDIT - actual_close_cost) * 100, 2)

    print(f"\n=== CONDOR CLOSED ===")
    print(f"Actual close cost: ${actual_close_cost:.2f}/sh")
    print(f"Realized P&L: ${actual_pnl:.2f}")

    msg = (
        f"<b>INTC Iron Condor CLOSED</b>\n"
        f"Reason: {reason}\n\n"
        f"BUY  90P  @ ${buy_90p:.2f}\n"
        f"BUY  112C @ ${buy_112c:.2f}\n"
        f"SELL 85P  @ ${sell_85p:.2f}\n"
        f"SELL 117C @ ${sell_117c:.2f}\n\n"
        f"Entry credit:  ${ENTRY_CREDIT:.2f}/sh\n"
        f"Close cost:    ${actual_close_cost:.2f}/sh\n"
        f"<b>Realized P&L: ${actual_pnl:+.2f}</b>\n"
        f"({round(actual_pnl/MAX_PROFIT*100,1)}% of max ${MAX_PROFIT:.0f})"
    )
    send_telegram(msg)
else:
    msg = (
        f"<b>INTC IC CLOSE FAILED</b> - {now_et.strftime('%H:%M ET')}\n"
        f"Could not fill all 4 legs. Check IBKR immediately!\n"
        f"Expiry: {EXPIRY}"
    )
    send_telegram(msg)
    print("CLOSE FAILED — check IBKR manually!")

ib.disconnect()
print("Done.")
