"""
AXP Iron Condor monitor — close when profitable.
Entry credit: $0.63/sh = $63 total  (2x implied move structure)
Target: close at >= 50% of max profit ($31+), hard close at 3:45 PM ET.
Legs: LONG 315P / SHORT 320P / SHORT 360C / LONG 365C, expiry 20260724

AXP reports BMO Jul 24 — gap happens at open, IV crushes immediately.
Run every 10 min via Task Scheduler on Jul 24.
"""
import sys, io, time, json, os
from datetime import datetime
from zoneinfo import ZoneInfo
from ib_insync import IB, Option, LimitOrder

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PORT, CLIENT = 7496, 41
TICKER  = "AXP"
EXPIRY  = "20260724"
ET      = ZoneInfo("America/New_York")

ENTRY_CREDIT  = 0.63   # per share
MAX_PROFIT    = 63.0   # dollars
PROFIT_TARGET = 31.0   # 50% of max
HARD_CLOSE_HR = 15
HARD_CLOSE_MI = 45

LEGS = [
    (315.0, "P", "LONG",  "SELL"),
    (320.0, "P", "SHORT", "BUY"),
    (360.0, "C", "SHORT", "BUY"),
    (365.0, "C", "LONG",  "SELL"),
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
    o = Option(TICKER, EXPIRY, float(strike), right, "SMART", "100", "USD")
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
    print(f"\n=== CLOSING ALL LEGS ({reason}) ===")
    trades = []
    for strike, right, position, action in LEGS:
        c = qualify(ib, strike, right)
        if not c:
            print(f"  Could not qualify {strike}{right} — aborting")
            return None
        mid, bid, ask = get_quote(ib, c)
        lmt = round((ask or mid) + 0.01, 2) if action == "BUY" else max(round((bid or mid) - 0.01, 2), 0.01)
        trade = ib.placeOrder(c, LimitOrder(action, 1, lmt, tif="DAY"))
        label = f"{action} {strike:.0f}{right}"
        print(f"  Placed {label} @ ${lmt}")
        trades.append((trade, label, strike, right, action))

    fills = {}
    for trade, label, strike, right, action in trades:
        fill = wait_fill(ib, trade, label, timeout=90)
        fills[(strike, right, action)] = fill

    if all(v is not None for v in fills.values()):
        return fills
    return None

# ─────────────────────────────────────────────────────────────────────────────
now_et = datetime.now(ET)
print(f"AXP IC Monitor — {now_et.strftime('%Y-%m-%d %H:%M ET')}")

market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
if not (market_open <= now_et <= market_close):
    print("Market closed — nothing to do.")
    sys.exit(0)

hard_close_time = now_et.replace(hour=HARD_CLOSE_HR, minute=HARD_CLOSE_MI, second=0, microsecond=0)
force_close = now_et >= hard_close_time

ib = IB()
ib.connect("127.0.0.1", PORT, clientId=CLIENT)
ib.reqMarketDataType(1)
print("Connected to IBKR")

# ── Check positions still open ────────────────────────────────────────────────
ib.sleep(2)
portfolio = {
    (getattr(p.contract, 'strike', 0), getattr(p.contract, 'right', '')): p.position
    for p in ib.portfolio()
    if p.contract.symbol == TICKER
    and getattr(p.contract, 'lastTradeDateOrContractMonth', '') == EXPIRY
}
print(f"AXP positions: {portfolio}")

expected = {(315.0,'P'): 1, (320.0,'P'): -1, (360.0,'C'): -1, (365.0,'C'): 1}
if not all(portfolio.get(k, 0) == v for k, v in expected.items()):
    print("Condor already closed or missing legs — exiting.")
    send_telegram("<b>AXP IC Monitor</b>: Condor already closed or no positions found.")
    ib.disconnect()
    sys.exit(0)

# ── Live quotes ───────────────────────────────────────────────────────────────
print("\nFetching live quotes...")
quotes = {}
for strike, right, position, action in LEGS:
    c = qualify(ib, strike, right)
    mid, bid, ask = get_quote(ib, c)
    quotes[(strike, right)] = (mid, bid, ask)
    print(f"  {strike:.0f}{right} ({position}): mid=${mid}  b/a={bid}/{ask}")

# Cost to close: BUY back shorts at ask, SELL wings at bid
close_cost = round(
    (quotes[(320.0,'P')][2] or quotes[(320.0,'P')][0]) +
    (quotes[(360.0,'C')][2] or quotes[(360.0,'C')][0]) -
    (quotes[(315.0,'P')][1] or quotes[(315.0,'P')][0]) -
    (quotes[(365.0,'C')][1] or quotes[(365.0,'C')][0]),
    2
)
current_pnl = round((ENTRY_CREDIT - close_cost) * 100, 2)
pnl_pct     = round(current_pnl / MAX_PROFIT * 100, 1)

print(f"\nClose cost: ${close_cost:.2f}/sh")
print(f"Current P&L: ${current_pnl:.2f}  ({pnl_pct}% of max ${MAX_PROFIT:.0f})")
print(f"Force close at 3:45 PM: {force_close}")

should_close = current_pnl >= PROFIT_TARGET or force_close

if not should_close:
    status = "GREEN" if current_pnl > 0 else "RED"
    msg = (
        f"<b>AXP IC Status [{status}]</b> - {now_et.strftime('%H:%M ET')}\n\n"
        f"P&L: <b>${current_pnl:+.2f}</b>  ({pnl_pct}% of max)\n"
        f"Close cost: ${close_cost:.2f}/sh  (entry: ${ENTRY_CREDIT:.2f})\n\n"
        f"315P: ${quotes[(315.0,'P')][0]}  |  320P: ${quotes[(320.0,'P')][0]}\n"
        f"360C: ${quotes[(360.0,'C')][0]}  |  365C: ${quotes[(365.0,'C')][0]}\n\n"
        f"Target: +${PROFIT_TARGET:.0f}  |  Hard close: 3:45 PM\n"
        f"Checking again in ~10 min."
    )
    print("Not yet at target — sending status update.")
    send_telegram(msg)
    ib.disconnect()
    sys.exit(0)

# ── CLOSE ─────────────────────────────────────────────────────────────────────
reason = "3:45 PM hard close" if force_close else f"profit target hit (${current_pnl:.0f} >= ${PROFIT_TARGET:.0f})"
fills = close_all_legs(ib, reason)

if fills:
    buy_320p  = fills.get((320.0,'P','BUY'))  or quotes[(320.0,'P')][0]
    buy_360c  = fills.get((360.0,'C','BUY'))  or quotes[(360.0,'C')][0]
    sell_315p = fills.get((315.0,'P','SELL')) or quotes[(315.0,'P')][0]
    sell_365c = fills.get((365.0,'C','SELL')) or quotes[(365.0,'C')][0]

    actual_close = round((buy_320p + buy_360c) - (sell_315p + sell_365c), 2)
    actual_pnl   = round((ENTRY_CREDIT - actual_close) * 100, 2)

    print(f"\n=== AXP CONDOR CLOSED ===")
    print(f"Realized P&L: ${actual_pnl:.2f}")

    msg = (
        f"<b>AXP Iron Condor CLOSED</b>\n"
        f"Reason: {reason}\n\n"
        f"BUY  320P  @ ${buy_320p:.2f}\n"
        f"BUY  360C  @ ${buy_360c:.2f}\n"
        f"SELL 315P  @ ${sell_315p:.2f}\n"
        f"SELL 365C  @ ${sell_365c:.2f}\n\n"
        f"Entry credit:  ${ENTRY_CREDIT:.2f}/sh\n"
        f"Close cost:    ${actual_close:.2f}/sh\n"
        f"<b>Realized P&L: ${actual_pnl:+.2f}</b>\n"
        f"({round(actual_pnl/MAX_PROFIT*100,1)}% of max ${MAX_PROFIT:.0f})"
    )
    send_telegram(msg)
else:
    msg = (
        f"<b>AXP IC CLOSE FAILED</b> - {now_et.strftime('%H:%M ET')}\n"
        f"Could not fill all 4 legs. Check IBKR immediately!\n"
        f"Expiry: {EXPIRY}"
    )
    send_telegram(msg)
    print("CLOSE FAILED — check IBKR manually!")

ib.disconnect()
print("Done.")
