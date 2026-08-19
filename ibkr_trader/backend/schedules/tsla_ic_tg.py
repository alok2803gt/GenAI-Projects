"""
TSLA Iron Condor — Monitor + Telegram Alert (permanent, session-independent)
Reads live prices from IBKR, assesses condor status, sends Telegram update.
Usage: python tsla_ic_tg.py [open|check|expiry]
"""
import sys, json, io, requests
from datetime import date, datetime
from ib_insync import IB, Stock, Option

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR   = __import__('pathlib').Path(__file__).parent
BACKEND_DIR  = SCRIPT_DIR.parent
SCANNER_CFG  = BACKEND_DIR / "scanner_config.json"
SCHEDULES_DB = SCRIPT_DIR / "session_schedules.json"

PORT, CLIENT = 7496, 26
TICKER, EXPIRY = "TSLA", "20260724"

LONG_345P_COST = 1.84
LONG_400C_COST = 2.77
SHORT_355P_TGT = 3.70
SHORT_390C_TGT = 5.04
WINGS_COST     = LONG_345P_COST + LONG_400C_COST  # 4.61

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"


def load_tg():
    try:
        cfg = json.loads(SCANNER_CFG.read_text())
        return cfg.get("telegram_token", ""), cfg.get("telegram_chat_id", "")
    except Exception as e:
        print(f"Could not load scanner_config.json: {e}")
        return "", ""


def tg_send(token, chat_id, text):
    if not token or not chat_id:
        print("Telegram not configured")
        return
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
        timeout=10,
    )
    print("Telegram OK" if r.ok else f"Telegram error {r.status_code}: {r.text}")


def quote(ib, strike, right):
    opt = Option(TICKER, EXPIRY, float(strike), right, "SMART", "100", "USD")
    q = ib.qualifyContracts(opt)
    if not q:
        return None
    [td] = ib.reqTickers(q[0])
    ib.sleep(1.5)
    bid = td.bid if (td.bid and td.bid > 0) else None
    ask = td.ask if (td.ask and td.ask > 0) else None
    return round((bid + ask) / 2, 2) if bid and ask else (td.last or td.close)


# ── Connect ───────────────────────────────────────────────────────────────────
print(f"[{datetime.now():%H:%M:%S}] TSLA condor check (mode={MODE})")
ib = IB()
ib.connect("127.0.0.1", PORT, clientId=CLIENT)
ib.reqMarketDataType(1)

# ── Spot ──────────────────────────────────────────────────────────────────────
stk = Stock(TICKER, "SMART", "USD")
ib.qualifyContracts(stk)
[td] = ib.reqTickers(stk)
ib.sleep(2)
spot = td.marketPrice() or td.last or td.close or 0
print(f"TSLA spot: ${spot:.2f}")

# ── Option quotes ─────────────────────────────────────────────────────────────
m345p = quote(ib, 345, "P")
m355p = quote(ib, 355, "P")
m390c = quote(ib, 390, "C")
m400c = quote(ib, 400, "C")

# ── Order / position status ───────────────────────────────────────────────────
ib.reqAllOpenOrders()
ib.sleep(2)

short_355p_status = "UNKNOWN"
short_390c_status = "UNKNOWN"
short_355p_fill   = None
short_390c_fill   = None

for trade in ib.openTrades():
    sym = getattr(trade.contract, 'localSymbol', '')
    if sym.endswith('P00355000'):
        short_355p_status = trade.orderStatus.status
    if sym.endswith('C00390000'):
        short_390c_status = trade.orderStatus.status

for pos in ib.portfolio():
    sym = getattr(pos.contract, 'localSymbol', '')
    if sym.endswith('P00355000') and pos.position < 0:
        short_355p_status = "Filled"
        short_355p_fill   = round(abs(pos.averageCost) / 100, 2) if pos.averageCost else None
    if sym.endswith('C00390000') and pos.position < 0:
        short_390c_status = "Filled"
        short_390c_fill   = round(abs(pos.averageCost) / 100, 2) if pos.averageCost else None

ib.disconnect()

# ── P&L ──────────────────────────────────────────────────────────────────────
short_credit    = (short_355p_fill or SHORT_355P_TGT) + (short_390c_fill or SHORT_390C_TGT)
net_basis       = WINGS_COST - short_credit
wings_now       = (m345p or 0) + (m400c or 0)
shorts_now      = (m355p or 0) + (m390c or 0)
pnl_now         = round((wings_now - shorts_now - net_basis) * 100, 0)
max_profit      = round(-net_basis * 100, 0)
days_left       = (date(2026, 7, 24) - date.today()).days
shorts_filled   = short_355p_fill and short_390c_fill

# ── Status ───────────────────────────────────────────────────────────────────
if not shorts_filled and short_355p_status not in ("Filled",) and short_390c_status not in ("Filled",):
    status_emoji = "⏳"
    status_label = f"SHORT LEGS PENDING ({short_355p_status}/{short_390c_status})"
elif spot < 345:
    status_emoji = "🚨"
    status_label = "RED — BELOW WING (close put spread NOW)"
elif spot < 355:
    status_emoji = "🟠"
    status_label = f"ORANGE — ${355-spot:.1f} below short put ($355)"
elif spot > 400:
    status_emoji = "🚨"
    status_label = "RED — ABOVE WING (close call spread NOW)"
elif spot > 390:
    status_emoji = "🟠"
    status_label = f"ORANGE — ${spot-390:.1f} above short call ($390)"
else:
    status_emoji = "✅"
    status_label = "GREEN — inside $355–$390 range"

# ── Telegram message ──────────────────────────────────────────────────────────
headers = {
    "open":   f"<b>TSLA Condor — Market Open {date.today()}</b>",
    "check":  f"<b>TSLA Condor — Intraday Check {datetime.now():%H:%M}</b>",
    "expiry": f"<b>TSLA Condor — EXPIRY FINAL (16 min to close!)</b>",
}
header = headers.get(MODE, headers["check"])

order_line = (
    f"\n<b>Orders:</b> 355P {short_355p_status} | 390C {short_390c_status}"
    if not shorts_filled else
    f"\n<b>Fills:</b> 355P @ ${short_355p_fill:.2f} | 390C @ ${short_390c_fill:.2f}"
)

msg = (
    f"{header}\n\n"
    f"{status_emoji} <b>{status_label}</b>\n\n"
    f"<b>TSLA:</b> ${spot:.2f}    DTE: {days_left}{order_line}\n\n"
    f"<b>Mids:</b>  345P ${m345p}  355P ${m355p}  390C ${m390c}  400C ${m400c}\n\n"
    f"<b>P&amp;L:</b> {'+' if pnl_now >= 0 else ''}${pnl_now:.0f} unrealized   "
    f"<b>Max profit:</b> ${max_profit:.0f}\n"
)

if spot < 345 or spot > 400:
    msg += "\n<b>ACTION: Close breached spread — wing ITM, stop bleeding.</b>"
elif spot < 355:
    msg += f"\n<b>WARNING:</b> 355P short is ITM. Consider buying it back."
elif spot > 390:
    msg += f"\n<b>WARNING:</b> 390C short is ITM. Consider buying it back."
elif MODE == "expiry" and 355 <= spot <= 390:
    msg += f"\n\n<b>MAX PROFIT — both shorts expire worthless. +${max_profit:.0f} collected.</b>"
else:
    msg += "\nNo action needed."

print(msg)
token, chat_id = load_tg()
tg_send(token, chat_id, msg)

# ── Expiry auto-close: buy back any ITM short before exercise ─────────────────
if MODE == "expiry" and spot > 0:
    from ib_insync import LimitOrder, MarketOrder
    itm_closes = []
    if spot < 355:   # 355P is ITM — buy it back
        itm_closes.append((355, "P", "355P short ITM"))
    if spot > 390:   # 390C is ITM — buy it back
        itm_closes.append((390, "C", "390C short ITM"))

    if itm_closes:
        print(f"\nAUTO-CLOSE: {len(itm_closes)} ITM short(s) — connecting for close orders")
        ib2 = IB()
        ib2.connect("127.0.0.1", PORT, clientId=CLIENT + 1)
        ib2.reqMarketDataType(1)
        close_results = []
        for strike, right, label in itm_closes:
            opt = Option(TICKER, EXPIRY, float(strike), right, "SMART", "100", "USD")
            q = ib2.qualifyContracts(opt)
            if not q:
                close_results.append(f"{label}: QUALIFY FAILED")
                continue
            order = MarketOrder("BUY", 1, tif="DAY")
            trade = ib2.placeOrder(q[0], order)
            for _ in range(20):
                ib2.sleep(1)
                if trade.orderStatus.status in ("Filled", "Cancelled", "Inactive"):
                    break
            st   = trade.orderStatus.status
            fill = trade.orderStatus.avgFillPrice
            close_results.append(f"{label}: {st} @ ${fill:.2f}" if fill else f"{label}: {st}")
            print(f"  {label} → {st} fill=${fill}")

        ib2.disconnect()
        close_summary = "\n".join(close_results)
        follow_up = f"<b>AUTO-CLOSE executed (expiry day):</b>\n{close_summary}"
        print(follow_up)
        tg_send(token, chat_id, follow_up)
