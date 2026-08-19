"""
Sell GOOG 260731 400C at market open.
Places limit at bid (or $0.01 if no bid), sends Telegram with result.
"""
import sys, io, json, requests
from datetime import datetime
from ib_insync import IB, Option, LimitOrder

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR  = __import__('pathlib').Path(__file__).parent
SCANNER_CFG = SCRIPT_DIR.parent / "scanner_config.json"

PORT, CLIENT = 7496, 29
TICKER, EXPIRY, STRIKE, RIGHT = "GOOG", "20260731", 400.0, "C"
ENTRY_PRICE = 0.60

def load_tg():
    try:
        cfg = json.loads(SCANNER_CFG.read_text())
        return cfg.get("telegram_token", ""), cfg.get("telegram_chat_id", "")
    except:
        return "", ""

def tg(token, chat_id, text):
    if not token or not chat_id: return
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                  json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)

print(f"[{datetime.now():%H:%M:%S}] GOOG 400C sell at open")

ib = IB()
ib.connect("127.0.0.1", PORT, clientId=CLIENT)
ib.reqMarketDataType(1)

opt = Option(TICKER, EXPIRY, STRIKE, RIGHT, "SMART", "100", "USD")
qualified = ib.qualifyContracts(opt)
if not qualified:
    msg = "GOOG 400C sell FAILED — could not qualify contract"
    print(msg)
    token, chat_id = load_tg()
    tg(token, chat_id, f"<b>GOOG 400C</b>\n{msg}")
    ib.disconnect(); sys.exit(1)

opt = qualified[0]
[td] = ib.reqTickers(opt)
ib.sleep(3)

bid  = td.bid  if td.bid  and td.bid  > 0 else None
ask  = td.ask  if td.ask  and td.ask  > 0 else None
last = td.last if td.last and td.last > 0 else None
mid  = round((bid + ask) / 2, 2) if bid and ask else None

print(f"Quote: bid={bid}  ask={ask}  last={last}  mid={mid}")

# Price to sell: bid if available, else $0.01 (take whatever is there)
if bid and bid > 0:
    lmt = round(bid, 2)
    price_note = f"at bid ${lmt:.2f}"
elif last and last > 0:
    lmt = round(last, 2)
    price_note = f"at last ${lmt:.2f}"
else:
    lmt = 0.01
    price_note = "at $0.01 (no bid — likely near worthless)"

print(f"Placing SELL 1x {opt.localSymbol} LMT@${lmt:.2f}  [{price_note}]")
order = LimitOrder("SELL", 1, lmt, tif="DAY")
trade = ib.placeOrder(opt, order)

for i in range(45):
    ib.sleep(1)
    st = trade.orderStatus.status
    print(f"  [{i+1:02d}s] {st}  filled={trade.orderStatus.filled}")
    if st in ("Filled", "Cancelled", "Inactive"):
        break

fill   = trade.orderStatus.avgFillPrice
status = trade.orderStatus.status
errs   = [e.message[:100] for e in trade.log if e.message]

ib.disconnect()

token, chat_id = load_tg()
if status == "Filled":
    pnl = round((fill - ENTRY_PRICE) * 100, 2)
    msg = (
        f"<b>GOOG 400C SOLD</b>\n\n"
        f"Fill: ${fill:.2f}  (paid ${ENTRY_PRICE:.2f})\n"
        f"P&amp;L: ${pnl:+.2f}\n"
        f"{'Loss — deep OTM, expected outcome.' if pnl < 0 else 'Profit!'}"
    )
    print(f"\nFILLED @ ${fill:.2f}  P&L: ${pnl:+.2f}")
else:
    msg = (
        f"<b>GOOG 400C sell {status}</b>\n"
        f"Limit was ${lmt:.2f}  {price_note}\n"
        f"{'Error: ' + errs[-1] if errs else 'Order may still be working.'}"
    )
    print(f"\n{status}  {errs[-1] if errs else ''}")

tg(token, chat_id, msg)
print("Done.")
