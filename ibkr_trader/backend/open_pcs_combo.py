"""
Open 3 Put Credit Spreads using BAG combo orders.
Combo orders send both legs simultaneously → IBKR charges spread margin, not naked margin.

  1. QCOM P145/P135 Aug7
  2. NVDA P195/P185 Aug14
  3. LRCX P270/P260 Aug7
"""
import sys, io, time, json, datetime
from ib_insync import IB, Option, Contract, ComboLeg, LimitOrder, Order

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PORT, CLIENT = 7496, 45

SPREADS = [
    {"name":"QCOM Bull Put Spread Aug7",  "ticker":"QCOM", "expiry":"20260807", "short":145.0, "wing":135.0},
    {"name":"NVDA Bull Put Spread Aug14", "ticker":"NVDA", "expiry":"20260814", "short":195.0, "wing":185.0},
    {"name":"LRCX Bull Put Spread Aug7",  "ticker":"LRCX", "expiry":"20260807", "short":270.0, "wing":260.0},
]

ib = IB()
ib.connect("127.0.0.1", PORT, clientId=CLIENT)
ib.reqMarketDataType(1)
print("Connected — using BAG combo orders (spread margin, not naked margin)\n")

def qualify_put(ticker, expiry, strike):
    c = Option(ticker, expiry, float(strike), "P", "SMART", "100", "USD")
    q = ib.qualifyContracts(c)
    if not q:
        raise RuntimeError(f"Could not qualify {ticker} {expiry} {strike}P")
    return q[0]

def get_mid(contract):
    [td] = ib.reqTickers(contract)
    ib.sleep(2)
    b, a = td.bid, td.ask
    if b and a and b > 0 and a > 0:
        return round((b+a)/2,2), b, a
    last = td.last or td.close or 0.01
    return round(last,2), last, last

def wait_fill(trade, label, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        ib.sleep(2)
        st = trade.orderStatus.status
        if st == "Filled":
            p = trade.orderStatus.avgFillPrice
            print(f"  {label}: FILLED @ ${p:.2f}/sh credit")
            return p
        if st in ("Cancelled","ApiCancelled","Inactive"):
            print(f"  {label}: {st}")
            return None
        print(f"  {label}: {st}...", end="\r")
    print(f"\n  {label}: TIMEOUT")
    return None

fills_summary = []

for sp in SPREADS:
    print(f"\n{'='*62}")
    print(f"  {sp['name']}")
    print(f"  BAG: SELL P{sp['short']:.0f} / BUY P{sp['wing']:.0f}  ({sp['expiry']})")
    print(f"{'='*62}")
    try:
        # Qualify both legs
        c_short = qualify_put(sp["ticker"], sp["expiry"], sp["short"])
        c_wing  = qualify_put(sp["ticker"], sp["expiry"], sp["wing"])
        print(f"  Short conId={c_short.conId}  Wing conId={c_wing.conId}")

        # Live quotes for price reference
        short_mid, short_bid, short_ask = get_mid(c_short)
        wing_mid,  wing_bid,  wing_ask  = get_mid(c_wing)
        net_credit = round(short_mid - wing_mid, 2)
        print(f"  Short P{sp['short']:.0f}: bid={short_bid:.2f} ask={short_ask:.2f}")
        print(f"  Wing  P{sp['wing']:.0f}: bid={wing_bid:.2f} ask={wing_ask:.2f}")
        print(f"  Indicative net credit: ${net_credit:.2f}/sh (${net_credit*100:.0f})")

        if net_credit <= 0:
            print(f"  !! Net credit <=0 — skipping")
            fills_summary.append({**sp,"status":"SKIPPED","credit":net_credit})
            continue

        # Build BAG combo contract
        bag = Contract()
        bag.symbol   = sp["ticker"]
        bag.secType  = "BAG"
        bag.currency = "USD"
        bag.exchange = "SMART"

        leg_sell = ComboLeg()
        leg_sell.conId    = c_short.conId
        leg_sell.ratio    = 1
        leg_sell.action   = "SELL"    # sell the short put
        leg_sell.exchange = "SMART"

        leg_buy = ComboLeg()
        leg_buy.conId    = c_wing.conId
        leg_buy.ratio    = 1
        leg_buy.action   = "BUY"     # buy the wing put
        leg_buy.exchange = "SMART"

        bag.comboLegs = [leg_sell, leg_buy]

        # For a put credit spread BAG: action=SELL, lmtPrice = NET CREDIT we want to receive
        # IBKR interprets: positive lmtPrice = credit received
        lmt = max(round(net_credit - 0.05, 2), 0.05)   # slightly below mid for a quick fill
        print(f"  Placing BAG SELL @ ${lmt:.2f} net credit limit...")

        order = Order()
        order.action        = "SELL"
        order.orderType     = "LMT"
        order.totalQuantity = 1
        order.lmtPrice      = lmt
        order.tif           = "DAY"

        trade = ib.placeOrder(bag, order)
        fill_credit = wait_fill(trade, f"{sp['ticker']} PCS combo")

        if fill_credit is None:
            # Retry at a lower limit (more aggressive)
            lmt2 = max(round(net_credit * 0.85, 2), 0.05)
            print(f"  Retrying at ${lmt2:.2f} (85% of mid)...")
            ib.cancelOrder(trade.order)
            ib.sleep(2)
            order2 = Order()
            order2.action = "SELL"; order2.orderType = "LMT"
            order2.totalQuantity = 1; order2.lmtPrice = lmt2; order2.tif = "DAY"
            trade2 = ib.placeOrder(bag, order2)
            fill_credit = wait_fill(trade2, f"{sp['ticker']} PCS retry")

        if fill_credit is None:
            fills_summary.append({**sp,"status":"FAILED","credit":net_credit})
            continue

        # Filled
        credit_usd  = round(fill_credit * 100, 0)
        width       = sp["short"] - sp["wing"]
        max_loss    = round((width - fill_credit) * 100, 0)
        bep         = round(sp["short"] - fill_credit, 2)
        pt_usd      = round(credit_usd * 0.50, 0)
        sl_usd      = -round(max_loss * 0.65, 0)

        print(f"  Net credit received: ${fill_credit:.2f}/sh = ${credit_usd:.0f}")
        print(f"  BEP: ${bep:.2f}  Max loss: -${max_loss:.0f}  PT: +${pt_usd:.0f}  SL: ${sl_usd:.0f}")

        fills_summary.append({
            **sp, "status":"FILLED",
            "credit":fill_credit, "credit_usd":credit_usd,
            "bep":bep, "max_loss":max_loss,
            "pt_usd":pt_usd, "sl_usd":sl_usd,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        fills_summary.append({**sp,"status":f"ERROR: {e}","credit":0})

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("  EXECUTION SUMMARY")
print(f"{'='*62}")
total_credit = 0
for r in fills_summary:
    icon = "✅" if r["status"]=="FILLED" else "❌"
    cr = r.get("credit_usd",0)
    if r["status"]=="FILLED": total_credit += cr
    print(f"  {icon}  {r['name']}")
    if r["status"]=="FILLED":
        print(f"       Credit: ${r['credit']:.2f}/sh (${cr:.0f}) | BEP: ${r['bep']:.2f} | "
              f"PT: +${r['pt_usd']:.0f} | SL: ${r['sl_usd']:.0f}")
    else:
        print(f"       Status: {r['status']}")
print(f"\n  Total credit: ${total_credit:.0f}")

# ── Register in manual trader state ──────────────────────────────────────────
try:
    state_path = r"c:\Projects\GenAI-Projects\ibkr_trader\backend\manual_trader_state.json"
    with open(state_path) as f: state = json.load(f)

    now_str = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-4))).isoformat()
    for r in fills_summary:
        if r["status"] != "FILLED": continue
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pos_id = f"{r['ticker']}_bull_put_spread_{ts}"
        earn_note = ""
        if r["ticker"] in ("QCOM","LRCX") and r["expiry"]=="20260807":
            earn_note = "Earnings Jul29 in window."
        state["positions"][pos_id] = {
            "pos_id": pos_id, "name": r["name"],
            "ticker": r["ticker"], "strategy": "bull_put_spread",
            "expiry": r["expiry"],
            "legs": [
                {"ticker":r["ticker"],"expiry":r["expiry"],"strike":r["wing"],"right":"P",
                 "action":"BUY","qty":1,"fill_price":round(r["credit"]*0.4,2),"multiplier":100},
                {"ticker":r["ticker"],"expiry":r["expiry"],"strike":r["short"],"right":"P",
                 "action":"SELL","qty":1,"fill_price":round(r["credit"]*0.4+r["credit"],2),"multiplier":100},
            ],
            "net_entry":         r["credit_usd"],
            "profit_target_usd": r["pt_usd"],
            "stop_loss_usd":     r["sl_usd"],
            "hard_close_time":   "15:45",
            "notes": f"Tech selloff PCS Jul24. Credit ${r['credit']:.2f}/sh. BEP ${r['bep']:.2f}. {earn_note}",
            "phase":"open","entry_time":now_str,
            "live_pnl":0.0,"close_reason":None,"close_pnl":None,"closed_at":None,
        }
        print(f"\n  Registered: {pos_id}")
    with open(state_path,"w") as f: json.dump(state,f,indent=2)
    print("  State saved.")
except Exception as e:
    print(f"  State save failed: {e}")

# ── Telegram ──────────────────────────────────────────────────────────────────
try:
    import requests as rq
    with open(r"c:\Projects\GenAI-Projects\ibkr_trader\backend\scanner_config.json") as f:
        cfg = json.load(f)
    lines = ["📉 <b>Tech Selloff PCS — 3 Spreads</b>\n"]
    for r in fills_summary:
        icon = "✅" if r["status"]=="FILLED" else "❌"
        lines.append(f"{icon} <b>{r['name']}</b>")
        if r["status"]=="FILLED":
            lines.append(f"   SELL P{r['short']:.0f}/BUY P{r['wing']:.0f} | Credit ${r['credit']:.2f} (${r['credit_usd']:.0f})")
            lines.append(f"   BEP ${r['bep']:.2f} | PT +${r['pt_usd']:.0f} | SL ${r['sl_usd']:.0f}\n")
        else:
            lines.append(f"   {r['status']}\n")
    lines.append(f"<b>Total credit: ${total_credit:.0f}</b>")
    rq.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id":cfg["telegram_chat_id"],"text":"\n".join(lines),"parse_mode":"HTML"},timeout=10)
    print("  Telegram sent.")
except Exception as e:
    print(f"  Telegram failed: {e}")

ib.disconnect()
print("Done.")
