"""
ORCL Monday Morning Monitor — Jul 27, 2026
Watches ORCL open and auto-closes spreads based on 3 scenarios:
  > $116       → HOLD  (relief bounce)
  $115–$116    → WATCH (close if drift detected below $115)
  < $115       → CLOSE immediately (gap down)

Runs at 9:25 AM ET, fires at open, auto-executes closes if needed.
"""
import sys, io, time, json, datetime, requests
import yfinance as yf
from ib_insync import IB, Option, Contract, ComboLeg, Order

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ET = datetime.timezone(datetime.timedelta(hours=-4))

# ── Config ────────────────────────────────────────────────────────────────────
IBKR_PORT    = 7496
IBKR_CLIENT  = 46
HOLD_ABOVE   = 116.00   # hold if opens above this
CLOSE_BELOW  = 115.00   # close immediately if opens below this
DRIFT_WINDOW = 10       # minutes to watch for drift after open
DRIFT_CHECK  = 60       # check every N seconds during drift window

ORCL_POSITIONS = {
    "play_a": {
        "name": "Play A — Bull Put Spread P115/P110",
        "expiry": "20260821",
        "legs": [
            {"strike": 115.0, "right": "P", "action": "BUY"},   # close short
            {"strike": 110.0, "right": "P", "action": "SELL"},  # close long
        ],
        "close_type": "debit",   # we pay to close
        "pos_id": "ORCL_bull_put_spread_20260724_135308",
    },
    "play_b": {
        "name": "Play B — Bull Call Spread C120/C130",
        "expiry": "20260821",
        "legs": [
            {"strike": 120.0, "right": "C", "action": "SELL"},  # close long
            {"strike": 130.0, "right": "C", "action": "BUY"},   # close short
        ],
        "close_type": "credit",  # we receive to close
        "pos_id": "ORCL_bull_call_spread_20260724_135319",
    },
}

# ── Telegram ──────────────────────────────────────────────────────────────────
def _tg(msg):
    try:
        with open(r"c:\Projects\GenAI-Projects\ibkr_trader\backend\scanner_config.json") as f:
            cfg = json.load(f)
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"  Telegram failed: {e}")

def log(msg):
    ts = datetime.datetime.now(ET).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# ── Price fetch ───────────────────────────────────────────────────────────────
def get_orcl_price(source="yf"):
    """Get current ORCL price including pre-market via yfinance."""
    try:
        tk = yf.Ticker("ORCL")
        hist = tk.history(period="1d", interval="1m", prepost=True)
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            return price
    except:
        pass
    return None

def get_orcl_live(ib):
    """Get live ORCL price via IBKR."""
    from ib_insync import Stock
    contract = Stock("ORCL", "SMART", "USD")
    ib.qualifyContracts(contract)
    [td] = ib.reqTickers(contract)
    ib.sleep(2)
    price = td.last or td.bid or td.close
    return float(price) if price else None

# ── Close execution ───────────────────────────────────────────────────────────
def close_spread(ib, play_key, reason):
    """Close one ORCL spread via BAG combo MKT order."""
    pos = ORCL_POSITIONS[play_key]
    log(f"  Closing {pos['name']} — reason: {reason}")

    try:
        # Qualify both legs
        contracts = []
        for leg in pos["legs"]:
            c = Option("ORCL", pos["expiry"], leg["strike"], leg["right"], "SMART", "100", "USD")
            q = ib.qualifyContracts(c)
            if not q:
                log(f"  !! Could not qualify ORCL {pos['expiry']} {leg['strike']}{leg['right']}")
                return None
            contracts.append((leg, q[0]))

        # Get indicative prices
        tickers = ib.reqTickers(*[c for _, c in contracts])
        ib.sleep(2)

        leg_prices = []
        for (leg, c), td in zip(contracts, tickers):
            b = td.bid or 0; a = td.ask or 0
            mid = round((b+a)/2, 2) if b > 0 and a > 0 else (td.last or 0)
            leg_prices.append({"leg": leg, "mid": mid, "bid": b, "ask": a})
            log(f"    {leg['action']} {leg['right']}{leg['strike']:.0f}: bid={b:.2f} ask={a:.2f} mid={mid:.2f}")

        # Build BAG combo
        bag = Contract()
        bag.symbol   = "ORCL"
        bag.secType  = "BAG"
        bag.currency = "USD"
        bag.exchange = "SMART"
        bag.comboLegs = []
        for (leg, c) in contracts:
            cl = ComboLeg()
            cl.conId    = c.conId
            cl.ratio    = 1
            cl.action   = leg["action"]
            cl.exchange = "SMART"
            bag.comboLegs.append(cl)

        # Calculate limit price
        if pos["close_type"] == "debit":
            # We're buying the spread back — pay mid of the spread
            net = sum(lp["mid"] if lp["leg"]["action"] == "BUY" else -lp["mid"] for lp in leg_prices)
            lmt = max(round(net + 0.10, 2), 0.05)   # pay up to mid+0.10 for quick fill
            order_action = "BUY"
        else:
            # We're receiving credit to close
            net = sum(lp["mid"] if lp["leg"]["action"] == "SELL" else -lp["mid"] for lp in leg_prices)
            lmt = max(round(net - 0.10, 2), 0.05)   # take mid-0.10 for quick fill
            order_action = "SELL"

        log(f"    Placing BAG {order_action} @ ${lmt:.2f} ({'debit' if pos['close_type']=='debit' else 'credit'})")

        ord_ = Order()
        ord_.action        = order_action
        ord_.orderType     = "LMT"
        ord_.totalQuantity = 1
        ord_.lmtPrice      = lmt
        ord_.tif           = "DAY"

        trade = ib.placeOrder(bag, ord_)

        # Wait up to 60s for fill
        start = time.time()
        while time.time() - start < 60:
            ib.sleep(2)
            st = trade.orderStatus.status
            if st == "Filled":
                fill = trade.orderStatus.avgFillPrice
                log(f"    FILLED @ ${fill:.2f}")
                return fill
            if st in ("Cancelled", "ApiCancelled", "Inactive"):
                log(f"    Order {st} — trying MKT order")
                break
            log(f"    {st}...", end="\r")

        # Fallback: MKT order
        ib.cancelOrder(trade.order)
        ib.sleep(1)
        mkt_ord = Order()
        mkt_ord.action = order_action; mkt_ord.orderType = "MKT"
        mkt_ord.totalQuantity = 1; mkt_ord.tif = "DAY"
        trade2 = ib.placeOrder(bag, mkt_ord)
        start2 = time.time()
        while time.time() - start2 < 45:
            ib.sleep(2)
            st = trade2.orderStatus.status
            if st == "Filled":
                fill = trade2.orderStatus.avgFillPrice
                log(f"    MKT FILLED @ ${fill:.2f}")
                return fill
            if st in ("Cancelled", "ApiCancelled", "Inactive"):
                log(f"    MKT order {st}")
                return None

        log(f"    Could not fill {pos['name']}")
        return None

    except Exception as e:
        import traceback; traceback.print_exc()
        return None


def close_all(ib, reason, orcl_price):
    """Close both ORCL spreads and notify."""
    log(f"CLOSING ALL ORCL POSITIONS — {reason} (ORCL @ ${orcl_price:.2f})")
    _tg(f"⚠️ <b>ORCL AUTO-CLOSE TRIGGERED</b>\n\nReason: {reason}\nORCL price: ${orcl_price:.2f}\n\nClosing Play A and Play B now...")

    results = {}
    for key in ["play_a", "play_b"]:
        fill = close_spread(ib, key, reason)
        results[key] = fill

    # Update state
    try:
        state_path = r"c:\Projects\GenAI-Projects\ibkr_trader\backend\manual_trader_state.json"
        with open(state_path) as f: state = json.load(f)
        now_str = datetime.datetime.now(ET).isoformat()
        for key, pos in ORCL_POSITIONS.items():
            pid = pos["pos_id"]
            if pid in state["positions"]:
                p = state["positions"].pop(pid)
                p["phase"]        = "closed"
                p["close_reason"] = reason
                p["closed_at"]    = now_str
                p["close_pnl"]    = 0.0
                p["close_fills"]  = {key: results.get(key)}
                state["closed"].append(p)
        with open(state_path, "w") as f: json.dump(state, f, indent=2)
        log("State updated.")
    except Exception as e:
        log(f"State update failed: {e}")

    # Summary Telegram
    a_fill = results.get("play_a")
    b_fill = results.get("play_b")
    msg = (
        f"📋 <b>ORCL Positions Closed</b>\n\n"
        f"Trigger: {reason}\nORCL @ ${orcl_price:.2f}\n\n"
        f"Play A (P115/P110): {'filled @ $'+str(round(a_fill,2)) if a_fill else '⚠ check TWS'}\n"
        f"Play B (C120/C130): {'filled @ $'+str(round(b_fill,2)) if b_fill else '⚠ check TWS'}\n\n"
        f"Check TWS to confirm both legs closed."
    )
    _tg(msg)
    return results


# ── Main monitoring logic ─────────────────────────────────────────────────────
def main():
    now_et = datetime.datetime.now(ET)
    log(f"ORCL Monday Monitor started — {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    log(f"Rules: HOLD > ${HOLD_ABOVE} | WATCH ${CLOSE_BELOW}–${HOLD_ABOVE} | CLOSE < ${CLOSE_BELOW}")

    # ── 9:25 AM: pre-market snapshot ─────────────────────────────────────────
    log("Fetching pre-market price...")
    pm_price = get_orcl_price()

    if pm_price:
        log(f"ORCL pre-market: ${pm_price:.2f}")
        if pm_price >= HOLD_ABOVE:
            pm_signal = f"✅ <b>HOLD</b> — pre-market ${pm_price:.2f} (above $116). Relief bounce scenario."
        elif pm_price < CLOSE_BELOW:
            pm_signal = f"🔴 <b>CLOSE AT OPEN</b> — pre-market ${pm_price:.2f} (below $115). Gap down confirmed."
        else:
            pm_signal = f"👁 <b>WATCH</b> — pre-market ${pm_price:.2f} ($115–$116). Monitor drift at open."

        _tg(f"🌅 <b>ORCL Pre-Market Alert (9:25 AM)</b>\n\n{pm_signal}\n\nPositions open:\n• Play A: SELL P115/BUY P110 Aug21\n• Play B: BUY C120/SELL C130 Aug21")
        log(f"Pre-market signal sent: {pm_signal[:60]}")
    else:
        log("Could not get pre-market price")
        _tg("⚠️ ORCL monitor: could not fetch pre-market price. Monitor manually.")

    # ── Wait for 9:30 AM market open ─────────────────────────────────────────
    open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    if now_et < open_time:
        wait_secs = (open_time - now_et).total_seconds()
        log(f"Waiting {wait_secs:.0f}s for market open at 9:30 AM...")
        time.sleep(wait_secs + 5)   # +5s for first print to appear

    # ── Connect to IBKR ───────────────────────────────────────────────────────
    log("Connecting to IBKR...")
    ib = IB()
    ib.connect("127.0.0.1", IBKR_PORT, clientId=IBKR_CLIENT)
    ib.reqMarketDataType(1)
    log("Connected.")

    # ── Get opening price ─────────────────────────────────────────────────────
    open_price = get_orcl_live(ib)
    if open_price is None:
        open_price = get_orcl_price()

    if open_price is None:
        log("!! Could not get opening price — check TWS manually")
        _tg("⚠️ ORCL monitor: could not get opening price from IBKR. Check TWS!")
        ib.disconnect()
        return

    log(f"ORCL opening print: ${open_price:.2f}")

    # ── Scenario 1: Gap above $116 → HOLD ────────────────────────────────────
    if open_price >= HOLD_ABOVE:
        msg = (
            f"✅ <b>ORCL HOLD — Relief Bounce</b>\n\n"
            f"Open: ${open_price:.2f} (above $116)\n\n"
            f"Thesis: $116 recaptured. Play A gaining fast. Play B watching $123.62 BEP.\n"
            f"Monitor throughout session. No action needed at open."
        )
        log(f"HOLD — opening at ${open_price:.2f}, above ${HOLD_ABOVE}")
        _tg(msg)
        ib.disconnect()
        return

    # ── Scenario 2: Gap below $115 → CLOSE immediately ───────────────────────
    if open_price < CLOSE_BELOW:
        log(f"GAP DOWN — opening at ${open_price:.2f}, below ${CLOSE_BELOW}. Closing immediately.")
        close_all(ib, f"Gap below $115 at open (${open_price:.2f})", open_price)
        ib.disconnect()
        return

    # ── Scenario 3: $115–$116 → Watch for drift ──────────────────────────────
    log(f"WATCH ZONE — opening at ${open_price:.2f} ($115–$116). Monitoring {DRIFT_WINDOW} min for drift...")
    _tg(
        f"👁 <b>ORCL Watch Zone</b>\n\n"
        f"Open: ${open_price:.2f} ($115–$116 zone)\n\n"
        f"Monitoring {DRIFT_WINDOW} min for drift. Will auto-close if ORCL breaks below $115."
    )

    deadline = datetime.datetime.now(ET) + datetime.timedelta(minutes=DRIFT_WINDOW)
    last_prices = [open_price]
    closed = False

    while datetime.datetime.now(ET) < deadline:
        time.sleep(DRIFT_CHECK)
        price = get_orcl_live(ib)
        if price is None:
            price = get_orcl_price()
        if price is None:
            log("  Price fetch failed — retrying next cycle")
            continue

        last_prices.append(price)
        log(f"  ORCL: ${price:.2f}  (last 3: {[f'${p:.2f}' for p in last_prices[-3:]]})")

        # Close trigger: drops below $115
        if price < CLOSE_BELOW:
            log(f"  DRIFT CLOSE: ORCL broke ${CLOSE_BELOW} (now ${price:.2f})")
            close_all(ib, f"Drift below $115 ({DRIFT_WINDOW}-min window) at ${price:.2f}", price)
            closed = True
            break

        # Bonus: if price recovers above $116 during watch window → switch to HOLD
        if price >= HOLD_ABOVE:
            log(f"  Recovered above ${HOLD_ABOVE} — switching to HOLD")
            _tg(
                f"✅ <b>ORCL Recovered — HOLD</b>\n\n"
                f"Price rallied from ${open_price:.2f} → ${price:.2f} during watch window.\n"
                f"Above $116, switching to hold. No close needed."
            )
            closed = True
            break

    if not closed:
        # Watch window ended with price still in $115–$116 → close to be safe
        final_price = last_prices[-1]
        log(f"  Watch window expired. ORCL at ${final_price:.2f}. Closing to be safe.")
        close_all(ib, f"Watch window expired — no clear direction (${final_price:.2f})", final_price)

    ib.disconnect()
    log("Monitor complete.")


if __name__ == "__main__":
    main()
