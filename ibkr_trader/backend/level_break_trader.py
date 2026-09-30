"""
SPY 0DTE level-break day trader (CEO plan, 2026-09-24).

The plan, as given:
    UPSIDE   reclaim pre-market resistance -> long call, play the daily gap
             move / hourly (ETH) mean reversion back toward the prior close
    DOWNSIDE (DISABLED 2026-09-24 after backtesting -- see below)
    VIX      key level 16 (regime context, logged; optional gate)

BACKTEST, 631 trades over 544 sessions (2024-02 .. 2026-09), real SPY 1-min
bars and real 0DTE option prices, same rules and costs:
    upside    n=356  +$7.10/trade  t=+3.51  win 66%  (hits target 95.8% of the
              time; positive in 9 of 11 quarters)
    downside  n=275  +$6.90/trade  t=+0.86  win 32%  (median -$45, stops out
              64% of the time, and the last 6 months ran -$32/trade at t=-2.94)
The downside leg is a lottery whose rare winners stopped arriving, and it
cancelled out the upside's real edge (combined t fell to 1.90), so only the
upside trades. See spy_0dte/level_break_backtest.py.

Levels are computed from real data each morning, not typed in:
    pre-market high/low   04:00-09:29 ET
    overnight range       from the prior RTH close through 09:29
    prior RTH close       the gap-fill target
    ATR14 (daily)         +/-1 ATR targets off the prior close

LONG OPTIONS ONLY. This account cannot sell options below $2,000 net liq
(IBKR error 201), and a debit position also caps the loss at the premium --
which matters for an unvalidated discretionary rule. Nothing here is
backtested yet: run with --alert-only (the default) until it is.

    python level_break_trader.py                  # alert only, places nothing
    python level_break_trader.py --live           # real orders, qty 1
    python level_break_trader.py --show-levels    # print today's levels and exit
"""
import argparse
import json
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from ib_insync import IB, Index, Option, Stock

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
STATE_FILE = HERE / "level_break_state.json"
LOG_FILE = HERE / "level_break_log.jsonl"
PAUSE_FLAG = HERE / "level_break_PAUSED.flag"

TWS_PORT, CLIENT_ID = 7496, 1640
SESSION_START = dtime(9, 30)
NO_NEW_ENTRIES_AFTER = dtime(14, 30)
HARD_CLOSE = dtime(15, 45)          # never hold 0DTE into the bell
BREAK_BUFFER = 0.10                 # $ beyond the level before it counts as a break
MAX_PREMIUM_USD = 250               # per contract, hard cap
QTY = 1
ENABLE_DOWNSIDE = False             # backtested t=+0.86, last 6 months t=-2.94
MAX_TRADES_PER_DAY = 1              # upside only
STOP_PREMIUM_PCT = 0.50             # cut at -50% of premium paid
POLL_S = 20


def now_et():
    return datetime.now(ET)


def log(action, **d):
    rec = dict(time=now_et().isoformat(), action=action, **d)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(f"[{rec['time'][11:19]}] {action}: {json.dumps(d, default=str)[:230]}", flush=True)


def telegram(text, high=False):
    try:
        cfg = json.loads((HERE / "scanner_config.json").read_text())
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"],
                            "text": ("[HIGH] " if high else "") + "LEVEL-BREAK: " + text}, timeout=10)
    except Exception as e:
        print(f"telegram failed: {e}")


def load_state():
    if STATE_FILE.exists():
        s = json.loads(STATE_FILE.read_text())
        if s.get("session") == str(now_et().date()):
            return s
    return {"session": str(now_et().date()), "trades": [], "open": None}


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=1, default=str))


# ----------------------------------------------------------------- levels
def todays_levels(ib) -> dict:
    """Pre-market/overnight structure from Alpaca 1-min bars (includes extended
    hours, which IBKR's RTH history does not), ATR from IBKR daily bars."""
    cfg = json.loads((HERE / "scanner_config.json").read_text())
    H = {"APCA-API-KEY-ID": cfg["alpaca_api_key"], "APCA-API-SECRET-KEY": cfg["alpaca_secret_key"]}
    start = (now_et() - timedelta(days=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = requests.get("https://data.alpaca.markets/v2/stocks/SPY/bars",
                     params={"timeframe": "1Min", "start": start, "limit": 10000, "feed": "sip"},
                     headers=H, timeout=30)
    b = pd.DataFrame(r.json().get("bars", []))
    if b.empty:
        raise RuntimeError("no SPY bars from Alpaca")
    b["t"] = pd.to_datetime(b["t"]).dt.tz_convert(ET)
    b = b.set_index("t")
    today = b[b.index.date == now_et().date()]
    prior = b[b.index.date < now_et().date()]
    prior_rth = prior.between_time("09:30", "15:59")
    prior_day = prior_rth[prior_rth.index.date == prior_rth.index.date.max()]
    overnight = b[b.index > prior_day.index.max()]
    pre = today.between_time("04:00", "09:29")
    if pre.empty:
        pre = overnight

    s = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(s)
    bars = ib.reqHistoricalData(s, "", "30 D", "1 day", "TRADES", useRTH=True)
    df = pd.DataFrame([{"h": x.high, "l": x.low, "c": x.close} for x in bars])
    tr = pd.concat([df.h - df.l, (df.h - df.c.shift()).abs(), (df.l - df.c.shift()).abs()], axis=1).max(axis=1)
    atr14 = float(tr.tail(14).mean())
    prior_close = float(prior_day["c"].iloc[-1])

    vix = None
    try:
        vc = Index("VIX", "CBOE")
        ib.qualifyContracts(vc)
        td = ib.reqMktData(vc, "", False, False)
        ib.sleep(3)
        vix = float(td.last or td.close)
        ib.cancelMktData(vc)
    except Exception:
        pass

    return dict(prior_close=prior_close, prior_high=float(prior_day["h"].max()), prior_low=float(prior_day["l"].min()),
                overnight_high=float(overnight["h"].max()), overnight_low=float(overnight["l"].min()),
                premarket_high=float(pre["h"].max()), premarket_low=float(pre["l"].min()),
                atr14=atr14, target_up=prior_close, target_down=prior_close - atr14, vix=vix)


def spot(ib):
    s = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(s)
    td = ib.reqMktData(s, "", False, False)
    ib.sleep(2)
    px = td.last or td.close or td.marketPrice()
    ib.cancelMktData(s)
    return float(px) if px and px == px else None


def option_at(ib, strike, right, expiry):
    c = Option("SPY", expiry, float(strike), right, "SMART")
    ib.qualifyContracts(c)
    td = ib.reqMktData(c, "", False, False)
    ib.sleep(3)
    bid, ask = td.bid, td.ask
    ib.cancelMktData(c)
    ok = lambda v: v is not None and v == v and v > 0
    return c, (bid if ok(bid) else None), (ask if ok(ask) else None)


# ----------------------------------------------------------------- trading
def enter(ib, state, side, strike, lv, px, live, expiry):
    right = "C" if side == "upside" else "P"
    c, bid, ask = option_at(ib, strike, right, expiry)
    if ask is None:
        log("entry_skipped", side=side, reason="no option quote", strike=strike)
        return
    cost = ask * 100 * QTY
    if cost > MAX_PREMIUM_USD:
        log("entry_skipped", side=side, reason=f"premium ${cost:.0f} over cap ${MAX_PREMIUM_USD}", strike=strike)
        telegram(f"{side} break at SPY {px:.2f} but {strike}{right} costs ${cost:.0f} (cap ${MAX_PREMIUM_USD}) -- skipped")
        return
    target = lv["target_up"] if side == "upside" else lv["target_down"]
    log("SIGNAL", side=side, spot=px, strike=strike, right=right, ask=ask, cost=cost, target=target)
    telegram(f"{side.upper()} break: SPY {px:.2f} through {lv['premarket_high'] if side=='upside' else lv['overnight_low']:.2f}"
             f" -> BUY {strike}{right} @ ~${ask:.2f} (${cost:.0f}), target SPY {target:.2f}"
             + ("" if live else "  [ALERT ONLY -- no order placed]"), high=True)
    if not live:
        state["trades"].append(dict(side=side, alert_only=True, time=now_et().isoformat(), spot=px, strike=strike))
        save_state(state)
        return
    # Premium the option should be worth when SPY reaches the target level,
    # from its live delta (falls back to a conservative 0.5 if greeks are absent).
    delta = None
    try:
        td = ib.reqMktData(c, "106", False, False)
        ib.sleep(3)
        delta = td.modelGreeks.delta if td.modelGreeks else None
        ib.cancelMktData(c)
    except Exception:
        pass
    delta = abs(delta) if delta else 0.5
    tp_premium = max(round(ask + delta * (target - px), 2), round(ask * 1.10, 2))
    sl_premium = round(ask * (1 - STOP_PREMIUM_PCT), 2)

    # ONE bracket: entry, target and stop go to IBKR together, so the exits are
    # live on the exchange rather than depending on this process staying up.
    bracket = ib.bracketOrder("BUY", QTY, limitPrice=round(ask, 2),
                              takeProfitPrice=tp_premium, stopLossPrice=sl_premium)
    for o in bracket:
        o.tif = "DAY"
        ib.placeOrder(c, o)
    ib.sleep(5)
    parent = bracket[0]
    trade = next((t for t in ib.trades() if t.order.orderId == parent.orderId), None)
    filled = trade.orderStatus.status == "Filled" if trade else False
    fill = trade.orderStatus.avgFillPrice if filled else None
    log("bracket_placed", side=side, strike=strike, entry_limit=round(ask, 2), take_profit=tp_premium,
        stop=sl_premium, delta=round(delta, 3), filled=filled, fill=fill)
    if not filled:
        telegram(f"{side} bracket working: BUY {strike}{right} @ ${ask:.2f}, TP ${tp_premium:.2f}, "
                 f"SL ${sl_premium:.2f} (not filled yet)")
    pos = dict(side=side, strike=strike, right=right, entry=fill or ask, qty=QTY, target=target,
               stop_premium=sl_premium, take_profit=tp_premium, parent_id=parent.orderId,
               bracket_ids=[o.orderId for o in bracket], time=now_et().isoformat())
    state["open"] = pos
    state["trades"].append(pos)
    save_state(state)
    log("ENTERED", **pos)
    telegram(f"{strike}{right} bracket live: entry ${pos['entry']:.2f}, target ${tp_premium:.2f} "
             f"(SPY {target:.2f}), stop ${sl_premium:.2f}, hard close {HARD_CLOSE:%H:%M}")


def manage(ib, state, lv, px, live, expiry):
    pos = state.get("open")
    if not pos:
        return
    # The take-profit and stop sit at IBKR as part of the bracket. If either
    # filled, the position is gone -- detect that and record it. The only rule
    # this loop still owns is the 15:45 flat deadline.
    live_qty = sum(p.position for p in ib.positions()
                   if p.contract.secType == "OPT" and p.contract.symbol == "SPY"
                   and float(p.contract.strike) == float(pos["strike"])
                   and p.contract.right == pos["right"])
    if live_qty == 0:
        log("bracket_exited", side=pos["side"], strike=pos["strike"], note="TP or stop filled at IBKR")
        telegram(f"{pos['strike']}{pos['right']} closed by its IBKR bracket (target or stop)")
        state["open"] = None
        save_state(state)
        return
    t = now_et().time()
    if t < HARD_CLOSE:
        return
    reason = "hard close"
    for oid in pos.get("bracket_ids", []):          # pull the resting exits first
        for tr in ib.openTrades():
            if tr.order.orderId == oid:
                ib.cancelOrder(tr.order)
    ib.sleep(2)
    c, bid, ask = option_at(ib, pos["strike"], pos["right"], expiry)
    log("EXIT_SIGNAL", reason=reason, spot=px, bid=bid, entry=pos["entry"])
    if not live:
        telegram(f"EXIT ({reason}): {pos['strike']}{pos['right']} bid ${bid} vs entry ${pos['entry']:.2f} [ALERT ONLY]")
        state["open"] = None
        save_state(state)
        return
    from ibkr_0dte_common import ibkr_place_leg_with_ladder
    mid = (bid + ask) / 2 if bid and ask else bid
    ok, fill = ibkr_place_leg_with_ladder(ib, c, "SELL", f"exit {pos['side']}", pos["qty"], bid, ask, mid)
    pnl = ((fill or 0) - pos["entry"]) * 100 * pos["qty"]
    log("EXITED", reason=reason, fill=fill, pnl=pnl)
    telegram(f"EXIT ({reason}) {pos['strike']}{pos['right']} @ ${fill}: {pnl:+.0f}", high=(pnl < 0))
    state["open"] = None
    save_state(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="place real orders (default: alert only)")
    ap.add_argument("--show-levels", action="store_true")
    args = ap.parse_args()
    if PAUSE_FLAG.exists():
        sys.exit(f"{PAUSE_FLAG.name} present: {PAUSE_FLAG.read_text()}")

    ib = IB()
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    lv = todays_levels(ib)
    expiry = f"{now_et():%Y%m%d}"
    up_trigger = lv["premarket_high"] + BREAK_BUFFER
    dn_trigger = lv["overnight_low"] - BREAK_BUFFER
    import math
    # strike just BELOW each level -- a call at the resistance it just reclaimed,
    # a put at the low it just lost (matches the CEO's own 766C / 761P)
    up_strike, dn_strike = math.floor(lv["premarket_high"]), math.floor(lv["overnight_low"])
    log("levels", **{k: round(v, 2) if isinstance(v, float) else v for k, v in lv.items()},
        up_trigger=round(up_trigger, 2), dn_trigger=round(dn_trigger, 2),
        up_strike=up_strike, dn_strike=dn_strike, mode="LIVE" if args.live else "alert-only")
    if args.show_levels:
        ib.disconnect()
        return
    telegram(f"armed {'LIVE' if args.live else '(alert only)'} | UPSIDE ONLY: break {up_trigger:.2f} "
             f"-> buy {up_strike}C, target SPY {lv['target_up']:.2f} | VIX {lv['vix']} "
             f"| downside leg disabled (backtest t=+0.86, last 6m t=-2.94)")

    state = load_state()
    try:
        while now_et().time() < HARD_CLOSE:
            if now_et().time() < SESSION_START:
                ib.sleep(POLL_S)
                continue
            px = spot(ib)
            if px is None:
                ib.sleep(POLL_S)
                continue
            if state.get("open"):
                manage(ib, state, lv, px, args.live, expiry)
            elif (len(state["trades"]) < MAX_TRADES_PER_DAY
                  and now_et().time() < NO_NEW_ENTRIES_AFTER):
                done = {t["side"] for t in state["trades"]}
                if px > up_trigger and "upside" not in done:
                    enter(ib, state, "upside", up_strike, lv, px, args.live, expiry)
                elif ENABLE_DOWNSIDE and px < dn_trigger and "downside" not in done:
                    enter(ib, state, "downside", dn_strike, lv, px, args.live, expiry)
            ib.sleep(POLL_S)
        if state.get("open"):
            manage(ib, state, lv, spot(ib) or 0, args.live, expiry)
        log("day_end", trades=len(state["trades"]))
    finally:
        save_state(state)
        ib.disconnect()


if __name__ == "__main__":
    main()
