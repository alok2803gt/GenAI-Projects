"""
SPY/QQQ 0DTE iron condor V2 -- staggered entries, stop at 0.5x credit, hold to expiry.

Strategy (validated 2026-09-22; see spy_0dte/TRADE_PLAN_condor.md and
ibkr_trader/docs/STRATEGIES.md):
  entries  every 30 min, 10:01 .. 14:31 ET (10 per day), one condor each
  strikes  short legs nearest S +/- 0.5u, wings $5 further out
           u = S * ATM_IV * sqrt(minutes_to_1600 / 98280)
  stop     close the whole condor the first time its mark-to-market loss
           reaches 0.5 x the credit received
  exit     otherwise hold to expiry (no profit target -- none was tested)

Backtest, 1 contract per entry (SPY, 660 sessions; QQQ pre-registered
confirmation passed at t 4.94):
  day mean +$101, t 4.94, worst day -$1,255, max drawdown -$5,224,
  stop fires on 42% of condors, peak margin ~$4,100, capital ~$15,000/unit.
Under worst-case after-hours exercise the day mean halves (~$48) -- that is
what NEAR_MONEY_CLOSE below exists to limit.

SAFETY RAILS (same disciplines as the other auto-traders on this account):
  - qty fixed at 1 per entry unless explicitly raised; no auto-scaling.
  - --mode paper is the DEFAULT (TWS paper port). Live requires --mode live.
  - --dry-run prices and logs everything but places no orders.
  - Pause flag file blocks all new entries until a human clears it; it is
    written on any partial fill (uncovered leg) or unexpected state.
  - DAILY_STOP: stops NEW entries once the day is down that much. This is a
    safety rail, NOT a backtested rule (the tested circuit breaker added
    nothing), so it sits well outside normal daily range.
  - NEAR_MONEY_CLOSE: at 15:58 close any condor whose short strike is within
    $0.25 of spot, to avoid after-hours assignment on physically settled
    ETF options. Tested only as a cost estimate, not as an edge.

Usage:
  python condor_v2_trader.py --ticker spy --mode paper
  python condor_v2_trader.py --ticker spy --mode paper --dry-run
"""
import argparse
import json
import math
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import requests
from ib_insync import IB

from alpaca_0dte_common import (
    load_config, now_et, etf_option, get_quotes_batch, get_quote, safe_px,
    place_condor_sequential as alpaca_place_condor, close_condor_sequential as alpaca_close_condor,
    place_leg_with_ladder as alpaca_place_leg,
)
from ibkr_0dte_common import (
    ibkr_place_condor_sequential, ibkr_close_condor_sequential, ibkr_place_leg_with_ladder,
    occ_symbol as alpaca_occ,          # OCC symbol builder, shared by both brokers
)

ENTRY_TIMES = [dtime(10, 1), dtime(10, 31), dtime(11, 1), dtime(11, 31), dtime(12, 1),
               dtime(12, 31), dtime(13, 1), dtime(13, 31), dtime(14, 1), dtime(14, 31)]
SHORT_U = 0.5                 # short strikes at +/- 0.5 expected moves
WING = 5.0                    # dollars beyond each short strike
STOP_MULT = 0.5               # close when loss reaches 0.5 x credit
MIN_CREDIT = 0.10             # $/share conservative credit floor
QTY = 1
MONITOR_S = 45
NEAR_MONEY_CLOSE = dtime(15, 58)
NEAR_MONEY_DIST = 0.25
DAILY_STOP = -1500.0 * QTY    # safety rail, not backtested
MIN_PER_YEAR = 390 * 252
PORTS = {"paper": 7497, "live": 7496}
ALLOW_ALPACA = False          # set by --alpaca-plumbing-test
CLIENT_ID = 1580
STATE_FILE = "condor_v2_state.json"
LOG_FILE = "condor_v2_log.jsonl"
PAUSE_FLAG = "condor_v2_PAUSED.flag"


# ----------------------------------------------------------------- plumbing
def telegram(text, high_priority=False):
    try:
        cfg = load_config()
        prefix = "[HIGH] " if high_priority else ""
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"], "text": prefix + "CONDOR V2: " + text}, timeout=10)
    except Exception as e:
        print(f"telegram send failed: {e}")


def log(action, **detail):
    rec = dict(time=now_et().isoformat(), action=action, **detail)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(f"[{rec['time'][11:19]}] {action}: {json.dumps({k: v for k, v in detail.items()}, default=str)[:200]}")


def load_state():
    p = Path(STATE_FILE)
    if p.exists():
        s = json.loads(p.read_text())
        if s.get("session") == str(now_et().date()):
            return s
    return {"session": str(now_et().date()), "open": [], "closed": [], "realized": 0.0}


def save_state(s):
    Path(STATE_FILE).write_text(json.dumps(s, indent=1, default=str))


def pause(reason):
    Path(PAUSE_FLAG).write_text(f"{now_et().isoformat()} {reason}\n")
    telegram(f"PAUSED: {reason}. Clear {PAUSE_FLAG} after review.", high_priority=True)
    log("paused", reason=reason)


# ----------------------------------------------------------------- pricing
def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, mins, sigma, right):
    T = max(mins, 0.5) / MIN_PER_YEAR
    if sigma <= 0:
        return max(0.0, S - K) if right == "C" else max(0.0, K - S)
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sq
    d2 = d1 - sq
    return S * _ncdf(d1) - K * _ncdf(d2) if right == "C" else K * _ncdf(-d2) - S * _ncdf(-d1)


def implied_vol(price, S, K, mins, right):
    lo, hi = 1e-4, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, mins, mid, right) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def minutes_to_close():
    now = now_et()
    close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return max((close - now).total_seconds() / 60.0, 1.0)


def spot(ib, ticker):
    """Spot for strike selection only. Same fallback chain as
    alpaca_0dte_common.spy_spot(): a live ETF stock quote can legitimately
    return NaN bid/ask while last/close are fine."""
    from ib_insync import Stock
    c = Stock(ticker.upper(), "SMART", "USD")
    ib.qualifyContracts(c)
    td = ib.reqMktData(c, "", False, False)
    ib.sleep(3)
    bid, ask = safe_px(td.bid), safe_px(td.ask)
    mid = (bid + ask) / 2 if bid and ask else None
    px = mid or bid or ask or safe_px(td.last) or safe_px(td.close) or safe_px(td.marketPrice())
    ib.cancelMktData(c)
    return px


def expected_move(ib, ticker, expiry, S):
    """u = S * ATM implied vol * sqrt(T). ATM IV from the nearest-strike call and put mids."""
    mins = minutes_to_close()
    k = round(S)
    qs = get_quotes_batch(ib, {"c": etf_option(ticker.upper(), expiry, k, "C"),
                               "p": etf_option(ticker.upper(), expiry, k, "P")})
    ivs = [implied_vol(qs[n]["mid"], S, k, mins, r) for n, r in (("c", "C"), ("p", "P")) if qs[n]["mid"]]
    ivs = [v for v in ivs if 0.01 < v < 3.0]
    if not ivs:
        return None, None
    iv = sum(ivs) / len(ivs)
    return S * iv * math.sqrt(mins / MIN_PER_YEAR), iv


# ----------------------------------------------------------------- broker
class Broker:
    """Order routing. Quotes always come from IBKR (real NBBO); Alpaca's free
    'indicative' option feed shows spreads several times wider than the real
    market, which would corrupt both entry pricing and the stop.

    ALPACA IS NOT SUITABLE FOR THIS STRATEGY (verified live 2026-09-22):
      1. opening a position near expiry is rejected outright --
         'contract ... expires soon, unable to open new positions';
      2. Alpaca force-closes every 0DTE option position at 15:45 ET
         (this account's own incident, 2026-09-02, see
         alpaca_0dte_butterfly_trader.py).
    The whole edge here is holding to the 16:00 expiry: the same condor
    closed at 15:45 backtests at -$0.38/day versus +$8.37/day held to
    expiry. Running this on Alpaca would trade the version already shown
    to have no edge, so --broker alpaca_paper requires an explicit
    override flag and exists only for plumbing tests."""

    def __init__(self, kind, cfg, expiry, ticker):
        self.kind, self.expiry, self.ticker = kind, expiry, ticker
        if kind == "alpaca_paper":
            if not ALLOW_ALPACA:
                sys.exit("REFUSING: Alpaca force-closes 0DTE at 15:45 and blocks opening near expiry, "
                         "which removes this strategy's entire edge (see Broker docstring). "
                         "Use --broker ibkr against TWS paper (port 7497), or pass "
                         "--alpaca-plumbing-test to place throwaway test orders anyway.")
            from alpaca.trading.client import TradingClient
            for k in ("alpaca_paper_api_key", "alpaca_paper_secret_key"):
                if not cfg.get(k):
                    sys.exit(f"missing {k} in scanner_config.json -- add your Alpaca PAPER keys "
                             "(the live data keys will not work for paper trading)")
            self.client = TradingClient(cfg["alpaca_paper_api_key"], cfg["alpaca_paper_secret_key"], paper=True)
            acct = self.client.get_account()
            log("alpaca_account", status=str(acct.status), equity=acct.equity,
                options_level=getattr(acct, "options_trading_level", None))

    def _syms(self, strikes):
        return {n: alpaca_occ(self.ticker.upper(), k, "P" if "put" in n else "C", self.expiry[2:])
                for n, k in strikes.items()}

    def place(self, ib, contracts, limits, strikes, qty):
        if self.kind == "ibkr":
            return ibkr_place_condor_sequential(ib, contracts, limits, qty=qty)
        return alpaca_place_condor(self.client, self._syms(strikes), limits, qty=qty)

    def close(self, ib, contracts, limits, strikes, qty):
        if self.kind == "ibkr":
            return ibkr_close_condor_sequential(ib, contracts, limits, qty=qty)
        return alpaca_close_condor(self.client, self._syms(strikes), limits, qty=qty)

    def leg(self, ib, contract, sym, action, label, qty, bid, ask, mid):
        if self.kind == "ibkr":
            return ibkr_place_leg_with_ladder(ib, contract, action, label, qty, bid, ask, mid)
        from alpaca.trading.enums import OrderSide
        side = OrderSide.BUY if action == "BUY" else OrderSide.SELL
        return alpaca_place_leg(self.client, sym, side, label, qty, bid, ask, mid)


def normalize_quotes(q):
    """A cheap long wing legitimately has NO bid (bid 0 / ask 0.01), and
    safe_px() reports a 0 bid as missing. Treat "no bid but a real ask" as
    bid 0 rather than throwing the whole condor away; a leg with no ask at
    all stays unusable."""
    out = {}
    for name, v in q.items():
        bid, ask = v["bid"], v["ask"]
        if bid is None and ask is not None:
            bid = 0.0
        mid = (bid + ask) / 2 if (bid is not None and ask is not None) else None
        out[name] = {"bid": bid, "ask": ask, "mid": mid}
    return out


# ----------------------------------------------------------------- trading
def build_condor(ib, ticker, expiry, S, u):
    strikes = {"short_put": round(S - SHORT_U * u), "short_call": round(S + SHORT_U * u)}
    strikes["long_put"] = strikes["short_put"] - WING
    strikes["long_call"] = strikes["short_call"] + WING
    contracts = {n: etf_option(ticker.upper(), expiry, k, "P" if "put" in n else "C") for n, k in strikes.items()}
    quotes = normalize_quotes(get_quotes_batch(ib, contracts))
    missing = [n for n, q in quotes.items() if q["mid"] is None]
    if missing:
        return None, f"no quote for {', '.join(missing)}"
    credit_mid = (quotes["short_put"]["mid"] - quotes["long_put"]["mid"]
                  + quotes["short_call"]["mid"] - quotes["long_call"]["mid"])
    conservative = (quotes["short_put"]["bid"] - quotes["long_put"]["ask"]
                    + quotes["short_call"]["bid"] - quotes["long_call"]["ask"])
    if conservative < MIN_CREDIT:
        return None, f"credit too thin (conservative ${conservative:.2f})"
    limits = {n: (quotes[n]["bid"], quotes[n]["ask"], quotes[n]["mid"]) for n in contracts}
    return dict(strikes=strikes, contracts=contracts, quotes=quotes, limits=limits,
                credit_mid=credit_mid, conservative=conservative), None


def condor_value(ib, ticker, expiry, strikes):
    contracts = {n: etf_option(ticker.upper(), expiry, k, "P" if "put" in n else "C") for n, k in strikes.items()}
    q = normalize_quotes(get_quotes_batch(ib, contracts))
    if any(v["mid"] is None for v in q.values()):
        return None, contracts, q
    val = (q["short_put"]["mid"] - q["long_put"]["mid"]) + (q["short_call"]["mid"] - q["long_call"]["mid"])
    return val, contracts, q


def enter(ib, broker, ticker, expiry, state, dry, slot):
    S = spot(ib, ticker)
    if not S:
        log("entry_skipped", reason="no spot")
        return
    u, iv = expected_move(ib, ticker, expiry, S)
    if not u:
        log("entry_skipped", reason="no ATM IV")
        return
    plan, err = build_condor(ib, ticker, expiry, S, u)
    if plan is None:
        log("entry_skipped", reason=err, spot=S, u=round(u, 2))
        return
    st = plan["strikes"]
    log("entry_plan", spot=round(S, 2), iv=round(iv, 4), u=round(u, 2), strikes=st,
        credit_mid=round(plan["credit_mid"], 2), conservative=round(plan["conservative"], 2))
    if dry:
        done_note = {"slot": slot, "strikes": st}
        log("entry_dry_run", **done_note)
        return
    ok, fills, status = broker.place(ib, plan["contracts"], plan["limits"], st, QTY)
    credit = ((fills.get("short_put") or 0) - (fills.get("long_put") or 0)
              + (fills.get("short_call") or 0) - (fills.get("long_call") or 0))
    if not ok:
        pause(f"partial fill on entry ({status}, fills={fills})")
        return
    pos = dict(id=f"{now_et():%H%M%S}", slot=slot, strikes=st, credit=credit, qty=QTY,
               stop_loss_usd=STOP_MULT * credit * 100 * QTY, entry_time=now_et().isoformat(), fills=fills)
    state["open"].append(pos)
    save_state(state)
    log("entered", **pos)
    telegram(f"{ticker.upper()} condor {st['long_put']:g}/{st['short_put']:g}P {st['short_call']:g}/{st['long_call']:g}C "
             f"credit ${credit * 100:.0f}, stop at -${pos['stop_loss_usd']:.0f}")


def close_condor(ib, broker, ticker, expiry, pos, state, reason, dry):
    val, contracts, q = condor_value(ib, ticker, expiry, pos["strikes"])
    if val is None:
        log("close_failed", reason="no quotes", pos=pos["id"])
        return False
    pnl = (pos["credit"] - val) * 100 * pos["qty"]
    if dry:
        log("close_plan", pos=pos["id"], reason=reason, value=round(val, 2), pnl=round(pnl, 0))
        return True
    limits = {n: (q[n]["bid"], q[n]["ask"], q[n]["mid"]) for n in contracts}
    ok, fills = broker.close(ib, contracts, limits, pos["strikes"], pos["qty"])
    exit_val = ((fills.get("short_put") or 0) - (fills.get("long_put") or 0)
                + (fills.get("short_call") or 0) - (fills.get("long_call") or 0))
    realized = (pos["credit"] - exit_val) * 100 * pos["qty"]
    if not ok:
        pause(f"partial close on {pos['id']} (fills={fills})")
    state["open"] = [p for p in state["open"] if p["id"] != pos["id"]]
    state["closed"].append({**pos, "close_reason": reason, "pnl": realized})
    state["realized"] += realized
    save_state(state)
    log("closed", pos=pos["id"], reason=reason, pnl=round(realized, 0), fills=fills)
    telegram(f"{ticker.upper()} condor {pos['id']} closed ({reason}): ${realized:+.0f}")
    return True


def close_near_money_side(ib, broker, ticker, expiry, pos, S, state, dry):
    """15:58: close only the side whose short strike is near spot (cheapest way
    to avoid after-hours assignment on a physically settled ETF option)."""
    for side, sk, lk, right in (("put", "short_put", "long_put", "P"), ("call", "short_call", "long_call", "C")):
        if abs(S - pos["strikes"][sk]) > NEAR_MONEY_DIST:
            continue
        cons = {n: etf_option(ticker.upper(), expiry, pos["strikes"][n], right) for n in (sk, lk)}
        q = normalize_quotes(get_quotes_batch(ib, cons))
        log("near_money_close", pos=pos["id"], side=side, spot=round(S, 2), strike=pos["strikes"][sk])
        if dry:
            continue
        syms = {n: alpaca_occ(ticker.upper(), pos["strikes"][n], right, expiry[2:]) for n in (sk, lk)}
        broker.leg(ib, cons[sk], syms[sk], "BUY", f"close short {side}", pos["qty"],
                   q[sk]["bid"], q[sk]["ask"], q[sk]["mid"])
        broker.leg(ib, cons[lk], syms[lk], "SELL", f"close long {side}", pos["qty"],
                   q[lk]["bid"], q[lk]["ask"], q[lk]["mid"])
        pos.setdefault("legs_closed", []).append(side)
        save_state(state)


# ----------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="spy", choices=["spy", "qqq"])
    ap.add_argument("--mode", default="paper", choices=["paper", "live"],
                    help="which TWS port to take QUOTES from")
    ap.add_argument("--broker", default="ibkr", choices=["ibkr", "alpaca_paper"],
                    help="where ORDERS go (IBKR; Alpaca cannot hold 0DTE to expiry)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--alpaca-plumbing-test", action="store_true",
                    help="allow --broker alpaca_paper for order-routing tests only (edge does NOT survive there)")
    args = ap.parse_args()
    dry = args.dry_run
    global ALLOW_ALPACA
    ALLOW_ALPACA = args.alpaca_plumbing_test

    if Path(PAUSE_FLAG).exists():
        print(f"{PAUSE_FLAG} present -- refusing to run. {Path(PAUSE_FLAG).read_text()}")
        sys.exit(1)
    now = now_et()
    if now.weekday() >= 5:
        print("weekend -- nothing to do")
        return
    expiry = f"{now:%Y%m%d}"
    state = load_state()

    ib = IB()
    ib.connect("127.0.0.1", PORTS[args.mode], clientId=CLIENT_ID, timeout=20)
    broker = Broker("ibkr" if dry else args.broker, load_config(), expiry, args.ticker)
    log("start", ticker=args.ticker, quotes=f"ibkr:{args.mode}", orders=args.broker, dry_run=dry, expiry=expiry,
        entries_done=len(state["closed"]) + len(state["open"]))
    telegram(f"started {args.ticker.upper()} quotes=IBKR orders={args.broker}{' DRY-RUN' if dry else ''}, "
             f"{len(ENTRY_TIMES)} entries planned, stop {STOP_MULT:g}x credit")
    try:
        done = {p.get("slot") for p in state["open"] + state["closed"]}   # scheduled slot, not fill time
        while now_et().time() < dtime(16, 0):
            t = now_et().time()
            # 1) entries
            for et in ENTRY_TIMES:
                key = f"{et:%H:%M}"
                window_end = (datetime.combine(now_et().date(), et) + timedelta(minutes=4)).time()
                if key not in done and et <= t < window_end:
                    open_pnl = sum((p["credit"] - (condor_value(ib, args.ticker, expiry, p["strikes"])[0] or p["credit"]))
                                   * 100 * p["qty"] for p in state["open"])
                    if state["realized"] + open_pnl <= DAILY_STOP:
                        log("entry_blocked", reason="daily stop", day_pnl=round(state["realized"] + open_pnl, 0))
                        done.add(key)
                        continue
                    enter(ib, broker, args.ticker, expiry, state, dry, key)
                    done.add(key)
            # 2) stops
            for pos in list(state["open"]):
                val, _, _ = condor_value(ib, args.ticker, expiry, pos["strikes"])
                if val is None:
                    continue
                loss = (val - pos["credit"]) * 100 * pos["qty"]
                if loss >= pos["stop_loss_usd"]:
                    close_condor(ib, broker, args.ticker, expiry, pos, state, f"stop {STOP_MULT:g}x", dry)
            # 3) near-the-money close before the bell
            if t >= NEAR_MONEY_CLOSE:
                S = spot(ib, args.ticker)
                if S:
                    for pos in list(state["open"]):
                        close_near_money_side(ib, broker, args.ticker, expiry, pos, S, state, dry)
                break
            ib.sleep(MONITOR_S)
        held = len(state["open"])
        msg = (f"day done: {len(state['closed'])} closed (realized ${state['realized']:+.0f}), "
               f"{held} held to expiry")
        log("day_end", closed=len(state["closed"]), realized=round(state["realized"], 0), held=held)
        telegram(msg)
    except Exception as e:
        pause(f"unhandled error: {type(e).__name__} {e}")
        raise
    finally:
        save_state(state)
        ib.disconnect()


if __name__ == "__main__":
    main()
