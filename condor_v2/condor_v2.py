#!/usr/bin/env python3
"""
0DTE iron condor V2 -- standalone trader for IBKR (TWS / IB Gateway).

Self-contained: needs only ib_insync (and requests, if Telegram alerts are on).
Nothing else from the original repo is required.

STRATEGY (validated on SPY 660 sessions and confirmed on QQQ under a
pre-registered test; see README):
  entries  every 30 min, 10:01 .. 14:31 ET -- 10 condors per session
  strikes  short legs nearest spot +/- 0.5u, wings $5 further out, where
           u = spot * ATM implied vol * sqrt(minutes_to_1600 / 98280)
  stop     close the whole condor the first time its mark-to-market loss
           reaches 0.5 x the credit received
  exit     otherwise HOLD TO EXPIRY -- this is where the edge lives. A broker
           that force-closes 0DTE early (e.g. Alpaca at 15:45) removes it.
  15:58    close only the side whose short strike is within $0.25 of spot,
           to limit after-hours assignment on physically settled ETF options.

Backtest per 1-contract unit: +$101/day, worst day -$1,255, max drawdown
-$5,224, stop fires on ~42% of condors, peak margin ~$4,100, capital ~$15,000.

SAFETY: paper port by default; qty 1; a pause flag blocks all new entries
after any partial fill; a daily stop halts new entries; --dry-run places
nothing. Read README.md before the first run.
"""
import argparse
import json
import math
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import IB, LimitOrder, Option, Stock

ET = ZoneInfo("America/New_York")
MIN_PER_YEAR = 390 * 252
HERE = Path(__file__).resolve().parent
DEFAULTS = {
    "ticker": "SPY",
    "qty": 1,
    "tws_host": "127.0.0.1",
    "tws_port": 7497,                 # 7497 TWS paper / 7496 TWS live / 4002 gateway paper
    "client_id": 1580,
    "entry_times": ["10:01", "10:31", "11:01", "11:31", "12:01",
                    "12:31", "13:01", "13:31", "14:01", "14:31"],
    "short_u": 0.5,
    "wing": 5.0,
    "stop_mult": 0.5,
    "min_credit": 0.10,
    "monitor_seconds": 45,
    # Near-money passes close the sides whose short is in the money or within `dist` of spot.
    # The early pass runs before IBKR's own pre-expiry liquidation, which on 2026-10-06 bought
    # back 6 of our 7 short 779P at 15:42 (assignment would have exceeded the account's equity).
    "near_money_early": "15:35",
    "near_money_dist_early": 1.0,
    "near_money_close": "15:58",
    "near_money_dist": 0.25,
    "daily_stop_usd": -1500.0,
    "fill_wait_seconds": 20,
    "entry_window_min": 10,
    "telegram_token": "",
    "telegram_chat_id": "",
}
STATE_FILE = HERE / "state.json"
LOG_FILE = HERE / "condor_v2_log.jsonl"
PAUSE_FLAG = HERE / "PAUSED.flag"
IB_DONE_NOT_FILLED = {"Cancelled", "ApiCancelled", "Inactive", "Expired"}
CROSS = 0.02
LEGS = ("long_put", "long_call", "short_put", "short_call")


# ----------------------------------------------------------------- plumbing
def now_et():
    return datetime.now(ET)


def hhmm(s):
    h, m = s.split(":")
    return dtime(int(h), int(m))


def load_config(base=HERE):
    p = base / "config.json"
    cfg = dict(DEFAULTS)
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    else:
        print(f"note: {p} not found -- using defaults (paper port {cfg['tws_port']})")
    return cfg


def telegram(cfg, text, high_priority=False):
    if not cfg.get("telegram_token") or not cfg.get("telegram_chat_id"):
        return
    try:
        import requests
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"],
                            "text": ("[HIGH] " if high_priority else "") + "CONDOR V2: " + text}, timeout=10)
    except Exception as e:
        print(f"telegram send failed: {e}")


def log(action, **detail):
    rec = dict(time=now_et().isoformat(), action=action, **detail)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(f"[{rec['time'][11:19]}] {action}: {json.dumps(detail, default=str)[:220]}", flush=True)


def load_state():
    if STATE_FILE.exists():
        s = json.loads(STATE_FILE.read_text())
        if s.get("session") == str(now_et().date()):
            return s
    return {"session": str(now_et().date()), "open": [], "closed": [], "realized": 0.0}


def save_state(s):
    STATE_FILE.write_text(json.dumps(s, indent=1, default=str))


def pause(cfg, reason):
    PAUSE_FLAG.write_text(f"{now_et().isoformat()} {reason}\n")
    telegram(cfg, f"PAUSED: {reason}. Clear {PAUSE_FLAG.name} after review.", high_priority=True)
    log("paused", reason=reason)


# ----------------------------------------------------------------- quotes
def safe_px(v):
    if v is None:
        return None
    try:
        f = float(v)
        return f if f > 0 and f == f else None
    except (TypeError, ValueError):
        return None


def option_contract(ticker, expiry, strike, right):
    return Option(ticker.upper(), expiry, strike, right, "SMART")


def get_quotes(ib, contracts):
    """{name: {bid, ask, mid}} for several contracts, one settle wait for all.
    A cheap long wing legitimately has NO bid (bid 0 / ask 0.01): treat "no bid
    but a real ask" as bid 0 rather than discarding the whole condor."""
    items = list(contracts.items())
    ib.qualifyContracts(*[c for _, c in items])
    missing = [n for n, c in items if not c.conId]
    if missing:
        raise ValueError(f"could not qualify: {missing}")
    tickers = {n: ib.reqMktData(c, "", False, False) for n, c in items}
    ib.sleep(3)
    out = {}
    for n, td in tickers.items():
        bid, ask = safe_px(td.bid), safe_px(td.ask)
        if bid is None and ask is not None:
            bid = 0.0
        out[n] = {"bid": bid, "ask": ask, "mid": (bid + ask) / 2 if (bid is not None and ask is not None) else None}
    for _, c in items:
        ib.cancelMktData(c)
    return out


def spot(ib, ticker):
    """Spot for strike selection. A live ETF quote can return NaN bid/ask while
    last/close are fine, so fall back through them."""
    c = Stock(ticker.upper(), "SMART", "USD")
    ib.qualifyContracts(c)
    td = ib.reqMktData(c, "", False, False)
    ib.sleep(3)
    bid, ask = safe_px(td.bid), safe_px(td.ask)
    mid = (bid + ask) / 2 if bid and ask else None
    px = mid or bid or ask or safe_px(td.last) or safe_px(td.close) or safe_px(td.marketPrice())
    ib.cancelMktData(c)
    return px


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
    n = now_et()
    return max((n.replace(hour=16, minute=0, second=0, microsecond=0) - n).total_seconds() / 60.0, 1.0)


def expected_move(ib, cfg, expiry, S):
    mins = minutes_to_close()
    k = round(S)
    q = get_quotes(ib, {"c": option_contract(cfg["ticker"], expiry, k, "C"),
                        "p": option_contract(cfg["ticker"], expiry, k, "P")})
    ivs = [implied_vol(q[n]["mid"], S, k, mins, r) for n, r in (("c", "C"), ("p", "P")) if q[n]["mid"]]
    ivs = [v for v in ivs if 0.01 < v < 3.0]
    if not ivs:
        return None, None
    iv = sum(ivs) / len(ivs)
    return S * iv * math.sqrt(mins / MIN_PER_YEAR), iv


# ----------------------------------------------------------------- orders
def place_leg(ib, cfg, contract, action, limit_price, label, qty):
    px = round(limit_price, 2)
    print(f"  {label}: {action} {qty}x {contract.localSymbol or contract.symbol} @ ${px}...")
    trade = None
    order = LimitOrder(action, qty, px, tif="DAY")
    try:
        trade = ib.placeOrder(contract, order)
    except Exception as exc:
        print(f"    SUBMIT FAILED: {exc}")
        return False, None
    for _ in range(cfg["fill_wait_seconds"]):
        ib.sleep(1)
        s = trade.orderStatus.status
        if s == "Filled":
            print(f"    FILLED @ ${trade.orderStatus.avgFillPrice}")
            return True, float(trade.orderStatus.avgFillPrice)
        if s in IB_DONE_NOT_FILLED:
            print(f"    {s}")
            return False, None
    try:
        ib.cancelOrder(order)
    except Exception:
        pass
    # An order can fill while the cancel is in flight (seen live 2026-10-01: filled ~1s after a
    # 5s wait gave up). Wait for a terminal status; never assume "not filled" without one.
    for _ in range(60):
        ib.sleep(1)
        if trade.orderStatus.status == "Filled":
            print(f"    actually FILLED @ ${trade.orderStatus.avgFillPrice} (raced the cancel)")
            return True, float(trade.orderStatus.avgFillPrice)
        if trade.orderStatus.status in IB_DONE_NOT_FILLED:
            print("    did not fill in time -- cancelled")
            return False, None
    raise RuntimeError(f"{label}: cancel never confirmed (status {trade.orderStatus.status}) -- fill state unknown")


def price_ladder(action, bid, ask, mid, urgent=False):
    """[favorable, mid, aggressive]. The last step crosses the quote by CROSS so it actually fills:
    the backtest assumed every leg filled, and cheap wings often sit unfilled at the touch.
    urgent (stop exits): every step crosses the quote, because the backtest exits in the minute
    the stop is hit and a patient ladder lets a fast-moving short run away (2026-10-01: -$78)."""
    if urgent and bid is not None and ask is not None:
        if action == "BUY":
            return [round(ask + d, 2) for d in (CROSS, 0.05, 0.10)]
        return [round(max(bid - d, 0.01), 2) for d in (CROSS, 0.05, 0.10)]
    if mid is None:
        fb = (ask if action == "SELL" else bid) or 0.01
        return [round(fb, 2)] * 3
    mid = round(mid, 2)
    if action == "SELL":
        return [round(mid + (ask - mid) * 0.75, 2), mid,
                round(max(bid - CROSS, 0.01), 2) if bid else round(mid * 0.5, 2)]
    return [round(mid - (mid - bid) * 0.75, 2) if bid else round(mid * 0.75, 2), mid, round(ask + CROSS, 2)]


def place_leg_ladder(ib, cfg, contract, action, label, qty, bid, ask, mid, urgent=False):
    for i, px in enumerate(price_ladder(action, bid, ask, mid, urgent)):
        print(f"  [{label}] step {i + 1}: {action} @ ${px}")
        ok, fill = place_leg(ib, cfg, contract, action, px, label, qty)
        if ok:
            return True, fill
    return False, None


def place_condor(ib, cfg, contracts, limits, qty):
    """Sequential single-leg orders, longs first so no leg is ever uncovered. Each leg is re-quoted
    just before it is placed: legging takes a minute or more, and limits from the plan go stale in a
    moving market (2026-10-02 10:01: both short puts priced above a falling bid, never filled)."""
    fills = {}
    for name, action, label in (("long_put", "BUY", "long put"), ("long_call", "BUY", "long call"),
                                ("short_put", "SELL", "short put"), ("short_call", "SELL", "short call")):
        fresh = get_quotes(ib, {name: contracts[name]})[name]
        b, a, m = (fresh["bid"], fresh["ask"], fresh["mid"]) if fresh["mid"] is not None else limits[name]
        ok, px = place_leg_ladder(ib, cfg, contracts[name], action, label, qty, b, a, m)
        fills[name] = px
        if not ok:
            return False, fills, f"{name}_failed"
    return True, fills, "all_4_filled"


def close_condor_legs(ib, cfg, contracts, limits, qty, urgent=True):
    """Two-phase close of whichever legs are in `contracts`: buy back the shorts first (removes
    risk), then sell the longs. A wing is only sold once its short is gone -- otherwise it stays
    as cover. Returns (all_ok, {leg: fill price or None}) for the legs attempted."""
    fills, all_ok = {}, True
    for name, label in (("short_call", "close short call"), ("short_put", "close short put")):
        if name not in contracts:
            continue
        b, a, m = limits[name]
        ok, px = place_leg_ladder(ib, cfg, contracts[name], "BUY", label, qty, b, a, m, urgent=urgent)
        fills[name] = px
        all_ok = all_ok and ok
    for name, short, label in (("long_call", "short_call", "close long call"),
                               ("long_put", "short_put", "close long put")):
        if name not in contracts:
            continue
        if short in contracts and fills.get(short) is None:
            all_ok = False
            continue
        b, a, m = limits[name]
        ok, px = place_leg_ladder(ib, cfg, contracts[name], "SELL", label, qty, b, a, m, urgent=urgent)
        fills[name] = px
        all_ok = all_ok and ok
    return all_ok, fills


def unwind_entry(ib, cfg, contracts, fills, qty):
    """Undo a partly filled entry so the trader is flat again: close the filled legs, shorts first.
    Returns (all_closed, {leg: unwind fill or None})."""
    held = {n: contracts[n] for n, px in fills.items() if px is not None}
    q = get_quotes(ib, held)
    limits = {n: (q[n]["bid"], q[n]["ask"], q[n]["mid"]) for n in held}
    ok, out = close_condor_legs(ib, cfg, held, limits, qty)
    return ok and all(out.get(n) is not None for n in held), out


# ----------------------------------------------------------------- strategy
def build_condor(ib, cfg, expiry, S, u):
    st = {"short_put": round(S - cfg["short_u"] * u), "short_call": round(S + cfg["short_u"] * u)}
    st["long_put"] = st["short_put"] - cfg["wing"]
    st["long_call"] = st["short_call"] + cfg["wing"]
    contracts = {n: option_contract(cfg["ticker"], expiry, k, "P" if "put" in n else "C") for n, k in st.items()}
    q = get_quotes(ib, contracts)
    missing = [n for n, v in q.items() if v["mid"] is None]
    if missing:
        return None, f"no quote for {', '.join(missing)}"
    credit_mid = (q["short_put"]["mid"] - q["long_put"]["mid"]) + (q["short_call"]["mid"] - q["long_call"]["mid"])
    conservative = (q["short_put"]["bid"] - q["long_put"]["ask"]) + (q["short_call"]["bid"] - q["long_call"]["ask"])
    if conservative < cfg["min_credit"]:
        return None, f"credit too thin (conservative ${conservative:.2f})"
    return dict(strikes=st, contracts=contracts, quotes=q, credit_mid=credit_mid, conservative=conservative,
                limits={n: (q[n]["bid"], q[n]["ask"], q[n]["mid"]) for n in contracts}), None


def side_of(leg):
    return "put" if "put" in leg else "call"


def live_strikes(pos):
    """The legs a position still holds: a side closed by a near-money pass is gone."""
    gone = set(pos.get("legs_closed", []))
    return {n: k for n, k in pos["strikes"].items() if side_of(n) not in gone}


def condor_value(ib, cfg, expiry, strikes):
    """Mid cost to close the given legs (any subset of the four): shorts bought, longs sold."""
    contracts = {n: option_contract(cfg["ticker"], expiry, k, "P" if "put" in n else "C") for n, k in strikes.items()}
    q = get_quotes(ib, contracts)
    if any(v["mid"] is None for v in q.values()):
        return None, contracts, q
    val = sum(q[n]["mid"] * (1 if n.startswith("short") else -1) for n in strikes)
    return val, contracts, q


def enter(ib, cfg, expiry, state, dry, slot):
    S = spot(ib, cfg["ticker"])
    if not S:
        log("entry_skipped", slot=slot, reason="no spot")
        return
    u, iv = expected_move(ib, cfg, expiry, S)
    if not u:
        log("entry_skipped", slot=slot, reason="no ATM IV")
        return
    plan, err = build_condor(ib, cfg, expiry, S, u)
    if plan is None:
        log("entry_skipped", slot=slot, reason=err, spot=round(S, 2), u=round(u, 2))
        return
    st = plan["strikes"]
    log("entry_plan", slot=slot, spot=round(S, 2), iv=round(iv, 4), u=round(u, 2), strikes=st,
        credit_mid=round(plan["credit_mid"] * 100, 0), conservative=round(plan["conservative"] * 100, 0))
    if dry:
        return
    ok, fills, status = place_condor(ib, cfg, plan["contracts"], plan["limits"], cfg["qty"])
    if not ok:
        filled = {n: px for n, px in fills.items() if px is not None}
        if not filled:
            log("entry_skipped", slot=slot, reason=f"first leg never filled ({status}) -- no position")
            return
        # The backtest only knows complete condors: undo the partial entry and keep trading.
        log("entry_partial", slot=slot, status=status, filled=filled)
        uok, uf = unwind_entry(ib, cfg, plan["contracts"], fills, cfg["qty"])
        pnl = 100 * cfg["qty"] * sum((uf[n] - px) if n.startswith("long") else (px - uf[n])
                                     for n, px in filled.items() if uf.get(n) is not None)
        state.setdefault("unwound", []).append(dict(slot=slot, time=now_et().isoformat(), status=status,
                                                    entry_fills=filled, unwind_fills=uf, pnl=round(pnl, 2)))
        state["realized"] += pnl
        if uok:
            save_state(state)
            log("entry_unwound", slot=slot, status=status, pnl=round(pnl, 2))
            telegram(cfg, f"{cfg['ticker']} {slot} entry only partly filled ({status}) -- unwound flat, ${pnl:+.0f}")
            return
        still = {n: st[n] for n in filled if uf.get(n) is None}
        state.setdefault("partial_legs", []).append(dict(slot=slot, time=now_et().isoformat(), legs=still))
        save_state(state)
        pause(cfg, f"partial entry could not be unwound ({status}) -- still open in the account: {still}")
        return
    credit = (fills["short_put"] - fills["long_put"]) + (fills["short_call"] - fills["long_call"])
    pos = dict(id=f"{now_et():%H%M%S}", slot=slot, strikes=st, credit=credit, qty=cfg["qty"],
               stop_loss_usd=cfg["stop_mult"] * credit * 100 * cfg["qty"],
               entry_time=now_et().isoformat(), fills=fills)
    state["open"].append(pos)
    save_state(state)
    log("entered", **pos)
    telegram(cfg, f"{cfg['ticker']} {st['long_put']:g}/{st['short_put']:g}P {st['short_call']:g}/{st['long_call']:g}C "
                  f"credit ${credit * 100:.0f}, stop -${pos['stop_loss_usd']:.0f}")


def close_condor(ib, cfg, expiry, pos, state, reason, dry):
    live = live_strikes(pos)
    val, contracts, q = condor_value(ib, cfg, expiry, live)
    if val is None:
        log("close_failed", pos=pos["id"], reason="no quotes")
        return
    if dry:
        log("close_plan", pos=pos["id"], reason=reason,
            pnl=round((pos["credit"] - val - pos.get("closed_cost", 0.0)) * 100 * pos["qty"], 0))
        return
    # The backtest closes the whole condor at the stop: legs that don't fill stay tracked and are
    # retried every monitor cycle until all four are closed.
    done = pos.setdefault("close_fills", {})
    pos.setdefault("close_reason", reason)
    todo = {n: c for n, c in contracts.items() if n not in done}
    limits = {n: (q[n]["bid"], q[n]["ask"], q[n]["mid"]) for n in todo}
    _, fills = close_condor_legs(ib, cfg, todo, limits, pos["qty"])
    done.update({n: px for n, px in fills.items() if px is not None})
    if any(n not in done for n in live):
        pos["close_attempts"] = pos.get("close_attempts", 0) + 1
        save_state(state)
        still = sorted(n for n in live if n not in done)
        log("close_incomplete", pos=pos["id"], attempt=pos["close_attempts"], still_open=still)
        if pos["close_attempts"] == 3:
            telegram(cfg, f"{cfg['ticker']} {pos['id']} stop-close still incomplete after 3 tries "
                          f"(open: {still}) -- retrying every cycle", high_priority=True)
        return
    state["open"] = [p for p in state["open"] if p["id"] != pos["id"]]
    reason = pos["close_reason"]
    ev = sum(done[n] * (1 if "short" in n else -1) for n in live) + pos.get("closed_cost", 0.0)
    realized = (pos["credit"] - ev) * 100 * pos["qty"]
    state["closed"].append({**pos, "close_reason": reason, "pnl": realized})
    state["realized"] += realized
    save_state(state)
    log("closed", pos=pos["id"], reason=reason, pnl=round(realized, 0))
    telegram(cfg, f"{cfg['ticker']} {pos['id']} closed ({reason}): ${realized:+.0f}")


def held_options(ib, cfg, expiry):
    """{(strike, right): position} for today's options on the ticker, as the ACCOUNT holds them --
    IBKR can change that on its own (2026-10-06 15:42: it bought back 6 of our 7 short 779P)."""
    out = {}
    for p in ib.positions():
        c = p.contract
        if c.secType == "OPT" and c.symbol == cfg["ticker"].upper() and c.lastTradeDateOrContractMonth == expiry:
            key = (float(c.strike), c.right)
            out[key] = out.get(key, 0) + p.position
    return out


def close_near_money_side(ib, cfg, expiry, pos, S, state, dry, dist=None, held=None):
    """Close each side whose short is in the money or within `dist` of spot: buy the short back, then
    sell its wing. 2026-10-06: the old rule (|spot - strike| <= dist) skipped a short put $0.79 in the
    money, which was then assigned. `held` (held_options(), shared across one pass) caps each order
    at what the account really holds -- the same day the 15:58 close re-bought four 779P that IBKR
    had already bought back, leaving the account long -- and is debited as legs close."""
    dist = cfg["near_money_dist"] if dist is None else dist
    for side, sk, lk, right in (("put", "short_put", "long_put", "P"), ("call", "short_call", "long_call", "C")):
        if side in pos.get("legs_closed", []):
            continue
        k = pos["strikes"][sk]
        if (S - k if right == "P" else k - S) > dist:  # out of the money by more than dist
            continue
        qty = pos["qty"]
        short_key, long_key = (float(k), right), (float(pos["strikes"][lk]), right)
        buy = qty if held is None else min(qty, max(-held.get(short_key, 0), 0))
        cons = {n: option_contract(cfg["ticker"], expiry, pos["strikes"][n], right) for n in (sk, lk)}
        q = get_quotes(ib, cons)
        log("near_money_close", pos=pos["id"], side=side, spot=round(S, 2), strike=k, dist=dist,
            covered_by_broker=qty - buy)
        if dry:
            continue
        cost = 0.0
        if buy:
            ok, px = place_leg_ladder(ib, cfg, cons[sk], "BUY", f"close short {side}", buy,
                                      q[sk]["bid"], q[sk]["ask"], q[sk]["mid"], urgent=True)
            if not ok:
                log("near_money_close_failed", pos=pos["id"], side=side, leg=sk)
                telegram(cfg, f"{cfg['ticker']} {pos['id']}: could not buy back short {side} {k:g} "
                              f"near the close -- assignment risk, close by hand", high_priority=True)
                continue
            cost += px * buy / qty
            if held is not None:
                held[short_key] = held.get(short_key, 0) + buy
        if buy < qty:
            # IBKR's pre-expiry liquidation got there first; its fill isn't visible here, so book it at mid.
            cost += (q[sk]["mid"] or 0.0) * (qty - buy) / qty
            log("near_money_short_gone", pos=pos["id"], side=side, strike=k, covered_by_broker=qty - buy)
        sell = qty if held is None else min(qty, max(held.get(long_key, 0), 0))
        if sell:
            ok, px = place_leg_ladder(ib, cfg, cons[lk], "SELL", f"close long {side}", sell,
                                      q[lk]["bid"], q[lk]["ask"], q[lk]["mid"], urgent=True)
            if ok:
                cost -= px * sell / qty
                if held is not None:
                    held[long_key] = held.get(long_key, 0) - sell
        pos.setdefault("legs_closed", []).append(side)
        pos["closed_cost"] = pos.get("closed_cost", 0.0) + cost
        save_state(state)
    if {"put", "call"} <= set(pos.get("legs_closed", [])) and pos in state["open"]:
        state["open"] = [p for p in state["open"] if p["id"] != pos["id"]]
        realized = (pos["credit"] - pos.get("closed_cost", 0.0)) * 100 * pos["qty"]
        state["closed"].append({**pos, "close_reason": "near-money close", "pnl": realized})
        state["realized"] += realized
        save_state(state)
        log("closed", pos=pos["id"], reason="near-money close", pnl=round(realized, 0))


def near_money_pass(ib, cfg, expiry, state, dry, dist):
    S = spot(ib, cfg["ticker"])
    if not S:
        return
    held = held_options(ib, cfg, expiry)
    for pos in list(state["open"]):
        if "close_fills" not in pos:
            close_near_money_side(ib, cfg, expiry, pos, S, state, dry, dist, held)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="0DTE iron condor V2 (IBKR)")
    ap.add_argument("--ticker", help="override config ticker (SPY/QQQ)")
    ap.add_argument("--port", type=int, help="override TWS port (7497 paper, 7496 live)")
    ap.add_argument("--qty", type=int, help="contracts per condor")
    ap.add_argument("--dry-run", action="store_true", help="price and log everything, place nothing")
    ap.add_argument("--dir", help="instance folder with its own config.json, state.json, log and PAUSED.flag "
                                  "(default: this script's folder); lets several tickers run in parallel")
    args = ap.parse_args()

    global STATE_FILE, LOG_FILE, PAUSE_FLAG
    base = Path(args.dir).resolve() if args.dir else HERE
    STATE_FILE, LOG_FILE, PAUSE_FLAG = base / "state.json", base / "condor_v2_log.jsonl", base / "PAUSED.flag"
    cfg = load_config(base)
    for k, v in (("ticker", args.ticker), ("tws_port", args.port), ("qty", args.qty)):
        if v:
            cfg[k] = v
    dry = args.dry_run

    if PAUSE_FLAG.exists():
        print(f"{PAUSE_FLAG} present -- refusing to run:\n{PAUSE_FLAG.read_text()}")
        sys.exit(1)
    now = now_et()
    if now.weekday() >= 5:
        print("weekend -- nothing to do")
        return
    expiry, state = f"{now:%Y%m%d}", load_state()
    entry_times = [hhmm(s) for s in cfg["entry_times"]]
    near_close = hhmm(cfg["near_money_close"])
    near_early = hhmm(cfg["near_money_early"]) if cfg.get("near_money_early") else None

    ib = IB()
    try:
        ib.connect(cfg["tws_host"], cfg["tws_port"], clientId=cfg["client_id"], timeout=20)
    except Exception as e:
        print(f"\nCannot reach TWS/IB Gateway at {cfg['tws_host']}:{cfg['tws_port']} ({type(e).__name__}).\n"
              "  1. Is TWS (or IB Gateway) running and logged in?\n"
              "  2. Global Configuration -> API -> Settings: 'Enable ActiveX and Socket Clients' ticked,\n"
              f"     and the socket port set to {cfg['tws_port']} (7497 TWS paper, 7496 TWS live,\n"
              "     4002 Gateway paper, 4001 Gateway live).\n"
              "  3. 127.0.0.1 in Trusted IPs, and 'Read-Only API' unticked.\n"
              f"  4. client_id {cfg['client_id']} not already used by another script.\n")
        sys.exit(2)
    live_port = cfg["tws_port"] in (7496, 4001)
    log("start", ticker=cfg["ticker"], port=cfg["tws_port"], account="LIVE" if live_port else "paper",
        dry_run=dry, qty=cfg["qty"], expiry=expiry, accounts=ib.managedAccounts())
    telegram(cfg, f"started {cfg['ticker']} on port {cfg['tws_port']}"
                  f"{' DRY-RUN' if dry else ''}, {len(entry_times)} entries planned")
    try:
        done = {p.get("slot") for p in state["open"] + state["closed"]
                + state.get("unwound", []) + state.get("partial_legs", [])}
        early_done = False
        while now_et().time() < dtime(16, 0):
            t = now_et().time()
            for et in entry_times:
                slot = f"{et:%H:%M}"
                end = (datetime.combine(now_et().date(), et) + timedelta(minutes=cfg["entry_window_min"])).time()
                if slot in done or not (et <= t < end):
                    continue
                if PAUSE_FLAG.exists():
                    log("entry_blocked", slot=slot, reason="paused mid-session -- no new entries")
                    done.add(slot)
                    continue
                open_pnl = 0.0
                for p in state["open"]:
                    v = condor_value(ib, cfg, expiry, live_strikes(p))[0] if live_strikes(p) else 0.0
                    if v is not None:
                        open_pnl += (p["credit"] - v - p.get("closed_cost", 0.0)) * 100 * p["qty"]
                if state["realized"] + open_pnl <= cfg["daily_stop_usd"]:
                    log("entry_blocked", slot=slot, reason="daily stop", day_pnl=round(state["realized"] + open_pnl, 0))
                else:
                    enter(ib, cfg, expiry, state, dry, slot)
                done.add(slot)
            for pos in list(state["open"]):
                if "close_fills" in pos:
                    close_condor(ib, cfg, expiry, pos, state, pos.get("close_reason", "stop"), dry)
                    continue
                if not live_strikes(pos):
                    continue
                # a side closed by the 15:35 pass counts at what it cost to close
                val = condor_value(ib, cfg, expiry, live_strikes(pos))[0]
                if val is not None and (val + pos.get("closed_cost", 0.0) - pos["credit"]) * 100 * pos["qty"] \
                        >= pos["stop_loss_usd"]:
                    close_condor(ib, cfg, expiry, pos, state, f"stop {cfg['stop_mult']:g}x", dry)
            if near_early and not early_done and near_early <= t < near_close:
                early_done = True
                near_money_pass(ib, cfg, expiry, state, dry, cfg["near_money_dist_early"])
            if t >= near_close:
                near_money_pass(ib, cfg, expiry, state, dry, cfg["near_money_dist"])
                break
            ib.sleep(cfg["monitor_seconds"])
        log("day_end", closed=len(state["closed"]), realized=round(state["realized"], 0), held=len(state["open"]))
        telegram(cfg, f"day done: {len(state['closed'])} closed (${state['realized']:+.0f}), "
                      f"{len(state['open'])} held to expiry")
    except Exception as e:
        pause(cfg, f"unhandled error: {type(e).__name__} {e}")
        raise
    finally:
        save_state(state)
        ib.disconnect()


if __name__ == "__main__":
    main()
