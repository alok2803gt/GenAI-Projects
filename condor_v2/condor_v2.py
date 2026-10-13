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
    # 2026-10-13 (user call, from the 258-day replay): the 10:01/10:31 entries lost in both
    # half-years, and a 1x stop nearly doubled the year's net vs 0.5x (+$8.3k vs +$4.7k per lot).
    "entry_times": ["11:01", "11:31", "12:01", "12:31", "13:01", "13:31", "14:01", "14:31"],
    "short_u": 0.5,
    "wing": 5.0,
    "stop_mult": 1.0,
    "min_credit": 0.10,
    "monitor_seconds": 45,
    # Near-money passes close the sides whose short is in the money or within `dist` of spot.
    # IBKR's own pre-expiry liquidation (2026-10-06 15:42: 6 of our 7 short 779P) is met by
    # reconcile_broker() every cycle from reconcile_from rather than pre-empted (user call
    # 2026-10-08). The early pass stays available ("15:35") but is off.
    "near_money_early": None,
    "near_money_dist_early": 1.0,
    "near_money_close": "15:57",       # was 15:58; on 2026-10-12 the pass only started at 15:59:23 (stop cycle ran first)
    "near_money_dist": 0.25,
    "reconcile_from": "14:45",        # after the last entry window, so our own fills can't race it
    "daily_stop_usd": -1500.0,
    "fill_wait_seconds": 20,
    "urgent_fill_wait_seconds": 6,    # stop/close steps: re-quoted each step, so short waits
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


def keep_awake(on=True):
    """Hold off Windows idle sleep and screen-off while the trader runs. On 2026-10-07 at 15:15 the
    laptop (Modern Standby) went to standby on AC once its 5-hour screen timeout passed with the
    user away, and the trader, monitor and TWS froze through the close. Lid-close and the power
    button still sleep it."""
    if sys.platform != "win32":
        return
    import ctypes
    es_continuous, es_system, es_display = 0x80000000, 0x00000001, 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | ((es_system | es_display) if on else 0))


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
def place_leg(ib, cfg, contract, action, limit_price, label, qty, wait=None):
    px = round(limit_price, 2)
    print(f"  {label}: {action} {qty}x {contract.localSymbol or contract.symbol} @ ${px}...")
    trade = None
    order = LimitOrder(action, qty, px, tif="DAY")
    try:
        trade = ib.placeOrder(contract, order)
    except Exception as exc:
        print(f"    SUBMIT FAILED: {exc}")
        return False, None
    for _ in range(wait or cfg["fill_wait_seconds"]):
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
    the stop is hit and a patient ladder lets a fast-moving short run away (2026-10-01: -$78).
    The last urgent step crosses by the larger of $0.10 and 25% of the price: on 2026-10-08 a
    short put being bought back in a falling market outran a $0.10 cross for 16 minutes (-$131
    against -$36 in the replay)."""
    if urgent and bid is not None and ask is not None:
        if action == "BUY":
            return [round(ask + d, 2) for d in (CROSS, 0.05, max(0.10, 0.25 * ask))]
        return [round(max(bid - d, 0.01), 2) for d in (CROSS, 0.05, max(0.10, 0.25 * bid))]
    if mid is None:
        fb = (ask if action == "SELL" else bid) or 0.01
        return [round(fb, 2)] * 3
    mid = round(mid, 2)
    if action == "SELL":
        return [round(mid + (ask - mid) * 0.75, 2), mid,
                round(max(bid - CROSS, 0.01), 2) if bid else round(mid * 0.5, 2)]
    return [round(mid - (mid - bid) * 0.75, 2) if bid else round(mid * 0.75, 2), mid, round(ask + CROSS, 2)]


def place_leg_ladder(ib, cfg, contract, action, label, qty, bid, ask, mid, urgent=False):
    """Up to three limit orders, each cancelled before the next. Urgent ladders (stops, closes,
    unwinds) re-quote before every step after the first and wait only urgent_fill_wait_seconds:
    one quote held for three 20-second steps chased a moving short for minutes on 2026-10-08."""
    wait = cfg.get("urgent_fill_wait_seconds") if urgent else None
    for i in range(3):
        if urgent and i:
            try:
                fresh = get_quotes(ib, {"leg": contract})["leg"]
                if fresh["bid"] is not None and fresh["ask"] is not None:
                    bid, ask, mid = fresh["bid"], fresh["ask"], fresh["mid"]
            except Exception as exc:
                print(f"  [{label}] re-quote failed ({exc}) -- using the last quote")
        px = price_ladder(action, bid, ask, mid, urgent)[i]
        print(f"  [{label}] step {i + 1}: {action} @ ${px}")
        ok, fill = place_leg(ib, cfg, contract, action, px, label, qty, wait)
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
    # The threatened short first: the dearer one is the one the market is running at. On 2026-10-08
    # a stop bought the cheap short call first while the short put it was stopping out ran away.
    shorts = sorted((n for n in ("short_call", "short_put") if n in contracts),
                    key=lambda n: -(limits[n][2] or limits[n][1] or 0.0))
    for name in shorts:
        label = f"close {name.replace('_', ' ')}"
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
    """The legs a position still holds: not in a side closed by a near-money pass (legs_closed, the
    2026-10-06 format) and not closed one by one since (legs_gone)."""
    gone_sides, gone_legs = set(pos.get("legs_closed", [])), set(pos.get("legs_gone", []))
    return {n: k for n, k in pos["strikes"].items() if side_of(n) not in gone_sides and n not in gone_legs}


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


SIDES = (("put", "short_put", "long_put", "P"), ("call", "short_call", "long_call", "C"))


def mark_gone(pos, leg, px):
    """Record a leg as no longer held, at what closing it cost (a short bought back adds, a wing sold
    subtracts)."""
    pos.setdefault("legs_gone", []).append(leg)
    pos["closed_cost"] = pos.get("closed_cost", 0.0) + (px if leg.startswith("short") else -px)


def sell_wing(ib, cfg, pos, lk, contract, quote):
    """Sell a wing whose short is gone. One with no bid stays on the books and expires."""
    if not quote["bid"]:
        return
    ok, px = place_leg_ladder(ib, cfg, contract, "SELL", f"close {lk.replace('_', ' ')}", pos["qty"],
                              quote["bid"], quote["ask"], quote["mid"], urgent=True)
    if ok:
        mark_gone(pos, lk, px)


def book_if_done(pos, state, reason):
    """A position with no live legs left is booked at its credit minus what closing it cost."""
    if live_strikes(pos) or pos not in state["open"]:
        return
    state["open"] = [p for p in state["open"] if p["id"] != pos["id"]]
    realized = (pos["credit"] - pos.get("closed_cost", 0.0)) * 100 * pos["qty"]
    state["closed"].append({**pos, "close_reason": reason, "pnl": realized})
    state["realized"] += realized
    save_state(state)
    log("closed", pos=pos["id"], reason=reason, pnl=round(realized, 0))


def booked_net(state):
    """{(strike, right): net position} the books say this trader holds: the live legs of open positions,
    minus legs a stop-close in progress has already closed."""
    net = {}
    for p in state["open"]:
        done = p.get("close_fills", {})
        for leg, k in live_strikes(p).items():
            if leg not in done:
                key = (float(k), "P" if "put" in leg else "C")
                net[key] = net.get(key, 0) + p["qty"] * (-1 if leg.startswith("short") else 1)
    return net


def reconcile_broker(ib, cfg, expiry, state, dry):
    """React to IBKR's own pre-expiry liquidation rather than pre-empting it (user call 2026-10-08).
    On 2026-10-06 IBKR bought back 6 of our 7 short 779P at 15:42, because assignment would have
    exceeded the account's equity. Wherever the account is less short than the books, the difference
    was bought back for us. Totals are compared NET per strike, since one condor's wing can share a
    strike with another's short. Those shorts are marked gone at mid (IBKR's fill isn't visible
    here) and their wings sold, so stops and the 15:58 pass only ever trade legs that are really
    there."""
    held = held_options(ib, cfg, expiry)
    for key, want in booked_net(state).items():
        extra = held.get(key, 0) - want  # > 0: fewer shorts in the account than on the books
        if extra <= 0:
            continue
        side = "put" if key[1] == "P" else "call"
        sk, lk = f"short_{side}", f"long_{side}"
        for pos in list(state["open"]):
            if extra < pos["qty"]:
                break
            live = live_strikes(pos)
            if sk not in live or float(live[sk]) != key[0] or sk in pos.get("close_fills", {}):
                continue
            extra -= pos["qty"]
            wing_live = lk in live and lk not in pos.get("close_fills", {})
            cons = {sk: option_contract(cfg["ticker"], expiry, key[0], key[1])}
            if wing_live:
                cons[lk] = option_contract(cfg["ticker"], expiry, live[lk], key[1])
            q = get_quotes(ib, cons)
            log("short_closed_by_broker", pos=pos["id"], side=side, strike=key[0])
            if dry:
                continue
            mid = q[sk]["mid"] or 0.0
            if "close_fills" in pos:
                pos["close_fills"][sk] = mid  # a stop-close in progress: close_condor sells the wing
            else:
                mark_gone(pos, sk, mid)
                if wing_live:
                    sell_wing(ib, cfg, pos, lk, cons[lk], q[lk])
            save_state(state)
            telegram(cfg, f"{cfg['ticker']} {pos['id']}: IBKR bought back our short {key[0]:g}{key[1]} itself "
                          f"(pre-expiry liquidation) -- booked at ~${mid:.2f}, wing sold if it had a bid")
            if "close_fills" not in pos:
                book_if_done(pos, state, "closed by IBKR")


def close_near_money_side(ib, cfg, expiry, pos, S, state, dry, dist=None):
    """Close each side whose short is in the money or within `dist` of spot: buy the short back, then
    sell its wing. 2026-10-06: the old rule (|spot - strike| <= dist) skipped a short put $0.79 in the
    money, which was then assigned. Run after reconcile_broker(), so a short IBKR already bought back
    is no longer live here and is never bought twice."""
    dist = cfg["near_money_dist"] if dist is None else dist
    live = live_strikes(pos)
    for side, sk, lk, right in SIDES:
        if sk not in live:
            continue
        k = live[sk]
        if (S - k if right == "P" else k - S) > dist:  # out of the money by more than dist
            continue
        cons = {n: option_contract(cfg["ticker"], expiry, live[n], right) for n in (sk, lk) if n in live}
        q = get_quotes(ib, cons)
        log("near_money_close", pos=pos["id"], side=side, spot=round(S, 2), strike=k, dist=dist)
        if dry:
            continue
        ok, px = place_leg_ladder(ib, cfg, cons[sk], "BUY", f"close short {side}", pos["qty"],
                                  q[sk]["bid"], q[sk]["ask"], q[sk]["mid"], urgent=True)
        if not ok:
            log("near_money_close_failed", pos=pos["id"], side=side, leg=sk)
            telegram(cfg, f"{cfg['ticker']} {pos['id']}: could not buy back short {side} {k:g} "
                          f"near the close -- assignment risk, close by hand", high_priority=True)
            continue
        mark_gone(pos, sk, px)
        if lk in cons:
            sell_wing(ib, cfg, pos, lk, cons[lk], q[lk])
        save_state(state)
    book_if_done(pos, state, "near-money close")


def near_money_pass(ib, cfg, expiry, state, dry, dist):
    S = spot(ib, cfg["ticker"])
    if not S:
        return
    reconcile_broker(ib, cfg, expiry, state, dry)  # anything IBKR closed since the last cycle
    for pos in list(state["open"]):
        if "close_fills" not in pos:
            close_near_money_side(ib, cfg, expiry, pos, S, state, dry, dist)


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
    reconcile_from = hhmm(cfg["reconcile_from"])

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
    keep_awake(True)
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
            if t >= reconcile_from:
                reconcile_broker(ib, cfg, expiry, state, dry)
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
        keep_awake(False)
        save_state(state)
        ib.disconnect()


if __name__ == "__main__":
    main()
