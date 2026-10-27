#!/usr/bin/env python3
"""AE-clone: free rule-based version of Collective2's AELong, PAPER ONLY (user ask 2026-10-26).

Rules come from research/analyze_entries.py + analyze_exits.py (AELong's public trade history). Parameters
were picked on 2022-07..2024-06 and tested on 2024-07..2026-10 (research/backtest_clone.py):
  regime   no new entries while SPY closes below its 200-day SMA
  setup    S&P 100 stock, 12-1 month momentum > 0, 8-30% below its 52-week high, below its 50-day SMA,
           RSI14 < 50; strongest 12-1 momentum first; up to 20 equal slots (equity / 20 each)
  exits    stop: close <= entry * 0.80
           trail: once a close is >= entry * 1.10, exit when close <= peak * 0.92
           time: 250 trading days held
  cash     idle cash earns SPY's daily return (AELong parks cash in VOO/QQQ/SPMO)
  fills    signals on the close, paper-filled at the next open, 5 bps slippage + $1 per order

Each run processes every completed trading day since the last one (catches up after missed runs), then
messages Telegram: today's fills, tomorrow's orders, equity vs SPY. No broker connection, no orders.

    python ae_clone.py            # daily pass (task C2-AEClone, weekdays 17:00)
    python ae_clone.py --report   # positions, closed trades, performance vs SPY
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
STATE = HERE / "ae_clone_state.json"
LOG = HERE / "ae_clone_log.jsonl"
UNIVERSE_FILE = HERE / "research" / "sp100.txt"
SCANNER_CONFIG = HERE.parent / "ibkr_trader" / "backend" / "scanner_config.json"
ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0"}

P = {"STOP": 0.20, "TRAIL": 0.08, "TRAIL_ARM": 0.10, "MAX_HOLD": 250, "MAX_POS": 20,
     "LO": -0.30, "HI": -0.08, "SLIP": 0.0005, "COMM": 1.0, "CAPITAL": 100_000.0}


def now():
    return datetime.now(ET)


def log(action, **kw):
    rec = {"time": now().isoformat(), "action": action, **kw}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{now():%Y-%m-%d %H:%M:%S}] {action}: {json.dumps(kw)[:300]}", flush=True)


def save_state(st):
    """Atomic: a crash mid-write leaves the previous state intact (and a .bak of it)."""
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    if STATE.exists():
        STATE.replace(STATE.with_suffix(".bak"))
    tmp.replace(STATE)


def tg(text):
    """Alert bot (the same one as the traders), not the user's account."""
    try:
        cfg = json.loads(SCANNER_CONFIG.read_text(encoding="utf-8"))
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"], "text": "AE CLONE: " + text}, timeout=10)
    except Exception as e:
        print(f"telegram failed: {e}")


# ---------------------------------------------------------------- data

def fetch_bars(sym, days=800):
    """Daily bars, split/dividend-adjusted (Yahoo chart API)."""
    p1 = int(time.time()) - days * 86400
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={int(time.time())}"
           f"&interval=1d&events=div%2Csplit&includeAdjustedClose=true")
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q, adj = res["indicators"]["quote"][0], res["indicators"]["adjclose"][0]["adjclose"]
    bars = []
    for i, t in enumerate(res["timestamp"]):
        if q["close"][i] is None or adj[i] is None or q["open"][i] is None:
            continue
        f = adj[i] / q["close"][i]
        bars.append({"d": time.strftime("%Y-%m-%d", time.gmtime(t)), "o": q["open"][i] * f, "c": adj[i]})
    return bars


def load_all(universe):
    px, failed = {}, []
    for s in ["SPY"] + universe:
        try:
            px[s] = fetch_bars(s)
        except Exception as e:
            failed.append(s)
            print(f"{s}: fetch failed {e}")
        time.sleep(0.5)
    return px, failed


# ---------------------------------------------------------------- engine (mirrors research/backtest_clone.py)

def indicators_at(bars, i):
    if i < 253:
        return None
    c = [b["c"] for b in bars[i - 253:i + 1]]
    g = sum(max(c[k] - c[k - 1], 0) for k in range(len(c) - 14, len(c)))
    l = sum(max(c[k - 1] - c[k], 0) for k in range(len(c) - 14, len(c)))
    return {"sma50": sum(c[-50:]) / 50, "sma200": sum(c[-200:]) / 200, "hi252": max(c[-252:]),
            "mom12_1": c[-22] / c[-253] - 1, "rsi": 100 - 100 / (1 + g / (l or 1e-9))}


def new_state(start_day, spy_close):
    return {"cash": P["CAPITAL"], "pos": {}, "pending_buys": [], "pending_sells": [], "closed": [],
            "last_day": start_day, "start_day": start_day, "spy_start": spy_close, "curve": []}


def step(st, d, px, idx, universe, prev_d):
    """Process trading day d: fills at the open, cash sweep, mark + exits at the close, new signals."""
    events = {"day": d, "bought": [], "sold": []}
    pos = st["pos"]
    for s in st["pending_sells"]:
        i = idx[s].get(d) if s in idx else None
        if s in pos and i is not None:
            p = pos.pop(s)
            fill = px[s][i]["o"] * (1 - P["SLIP"])
            st["cash"] += p["qty"] * fill - P["COMM"]
            tr = {"sym": s, "entry": p["entry"], "exit": fill, "ret": fill / p["entry"] - 1, "held": p["held"],
                  "opened": p["opened"], "closed": d, "reason": p.get("exit_reason")}
            st["closed"].append(tr)
            events["sold"].append(tr)
    equity_open = st["cash"] + sum(p["qty"] * px[s][idx[s][d]]["o"] for s, p in pos.items() if d in idx.get(s, {}))
    for s in st["pending_buys"]:
        i = idx[s].get(d) if s in idx else None
        if i is None or s in pos or len(pos) >= P["MAX_POS"]:
            continue
        fill = px[s][i]["o"] * (1 + P["SLIP"])
        qty = int(min(equity_open / P["MAX_POS"], st["cash"] - P["COMM"]) // fill)
        if qty <= 0:
            continue
        st["cash"] -= qty * fill + P["COMM"]
        pos[s] = {"qty": qty, "entry": fill, "peak": fill, "held": 0, "opened": d}
        events["bought"].append({"sym": s, "qty": qty, "entry": fill})
    st["pending_buys"], st["pending_sells"] = [], []
    if prev_d and prev_d in idx["SPY"] and d in idx["SPY"]:
        st["cash"] *= px["SPY"][idx["SPY"][d]]["c"] / px["SPY"][idx["SPY"][prev_d]]["c"]
    equity = st["cash"]
    for s, p in pos.items():
        i = idx[s].get(d) if s in idx else None
        if i is None:
            continue
        c = px[s][i]["c"]
        p["held"] += 1
        p["peak"] = max(p["peak"], c)
        p["last"] = c
        equity += p["qty"] * c
        armed = p["peak"] >= p["entry"] * (1 + P["TRAIL_ARM"])
        reason = ("stop" if c <= p["entry"] * (1 - P["STOP"]) else
                  "trail" if armed and c <= p["peak"] * (1 - P["TRAIL"]) else
                  "time" if p["held"] >= P["MAX_HOLD"] else None)
        if reason:
            p["exit_reason"] = reason
            st["pending_sells"].append(s)
    st["curve"].append([d, round(equity, 2), round(px["SPY"][idx["SPY"][d]]["c"], 4)])
    spy_i = idx["SPY"][d]
    spy_f = indicators_at(px["SPY"], spy_i)
    events["regime_ok"] = bool(spy_f and px["SPY"][spy_i]["c"] >= spy_f["sma200"])
    if events["regime_ok"]:
        slots = P["MAX_POS"] - len(pos) + len(st["pending_sells"])
        cands = []
        for s in universe:
            i = idx[s].get(d) if s in idx else None
            if i is None or s in pos or slots <= 0:
                continue
            f = indicators_at(px[s], i)
            if not f:
                continue
            c = px[s][i]["c"]
            if f["mom12_1"] > 0 and P["LO"] <= c / f["hi252"] - 1 <= P["HI"] and c < f["sma50"] and f["rsi"] < 50:
                cands.append((f["mom12_1"], s))
        st["pending_buys"] = [s for _, s in sorted(cands, reverse=True)[:max(slots, 0)]]
    st["last_day"] = d
    events["equity"] = equity
    return events


def completed_days(px, after):
    """Trading days (SPY calendar) after `after` whose daily bar is final."""
    today = now().date().isoformat()
    cutoff_ok = now().hour * 60 + now().minute >= 16 * 60 + 30
    return [b["d"] for b in px["SPY"] if b["d"] > after and (b["d"] < today or (b["d"] == today and cutoff_ok))]


# ---------------------------------------------------------------- main

def run():
    universe = [s for s in UNIVERSE_FILE.read_text().split() if s != "BK"]
    px, failed = load_all(universe)
    if "SPY" not in px:
        log("fetch_failed", error="SPY unavailable; nothing processed")
        tg("data fetch failed (SPY); will retry next run.")
        return
    idx = {s: {b["d"]: i for i, b in enumerate(bars)} for s, bars in px.items()}
    st = json.loads(STATE.read_text()) if STATE.exists() else None
    if st is None:
        days = completed_days(px, "0000")
        d0 = days[-1]
        st = new_state(d0, px["SPY"][idx["SPY"][d0]]["c"])
        ev = step(st, d0, px, idx, universe, None)  # first signals only; nothing held yet
        log("started", day=d0, pending_buys=st["pending_buys"], regime_ok=ev["regime_ok"])
        save_state(st)
        tg(f"paper tracking started {d0} with ${P['CAPITAL']:,.0f}. SPY regime "
           f"{'OK' if ev['regime_ok'] else 'OFF (below 200d)'}. Buys at next open: "
           f"{', '.join(st['pending_buys']) or 'none'}.")
        return
    # A held or queued symbol without data would be skipped (unmarked, exit unchecked, queued order dropped).
    # Don't advance at all; the state is untouched, so the next run retries the same days.
    needed = set(st["pos"]) | set(st["pending_buys"]) | set(st["pending_sells"])
    missing = sorted(needed & set(failed))
    if missing:
        log("fetch_failed", error="held/queued symbols missing; nothing processed", symbols=missing)
        tg(f"data fetch failed for held/queued {', '.join(missing)}; no day processed, will retry next run.")
        return
    days = completed_days(px, st["last_day"])
    if not days:
        log("no_new_day", last_day=st["last_day"])
        return
    prev = st["last_day"]
    msgs = []
    for d in days:
        ev = step(st, d, px, idx, universe, prev)
        prev = d
        log("day", day=d, bought=ev["bought"], sold=ev["sold"], equity=round(ev["equity"], 2),
            pending_buys=st["pending_buys"], pending_sells=st["pending_sells"], regime_ok=ev["regime_ok"])
        for b in ev["bought"]:
            msgs.append(f"{d} BOUGHT {b['qty']} {b['sym']} @ {b['entry']:.2f}")
        for t in ev["sold"]:
            msgs.append(f"{d} SOLD {t['sym']} @ {t['exit']:.2f} ({t['ret']:+.1%}, {t['held']}d, {t['reason']})")
    save_state(st)
    eq, spy_now = st["curve"][-1][1], st["curve"][-1][2]
    perf = f"equity ${eq:,.0f} ({eq / P['CAPITAL'] - 1:+.1%}) vs SPY {spy_now / st['spy_start'] - 1:+.1%} since {st['start_day']}"
    invested = sum(p["qty"] * p.get("last", p["entry"]) for p in st["pos"].values())
    tg("\n".join(msgs + [
        f"{perf}; {len(st['pos'])} positions, {invested / eq:.0%} invested (rest tracks SPY).",
        f"next open: buy {', '.join(st['pending_buys']) or 'none'}; sell {', '.join(st['pending_sells']) or 'none'}"
        f"{'' if ev['regime_ok'] else ' (SPY below 200d: no new buys)'}."]
        + ([f"data fetch failed for {', '.join(failed)}"] if failed else [])))


def report():
    st = json.loads(STATE.read_text())
    eq, spy_now = st["curve"][-1][1], st["curve"][-1][2]
    peak, mdd = 0, 0
    for _, e, _ in st["curve"]:
        peak = max(peak, e)
        mdd = min(mdd, e / peak - 1)
    print(f"AE-clone paper since {st['start_day']} (last processed {st['last_day']})")
    print(f"  equity ${eq:,.0f} ({eq / P['CAPITAL'] - 1:+.2%}), max drawdown {mdd:.1%}; "
          f"SPY {spy_now / st['spy_start'] - 1:+.2%}")
    print(f"  open positions ({len(st['pos'])}):")
    for s, p in sorted(st["pos"].items()):
        last = p.get("last", p["entry"])
        print(f"    {s:6} {p['qty']:>5} @ {p['entry']:8.2f}  last {last:8.2f}  {last / p['entry'] - 1:+6.1%}  "
              f"held {p['held']:3}d  since {p['opened']}")
    cl = st["closed"]
    if cl:
        w = sum(t["ret"] > 0 for t in cl)
        print(f"  closed trades: {len(cl)}, winners {w} ({w / len(cl):.0%}), avg {sum(t['ret'] for t in cl) / len(cl):+.2%}")
        for t in cl[-10:]:
            print(f"    {t['closed']} {t['sym']:6} {t['ret']:+6.1%} {t['held']:3}d {t['reason']}")
    print(f"  pending: buy {st['pending_buys']} sell {st['pending_sells']}")


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
