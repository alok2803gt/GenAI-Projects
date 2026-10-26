#!/usr/bin/env python3
"""Shadow copy-trader for the "VNKUMAR Trades - PREMIUM Group" Telegram channel (user ask 2026-10-16).

SHADOW ONLY: it places no orders. Each alert is priced at the REAL market the moment it is seen,
through a READ-ONLY IBKR connection, and booked in a ledger with two P&L columns:
  his        the prices he quotes in his messages (what the channel claims)
  realistic  buy at the ask and sell at the bid when we saw the alert (what following would have got)
The gap between them is the cost of acting on his alerts after he posts them.

Message handling (he writes free text, so this is conservative):
  entry   "$SPY 773c 10/16 0.54" / "Try $770p $SPY 10/09 $0.9" / "$SPY 750/735 debit put spread 11/20
          expiration for $2.65". Any ticker. Recap posts ("Open positions ...") never open anything.
  exit    any close word (close / closing / closed / closure / sold). The position is picked by the
          ticker, strike+right, spread strikes or expiry the message names; if it names none, the
          position he mentioned most recently ("Closed for 0.75" right after an entry). "half" closes
          half. "you can ... or hold" is advice, not an exit, and is only logged.
  expiry  anything still open at 15:58 on its expiry day is closed at the bid ("expired").
Every decision goes to vnkumar_shadow_log.jsonl and to Telegram, so a wrong match is visible.

Runs continuously (watchdog run_vnkumar_shadow.ps1, task VNKumar-Shadow at logon): every message he
posts is forwarded to the user's Telegram immediately, day or night; trade alerts posted outside
09:30-16:00 ET queue and are priced at the next open (and flagged late).

    python vnkumar_shadow.py            # run forever (the watchdog restarts it if it dies)
    python vnkumar_shadow.py --report   # closed trades + open positions, his vs realistic
"""
import asyncio
import json
import re
from collections import defaultdict
import sys
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from ib_insync import IB, Option
from telethon import TelegramClient
from telethon.sessions import SQLiteSession, StringSession

import reader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
STATE = HERE / "vnkumar_shadow_state.json"
LOG = HERE / "vnkumar_shadow_log.jsonl"
CHANNEL_ID = -1001647992135
CLIENT_ID = 1650
ET = ZoneInfo("America/New_York")
POLL_SECONDS = 20
OPEN_T, EXPIRE_T, STOP_T = dtime(9, 30), dtime(15, 58), dtime(16, 10)

TICK = r"(\d+(?:\.\d+)?)"
PRICE = r"\$?(\d*\.\d+|\d+)"
RE_SINGLE = re.compile(r"\$([A-Z]{1,5})\s+\$?" + TICK + r"\s*([cCpP])\s+(\d{1,2}/\d{1,2})\s+(?:for\s+)?" + PRICE)
RE_SINGLE_ALT = re.compile(r"\$?" + TICK + r"([cCpP])\s+\$([A-Z]{1,5})\s+(\d{1,2}/\d{1,2})\s+" + PRICE)
RE_SPREAD = re.compile(r"\$([A-Z]{1,5})\s+" + TICK + r"/" + TICK + r"\s+(debit|credit)\s+(call|put)\s+spread\s+"
                       r"(\d{1,2}/\d{1,2})[^$\d]*" + PRICE, re.I)
RE_CLOSE = re.compile(r"\b(close|closed|closing|closure|sold|exit|exited)\b", re.I)
# Plans and conditions are not exits (2026-10-19: "If it doesn't work will close on Thursday" closed a
# position he was holding). The exit he then posts ("Closing for $1.56") is what closes it.
RE_ADVICE = re.compile(r"\byou can\b|\bor hold\b|\bwill (?:close|exit|sell)\b|\bif\b|\bclose on\b|"
                       r"\bplan(?:ning)? to\b|\bgoing to\b|\btry to hold\b", re.I)
RE_RECAP = re.compile(r"open positions?", re.I)
RE_STRIKE = re.compile(TICK + r"\s*([cCpP])\b")
RE_PAIR = re.compile(r"(\d+)/(\d+)")
RE_EXP = re.compile(r"\b(\d{1,2}/\d{1,2})\b")
RE_HIS_EXIT = re.compile(r"(?:for|@|fo)\s*\$?(\d*\.\d+|\d+)", re.I)


def now():
    return datetime.now(ET)


def log(action, **kw):
    rec = {"time": now().isoformat(), "action": action, **kw}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{now():%H:%M:%S}] {action}: {json.dumps(kw)[:300]}", flush=True)


def tg(text):
    """Alert bot (the same one as the traders), not the user's account."""
    try:
        cfg = json.loads(reader.SCANNER_CONFIG.read_text(encoding="utf-8"))
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"], "text": "VNKUMAR SHADOW: " + text}, timeout=10)
    except Exception as e:
        print(f"telegram failed: {e}")


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"last_id": None, "open": {}, "closed": [], "seq": 0}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))


def expiry_of(mmdd, ref):
    m, d = (int(x) for x in mmdd.split("/"))
    e = date(ref.year, m, d)
    if e < ref - timedelta(days=180):  # "1/16" posted in December
        e = date(ref.year + 1, m, d)
    return e


# ---------------------------------------------------------------- parsing
def parse_entries(text, ref):
    """Positions a message opens: [{ticker, legs: [(strike, right, +1|-1)], expiry, his}]."""
    if RE_RECAP.search(text) or RE_CLOSE.search(text):
        return []
    out = []
    for t, a, b, kind, cp, exp, px in RE_SPREAD.findall(text):
        r = "C" if cp.lower() == "call" else "P"
        sign = (1, -1) if kind.lower() == "debit" else (-1, 1)  # debit: buy the first strike
        out.append({"ticker": t, "legs": [(float(a), r, sign[0]), (float(b), r, sign[1])],
                    "expiry": expiry_of(exp, ref).isoformat(), "his": float(px), "kind": f"{kind.lower()} spread"})
    if out:
        return out
    for t, k, r, exp, px in RE_SINGLE.findall(text):
        out.append({"ticker": t, "legs": [(float(k), r.upper(), 1)], "expiry": expiry_of(exp, ref).isoformat(),
                    "his": float(px), "kind": "single"})
    for k, r, t, exp, px in RE_SINGLE_ALT.findall(text):
        out.append({"ticker": t, "legs": [(float(k), r.upper(), 1)], "expiry": expiry_of(exp, ref).isoformat(),
                    "his": float(px), "kind": "single"})
    return out


def label(p):
    legs = p["legs"]
    s = (f"{legs[0][0]:g}/{legs[1][0]:g}{legs[0][1]}" if len(legs) == 2 else f"{legs[0][0]:g}{legs[0][1]}")
    return f"{p['ticker']} {s} {p['expiry'][5:]}"


# All-caps words that are commentary, not tickers ("Closing NVDA" names a ticker; "ER", "FOMC" don't).
NOT_TICKERS = {"ER", "FOMC", "CPI", "PPI", "NFP", "ATH", "EMA", "SMA", "VWAP", "TP", "SL", "ITM", "OTM", "ATM",
               "DTE", "EOD", "AM", "PM", "OK", "ET", "EST", "PT", "STRANGLE", "STRADDLE", "CALL", "PUT", "OPEX",
               "GDP", "PCE", "RSI", "IV", "USA", "US", "I", "A"}


def named_tickers(text):
    """Every ticker a message names: $-prefixed in any case ("$SPcX" -> SPCX, 2026-10-21) plus bare
    all-caps words that aren't commentary ("Closing NVDA", 2026-10-23)."""
    out = {t.upper() for t in re.findall(r"\$([A-Za-z]{1,5})\b", text)}
    out |= {w for w in re.findall(r"\b([A-Z]{2,5})\b", text) if w not in NOT_TICKERS}
    return out


def match_exit(text, opens, ref):
    """(position ids to close, how matched). Filters by every identifier the message names."""
    cands = dict(opens)
    named = False
    said = named_tickers(text)
    tickers = {p["ticker"] for p in opens.values() if p["ticker"] in said}
    if said and not tickers:
        return [], f"names {sorted(said)}, none of which is open"  # never guess another position
    if tickers:
        named = True
        cands = {k: p for k, p in cands.items() if p["ticker"] in tickers}
    # "750/735" is a spread; "10/16" is a date -- strikes above 31 tell them apart
    pairs = {(float(a), float(b)) for a, b in RE_PAIR.findall(text) if max(int(a), int(b)) > 31}
    strikes = [(float(k), r.upper()) for k, r in RE_STRIKE.findall(text)]
    if pairs:
        named = True
        cands = {k: p for k, p in cands.items()
                 if len(p["legs"]) == 2 and (p["legs"][0][0], p["legs"][1][0]) in pairs}
    elif strikes:
        named = True
        cands = {k: p for k, p in cands.items() if any((leg[0], leg[1]) in strikes for leg in p["legs"])}
    exps = [expiry_of(e, ref).isoformat() for e in RE_EXP.findall(text)
            if int(e.split("/")[0]) <= 12 and int(e.split("/")[1]) <= 31]
    if exps and len(cands) > 1:
        cands = {k: p for k, p in cands.items() if p["expiry"] in exps} or cands
    if not named:
        if not opens:
            return [], "no open positions"
        last = max(opens, key=lambda k: opens[k]["mentioned"])
        return [last], "inferred: most recently mentioned"
    if len(cands) == 1:
        return list(cands), "named"
    # "Closed $SPcX STRANGLE for $3.75" (2026-10-21): one ticker, both legs -> close every leg of it
    if (len(tickers) == 1 and not pairs and not strikes
            and re.search(r"\b(strangle|straddle|both|all legs)\b", text, re.I)):
        return list(cands), "named ticker, all legs"
    # "Closing $SPY and $AMD near open" (2026-10-22): several tickers, one open position each -> all of them
    if len(tickers) > 1 and not pairs and not strikes:
        per = defaultdict(list)
        for k, p in cands.items():
            per[p["ticker"]].append(k)
        if all(len(v) == 1 for v in per.values()):
            return [v[0] for v in per.values()], f"named {len(per)} tickers"
    return [], f"ambiguous ({len(cands)} candidates)" if cands else "names nothing open"


def his_exit_price(text, pos):
    if re.search(r"break\s*even", text, re.I):
        return pos["his"]
    m = RE_HIS_EXIT.search(text)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------- pricing
async def leg_quotes(ib, p):
    """[(bid, ask)] per leg, or None if any leg has no two-sided quote."""
    out = []
    for strike, right, _ in p["legs"]:
        c = Option(p["ticker"], p["expiry"].replace("-", ""), strike, right, "SMART")
        await ib.qualifyContractsAsync(c)
        if not c.conId:
            return None
        t = ib.reqMktData(c, "", False, False)
        for _ in range(10):
            await asyncio.sleep(0.5)
            if t.bid == t.bid and t.ask == t.ask and t.bid is not None and t.ask and t.ask > 0:
                break
        ib.cancelMktData(c)
        bid = t.bid if t.bid == t.bid and t.bid is not None and t.bid >= 0 else None
        ask = t.ask if t.ask == t.ask and t.ask and t.ask > 0 else None
        if ask is None:
            return None
        out.append((bid or 0.0, ask))
    return out


def buy_cost(p, q):
    """Debit to open at the real market: long legs at the ask, short legs at the bid."""
    return round(sum((a if s > 0 else -b) for (b, a), (_, _, s) in zip(q, p["legs"])), 2)


def sell_value(p, q):
    """Credit to close at the real market: long legs at the bid, short legs at the ask."""
    return round(sum((b if s > 0 else -a) for (b, a), (_, _, s) in zip(q, p["legs"])), 2)


# ---------------------------------------------------------------- actions
async def open_pos(ib, st, e, msg):
    if e["expiry"] < msg["date"][:10]:  # "$SPY 768c 10/20" posted 10/23: a typo, not a real contract
        log("entry_bad_expiry", pos=label(e), msg=msg["text"][:120])
        tg(f"Skipped {label(e)}: expiry is before the post date (typo in his alert?) -- not tracked.")
        return
    key = (e["ticker"], tuple(map(tuple, e["legs"])), e["expiry"])
    for p in st["open"].values():
        if (p["ticker"], tuple(map(tuple, p["legs"])), p["expiry"]) == key:
            log("duplicate_entry_ignored", pos=label(p), msg=msg["text"][:120])
            return
    q = await leg_quotes(ib, e)
    st["seq"] += 1
    pid = str(st["seq"])
    p = {**e, "id": pid, "qty": 1.0, "opened_at": msg["date"], "seen_at": now().isoformat(),
         "real": buy_cost(e, q) if q else None, "mentioned": msg["id"], "closes": []}
    st["open"][pid] = p
    late = (now() - datetime.fromisoformat(msg["date"])).total_seconds()
    log("entered", pos=label(p), his=p["his"], real=p["real"], quotes=q, late_s=round(late), msg=msg["text"][:200])
    tg(f"ENTRY {label(p)} -- his ${p['his']}, real market ${p['real']} (ask{'/bid' if len(p['legs']) > 1 else ''}, "
       f"{late:.0f}s after his post). Shadow only, no order.")


async def close_pos(ib, st, pid, frac, his_px, why, msg_text):
    p = st["open"][pid]
    q = await leg_quotes(ib, p)
    real = sell_value(p, q) if q else None
    qty = round(p["qty"] * frac, 4) if frac < 1 else p["qty"]
    his_pnl = round((his_px - p["his"]) * 100 * qty, 2) if his_px is not None else None
    real_pnl = round((real - p["real"]) * 100 * qty, 2) if real is not None and p["real"] is not None else None
    p["qty"] = round(p["qty"] - qty, 4)
    rec = {"pos": label(p), "id": pid, "qty": qty, "his_in": p["his"], "his_out": his_px, "real_in": p["real"],
           "real_out": real, "his_pnl": his_pnl, "real_pnl": real_pnl, "why": why, "time": now().isoformat()}
    st["closed"].append(rec)
    if p["qty"] <= 0.0001:
        del st["open"][pid]
    log("closed", **rec, msg=msg_text[:200])
    tg(f"EXIT {qty:g}x {label(p)} ({why}) -- his ${his_px} (P&L {his_pnl}), real bid ${real} (P&L {real_pnl})")


async def handle(ib, st, msg):
    text, ref = msg["text"], datetime.fromisoformat(msg["date"]).date()
    entries = parse_entries(text, ref)
    for e in entries:
        await open_pos(ib, st, e, msg)
    if entries:
        return
    if RE_CLOSE.search(text) and not RE_RECAP.search(text):
        if RE_ADVICE.search(text):
            log("advice_only", msg=text[:200])
            return
        ids, how = match_exit(text, st["open"], ref)
        if not ids:
            log("exit_unmatched", how=how, open=[label(p) for p in st["open"].values()], msg=text[:200])
            if st["open"]:
                tg(f"Exit message not matched ({how}): \"{text[:150]}\" -- open: "
                   + ", ".join(label(p) for p in st["open"].values()))
            return
        frac = 0.5 if re.search(r"\bhalf\b", text, re.I) and not re.search(r"\bfull\b", text, re.I) else 1.0
        for pid in ids:
            await close_pos(ib, st, pid, frac, his_exit_price(text, st["open"][pid]), how, text)
        return
    # a non-trade message that names an open position still counts as "mentioning" it
    for pid in match_exit(text, st["open"], ref)[0] if RE_STRIKE.search(text) else []:
        st["open"][pid]["mentioned"] = msg["id"]


async def expire_due(ib, st):
    today = now().date().isoformat()
    for pid, p in list(st["open"].items()):
        if p["expiry"] < today:
            st["closed"].append({"pos": label(p), "id": pid, "qty": p["qty"], "his_in": p["his"], "his_out": None,
                                 "real_in": p["real"], "real_out": None, "his_pnl": None, "real_pnl": None,
                                 "why": "expired while the tracker was off -- unpriced", "time": now().isoformat()})
            del st["open"][pid]
            log("expired_unpriced", pos=label(p))
        elif p["expiry"] == today and now().time() >= EXPIRE_T:
            await close_pos(ib, st, pid, 1.0, None, "expired: no exit alert by 15:58", "")


def summary(st, day=None):
    rows = [c for c in st["closed"] if day is None or c["time"].startswith(day)]
    his = sum(c["his_pnl"] or 0 for c in rows if c["his_pnl"] is not None and c["real_pnl"] is not None)
    real = sum(c["real_pnl"] for c in rows if c["his_pnl"] is not None and c["real_pnl"] is not None)
    return rows, round(his, 2), round(real, 2)


def market_open(t=None):
    t = t or now()
    return t.weekday() < 5 and OPEN_T <= t.time() < dtime(16, 0)


def forward_text(m):
    """His message as it will appear on the user's Telegram (photos/polls can't be read as text)."""
    text = (m.message or "").strip()
    kind = type(m.media).__name__.replace("MessageMedia", "").lower() if m.media else ""
    if kind and kind != "webpage":
        text = f"[{kind}] {text}".strip()
    return text[:3500]


async def ensure_ib(ib, st):
    """Reconnect after TWS's nightly restart; one Telegram per outage, not per poll."""
    if ib.isConnected():
        return True
    try:
        await ib.connectAsync("127.0.0.1", 7497, clientId=CLIENT_ID, timeout=15, readonly=True)
        if st.get("ib_down_since"):
            tg("TWS is back -- pricing queued alerts now")
            st.pop("ib_down_since")
        log("ibkr_connected")
        return True
    except Exception as e:
        if not st.get("ib_down_since"):
            st["ib_down_since"] = now().isoformat()
            tg(f"TWS not answering ({type(e).__name__}) -- his messages are still forwarded; trades queue "
               f"until TWS is back")
        return False


async def run():
    """Runs continuously under run_vnkumar_shadow.ps1 (2026-10-16 user ask). Every message he posts is
    forwarded to the user's Telegram at once, at any hour; trade alerts queue in state["pending"] and
    are priced only while the market is open -- an evening alert is priced at the next 09:30."""
    st = load_state()
    st.setdefault("pending", [])
    sess = SQLiteSession(str(reader.SESSION))  # copy the login into memory: no sqlite lock with reader.py
    tc = TelegramClient(StringSession(StringSession.save(sess)), *reader.credentials())
    sess.close()
    await tc.connect()
    if not await tc.is_user_authorized():
        sys.exit("Telegram session not authorized -- see reader.py")
    if st["last_id"] is None:  # first run: start from now, never replay history as new trades
        async for m in tc.iter_messages(CHANNEL_ID, limit=1):
            st["last_id"] = m.id
        save_state(st)
        log("start_fresh", last_id=st["last_id"])
    ib = IB()
    log("running", open=[label(p) for p in st["open"].values()], pending=len(st["pending"]))
    try:
        while True:
            if not tc.is_connected():
                await tc.connect()
            for m in [m async for m in tc.iter_messages(CHANNEL_ID, min_id=st["last_id"], reverse=True)]:
                text = forward_text(m)
                if text:
                    tg(f"[{m.date.astimezone(ET):%H:%M}] {text}")
                if m.message:
                    st["pending"].append({"id": m.id, "text": m.message.strip(),
                                          "date": m.date.astimezone(ET).isoformat()})
                st["last_id"] = m.id
                save_state(st)
            if market_open() and await ensure_ib(ib, st):
                while st["pending"]:
                    await handle(ib, st, st["pending"][0])
                    st["pending"].pop(0)
                    save_state(st)
                await expire_due(ib, st)
                save_state(st)
            today = now().date().isoformat()
            if now().weekday() < 5 and now().time() >= STOP_T and st.get("summary_day") != today:
                rows, his, real = summary(st, today)
                tg(f"DAY {today}: {len(rows)} exits -- his P&L ${his:+.0f}, realistic ${real:+.0f} "
                   f"(1 contract each). Open: " + (", ".join(label(p) for p in st["open"].values()) or "none"))
                st["summary_day"] = today
                save_state(st)
                if ib.isConnected():
                    ib.disconnect()  # free the client id overnight; reconnects at the open
            await asyncio.sleep(POLL_SECONDS)
    finally:
        ib.disconnect()
        await tc.disconnect()


def report():
    st = load_state()
    rows, his, real = summary(st)
    print(f"{'closed':<28}{'qty':>5}{'his in':>8}{'his out':>8}{'real in':>8}{'real out':>9}{'his P&L':>9}{'real P&L':>9}  how")
    for c in rows:
        f = lambda v: "" if v is None else f"{v:g}"
        print(f"{c['pos']:<28}{c['qty']:>5g}{f(c['his_in']):>8}{f(c['his_out']):>8}{f(c['real_in']):>8}"
              f"{f(c['real_out']):>9}{f(c['his_pnl']):>9}{f(c['real_pnl']):>9}  {c['why']}")
    print(f"TOTAL (pairs priced both ways): his ${his:+.2f}, realistic ${real:+.2f}")
    print("open:", ", ".join(f"{label(p)} x{p['qty']:g} (his {p['his']}, real {p['real']})"
                             for p in st["open"].values()) or "none")


if __name__ == "__main__":
    if "--report" in sys.argv:
        report()
    else:
        asyncio.run(run())
