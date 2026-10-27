#!/usr/bin/env python3
"""Paper-track Collective2 strategies before paying for them (user ask 2026-10-26).

SHADOW ONLY: no orders, no C2 account. Uses only what Collective2 shows the public:
  - the closed-trade CSV (/strategy/csv/<id>): every closed trade with entry/exit time and price
  - the strategy page's embedded JSON: real broker fills of the manager and of live AutoTrade
    subscribers for each trade leg, plus subscriber count / AUM / live AutoTraders
Open positions are hidden from non-subscribers, so a trade is only seen once it closes.

Each run (task C2-Shadow, weekdays 17:15):
  - books every trade that closed since tracking began (first run = baseline, nothing booked)
  - flags whether the trade was also OPENED after tracking began ("copyable": a copier starting
    on the tracking date would have held it) or was already open before
  - measures slippage of real fills vs the model price C2 reports, in bps (positive = worse
    than model), so we see whether the published record is achievable
  - snapshots subscribers / AUM / live AutoTraders, and messages Telegram on new closes

    python c2_shadow.py            # one pass (what the task runs)
    python c2_shadow.py --report   # summary per strategy since tracking began
"""
import csv
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
STATE = HERE / "c2_shadow_state.json"
LOG = HERE / "c2_shadow_log.jsonl"
SCANNER_CONFIG = HERE.parent / "ibkr_trader" / "backend" / "scanner_config.json"
ET = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0"}

STRATEGIES = {
    144585915: "AELong",
    132169961: "Easy Value",
}


def now():
    return datetime.now(ET)


def log(action, **kw):
    rec = {"time": now().isoformat(), "action": action, **kw}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{now():%Y-%m-%d %H:%M:%S}] {action}: {json.dumps(kw)[:300]}", flush=True)


def tg(text):
    """Alert bot (the same one as the traders), not the user's account."""
    try:
        cfg = json.loads(SCANNER_CONFIG.read_text(encoding="utf-8"))
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      json={"chat_id": cfg["telegram_chat_id"], "text": "C2 SHADOW: " + text}, timeout=10)
    except Exception as e:
        print(f"telegram failed: {e}")


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"strategies": {}}


def save_state(s):
    """Atomic: a crash mid-write leaves the previous state intact (and a .bak of it)."""
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(s, indent=1), encoding="utf-8")
    if STATE.exists():
        STATE.replace(STATE.with_suffix(".bak"))
    tmp.replace(STATE)


# ---------------------------------------------------------------- fetch / parse

def fetch_csv(sid):
    r = requests.get(f"https://collective2.com/strategy/csv/{sid}", headers=UA, timeout=60)
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.text)))
    if rows and "Trade ID" not in rows[0]:
        raise ValueError(f"unexpected CSV header for {sid}: {list(rows[0])[:5]}")
    return rows


def fetch_page_json(sid):
    """trust block + first page of closed trades (with per-fill detail) from the strategy page."""
    r = requests.get(f"https://collective2.com/details/{sid}", headers=UA, timeout=60)
    r.raise_for_status()
    html = r.text
    dec = json.JSONDecoder()
    out = {"trust": None, "closed": []}
    i = html.find('"trust":')
    if i >= 0:
        out["trust"], _ = dec.raw_decode(html, i + len('"trust":'))
    i = html.find('"closedFirstPage":')
    if i >= 0:
        out["closed"], _ = dec.raw_decode(html, i + len('"closedFirstPage":'))
    return out


def slippage(trade):
    """Average adverse slippage (bps) of manager and subscriber fills vs the model leg price."""
    res = {"manager": [], "subscribers": []}
    for leg in trade.get("legs", []):
        model = leg.get("price")
        act = leg.get("action", "")
        if not model:
            continue
        buy = act in ("BTO", "BTC")
        for f in leg.get("fills", []):
            if not f.get("price"):
                continue
            bps = (f["price"] - model) / model * 1e4 * (1 if buy else -1)
            res["manager" if f.get("isManager") else "subscribers"].append(bps)
    return {k: (round(sum(v) / len(v), 1) if v else None) for k, v in res.items()} | {
        "n_sub_fills": len(res["subscribers"])}


def match_fills(row, closed_json):
    """Find the page-JSON trade matching a CSV row (same symbol, close within 2 minutes)."""
    try:
        closed_t = datetime.strptime(row["Closed Time ET"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=ET).timestamp()
    except ValueError:
        return None
    for t in closed_json:
        if t.get("symbol") != row["Symbol"]:
            continue
        last = max((leg.get("epoch", 0) for leg in t.get("legs", [])), default=0)
        if abs(last - closed_t) <= 120:
            return t
    return None


def trade_pct(row):
    o, c = float(row["Avg Price Open"]), float(row["Avg Price Closed"])
    sign = -1 if row["Side"].upper().startswith("SHORT") else 1
    return round((c - o) / o * 100 * sign, 2)


# ---------------------------------------------------------------- main pass

def run():
    st = load_state()
    for sid, name in STRATEGIES.items():
        key = str(sid)
        try:
            rows = fetch_csv(sid)
            page = fetch_page_json(sid)
        except Exception as e:
            log("fetch_failed", strategy=name, error=str(e)[:200])
            continue
        trust = page["trust"] or {}
        s = st["strategies"].get(key)
        if s is None:  # baseline: everything already closed is history, not tracked
            s = {"name": name, "since": now().isoformat(), "seen": [r["Trade ID"] for r in rows],
                 "booked": [], "trust": []}
            st["strategies"][key] = s
            log("baseline", strategy=name, prior_trades=len(rows), trust=trust)
            tg(f"tracking started for {name} (C2 {sid}). {len(rows)} past trades ignored. "
               f"Subscribers {trust.get('numSubs')}, live AutoTraders {trust.get('liveAutoTraders')}, "
               f"AUM ${trust.get('aumDollars', 0):,}.")
        since = datetime.fromisoformat(s["since"])
        seen = set(s["seen"])
        for r in rows:
            tid = r["Trade ID"]
            if tid in seen:
                continue
            seen.add(tid)
            if not r.get("Closed Time ET"):
                continue
            opened = datetime.strptime(r["Open Time ET"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=ET)
            fills = match_fills(r, page["closed"])
            rec = {
                "trade_id": tid, "symbol": r["Symbol"], "side": r["Side"], "qty": r["Qty Closed"],
                "opened": r["Open Time ET"], "closed": r["Closed Time ET"],
                "entry": float(r["Avg Price Open"]), "exit": float(r["Avg Price Closed"]),
                "pct": trade_pct(r), "pl_model": float(r["Trade P/L"] or 0),
                "hold_days": round((datetime.strptime(r["Closed Time ET"], "%Y-%m-%d %H:%M:%S")
                                    - opened.replace(tzinfo=None)).total_seconds() / 86400, 1),
                "copyable": opened >= since,
                "slippage_bps": slippage(fills) if fills else None,
            }
            s["booked"].append(rec)
            log("closed_trade", strategy=name, **rec)
            slip = rec["slippage_bps"] or {}
            tg(f"{name}: closed {rec['side']} {rec['symbol']} {rec['entry']:.2f} -> {rec['exit']:.2f} "
               f"({rec['pct']:+.2f}%, model ${rec['pl_model']:,.0f}, held {rec['hold_days']}d)"
               f"{'' if rec['copyable'] else ' [opened before tracking]'}. "
               f"Sub fills vs model: {slip.get('subscribers')} bps.")
        s["seen"] = sorted(seen)
        if not s["trust"] or s["trust"][-1].get("date") != now().date().isoformat():
            s["trust"].append({"date": now().date().isoformat(), **{k: trust.get(k) for k in
                               ("numSubs", "aumDollars", "liveAutoTraders")}})
    save_state(st)


def report():
    st = load_state()
    for key, s in st["strategies"].items():
        b = s["booked"]
        cp = [x for x in b if x["copyable"]]
        t0, t1 = (s["trust"][0], s["trust"][-1]) if s["trust"] else ({}, {})
        print(f"\n== {s['name']} (C2 {key}) tracking since {s['since'][:10]}")
        print(f"   subscribers {t0.get('numSubs')} -> {t1.get('numSubs')}, live AutoTraders "
              f"{t0.get('liveAutoTraders')} -> {t1.get('liveAutoTraders')}, AUM ${t1.get('aumDollars') or 0:,}")
        for label, xs in (("all closes", b), ("copyable (opened after start)", cp)):
            if not xs:
                print(f"   {label}: none yet")
                continue
            wins = sum(1 for x in xs if x["pct"] > 0)
            slips = [x["slippage_bps"]["subscribers"] for x in xs
                     if x["slippage_bps"] and x["slippage_bps"]["subscribers"] is not None]
            print(f"   {label}: {len(xs)} trades, {wins} winners ({wins / len(xs):.0%}), "
                  f"model P/L ${sum(x['pl_model'] for x in xs):,.0f}, avg trade {sum(x['pct'] for x in xs) / len(xs):+.2f}%, "
                  f"avg sub slippage {sum(slips) / len(slips):+.1f} bps" if slips else "")
        for x in b[-10:]:
            print(f"     {x['closed'][:10]} {x['side']:5} {x['symbol']:6} {x['pct']:+7.2f}%  ${x['pl_model']:>8,.0f}  "
                  f"{x['hold_days']:>5}d {'' if x['copyable'] else '(pre-start)'}")


if __name__ == "__main__":
    report() if "--report" in sys.argv else run()
