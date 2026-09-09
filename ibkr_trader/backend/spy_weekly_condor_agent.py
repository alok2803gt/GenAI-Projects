"""
Standalone SPY Weekly Condor agent -- full extraction from main.py (CEO
instruction 2026-09-07), following the exact precedent set by Day Trader's
own 2026-08-27 extraction (day_trader_agent.py) for the same reason: a
main.py crash/restart used to be able to silently interrupt this
strategy's Monday-entry/Friday-exit cycle or drop an in-flight order
attempt, with no way to recover except a human noticing.

This strategy has never gone live (state["spy_weekly_condor"]["enabled"]
was False, zero real positions, at the time of this extraction) -- the
safest possible strategy to move, since there was no in-flight position or
Monday/Friday cycle state to migrate.

Architecture (mirrors day_trader_agent.py):
  - Owns its own single IBKR connection (clientId=996) for both quoting
    and order execution -- IBKR-only since the 2026-09-07 Alpaca-removal
    migration (this file never touches Alpaca at all, unlike EVC which
    remains in main.py and still executes there).
  - Persists to its own spy_weekly_condor_state.json (previously held only
    in main.py's in-memory `state` dict + SPY_CONDOR_STATE_PATH -- same
    filename, so nothing else needs to change to keep reading it).
  - Exposes its own FastAPI server on port 8011 with the same endpoint
    shapes main.py used to serve (/spy-condor/status, /enable, /config,
    /close/{pos_id}, /enter-now). main.py's own /spy-condor/* routes
    become thin proxies, exactly like /day-trader/* already are.
  - Runs its own monitor loop (Monday entry window, Friday exit window,
    continuous stop-loss check) as a background asyncio task alongside the
    FastAPI server, both driven by one asyncio.run(main()) -- no more
    _run_in_streaming_loop cross-thread dance, since this process has only
    one event loop to begin with (that machinery existed in main.py only
    because of ITS separate streaming-loop-thread architecture).
  - Own connection watchdog -- real, recurring IBKR/TWS disconnects were
    observed live during this same day's testing (Error 1100, twice in 10
    minutes) -- ported from day_trader_agent.py's own hard-learned pattern
    (found missing the morning after ITS extraction, when TWS's routine
    daily restart silently killed that agent's connection for the rest of
    the day).
  - Own watchdog (run_spy_weekly_condor_agent.ps1) restarts on crash,
    matching every other standalone script in this codebase.
"""
import asyncio
import json
import logging
import logging.handlers
import os
import sqlite3
import sys
import threading
import traceback
from datetime import date, datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ib_insync import IB, Option as IbOpt, Stock as IbStock
from pydantic import BaseModel

sys.path.insert(0, ".")

ET = ZoneInfo("America/New_York")
IBKR_PORT       = int(os.environ.get("IBKR_PORT", "7496"))
IBKR_CLIENT_ID  = 996   # standalone-agent range: main.py=3, ashleyklieu=994, day_trader=995
AGENT_PORT      = 8011
SPY_CONDOR_STATE_PATH = "spy_weekly_condor_state.json"   # SAME filename main.py used
JOURNAL_DB_PATH = "trade_journal.db"
SCANNER_CFG_PATH = "scanner_config.json"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_event_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S")
_event_log_handler = logging.handlers.RotatingFileHandler(
    os.path.join(_SCRIPT_DIR, "spy_weekly_condor_agent_events.log"),
    maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
_event_log_handler.setFormatter(_event_log_fmt)
event_log = logging.getLogger("spy_weekly_condor_agent_events")
event_log.setLevel(logging.INFO)
event_log.addHandler(_event_log_handler)
event_log.propagate = False


# ── time / telegram / oversight helpers (self-contained, same pattern as
#    day_trader_agent.py -- no dependency on main.py's process) ────────────

def now_et() -> datetime:
    return datetime.now(ET)


def _parse_utc(s: str) -> datetime:
    """Parse any ISO datetime string and return a UTC-aware datetime.
    Handles naive strings (assumes UTC), '+HH:MM' offsets, and 'Z' suffix."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _load_telegram_creds() -> tuple[str, str]:
    try:
        with open(SCANNER_CFG_PATH) as f:
            cfg = json.load(f)
        return cfg.get("telegram_token", ""), cfg.get("telegram_chat_id", "")
    except Exception:
        return "", ""


def _send_telegram_sync(text: str) -> None:
    token, chat_id = _load_telegram_creds()
    if not (token and chat_id):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat_id, "text": text}, timeout=10)
    except Exception as exc:
        print(f"Telegram send failed: {exc}")


def notify(text: str, high_priority: bool = False) -> None:
    prefix = "\U0001F6A8\U0001F6A8 HIGH PRIORITY — ACTION NEEDED \U0001F6A8\U0001F6A8\n" if high_priority \
        else "\U0001F6E0️ SPY Weekly Condor agent:\n"
    threading.Thread(target=_send_telegram_sync, args=(prefix + text,), daemon=True).start()


def oversight_log(summary: str, rationale: str = "", outcome: str = "", actor: str = "trader") -> None:
    entry = {"time": now_et().astimezone(timezone.utc).isoformat(), "actor": actor,
              "category": "spy_weekly_condor_agent",
              "summary": summary, "rationale": rationale, "outcome": outcome, "pnl_impact": None}
    try:
        with open("oversight_log.jsonl", "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        print(f"Oversight log write failed: {exc}")


# ── State ────────────────────────────────────────────────────────────────

sp: dict = {
    "enabled": False,
    "config": {
        "cushion_pct":       5.0,
        "wing_cushion_pct":  7.0,
        "entry_weekday":     0,     # Monday
        "entry_start":       "09:45",
        "entry_cutoff":      "10:30",
        "exit_weekday":      4,     # Friday
        "exit_start":        "15:30",
        "exit_cutoff":       "15:55",
        "min_credit":        0.30,
        "max_positions":     1,
        "max_position_pct":  5.0,
        "max_loss_pct":      50,
    },
    "positions":    {},
    "closed_today": [],
    "decisions":    [],
    "entered_this_week": None,
}

_ib: Optional[IB] = None


def _spy_log(action: str, detail: str) -> None:
    t = now_et().strftime("%H:%M:%S ET")
    sp["decisions"].append({"time": t, "action": action, "detail": detail})
    if len(sp["decisions"]) > 200:
        sp["decisions"] = sp["decisions"][-200:]
    event_log.info("SPY-CONDOR [%s] — %s", action, detail)
    print(f"SPY-CONDOR [{action}] {detail}")


def _spy_save_state() -> None:
    try:
        with open(SPY_CONDOR_STATE_PATH, "w") as f:
            json.dump({
                "enabled":            sp["enabled"],
                "config":             sp["config"],
                "positions":          sp["positions"],
                "closed_today":       sp["closed_today"],
                "decisions":          sp["decisions"][-50:],
                "entered_this_week":  sp["entered_this_week"],
                "_save_date":         date.today().isoformat(),
            }, f, default=str)
    except Exception as e:
        print(f"SPY-CONDOR state save failed: {e}")


def _spy_load_state() -> None:
    if not os.path.exists(SPY_CONDOR_STATE_PATH):
        return
    try:
        with open(SPY_CONDOR_STATE_PATH, "r") as f:
            saved = json.load(f)
        if "config" in saved:
            sp["config"].update(saved["config"])
        sp["enabled"]     = saved.get("enabled", False)
        sp["decisions"]   = saved.get("decisions", [])
        # Restore by expiry, not by same-day entry (same fix as EVC's
        # 2026-08-26 bug) -- this strategy spans a full week (Monday entry,
        # Friday exit), so a restart mid-week must not drop it.
        today_ymd = date.today().strftime("%Y%m%d")
        sp["positions"] = {
            k: v for k, v in saved.get("positions", {}).items()
            if v.get("phase") == "open" and v.get("expiry", "00000000") >= today_ymd
        }
        this_week = date.today().isocalendar()[:2]
        saved_week = tuple(saved.get("entered_this_week")) if saved.get("entered_this_week") else None
        sp["entered_this_week"] = saved.get("entered_this_week") if saved_week == tuple(this_week) else None
        sp["closed_today"] = saved.get("closed_today", []) if saved.get("_save_date") == date.today().isoformat() else []
        print(f"SPY-CONDOR state restored: {len(sp['positions'])} open positions")
    except Exception as e:
        print(f"SPY-CONDOR load_state error: {e}")


def _is_paper() -> int:
    try:
        accts = _ib.managedAccounts() if _ib else []
        return 0 if accts and not accts[0].startswith("D") else 1
    except Exception:
        return 0


def _get_net_liq(ib) -> float:
    for av in ib.accountValues():
        if av.tag == "NetLiquidation" and av.currency in ("USD", "", "BASE"):
            try:
                v = float(av.value)
                if v > 0:
                    return v
            except (ValueError, TypeError):
                pass
    return 0.0


def _spx_safe_px(v) -> float:
    import math
    try:
        f = float(v)
        return f if (f > 0 and not math.isnan(f) and not math.isinf(f)) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _spx_mid(td) -> float:
    b = _spx_safe_px(td.bid)
    a = _spx_safe_px(td.ask)
    if b > 0 and a > 0:
        return (b + a) / 2
    return _spx_safe_px(td.last)


# ── Core strategy logic (ported verbatim from main.py's already-IBKR-native
#    version, 2026-09-07 -- only the state/ib/logging plumbing changed to
#    standalone-process shape) ──────────────────────────────────────────────

async def _spy_quote_condor(ib) -> dict:
    """Real quote for a symmetric, percentage-cushion SPY iron condor."""
    cfg = sp["config"]

    stk = IbStock("SPY", "SMART", "USD")
    await ib.qualifyContractsAsync(stk)
    td = ib.reqMktData(stk, "", False, False)
    await asyncio.sleep(2)
    spot = _spx_safe_px(td.last) or _spx_safe_px(td.close)
    ib.cancelMktData(stk)
    if not spot or spot <= 0:
        raise ValueError("no SPY spot price")

    today = date.today()
    days_to_friday = (4 - today.weekday()) % 7
    from datetime import timedelta
    target_expiry_date = today + timedelta(days=days_to_friday or 7)
    expiry = target_expiry_date.strftime("%Y%m%d")

    async def real_strikes(right):
        c = IbOpt("SPY", expiry, 0, right, "SMART", "100", "USD", tradingClass="SPY")
        details = await ib.reqContractDetailsAsync(c)
        return sorted(set(d.contract.strike for d in details))

    put_strikes  = await real_strikes("P")
    call_strikes = await real_strikes("C")
    if not put_strikes or not call_strikes:
        raise ValueError(f"no real strikes listed for SPY {expiry}")

    def nearest(strikes, target):
        return min(strikes, key=lambda s: abs(s - target))

    cushion   = cfg["cushion_pct"] / 100
    wing      = cfg["wing_cushion_pct"] / 100
    short_put  = nearest(put_strikes,  spot * (1 - cushion))
    long_put   = nearest(put_strikes,  spot * (1 - wing))
    short_call = nearest(call_strikes, spot * (1 + cushion))
    long_call  = nearest(call_strikes, spot * (1 + wing))
    if long_put >= short_put or short_call >= long_call:
        raise ValueError(f"invalid condor strikes: {long_put}/{short_put}/{short_call}/{long_call}")

    async def leg_mid(strike, right):
        c = IbOpt("SPY", expiry, strike, right, "SMART", "100", "USD")
        await ib.qualifyContractsAsync(c)
        if not c.conId:
            return None, None
        td = ib.reqMktData(c, "", False, False)
        await asyncio.sleep(3)
        mid = _spx_mid(td)
        ib.cancelMktData(c)
        return mid, c.conId

    lp_mid, lp_conid = await leg_mid(long_put,   "P")
    sp_mid, sp_conid = await leg_mid(short_put,  "P")
    sc_mid, sc_conid = await leg_mid(short_call, "C")
    lc_mid, lc_conid = await leg_mid(long_call,  "C")
    if any(v is None or v <= 0 for v in (sp_mid, sc_mid)):
        raise ValueError(f"no real quotes on short legs: sp={sp_mid} sc={sc_mid}")

    net_credit = round((sp_mid + sc_mid) - ((lp_mid or 0) + (lc_mid or 0)), 2)
    if net_credit < cfg["min_credit"]:
        raise ValueError(f"net credit {net_credit:.2f} below minimum {cfg['min_credit']:.2f}")

    return {
        "spot": spot, "expiry": expiry,
        "short_put": short_put, "long_put": long_put,
        "short_call": short_call, "long_call": long_call,
        "short_put_conid": sp_conid, "long_put_conid": lp_conid,
        "short_call_conid": sc_conid, "long_call_conid": lc_conid,
        "net_credit": net_credit,
    }


def _spy_pretrade_review(quote: dict, ibkr_net_liq: float, cfg: dict) -> dict:
    """Simpler than EVC's review -- no earnings event, so no per-ticker
    historical-move check. Two real things: (1) the account's own real
    10-year backtest result for whichever cushion this quote actually used
    (static reference, not recomputed live), and (2) the same CFO
    position-size check every strategy here uses."""
    findings: list[str] = []
    approved = True

    width = max(quote["short_put"] - quote["long_put"], quote["long_call"] - quote["short_call"])
    credit = quote["net_credit"]
    max_risk = round(width * 100 - credit * 100, 2)

    cushion = cfg["cushion_pct"]
    backtest_ref = {
        5.0: ("95.2% avg containment, but only 79.2% in 2020 (worst real year)"),
        7.0: ("98.1% avg containment, 90.6% even in 2020 (worst real year)"),
    }.get(cushion, f"no exact backtest entry for {cushion}% cushion -- nearest tested levels are 5% and 7%")
    findings.append(f"Real 10y backtest at {cushion}% cushion: {backtest_ref}")

    combined_net_liq = ibkr_net_liq or 0
    max_pct = float(cfg.get("max_position_pct", 5.0)) / 100
    pct_of_acct = (max_risk / combined_net_liq) if combined_net_liq else 1.0
    findings.append(f"CFO: max risk ${max_risk:,.0f} = {pct_of_acct:.1%} of combined net liq ${combined_net_liq:,.0f}")
    if pct_of_acct > max_pct:
        findings.append(f"EXCEEDS {max_pct:.0%} position-size ceiling (config.max_position_pct)")
        approved = False

    return {"approved": approved, "findings": findings, "max_risk": max_risk, "width": width}


async def _spy_place_condor(ib, quote: dict) -> bool:
    """Entry via IBKR sequential legs. Same long-legs-first mechanism every
    other strategy in this codebase uses."""
    from ibkr_0dte_common import ibkr_place_condor_sequential_async
    cfg = sp["config"]

    ibkr_net_liq = _get_net_liq(ib) if ib and ib.isConnected() else 0.0

    review = _spy_pretrade_review(quote, ibkr_net_liq, cfg)
    _spy_log("PRETRADE_REVIEW", f"{'APPROVED' if review['approved'] else 'REJECTED'} — " + " | ".join(review["findings"]))
    oversight_log("SPY weekly condor: " + " | ".join(review["findings"]),
                  outcome="APPROVED" if review["approved"] else "REJECTED — no orders placed",
                  actor="risk_manager" if not review["approved"] else "trader")
    if not review["approved"]:
        notify(f"SPY weekly condor pre-trade review REJECTED: {review['findings']}")
        return False

    contracts = {
        "long_put":   IbOpt("SPY", quote["expiry"], quote["long_put"],   "P", "SMART", "100", "USD"),
        "short_put":  IbOpt("SPY", quote["expiry"], quote["short_put"],  "P", "SMART", "100", "USD"),
        "short_call": IbOpt("SPY", quote["expiry"], quote["short_call"], "C", "SMART", "100", "USD"),
        "long_call":  IbOpt("SPY", quote["expiry"], quote["long_call"],  "C", "SMART", "100", "USD"),
    }
    await ib.qualifyContractsAsync(*contracts.values())

    async def _live_bid_ask(contract):
        td = ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(3)
        b, a = _spx_safe_px(td.bid), _spx_safe_px(td.ask)
        ib.cancelMktData(contract)
        return b, a

    lp_b, lp_a = await _live_bid_ask(contracts["long_put"])
    sp_b, sp_a = await _live_bid_ask(contracts["short_put"])
    sc_b, sc_a = await _live_bid_ask(contracts["short_call"])
    lc_b, lc_a = await _live_bid_ask(contracts["long_call"])
    if any(v is None or v <= 0 for v in (lp_b, lp_a, sp_b, sp_a, sc_b, sc_a, lc_b, lc_a)):
        _spy_log("NO_FILL", "stale/missing live bid-ask on one or more legs — aborting before any orders placed")
        return False
    conservative_credit = round((sp_b + sc_b) - (lp_a + lc_a), 2)
    if conservative_credit < cfg["min_credit"]:
        _spy_log("NO_FILL", f"conservative live credit {conservative_credit:.2f} below minimum {cfg['min_credit']:.2f} — aborting")
        return False

    lp_mid, lc_mid, sp_mid, sc_mid = (lp_a+lp_b)/2, (lc_a+lc_b)/2, (sp_a+sp_b)/2, (sc_a+sc_b)/2
    entry_limits = {
        "long_put":   round(lp_a - (lp_a - lp_mid) * 0.40, 2),
        "long_call":  round(lc_a - (lc_a - lc_mid) * 0.40, 2),
        "short_put":  round(sp_b + (sp_mid - sp_b) * 0.40, 2),
        "short_call": round(sc_b + (sc_mid - sc_b) * 0.40, 2),
    }

    try:
        ok, fills, fill_state = await ibkr_place_condor_sequential_async(ib, contracts, entry_limits, 1)
    except Exception as exc:
        ok, fills, fill_state = False, {}, f"unexpected_exception: {exc}"

    if not ok:
        msg = f"SPY weekly condor ENTRY INCOMPLETE (state={fill_state}, fills={fills}) — check IBKR positions manually NOW."
        _spy_log("ENTRY_INCOMPLETE", msg)
        notify(msg, high_priority=True)
        oversight_log(f"SPY condor entry incomplete: state={fill_state} fills={fills}",
                      outcome="needs manual review — possible naked leg")
        return False

    filled_credit = round((fills["short_put"] + fills["short_call"]) - (fills["long_put"] + fills["long_call"]), 2)
    pos_id = f"SPY_{date.today().strftime('%Y%m%d')}"
    pos = {
        "pos_id": pos_id, "ticker": "SPY", "date": date.today().isoformat(), "expiry": quote["expiry"],
        "spot_at_entry": quote["spot"],
        "short_put": quote["short_put"], "short_call": quote["short_call"],
        "long_put": quote["long_put"], "long_call": quote["long_call"],
        "qty": 1,
        "conids": {"long_put": quote["long_put_conid"], "short_put": quote["short_put_conid"],
                   "short_call": quote["short_call_conid"], "long_call": quote["long_call_conid"]},
        "venue": "ibkr", "net_credit": round(filled_credit, 2),
        "leg_fills": fills, "phase": "open", "live_pnl": 0.0,
        "entry_time": datetime.now(timezone.utc).isoformat(),
    }
    sp["positions"][pos_id] = pos
    sp["entered_this_week"] = list(date.today().isocalendar()[:2])
    _spy_log("ENTERED", f"condor {quote['long_put']}/{quote['short_put']}P | {quote['short_call']}/{quote['long_call']}C "
             f"credit={filled_credit:.2f} via IBKR (LP={fills['long_put']:.2f} LC={fills['long_call']:.2f} "
             f"SP={fills['short_put']:.2f} SC={fills['short_call']:.2f})")
    try:
        con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
        con.execute("""INSERT INTO trade_journal (opened_at, ticker, action, qty, entry_price, strategy_type, spot_price, is_paper)
                        VALUES (?,?,?,?,?,?,?,?)""",
                    (date.today().isoformat(), "SPY", "SELL_CONDOR", 1, filled_credit, "spy_weekly_condor", quote["spot"], _is_paper()))
        con.commit(); con.close()
    except Exception as exc:
        print(f"SPY-CONDOR journal insert failed: {exc}")
    _spy_save_state()
    return True


async def _spy_close_position(ib, pos_id: str, reason: str) -> None:
    """Close via IBKR sequential legs -- shorts first, then longs.
    Market-hours gate: options only fill 9:30-16:00 ET regardless of caller."""
    from ibkr_0dte_common import ibkr_close_condor_sequential_async
    pos = sp["positions"].get(pos_id)
    if not pos or pos["phase"] != "open":
        return

    now = now_et()
    opt_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    opt_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    if not ((opt_open <= now <= opt_close) and now.weekday() < 5):
        _spy_log("CLOSE_DEFERRED", f"close requested (reason={reason}) outside options trading hours — refusing, will retry once market reopens")
        return

    pos["phase"] = "closing"
    expiry = pos["expiry"]
    contracts = {
        "long_put":   IbOpt("SPY", expiry, pos["long_put"],   "P", "SMART", "100", "USD"),
        "long_call":  IbOpt("SPY", expiry, pos["long_call"],  "C", "SMART", "100", "USD"),
        "short_put":  IbOpt("SPY", expiry, pos["short_put"],  "P", "SMART", "100", "USD"),
        "short_call": IbOpt("SPY", expiry, pos["short_call"], "C", "SMART", "100", "USD"),
    }
    await ib.qualifyContractsAsync(*contracts.values())

    async def _live_bid_ask(contract):
        td = ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(3)
        b, a = _spx_safe_px(td.bid), _spx_safe_px(td.ask)
        ib.cancelMktData(contract)
        return b, a

    quotes = {name: await _live_bid_ask(c) for name, c in contracts.items()}
    close_limits = {}
    for name, (b, a) in quotes.items():
        mid = (a + b) / 2 if (b and a and b > 0 and a > 0) else None
        close_limits[name] = (b, a, mid)

    ok, fills = await ibkr_close_condor_sequential_async(ib, contracts, close_limits, pos["qty"])
    if not ok:
        msg = f"SPY weekly condor close INCOMPLETE (fills={fills}) — check IBKR positions manually NOW."
        _spy_log("CLOSE_INCOMPLETE", msg)
        notify(msg, high_priority=True)
        return  # leave phase="closing" -- needs a human look, not an auto-retry

    close_cost = 0.0
    for name, sign in (("long_put", -1), ("short_put", 1), ("short_call", 1), ("long_call", -1)):
        close_cost += sign * (fills.get(name) or 0.0)
    net_credit = pos["net_credit"]
    pnl        = round((net_credit - close_cost) * pos["qty"] * 100, 2)
    pnl_pct    = round(pnl / (net_credit * pos["qty"] * 100) * 100, 1) if net_credit else 0
    sp["closed_today"].append({
        "pos_id": pos_id, "date": pos["date"], "expiry": expiry,
        "short_put": pos["short_put"], "short_call": pos["short_call"],
        "net_credit": net_credit, "close_cost": round(close_cost, 2),
        "pnl": pnl, "pnl_pct": pnl_pct, "win": 1 if pnl > 0 else 0, "exit_reason": reason,
    })
    sp["positions"].pop(pos_id, None)
    _spy_log("CLOSED", f"exit={reason}  P&L=${pnl:.2f} ({pnl_pct:.1f}%) via IBKR")
    _spy_save_state()


async def _spy_enter_now_helper(ib) -> dict:
    quote = await _spy_quote_condor(ib)
    success = await _spy_place_condor(ib, quote)
    return {
        "success": success,
        "strikes": f"{quote['long_put']}/{quote['short_put']}P | {quote['short_call']}/{quote['long_call']}C",
        "credit": quote["net_credit"],
    }


async def monitor_loop(ib) -> None:
    """Monday entry window, Friday exit window, continuous stop-loss check."""
    await asyncio.sleep(15)
    while True:
        try:
            now = now_et()
            cfg = sp["config"]

            if sp["enabled"] and ib and ib.isConnected() and now.weekday() < 5:
                this_week = list(now.isocalendar()[:2])
                already_entered = sp.get("entered_this_week") == this_week

                if (now.weekday() == cfg["entry_weekday"] and not already_entered
                        and len(sp["positions"]) < cfg["max_positions"]):
                    h, m = map(int, cfg["entry_start"].split(":"))
                    cutoff_h, cutoff_m = map(int, cfg["entry_cutoff"].split(":"))
                    win_start = now.replace(hour=h, minute=m, second=0, microsecond=0)
                    win_end   = now.replace(hour=cutoff_h, minute=cutoff_m, second=0, microsecond=0)
                    if win_start <= now <= win_end:
                        try:
                            quote = await _spy_quote_condor(ib)
                            await _spy_place_condor(ib, quote)
                        except Exception as exc:
                            _spy_log("NO_FILL", f"entry attempt failed: {exc}")
                            sp["entered_this_week"] = this_week

                if now.weekday() == cfg["exit_weekday"]:
                    h, m = map(int, cfg["exit_start"].split(":"))
                    cutoff_h, cutoff_m = map(int, cfg["exit_cutoff"].split(":"))
                    win_start = now.replace(hour=h, minute=m, second=0, microsecond=0)
                    win_end   = now.replace(hour=cutoff_h, minute=cutoff_m, second=0, microsecond=0)
                    if win_start <= now <= win_end:
                        for pos_id, pos in list(sp["positions"].items()):
                            if pos["phase"] == "open":
                                await _spy_close_position(ib, pos_id, "friday_exit")

                if sp["positions"]:
                    for pos_id, pos in list(sp["positions"].items()):
                        if pos["phase"] != "open":
                            continue
                        try:
                            leg_map = [
                                (pos["short_put"],  "P", pos["conids"]["short_put"],  +1),
                                (pos["short_call"], "C", pos["conids"]["short_call"], +1),
                                (pos["long_put"],   "P", pos["conids"]["long_put"],   -1),
                                (pos["long_call"],  "C", pos["conids"]["long_call"],  -1),
                            ]
                            current_cost = 0.0
                            for strike, right, conid, sign in leg_map:
                                c = IbOpt("SPY", pos["expiry"], strike, right, "SMART", "100", "USD")
                                c.conId = conid
                                td = ib.reqMktData(c, "100,101,106", False, False)
                                await asyncio.sleep(2)
                                current_cost += sign * _spx_mid(td)
                                ib.cancelMktData(c)
                            live_pnl = round((pos["net_credit"] - current_cost) * pos["qty"] * 100, 2)
                            pos["live_pnl"] = live_pnl
                            max_loss = pos["net_credit"] * pos["qty"] * 100 * cfg["max_loss_pct"] / 100
                            entry_age = (datetime.now(timezone.utc) - _parse_utc(pos["entry_time"])).total_seconds()
                            if live_pnl < -max_loss and entry_age > 300:
                                _spy_log("STOP", f"live_pnl=${live_pnl:.2f} < max_loss=${-max_loss:.2f}")
                                await _spy_close_position(ib, pos_id, "max_loss_stop")
                        except Exception as exc:
                            print(f"SPY-CONDOR live P&L check failed for {pos_id}: {exc}")
        except Exception as exc:
            event_log.error("monitor_loop error: %s\n%s", exc, traceback.format_exc())
        await asyncio.sleep(60)


async def connection_watchdog(ib: IB) -> None:
    """Reconnect IBKR if the connection drops -- real, recurring Error 1100
    (TWS connectivity loss) was observed live during this same day's
    testing, twice within 10 minutes, before this agent even existed.
    Mirrors day_trader_agent.py's own watchdog, built after that agent hit
    the same real gap the morning after ITS extraction (TWS's routine
    daily restart silently killed its connection with nothing to recover
    it)."""
    consecutive_failures = 0
    disconnect_notified = False
    while True:
        await asyncio.sleep(20)
        if _ib is not None and not _ib.isConnected():
            if not disconnect_notified:
                print("IBKR disconnected -- attempting reconnect...")
                notify("IBKR disconnected on this agent's own connection -- auto-reconnect in progress.")
                disconnect_notified = True
            try:
                await _ib.connectAsync("127.0.0.1", IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=15)
                print("IBKR reconnected.")
                notify("IBKR reconnected.")
                disconnect_notified = False
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                print(f"Reconnect attempt failed ({consecutive_failures}): {exc}")
                if consecutive_failures == 5:
                    notify(f"IBKR reconnect has failed {consecutive_failures} times in a row — "
                           f"this agent cannot trade until connectivity is restored. Check TWS/Gateway.",
                           high_priority=True)


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _log_unhandled_exception(request: Request, exc: Exception):
    """Same real gap day_trader_agent.py fixed 2026-09-01: an unhandled
    exception here must not become a bare, untraceable 500 -- exactly what
    happened during this agent's own live test on main.py earlier today
    (a real IBKR connectivity drop hung the request until an outer 300s
    timeout fired, and the only way to find out why was digging through a
    separate crash_logs file). Log the full traceback to the durable event
    log before returning 500."""
    event_log.error("UNHANDLED EXCEPTION on %s %s: %s\n%s",
                     request.method, request.url.path, exc, traceback.format_exc())
    return JSONResponse(status_code=500,
                         content={"status": "error", "detail": f"internal error: {exc}"})


class SPYCondorConfigRequest(BaseModel):
    cushion_pct:      Optional[float] = None
    wing_cushion_pct: Optional[float] = None
    entry_weekday:    Optional[int]   = None
    entry_start:      Optional[str]   = None
    entry_cutoff:     Optional[str]   = None
    exit_weekday:     Optional[int]   = None
    exit_start:       Optional[str]   = None
    exit_cutoff:      Optional[str]   = None
    min_credit:       Optional[float] = None
    max_positions:    Optional[int]   = None
    max_position_pct: Optional[float] = None
    max_loss_pct:     Optional[float] = None


@app.get("/spy-condor/status")
def spy_condor_status():
    closed = sp.get("closed_today", [])
    return {
        "enabled":   sp["enabled"],
        "config":    sp["config"],
        "positions": sp["positions"],
        "closed_today": closed,
        "decisions": sp.get("decisions", [])[-50:],
        "entered_this_week": sp.get("entered_this_week"),
        "summary": {
            "open_positions": len(sp["positions"]),
            "closed_today":   len(closed),
            "today_pnl":      round(sum(r.get("pnl", 0) for r in closed), 2),
        },
    }


@app.post("/spy-condor/enable")
def spy_condor_enable(enabled: bool = True):
    sp["enabled"] = enabled
    _spy_log("CONFIG", f"{'enabled' if enabled else 'disabled'} by user")
    _spy_save_state()
    return {"enabled": sp["enabled"]}


@app.post("/spy-condor/config")
def spy_condor_config(req: SPYCondorConfigRequest):
    cfg = sp["config"]
    updates = req.model_dump(exclude_none=True)
    cfg.update(updates)
    _spy_log("CONFIG", f"updated: {updates}")
    _spy_save_state()
    return {"config": cfg}


@app.post("/spy-condor/close/{pos_id}")
async def spy_condor_close(pos_id: str):
    if pos_id not in sp["positions"]:
        raise HTTPException(404, f"{pos_id} not found in open positions")
    if not _ib or not _ib.isConnected():
        raise HTTPException(503, "IBKR not connected")
    _spy_log("MANUAL_CLOSE", f"manual close requested for {pos_id}")
    await _spy_close_position(_ib, pos_id, "manual_close")
    return {"status": "closing", "pos_id": pos_id}


@app.post("/spy-condor/enter-now")
async def spy_condor_enter_now():
    """Manual override: quote and (if it passes review) enter a SPY weekly
    condor right now, bypassing the Monday-only window -- for controlled
    testing, not for routine use. Real orders if approved."""
    if not _ib or not _ib.isConnected():
        raise HTTPException(503, "IBKR not connected")
    if sp["positions"]:
        raise HTTPException(409, "a SPY weekly condor position is already open")
    try:
        result = await _spy_enter_now_helper(_ib)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    return result


@app.get("/health")
def health():
    return {"ok": True, "ibkr_connected": bool(_ib and _ib.isConnected()), "enabled": sp["enabled"]}


async def main():
    global _ib
    _spy_load_state()
    _ib = IB()
    await _ib.connectAsync("127.0.0.1", IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=15)

    print(f"SPY Weekly Condor agent started. IBKR connected (clientId={IBKR_CLIENT_ID}), "
          f"serving on port {AGENT_PORT}. enabled={sp['enabled']}")
    oversight_log("SPY Weekly Condor agent started as standalone process (extracted from main.py 2026-09-07).",
                  "This strategy had never gone live (disabled, zero positions) at extraction time -- "
                  "same real motivation as Day Trader's 2026-08-27 extraction: isolate this strategy's "
                  "Monday-entry/Friday-exit cycle from any main.py crash/restart.")

    config = uvicorn.Config(app, host="127.0.0.1", port=AGENT_PORT, log_level="warning")
    server = uvicorn.Server(config)

    try:
        await asyncio.gather(server.serve(), monitor_loop(_ib), connection_watchdog(_ib))
    finally:
        if _ib.isConnected():
            _ib.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
