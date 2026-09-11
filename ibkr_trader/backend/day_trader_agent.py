"""
Standalone Day Trader agent -- full extraction from main.py (CEO instruction
2026-08-27: "may be we should make day trader independent of backend
restarts as its own agent running end to end").

Real incident that motivated this: main.py crashed and restarted 5 times
between 8:48-9:37 AM on 2026-08-27 (a Windows asyncio ProactorEventLoop
AssertionError under heavy concurrent I/O -- a low-level Python/Windows
bug, not this codebase's own logic). daytrader_scanner.py's top-ranked
candidate that morning (CRWD, score 95.5, real +10.08% earnings gap) was
submitted to /day-trader/signal at the exact moment main.py was mid-restart
and got a connection error instead of a real response -- the day's best
signal was silently lost to bad timing. Running end-to-end as its own
process means a main.py crash/restart can no longer drop an in-flight
signal or interrupt position watching/exits.

Architecture:
  - Owns its own single IBKR connection (clientId=995) for BOTH watching
    (price/volume snapshots, ATR/23-DMA/sigma entry-filter bars) AND order
    execution (entry LMT, exit TRAIL, EOD/manual MKT close) -- as of the
    2026-09-06 migration off Alpaca, specifically for this strategy. Real
    order-history evidence motivated this: with watch-on-IBKR/execute-on-
    Alpaca, entries averaged +3.47bps adverse slippage vs the IBKR-watched
    confirmation price (n=14 real trades, 73% adverse), stacked on a
    separate -4.11bps average exit-fill slippage -- both real, meaningful
    fractions of a 20-30bps trailing stop. A cro_cfo_capital_budget Alpaca
    client is still kept (see the alpaca_0dte_common import below) purely
    for the cross-strategy portfolio-wide capital view -- OTHER strategies
    (SPX 0DTE, EVC condors) still hold real capital at Alpaca regardless of
    which venue Day Trader itself trades on.
  - Persists to the SAME day_trader_state.json file main.py used to own,
    so existing tooling/dashboards that read that file keep working.
  - Exposes its own FastAPI server on port 8010 with the same endpoint
    shapes main.py used to serve (/day-trader/signal, /status, /config,
    /enable, /close/{ticker}) -- daytrader_scanner.py now posts candidates
    directly here instead of to main.py, so the time-critical entry path
    never touches main.py's process at all. main.py's own /day-trader/*
    routes become thin proxies/file-readers for frontend compatibility.
  - Runs its own monitor loop (confirmation-gate watching, phase 0/1/3
    position tracking, EOD force-close) as a background asyncio task
    alongside the FastAPI server, both driven by ONE asyncio.run(main()).
  - Own watchdog (run_day_trader_agent.ps1) restarts on crash, matching
    every other standalone script in this codebase (daytrader_scanner.py,
    ashleyklieu_trigger_executor.py, alpaca_0dte_trader.py).
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
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

# Real root cause found 2026-09-01, live-confirmed via a deliberate re-trigger:
# dt_log()'s raw print() crashed with UnicodeEncodeError on a plain sigma
# character ("σ filter: ...") the instant stdout got redirected to a file
# without explicit encoding (run_day_trader_agent.ps1's watchdog does this)
# -- Windows defaults print()'s stream to the legacy system codepage, which
# can't represent sigma, and unlike Python's logging module (which catches
# emit-time encoding errors), a raw print() failure propagates and crashes
# the whole request. This is exactly what silently killed all 20 of
# 2026-09-01's real candidate submissions with a bare HTTP 500. This
# codebase also prints/logs the multiplication sign (x) and various emoji
# elsewhere -- force real UTF-8 so none of those can trigger the same crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ib_insync import IB, Stock as IbStock, Order, LimitOrder, MarketOrder, ExecutionFilter
from pydantic import BaseModel

from macro_calendar import is_macro_day

# ── Durable event log -- survives restarts (real gap found 2026-09-01) ──────
# run_day_trader_agent.ps1's Start-Process -RedirectStandardOutput/-Error
# OVERWRITES day_trader_agent.log/.err.log on every relaunch (no append),
# so a restart done for any unrelated reason silently destroys whatever
# diagnostic evidence the previous process instance had written -- this is
# exactly what happened 2026-09-01: a real 20/20 HTTP 500 batch of
# submission failures at 09:35 ET had its cause permanently erased by three
# later same-day restarts before anyone looked. RotatingFileHandler opens
# in append mode by default (same real pattern already proven in
# daytrader_scanner.py) -- writing to a SEPARATE file from the OS-redirected
# one avoids two independent file handles fighting over the same path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_event_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                    datefmt="%Y-%m-%d %H:%M:%S")
_event_log_handler = logging.handlers.RotatingFileHandler(
    os.path.join(_SCRIPT_DIR, "day_trader_agent_events.log"),
    maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
)
_event_log_handler.setFormatter(_event_log_fmt)
event_log = logging.getLogger("day_trader_agent_events")
event_log.setLevel(logging.INFO)
event_log.addHandler(_event_log_handler)
event_log.propagate = False

sys.path.insert(0, ".")
# All of Day Trader's OWN order execution moved to native IBKR orders
# 2026-09-06 (see ib_submit_* helpers below) to cut Day Trader's real
# entry-to-fill latency: real order history showed the IBKR-watch/Alpaca-
# execute hop was adding +3.47bps average adverse slippage on entries
# (n=14 real trades, 73% adverse) on top of a separate -4.11bps average on
# exits -- both real, on a strategy trading a 20-30bps trailing stop.
# Zero Alpaca dependency as of 2026-09-07 (CEO instruction) -- the
# cro_cfo_capital_budget call below used to also pass an Alpaca client
# purely to see capital held at OTHER strategies still executing there;
# now IBKR-only (client omitted, see that function's own docstring). Real
# tradeoff: this risk view no longer sees capital any strategy still on
# Alpaca (as of this date, only EVC) has deployed -- accepted explicitly,
# not an oversight.
from alpaca_0dte_common import cro_cfo_capital_budget

ET = ZoneInfo("America/New_York")
IBKR_PORT       = int(os.environ.get("IBKR_PORT", "7496"))
IBKR_CLIENT_ID  = 995   # standalone-agent range, distinct from main.py (3), tape (20-29), ashleyklieu (994)
AGENT_PORT      = 8010
DT_STATE_PATH   = "day_trader_state.json"     # SAME file main.py used -- keeps existing readers working
JOURNAL_DB_PATH = "trade_journal.db"
TAPE_DB_PATH    = "tape_data.db"
SCANNER_CFG_PATH = "scanner_config.json"       # shared telegram creds, same file every other script reads

DEFAULT_CONFIG = {
    "position_size":        5000,
    "position_size_pct":    10.0,
    "max_positions":        10,
    "hard_stop_pct":        3.0,
    "profit_target_pct":    2.0,
    "force_close_time":     "15:45",
    "signal_freshness_min": 30,
    "limit_buffer_pct":     0.10,
    "daily_profit_target":  200.0,
    "expected_return_pct":  0.1369,
    "win_rate_est":         0.505,
    "min_composite_score":  75.0,
    "use_entry_filters":    True,
    "atr_period":           14,
    "atr_multiplier":       1.8,
    "std_dev_period":       20,
    "std_dev_threshold":    3.7,
    "use_vol_filter":       True,
    "min_atr_pct":          2.5,
    "use_confirmation_gate": True,
    "confirm_pct":           0.35,
    "confirm_window_min":    60,
    "rvol_threshold":        1.2,   # 2026-09-07: real cumulative-volume-vs-20day-avg baseline,
                                    # replaces the old self-referential median-of-this-watch's-own-
                                    # polls check -- real backtest (confirm_volume_method_test.py,
                                    # full 820-candidate dataset, same 0.35% price confirm held
                                    # fixed): 62.5%->67.4% win, +0.239%->+0.281% avg at this threshold
    "max_spread_pct":        1.0,   # 2026-09-07: real bid-ask spread at the exact confirmation
                                    # moment, from the same tick driving confirmation -- no new
                                    # data source needed. Real backtest (spread_at_confirm_test.py,
                                    # full 820-candidate dataset): excluding the widest spread
                                    # quartile (>~0.95%) improved 67.5%->70.5% win, +0.281%->
                                    # +0.303% avg, t=1.96 vs the widest-quartile-only group --
                                    # strongest statistical result of any factor tested this session
    "use_trailing_stop":     True,
    "trailing_stop_pct":     0.3,
    "max_risk_override_usd": None,  # CEO override of the shared CRO/CFO per-strategy cap for
                                     # THIS strategy only -- never affects the portfolio-wide
                                     # headroom check. Same precedent as GOOG condor's 15%
                                     # override and the 0DTE butterflies' --max-risk-override.
}

dt: dict = {
    "enabled":      False,
    "config":       dict(DEFAULT_CONFIG),
    "positions":    {},
    "watching":     {},
    "closed_today": [],
    "decisions":    [],
}

_ib: Optional[IB] = None
_watch_contracts: dict[str, IbStock] = {}  # ticker -> subscribed contract, in-memory only (not persisted) --
                                            # backs the real-time reqMktData streams that drive confirmation
                                            # detection (2026-09-07), see start_watching_stream() below

# ── time / telegram / oversight helpers (self-contained, same pattern as
#    ashleyklieu_alert_monitor.py -- no dependency on main.py's process) ──

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(s: str) -> datetime:
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


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
    prefix = "🚨🚨 HIGH PRIORITY — ACTION NEEDED 🚨🚨\n" if high_priority else "🛠️ Day Trader agent:\n"
    threading.Thread(target=_send_telegram_sync, args=(prefix + text,), daemon=True).start()


def oversight_log(summary: str, rationale: str = "", outcome: str = "") -> None:
    entry = {"time": _utcnow().isoformat(), "actor": "trader", "category": "day_trader_agent",
              "summary": summary, "rationale": rationale, "outcome": outcome, "pnl_impact": None}
    try:
        with open("oversight_log.jsonl", "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        print(f"Oversight log write failed: {exc}")


def dt_log(action: str, ticker: str, detail: str) -> None:
    entry = {"time": datetime.now(ET).strftime("%H:%M:%S ET"), "action": action,
              "ticker": ticker, "detail": detail}
    dt["decisions"].append(entry)
    dt["decisions"] = dt["decisions"][-200:]
    line = f"[{entry['time']}] {action} {ticker}: {detail}"
    print(line)
    event_log.info(line)  # durable, append-mode -- survives a restart done for any unrelated reason


# ── state persistence (same file/shape main.py used) ──────────────────────

def save_state() -> None:
    try:
        with open(DT_STATE_PATH, "w") as f:
            json.dump({
                "enabled":      dt["enabled"],
                "config":       dt["config"],
                "positions":    dt["positions"],
                "watching":     dt.get("watching", {}),
                "closed_today": dt.get("closed_today", [])[-100:],
                "decisions":    dt.get("decisions", [])[-200:],
            }, f, indent=2, default=str)
    except Exception as exc:
        print(f"Day trader state save failed: {exc}")


def load_state() -> None:
    if not os.path.exists(DT_STATE_PATH):
        return
    try:
        with open(DT_STATE_PATH) as f:
            saved = json.load(f)
        if "config" in saved:
            dt["config"].update(saved["config"])
        if "enabled" in saved:
            dt["enabled"] = saved["enabled"]
        today = date.today().isoformat()
        restored = {tk: p for tk, p in saved.get("positions", {}).items()
                    if p.get("entry_date") == today}
        for p in restored.values():
            p.pop("live_price", None)
            p.pop("live_pnl", None)
        dt["positions"]    = restored
        dt["watching"]     = saved.get("watching", {})
        dt["closed_today"] = [r for r in saved.get("closed_today", []) if r.get("exit_date") == today]
        dt["decisions"]    = saved.get("decisions", [])
        print(f"Day trader state restored: {len(restored)} open, {len(dt['closed_today'])} closed today")
    except Exception as exc:
        print(f"Day trader state load failed: {exc}")


def is_paper() -> int:
    try:
        accts = _ib.managedAccounts() if _ib else []
        return 1 if any(a.startswith("DU") for a in accts) else 0
    except Exception:
        return 0


def journal_close(ticker: str, pos: dict, exit_px: float, exit_type: str, pnl: float) -> None:
    entry_px = pos.get("entry_price", exit_px)
    pnl_pct  = round((exit_px - entry_px) / entry_px * 100, 3) if entry_px else 0.0
    record = {
        "ticker": ticker, "entry_date": pos.get("entry_date"), "exit_date": date.today().isoformat(),
        "entry_price": round(entry_px, 4), "exit_price": round(exit_px, 4),
        "shares": pos.get("shares", 0), "pnl": round(pnl, 2), "pnl_pct": pnl_pct,
        "exit_type": exit_type, "win": pnl > 0,
        "source": pos.get("source"),  # 2026-09-01: so closed_today/eod_summary can be split by pipeline
    }
    dt["closed_today"].append(record)
    dt["closed_today"] = dt["closed_today"][-100:]
    try:
        con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
        con.execute("""
            INSERT INTO trade_journal
                (opened_at, closed_at, ticker, action, qty,
                 entry_price, exit_price, pnl, pnl_pct, win,
                 exit_reason, strategy_type, score, vol_ratio, commission, is_paper, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pos.get("entry_date"), date.today().isoformat(), ticker, "BUY", pos.get("shares", 0),
            round(entry_px, 4), round(exit_px, 4), round(pnl, 2), pnl_pct, 1 if pnl > 0 else 0,
            exit_type, "DAY_BREAKOUT", pos.get("composite_score"), pos.get("vol_ratio"), 0.0, is_paper(),
            f"venue={pos.get('venue', 'ibkr')}",  # 2026-09-06: distinguishes IBKR-era trades from
                                                   # the Alpaca-era ones already in this table
        ))
        con.commit()
        con.close()
    except Exception as exc:
        print(f"Day trade journal insert failed: {exc}")
    close_tag = f"[{pos.get('source')}] " if pos.get("source") else ""
    dt_log(exit_type.upper(), ticker, f"{close_tag}exit={exit_px:.2f} pnl={'+' if pnl >= 0 else ''}{pnl:.2f} ({pnl_pct:+.2f}%)")
    notify(f"{'🟢' if pnl >= 0 else '🔴'} Day Trader: {close_tag}{ticker} closed ({exit_type}) "
           f"@ {exit_px:.2f} -- {'+' if pnl >= 0 else ''}{pnl:.2f} ({pnl_pct:+.2f}%)")
    save_state()


# ── Native IBKR execution (2026-09-06 migration off Alpaca for Day Trader
#    specifically -- see the import-block note above for why) ──────────────
# ib_insync keeps a live, locally-cached ib.trades()/ib.fills()/ib.positions()
# that update automatically from IBKR's push events on this agent's own
# persistent connection -- no per-order REST round-trip needed the way
# Alpaca's get_order_by_id polling required. reqAllOpenOrdersAsync() at
# startup (see main()) rehydrates any orders still working from before a
# restart; ib.positions() is the source of truth for whether a position is
# still open (never trust dt["positions"] alone -- same restart-safety
# discipline this account applies everywhere else).

def ib_submit_trailing_stop_sell(ib: IB, ticker: str, shares: int, trail_pct: float):
    """Standalone trail sell -- kept ONLY as an emergency fallback for the
    rare case a bracket's trail CHILD leg gets rejected even though the
    parent filled (see _handle_entry_fill's health check). The normal path
    is ib_submit_bracket_trailing_buy below, which submits both legs
    together so the position is never unprotected in between."""
    contract = IbStock(ticker, "SMART", "USD")
    order = Order(action="SELL", orderType="TRAIL", totalQuantity=shares,
                   trailingPercent=trail_pct, tif="GTC")
    return ib.placeOrder(contract, order)


def ib_submit_bracket_trailing_buy(ib: IB, ticker: str, shares: int, lmt_px: float, trail_pct: float):
    """Entry (parent LMT BUY) + exit (child TRAIL SELL) submitted together,
    atomically -- IBKR activates the trail automatically once the parent
    fills, anchoring it to the REAL fill price server-side (2026-09-07,
    closes a real gap the old submit-entry-then-detect-fill-then-submit-
    trail flow had: a filled position was briefly unprotected if the trail
    submission step was delayed or failed). Verified live: IBKR accepts a
    TRAIL order as a bracket child (parentId set, transmit=True on the
    child, False on the parent)."""
    contract = IbStock(ticker, "SMART", "USD")
    parent_id = ib.client.getReqId()
    parent = LimitOrder("BUY", shares, round(lmt_px, 2), tif="DAY")
    parent.orderId = parent_id
    parent.transmit = False
    trail_id = ib.client.getReqId()
    trail = Order(action="SELL", orderType="TRAIL", totalQuantity=shares,
                   trailingPercent=trail_pct, tif="GTC")
    trail.orderId = trail_id
    trail.parentId = parent_id
    trail.transmit = True
    parent_trade = ib.placeOrder(contract, parent)
    trail_trade = ib.placeOrder(contract, trail)
    return parent_trade, trail_trade


def ib_submit_bracket_buy(ib: IB, ticker: str, shares: int, lmt_px: float,
                           stop_px: float, profit_px: float):
    contract = IbStock(ticker, "SMART", "USD")
    bracket = ib.bracketOrder("BUY", shares, round(lmt_px, 2),
                               round(profit_px, 2), round(stop_px, 2))
    trades = [ib.placeOrder(contract, o) for o in bracket]
    return trades[0]  # parent (entry) trade -- child stop/profit legs transmit automatically


def ib_submit_market_sell(ib: IB, ticker: str, shares: int):
    contract = IbStock(ticker, "SMART", "USD")
    order = MarketOrder("SELL", shares, tif="DAY")
    return ib.placeOrder(contract, order)


def ib_find_trade(ib: IB, order_id: int):
    for t in ib.trades():
        if t.order.orderId == order_id:
            return t
    return None


def ib_real_position_qty(ib: IB, ticker: str) -> float:
    """Real, current share count at IBKR for ticker -- checked fresh each
    cycle, never assumed from local state (2026-08-07's lesson, applied here
    the same way ib.positions()/ib.trades() are used everywhere else in this
    account for restart-safety)."""
    for p in ib.positions():
        if p.contract.symbol == ticker and p.contract.secType == "STK":
            return float(p.position)
    return 0.0


async def ib_last_closing_fill(ib: IB, ticker: str):
    """Most recent real SELL execution for ticker today, or None. Tries the
    live in-memory ib.fills() first (fast, already-cached from this
    session's push events); falls back to a fresh reqExecutionsAsync (hits
    IBKR's server directly) if that's empty -- covers the case where the
    position closed while this agent was down/restarting and this session
    never saw the fill event locally."""
    sells = [f for f in ib.fills()
             if f.contract.symbol == ticker and f.execution.side == "SLD"]
    if not sells:
        today = date.today().strftime("%Y%m%d")
        try:
            fills = await ib.reqExecutionsAsync(ExecutionFilter(symbol=ticker, time=today + " 00:00:00"))
            sells = [f for f in fills if f.contract.symbol == ticker and f.execution.side == "SLD"]
        except Exception:
            sells = []
    if not sells:
        return None
    return max(sells, key=lambda f: f.execution.time)


async def fetch_entry_metrics(ib: IB, ticker: str) -> dict:
    """ATR(14), 23-DMA, today's sigma score, atr_mult -- ported verbatim from
    main.py's _fetch_entry_metrics (500-ticker/5yr sp500_daytrade_study.py
    validated this exact methodology, see min_atr_pct's history in config)."""
    contract = IbStock(ticker, "SMART", "USD")
    # Real fix 2026-09-01: this is a SEPARATE live IBKR call from the plain
    # connection check in /day-trader/signal -- a transient hiccup on just
    # THIS request (not a full disconnect) used to be swallowed as a silent
    # "skip" with zero retry, same class of problem as the connection-level
    # one, just per-ticker instead of pipeline-wide. One retry after a short
    # pause before giving up on this candidate.
    for attempt in (1, 2):
        try:
            bars = await asyncio.wait_for(
                ib.reqHistoricalDataAsync(contract, endDateTime="", durationStr="40 D",
                                            barSizeSetting="1 day", whatToShow="TRADES",
                                            useRTH=True, keepUpToDate=False),
                timeout=15)
            break
        except Exception as exc:
            # No concurrency limit on this call as of 2026-09-05 (removed after
            # a real test showed 33 concurrent reqHistoricalDataAsync calls --
            # this exact call, same params -- succeed cleanly with zero pacing
            # errors at today's realistic candidate counts). Watching for a real
            # occurrence instead of pre-guarding against a risk that isn't
            # empirically showing up: flag it loudly here if IBKR's own pacing
            # language ever appears, so it's easy to notice and re-add
            # protection (e.g. a semaphore) at that point rather than silently
            # blending into routine per-ticker fetch failures.
            if "pacing" in str(exc).lower():
                print(f"*** POSSIBLE IBKR PACING VIOLATION *** fetch_entry_metrics {ticker} "
                      f"(attempt {attempt}): {exc}")
            else:
                print(f"fetch_entry_metrics {ticker} (attempt {attempt}): {exc}")
            if attempt == 2:
                return {}
            await asyncio.sleep(5)
    if len(bars) < 15:
        return {}
    closes = [b.close for b in bars]
    highs  = [b.high for b in bars]
    lows   = [b.low for b in bars]
    trs = []
    for i in range(1, len(bars)):
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    atr14 = sum(trs[-14:]) / 14 if len(trs) >= 14 else sum(trs) / max(len(trs), 1)
    dma23 = sum(closes[-23:]) / 23 if len(closes) >= 23 else sum(closes) / len(closes)
    atr_mult = (highs[-1] - lows[-1]) / atr14 if atr14 > 0 else 0.0
    std_score = 0.0
    if len(closes) >= 2:
        daily_rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        period = min(20, len(daily_rets) - 1)
        if period >= 5:
            ret_std = float(np.std(daily_rets[-period:]))
            std_score = abs(daily_rets[-1]) / ret_std if ret_std > 0 else 0.0
    atr_pct = (atr14 / closes[-1] * 100) if closes[-1] > 0 else 0.0
    # Real RVOL baseline (2026-09-07): 20-day avg daily volume from the SAME
    # already-fetched daily bars -- no new IBKR call needed. bars[-1] is
    # today's still-forming session (mid-market-hours when this fires), so
    # excluded via bars[:-1] the same way ATR/DMA above implicitly rely on
    # complete prior sessions. Real backtest (confirm_volume_method_test.py,
    # full 820-candidate dataset, SAME 0.35% price confirm held fixed):
    # replacing the old self-referential median-of-this-watch's-own-polls
    # volume check with this real RVOL baseline improved win rate 62.5%->
    # 67.4% and avg return +0.239%->+0.281% at the 1.2x threshold, robust to
    # removing the top 3/5/10 winners from each side (z=1.73, t=1.51 --
    # real and consistent across every threshold 0.8x-2.0x tested, not
    # overwhelming at any single one).
    hist_bars = bars[:-1]
    avg_daily_vol = (sum(b.volume for b in hist_bars[-20:]) / len(hist_bars[-20:])
                      if len(hist_bars) >= 5 else None)
    return {"atr14": round(atr14, 4), "atr_pct": round(atr_pct, 3), "dma23": round(dma23, 4),
            "atr_mult": round(atr_mult, 2), "std_score": round(std_score, 2), "price": closes[-1],
            "avg_daily_vol": round(avg_daily_vol, 1) if avg_daily_vol else None}


def get_net_liq(ib: IB) -> float:
    for av in ib.accountValues():
        if av.tag == "NetLiquidation" and av.currency in ("USD", "", "BASE"):
            try:
                v = float(av.value)
                if v > 0:
                    return v
            except (ValueError, TypeError):
                pass
    return 0.0


def pretrade_review(ib: IB, ticker: str, shares: int, entry_price: float,
                     cost: float, stop_price, profit_price, cfg: dict) -> dict:
    """Real-time CRO/CFO gate -- ported verbatim from main.py's _dt_pretrade_review."""
    findings: list[str] = []
    approved = True

    if stop_price is not None and profit_price is not None:
        scenarios = [
            {"case": "stop hit", "price": stop_price, "pnl": round((stop_price - entry_price) * shares, 2)},
            {"case": "flat", "price": entry_price, "pnl": 0.0},
            {"case": "target hit", "price": profit_price, "pnl": round((profit_price - entry_price) * shares, 2)},
        ]
    else:
        trail_pct = float(cfg.get("trailing_stop_pct", 0.3))
        worst_pnl = round(-entry_price * (trail_pct / 100) * shares, 2)
        scenarios = [{"case": f"trailing stop ({trail_pct}% worst-case from entry)",
                      "price": round(entry_price * (1 - trail_pct / 100), 2), "pnl": worst_pnl}]
        findings.append(f"trailing-stop mode: no fixed target, worst-case ~${worst_pnl:.0f} "
                         f"if stopped immediately at {trail_pct}%")

    try:
        # cap_pct raised 5%->20% (CEO decision 2026-09-08, same mechanism/
        # precedent as GOOG Condor's existing 15% override) -- this account
        # is small enough that the 5% default (~$110/trade at today's net
        # liq) was leaving Day Trader's own already-validated per-trade
        # sizing (main.py cfg position_size / trailing-stop risk) far below
        # what the strategy was actually designed to size at.
        budget = cro_cfo_capital_budget(ib, cap_pct=0.20)
        override = cfg.get("max_risk_override_usd")
        effective_cap = override if override is not None else budget["per_strategy_cap"]
        findings.append(f"CFO: trade cost ${cost:,.0f} vs per-strategy cap ${effective_cap:,.0f}"
                         f"{' (CEO override)' if override is not None else ''}, "
                         f"portfolio headroom ${budget['headroom']:,.0f} of ${budget['total_budget']:,.0f} budget")
        if cost > effective_cap:
            findings.append(f"EXCEEDS per-strategy cap (${effective_cap:,.0f})")
            approved = False
        if cost > budget["headroom"]:
            findings.append(f"EXCEEDS remaining portfolio headroom (${budget['headroom']:,.0f})")
            approved = False
    except Exception as exc:
        findings.append(f"could not compute capital budget: {exc}")
        approved = False

    try:
        con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
        con.row_factory = sqlite3.Row
        rows = con.execute("""SELECT * FROM options_unusual_activity WHERE ticker = ?
                               ORDER BY scan_time DESC LIMIT 5""", (ticker.upper(),)).fetchall()
        con.close()
        if rows:
            flagged = [r for r in rows if r["is_unusual"]]
            findings.append(f"options flow: {len(flagged)}/{len(rows)} recent scans flagged unusual for {ticker}")
        else:
            findings.append(f"options flow: no recent UOA scan data for {ticker}")
    except Exception as exc:
        findings.append(f"options flow check failed: {exc}")

    if stop_price is None and not cfg.get("use_trailing_stop", True):
        findings.append("NO EXIT PLAN: neither a fixed stop nor trailing stop configured")
        approved = False
    else:
        findings.append(f"exit plan: {'trailing stop ' + str(cfg.get('trailing_stop_pct')) + '%' if stop_price is None else f'fixed stop @ {stop_price}'}")

    return {"approved": approved, "findings": findings, "scenarios": scenarios}


# ── Real-time confirmation via live IBKR ticks (2026-09-07) ─────────────────
# Previously, watching relied on a periodic snapshot (reqTickersAsync, once
# per monitor_cycle) -- confirmation could only ever be noticed on the NEXT
# scheduled poll, capping detection latency at that interval regardless of
# how fast the real market actually crossed the threshold. This replaces
# that with a persistent (non-snapshot) reqMktData subscription per watched
# ticker, feeding ib.pendingTickersEvent -- confirmation is now checked the
# instant a real tick arrives, not on a fixed schedule. Position management
# (phase 0/1/3, EOD close) stays on monitor_cycle's existing 30s cycle --
# it isn't latency-critical the same way entry detection is (exits are
# already broker-side self-managing native TRAIL orders; polling there only
# detects that one already fired, it doesn't cause the exit itself).

def start_watching_stream(ib: IB, ticker: str) -> None:
    if ticker in _watch_contracts:
        return
    try:
        contract = IbStock(ticker, "SMART", "USD")
        ib.reqMktData(contract, "", False, False)
        _watch_contracts[ticker] = contract
    except Exception as exc:
        print(f"start_watching_stream {ticker} failed: {exc}")


def stop_watching_stream(ib: IB, ticker: str) -> None:
    contract = _watch_contracts.pop(ticker, None)
    if contract is not None:
        try:
            ib.cancelMktData(contract)
        except Exception:
            pass


async def _handle_entry_fill(ticker: str, fill_px: float) -> None:
    """Idempotent: processes an entry fill exactly once, however it's
    detected -- a real-time filledEvent callback (the normal, fast path)
    OR monitor_cycle's 30s poll (kept as a backstop). Whichever fires
    first wins; the phase!=0 check makes the other a safe no-op. Never
    submits a new order -- the trail (if any) was already placed
    atomically alongside the entry, see ib_submit_bracket_trailing_buy."""
    pos = dt["positions"].get(ticker)
    if pos is None or pos.get("phase", 0) != 0:
        return
    pos["entry_price"] = fill_px
    pos["phase"] = 1
    ptag = f"[{pos.get('source')}] " if pos.get("source") else ""
    if pos.get("stop_order_id"):
        trail_pct = float(dt["config"].get("trailing_stop_pct", 0.3))
        pos["is_trailing"] = True
        pos["running_high"] = fill_px
        pos["stop_price"] = round(fill_px * (1 - trail_pct / 100), 2)
        pos["profit_price"] = None
        dt_log("FILLED", ticker,
               f"{ptag}fill={fill_px:.2f} x{pos['shares']}sh via IBKR TRAILING STOP {trail_pct}% "
               f"(already resting from entry, ord#{pos['stop_order_id']})")
        notify(f"🟢 Day Trader: {ptag}{ticker} FILLED @ {fill_px:.2f} x{pos['shares']}sh "
               f"(trailing stop {trail_pct}%)")
    else:
        dt_log("FILLED", ticker,
               f"{ptag}fill={fill_px:.2f} x{pos['shares']}sh via IBKR BRACKET "
               f"stop@{pos.get('stop_price')} target@{pos.get('profit_price')} (protected atomically at entry)")
        notify(f"🟢 Day Trader: {ptag}{ticker} FILLED @ {fill_px:.2f} x{pos['shares']}sh "
               f"(bracket stop@{pos.get('stop_price')} target@{pos.get('profit_price')})")
    save_state()


async def _handle_exit_fill(ticker: str, exit_px: float) -> None:
    """Idempotent: processes an exit fill exactly once (trail firing, EOD
    force-close, or manual close), however it's detected -- same
    real-time-event-plus-poll-backstop pattern as _handle_entry_fill."""
    pos = dt["positions"].pop(ticker, None)
    if pos is None:
        return
    pnl = round((exit_px - pos["entry_price"]) * pos["shares"], 2)
    exit_type = pos.get("pending_exit_type") or (
        "trailing_stop" if pos.get("is_trailing") else ("profit_target" if pnl >= 0 else "hard_stop"))
    journal_close(ticker, pos, exit_px, exit_type, pnl)


async def _try_confirm_and_enter(ib: IB, ticker: str, live_price: float, cum_vol: float | None,
                                  spread_pct: float | None = None) -> None:
    """Fired from a live tick (on_pending_tickers) for a ticker currently in
    dt["watching"]. Pops the ticker from watching the MOMENT confirmation is
    evaluated true (whether it goes on to a real order or gets dropped by a
    downstream gate) so a second tick arriving while this coroutine is still
    running (e.g. during the pretrade-review's real Alpaca network call)
    can't double-trigger the same candidate."""
    if not dt["enabled"]:
        return  # monitor_loop's own gate stops the periodic cycle when disabled -- this event-driven
                 # path runs independently of that loop, so it needs the same check explicitly
    cfg = dt["config"]
    watch = dt.get("watching", {}).get(ticker)
    if watch is None:
        return  # already handled by a prior trigger for this same ticker

    wtag = f"[{watch.get('source')}] " if watch.get("source") else ""
    confirm_window = int(cfg.get("confirm_window_min", 60))
    try:
        registered_at = _parse_utc(watch.get("registered_at", ""))
        age_min = (_utcnow() - registered_at).total_seconds() / 60
    except Exception:
        age_min = 0.0
    if age_min > confirm_window:
        return  # stale -- the periodic expiry sweep in monitor_cycle will remove it

    watch["live_price"] = live_price
    now_et = datetime.now(ET)
    confirm_pct = float(cfg.get("confirm_pct", 0.35))
    avg_daily_vol = watch.get("avg_daily_vol")
    if avg_daily_vol and cum_vol is not None:
        session_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        elapsed_min = max(1.0, (now_et - session_open).total_seconds() / 60.0)
        expected_by_now = avg_daily_vol * min(elapsed_min, 390.0) / 390.0
        rvol = cum_vol / expected_by_now if expected_by_now > 0 else 0.0
    else:
        rvol = 0.0

    confirm_price = watch["day_open"] * (1 + confirm_pct / 100)
    rvol_threshold = float(cfg.get("rvol_threshold", 1.2))
    max_spread_pct = float(cfg.get("max_spread_pct", 1.0))
    # Real order-book leverage (2026-09-07): require a HEALTHY real-time
    # bid-ask spread at the exact confirmation moment -- missing spread data
    # fails closed (same conservative pattern as missing avg_daily_vol/rvol
    # above), not open, since a real, known-good spread is exactly the
    # signal being required. Real backtest (spread_at_confirm_test.py, full
    # 820-candidate dataset): excluding the widest spread quartile at
    # confirmation improved win rate 67.5%->70.5%, avg return +0.281%->
    # +0.303%, t=1.96 vs the widest-quartile-only group -- the strongest
    # statistical result of any confirmation-gate factor tested this
    # session, robust to removing the top 5/10/20 winners from each side.
    spread_ok = spread_pct is not None and spread_pct <= max_spread_pct
    if not (live_price >= confirm_price and rvol >= rvol_threshold and spread_ok):
        return

    # Confirmed -- remove from watching (and stop the stream) immediately,
    # before any further await, per the race-safety note above.
    dt["watching"].pop(ticker, None)
    stop_watching_stream(ib, ticker)

    pos_size_pct = float(cfg.get("position_size_pct", 0))
    if pos_size_pct > 0:
        net_liq = get_net_liq(ib)
        position_size = (net_liq * pos_size_pct / 100) if net_liq > 0 else cfg["position_size"]
    else:
        position_size = cfg["position_size"]

    shares = max(1, int(position_size / live_price))
    buf_pct = 0.50 if (now_et.hour == 9 and now_et.minute < 45) else cfg["limit_buffer_pct"]
    lmt_px  = round(live_price * (1 + buf_pct / 100), 2)
    cost = round(shares * lmt_px, 2)

    # Real, unified Day-Trader-own-budget check (2026-09-07): replaces the
    # separate count-based capacity check (len(positions) >= max_positions)
    # and price-multiple affordability check (live_price > position_size *
    # 1.5x) -- both were proxies for the same underlying question, "would
    # this trade over-commit Day Trader's own capital slice?" A direct
    # dollar comparison answers that more accurately (two $2,000 positions
    # and ten $200 positions used to look identical to a position-COUNT
    # cap) and naturally subsumes affordability: a stock priced high enough
    # that ONE share alone blows the remaining budget fails this same
    # comparison, no separate price-ratio heuristic needed. day_trader_cap
    # mirrors exactly what max_positions always implicitly assumed (N slots
    # x target size), just expressed directly in dollars. The SEPARATE
    # account-wide CFO/CRO check below (cro_cfo_capital_budget) is left
    # untouched -- it needs real, live IBKR+Alpaca account state no local
    # check can substitute for, and other strategies (EVC, 0DTE) still hold
    # real capital at Alpaca today even though Day Trader itself no longer
    # executes there.
    day_trader_deployed = sum(p.get("cost", 0) for p in dt["positions"].values())
    day_trader_cap = cfg["max_positions"] * position_size
    if day_trader_deployed + cost > day_trader_cap:
        dt_log("WATCH_DROPPED", ticker,
               f"{wtag}confirmed but would exceed Day Trader's own budget: "
               f"${day_trader_deployed:.0f} deployed + ${cost:.0f} this trade > "
               f"${day_trader_cap:.0f} cap ({cfg['max_positions']}x${position_size:.0f})")
        save_state()
        return

    use_trailing = cfg.get("use_trailing_stop", True)
    stop_px_planned = profit_px_planned = None
    if not use_trailing:
        stop_px_planned   = round(lmt_px * (1 - cfg["hard_stop_pct"] / 100), 2)
        profit_px_planned = round(lmt_px * (1 + cfg["profit_target_pct"] / 100), 2)

    loop = asyncio.get_event_loop()
    # Run off-thread same as before -- pretrade_review's cro_cfo_capital_budget
    # call is a plain sync ib_insync call now (IBKR-only as of 2026-09-07, no
    # more Alpaca REST round trip), but keeping this off the main event loop
    # is still cheap insurance against blocking the live tick/order-status
    # socket reads happening concurrently for every other watched ticker.
    review = await loop.run_in_executor(
        None, pretrade_review, ib, ticker, shares, lmt_px, cost,
        stop_px_planned, profit_px_planned, cfg)
    review_summary = f"CRO/CFO pre-trade review for {ticker}: " + " | ".join(review["findings"])
    dt_log("PRETRADE_REVIEW", ticker,
           f"{wtag}{'APPROVED' if review['approved'] else 'REJECTED'} — {review_summary}")
    oversight_log(f"Day Trader {wtag}{ticker}: {review_summary}",
                  outcome="APPROVED" if review["approved"] else "REJECTED — no order placed")
    if not review["approved"]:
        notify(f"Day Trader pre-trade review REJECTED {wtag}{ticker}: {review_summary}")
        save_state()
        return

    trail_trade = None
    try:
        if use_trailing:
            trail_pct = float(cfg.get("trailing_stop_pct", 0.3))
            ib_trade, trail_trade = ib_submit_bracket_trailing_buy(ib, ticker, shares, lmt_px, trail_pct)
        else:
            ib_trade = ib_submit_bracket_buy(ib, ticker, shares, lmt_px,
                                              stop_px_planned, profit_px_planned)
    except Exception as exc:
        dt_log("ERROR", ticker, f"confirmed entry order failed: {exc}")
        save_state()
        return

    dt["positions"][ticker] = {
        "entry_date": date.today().isoformat(), "entry_price": live_price, "shares": shares,
        "cost": round(shares * live_price, 2), "buy_order_id": ib_trade.order.orderId,
        "stop_order_id": trail_trade.order.orderId if trail_trade is not None else None,
        "stop_price": stop_px_planned, "profit_price": profit_px_planned,
        "phase": 0, "venue": "ibkr", "alert_fired_at": _utcnow().isoformat(),
        "composite_score": watch.get("composite_score"), "vol_ratio": watch.get("vol_ratio"),
        "live_price": None, "live_pnl": None,
        "atr14": watch.get("atr14"), "atr_pct": watch.get("atr_pct"),
        "source": watch.get("source"),
    }
    pos_tag = f"[{watch.get('source')}] " if watch.get("source") else ""
    dt_log("CONFIRMED", ticker,
           f"{pos_tag}day_open={watch['day_open']:.2f} -> confirmed@{live_price:.2f} "
           f"({(live_price/watch['day_open']-1)*100:+.2f}%, RVOL={rvol:.2f}x vs {rvol_threshold}x threshold, "
           f"spread={spread_pct:.3f}% vs {max_spread_pct}% max) "
           f"— LIMIT BUY {shares}sh @ {lmt_px:.2f} via IBKR (ord#{ib_trade.order.orderId}, "
           f"{'trailing (atomic bracket)' if use_trailing else 'bracket'}) -- real-time tick, no poll wait")
    save_state()

    # Real-time fill detection (2026-09-07): fires the instant IBKR confirms
    # the fill, instead of waiting for the next 30s poll. The poll in
    # monitor_cycle stays as a backstop -- both paths funnel through the
    # same idempotent _handle_entry_fill/_handle_exit_fill, so whichever
    # detects a fill first wins and the other is a safe no-op.
    ib_trade.filledEvent += lambda trade: asyncio.create_task(
        _handle_entry_fill(ticker, round(float(trade.orderStatus.avgFillPrice), 4)))
    if trail_trade is not None:
        trail_trade.filledEvent += lambda trade: asyncio.create_task(
            _handle_exit_fill(ticker, round(float(trade.orderStatus.avgFillPrice), 4)))


async def on_pending_tickers(tickers) -> None:
    """ib.pendingTickersEvent handler -- fires with every batch of real
    ticks IBKR pushes for subscribed contracts. This is the actual
    real-time confirmation path; see the module note above."""
    if not dt.get("watching") or _ib is None:
        return
    for tk in tickers:
        sym = tk.contract.symbol
        if sym not in dt.get("watching", {}):
            continue
        live_price = None
        if tk.ask > 0 and tk.bid > 0:
            live_price = (tk.bid + tk.ask) / 2
        elif tk.close > 0:
            live_price = tk.close
        elif tk.last > 1.0:
            live_price = tk.last
        if live_price is None:
            continue
        cum_vol = float(tk.volume) if tk.volume and tk.volume > 0 else None
        # Real spread, only when both a real bid and ask are present -- same
        # condition as the primary live_price path above, so spread is never
        # computed off a close/last fallback price.
        spread_pct = None
        if tk.ask > 0 and tk.bid > 0:
            mid = (tk.bid + tk.ask) / 2
            spread_pct = (tk.ask - tk.bid) / mid * 100 if mid > 0 else None
        try:
            await _try_confirm_and_enter(_ib, sym, round(live_price, 4), cum_vol, spread_pct)
        except Exception as exc:
            print(f"_try_confirm_and_enter {sym} failed: {exc}")


# ── the monitor cycle -- ported from main.py's _day_trader_monitor_coro,
#    simplified: this script owns its OWN event loop and IBKR connection,
#    so every ib_insync/Alpaca call below is a direct native await/executor
#    call -- no _run_in_streaming_loop cross-thread bridging needed (that
#    existed only because main.py shared one IBKR connection/event loop
#    across a dozen other strategies at once). ──────────────────────────────

async def monitor_cycle(ib: IB) -> None:
    cfg = dt["config"]
    if not dt["positions"] and not dt.get("watching"):
        return

    now_et = datetime.now(ET)
    try:
        fc_h, fc_m = map(int, cfg["force_close_time"].split(":"))
        force_close_dt = now_et.replace(hour=fc_h, minute=fc_m, second=0, microsecond=0)
    except Exception:
        force_close_dt = now_et.replace(hour=15, minute=45, second=0, microsecond=0)

    ticker_snapshot: dict = {}
    phase1_tickers = [t for t, p in dt["positions"].items() if p.get("phase", 0) == 1]
    if phase1_tickers:
        try:
            contracts = [IbStock(t, "SMART", "USD") for t in phase1_tickers]
            tickers = await ib.reqTickersAsync(*contracts)
            for tk in tickers:
                sym = tk.contract.symbol
                mid = None
                if tk.ask > 0 and tk.bid > 0:
                    mid = (tk.bid + tk.ask) / 2
                elif tk.close > 0:
                    mid = tk.close
                elif tk.last > 1.0:
                    mid = tk.last
                if mid:
                    ticker_snapshot[sym] = round(mid, 4)
        except Exception as ex:
            print(f"Day trader reqTickers failed: {ex}")

    # ── Watching: expiry sweep only -- real confirmation now happens via
    #    on_pending_tickers(), fired directly off live IBKR ticks, not this
    #    periodic cycle (see the module note above _try_confirm_and_enter).
    if dt.get("watching"):
        confirm_window = int(cfg.get("confirm_window_min", 60))
        for ticker, watch in list(dt["watching"].items()):
            wtag = f"[{watch.get('source')}] " if watch.get("source") else ""
            try:
                registered_at = _parse_utc(watch.get("registered_at", ""))
                age_min = (_utcnow() - registered_at).total_seconds() / 60
            except Exception:
                age_min = 0.0
            if age_min > confirm_window:
                dt_log("WATCH_EXPIRED", ticker, f"{wtag}no confirmation within {confirm_window}min of registration — dropped")
                dt["watching"].pop(ticker, None)
                stop_watching_stream(ib, ticker)
        save_state()

    to_remove: list[str] = []

    for ticker, pos in list(dt["positions"].items()):
        ptag = f"[{pos.get('source')}] " if pos.get("source") else ""
        phase = pos.get("phase", 0)
        if ticker in ticker_snapshot:
            pos["live_price"] = ticker_snapshot[ticker]
        if pos.get("live_price") and pos.get("entry_price"):
            pos["live_pnl"] = round((pos["live_price"] - pos["entry_price"]) * pos.get("shares", 0), 2)
        if pos.get("is_trailing") and pos.get("live_price"):
            pos["running_high"] = max(pos.get("running_high", pos["live_price"]), pos["live_price"])
            trail_pct = float(cfg.get("trailing_stop_pct", 0.3))
            pos["stop_price"] = round(pos["running_high"] * (1 - trail_pct / 100), 2)

        IB_STILL_WORKING = {"PendingSubmit", "PreSubmitted", "Submitted", "ApiPending"}

        if phase == 0:
            buy_oid = pos.get("buy_order_id")
            if not buy_oid:
                continue
            trade = ib_find_trade(ib, buy_oid)
            if trade is None:
                # Not yet rehydrated (e.g. right after a restart, before
                # reqAllOpenOrdersAsync's callback lands) -- check real
                # position state directly rather than assume anything.
                real_qty = ib_real_position_qty(ib, ticker)
                if real_qty > 0:
                    fill_px = pos.get("entry_price")
                    dt_log("FILLED", ticker, f"{ptag}recovered post-restart: real IBKR position "
                                              f"{real_qty}sh found, order#{buy_oid} not in local trade cache")
                else:
                    continue  # wait for rehydration or a real fill to show up
            else:
                status = trade.orderStatus.status
                if status in IB_STILL_WORKING:
                    try:
                        alert_ts = pos.get("alert_fired_at", "")
                        at = _parse_utc(alert_ts) if alert_ts else None
                        age_min = (_utcnow() - at).total_seconds() / 60 if at is not None else 0
                    except Exception:
                        age_min = 0
                    if age_min > cfg.get("signal_freshness_min", 30):
                        try:
                            ib.cancelOrder(trade.order)
                        except Exception:
                            pass
                        dt_log("BUY_CANCELLED", ticker, f"{ptag}stale after {age_min:.0f}min — cancelled, slot freed")
                        to_remove.append(ticker)
                    continue
                if status != "Filled" or not trade.orderStatus.filled:
                    dt_log("BUY_LAPSED", ticker, f"{ptag}buy not filled (status={status}) — removing")
                    to_remove.append(ticker)
                    continue
                fill_px = round(float(trade.orderStatus.avgFillPrice), 4)

            # Poll-backstop path -- the normal path is ib_trade.filledEvent
            # (attached in _try_confirm_and_enter), which usually already
            # transitioned this position to phase 1 well before this poll
            # ever runs. _handle_entry_fill's phase!=0 check makes this a
            # safe no-op in that case.
            await _handle_entry_fill(ticker, fill_px)
            continue

        if phase == 3:
            sell_oid = pos.get("stop_order_id")
            if not sell_oid:
                continue
            trade = ib_find_trade(ib, sell_oid)
            real_qty = ib_real_position_qty(ib, ticker)
            if trade is not None and trade.orderStatus.status in IB_STILL_WORKING and real_qty > 0:
                continue
            closing = await ib_last_closing_fill(ib, ticker)
            if closing:
                exit_px = round(float(closing.execution.price), 4)
            elif trade is not None and trade.orderStatus.status == "Filled":
                exit_px = round(float(trade.orderStatus.avgFillPrice), 4)
            else:
                exit_px = pos.get("entry_price", 0)
            await _handle_exit_fill(ticker, exit_px)
            to_remove.append(ticker)
            continue

        # phase 1: active intraday position -- the IBKR TRAIL order (placed
        # atomically alongside the entry) is broker-side self-managing;
        # this poll (and trail_trade.filledEvent, the normal fast path) just
        # detect when it has fired.
        real_qty = ib_real_position_qty(ib, ticker)
        if real_qty <= 0:
            closing = await ib_last_closing_fill(ib, ticker)
            if closing:
                exit_px = round(float(closing.execution.price), 4)
            else:
                exit_px = pos.get("stop_price") or pos.get("entry_price", 0)
            await _handle_exit_fill(ticker, exit_px)
            to_remove.append(ticker)
            continue

        if now_et >= force_close_dt:
            try:
                stop_trade = ib_find_trade(ib, pos.get("stop_order_id"))
                if stop_trade is not None and stop_trade.orderStatus.status in IB_STILL_WORKING:
                    ib.cancelOrder(stop_trade.order)
                closing_trade = ib_submit_market_sell(ib, ticker, int(real_qty))
            except Exception as exc:
                dt_log("ERROR", ticker, f"{ptag}EOD force-close order failed: {exc} — CHECK IBKR MANUALLY")
                notify(f"Day Trader/IBKR: {ptag}{ticker} EOD force-close FAILED ({exc}) — check IBKR manually NOW.",
                       high_priority=True)
                continue
            pos["phase"] = 3
            pos["stop_order_id"] = closing_trade.order.orderId
            pos["pending_exit_type"] = "force_close"
            dt_log("FORCE_CLOSE", ticker, f"{ptag}EOD force-close @ {cfg['force_close_time']} ET via IBKR (ord#{closing_trade.order.orderId})")
            save_state()

    for ticker in to_remove:
        dt["positions"].pop(ticker, None)
    if to_remove:
        save_state()


async def monitor_loop(ib: IB) -> None:
    await asyncio.sleep(5)
    while True:
        await asyncio.sleep(30)
        if not dt["enabled"] or (not dt["positions"] and not dt.get("watching")):
            continue
        if not ib.isConnected():
            continue
        now_et = datetime.now(ET)
        if now_et.weekday() >= 5:
            continue
        mkt_open  = now_et.replace(hour=9, minute=25, second=0, microsecond=0)
        mkt_close = now_et.replace(hour=16, minute=15, second=0, microsecond=0)
        if not (mkt_open <= now_et <= mkt_close):
            continue
        try:
            await asyncio.wait_for(monitor_cycle(ib), timeout=25)
        except Exception as exc:
            print(f"Day trader monitor error: {exc}")


async def connection_watchdog(ib: IB) -> None:
    """Reconnect IBKR if the connection drops -- found missing 2026-08-28,
    the morning after this agent first shipped: it survived overnight fine
    (proving process independence works, the whole point of the extraction),
    but TWS's own routine daily restart (~7:30-8am ET) dropped the IBKR leg
    and nothing brought it back, since main() only ever called connectAsync
    once at startup. Every /day-trader/signal call would have 503'd for the
    rest of the day. main.py's own streaming_loop_async() has an equivalent
    reconnect-on-drop loop; this mirrors that same need for this process."""
    # Real gap found 2026-09-01: this healed itself fine (that's the whole
    # point of the loop) but never told anyone it was broken while healing --
    # only success ever got a notify(). 8/31's disconnect self-resolved in
    # under a minute and NOBODY knew any signals had been lost, because the
    # only visible-from-outside symptom was a scanner batch quietly saying
    # "0 entered" (now separately alerted on in daytrader_scanner.py). Adding
    # the missing half: say something the moment it breaks, and escalate if
    # the self-healing isn't actually healing.
    consecutive_failures = 0
    disconnect_notified = False
    while True:
        await asyncio.sleep(20)
        if _ib is not None and not _ib.isConnected():
            if not disconnect_notified:
                print("IBKR disconnected -- attempting reconnect...")
                notify("IBKR disconnected on this agent's own connection -- auto-reconnect in progress. "
                       "Any /day-trader/signal calls in the meantime will 503 and be lost.")
                disconnect_notified = True
            try:
                await _ib.connectAsync("127.0.0.1", IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=15)
                print("IBKR reconnected.")
                # A real disconnect drops IBKR's server-side reqMktData
                # subscriptions even though this is the same IB() object --
                # _watch_contracts still thinks they're live, so clear it and
                # re-subscribe every currently-watched ticker, or confirmation
                # would silently stop receiving ticks after any reconnect.
                _watch_contracts.clear()
                for ticker in list(dt.get("watching", {}).keys()):
                    start_watching_stream(_ib, ticker)
                down_for = consecutive_failures * 20
                notify(f"IBKR reconnected after a disconnect (down ~{down_for}s, likely TWS's daily restart).")
                consecutive_failures = 0
                disconnect_notified = False
            except Exception as exc:
                consecutive_failures += 1
                print(f"IBKR reconnect failed, will retry: {exc}")
                if consecutive_failures == 5:  # ~100s of failed auto-heal attempts
                    notify(f"Auto-reconnect has failed {consecutive_failures} times in a row "
                           f"(~{consecutive_failures*20}s) -- still retrying, but this isn't "
                           f"self-healing on its own. Last error: {exc}", high_priority=True)


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI()

# CORS (2026-09-02): needed so the frontend (served from a different port,
# 8001) can call this agent's own port (8010) DIRECTLY instead of only
# through main.py's /day-trader/* proxy. Without this, the enable/disable
# toggle -- while functionally independent of main.py's control logic as
# of today's decoupling work -- would still be misleading: it visibly
# lives in the same dashboard as everything else, but silently depended on
# main.py's proxy being reachable to actually take effect. Same wide-open
# policy as main.py's own CORSMiddleware, for consistency.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _log_unhandled_exception(request: Request, exc: Exception):
    """Real gap found 2026-09-01: an unhandled exception here used to become
    a bare FastAPI 500 with zero trace anywhere -- exactly what happened to
    a full 20/20-candidate scanner batch at 09:35 ET that day, and the
    underlying cause was never recoverable. Log the full traceback to the
    durable event log (survives restarts) before returning 500, so a repeat
    is actually diagnosable instead of just "-> error (HTTP 500)" again."""
    event_log.error("UNHANDLED EXCEPTION on %s %s: %s\n%s",
                     request.method, request.url.path, exc, traceback.format_exc())
    return JSONResponse(status_code=500,
                         content={"status": "error", "detail": f"internal error: {exc}"})


class StockSignalRequest(BaseModel):
    ticker:             str
    price:              float
    alert_fired_at:     Optional[str]   = None
    composite_score:    Optional[float] = None
    vol_ratio:          Optional[float] = None
    # Added 2026-09-01 for the dispersion-gated combo live test (CEO-approved,
    # $150/trade, max 2 concurrent, 20-trade/60-day review): source tags every
    # decision/position/closed-trade record so results can be split by which
    # pipeline actually produced the signal, instead of blending into Day
    # Trader's normal scanner-driven trades. skip_entry_filters lets ONLY this
    # explicitly-tagged path bypass Day Trader's own ATR/std-dev/vol filters --
    # those were tuned for the ATR-composite-score method, not this combo, and
    # stacking them would silently test something never backtested.
    source:             Optional[str]   = None
    skip_entry_filters: bool            = False


class DayConfigRequest(BaseModel):
    position_size:        Optional[float] = None
    position_size_pct:    Optional[float] = None
    max_positions:        Optional[int]   = None
    hard_stop_pct:        Optional[float] = None
    profit_target_pct:    Optional[float] = None
    force_close_time:     Optional[str]   = None
    signal_freshness_min: Optional[int]   = None
    limit_buffer_pct:     Optional[float] = None
    daily_profit_target:  Optional[float] = None
    expected_return_pct:  Optional[float] = None
    win_rate_est:         Optional[float] = None
    min_composite_score:  Optional[float] = None
    use_entry_filters:    Optional[bool]  = None
    atr_period:           Optional[int]   = None
    atr_multiplier:       Optional[float] = None
    std_dev_period:       Optional[int]   = None
    std_dev_threshold:    Optional[float] = None
    use_vol_filter:       Optional[bool]  = None
    min_atr_pct:          Optional[float] = None
    use_confirmation_gate: Optional[bool]  = None
    confirm_pct:           Optional[float] = None
    confirm_window_min:    Optional[int]   = None
    rvol_threshold:        Optional[float] = None
    max_spread_pct:        Optional[float] = None
    use_trailing_stop:     Optional[bool]  = None
    trailing_stop_pct:     Optional[float] = None
    max_risk_override_usd: Optional[float] = None


@app.post("/day-trader/signal")
async def day_trader_signal(req: StockSignalRequest):
    """Called by daytrader_scanner.py directly (bypasses main.py entirely --
    this is the fix for the 2026-08-27 CRWD near-miss)."""
    cfg = dt["config"]
    if not dt["enabled"]:
        return {"status": "skipped", "reason": "disabled"}

    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return {"status": "skipped", "reason": "weekend"}
    mkt_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    try:
        fc_h, fc_m = map(int, cfg["force_close_time"].split(":"))
        entry_cutoff = now_et.replace(hour=fc_h, minute=max(0, fc_m - 30), second=0, microsecond=0)
    except Exception:
        entry_cutoff = now_et.replace(hour=15, minute=15, second=0, microsecond=0)
    if not (mkt_open <= now_et <= entry_cutoff):
        return {"status": "skipped", "reason": "outside_hours"}

    # ── Macro calendar gate (added 2026-09-04, CEO-requested) ────────────────
    # Same real, live FOMC/NFP/CPI/PPI calendar SPX 0DTE and the 0DTE
    # butterflies now use. Blocks the whole day's signal intake at this one
    # real choke point (every candidate from daytrader_scanner.py funnels
    # through this endpoint) -- an already-open equity position still rides
    # its 0.3% trailing stop through any scheduled afternoon volatility
    # (FOMC decisions publish 2:00pm ET), which is real whipsaw risk this
    # strategy's own gates have no way to see coming.
    macro_skip, macro_reason = is_macro_day()
    if macro_skip:
        dt_log("SKIPPED", req.ticker.upper(), f"real {macro_reason} day -- macro calendar gate")
        return {"status": "skipped", "reason": "macro_calendar", "detail": macro_reason}

    if req.alert_fired_at:
        try:
            age_min = (_utcnow() - _parse_utc(req.alert_fired_at)).total_seconds() / 60
            if age_min > cfg["signal_freshness_min"]:
                return {"status": "skipped", "reason": "stale_signal", "age_min": round(age_min, 1)}
        except Exception:
            pass

    ticker = req.ticker.upper()
    # Source tag 2026-09-01: every dt_log line in this handler is prefixed so
    # the dispersion-gated combo live test's decisions/trades are visibly and
    # grep-ably distinct from Day Trader's normal scanner-driven ones, instead
    # of blending into the same untagged decision stream.
    tag = f"[{req.source}] " if req.source else ""
    if ticker in dt["positions"]:
        dt_log("SKIPPED", ticker, f"{tag}already have open position")
        return {"status": "skipped", "reason": "already_open"}
    if ticker in dt.get("watching", {}):
        dt_log("SKIPPED", ticker, f"{tag}already watching for confirmation")
        return {"status": "skipped", "reason": "already_watching"}
    if len(dt["positions"]) >= cfg["max_positions"]:
        dt_log("SKIPPED", ticker, f"{tag}at capacity ({len(dt['positions'])}/{cfg['max_positions']} positions)")
        return {"status": "skipped", "reason": "at_capacity"}

    min_score = float(cfg.get("min_composite_score", 0))
    if min_score > 0 and not req.skip_entry_filters:
        score = req.composite_score
        if score is None:
            dt_log("SKIPPED", ticker, f"{tag}no composite_score in signal — min={min_score:.0f} required")
            return {"status": "skipped", "reason": "no_score"}
        if score < min_score:
            dt_log("SKIPPED", ticker, f"{tag}score={score:.0f} < min={min_score:.0f}")
            return {"status": "skipped", "reason": "score_below_threshold", "score": score, "min": min_score}

    if not _ib or not _ib.isConnected():
        # Real fix 2026-09-01: this used to raise immediately, with zero
        # logging -- a whole scanner batch (20/20, 8/31) 503'd here in a
        # ~40s window and left no trace anywhere. connection_watchdog()
        # already retries the actual IBKR connection every 20s in the
        # background regardless of whether a request is waiting on it --
        # give that existing auto-heal a real chance to land (up to ~40s,
        # two reconnect cycles) before failing this signal outright, instead
        # of racing the exact instant a request happens to arrive mid-outage.
        # Only the first request during an outage pays this wait -- IBKR
        # either recovers within the window (subsequent requests see
        # isConnected()==True immediately) or it doesn't and every request
        # still 503s, same as before, just later and now logged either way.
        healed = False
        for _ in range(8):
            await asyncio.sleep(5)
            if _ib and _ib.isConnected():
                healed = True
                break
        if healed:
            dt_log("INFO", ticker, f"{tag}IBKR reconnected while this signal was waiting -- proceeding")
        else:
            dt_log("ERROR", ticker,
                   f"{tag}signal rejected: IBKR still not connected after waiting ~40s for auto-reconnect")
            raise HTTPException(503, "IBKR not connected")

    # skip_entry_filters 2026-09-01: Day Trader's ATR-multiple/std-dev/vol
    # filters were tuned for the ATR-composite-score selection method, not
    # for the dispersion-gated combo -- still fetch dt_metrics (dma23 is
    # used downstream regardless), just don't reject on these specific gates
    # for a request explicitly tagged as using a different methodology.
    apply_filters = not req.skip_entry_filters
    need_metrics = cfg.get("use_entry_filters", True) or cfg.get("use_vol_filter", True) or req.skip_entry_filters
    dt_metrics: dict = {}
    if need_metrics:
        try:
            dt_metrics = await asyncio.wait_for(fetch_entry_metrics(_ib, ticker), timeout=20)
        except Exception:
            dt_metrics = {}
        if not dt_metrics:
            dt_log("SKIPPED", ticker, f"{tag}entry-filter data unavailable — skipping")
            return {"status": "skipped", "reason": "filter_data_unavailable"}

    if apply_filters and cfg.get("use_entry_filters", True):
        atr_req = float(cfg.get("atr_multiplier", 1.8))
        sig_req = float(cfg.get("std_dev_threshold", 3.7))
        if dt_metrics["atr_mult"] < atr_req:
            dt_log("SKIPPED", ticker, f"{tag}ATR filter: range={dt_metrics['atr_mult']:.2f}× < {atr_req}× ATR14")
            return {"status": "skipped", "reason": "atr_filter", "atr_mult": dt_metrics["atr_mult"], "required": atr_req}
        if dt_metrics["std_score"] < sig_req:
            dt_log("SKIPPED", ticker, f"{tag}σ filter: {dt_metrics['std_score']:.2f}σ < {sig_req}σ threshold")
            return {"status": "skipped", "reason": "std_dev_filter", "std_score": dt_metrics["std_score"], "required": sig_req}

    if apply_filters and cfg.get("use_vol_filter", True):
        atr_pct_req = float(cfg.get("min_atr_pct", 2.5))
        atr_pct_val = dt_metrics.get("atr_pct", 0.0)
        if atr_pct_val < atr_pct_req:
            dt_log("SKIPPED", ticker,
                   f"{tag}DT-VOL filter: ATR%={atr_pct_val:.2f}% < {atr_pct_req}% "
                   f"(ticker too low-volatility for a reliable 0.5%+ intraday day)")
            return {"status": "skipped", "reason": "vol_filter", "atr_pct": atr_pct_val, "required": atr_pct_req}

    if dt_metrics:
        dt_log("FILTERS_PASS" if apply_filters else "FILTERS_BYPASSED", ticker,
               f"{tag}atr={dt_metrics.get('atr_mult', 0):.2f}×ATR14  σ={dt_metrics.get('std_score', 0):.2f}  "
               f"ATR%={dt_metrics.get('atr_pct', 0):.2f}%  dma23={dt_metrics.get('dma23', 0):.2f} (exit floor)")

    # Confirmation gate is this account's standing default (use_confirmation_gate=True) --
    # the old blind-entry IBKR path (use_confirmation_gate=False) is not ported here since
    # it's dead code under the current, real config (Alpaca-only execution since 2026-08-19).
    dt["watching"][ticker] = {
        "day_open": req.price, "registered_at": _utcnow().isoformat(),
        "composite_score": req.composite_score, "vol_ratio": req.vol_ratio,
        "atr14": dt_metrics.get("atr14"), "atr_pct": dt_metrics.get("atr_pct"),
        "avg_daily_vol": dt_metrics.get("avg_daily_vol"), "live_price": None,
        "source": req.source,
    }
    start_watching_stream(_ib, ticker)  # real-time tick stream -- confirmation fires off this, not a poll
    dt_log("WATCHING", ticker,
           f"{tag}day_open={req.price:.2f} -- awaiting {cfg.get('confirm_pct', 0.35)}% move + "
           f"RVOL>={cfg.get('rvol_threshold', 1.2)}x confirmation (window {cfg.get('confirm_window_min', 60)}min from open)")
    notify(f"📡 {tag}Day Trader signal: {ticker} @ {req.price:.2f} (score={req.composite_score}) "
           f"-- watching for {cfg.get('confirm_pct', 0.35)}% + RVOL>={cfg.get('rvol_threshold', 1.2)}x confirmation")
    save_state()
    return {"status": "watching", "ticker": ticker, "day_open": req.price}


@app.get("/day-trader/status")
def day_trader_status():
    cfg = dt["config"]
    open_positions = dt["positions"]
    closed = dt.get("closed_today", [])
    capital_deployed = sum(p.get("shares", 0) * p.get("entry_price", 0) for p in open_positions.values())
    closed_pnl = sum(r.get("pnl", 0) for r in closed)
    open_pnl = sum(p.get("live_pnl", 0) or 0 for p in open_positions.values() if p.get("phase", 0) == 1)
    today_pnl = closed_pnl + open_pnl
    n = len(closed)
    wins = [r for r in closed if r.get("win")]
    losses = [r for r in closed if not r.get("win")]
    gross_profit = sum(r["pnl"] for r in wins)
    gross_loss = sum(r["pnl"] for r in losses)
    avg_ret_pct = (sum(r.get("pnl_pct", 0) for r in closed) / n) if n else 0
    avg_win_pct = (sum(r.get("pnl_pct", 0) for r in wins) / len(wins)) if wins else 0
    avg_loss_pct = (sum(r.get("pnl_pct", 0) for r in losses) / len(losses)) if losses else 0
    best = max(closed, key=lambda r: r.get("pnl", 0), default=None)
    worst = min(closed, key=lambda r: r.get("pnl", 0), default=None)
    total_capital_traded = sum(r.get("entry_price", 0) * r.get("shares", 0) for r in closed) + capital_deployed
    exit_breakdown: dict = {}
    for r in closed:
        et = r.get("exit_type", "unknown")
        exit_breakdown[et] = exit_breakdown.get(et, 0) + 1

    eod_summary = {
        "total_trades": n, "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins) / n * 100, 1) if n else 0,
        "avg_return_pct": round(avg_ret_pct, 3), "avg_win_pct": round(avg_win_pct, 3),
        "avg_loss_pct": round(avg_loss_pct, 3), "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(gross_profit / abs(gross_loss), 2) if gross_loss else None,
        "best_trade": {"ticker": best["ticker"], "pnl": best["pnl"], "pnl_pct": best["pnl_pct"]} if best else None,
        "worst_trade": {"ticker": worst["ticker"], "pnl": worst["pnl"], "pnl_pct": worst["pnl_pct"]} if worst else None,
        "total_capital_traded": round(total_capital_traded, 2), "exit_breakdown": exit_breakdown,
        "profit_target_pct": cfg["profit_target_pct"], "hard_stop_pct": cfg["hard_stop_pct"],
        "use_trailing_stop": cfg.get("use_trailing_stop", True), "trailing_stop_pct": cfg.get("trailing_stop_pct"),
        "daily_profit_target": cfg["daily_profit_target"],
        "goal_achieved": today_pnl >= cfg["daily_profit_target"],
    }
    return {
        "enabled": dt["enabled"], "config": cfg, "positions": open_positions,
        "watching": dt.get("watching", {}), "closed_today": closed,
        "decisions": dt.get("decisions", [])[-50:], "eod_summary": eod_summary,
        "summary": {
            "open_positions": len(open_positions), "watching": len(dt.get("watching", {})),
            "capital_deployed": round(capital_deployed, 2), "closed_pnl": round(closed_pnl, 2),
            "open_pnl": round(open_pnl, 2), "today_pnl": round(today_pnl, 2), "today_trades": n,
            "goal_pct": round(today_pnl / cfg["daily_profit_target"] * 100, 1) if cfg["daily_profit_target"] > 0 else 0,
        },
    }


@app.post("/day-trader/enable")
def day_trader_enable(enabled: bool = True):
    dt["enabled"] = enabled
    dt_log("CONFIG", "—", f"{'enabled' if enabled else 'disabled'} by user")
    save_state()
    return {"enabled": dt["enabled"]}


@app.post("/day-trader/config")
def day_trader_config(req: DayConfigRequest):
    cfg = dt["config"]
    updates = req.model_dump(exclude_none=True)
    cfg.update(updates)
    if "profit_target_pct" in updates:
        new_pct = updates["profit_target_pct"]
        repriced = []
        for ticker, pos in dt["positions"].items():
            if pos.get("phase", 0) in (0, 1) and pos.get("entry_price"):
                old_target = pos.get("profit_price")
                pos["profit_price"] = round(pos["entry_price"] * (1 + new_pct / 100), 2)
                repriced.append(f"{ticker}: ${old_target}->${pos['profit_price']}")
        if repriced:
            dt_log("REPRICE", "—", f"profit_target→{new_pct}%: {', '.join(repriced)}")
    if "max_positions" in updates:
        dt_log("CONFIG", "—", f"max_positions→{updates['max_positions']} (currently {len(dt['positions'])} open)")
    dt_log("CONFIG", "—", f"updated: {updates}")
    save_state()
    return {"config": cfg}


@app.post("/day-trader/close/{ticker}")
async def day_trader_close(ticker: str):
    """Manually close a position immediately. Native IBKR since the
    2026-09-06 migration off Alpaca for Day Trader specifically (see the
    alpaca_0dte_common import-block note near the top of this file)."""
    ticker = ticker.upper()
    pos = dt["positions"].get(ticker)
    if not pos:
        raise HTTPException(404, f"{ticker} not in open day trader positions")
    if not _ib or not _ib.isConnected():
        raise HTTPException(503, "IBKR not connected")

    if pos.get("phase", 0) == 0:
        buy_oid = pos.get("buy_order_id")
        trade = ib_find_trade(_ib, buy_oid) if buy_oid else None
        try:
            if trade is not None:
                _ib.cancelOrder(trade.order)
        except Exception:
            pass
        dt["positions"].pop(ticker, None)
        dt_log("MANUAL_CANCEL", ticker, f"pending buy ord#{buy_oid} cancelled")
        save_state()
        return {"status": "cancelled", "ticker": ticker}

    real_qty = ib_real_position_qty(_ib, ticker)
    if real_qty <= 0:
        dt["positions"].pop(ticker, None)
        save_state()
        return {"status": "already_flat", "ticker": ticker}

    try:
        stop_trade = ib_find_trade(_ib, pos.get("stop_order_id"))
        if stop_trade is not None and stop_trade.orderStatus.status in (
                "PendingSubmit", "PreSubmitted", "Submitted", "ApiPending"):
            _ib.cancelOrder(stop_trade.order)
        closing_trade = ib_submit_market_sell(_ib, ticker, int(real_qty))
    except Exception as exc:
        raise HTTPException(500, f"IBKR close failed: {exc}")
    pos["phase"] = 3
    pos["stop_order_id"] = closing_trade.order.orderId
    pos["pending_exit_type"] = "manual_close"
    dt_log("MANUAL_CLOSE", ticker, f"MKT SELL {int(real_qty)}sh via IBKR (ord#{closing_trade.order.orderId})")
    save_state()
    return {"status": "closing", "ticker": ticker, "order_id": closing_trade.order.orderId}


@app.post("/reconcile")
async def day_trader_reconcile():
    """On-demand reconciliation of this agent's own state against real
    IBKR positions -- called by main.py's central auto-correcting engine
    (2026-09-10). External-close: a phase-1 position IBKR no longer holds
    -> process the exit exactly as monitor_loop would. Phantom-close: a
    record in closed_today that IBKR still holds -> alert (rare for a
    same-day strategy; not auto-reopened here)."""
    corrections: list = []
    alerts: list = []
    if not _ib or not _ib.isConnected():
        return {"corrections": corrections, "alerts": ["IBKR not connected"]}
    try:
        for ticker, pos in list(dt["positions"].items()):
            if pos.get("phase", 0) != 1:
                continue
            if ib_real_position_qty(_ib, ticker) <= 0:
                closing = await ib_last_closing_fill(_ib, ticker)
                exit_px = round(float(closing.execution.price), 4) if closing else (pos.get("stop_price") or pos.get("entry_price", 0))
                await _handle_exit_fill(ticker, exit_px)
                corrections.append(f"externally closed {ticker} @ {exit_px} (IBKR flat)")
                dt_log("RECONCILE", ticker, f"externally closed @ {exit_px} -- IBKR no longer holds it")
                notify(f"Day Trader reconcile: {ticker} was open in records but flat at IBKR -- closed @ {exit_px}.")
        held = {p.contract.symbol for p in _ib.positions() if p.contract.secType == "STK" and p.position != 0}
        for rec in dt.get("closed_today", []):
            tk = rec.get("ticker")
            if tk and tk in held and tk not in dt["positions"]:
                alerts.append(f"PHANTOM CLOSE: {tk} in closed_today but IBKR still holds it -- needs manual review")
        if corrections:
            save_state()
    except Exception as e:
        alerts.append(f"reconcile error: {e}")
    return {"corrections": corrections, "alerts": alerts}


@app.get("/day-trader/goal")
def day_trader_goal():
    cfg = dt["config"]
    target = cfg["daily_profit_target"]
    win_rate = cfg["win_rate_est"]
    exp_ret = cfg["expected_return_pct"]
    pos_size = cfg["position_size"]
    ev_per = pos_size * (exp_ret / 100)
    if ev_per <= 0:
        return {"error": "Invalid expected_return_pct (must be a positive avg net return per trade)"}
    req_pos = int(np.ceil(target / ev_per))
    req_capital = round(req_pos * pos_size, 2)
    today_pnl = sum(r.get("pnl", 0) for r in dt.get("closed_today", []))
    remaining = max(0.0, target - today_pnl)
    req_pos_remaining = int(np.ceil(remaining / ev_per)) if remaining > 0 else 0
    return {
        "daily_profit_target": target, "win_rate_est": win_rate, "expected_return_pct": exp_ret,
        "position_size": pos_size, "ev_per_position": round(ev_per, 2), "required_positions": req_pos,
        "required_capital": req_capital, "today_pnl": round(today_pnl, 2), "remaining_target": round(remaining, 2),
        "remaining_positions": req_pos_remaining, "current_max_positions": cfg["max_positions"],
        "positions_gap": max(0, req_pos - cfg["max_positions"]),
    }


@app.get("/health")
def health():
    return {"ok": True, "ibkr_connected": bool(_ib and _ib.isConnected()), "enabled": dt["enabled"]}


async def main():
    global _ib
    load_state()
    _ib = IB()
    await _ib.connectAsync("127.0.0.1", IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=15)
    # Rehydrate any orders still working from before a restart (same clientId
    # -> IBKR replays this client's own open orders into ib.trades()) so
    # phase 0/3 order-status checks don't have to wait for a NEW event to
    # populate the local cache -- matches this account's restart-safety rule
    # of verifying real broker state fresh rather than trusting local memory.
    try:
        await _ib.reqAllOpenOrdersAsync()
    except Exception as exc:
        print(f"reqAllOpenOrdersAsync at startup failed (non-fatal): {exc}")

    # Real-time confirmation (2026-09-07): register the tick handler, and
    # re-subscribe any watches restored from before this restart -- a fresh
    # IB() socket has no memory of the old process's reqMktData streams.
    _ib.pendingTickersEvent += on_pending_tickers
    for ticker in list(dt.get("watching", {}).keys()):
        start_watching_stream(_ib, ticker)
    if dt.get("watching"):
        print(f"Re-subscribed real-time streams for {len(dt['watching'])} restored watch(es).")

    print(f"Day Trader agent started. IBKR connected (clientId={IBKR_CLIENT_ID}), "
          f"serving on port {AGENT_PORT}. enabled={dt['enabled']}")
    oversight_log("Day Trader agent started as standalone process (extracted from main.py 2026-08-27).",
                  "main.py restarted 5x in one hour on 2026-08-27 (Windows asyncio bug), dropping CRWD's "
                  "top-ranked entry signal mid-restart -- running end-to-end isolates Day Trader from that.")

    config = uvicorn.Config(app, host="127.0.0.1", port=AGENT_PORT, log_level="warning")
    server = uvicorn.Server(config)

    try:
        await asyncio.gather(server.serve(), monitor_loop(_ib), connection_watchdog(_ib))
    finally:
        if _ib.isConnected():
            _ib.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
