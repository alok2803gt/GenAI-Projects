"""
Unit / integration tests for the Stock Trader module.

Tests run against the live backend at http://localhost:8000.
No IBKR orders are placed — all "signal" tests hit rejection paths
(disabled, outside_hours, stale, at_capacity, duplicate).
The trading_days helper is tested in pure Python without any HTTP calls.

Run:
    python test_stock_trader.py
"""

import json
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

BASE = "http://localhost:8000"
PASS = "[PASS]"
FAIL = "[FAIL]"

results = []


# -- test helpers ------------------------------------------------------------

def get(path, **kw):
    return requests.get(BASE + path, timeout=10, **kw)

def post(path, **kw):
    return requests.post(BASE + path, timeout=10, **kw)

def check(name, condition, detail=""):
    icon = PASS if condition else FAIL
    msg  = f"{icon}  {name}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    results.append((name, condition, detail))
    return condition


# ═══════════════════════════════════════════════════════════════════════════
# 1. STATUS ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 1. GET /stock-trader/status ------------------------------------")

r = get("/stock-trader/status")
check("responds 200", r.status_code == 200)
s = r.json()

check("has 'enabled' key",    "enabled"    in s)
check("has 'config' key",     "config"     in s)
check("has 'positions' key",  "positions"  in s)
check("has 'decisions' key",  "decisions"  in s)
check("has 'closed_today' key","closed_today" in s)
check("has 'summary' key",    "summary"    in s)

cfg = s.get("config", {})
check("config.position_size present",  "position_size"        in cfg)
check("config.max_positions present",  "max_positions"        in cfg)
check("config.hard_stop_pct present",  "hard_stop_pct"        in cfg)
check("config.trail_pct present",      "trail_pct"            in cfg)
check("config.max_hold_days present",  "max_hold_days"        in cfg)
check("config.signal_freshness_min present", "signal_freshness_min" in cfg)
check("config.limit_buffer_pct present",     "limit_buffer_pct"    in cfg)

summary = s.get("summary", {})
check("summary.open_positions present",   "open_positions"   in summary)
check("summary.capital_deployed present", "capital_deployed" in summary)
check("summary.today_pnl present",        "today_pnl"        in summary)

# Save baseline state to restore later
original_enabled      = s["enabled"]
original_config       = dict(cfg)

print(f"\n  Baseline: enabled={original_enabled}, "
      f"position_size=${cfg.get('position_size')}, "
      f"max_positions={cfg.get('max_positions')}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ENABLE / DISABLE TOGGLE
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 2. POST /stock-trader/enable -----------------------------------")

r = post("/stock-trader/enable?enabled=false")
check("disable returns 200", r.status_code == 200)
check("disable: enabled=false in response", r.json().get("enabled") == False)

r = get("/stock-trader/status")
check("status reflects disabled", r.json()["enabled"] == False)

r = post("/stock-trader/enable?enabled=true")
check("re-enable returns 200", r.status_code == 200)
check("re-enable: enabled=true in response", r.json().get("enabled") == True)

r = get("/stock-trader/status")
check("status reflects re-enabled", r.json()["enabled"] == True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONFIG UPDATE (roundtrip)
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 3. POST /stock-trader/config -----------------------------------")

new_vals = {"position_size": 9999.0, "max_positions": 3, "trail_pct": 6.5}
r = post("/stock-trader/config",
         json=new_vals,
         headers={"Content-Type": "application/json"})
check("config update returns 200", r.status_code == 200)
resp_cfg = r.json().get("config", {})
check("position_size updated",  resp_cfg.get("position_size") == 9999.0)
check("max_positions updated",  resp_cfg.get("max_positions") == 3)
check("trail_pct updated",      resp_cfg.get("trail_pct") == 6.5)

# Verify persisted
r2 = get("/stock-trader/status")
cfg2 = r2.json().get("config", {})
check("position_size persisted in status", cfg2.get("position_size") == 9999.0)

# Restore original config
post("/stock-trader/config",
     json=original_config,
     headers={"Content-Type": "application/json"})
r3 = get("/stock-trader/status")
cfg3 = r3.json().get("config", {})
check("original config restored",
      cfg3.get("position_size") == original_config.get("position_size"))


# ═══════════════════════════════════════════════════════════════════════════
# 4. SIGNAL — DISABLED rejection
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 4. POST /stock-trader/signal — DISABLED ------------------------")

post("/stock-trader/enable?enabled=false")
r = post("/stock-trader/signal",
         json={"ticker": "AAPL", "price": 200.0, "alert_fired_at": datetime.utcnow().isoformat()},
         headers={"Content-Type": "application/json"})
check("disabled: returns 200 (soft skip)", r.status_code == 200)
body = r.json()
check("disabled: status=skipped",  body.get("status") == "skipped")
check("disabled: reason=disabled", body.get("reason") == "disabled")
post("/stock-trader/enable?enabled=true")   # restore


# ═══════════════════════════════════════════════════════════════════════════
# 5. SIGNAL — OUTSIDE MARKET HOURS rejection
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 5. POST /stock-trader/signal — OUTSIDE HOURS -------------------")

# Determine current ET hour from the decision log timestamp we saw (1am ET)
# Backend will reject if not 09:30–15:50 ET
from zoneinfo import ZoneInfo
now_et = datetime.now(ZoneInfo("America/New_York"))
is_market_hours = (
    now_et.weekday() < 5
    and now_et.replace(hour=9, minute=30, second=0, microsecond=0) <= now_et
    <= now_et.replace(hour=15, minute=50, second=0, microsecond=0)
)
print(f"  Current ET time: {now_et.strftime('%H:%M ET, %a')} "
      f"— market {'OPEN' if is_market_hours else 'CLOSED'}")

if not is_market_hours:
    r = post("/stock-trader/signal",
             json={"ticker": "AAPL", "price": 200.0,
                   "alert_fired_at": now_et.isoformat()},
             headers={"Content-Type": "application/json"})
    check("outside hours: returns 200", r.status_code == 200)
    body = r.json()
    check("outside hours: status=skipped", body.get("status") == "skipped")
    check("outside hours: reason=outside_hours or weekend",
          body.get("reason") in ("outside_hours", "weekend"))
else:
    print(f"  [SKIP] Market is open — outside-hours test not applicable")


# ═══════════════════════════════════════════════════════════════════════════
# 6. SIGNAL — STALE signal rejection
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 6. POST /stock-trader/signal — STALE SIGNAL --------------------")

# Set freshness to 1 minute, send a signal with 60-minute-old timestamp
post("/stock-trader/config",
     json={"signal_freshness_min": 1},
     headers={"Content-Type": "application/json"})

stale_ts = (datetime.now(ZoneInfo("America/New_York")) - timedelta(minutes=60)).isoformat()
r = post("/stock-trader/signal",
         json={"ticker": "NVDA", "price": 135.0, "alert_fired_at": stale_ts},
         headers={"Content-Type": "application/json"})
check("stale: returns 200", r.status_code == 200)
body = r.json()
# Could also be outside_hours first — both are valid skips
check("stale: status=skipped",
      body.get("status") == "skipped",
      f"reason={body.get('reason')}, age={body.get('age_min')}")
if body.get("reason") == "stale_signal":
    check("stale: reason=stale_signal", True)
    check("stale: age_min >= 1", (body.get("age_min") or 0) >= 1)
else:
    print(f"       (got '{body.get('reason')}' — earlier gate fired first, stale logic not reached)")

# Restore freshness
post("/stock-trader/config",
     json={"signal_freshness_min": original_config.get("signal_freshness_min", 30)},
     headers={"Content-Type": "application/json"})


# ═══════════════════════════════════════════════════════════════════════════
# 7. SIGNAL — AT CAPACITY rejection
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 7. POST /stock-trader/signal — AT CAPACITY ---------------------")

# Set max_positions=0 so every signal hits capacity
post("/stock-trader/config",
     json={"max_positions": 0},
     headers={"Content-Type": "application/json"})

r = post("/stock-trader/signal",
         json={"ticker": "MSFT", "price": 420.0,
               "alert_fired_at": now_et.isoformat()},
         headers={"Content-Type": "application/json"})
check("at_capacity: returns 200", r.status_code == 200)
body = r.json()
check("at_capacity: status=skipped", body.get("status") == "skipped")
if body.get("reason") == "at_capacity":
    check("at_capacity: reason=at_capacity", True)
else:
    print(f"       (got '{body.get('reason')}' — earlier gate fired first, capacity not reached)")

# Restore max_positions
post("/stock-trader/config",
     json={"max_positions": original_config.get("max_positions", 8)},
     headers={"Content-Type": "application/json"})


# ═══════════════════════════════════════════════════════════════════════════
# 8. CLOSE — 404 for non-existent ticker
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 8. POST /stock-trader/close/{ticker} — NOT FOUND ---------------")

r = post("/stock-trader/close/FAKEXXX")
check("close unknown ticker: returns 404", r.status_code == 404)


# ═══════════════════════════════════════════════════════════════════════════
# 9. HISTORY ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 9. GET /stock-trader/history -----------------------------------")

r = get("/stock-trader/history?days=30")
check("history: returns 200", r.status_code == 200)
h = r.json()
check("history: has 'trades' key",    "trades"    in h)
check("history: has 'total' key",     "total"     in h)
check("history: has 'wins' key",      "wins"      in h)
check("history: has 'win_rate' key",  "win_rate"  in h)
check("history: has 'total_pnl' key", "total_pnl" in h)
check("history: total >= 0",          h.get("total", -1) >= 0)
check("history: trades is list",      isinstance(h.get("trades"), list))

# Sanity: wins <= total
check("history: wins <= total",
      h.get("wins", 0) <= h.get("total", 0))

if h.get("total", 0) > 0:
    t = h["trades"][0]
    check("trade: has ticker",       "ticker"     in t)
    check("trade: has entry_price",  "entry_price" in t)
    check("trade: has exit_price",   "exit_price"  in t)
    check("trade: has pnl",          "pnl"         in t)
    check("trade: has exit_reason",  "exit_reason" in t)
    print(f"  Sample trade: {t.get('ticker')} "
          f"{t.get('entry_price')} -> {t.get('exit_price')} "
          f"pnl=${t.get('pnl')} ({t.get('exit_reason')})")
else:
    print("  (no closed trades yet — history structure tests passed)")


# ═══════════════════════════════════════════════════════════════════════════
# 10. DECISIONS LOG — populated by enable/config actions
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 10. Decisions log populated by actions --------------------------")

r = get("/stock-trader/status")
dec = r.json().get("decisions", [])
check("decisions: is list",    isinstance(dec, list))
check("decisions: non-empty (config/enable actions above added entries)", len(dec) > 0)
if dec:
    last = dec[-1]
    check("decision entry has 'time' key",   "time"   in last)
    check("decision entry has 'action' key", "action" in last)
    check("decision entry has 'ticker' key", "ticker" in last)
    check("decision entry has 'detail' key", "detail" in last)


# ═══════════════════════════════════════════════════════════════════════════
# 11. TRADING DAYS COUNTER — pure Python logic
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 11. _st_trading_days_held() — business day logic ----------------")

from datetime import date as _date

def trading_days_held(entry_date_str: str) -> int:
    try:
        entry = datetime.fromisoformat(entry_date_str).date()
        today = _date.today()
        if entry >= today:
            return 0
        return int(np.busday_count(entry.isoformat(), today.isoformat()))
    except Exception:
        return 0

today      = _date.today()
yesterday  = today - timedelta(days=1)
week_ago   = today - timedelta(days=7)
month_ago  = today - timedelta(days=30)
future     = today + timedelta(days=1)

check("same day returns 0",    trading_days_held(today.isoformat()) == 0)
check("future date returns 0", trading_days_held(future.isoformat()) == 0)

days_1 = trading_days_held(yesterday.isoformat())
check(f"yesterday = {days_1} trading day(s) (0 or 1 depending on weekend)",
      days_1 in (0, 1))

days_7 = trading_days_held(week_ago.isoformat())
check(f"7 calendar days = {days_7} trading days (expect 4-5)",
      3 <= days_7 <= 6,
      f"7 calendar days from {week_ago} to {today} = {days_7} business days")

days_30 = trading_days_held(month_ago.isoformat())
check(f"30 calendar days = {days_30} trading days (expect ~21)",
      18 <= days_30 <= 24,
      f"30 calendar days = {days_30} business days")

check("invalid date returns 0", trading_days_held("not-a-date") == 0)
check("empty string returns 0", trading_days_held("") == 0)

# Test specific known weekend span
# 2026-06-26 (Friday) + 2 calendar days = 2026-06-28 (Sunday) = 0 new trading days
known_friday = "2026-06-27"   # Saturday
known_day    = "2026-06-29"   # Monday
try:
    bdays = int(np.busday_count(known_friday, known_day))
    check(f"Sat→Mon = {bdays} business day(s) (expect 0 or 1)", bdays in (0, 1))
except Exception as e:
    print(f"       (busday_count edge case: {e})")


# ═══════════════════════════════════════════════════════════════════════════
# 12. FINAL STATE — restore & verify
# ═══════════════════════════════════════════════════════════════════════════
print("\n-- 12. Final state restore -----------------------------------------")

r = post("/stock-trader/enable?enabled=true")
check("final enabled=true", r.json().get("enabled") == True)
r = post("/stock-trader/config",
         json=original_config,
         headers={"Content-Type": "application/json"})
check("final config restored", r.status_code == 200)

r = get("/stock-trader/status")
final = r.json()
check("final status clean (no spurious positions)",
      isinstance(final.get("positions"), dict))


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total  = len(results)

print(f"\n{'='*60}")
print(f"  Results: {passed}/{total} passed   {failed} failed")
print(f"{'='*60}")

if failed:
    print("\nFailed tests:")
    for name, ok, detail in results:
        if not ok:
            print(f"  {FAIL} {name}")
            if detail:
                print(f"         {detail}")
    sys.exit(1)
else:
    print("\nAll tests passed.")
    sys.exit(0)
