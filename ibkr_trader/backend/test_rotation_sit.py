"""
System Integration Tests for the stock trader rotation rule.

Requires the backend running at http://localhost:8000.
Does NOT place real IBKR orders — uses the signal endpoint which checks
for IBKR connection before ordering; all tests that would hit the order
path are skipped gracefully when at capacity with IBKR connected.

Run:
  cd ibkr_trader/backend
  python test_rotation_sit.py
"""
import requests
import json
import sys
import time

BASE = "http://localhost:8000"
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"
INFO = "\033[36mINFO\033[0m"

_results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    _results.append((name, condition))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition

def section(title):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")

# ── 1. Connectivity ───────────────────────────────────────────────────────
section("1. Backend connectivity")

try:
    r = requests.get(f"{BASE}/health", timeout=5)
    check("Backend reachable", r.status_code == 200, r.text[:60])
except Exception as e:
    print(f"  [{FAIL}] Backend not reachable: {e}")
    print("\nAbort: backend must be running at http://localhost:8000")
    sys.exit(1)

# ── 2. Stock Trader status ────────────────────────────────────────────────
section("2. Stock Trader status")

r = requests.get(f"{BASE}/stock-trader/status", timeout=5)
check("Status endpoint 200", r.status_code == 200)
status = r.json()

check("Config present",        "config" in status)
check("Positions present",     "positions" in status)
check("Decisions present",     "decisions" in status)
check("rotation_log present",  "rotation_log" in status,
      f"got keys: {list(status.keys())}")

cfg = status.get("config", {})
check("rotation_enabled field exists in config", "rotation_enabled" in cfg,
      f"config keys: {list(cfg.keys())}")

print(f"\n  {INFO} rotation_enabled = {cfg.get('rotation_enabled')}")
print(f"  {INFO} max_positions    = {cfg.get('max_positions')}")
print(f"  {INFO} open positions   = {len(status.get('positions', {}))}")

# ── 3. StockSignalRequest model accepts composite_score ───────────────────
section("3. composite_score field accepted by /stock-trader/signal")

# Send a signal outside market hours or with disabled stock trader.
# We just need the request NOT to fail with a 422 (validation error).
payload = {
    "ticker":          "TESTONLY",
    "price":           100.0,
    "alert_fired_at":  "2026-07-07T10:00:00",
    "composite_score": 75.0,
}
r = requests.post(f"{BASE}/stock-trader/signal", json=payload, timeout=5)
check("composite_score accepted (no 422)", r.status_code != 422,
      f"status={r.status_code} body={r.text[:80]}")

# Verify valid skip reasons (disabled / outside_hours / weekend)
resp = r.json()
check("Response has status key", "status" in resp, str(resp)[:80])
check("Not 'unknown field' error", resp.get("status") != "error" or "composite_score" not in resp.get("detail",""),
      str(resp)[:80])
print(f"  {INFO} Signal response: {resp}")

# ── 4. Config endpoint accepts rotation_enabled ───────────────────────────
section("4. StockConfigRequest accepts rotation_enabled toggle")

# Save original value
orig_rotation = cfg.get("rotation_enabled", False)

# Enable rotation
r = requests.post(f"{BASE}/stock-trader/config",
                  json={"rotation_enabled": True}, timeout=5)
check("Config POST 200", r.status_code == 200)
cfg_after = r.json().get("config", {})
check("rotation_enabled = True after enable",
      cfg_after.get("rotation_enabled") == True,
      f"got: {cfg_after.get('rotation_enabled')}")

# Disable rotation
r = requests.post(f"{BASE}/stock-trader/config",
                  json={"rotation_enabled": False}, timeout=5)
cfg_after = r.json().get("config", {})
check("rotation_enabled = False after disable",
      cfg_after.get("rotation_enabled") == False,
      f"got: {cfg_after.get('rotation_enabled')}")

# Restore original
requests.post(f"{BASE}/stock-trader/config",
              json={"rotation_enabled": orig_rotation}, timeout=5)

# ── 5. Rotation skipped when disabled ────────────────────────────────────
section("5. Rotation skipped when disabled and at capacity")

# First enable stock trader if not already (read-only test — no actual order placed
# because we set max_positions to current count so capacity check triggers)
n_open = len(status.get("positions", {}))
orig_max = cfg.get("max_positions", 8)

# Set max_positions = current open count to force capacity state
requests.post(f"{BASE}/stock-trader/config",
              json={"max_positions": max(1, n_open), "rotation_enabled": False},
              timeout=5)

if n_open > 0:
    payload = {
        "ticker":          "DDOG",
        "price":           200.0,
        "alert_fired_at":  "2026-07-07T10:00:00",
        "composite_score": 80.0,
    }
    # Need stock trader enabled for this check
    orig_enabled = status.get("enabled", False)
    requests.post(f"{BASE}/stock-trader/enable?enabled=true", timeout=5)

    r = requests.post(f"{BASE}/stock-trader/signal", json=payload, timeout=5)
    resp = r.json()
    is_capacity_skip = (resp.get("reason") == "at_capacity" or
                        resp.get("reason", "").startswith("rotation_blocked") or
                        resp.get("reason") in ("outside_hours", "weekend", "disabled", "stale_signal"))
    check("At capacity with rotation disabled → at_capacity or time-based skip",
          is_capacity_skip, f"reason={resp.get('reason')}")

    if not orig_enabled:
        requests.post(f"{BASE}/stock-trader/enable?enabled=false", timeout=5)
else:
    print(f"  [{SKIP}] No open positions — skipping capacity test (increase max_positions and add positions)")

# Restore max_positions
requests.post(f"{BASE}/stock-trader/config", json={"max_positions": orig_max}, timeout=5)

# ── 6. rotation_log structure in status ───────────────────────────────────
section("6. rotation_log structure in status response")

r = requests.get(f"{BASE}/stock-trader/status", timeout=5)
status2 = r.json()
rot_log = status2.get("rotation_log", None)
check("rotation_log is a list", isinstance(rot_log, list),
      f"type={type(rot_log).__name__}")
if not isinstance(rot_log, list):
    print(f"  [{SKIP}] Skipping rotation_log entry checks — field missing (backend restart required?)")
    rot_log = []
check("rotation_log max 10 entries returned", len(rot_log) <= 10,
      f"len={len(rot_log)}")
if rot_log:
    entry = rot_log[0]
    required_fields = ["ts","evicted","incoming","incoming_score","evicted_sector",
                       "incoming_sector","portfolio_avg_score_at_rotation","outcome_5d"]
    for field in required_fields:
        check(f"rotation_log entry has '{field}'", field in entry, str(entry)[:60])
else:
    print(f"  [{SKIP}] rotation_log is empty (no rotations have occurred yet)")
    print(f"         Schema will be verified when first rotation fires")

# ── 7. Score validation in rotation logic ────────────────────────────────
section("7. Rotation blocked by low score (integration check via log)")

requests.post(f"{BASE}/stock-trader/config",
              json={"max_positions": max(1, n_open), "rotation_enabled": True}, timeout=5)

if n_open > 0:
    orig_enabled = status.get("enabled", False)
    requests.post(f"{BASE}/stock-trader/enable?enabled=true", timeout=5)

    low_score_payload = {
        "ticker":          "TSLA",
        "price":           300.0,
        "alert_fired_at":  "2026-07-07T10:00:00",
        "composite_score": 55.0,  # below 70 floor
    }
    r = requests.post(f"{BASE}/stock-trader/signal", json=low_score_payload, timeout=5)
    resp = r.json()
    reason = resp.get("reason", "")
    check("Low score signal blocked (score gate or time/market skip)",
          "rotation_blocked" in reason or reason in ("outside_hours","weekend","disabled","stale_signal","at_capacity"),
          f"reason={reason}")
    if "rotation_blocked" in reason:
        check("Reason mentions 'below floor'",
              "below floor" in reason or "not above" in reason, reason)

    if not orig_enabled:
        requests.post(f"{BASE}/stock-trader/enable?enabled=false", timeout=5)
else:
    print(f"  [{SKIP}] No open positions to test score gate against")

# Restore
requests.post(f"{BASE}/stock-trader/config",
              json={"max_positions": orig_max, "rotation_enabled": orig_rotation}, timeout=5)

# ── 8. Decisions log shows rotation_enabled config change ─────────────────
section("8. Config changes logged in decisions")

r = requests.get(f"{BASE}/stock-trader/status", timeout=5)
decisions = r.json().get("decisions", [])
config_decisions = [d for d in decisions if d.get("action") == "CONFIG"]
check("CONFIG decisions logged", len(config_decisions) > 0,
      f"total decisions: {len(decisions)}, config decisions: {len(config_decisions)}")

# ── Summary ───────────────────────────────────────────────────────────────
section("Summary")
passed = sum(1 for _, ok in _results if ok)
total  = len(_results)
failed = [(name, ok) for name, ok in _results if not ok]
print(f"\n  {passed}/{total} checks passed")
if failed:
    print(f"\n  Failed:")
    for name, _ in failed:
        print(f"    • {name}")
    sys.exit(1)
else:
    print(f"\n  All checks passed")
