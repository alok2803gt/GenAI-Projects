"""
Deterministic, unattended hourly portfolio-oversight check.

This is NOT an LLM agent -- it's a plain script with a fixed set of checks,
designed to run headless via Windows Task Scheduler so the account still
gets watched even when no Claude Code session is open. It intentionally
gives up the nuanced narrative judgment an attended session brings (e.g.
recognizing "EVC re-enabled with unconfirmed source" as worth digging into)
in exchange for genuine durability and zero permission-bypass risk -- it
never places a trade, never changes config, only reads state, logs, and
alerts. Real analysis/decisions still belong in an attended session.

Checks each run:
  - /account/summary, /risk/status: violations, VIX threshold breach
  - Every known trader's /status: enabled/disabled state changes vs the
    last run (this is what actually caught the real "EVC/Day Trader
    re-enabled, unconfirmed source" incident this account already had)
  - Alpaca positions: anything expiring today (0 DTE) flagged for
    awareness, not action
  - Writes one line to oversight_log.jsonl every run (actor="trader",
    category="routine_check_unattended") so the audit trail keeps moving
    even with no session open
  - Sends a Telegram alert ONLY when something changed or looks off --
    never spams a clean sweep

State persisted in oversight_check_state.json (gitignored, runtime data)
so state-change detection works across runs/reboots.

Usage: python portfolio_oversight_check.py
"""
import json
import sys
from datetime import datetime, timezone, date

import requests

BACKEND = "http://localhost:8000"
STATE_FILE = "oversight_check_state.json"
LOG_FILE = "oversight_log.jsonl"
TIMEOUT = 10

TRADER_ENDPOINTS = {
    "day_trader": "/day-trader/status",
    "evc": "/earnings-vol-crush/status",
    "autotrader": "/autotrader/status",
    "fx_trader": "/fx-trader/status",
    "signal_trader": "/signal-trader/status",
    "manual_trader": "/manual-trader/status",
}


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get(path: str):
    try:
        r = requests.get(f"{BACKEND}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"_error": str(e)}


def telegram(msg: str, high_priority: bool = False):
    try:
        with open("scanner_config.json") as f:
            cfg = json.load(f)
        prefix = "🚨 " if high_priority else ""
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": prefix + msg, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as e:
        print(f"Telegram send failed: {e}")


def log_line(summary: str, outcome: str, rationale: str = "Unattended hourly check (Task Scheduler, no live session)."):
    entry = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(),
        "actor": "trader",
        "category": "routine_check_unattended",
        "summary": summary,
        "rationale": rationale,
        "outcome": outcome,
        "pnl_impact": None,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    if datetime.now().weekday() >= 5:  # Sat=5, Sun=6 -- weekdays-only, enforced here since the
        return                          # OS trigger repeats 24/7 for reliable hourly repetition

    state = load_state()
    findings = []

    account = get("/account/summary")
    risk = get("/risk/status")

    if "_error" in account or "_error" in risk:
        msg = f"Unattended oversight check FAILED to reach backend: account_err={account.get('_error')} risk_err={risk.get('_error')}"
        telegram(msg, high_priority=True)
        log_line(msg, outcome="Backend unreachable -- alerted, no further checks run this cycle.")
        sys.exit(1)

    net_liq = account.get("net_liquidation")
    violations = risk.get("summary", {}).get("total_violations", 0)
    critical = risk.get("summary", {}).get("critical", 0)

    if critical:
        findings.append(f"CRITICAL risk violation(s): {risk.get('violations')}")
    elif violations:
        findings.append(f"{violations} risk violation(s) (non-critical): {risk.get('violations')}")

    prev_traders = state.get("trader_enabled", {})
    cur_traders = {}
    for name, path in TRADER_ENDPOINTS.items():
        st = get(path)
        enabled = st.get("enabled")
        cur_traders[name] = enabled
        if name in prev_traders and prev_traders[name] != enabled and enabled is not None:
            findings.append(f"{name} state CHANGED: {prev_traders[name]} -> {enabled} (unconfirmed source -- verify in an attended session before trusting)")

    alpaca = get("/alpaca/positions")
    expiring_today = []
    if "positions" in alpaca:
        today_str = date.today().strftime("%y%m%d")
        for p in alpaca["positions"]:
            sym = p.get("symbol", "")
            if len(sym) > 15 and today_str in sym[:20]:
                expiring_today.append(sym)
    if expiring_today:
        findings.append(f"{len(expiring_today)} option leg(s) expire today: {', '.join(expiring_today)}")

    prev_net_liq = state.get("last_net_liq")
    net_liq_note = ""
    if prev_net_liq is not None and net_liq is not None:
        delta = net_liq - prev_net_liq
        if abs(delta) > 50:
            findings.append(f"IBKR net liq moved ${delta:+.2f} since last check (${prev_net_liq:.2f} -> ${net_liq:.2f})")

    summary = f"Unattended hourly sweep: net liq ${net_liq}, {violations} violations. " + ("; ".join(findings) if findings else "Clean, nothing new.")

    if findings:
        telegram("<b>Unattended oversight check -- findings:</b>\n" + "\n".join(f"- {f}" for f in findings) + f"\n\nNet liq: ${net_liq}", high_priority=bool(critical))
        outcome = "Findings sent to Telegram for review in the next attended session -- no action taken (this script never trades/changes config)."
    else:
        outcome = "Clean sweep, no Telegram sent."

    log_line(summary, outcome=outcome)

    state["trader_enabled"] = cur_traders
    state["last_net_liq"] = net_liq
    state["last_run"] = datetime.now(timezone.utc).astimezone().isoformat()
    save_state(state)

    print(summary)


if __name__ == "__main__":
    main()
