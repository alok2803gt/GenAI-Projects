"""
Opportunity monitor for the residual GOOG long put (317.5P, exp 2026-08-28)
left over from the 2026-08-19 off-cycle condor's partial close (2026-08-24
-- 3 of 4 legs closed for +$63 realized; this long put's sell-to-close hit
a real Alpaca order-routing quirk -- see close_goog_condor_20260819.py's
docstring). Baseline: entry cost $0.66 ($66), currently worth ~$0.14 ($14),
GOOG spot ~$345 vs the $317.50 strike (~8% OTM).

DIFFERENT DESIGN FROM pdd_condor_babysitter.py: that one watches for RISK
(a short strike getting threatened). This one watches for OPPORTUNITY -- a
long put gains value as the underlying falls, so a real GOOG drop is
something to actively benefit from, not defend against. Alert-only, same
as every other monitor on this account (no auto-close) -- especially given
the known sell-to-close quirk on this exact leg means acting on an alert
may itself need a careful, watched attempt, not a blind unattended retry.

Zones (checked against REAL Alpaca marks + real GOOG spot):
  - QUIET (log only): GOOG > 5% above strike AND put value < $0.30
  - WATCH (normal Telegram): GOOG within 5% of strike, OR put value >= $0.30
    -- real opportunity building, worth knowing
  - ACT (high-priority Telegram): put value >= $0.50 (approaching/past its
    own $0.66 entry cost -- a real, capturable gain), OR GOOG has breached
    the $317.50 strike outright -- worth actively considering a close
  Self-expiring: exits quietly once run after the 2026-08-28 expiry, or if
  the position is no longer held (closed, or settled).

Usage: python goog_residual_put_monitor.py
"""
import json
from datetime import date, datetime, timezone

import requests
import yfinance as yf

BACKEND = "http://localhost:8000"
LOG_FILE = "oversight_log.jsonl"
TICKER = "GOOG"
SYMBOL = "GOOG260828P00317500"
STRIKE = 317.50
EXPIRY_DATE = date(2026, 8, 28)
ENTRY_COST = 0.66
WATCH_VALUE = 0.30
ACT_VALUE = 0.50
WATCH_DIST_PCT = 5.0
STATE_FILE = "goog_residual_put_monitor_state.json"


def telegram(msg: str, high_priority: bool = False):
    try:
        with open("scanner_config.json") as f:
            cfg = json.load(f)
        prefix = "🚨 " if high_priority else "💡 "
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": prefix + msg, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as e:
        print(f"Telegram send failed: {e}")


def log_line(summary: str, outcome: str):
    entry = {
        "time": datetime.now(timezone.utc).astimezone().isoformat(),
        "actor": "trader",
        "category": "goog_residual_put_monitor",
        "summary": summary,
        "rationale": "Opportunity monitor for the residual GOOG 317.5P long put -- watches for a real GOOG drop worth capturing, not a risk to defend against.",
        "outcome": outcome,
        "pnl_impact": None,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"last_zone": None}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def main():
    today = date.today()
    if today > EXPIRY_DATE:
        log_line(f"GOOG residual put expiry ({EXPIRY_DATE}) has passed -- monitoring complete.",
                 "No further checks needed. Remove this Task Scheduler job.")
        print("Past expiry -- nothing to monitor. Remove the scheduled task.")
        return

    try:
        r = requests.get(f"{BACKEND}/alpaca/positions", timeout=10)
        r.raise_for_status()
        positions = r.json().get("positions", [])
    except Exception as e:
        telegram(f"GOOG residual put monitor could not reach backend: {e}", high_priority=True)
        log_line(f"Backend unreachable: {e}", "Alerted -- could not check this cycle.")
        return

    leg = next((p for p in positions if p["symbol"] == SYMBOL), None)
    if not leg:
        log_line("GOOG residual put no longer held (closed, or already expired/settled).",
                 "No further monitoring needed -- position is gone.")
        print("Position not found -- already closed or settled.")
        return

    put_value = leg.get("current_price") or 0.0
    unrealized = leg.get("unrealized_pl", 0) or 0

    try:
        spot = yf.Ticker(TICKER).fast_info.get("lastPrice")
    except Exception as e:
        telegram(f"GOOG residual put monitor could not get spot price: {e}", high_priority=True)
        log_line(f"Could not fetch GOOG spot price: {e}", "Alerted -- could not assess this cycle.")
        return

    if not spot or spot <= 0:
        telegram("GOOG residual put monitor got an invalid spot price -- check manually.", high_priority=True)
        log_line("Invalid GOOG spot price returned.", "Alerted -- could not assess this cycle.")
        return

    dist_pct = (spot - STRIKE) / spot * 100

    if spot <= STRIKE:
        zone = "ACT"
        msg = (f"GOOG has BREACHED the $317.50 strike -- spot ${spot:.2f}. Residual put is now real ITM value "
               f"(${put_value:.2f}, vs $0.66 entry). Worth actively considering a close to capture this "
               f"(note: sell-to-close on this leg hit a real Alpaca quirk once already -- may need a careful retry).")
    elif put_value >= ACT_VALUE:
        zone = "ACT"
        msg = (f"GOOG residual put value (${put_value:.2f}) has reached/exceeded its own $0.66 entry cost -- "
               f"spot ${spot:.2f}, {dist_pct:.1f}% above strike. Real capturable gain building. Worth considering "
               f"a close (note: sell-to-close on this leg hit a real Alpaca quirk once already -- may need a careful retry).")
    elif dist_pct <= WATCH_DIST_PCT or put_value >= WATCH_VALUE:
        zone = "WATCH"
        msg = (f"GOOG residual put: spot ${spot:.2f} is {dist_pct:.1f}% above the $317.50 strike, "
               f"put value ${put_value:.2f} (entry $0.66). Real move building, not yet worth acting on.")
    else:
        zone = "QUIET"
        msg = (f"GOOG residual put quiet: spot ${spot:.2f}, {dist_pct:.1f}% above strike ($317.50), "
               f"put value ${put_value:.2f} (unrealized ${unrealized:.2f}).")

    state = load_state()
    zone_changed = state.get("last_zone") != zone

    telegram_sent = False
    if zone == "ACT":
        telegram(msg, high_priority=True)
        telegram_sent = True
    elif zone == "WATCH" and zone_changed:
        telegram(msg, high_priority=False)
        telegram_sent = True

    print(f"[{zone}] {msg}")
    log_line(msg, f"Zone: {zone}. " + ("Telegram sent." if telegram_sent else "No Telegram (quiet/unchanged)."))

    state["last_zone"] = zone
    state["last_check"] = datetime.now(timezone.utc).astimezone().isoformat()
    save_state(state)


if __name__ == "__main__":
    main()
