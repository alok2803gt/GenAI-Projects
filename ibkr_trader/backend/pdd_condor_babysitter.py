"""
Dedicated babysitter for the real PDD earnings condor entered Fri 2026-08-21
15:44:15 ET (EVC, via pre-trade review): short 82P/long 78P, short 95C/long
98C, exp 2026-08-28, net credit $0.97 (qty 1). CEO asked for active
monitoring through expiry to confirm the trade stays safe.

Same "escalation-only" design as the CSCO/COHR babysitter this account ran
2026-08-14 (cron f6a6c589) -- quiet in oversight_log.jsonl on every clean
check, Telegram only when something real changes. Rebuilt as a Task
Scheduler job instead of a session-scoped cron this time: every standing
in-session cron on this account was found silently missing tonight
(2026-08-23/24, root-caused to session-scoped CronCreate jobs not
surviving), and a 4-day monitoring window (today through Fri 8/28 expiry)
is exactly the kind of thing that can't be allowed to silently stop.

Thresholds (checked against REAL Alpaca marks, not modeled):
  - GREEN (log only): spot > 5% away from both short strikes (82P, 95C)
  - WARNING (normal Telegram): spot within 5% of either short strike
  - HIGH (urgent Telegram): spot has breached a short strike (real ITM risk)
  - Self-expiring: exits quietly (still logs once) once run after the
    2026-08-28 expiry date, or if the position is no longer held (closed
    early, or expired) -- no manual cleanup required, though removing the
    Task Scheduler job itself after expiry is still recommended hygiene.

Usage: python pdd_condor_babysitter.py
"""
import json
from datetime import date, datetime, timezone

import requests
import yfinance as yf

BACKEND = "http://localhost:8000"
LOG_FILE = "oversight_log.jsonl"
TICKER = "PDD"
EXPIRY_DATE = date(2026, 8, 28)
SHORT_PUT, LONG_PUT = 82.0, 78.0
SHORT_CALL, LONG_CALL = 95.0, 98.0
ENTRY_CREDIT = 0.97
WARN_BUFFER_PCT = 5.0
STATE_FILE = "pdd_condor_babysitter_state.json"


def telegram(msg: str, high_priority: bool = False):
    try:
        with open("scanner_config.json") as f:
            cfg = json.load(f)
        prefix = "🚨 " if high_priority else "⚠️ "
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
        "category": "pdd_condor_babysitter",
        "summary": summary,
        "rationale": "Dedicated active monitoring, CEO-requested, for the real PDD condor through its 2026-08-28 expiry.",
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
        log_line(f"PDD condor expiry ({EXPIRY_DATE}) has passed -- monitoring complete.",
                 "No further checks needed. Remove this Task Scheduler job.")
        print("Past expiry -- nothing to monitor. Remove the scheduled task.")
        return

    try:
        r = requests.get(f"{BACKEND}/alpaca/positions", timeout=10)
        r.raise_for_status()
        positions = r.json().get("positions", [])
    except Exception as e:
        telegram(f"PDD babysitter could not reach backend: {e}", high_priority=True)
        log_line(f"Backend unreachable: {e}", "Alerted -- could not check the position this cycle.")
        return

    pdd_legs = [p for p in positions if p["symbol"].startswith(TICKER) and "2608" in p["symbol"]]
    if not pdd_legs:
        log_line("PDD condor no longer held (closed early or already expired/settled).",
                 "No further monitoring needed -- position is gone.")
        print("No PDD position found -- already closed or settled.")
        return

    total_unrealized = sum(p.get("unrealized_pl", 0) or 0 for p in pdd_legs)

    # Same fix as intu_condor_babysitter.py (found 2026-08-25): fast_info's
    # lastPrice/regularMarketPrice is frozen at the regular-session close and
    # does NOT update pre/post-market -- prefer the real extended-hours price
    # when the market is actually in that state.
    try:
        t = yf.Ticker(TICKER)
        info = t.get_info()
        market_state = info.get("marketState")
        if market_state == "POST" and info.get("postMarketPrice"):
            spot = info["postMarketPrice"]
        elif market_state == "PRE" and info.get("preMarketPrice"):
            spot = info["preMarketPrice"]
        else:
            spot = info.get("regularMarketPrice") or t.fast_info.get("lastPrice")
    except Exception as e:
        telegram(f"PDD babysitter could not get spot price: {e}", high_priority=True)
        log_line(f"Could not fetch PDD spot price: {e}", "Alerted -- could not assess risk this cycle.")
        return

    if not spot or spot <= 0:
        telegram("PDD babysitter got an invalid spot price (0 or None) -- check manually.", high_priority=True)
        log_line("Invalid PDD spot price returned.", "Alerted -- could not assess risk this cycle.")
        return

    dist_put_pct = (spot - SHORT_PUT) / spot * 100
    dist_call_pct = (SHORT_CALL - spot) / spot * 100
    closest_dist = min(dist_put_pct, dist_call_pct)

    if spot <= SHORT_PUT or spot >= SHORT_CALL:
        zone = "HIGH"
        breach_side = "put" if spot <= SHORT_PUT else "call"
        breach_strike = SHORT_PUT if breach_side == "put" else SHORT_CALL
        msg = (f"PDD condor BREACHED the short {breach_side} strike (${breach_strike:.0f}) -- "
               f"spot ${spot:.2f}. Real ITM risk on that leg now. Unrealized P&L ${total_unrealized:.2f} "
               f"(entry credit ${ENTRY_CREDIT*100:.0f}, exp {EXPIRY_DATE}).")
    elif closest_dist <= WARN_BUFFER_PCT:
        zone = "WARNING"
        side = "put" if dist_put_pct < dist_call_pct else "call"
        strike = SHORT_PUT if side == "put" else SHORT_CALL
        msg = (f"PDD condor getting close: spot ${spot:.2f} is only {closest_dist:.1f}% from the "
               f"short {side} strike (${strike:.0f}). Unrealized P&L ${total_unrealized:.2f} "
               f"(entry credit ${ENTRY_CREDIT*100:.0f}, exp {EXPIRY_DATE}).")
    else:
        zone = "GREEN"
        msg = (f"PDD condor safely contained: spot ${spot:.2f}, {dist_put_pct:.1f}% above short put (${SHORT_PUT:.0f}), "
               f"{dist_call_pct:.1f}% below short call (${SHORT_CALL:.0f}). Unrealized P&L ${total_unrealized:.2f}.")

    state = load_state()
    zone_changed = state.get("last_zone") != zone
    had_prior_zone = state.get("last_zone") is not None

    # Always alert on WARNING/HIGH; for GREEN, only alert once when transitioning
    # back to safe from a worse zone (so a real recovery is confirmed, not silent) --
    # never on the very first run, where there's no prior zone to "recover" from.
    telegram_sent = False
    if zone == "HIGH":
        telegram(msg, high_priority=True)
        telegram_sent = True
    elif zone == "WARNING":
        telegram(msg, high_priority=False)
        telegram_sent = True
    elif zone == "GREEN" and zone_changed and had_prior_zone:
        telegram(f"PDD condor back to safely contained: {msg}", high_priority=False)
        telegram_sent = True

    print(f"[{zone}] {msg}")
    log_line(msg, f"Zone: {zone}. " + ("Telegram sent." if telegram_sent else "No Telegram (quiet/clean check)."))

    state["last_zone"] = zone
    state["last_check"] = datetime.now(timezone.utc).astimezone().isoformat()
    save_state(state)


if __name__ == "__main__":
    main()
