"""
Dedicated babysitter for the real INTU earnings condor entered Tue 2026-08-25
15:51:13 ET (EVC, via CRO/CFO pre-trade review, put_cushion_mult=1.3): short
312.5P/long 307.5P, short 390C/long 405C, exp 2026-08-28, net credit $2.74
(qty 1). Built the same night after finding EVC's own reconciliation logic
was wrongly stamping fresh Alpaca-venue positions "externally_closed" (fixed
in main.py the same night) -- this is independent, redundant monitoring on
top of that fix, not a replacement for it.

Same "escalation-only" design as the PDD condor babysitter (2026-08-24):
quiet in oversight_log.jsonl on every clean check, Telegram only when
something real changes.

Thresholds (checked against REAL Alpaca marks, not modeled):
  - GREEN (log only): spot > 5% away from both short strikes (312.5P, 390C)
  - WARNING (normal Telegram): spot within 5% of either short strike
  - HIGH (urgent Telegram): spot has breached a short strike (real ITM risk)
  - Self-expiring: exits quietly (still logs once) once run after the
    2026-08-28 expiry date, or if the position is no longer held (closed
    early, or expired) -- no manual cleanup required, though removing the
    Task Scheduler job itself after expiry is still recommended hygiene.

Usage: python intu_condor_babysitter.py
"""
import json
from datetime import date, datetime, timezone

import requests
import yfinance as yf

BACKEND = "http://localhost:8000"
LOG_FILE = "oversight_log.jsonl"
TICKER = "INTU"
EXPIRY_DATE = date(2026, 8, 28)
SHORT_PUT, LONG_PUT = 312.5, 307.5
SHORT_CALL, LONG_CALL = 390.0, 405.0
ENTRY_CREDIT = 2.74
WARN_BUFFER_PCT = 5.0
STATE_FILE = "intu_condor_babysitter_state.json"


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
        "category": "intu_condor_babysitter",
        "summary": summary,
        "rationale": "Dedicated active monitoring for the real INTU condor through its 2026-08-28 expiry -- redundant on top of the same-night reconciliation-bug fix, not a replacement for it.",
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
        log_line(f"INTU condor expiry ({EXPIRY_DATE}) has passed -- monitoring complete.",
                 "No further checks needed. Remove this Task Scheduler job.")
        print("Past expiry -- nothing to monitor. Remove the scheduled task.")
        return

    try:
        r = requests.get(f"{BACKEND}/alpaca/positions", timeout=10)
        r.raise_for_status()
        positions = r.json().get("positions", [])
    except Exception as e:
        telegram(f"INTU babysitter could not reach backend: {e}", high_priority=True)
        log_line(f"Backend unreachable: {e}", "Alerted -- could not check the position this cycle.")
        return

    intu_legs = [p for p in positions if p["symbol"].startswith(TICKER) and "2608" in p["symbol"]]
    if not intu_legs:
        log_line("INTU condor no longer held (closed early or already expired/settled).",
                 "No further monitoring needed -- position is gone.")
        print("No INTU position found -- already closed or settled.")
        return

    total_unrealized = sum(p.get("unrealized_pl", 0) or 0 for p in intu_legs)

    # Real bug found 2026-08-25, ~20 min after this script's own first run:
    # fast_info.lastPrice/regularMarketPrice is the REGULAR SESSION CLOSE --
    # it does NOT update in pre/post-market. INTU reports AH tonight, so
    # right when this position needs watching most, the naive price source
    # was silently frozen at the 4pm close ($357.46) while the real
    # after-hours price had already dropped to $327.80 (postMarketPrice) --
    # an 8.3% real move the first run completely missed, reporting a false
    # GREEN when the real distance to the short put was already inside the
    # 5% warning band. Prefer postMarketPrice/preMarketPrice when the market
    # is actually in that state; fall back to the regular price otherwise.
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
        telegram(f"INTU babysitter could not get spot price: {e}", high_priority=True)
        log_line(f"Could not fetch INTU spot price: {e}", "Alerted -- could not assess risk this cycle.")
        return

    if not spot or spot <= 0:
        telegram("INTU babysitter got an invalid spot price (0 or None) -- check manually.", high_priority=True)
        log_line("Invalid INTU spot price returned.", "Alerted -- could not assess risk this cycle.")
        return

    dist_put_pct = (spot - SHORT_PUT) / spot * 100
    dist_call_pct = (SHORT_CALL - spot) / spot * 100
    closest_dist = min(dist_put_pct, dist_call_pct)

    if spot <= SHORT_PUT or spot >= SHORT_CALL:
        zone = "HIGH"
        breach_side = "put" if spot <= SHORT_PUT else "call"
        breach_strike = SHORT_PUT if breach_side == "put" else SHORT_CALL
        msg = (f"INTU condor BREACHED the short {breach_side} strike (${breach_strike:.0f}) -- "
               f"spot ${spot:.2f}. Real ITM risk on that leg now. Unrealized P&L ${total_unrealized:.2f} "
               f"(entry credit ${ENTRY_CREDIT*100:.0f}, exp {EXPIRY_DATE}).")
    elif closest_dist <= WARN_BUFFER_PCT:
        zone = "WARNING"
        side = "put" if dist_put_pct < dist_call_pct else "call"
        strike = SHORT_PUT if side == "put" else SHORT_CALL
        msg = (f"INTU condor getting close: spot ${spot:.2f} is only {closest_dist:.1f}% from the "
               f"short {side} strike (${strike:.0f}). Unrealized P&L ${total_unrealized:.2f} "
               f"(entry credit ${ENTRY_CREDIT*100:.0f}, exp {EXPIRY_DATE}).")
    else:
        zone = "GREEN"
        msg = (f"INTU condor safely contained: spot ${spot:.2f}, {dist_put_pct:.1f}% above short put (${SHORT_PUT:.0f}), "
               f"{dist_call_pct:.1f}% below short call (${SHORT_CALL:.0f}). Unrealized P&L ${total_unrealized:.2f}.")

    state = load_state()
    zone_changed = state.get("last_zone") != zone
    had_prior_zone = state.get("last_zone") is not None

    telegram_sent = False
    if zone == "HIGH":
        telegram(msg, high_priority=True)
        telegram_sent = True
    elif zone == "WARNING":
        telegram(msg, high_priority=False)
        telegram_sent = True
    elif zone == "GREEN" and zone_changed and had_prior_zone:
        telegram(f"INTU condor back to safely contained: {msg}", high_priority=False)
        telegram_sent = True

    print(f"[{zone}] {msg}")
    log_line(msg, f"Zone: {zone}. " + ("Telegram sent." if telegram_sent else "No Telegram (quiet/clean check)."))

    state["last_zone"] = zone
    state["last_check"] = datetime.now(timezone.utc).astimezone().isoformat()
    save_state(state)


if __name__ == "__main__":
    main()
