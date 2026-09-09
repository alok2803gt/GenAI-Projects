"""
Generic babysitter for ALL currently-open EVC iron condors -- reads live
from /earnings-vol-crush/status instead of hardcoding a ticker/strikes per
script. The INTU/PDD condor babysitters this replaces each needed a brand
new script written per position; this one needs none, ever -- it just
watches whatever EVC currently has open (CRM/CRWD/NVDA as of 2026-08-26,
automatically picking up anything entered or closed after).

Same "escalation-only" design as the INTU/PDD precedents: quiet in
oversight_log.jsonl on every clean check, Telegram only when a position's
zone changes or is WARNING/HIGH. Real unrealized P&L is summed from actual
Alpaca leg fills (not the backend's own live_pnl snapshot, which only
updates on the monitor loop's own ~30s tick) -- same verify-against-the-
real-broker discipline used throughout this account.

Thresholds per position (checked against REAL yfinance spot -- pre/post-
market aware, same fix already proven on the INTU/PDD scripts: naive
regularMarketPrice/fast_info.lastPrice freezes at the regular-session
close and misses AH/BMO moves that matter most for these positions):
  - GREEN (log only): spot > 5% away from both short strikes
  - WARNING (normal Telegram): spot within 5% of either short strike
  - HIGH (urgent Telegram): spot has breached a short strike (real ITM risk)

Self-looping (checks every 15 min) -- run standalone, no external scheduler
needed: python evc_condor_babysitter.py
"""
import json
import time
from datetime import datetime, timezone

import requests
import yfinance as yf

BACKEND = "http://localhost:8000"
LOG_FILE = "oversight_log.jsonl"
STATE_FILE = "evc_condor_babysitter_state.json"
WARN_BUFFER_PCT = 5.0
CHECK_INTERVAL_S = 900


def telegram(msg: str, high_priority: bool = False):
    try:
        with open("scanner_config.json") as f:
            cfg = json.load(f)
        prefix = "\U0001F6A8 " if high_priority else "⚠️ "
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
        "category": "evc_condor_babysitter",
        "summary": summary,
        "rationale": "Generic active monitoring for every currently-open EVC condor -- redundant on top of the backend's own monitor loop, not a replacement for it.",
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
        return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_spot(ticker: str):
    t = yf.Ticker(ticker)
    info = t.get_info()
    market_state = info.get("marketState")
    if market_state == "POST" and info.get("postMarketPrice"):
        return info["postMarketPrice"]
    elif market_state == "PRE" and info.get("preMarketPrice"):
        return info["preMarketPrice"]
    return info.get("regularMarketPrice") or t.fast_info.get("lastPrice")


def check_once():
    try:
        r = requests.get(f"{BACKEND}/earnings-vol-crush/status", timeout=10)
        r.raise_for_status()
        positions = r.json().get("positions", {})
    except Exception as e:
        telegram(f"EVC babysitter could not reach backend: {e}", high_priority=True)
        log_line(f"Backend unreachable: {e}", "Alerted -- could not check any position this cycle.")
        return

    open_positions = {pid: p for pid, p in positions.items() if p.get("phase") == "open"}

    try:
        r2 = requests.get(f"{BACKEND}/alpaca/positions", timeout=10)
        r2.raise_for_status()
        alpaca_positions = r2.json().get("positions", [])
    except Exception as e:
        alpaca_positions = []
        print(f"Could not fetch Alpaca positions for real P&L: {e}")

    state = load_state()
    if not open_positions:
        if state:
            log_line("No EVC positions currently open.", "Nothing to monitor this cycle.")
            save_state({})
        print("No open EVC positions.")
        return

    for pos_id, pos in open_positions.items():
        ticker      = pos["ticker"]
        short_put   = pos["short_put"]
        short_call  = pos["short_call"]
        net_credit  = pos["net_credit"]
        expiry      = pos["expiry"]

        alp_symbols = set(pos.get("alpaca_symbols", {}).values())
        legs = [p for p in alpaca_positions if p["symbol"] in alp_symbols]
        total_unrealized = sum(p.get("unrealized_pl", 0) or 0 for p in legs)

        try:
            spot = get_spot(ticker)
        except Exception as e:
            telegram(f"EVC babysitter ({ticker}) could not get spot price: {e}", high_priority=True)
            log_line(f"Could not fetch {ticker} spot price: {e}", "Alerted -- could not assess risk this cycle.")
            continue
        if not spot or spot <= 0:
            telegram(f"EVC babysitter ({ticker}) got an invalid spot price -- check manually.", high_priority=True)
            log_line(f"Invalid {ticker} spot price returned.", "Alerted -- could not assess risk this cycle.")
            continue

        dist_put_pct  = (spot - short_put) / spot * 100
        dist_call_pct = (short_call - spot) / spot * 100
        closest_dist  = min(dist_put_pct, dist_call_pct)

        if spot <= short_put or spot >= short_call:
            zone = "HIGH"
            breach_side   = "put" if spot <= short_put else "call"
            breach_strike = short_put if breach_side == "put" else short_call
            msg = (f"{ticker} condor BREACHED the short {breach_side} strike (${breach_strike:.0f}) -- "
                   f"spot ${spot:.2f}. Real ITM risk on that leg now. Unrealized P&L ${total_unrealized:.2f} "
                   f"(entry credit ${net_credit*100:.0f}, exp {expiry}).")
        elif closest_dist <= WARN_BUFFER_PCT:
            zone = "WARNING"
            side   = "put" if dist_put_pct < dist_call_pct else "call"
            strike = short_put if side == "put" else short_call
            msg = (f"{ticker} condor getting close: spot ${spot:.2f} is only {closest_dist:.1f}% from the "
                   f"short {side} strike (${strike:.0f}). Unrealized P&L ${total_unrealized:.2f} "
                   f"(entry credit ${net_credit*100:.0f}, exp {expiry}).")
        else:
            zone = "GREEN"
            msg = (f"{ticker} condor safely contained: spot ${spot:.2f}, {dist_put_pct:.1f}% above short put "
                   f"(${short_put:.0f}), {dist_call_pct:.1f}% below short call (${short_call:.0f}). "
                   f"Unrealized P&L ${total_unrealized:.2f}.")

        prior          = state.get(ticker, {})
        zone_changed   = prior.get("last_zone") != zone
        had_prior_zone = prior.get("last_zone") is not None

        telegram_sent = False
        if zone == "HIGH":
            telegram(msg, high_priority=True)
            telegram_sent = True
        elif zone == "WARNING":
            telegram(msg, high_priority=False)
            telegram_sent = True
        elif zone == "GREEN" and zone_changed and had_prior_zone:
            telegram(f"{ticker} condor back to safely contained: {msg}", high_priority=False)
            telegram_sent = True

        print(f"[{zone}] {msg}")
        log_line(msg, f"Zone: {zone}. " + ("Telegram sent." if telegram_sent else "No Telegram (quiet/clean check)."))

        state[ticker] = {"last_zone": zone, "last_check": datetime.now(timezone.utc).astimezone().isoformat()}

    still_open_tickers = {p["ticker"] for p in open_positions.values()}
    state = {t: v for t, v in state.items() if t in still_open_tickers}
    save_state(state)


def main():
    print(f"EVC condor babysitter started {datetime.now(timezone.utc).astimezone().isoformat()} "
          f"(checking every {CHECK_INTERVAL_S}s)")
    while True:
        try:
            check_once()
        except Exception as e:
            print(f"Check cycle error: {e}")
        time.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    main()
