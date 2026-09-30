"""
QQQ IV/Price Divergence Monitor -- built 2026-09-17 (CEO request) to watch,
day by day, for the specific pattern found in the June30-Jul29 2026 slide:
price making successive LOWER highs on failed relief bounces while IV makes
successive HIGHER lows at those same bounce peaks -- a real, single-instance
"distribution" divergence (each rally provided less complete IV relief than
the last, baseline anxiety ratcheting up even on the up days) that preceded
the actual capitulation low and the following +9.4% week-long rally.

IMPORTANT, stated once here and worth remembering: that July pattern is ONE
real case study, not a backtested/validated signal. This monitor watches for
the SAME shape going forward (useful, honest, real-time pattern-matching)
but does not claim the shape is statistically proven to predict anything --
consistent with this whole session's finding that no lead-lag IV->price
signal survived testing (pooled 112-ticker lag-correlation study, ~6,600 obs,
r in the noise range at every lag). Read this monitor's alerts as "here is
what the tape structurally looks like right now, and how it compares to the
one real precedent we have," not as a trade signal.

Why daily, not intraday "real-time": the only real IV history source this
system has is IBKR's OPTION_IMPLIED_VOLATILITY historical series, which is
itself a DAILY bar (see calc_gex_vex.py / archive_iv_daily.py) -- there is
no genuine intraday IV feed to poll. Running this every 30 min would just
re-check an unchanged daily bar. Scheduled once/day, after close (16:20 ET,
5 min after archive_iv_daily.py's 16:15 run so today's IV bar is already
fresh in iv_history.jsonl), matching this codebase's existing end-of-day
archiver pattern.

Swing detection: a simple threshold zigzag on QQQ's daily close. A pending
high/low extends as long as price keeps making new extremes; it CONFIRMS as
a real swing once price reverses by more than REVERSAL_PCT from that extreme
(1.0% -- roughly one day's realized move at current ~18-19% IV, i.e. sized
to filter ordinary daily noise, not real reversals). Every CONFIRMED swing
high's IV is compared to the prior confirmed swing high's IV: IV higher at
the new (lower) peak = divergence warning; IV lower = ordinary relief, no
flag. Symmetric check on swing lows during an uptrend (IV lower at a higher
low = melt-up complacency building, the topping mirror of the same idea) --
untested even as a single case study, included for symmetry, flagged as
such in the alert.

Also tracks proximity to the Oct 28-30 FOMC/earnings cluster (T-7/T-3/T-1
reminders) and flags when QQQ's IV crosses the 10y 25th/75th/90th percentile
bands established earlier this session (12.9/24.0/29.5%).

State persisted in qqq_iv_divergence_state.json so re-running (a retry, a
manual check) never double-alerts on the same confirmed swing.

Usage: python qqq_iv_divergence_monitor.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from ib_insync import IB, Stock

BACKEND_DIR = Path(__file__).parent
ET = timezone(timedelta(hours=-4))
STATE_FILE = BACKEND_DIR / "qqq_iv_divergence_state.json"
TWS_PORT = 7496
CLIENT_ID = 1852  # dedicated, avoids colliding with any other live clientId in this codebase

REVERSAL_PCT = 0.010          # 1.0% move from a running extreme confirms a swing
CLUSTER_DATE = date(2026, 10, 28)   # FOMC + MSFT/META/GOOGL; AAPL/AMZN follow 10/29, crush lags to ~10/30
CLUSTER_REMINDER_DAYS = {7, 3, 1, 0}
IV_PCTILE_BANDS = {"25th": 0.1287, "median": 0.1860, "75th": 0.2401, "90th": 0.2954}  # from the 10y history study


def now_et() -> datetime:
    return datetime.now(ET)


def load_cfg() -> dict:
    return json.loads((BACKEND_DIR / "scanner_config.json").read_text())


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "direction": None,           # "seeking_high" | "seeking_low"
        "extreme": None,             # {"date", "price", "iv"} -- the running extreme being tracked
        "confirmed_swings": [],      # list of {"type", "date", "price", "iv"}
        "last_run_date": None,
        "last_cluster_reminder_days": None,
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def telegram(cfg: dict, msg: str):
    import requests
    try:
        from telegram_alert_gate import alert_enabled
        if not alert_enabled("research_desk"):
            return
    except Exception:
        pass
    try:
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"Telegram send failed: {e}")


def oversight_log(summary: str, outcome: str, rationale: str = ""):
    entry = {
        "time": now_et().isoformat(), "actor": "trader", "category": "qqq_iv_divergence_monitor",
        "summary": summary, "rationale": rationale, "outcome": outcome, "pnl_impact": None,
    }
    with open(BACKEND_DIR / "oversight_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def fetch_latest_bar(ib: IB) -> dict:
    """Today's (or most recent trading day's) QQQ close + IV, straight from IBKR."""
    stk = Stock("QQQ", "SMART", "USD")
    ib.qualifyContracts(stk)
    px_bars = ib.reqHistoricalData(stk, endDateTime="", durationStr="5 D", barSizeSetting="1 day",
                                    whatToShow="TRADES", useRTH=True, formatDate=1, timeout=30)
    iv_bars = ib.reqHistoricalData(stk, endDateTime="", durationStr="5 D", barSizeSetting="1 day",
                                    whatToShow="OPTION_IMPLIED_VOLATILITY", useRTH=True, formatDate=1, timeout=30)
    px_map = {str(b.date): b.close for b in px_bars}
    iv_map = {str(b.date): b.close for b in iv_bars}
    common = sorted(set(px_map) & set(iv_map))
    latest = common[-1]
    return {"date": latest, "price": px_map[latest], "iv": iv_map[latest]}


def update_zigzag(state: dict, bar: dict) -> dict | None:
    """Feed today's bar into a standard single-direction threshold zigzag.
    Tracks ONE running extreme at a time (a pending high while price is
    rising, a pending low while price is falling); confirms and flips
    direction once price reverses REVERSAL_PCT off that extreme. (An
    earlier version of this tracked both a pending high and pending low
    simultaneously without retiring the stale one after a confirm -- caught
    in backtesting 2026-09-17: it re-confirmed the same June 30 high on
    every single subsequent day for a month, since nothing ever reset it.
    This version fixes that by keeping exactly one extreme and one
    direction, matching the standard zigzag definition.)"""
    if state.get("direction") is None:
        state["direction"] = "seeking_high"
        state["extreme"] = dict(bar)
        return None

    direction = state["direction"]
    extreme = state["extreme"]
    confirmed = None

    if direction == "seeking_high":
        if bar["price"] > extreme["price"]:
            state["extreme"] = dict(bar)
        elif bar["price"] <= extreme["price"] * (1 - REVERSAL_PCT):
            confirmed = {**extreme, "type": "high"}
            state["direction"] = "seeking_low"
            state["extreme"] = dict(bar)
    else:  # seeking_low
        if bar["price"] < extreme["price"]:
            state["extreme"] = dict(bar)
        elif bar["price"] >= extreme["price"] * (1 + REVERSAL_PCT):
            confirmed = {**extreme, "type": "low"}
            state["direction"] = "seeking_high"
            state["extreme"] = dict(bar)

    if confirmed:
        state["confirmed_swings"].append(confirmed)
        state["confirmed_swings"] = state["confirmed_swings"][-20:]
    return confirmed


def check_divergence(state: dict, new_swing: dict) -> str | None:
    """Compare the new confirmed swing to the most recent PRIOR swing of the
    same type. Returns an alert string if the real (high-side) or mirrored
    (low-side) divergence pattern is present, else None."""
    prior_same_type = [s for s in state["confirmed_swings"][:-1] if s["type"] == new_swing["type"]]
    if not prior_same_type:
        return None
    prev = prior_same_type[-1]

    if new_swing["type"] == "high":
        lower_high = new_swing["price"] < prev["price"]
        higher_iv = new_swing["iv"] > prev["iv"]
        if lower_high and higher_iv:
            return (f"🔴 <b>Distribution divergence (real pattern, single-precedent basis):</b> QQQ bounce peak "
                    f"{new_swing['date']} at ${new_swing['price']:.2f} is a LOWER high than the prior bounce "
                    f"({prev['date']} ${prev['price']:.2f}), but IV at this peak ({new_swing['iv']*100:.1f}%) is "
                    f"HIGHER than IV at the prior peak ({prev['iv']*100:.1f}%). This is the exact shape seen into "
                    f"the 7/29/2026 capitulation low -- each bounce provided less IV relief than the last. "
                    f"Not a proven signal (n=1 precedent), a real structural match worth knowing about.")
    else:
        higher_low = new_swing["price"] > prev["price"]
        lower_iv = new_swing["iv"] < prev["iv"]
        if higher_low and lower_iv:
            return (f"🟡 <b>Complacency divergence (mirrored pattern, untested even as a precedent):</b> QQQ pullback "
                    f"low {new_swing['date']} at ${new_swing['price']:.2f} is a HIGHER low than the prior pullback "
                    f"({prev['date']} ${prev['price']:.2f}), but IV at this low ({new_swing['iv']*100:.1f}%) is "
                    f"LOWER than at the prior low ({prev['iv']*100:.1f}%) -- melt-up complacency building on dips. "
                    f"This is the topping mirror of the distribution pattern; unlike that one, this exact shape "
                    f"hasn't even been observed once yet in this analysis -- purely structural, flagged for awareness.")
    return None


def check_pctile_band(bar: dict) -> str | None:
    iv = bar["iv"]
    if iv >= IV_PCTILE_BANDS["90th"]:
        return f"⚠️ QQQ IV ({iv*100:.1f}%) is at/above the 10-year 90th percentile (29.5%) -- real stress-regime territory."
    if iv >= IV_PCTILE_BANDS["75th"]:
        return f"QQQ IV ({iv*100:.1f}%) crossed above the 10-year 75th percentile (24.0%) -- elevated regime."
    if iv <= IV_PCTILE_BANDS["25th"]:
        return f"QQQ IV ({iv*100:.1f}%) crossed below the 10-year 25th percentile (12.9%) -- genuinely cheap regime."
    return None


def check_cluster_proximity(bar_date: date) -> str | None:
    days_out = (CLUSTER_DATE - bar_date).days
    if days_out in CLUSTER_REMINDER_DAYS:
        if days_out == 0:
            return ("📅 <b>Today is 10/28</b> -- FOMC + MSFT/META/GOOGL earnings. Per the 7/29-30 precedent, expect "
                    "the real price/IV resolution to lag AAPL/AMZN's report (10/29) by roughly a day -- watch "
                    "for the crush into 10/30, not necessarily today.")
        return f"📅 T-minus {days_out} trading day(s) to the Oct 28-30 FOMC/mega-cap-earnings cluster."
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print what would be sent, don't actually Telegram")
    args = ap.parse_args()

    et = now_et()
    print(f"[{et.isoformat()}] qqq_iv_divergence_monitor starting")
    if et.weekday() >= 5:
        print("Weekend, nothing to do.")
        return

    cfg = load_cfg()
    state = load_state()

    ib = IB()
    ib.errorEvent += lambda *a: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    bar = fetch_latest_bar(ib)
    ib.disconnect()

    if bar["date"] == state.get("last_run_date"):
        print(f"Already processed {bar['date']}, nothing new. (price=${bar['price']:.2f} iv={bar['iv']*100:.1f}%)")
        return
    state["last_run_date"] = bar["date"]

    messages = []

    new_swing = update_zigzag(state, bar)
    if new_swing:
        print(f"Confirmed swing {new_swing['type']}: {new_swing['date']} ${new_swing['price']:.2f} iv={new_swing['iv']*100:.1f}%")
        div_alert = check_divergence(state, new_swing)
        if div_alert:
            messages.append(div_alert)

    pctile_alert = check_pctile_band(bar)
    if pctile_alert:
        messages.append(pctile_alert)

    bar_date = datetime.strptime(bar["date"], "%Y-%m-%d").date()
    cluster_alert = check_cluster_proximity(bar_date)
    if cluster_alert and state.get("last_cluster_reminder_days") != (CLUSTER_DATE - bar_date).days:
        messages.append(cluster_alert)
        state["last_cluster_reminder_days"] = (CLUSTER_DATE - bar_date).days

    save_state(state)

    if messages:
        full_msg = (f"<b>QQQ IV/Price Monitor</b> -- {bar['date']}: ${bar['price']:.2f}, IV {bar['iv']*100:.1f}%\n\n"
                     + "\n\n".join(messages))
        print(full_msg)
        if not args.dry_run:
            telegram(cfg, full_msg)
            oversight_log(f"QQQ IV/price monitor alert on {bar['date']}",
                          outcome="alerted", rationale="; ".join(m[:80] for m in messages))
        else:
            print("[DRY RUN -- not sent]")
    else:
        print(f"No new signal today. price=${bar['price']:.2f} iv={bar['iv']*100:.1f}% "
              f"(currently {state['direction']}, running extreme ${state['extreme']['price']:.2f} "
              f"on {state['extreme']['date']})")


if __name__ == "__main__":
    main()
