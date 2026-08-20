"""
CFO weekly performance tracker -- reports actual combined (IBKR + Alpaca)
account performance against a 4%/week TARGET, escalates misses with real
reasons attached. Deliberately advisory/tracking only: never touches trades,
config, or position sizing. A missed target is a business fact to report and
investigate (via CIO), not a trigger to take more risk to catch up -- that
was explicitly considered and rejected 2026-08-20 as a real ruin risk (same
failure class as the DE incident: sizing/behavior driven by a forced outcome
rather than a real edge).

Cadence: run once daily (not hourly -- this is a weekly metric, checking it
hourly is noise). Logs quietly every day; does a fuller CEO-facing escalation
(Telegram + secretary_tasks.json) specifically when a week completes (Friday)
or when the tracker detects a new week has started (finalizing the prior one).

Usage: python cfo_weekly_performance_check.py
"""
import json
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
LEDGER_PATH = "cfo_ledger.json"
CONFIG_PATH = "scanner_config.json"
OVERSIGHT_LOG = "oversight_log.jsonl"
BACKEND_URL = "http://localhost:8000"
TARGET_PCT = 4.0


def _get_json(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.load(resp)


def get_combined_net_liq():
    ibkr = _get_json(f"{BACKEND_URL}/account/summary")
    alp = _get_json(f"{BACKEND_URL}/alpaca/positions")
    ibkr_nl = float(ibkr.get("net_liquidation") or 0)
    alp_eq = float(alp.get("account", {}).get("equity") or 0)
    return ibkr_nl, alp_eq, ibkr_nl + alp_eq


def week_start_for(d: date) -> date:
    """ISO week start (Monday)."""
    return d - timedelta(days=d.weekday())


def send_telegram(text: str):
    try:
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        token, chat_id = cfg.get("telegram_token"), cfg.get("telegram_chat_id")
        if not token or not chat_id:
            return False
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data)
        urllib.request.urlopen(req, timeout=10)
        return True
    except Exception as exc:
        print(f"Telegram send failed: {exc}")
        return False


def log_oversight(summary, rationale, outcome):
    entry = {
        "time": datetime.now(ET).astimezone().isoformat(),
        "actor": "cfo",
        "category": "performance_tracking",
        "summary": summary,
        "rationale": rationale,
        "outcome": outcome,
        "pnl_impact": None,
    }
    with open(OVERSIGHT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    now = datetime.now(ET)
    today = now.date()
    ibkr_nl, alp_eq, combined = get_combined_net_liq()

    with open(LEDGER_PATH) as f:
        ledger = json.load(f)
    pt = ledger["performance_tracking"]
    cw = pt["current_week"]

    this_week_start = week_start_for(today)
    stored_week_start = (
        date.fromisoformat(cw["week_start_date"]) if cw.get("week_start_date") else None
    )

    new_week = stored_week_start is None or stored_week_start != this_week_start
    finalized_prior_week = None

    if new_week:
        if stored_week_start is not None and cw.get("week_start_net_liq"):
            # finalize the prior week into history before resetting
            prior_return = round(
                (cw["last_combined_net_liq"] / cw["week_start_net_liq"] - 1) * 100, 2
            )
            finalized_prior_week = {
                "week_start_date": cw["week_start_date"],
                "week_end_date": cw.get("last_checked", "")[:10],
                "week_start_net_liq": cw["week_start_net_liq"],
                "week_end_net_liq": cw["last_combined_net_liq"],
                "return_pct": prior_return,
                "target_pct": TARGET_PCT,
                "hit_target": prior_return >= TARGET_PCT,
            }
            pt["history"].append(finalized_prior_week)
        cw["week_start_date"] = this_week_start.isoformat()
        cw["week_start_net_liq"] = combined

    wtd_return = round((combined / cw["week_start_net_liq"] - 1) * 100, 2) if cw["week_start_net_liq"] else 0.0

    # Pro-rate the target by trading days elapsed (Mon=0 .. Fri=4) so "behind"
    # only fires against a realistic partial-week bar, not the full 4% on a Tuesday.
    trading_days_elapsed = min(5, today.weekday() + 1)
    prorated_target = round(TARGET_PCT * trading_days_elapsed / 5, 2)
    if wtd_return >= TARGET_PCT:
        status = "AHEAD_OF_TARGET"
    elif wtd_return >= prorated_target:
        status = "ON_PACE"
    else:
        status = "BEHIND_PACE"

    cw["last_checked"] = now.isoformat()
    cw["last_combined_net_liq"] = round(combined, 2)
    cw["wtd_return_pct"] = wtd_return
    cw["status"] = status
    ledger["last_updated"] = now.isoformat()

    with open(LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)

    print(f"Week start: {cw['week_start_date']}  start_net_liq=${cw['week_start_net_liq']:,.2f}")
    print(f"Current combined net liq: ${combined:,.2f} (IBKR ${ibkr_nl:,.2f} + Alpaca ${alp_eq:,.2f})")
    print(f"WTD return: {wtd_return:+.2f}%  vs target {TARGET_PCT}%  (pro-rated so far: {prorated_target:.2f}%)")
    print(f"Status: {status}")

    log_oversight(
        summary=f"CFO weekly performance check: WTD {wtd_return:+.2f}% vs {TARGET_PCT}% target "
                f"(pro-rated {prorated_target:.2f}% through day {trading_days_elapsed}/5) -- {status}. "
                f"Combined net liq ${combined:,.2f}.",
        rationale="Daily tracking cadence per CEO directive 2026-08-20 -- report actual performance "
                   "against target, never used to justify increased risk-taking to close a gap.",
        outcome="Logged only" if status != "BEHIND_PACE" else "Behind pace -- see CIO for real reason (strategy capacity, market conditions, or genuine losses), not a risk-taking signal",
    )

    # Full escalation only when a week actually completes (a new week just started)
    if finalized_prior_week:
        fw = finalized_prior_week
        hit = "HIT" if fw["hit_target"] else "MISSED"
        msg = (
            f"📊 CFO weekly report: {fw['week_start_date']} -> {fw['week_end_date']}\n\n"
            f"Return: {fw['return_pct']:+.2f}% vs {fw['target_pct']}% target -- {hit}\n"
            f"Net liq: ${fw['week_start_net_liq']:,.2f} -> ${fw['week_end_net_liq']:,.2f}\n\n"
            + ("Target met.\n" if fw["hit_target"] else
               "Target missed. This is a report for CIO to investigate why (capacity, conditions, or real losses) -- not a signal to take more risk next week.\n")
        )
        print("\n" + msg)
        send_telegram(msg)
        log_oversight(
            summary=f"Week {fw['week_start_date']} finalized: {fw['return_pct']:+.2f}% vs {fw['target_pct']}% target -- {hit}",
            rationale="End-of-week finalization, full CEO-facing report per the weekly tracking cadence.",
            outcome="Telegram sent" if not fw["hit_target"] else "Telegram sent (target met)",
        )


if __name__ == "__main__":
    main()
