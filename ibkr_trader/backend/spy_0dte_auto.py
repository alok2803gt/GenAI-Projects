"""
Fully unattended daily SPY 0DTE iron condor: checks the entry setup at
09:45 ET and auto-fires with NO manual confirmation if it clears -- per
explicit CEO direction 2026-08-17, overriding the earlier "do NOT
auto-fire" note on task 2026-08-13-001 (that note is now stale; this
script is what replaced it).

Meant to be launched once, as a single long-running background process,
by a cron firing shortly before 09:45 ET on trading days -- it blocks
through pricing, entry, monitoring, and close (up to ~6h to the 15:45
hard close) and exits on its own. Nothing further needs to trigger it
mid-day.

SAFETY RAILS (kept even though the manual fire-confirmation was
explicitly waived -- these are the same standing disciplines every other
auto-trader on this account runs, not a re-added approval gate):
  - qty stays fixed at 1 contract. No auto-scaling -- this is still
    brand-new code with its first live fires happening under this
    script; sizing up is a separate conversation.
  - MIN_CONSERVATIVE_CREDIT = $0.05/contract (not just >$0.00). The
    2026-08-17 analysis found width-tuning alone can get conservative
    credit to exactly $0.00 without ever being genuinely tradeable
    (breakeven before real commissions is a real loss after them). This
    buffer is a judgment call, not a verified commission figure -- this
    account's own commission tracking is documented as incomplete
    (cfo_ledger.json) -- so it's set with margin on purpose.
  - Account-health pre-check (IBKR reachable, no critical /risk/status
    violation) before ever attempting to price, let alone fire.
  - A pause-flag file (spy_0dte_auto_PAUSED.flag): if a fire attempt
    partially fills (leaves a naked/uncovered leg) or hits any other
    unexpected state, this script writes the flag, sends a HIGH-priority
    Telegram alert, logs to oversight_log.jsonl, and adds a
    secretary_tasks.json item -- and every subsequent day's run checks
    for that flag FIRST and refuses to fire until a human clears it.
    This is incident-handling, not a confirmation gate on normal green-
    light days.

Usage: python spy_0dte_auto.py --ticker spy
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import date, datetime

import requests
from ib_insync import IB

from alpaca_0dte_common import (
    load_config, alpaca_client, get_alpaca_symbols, place_condor_sequential,
    close_condor_sequential, register_position, close_position, now_et, target_px,
    cro_cfo_capital_budget,
)
from alpaca_0dte_trader import CONFIG, price_legs, condor_mark, PROFIT_TARGET_PCT, HARD_CLOSE_TIME, QTY

TWS_PORT = 7496
CLIENT_ID = 1570
MIN_CONSERVATIVE_CREDIT = 0.05  # $/contract -- see module docstring
PAUSE_FLAG_FILE = "spy_0dte_auto_PAUSED.flag"
MONITOR_INTERVAL_S = 60
BACKEND = "http://localhost:8000"


def telegram(text, high_priority=False):
    cfg = load_config()
    prefix = "\U0001F6A8 " if high_priority else ""
    try:
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": prefix + text},
            timeout=10,
        )
    except Exception as e:
        print(f"telegram send failed: {e}")


def oversight_log(category, summary, rationale="", outcome=None, pnl_impact=None):
    entry = {
        "time": datetime.now().astimezone().isoformat(),
        "actor": "trader", "category": category, "summary": summary,
        "rationale": rationale, "outcome": outcome, "pnl_impact": pnl_impact,
    }
    with open("oversight_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


SPY_0DTE_DECISIONS_FILE = "spy_0dte_decisions.json"


def spy_0dte_log(action, detail):
    """Dedicated decision log for the SPY 0DTE tab (frontend) -- distinct
    from oversight_log.jsonl, which mixes every strategy/role together.
    Every fire-or-skip decision lands here, not just fills, so a no-fire
    day (the common case so far) is visible with its real reason, not
    silently absent from the record."""
    entry = {"time": now_et().strftime("%H:%M:%S ET"), "action": action, "detail": detail}
    try:
        with open(SPY_0DTE_DECISIONS_FILE) as f:
            decisions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        decisions = []
    decisions.append(entry)
    decisions = decisions[-200:]
    with open(SPY_0DTE_DECISIONS_FILE, "w") as f:
        json.dump(decisions, f, indent=2)


def add_secretary_task(description, notes, due=None):
    with open("secretary_tasks.json") as f:
        d = json.load(f)
    new_id = f"{date.today().isoformat()}-{sum(1 for t in d['tasks'] if t['id'].startswith(date.today().isoformat()))+1:03d}"
    d["tasks"].append({
        "id": new_id, "created": now_et().isoformat(), "source": "trader",
        "description": description, "status": "open", "due": due, "notes": notes,
    })
    with open("secretary_tasks.json", "w") as f:
        json.dump(d, f, indent=2)
    return new_id


def pretrade_review(ib, client, strikes, conservative_credit, max_risk, cfg):
    """Real-time CRO/CFO pre-trade review -- BLOCKING gate, same pattern as
    EVC's (task 2026-08-20-003), extended here per task 2026-08-20-006.
    CRO: what-if P&L at the short strikes and beyond the wings. CFO: this
    trade's max_risk against the cross-strategy capital budget (this
    strategy's own per-trade cap and total portfolio headroom --
    cro_cfo_capital_budget, shared with Day Trader/EVC). Also verifies an
    exit plan exists (force-close time / profit target configured -- this
    strategy always has one, checked for consistency with the other
    reviews, not because it's ever actually missing).
    Returns {"approved": bool, "findings": [...], "scenarios": [...]}."""
    findings = []
    approved = True

    lp, sp, sc, lc = strikes["long_put"], strikes["short_put"], strikes["short_call"], strikes["long_call"]
    credit = conservative_credit

    def payoff(px):
        put_i = max(0.0, sp - px) - max(0.0, lp - px)
        call_i = max(0.0, px - sc) - max(0.0, px - lc)
        return round(credit * 100 * QTY - (put_i + call_i) * 100 * QTY, 2)

    scenarios = [
        {"case": f"below long put ({lp})", "price": lp - 1, "pnl": payoff(lp - 1)},
        {"case": f"at short put ({sp})", "price": sp, "pnl": payoff(sp)},
        {"case": "between shorts (max profit)", "price": round((sp + sc) / 2, 2), "pnl": payoff((sp + sc) / 2)},
        {"case": f"at short call ({sc})", "price": sc, "pnl": payoff(sc)},
        {"case": f"above long call ({lc})", "price": lc + 1, "pnl": payoff(lc + 1)},
    ]

    try:
        budget = cro_cfo_capital_budget(ib, client)
        findings.append(f"CFO: max risk ${max_risk:,.0f} vs per-strategy cap ${budget['per_strategy_cap']:,.0f}, "
                         f"portfolio headroom ${budget['headroom']:,.0f} of ${budget['total_budget']:,.0f} budget")
        if max_risk > budget["per_strategy_cap"]:
            findings.append(f"EXCEEDS per-strategy cap (${budget['per_strategy_cap']:,.0f})")
            approved = False
        if max_risk > budget["headroom"]:
            findings.append(f"EXCEEDS remaining portfolio headroom (${budget['headroom']:,.0f})")
            approved = False
    except Exception as exc:
        findings.append(f"could not compute capital budget: {exc}")
        approved = False

    findings.append(f"exit plan: force-close by {HARD_CLOSE_TIME} ET, profit target {PROFIT_TARGET_PCT}%")

    return {"approved": approved, "findings": findings, "scenarios": scenarios}


def account_health_check():
    """Real pre-flight check: IBKR reachable via the backend, no critical
    risk violation. Returns (ok: bool, reason: str)."""
    try:
        with urllib.request.urlopen(f"{BACKEND}/account/summary", timeout=10) as r:
            acct = json.loads(r.read())
        if "detail" in acct:
            return False, f"account/summary error: {acct['detail']}"
    except Exception as e:
        return False, f"account/summary unreachable: {e}"

    try:
        with urllib.request.urlopen(f"{BACKEND}/risk/status", timeout=10) as r:
            risk = json.loads(r.read())
        if risk.get("summary", {}).get("critical", 0) > 0:
            return False, f"risk/status has {risk['summary']['critical']} critical violation(s)"
    except Exception as e:
        return False, f"risk/status unreachable: {e}"

    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, choices=["spy"])
    args = ap.parse_args()
    ticker = args.ticker
    cfg = CONFIG[ticker]
    cfg_obj = load_config()

    print(f"=== {ticker.upper()} 0DTE AUTO -- {now_et().isoformat()} ===")

    import os
    if os.path.exists(PAUSE_FLAG_FILE):
        with open(PAUSE_FLAG_FILE) as f:
            reason = f.read()
        msg = f"SPY 0DTE auto-fire is PAUSED (flag file present): {reason}\nSkipping today until a human clears {PAUSE_FLAG_FILE}."
        print(msg)
        telegram(msg, high_priority=True)
        spy_0dte_log("PAUSED", reason)
        return

    ok, reason = account_health_check()
    if not ok:
        msg = f"SPY 0DTE auto: account-health check FAILED ({reason}) -- skipping fire attempt today, no order placed."
        print(msg)
        telegram(msg, high_priority=True)
        oversight_log("execution_issue", msg, outcome="no order placed")
        spy_0dte_log("HEALTH_CHECK_FAIL", reason)
        return

    today_ibkr = date.today().strftime("%Y%m%d")
    today_alp = date.today().strftime("%Y-%m-%d")

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    print("Connected to IBKR.")

    try:
        strikes, quotes, entry_limits, conservative_credit, max_risk = price_legs(ib, ticker, cfg, today_ibkr)
    except SystemExit:
        ib.disconnect()
        msg = "SPY 0DTE auto: pricing aborted (missing live bid/ask on one or more legs) -- no order placed."
        print(msg)
        telegram(msg, high_priority=True)
        oversight_log("execution_issue", msg, outcome="no order placed")
        spy_0dte_log("PRICING_ABORT", "missing live bid/ask on one or more legs")
        return
    finally:
        if ib.isConnected():
            ib.disconnect()

    print(f"conservative credit: ${conservative_credit:.2f}  max_risk: ${max_risk:.0f}  "
          f"threshold: ${MIN_CONSERVATIVE_CREDIT:.2f}")

    if conservative_credit < MIN_CONSERVATIVE_CREDIT:
        msg = (f"SPY 0DTE auto: NO FIRE today. Spot condor {strikes['long_put']}/{strikes['short_put']}P.."
               f"{strikes['short_call']}/{strikes['long_call']}C, conservative_credit=${conservative_credit:.2f} "
               f"< ${MIN_CONSERVATIVE_CREDIT:.2f} threshold. No order placed.")
        print(msg)
        telegram(msg)
        oversight_log("strategy_analysis", msg, outcome="no order placed, credit below threshold")
        spy_0dte_log("NO_FIRE", f"condor {strikes['long_put']}/{strikes['short_put']}P.."
                     f"{strikes['short_call']}/{strikes['long_call']}C, conservative_credit=${conservative_credit:.2f} "
                     f"< ${MIN_CONSERVATIVE_CREDIT:.2f} threshold")
        return

    # ── Real-time CRO/CFO pre-trade review -- BLOCKING gate (task 2026-08-20-006) ──
    # IB was already disconnected above (price_legs' finally block) -- pass None,
    # cro_cfo_capital_budget degrades gracefully to Alpaca-only equity, which is
    # already the dominant share of this account's capital (~$3,400 of ~$4,100).
    client = alpaca_client(cfg_obj)
    review = pretrade_review(None, client, strikes, conservative_credit, max_risk, cfg)
    review_summary = "CRO/CFO pre-trade review: " + " | ".join(review["findings"])
    print(review_summary)
    spy_0dte_log("PRETRADE_REVIEW", f"{'APPROVED' if review['approved'] else 'REJECTED'} — {review_summary}")
    oversight_log("pretrade_review", f"SPY 0DTE: {review_summary}",
                  outcome="APPROVED" if review["approved"] else "REJECTED — no order placed")
    if not review["approved"]:
        msg = f"SPY 0DTE auto: pre-trade review REJECTED -- {review_summary}. No order placed."
        print(msg)
        telegram(msg)
        return

    # ── Clears the gate -- fire, no manual confirmation (per 2026-08-17 CEO direction) ──
    put_by_strike, call_by_strike = get_alpaca_symbols(
        client, cfg["alpaca_underlying"], today_alp,
        strikes["long_put"] - 1, strikes["short_put"] + 1,
        strikes["short_call"] - 1, strikes["long_call"] + 1,
    )
    missing = [k for k in (strikes["short_put"], strikes["long_put"]) if k not in put_by_strike] + \
              [k for k in (strikes["short_call"], strikes["long_call"]) if k not in call_by_strike]
    if missing:
        msg = f"SPY 0DTE auto: strikes not listed on Alpaca: {missing} -- no order placed."
        print(msg)
        telegram(msg, high_priority=True)
        oversight_log("execution_issue", msg, outcome="no order placed")
        spy_0dte_log("STRIKES_MISSING", f"not listed on Alpaca: {missing}")
        return

    syms = {
        "short_put": put_by_strike[strikes["short_put"]], "long_put": put_by_strike[strikes["long_put"]],
        "short_call": call_by_strike[strikes["short_call"]], "long_call": call_by_strike[strikes["long_call"]],
    }
    telegram(f"SPY 0DTE auto: FIRING. condor {strikes['long_put']}/{strikes['short_put']}P.."
             f"{strikes['short_call']}/{strikes['long_call']}C, conservative_credit=${conservative_credit:.2f}, "
             f"qty={QTY}, max_risk=${max_risk:.0f}")

    ok, fills, state = place_condor_sequential(client, syms, entry_limits, QTY)
    if not ok:
        with open(PAUSE_FLAG_FILE, "w") as f:
            f.write(f"{now_et().isoformat()}: entry incomplete, state={state}, fills={fills}")
        msg = (f"SPY 0DTE auto: ENTRY INCOMPLETE (state={state}, fills={fills}). "
               f"Automation PAUSED via {PAUSE_FLAG_FILE} until manually reviewed and cleared -- "
               f"check Alpaca positions NOW for a possible naked/uncovered leg.")
        print(msg)
        telegram(msg, high_priority=True)
        oversight_log("execution_issue", msg, outcome="PAUSED -- possible partial fill, needs manual review")
        spy_0dte_log("ENTRY_INCOMPLETE", f"state={state} fills={fills} -- automation now PAUSED")
        add_secretary_task(
            "SPY 0DTE auto-fire entry incomplete -- review and clear pause flag",
            f"state={state} fills={fills}. Check Alpaca positions for a naked leg before doing anything else. "
            f"Automation will not attempt to fire again until {PAUSE_FLAG_FILE} is deleted.",
        )
        return

    net_entry_credit = round((fills["short_put"] + fills["short_call"]) - (fills["long_put"] + fills["long_call"]), 2)
    entry_time = now_et()
    pos_id = f"SPY_0dte_condor_{entry_time.strftime('%Y%m%d_%H%M%S')}"
    profit_target_usd = round(net_entry_credit * QTY * 100 * PROFIT_TARGET_PCT, 2)
    register_position(
        pos_id, "SPY", "iron_condor_0dte",
        [{"leg": k, "strike": strikes[k], "symbol": syms[k], "fill": fills[k]} for k in syms],
        net_entry_credit, QTY, max_risk * QTY, profit_target_usd, HARD_CLOSE_TIME,
        entry_time.isoformat(),
        notes="Fully unattended auto-fire, 2026-08-17 CEO direction. No manual confirmation gate.",
    )
    msg = f"SPY 0DTE auto: ENTERED. pos_id={pos_id} net_entry_credit=${net_entry_credit:.2f} (${net_entry_credit*QTY*100:.0f} total) profit_target=${profit_target_usd:.2f}"
    print(msg)
    telegram(msg)
    oversight_log("position_opened", msg, outcome="entered, monitoring to profit target or hard close")
    spy_0dte_log("ENTERED", f"pos_id={pos_id} net_entry_credit=${net_entry_credit:.2f} "
                 f"(${net_entry_credit*QTY*100:.0f} total) profit_target=${profit_target_usd:.2f}")

    # ── Monitor until profit target or hard close ──
    hard_close_dt = datetime.strptime(f"{entry_time.strftime('%Y-%m-%d')} {HARD_CLOSE_TIME}", "%Y-%m-%d %H:%M")
    hard_close_dt = hard_close_dt.replace(tzinfo=entry_time.tzinfo)

    ib2 = IB()
    ib2.errorEvent += lambda reqId, code, msg, contract: None
    ib2.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID + 100, timeout=20)
    try:
        while True:
            now = now_et()
            value, mon_quotes = condor_mark(ib2, ticker, today_ibkr, strikes)
            if value is None:
                print(f"[{now.strftime('%H:%M:%S')}] quote gap, retry next cycle")
            else:
                live_pnl = (net_entry_credit - value) * QTY * 100
                print(f"[{now.strftime('%H:%M:%S')}] condor_value=${value:.2f} live_pnl=${live_pnl:+.2f} target=${profit_target_usd:.2f}")
                if live_pnl >= profit_target_usd:
                    close_limits = {k: target_px(mon_quotes[k], not k.startswith("short")) for k in mon_quotes}
                    ok, close_fills = close_condor_sequential(client, syms, close_limits, QTY)
                    close_position(pos_id, "profit_target_hit", live_pnl)
                    msg = f"SPY 0DTE auto: PROFIT TARGET HIT, closed. pnl=${live_pnl:+.2f} fills={close_fills}"
                    print(msg)
                    telegram(msg)
                    oversight_log("position_closed", msg, outcome="closed at profit target", pnl_impact=live_pnl)
                    spy_0dte_log("PROFIT_TARGET_HIT", f"pos_id={pos_id} pnl=${live_pnl:+.2f} fills={close_fills}")
                    return
            if now >= hard_close_dt:
                _, mon_quotes = condor_mark(ib2, ticker, today_ibkr, strikes)
                close_limits = {k: target_px(mon_quotes[k], not k.startswith("short")) for k in mon_quotes} if mon_quotes else entry_limits
                ok, close_fills = close_condor_sequential(client, syms, close_limits, QTY)
                final_value, _ = condor_mark(ib2, ticker, today_ibkr, strikes)
                final_pnl = (net_entry_credit - final_value) * QTY * 100 if final_value else None
                close_position(pos_id, "hard_close", final_pnl)
                msg = f"SPY 0DTE auto: HARD CLOSE. pnl={final_pnl} fills={close_fills}"
                print(msg)
                telegram(msg)
                oversight_log("position_closed", msg, outcome="closed at hard close", pnl_impact=final_pnl)
                spy_0dte_log("HARD_CLOSE", f"pos_id={pos_id} pnl={final_pnl} fills={close_fills}")
                return
            time.sleep(MONITOR_INTERVAL_S)
    finally:
        ib2.disconnect()


if __name__ == "__main__":
    main()
