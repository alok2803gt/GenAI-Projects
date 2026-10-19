"""
Standing trade monitor: catches a REAL, currently-open IBKR position that
has fallen out of its owning process's ACTIVE tracking, even when every
other record looks fine.

Real incident that prompted this (2026-10-15): Ashley bought 770P, an exit
signal arrived, the sell failed to fill after 3 ladder attempts, and
ashleyklieu_trigger_executor.py's own open_positions dict popped the name
BEFORE confirming the fill (a real pre-existing bug, fixed separately in
that file) -- so the live daily-watch loop stopped watching it for the
rest of the day. Neither existing monitor caught this:
  - ibkr_position_reconciler.py only catches the OPPOSITE desync (registry
    says open, IBKR shows flat). Here registry and IBKR fully AGREED the
    position was open, so by that check's own definition there was nothing
    to correct. (It did correctly clean up 770P the following morning once
    it actually expired flat -- close_reason "reconcile_externally_closed"
    -- that direction works fine; this file covers the other one.)
  - housekeeping_agent.py's check_ashley_state() only flags a same-day
    option whose strike/right isn't claimed by ANY of today's alert setups
    at all. 770P WAS one of that day's four setups, so it was "claimed" by
    name and never flagged, even though the live loop had already stopped
    actively watching it.
770P happened to expire OTM and worthless that time -- lucky, not caught.
The gap itself (a known, registered-open position with nobody left
watching for its close) is real regardless of how any one instance
resolves, so this runs independently of whether the root-cause bug is
fixed anywhere else -- defense in depth.

Two checks, run together every 15 min during market hours:

1. UNIVERSAL ORPHAN SWEEP: every real open IBKR option position should be
   referenced by SOME record -- the shared registry
   (alpaca_0dte_positions.json) or a known per-strategy state file.
   Matching here is a best-effort text search (each strategy's own state
   schema differs too much to parse all of them precisely), so a finding
   says so and asks for a manual check rather than claiming certainty.

2. LIVE-TRACKING DRIFT (the specific 2026-10-15 bug, generalized): for each
   strategy that keeps its OWN separate live-open-positions file alongside
   the shared registry (currently just Ashley -- see LIVE_TRACKERS below),
   a registry entry marked phase=open for that strategy MUST also appear
   in the strategy's own live file. If the registry says open but the live
   file disagrees, the owning process has silently stopped watching a
   position it is still legally holding.

   Extending to another long-running strategy with its own live state
   (e.g. spycondor's spx_0dte_live_state.json "spreads" dict, or Day
   Trader's day_trader_state.json "positions" dict once it's re-enabled)
   is adding one function to LIVE_TRACKERS once its exact schema is
   understood -- not done here yet; only Ashley's is wired up, since
   that's the one with a proven incident. A scheduled/stateless babysitter
   that re-reads the registry fresh every run (the butterflies, the GOOG
   weekly condor) can't suffer this specific drift at all, since it keeps
   no second copy to fall out of sync with -- deliberately not in this list.

Usage: python trade_monitor.py [--dry-run]
Scheduled via launchd every 15 min, 9:35-16:30 ET weekdays, same pattern as
ibkr_reconciler (com.ibkrtrader.reconciler.plist). --dry-run changes
nothing either way -- this script never writes to any state file or
registry, FLAG only, same philosophy as housekeeping_agent: never
auto-repair something that grants real position ownership.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ib_insync import IB

sys.path.insert(0, str(Path(__file__).parent))
from alpaca_0dte_common import load_registry  # noqa: E402
from ibkr_0dte_common import parse_occ_symbol  # noqa: E402

HERE = Path(__file__).parent
CFG_PATH = HERE / "scanner_config.json"
OVERSIGHT_LOG = HERE / "oversight_log.jsonl"

ET = ZoneInfo("America/New_York")
TWS_PORT = 7496
CLIENT_ID = 2441  # distinct from ibkr_position_reconciler.py's 2440

# Known per-strategy state files to text-search in the universal orphan
# sweep. Best-effort only -- see module docstring, check #1.
KNOWN_STATE_FILES = [
    "safe_income_auto_state.json", "harami_trader_state.json",
    "day_trader_state.json", "spx_0dte_live_state.json", "spx_0dte_state.json",
    "qqq_iv_divergence_state.json", "earnings_vol_crush_state.json",
    "ashleyklieu_open_positions.json", "stock_state.json",
]


def now_et():
    return datetime.now(ET)


def load_cfg():
    try:
        return json.loads(CFG_PATH.read_text())
    except Exception:
        return {}


def telegram_text(cfg, text):
    import requests
    try:
        from telegram_alert_gate import alert_enabled
        if not alert_enabled("reconciliation"):
            return
    except Exception:
        pass
    if not cfg.get("telegram_token"):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      data={"chat_id": cfg["telegram_chat_id"], "text": text, "parse_mode": "HTML"},
                      timeout=10)
    except Exception as e:
        print(f"Telegram send failed: {e}")


def oversight_log(summary, rationale, outcome=None):
    entry = {"time": now_et().isoformat(), "actor": "programmer", "category": "trade_monitor",
              "summary": summary, "rationale": rationale, "outcome": outcome, "pnl_impact": None}
    with open(OVERSIGHT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def live_option_positions(ib):
    """(ticker, expiry 'YYYYMMDD', right, strike) -> Position, for every real,
    currently nonzero option position IBKR reports."""
    out = {}
    for p in ib.portfolio():
        c = p.contract
        if c.secType != "OPT" or p.position == 0:
            continue
        key = (c.symbol, c.lastTradeDateOrContractMonth, c.right, round(float(c.strike), 4))
        out[key] = p
    return out


def registry_open_keys():
    """key -> (pos_id, strategy) for every OPEN, single-leg registry position.
    Multi-leg spreads need their own key shape (a strike/right pair, not one)
    -- left for a future extension rather than silently mismatched here."""
    reg = load_registry()
    out = {}
    for pos_id, pos in reg.get("positions", {}).items():
        if pos.get("phase") != "open":
            continue
        legs = pos.get("legs", [])
        if len(legs) != 1:
            continue
        try:
            ticker, expiry, strike, right = parse_occ_symbol(legs[0]["symbol"])
        except (KeyError, ValueError, TypeError):
            continue
        out[(ticker, expiry, right, round(strike, 4))] = (pos_id, pos.get("strategy", "?"))
    return out


def _ashley_still_tracked(key) -> bool:
    try:
        live = json.loads((HERE / "ashleyklieu_open_positions.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    ticker, expiry, right, strike = key
    for rec in live.values():
        if (ticker == "SPY" and rec.get("expiry") == expiry and rec.get("right") == right
                and abs(float(rec.get("strike", -1)) - strike) < 1e-6):
            return True
    return False


# strategy name (as the registry's own "strategy" field spells it) -> a
# function(key) -> bool, true if that strategy's own authoritative live
# state still has this contract. See module docstring, check #2.
LIVE_TRACKERS = {
    "ashley_signal": _ashley_still_tracked,
}


def check_universal_orphans(live, registry_keys, cfg, dry_run):
    raw_text = ""
    for fname in KNOWN_STATE_FILES:
        p = HERE / fname
        if p.exists():
            try:
                raw_text += p.read_text() + "\n"
            except Exception:
                pass

    findings = []
    today_ibkr = now_et().strftime("%Y%m%d")
    for key, pos in live.items():
        ticker, expiry, right, strike = key
        if key in registry_keys:
            continue
        strike_txt = f"{strike:g}"
        if expiry in raw_text and strike_txt in raw_text:
            continue
        findings.append((key, pos, expiry == today_ibkr))

    for key, pos, expires_today in findings:
        ticker, expiry, right, strike = key
        urgency = "HIGH PRIORITY -- expires TODAY" if expires_today else "flag"
        msg = (f"\U0001F6A8 <b>trade_monitor:</b> real open {ticker} {strike:g}{right} (exp {expiry}, "
               f"qty {pos.position:g}) matches NO registry entry and isn't mentioned in any known "
               f"strategy state file -- {urgency}. Best-effort text search, not a precise parse: "
               f"verify in IBKR before assuming this is truly unowned.")
        print(msg)
        if not dry_run:
            telegram_text(cfg, msg)
            oversight_log(msg, "No registry entry and no known state file reference for a real open position.",
                          outcome="flagged_unregistered_position")
    return findings


def check_live_tracking_drift(registry_keys, cfg, dry_run):
    findings = []
    today_ibkr = now_et().strftime("%Y%m%d")
    for key, (pos_id, strategy) in registry_keys.items():
        checker = LIVE_TRACKERS.get(strategy)
        if checker is None or checker(key):
            continue
        ticker, expiry, right, strike = key
        findings.append((key, pos_id, strategy, expiry == today_ibkr))

    for key, pos_id, strategy, expires_today in findings:
        ticker, expiry, right, strike = key
        urgency = ("HIGH PRIORITY -- expires TODAY, physically-settled auto-exercise risk"
                   if expires_today else "flag")
        msg = (f"\U0001F6A8 <b>trade_monitor:</b> {pos_id} ({strategy}) {ticker} {strike:g}{right} "
               f"(exp {expiry}) is OPEN in the registry, but {strategy}'s OWN live tracking no "
               f"longer has it -- {urgency}. Real incident 2026-10-15: this exact pattern (a failed "
               f"exit silently dropped from active tracking) happened to 770P. Check manually NOW.")
        print(msg)
        if not dry_run:
            telegram_text(cfg, msg)
            oversight_log(msg, f"{strategy}'s own live tracking has drifted from the registry's "
                          f"'open' state -- the position is real but nobody is actively watching it.",
                          outcome="flagged_live_tracking_drift")
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report issues without alerting/logging")
    args = ap.parse_args()

    cfg = load_cfg()
    ib = IB()
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    try:
        live = live_option_positions(ib)
        registry_keys = registry_open_keys()
        orphans = check_universal_orphans(live, registry_keys, cfg, args.dry_run)
        drift = check_live_tracking_drift(registry_keys, cfg, args.dry_run)
        total = len(orphans) + len(drift)
        if total == 0:
            print(f"[{now_et().isoformat()}] trade_monitor: no issues found "
                  f"({len(live)} live option position(s), {len(registry_keys)} open registry entries).")
        else:
            print(f"[{now_et().isoformat()}] trade_monitor: {total} issue(s) "
                  f"{'found (dry-run)' if args.dry_run else 'flagged'}.")
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
