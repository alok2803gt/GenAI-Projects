"""
Persistent, DAILY version of the Ashley-alert trigger executor (generalized
2026-08-27 from the 2026-08-26 one-day version, which hardcoded that day's
specific strikes/zones). CEO instruction 2026-08-27: "i think we should
schedule daily during market hours" -- runs continuously, every trading
day, with no manual re-launch needed.

Each trading day:
  1. Wait for ashleyklieu to post her daily setups in Discord (same
     "<strike><right> Entry: <lo> to <hi>" format ashleyklieu_alert_monitor.py
     already parses -- same regex, reused verbatim for consistency). Gives
     up for the day (logs + informational Telegram) if nothing appears by
     WAIT_CUTOFF_ET.
  2. ENTRY: watches SPY spot via IBKR; the moment spot enters one of her
     stated zones for an unfired setup, places a real BUY limit order (1
     contract) for the corresponding 0DTE SPY option, via the
     favorable->mid->aggressive ladder (ibkr_place_leg_with_ladder_async,
     ibkr_0dte_common.py) proven 2026-08-26 on Alpaca -- no flat, no-retry
     price.
  3. EXIT: polls the same Discord channel for her follow-up messages. A
     clear, past-tense exit phrase ("took profit", "stopped out", "closed",
     "i'm out", etc.) on a setup this run currently holds triggers a real
     sell via the same ladder mechanism. Conservative on ambiguity:
     conditional/hypothetical phrasing ("if stopped...") is never treated
     as a fill trigger, and an exit signal with >1 position open and no
     clear match escalates via Telegram instead of guessing.
  4. FORCE-CLOSE SAFETY WINDOW at 15:55 ET: any position still open with no
     exit signal seen gets a real forced close attempted here, before
     market close. Self-terminates for the day right after (0DTE, nothing
     carries overnight); if the forced close somehow still didn't clear a
     position, flags it HIGH PRIORITY, then loops back to wait for the
     NEXT trading day.

CEO decision 2026-08-27 (asked directly, not assumed): keep running fully
autonomously at 1 contract per setup, no day-count cap -- flag anomalies
(no alert found by cutoff, unusual setup count, a materially bad day)
rather than pausing silently.

Execution migrated from Alpaca to native IBKR 2026-09-07 (Alpaca's real
account balance dropped to $0 buying power, fully consumed by an unrelated
existing position -- no longer usable for any strategy). This also closed
a real gap step 4 above depended on: Alpaca used to auto-close any
still-open 0DTE position at 15:45 ET on its own, which is what made
"alert if still open at market close" a safe enough fallback. IBKR has no
equivalent broker-side auto-close for single-leg equity options, so the
15:55 ET forced-close attempt (FORCE_CLOSE_ET) was added as this
strategy's own real safety net -- SPY options are physically settled, so
an ITM position left open past close risks real auto-exercise (SPY shares
delivered Monday), not just expiring worthless.
"""
import asyncio
import json
import re
import sys
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests
from ib_insync import IB, Option, Stock

sys.path.insert(0, ".")
from alpaca_0dte_common import (
    load_config,
    register_position as registry_register_position,
    close_position as registry_close_position,
)
from ibkr_0dte_common import (
    ibkr_place_leg_with_ladder_async, ibkr_has_open_position, ibkr_get_last_closing_fill,
)
from macro_calendar import is_macro_day

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ET = ZoneInfo("America/New_York")
CHANNEL_ID = "1540614267152371752"
TARGET_USERNAME = "ashleyklieu"
DISCORD_POLL_EVERY_N_TICKS = 3   # entry loop ticks at 5s -> Discord checked ~every 15s
WAIT_CUTOFF_ET = (13, 0)         # give up waiting for today's alert past 1:00 PM ET -- loosened
                                  # from 11:00 (CEO instruction 2026-08-27) after she posted her
                                  # daily setups at 12:20 PM ET that day, past the old cutoff,
                                  # and the run skipped a real, tradeable day for it. 1:00 PM still
                                  # leaves ~3 hours before the 4:00 PM close for a 0DTE entry+exit.
DAY_POLL_S = 30                  # how often the outer daily loop checks in
STATE_FILE = "ashleyklieu_trigger_executor_state.json"
FORCE_CLOSE_ET = (15, 55)        # real safety mechanism added 2026-09-07 (IBKR migration): Alpaca
                                  # used to auto-close any still-open 0DTE position at 15:45 ET on
                                  # its own, which is what made "alert if still open at market close"
                                  # an acceptable fallback rather than a real gap -- porting the
                                  # Alpaca-era logic verbatim to IBKR (which has NO equivalent
                                  # auto-close for single-leg equity options) would have silently
                                  # dropped the actual safety net while keeping only the alert.
                                  # SPY options are physically settled -- an ITM position left open
                                  # past close risks real auto-exercise (Monday SPY shares), not
                                  # just "expiring worthless" the way the old alert text implied.

# Same pattern as ashleyklieu_alert_monitor.py -- reused verbatim so both
# scripts parse her messages identically.
ENTRY_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*([CP])\b[^0-9]{0,15}Entry:?\s*(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

EXIT_PHRASES = [
    "took profit", "take profit", "tp hit", "closed my", "closed the",
    "closing my", "closing the", "i'm out", "im out", "stopped out",
    "stopped",  # REAL BUG found 2026-09-03: "stopped out" was listed but not
                # the bare "stopped" -- "stopped out" is not a substring of
                # "stopped", so her real 11:12 ET exit message ("stopped")
                # was silently NOT recognized as an exit signal. A 3-contract
                # SPY 770P position sat unmanaged for 2+ hours as a result,
                # decaying from a real, closeable value down to near-worthless
                # (real loss ~-$244.50 of a possible -$249 max, confirmed via
                # real Alpaca order history and current bid/ask). No
                # AMBIGUOUS-style hypothetical-marker guard needed here --
                # unlike "take profit"/"took profit", "stopped" doesn't
                # overlap with her casual non-actionable commentary style.
    "flat now", "out now", "banked", "locked in profit", "covering here",
    "covered here",
]
CONDITIONAL_PREFIXES = ("if ", "when ", "should ", "would ", "in case")
# Real clarification from Ashley herself, 2026-08-31 (CEO asked her directly
# when to take profit): "take profit is subjective, I'd go for a 10-20%. But
# when I share TP that's TP alert (take profit)." Two real things this
# confirms: (1) her casual "I'd take profit..."-style commentary is just her
# personal opinion, NOT a real signal -- exactly the failure mode already
# found 2026-08-30 (the SPY call closed on "SPY taking off, but I'd take
# profit already now", a hypothetical, not a real signal); (2) her actual,
# deliberate alert convention is specifically sharing "TP" -- confirmed by
# her own real historical usage (bare "TP" and "<price> TP" messages this
# week). AMBIGUOUS_EXIT_PHRASES lists the ones that overlap with casual,
# non-actionable commentary and need the extra hypothetical check; the rest
# of EXIT_PHRASES aren't said hypothetically in casual chat the same way.
AMBIGUOUS_EXIT_PHRASES = ("take profit", "took profit")
HYPOTHETICAL_MARKERS = ("i'd", "i would")  # she only ever describes her own trades in first person
TP_WORD = re.compile(r"\btp\b", re.IGNORECASE)

TELEGRAM_TOKEN = None
TELEGRAM_CHAT = None


CONTRACTS_OVERRIDE_FILE = "ashleyklieu_contracts_override.json"


def contracts_for_today() -> int:
    """Real per-setup contract quantity for TODAY only. Self-expiring by
    design (CEO ask 2026-08-31: "increase Ashley's SPY number of options to
    3 today") -- reads a small override file and only honors it if the
    stored date matches today, so tomorrow this silently reverts to the
    default of 1 without anyone needing to remember to change it back.
    """
    try:
        with open(CONTRACTS_OVERRIDE_FILE) as f:
            o = json.load(f)
        if o.get("date") == date.today().isoformat():
            return int(o["qty"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
        pass
    return 1


def occ_symbol(strike, right, yymmdd):
    """Standard OCC option symbol -- used only as a human-readable registry
    label since the 2026-09-07 IBKR migration, not for order routing
    (IBKR orders go by qualified Option contract, not this string)."""
    strike_str = f"{int(round(strike * 1000)):08d}"
    return f"SPY{yymmdd}{right}{strike_str}"


FIRED_STATE_FILE = "ashleyklieu_fired_today.json"


def load_fired_today() -> set:
    """Real persisted 'already fired today' setups. Real incident 2026-08-31:
    fired/open_positions lived only in the running process's memory -- a
    restart (needed mid-day to deploy any fix) forgot a setup was already
    done, INCLUDING one the CEO had deliberately closed manually, and could
    re-buy it the moment price revisited that zone. Self-scoped to today's
    date so it never bleeds into tomorrow (0DTE -- yesterday's strikes don't
    even exist tomorrow, but the date check keeps this honest regardless).
    """
    try:
        with open(FIRED_STATE_FILE) as f:
            d = json.load(f)
        if d.get("date") == date.today().isoformat():
            return set(d.get("fired", []))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return set()


def save_fired_today(fired: set) -> None:
    with open(FIRED_STATE_FILE, "w") as f:
        json.dump({"date": date.today().isoformat(), "fired": sorted(fired)}, f)


def reconcile_open_positions(ib, setups: list[dict], contracts: dict) -> dict:
    """Real IBKR position check for each of today's setups at startup --
    NOT trusting local bookkeeping, which is exactly what a restart loses.
    A genuinely still-open real position (e.g. the process crashed mid-day
    while holding one) gets restored into open_positions with its REAL qty
    and avg cost, so exit-signal monitoring resumes correctly. A setup with
    no matching real position (already closed, manually or otherwise, or
    never filled) is correctly left alone -- callers combine this with
    load_fired_today() to also mark it as "already done."

    contracts: dict of setup name -> already-qualified Option contract
    (built in run_daily_watch before this is called). IBKR's avgCost for
    an option position is the total cost PER CONTRACT (price x multiplier,
    i.e. already x100) -- divided back down here to the per-share premium
    price this script stores everywhere else. Migrated from Alpaca
    2026-09-07 -- not yet live-verified against a real restart-with-open-
    position scenario (market closed at build time); verify this
    specifically the first time it actually matters.
    """
    try:
        real_positions = {p.contract.conId: p for p in ib.positions() if p.position != 0}
    except Exception as exc:
        print(f"reconcile_open_positions: could not fetch real IBKR positions: {exc}")
        return {}
    restored = {}
    for s in setups:
        c = contracts.get(s["name"])
        if c is None:
            continue
        p = real_positions.get(c.conId)
        if p:
            restored[s["name"]] = {
                "symbol": occ_symbol(s["strike"], s["right"], date.today().strftime("%y%m%d")),
                "fill_px": abs(float(p.avgCost)) / 100,
                "qty": int(abs(p.position)), "opened_at": None,
            }
    return restored


def telegram(msg, high_priority=True):
    try:
        prefix = "\U0001F6A8 " if high_priority else "\U0001F4E1 "
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": prefix + msg, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as e:
        print(f"Telegram send failed: {e}")


def oversight_log(summary, rationale, outcome=None):
    entry = {
        "time": datetime.now(ET).isoformat(),
        "actor": "trader",
        "category": "ashleyklieu_trigger_executor",
        "summary": summary,
        "rationale": rationale,
        "outcome": outcome,
        "pnl_impact": None,
    }
    with open("oversight_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"last_processed_date": None, "last_message_id": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def is_market_hours(now):
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t


def is_exit_signal(content: str) -> bool:
    lower = content.strip().lower()
    if lower.startswith(CONDITIONAL_PREFIXES) or " if " in lower[:40]:
        return False
    # Her own explicit, deliberate alert convention (confirmed directly by
    # her, 2026-08-31) -- always real when present, regardless of anything
    # else in the message.
    if TP_WORD.search(lower):
        return True
    # "take profit"/"took profit" specifically overlap with her casual,
    # non-actionable commentary about profit-taking philosophy -- a
    # hypothetical marker anywhere in the message (not just at the very
    # start, which CONDITIONAL_PREFIXES already covers) means it's her
    # opinion, not a real trade she's making.
    if any(p in lower for p in AMBIGUOUS_EXIT_PHRASES):
        return not any(h in lower for h in HYPOTHETICAL_MARKERS)
    return any(p in lower for p in EXIT_PHRASES if p not in AMBIGUOUS_EXIT_PHRASES)


def msg_date_et(m: dict):
    """Real Discord message date, in ET -- used to reject stale content (a
    prior trading day's alert or exit message) from being treated as live.
    Real bug found 2026-08-30: without this, a message sitting unprocessed
    overnight (or across a restart) could get accepted as if posted today."""
    ts = m.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).astimezone(ET).date()
    except Exception:
        return None


def match_open_setup(content: str, open_positions: dict, setups: list):
    numbers = re.findall(r"\d+(?:\.\d+)?", content)
    for name in open_positions:
        strike = next(s["strike"] for s in setups if s["name"] == name)
        for n in numbers:
            if abs(float(n) - strike) < 1.0:
                return name
    return None


def setups_from_content(content: str) -> list[dict]:
    """Parse '<strike><right> Entry: <lo> to <hi>' setups from raw message
    content. Factored out of wait_for_daily_setups so the same parsing can
    be reused to re-check an already-seen message for a real edit."""
    setups = []
    for strike, right, lo, hi in ENTRY_PATTERN.findall(content):
        strike_f, lo_f, hi_f = float(strike), float(lo), float(hi)
        name = f"{strike_f:g}{right.upper()}"
        setups.append({"name": name, "strike": strike_f, "right": right.upper(),
                       "lo": min(lo_f, hi_f), "hi": max(lo_f, hi_f)})
    return setups


def fetch_message(headers, message_id: str) -> dict | None:
    """Fetch ONE specific Discord message fresh by ID -- used to detect a
    real edit to an already-processed alert. Real incident 2026-08-31:
    Ashley posted '773 C', then edited it to the correct '763 C' ~3 minutes
    later. The normal after=last_message_id poll design never revisits a
    message it's already consumed, so without this the executor would have
    kept watching the wrong, stale strike for the rest of the day with no
    warning -- caught only by manually cross-checking the live log against
    real Discord content."""
    try:
        r = requests.get(f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages/{message_id}",
                          headers=headers, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"fetch_message error for {message_id}: {exc}")
        return None


def wait_for_daily_setups(headers, state):
    """Poll Discord until ashleyklieu posts a message containing >=1 real
    '<strike><right> Entry: <lo> to <hi>' match, or WAIT_CUTOFF_ET passes.
    Returns (setups list or None, updated last_message_id, alert_edited_ts).
    last_message_id, when a match is found, IS the alert message's own ID
    (the loop returns on the same iteration it sets last_message_id = m["id"]
    for that message) -- reused by the caller as the message to re-check for
    edits throughout the day."""
    last_message_id = state.get("last_message_id")
    first_attempt = True
    while True:
        now = datetime.now(ET)
        cutoff = now.replace(hour=WAIT_CUTOFF_ET[0], minute=WAIT_CUTOFF_ET[1], second=0, microsecond=0)
        past_cutoff = (not is_market_hours(now) and now.hour >= WAIT_CUTOFF_ET[0]
                       and now.minute >= WAIT_CUTOFF_ET[1]) or now > cutoff
        # Real bug fixed 2026-09-01: this used to give up the instant wall-clock
        # time was past WAIT_CUTOFF_ET, WITHOUT EVER CHECKING DISCORD -- so any
        # restart after the cutoff (e.g. to deploy a fix) always reported "no
        # setups today," even when a real setup had already been posted and
        # traded earlier that same day (confirmed live: restarting at 15:22
        # after a real setup fired and closed before 13:00 reported "no setups
        # posted by 13:00" and gave up for the rest of the day). Always attempt
        # at least one real fetch before ever returning on cutoff, so an
        # already-posted setup from earlier today is still found after a
        # restart, however late in the day that restart happens.
        if past_cutoff and not first_attempt:
            return None, last_message_id, None
        first_attempt = False

        try:
            params = {"limit": 20}
            if last_message_id:
                params["after"] = last_message_id
            r = requests.get(f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
                              headers=headers, params=params, timeout=15)
            if r.status_code == 429:
                import time as _t
                _t.sleep(float(r.json().get("retry_after", 5)) + 1)
                continue
            r.raise_for_status()
            msgs = r.json()
            msgs.sort(key=lambda m: int(m["id"]))
            for m in msgs:
                last_message_id = m["id"]
                if m.get("author", {}).get("username", "") != TARGET_USERNAME:
                    continue
                content = (m.get("content") or "").strip()
                matches = ENTRY_PATTERN.findall(content)
                if not matches:
                    continue
                # Real bug found 2026-08-30 (2026-08-28 postmortem): this used
                # to accept the first Entry-pattern match after last_message_id
                # with no date check -- so a prior day's already-superseded
                # "Alerts Today" post (e.g. sitting unprocessed after a
                # restart) got treated as if Ashley posted it today, entering
                # a strike she'd already moved past. last_message_id has
                # already advanced above, so a stale match is never revisited.
                msg_date = msg_date_et(m)
                if msg_date != datetime.now(ET).date():
                    print(f"Ignoring Entry-pattern message from {msg_date} (not today): {content[:100]!r}")
                    continue
                return setups_from_content(content), last_message_id, m.get("edited_timestamp")
        except Exception as exc:
            print(f"Discord poll error while waiting for setups: {exc}")

        import time as _t
        _t.sleep(15)


async def close_position(ib, name, pos, setups, reason_text):
    strike = next(s["strike"] for s in setups if s["name"] == name)
    right  = next(s["right"] for s in setups if s["name"] == name)
    today_ibkr = date.today().strftime("%Y%m%d")
    c = Option("SPY", today_ibkr, strike, right, "SMART", tradingClass="SPY")
    await ib.qualifyContractsAsync(c)
    td = ib.reqMktData(c, "", False, False)
    await asyncio.sleep(3)
    bid = td.bid if td.bid and td.bid > 0 else None
    ask = td.ask if td.ask and td.ask > 0 else None
    mid = (bid + ask) / 2 if bid and ask else None
    ib.cancelMktData(c)

    qty = pos.get("qty", 1)  # must match what was actually bought -- never re-derive from today's
                              # override, since a position opened under an earlier qty must close
                              # at that SAME qty even if the override changes intraday.
    ok, fill_px = await ibkr_place_leg_with_ladder_async(
        ib, c, "SELL", f"ashley {name} exit", qty, bid, ask, mid)

    if ok:
        pnl = round((fill_px - pos["fill_px"]) * 100 * qty, 2)
        msg = (f"Ashley exit FILLED: sold {qty}x {name} @ ${fill_px} (bought @ ${pos['fill_px']}, "
               f"pnl=${pnl}). Trigger: \"{reason_text}\"")
        print(msg)
        telegram(msg, high_priority=True)
        oversight_log(msg, "Following Ashley's real-time exit (daily scheduled run).",
                      outcome=f"closed @ {fill_px}, pnl={pnl}")
        if pos.get("pos_id"):
            try:
                registry_close_position(pos["pos_id"], "automated_exit", pnl)
            except Exception as exc:
                print(f"registry_close_position failed (non-fatal): {exc}")
    else:
        msg = f"Ashley exit {name}: SELL to close did NOT fill (bid={bid} ask={ask}) -- check IBKR manually NOW."
        print(msg)
        telegram(msg, high_priority=True)
        oversight_log(msg, "Exit order placement attempted but not confirmed filled.", outcome="not_filled")
    return ok


async def run_daily_watch(setups, headers, last_message_id, alert_message_id=None, alert_edited_ts=None):
    cfg = load_config()
    global TELEGRAM_TOKEN, TELEGRAM_CHAT
    TELEGRAM_TOKEN = cfg["telegram_token"]
    TELEGRAM_CHAT = cfg["telegram_chat_id"]

    today_ibkr  = date.today().strftime("%Y%m%d")
    today_yymmdd = date.today().strftime("%y%m%d")

    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=994, timeout=15)

    spy = Stock("SPY", "SMART", "USD")
    await ib.qualifyContractsAsync(spy)
    spy_td = ib.reqMktData(spy, "", False, False)

    contracts = {}
    for s in setups:
        c = Option("SPY", today_ibkr, s["strike"], s["right"], "SMART", tradingClass="SPY")
        contracts[s["name"]] = c
    await ib.qualifyContractsAsync(*contracts.values())

    today_qty = contracts_for_today()
    print(f"Daily watch started {datetime.now(ET)}. Setups: {[s['name'] for s in setups]}. Qty/setup: {today_qty}")
    telegram(
        "Ashley-trigger daily watcher LIVE: "
        + ", ".join(f"{s['name']} @ SPY {s['lo']}-{s['hi']}" for s in setups)
        + f". Real IBKR orders (ladder-filled) on entry trigger AND on her exit messages. "
        + f"Qty/setup today: {today_qty}" + (" (one-day override)" if today_qty != 1 else ""),
        high_priority=False,
    )
    oversight_log(
        f"Ashley-trigger daily watcher started for today's {len(setups)} setups.",
        f"Scheduled daily run (CEO instruction 2026-08-27) -- {today_qty} contract(s) per setup, autonomous.",
    )

    # Restart recovery, built 2026-08-31: never start these blank without
    # checking. load_fired_today() carries forward anything already acted on
    # today (including a setup the CEO manually closed -- with no real
    # position left to find, it just stays correctly "done, don't re-enter").
    # reconcile_open_positions() independently checks REAL IBKR state for
    # anything still genuinely open (e.g. a crash mid-position) and restores
    # it with its real qty/avg cost so exit monitoring resumes correctly.
    fired = load_fired_today()
    open_positions = reconcile_open_positions(ib, setups, contracts)
    fired |= set(open_positions.keys())  # a real open position is definitely "fired"
    if fired or open_positions:
        save_fired_today(fired)
        txt = (f"Ashley-trigger restart recovery: already-fired today = {sorted(fired) or 'none'}, "
               f"real open positions restored = {sorted(open_positions.keys()) or 'none'}.")
        print(txt)
        telegram(txt, high_priority=False)
        oversight_log(txt, "Restart recovery from persisted fired-state + real IBKR reconciliation.")
    tick = 0
    force_close_attempted = False

    try:
        while True:
            now = datetime.now(ET)

            # Real forced-close safety window (see FORCE_CLOSE_ET comment) --
            # runs once, while the market is still open, before the
            # alert-only post-close check below even has a chance to matter.
            if (is_market_hours(now) and open_positions and not force_close_attempted
                    and (now.hour, now.minute) >= FORCE_CLOSE_ET):
                force_close_attempted = True
                for name, pos in list(open_positions.items()):
                    c = contracts[name]
                    txt = (f"Ashley {name}: no exit signal seen by {FORCE_CLOSE_ET[0]}:{FORCE_CLOSE_ET[1]:02d} ET -- "
                           f"attempting a real forced close now (IBKR has no broker-side auto-close).")
                    print(txt)
                    telegram(txt, high_priority=True)
                    td = ib.reqMktData(c, "", False, False)
                    await asyncio.sleep(3)
                    bid = td.bid if td.bid and td.bid > 0 else None
                    ask = td.ask if td.ask and td.ask > 0 else None
                    mid = (bid + ask) / 2 if bid and ask else None
                    ib.cancelMktData(c)
                    qty = pos.get("qty", 1)
                    ok, fill_px = await ibkr_place_leg_with_ladder_async(
                        ib, c, "SELL", f"ashley {name} forced close", qty, bid, ask, mid)
                    if ok:
                        open_positions.pop(name, None)
                        pnl = round((fill_px - pos["fill_px"]) * 100 * qty, 2)
                        msg = f"Ashley {name}: forced close FILLED @ ${fill_px} (pnl=${pnl})."
                        print(msg)
                        telegram(msg, high_priority=True)
                        oversight_log(msg, "Forced pre-close safety exit -- no exit signal seen in time.",
                                      outcome=f"forced_close @ {fill_px}, pnl={pnl}")
                        if pos.get("pos_id"):
                            try:
                                registry_close_position(pos["pos_id"], "forced_close_no_signal", pnl)
                            except Exception as exc:
                                print(f"registry_close_position failed (non-fatal): {exc}")
                    else:
                        msg = (f"Ashley {name}: forced close did NOT fill (bid={bid} ask={ask}) -- "
                               f"CHECK MANUALLY NOW, market closes soon.")
                        print(msg)
                        telegram(msg, high_priority=True)
                        oversight_log(msg, "Forced pre-close safety exit failed to fill.", outcome="not_filled")

            if not is_market_hours(now):
                if open_positions:
                    # REAL bug found 2026-09-03: her "stopped" exit message wasn't recognized
                    # (EXIT_PHRASES gap, since fixed), leaving a position this script thought
                    # was still open for 2+ hours after the broker had already force-closed it
                    # at 15:45 ET -- never reconciled, never recorded, dashboard just silently
                    # missing it. Before alerting "still open," check real broker state first:
                    # a missed signal (any future keyword gap, a Discord outage, etc.) can
                    # ALWAYS happen again, but the position not being tracked afterward
                    # shouldn't have to. Migrated to IBKR 2026-09-07 -- this account has no
                    # equivalent "auto-close at 15:45" broker behavior for options like Alpaca
                    # had, so a still-open 0DTE at this point is a real, unclosed position, not
                    # an expected broker action to reconcile against.
                    still_genuinely_open = []
                    for name, pos in list(open_positions.items()):
                        try:
                            c = contracts[name]
                            if ibkr_has_open_position(ib, c):
                                still_genuinely_open.append(name)
                                continue
                            fill = ibkr_get_last_closing_fill(ib, c)
                            if fill:
                                exit_px = float(fill.execution.price)
                                pnl = round((exit_px - pos["fill_px"]) * 100 * pos["qty"], 2)
                                if pos.get("pos_id"):
                                    try:
                                        registry_close_position(pos["pos_id"], "ibkr_close_recovered_eod", pnl)
                                    except Exception as exc:
                                        print(f"registry_close_position (EOD recovery) failed: {exc}")
                                msg = (f"Ashley {name}: no exit signal was seen, but the position was "
                                       f"already closed (real fill @ {exit_px}, pnl=${pnl}) -- recovered and "
                                       f"recorded, no manual action needed.")
                                print(msg)
                                telegram(msg, high_priority=False)
                                oversight_log(msg, "EOD reconciliation found a real close this "
                                              "script's own exit logic missed.",
                                              outcome=f"recovered @ {exit_px}, pnl={pnl}")
                            else:
                                still_genuinely_open.append(name)
                        except Exception as exc:
                            print(f"EOD reconciliation check failed for {name}: {exc}")
                            still_genuinely_open.append(name)
                    if still_genuinely_open:
                        telegram(
                            "Ashley-trigger daily watcher: market closed with "
                            + ", ".join(still_genuinely_open)
                            + " still OPEN despite the forced-close attempt -- these are 0DTE, "
                              "physically-settled SPY options: if ITM this risks real auto-exercise "
                              "(SPY shares Monday), not just expiring worthless. CHECK MANUALLY NOW.",
                            high_priority=True,
                        )
                        oversight_log("Daily watcher stopped at market close with open, unclosed positions: "
                                      + ", ".join(still_genuinely_open),
                                      "Forced-close attempt at 15:55 ET did not clear these -- "
                                      "physically-settled 0DTE options risk auto-exercise if ITM.",
                                      outcome="left_open_at_close")
                else:
                    telegram("Ashley-trigger daily watcher stopping: market closed, no open positions.",
                              high_priority=False)
                print("Outside market hours -- stopping today's watch.")
                break
            if len(fired) == len(setups) and not open_positions:
                print("All setups fired and closed -- stopping today's watch early.")
                break

            spot = spy_td.last or spy_td.close
            if spot and spot == spot:
                for s in setups:
                    if s["name"] in fired:
                        continue
                    if s["lo"] <= spot <= s["hi"]:
                        print(f"TRIGGER: {s['name']} -- SPY {spot} in zone [{s['lo']}, {s['hi']}]")
                        c = contracts[s["name"]]
                        td = ib.reqMktData(c, "", False, False)
                        await asyncio.sleep(3)
                        bid = td.bid if td.bid and td.bid > 0 else None
                        ask = td.ask if td.ask and td.ask > 0 else None
                        mid = (bid + ask) / 2 if bid and ask else None
                        ib.cancelMktData(c)

                        qty = contracts_for_today()
                        est_cost = (ask or bid or 0) * 100 * qty
                        try:
                            vals = {v.tag: v.value for v in ib.accountValues()}
                            avail = float(vals.get("AvailableFunds", 0.0))
                        except Exception:
                            avail = 0.0
                        if est_cost > 0 and avail > 0 and est_cost > avail:
                            msg = (f"Ashley trigger {s['name']}: SPY hit zone but est. cost ${est_cost:.2f} "
                                   f"(qty={qty}) exceeds available funds ${avail:.2f} -- SKIPPING, not firing.")
                            print(msg)
                            telegram(msg, high_priority=True)
                            oversight_log(msg, "Affordability gate blocked entry.", outcome="skipped")
                            fired.add(s["name"])
                            save_fired_today(fired)
                            continue

                        sym = occ_symbol(s["strike"], s["right"], today_yymmdd)
                        ok, fill_px = await ibkr_place_leg_with_ladder_async(
                            ib, c, "BUY", f"ashley {s['name']}", qty, bid, ask, mid)

                        fired.add(s["name"])
                        save_fired_today(fired)
                        if ok:
                            pos_id = f"ASHLEY_{s['name']}_{now.strftime('%Y%m%d_%H%M%S')}"
                            open_positions[s["name"]] = {
                                "symbol": sym, "fill_px": fill_px, "qty": qty,
                                "opened_at": now.isoformat(), "pos_id": pos_id,
                            }
                            # Macro calendar CAUTION only (added 2026-09-04, CEO-requested) --
                            # real motivating example was this exact position type (SPY 770C)
                            # stumbling on the real August-jobs-report morning. Deliberately
                            # does NOT skip/block: this script's whole job is following
                            # Ashley's own real-time call, and overriding her signal off a
                            # calendar would break that design. Same live FOMC/NFP/CPI/PPI
                            # calendar the butterflies/Day Trader/SPX 0DTE BLOCK on --
                            # here it's just recorded, for context if this trade underperforms.
                            macro_skip, macro_reason = is_macro_day()
                            macro_note = f" CAUTION: real {macro_reason} day." if macro_skip else ""
                            try:
                                registry_register_position(
                                    pos_id, "SPY", "ashley_signal",
                                    [{"leg": "long", "strike": s["strike"], "symbol": sym,
                                      "fill": fill_px, "qty": qty}],
                                    -fill_px, qty, round(fill_px * 100 * qty, 2), None, "15:45",
                                    now.isoformat(),
                                    notes=f"Ashley signal-follow, zone [{s['lo']},{s['hi']}], SPY spot {spot} at entry.{macro_note}",
                                )
                            except Exception as exc:
                                print(f"registry_register_position failed (non-fatal): {exc}")
                            msg = (f"Ashley trigger FILLED: bought {qty}x {s['name']} @ ${fill_px} "
                                   f"(SPY {spot}, zone [{s['lo']},{s['hi']}]). Now watching her "
                                   f"messages for the exit.{macro_note}")
                            print(msg)
                            telegram(msg, high_priority=True)
                            oversight_log(msg, "Real fill from Ashley-trigger daily pipeline.",
                                          outcome=f"filled @ {fill_px}")
                        else:
                            msg = (f"Ashley trigger {s['name']}: SPY hit zone (spot={spot}) but order did "
                                   f"NOT fill (bid={bid} ask={ask}) -- check IBKR manually.")
                            print(msg)
                            telegram(msg, high_priority=True)
                            oversight_log(msg, "Order placement attempted but not confirmed filled.",
                                          outcome="not_filled")

            tick += 1
            # Real bug found 2026-08-30 (2026-08-28 postmortem): this used to
            # only poll Discord "and open_positions" -- meaning it never read
            # her messages at all while flat. last_message_id sat frozen for
            # hours (once, overnight), and the moment a NEW position opened,
            # the very next poll dumped the whole backlog at once and fired
            # on the first exit-phrase match -- explaining two real 2026-08-28
            # trades that opened and closed 10-16 SECONDS apart on messages
            # that were actually hours (one, a full calendar day) old. Now
            # polls on the same cadence regardless of position state, so
            # last_message_id never falls behind and there's nothing stale
            # left to dump when a position does open.
            if tick % DISCORD_POLL_EVERY_N_TICKS == 0:
                try:
                    params = {"limit": 20}
                    if last_message_id:
                        params["after"] = last_message_id
                    r = requests.get(f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
                                      headers=headers, params=params, timeout=15)
                    if r.status_code == 429:
                        await asyncio.sleep(float(r.json().get("retry_after", 5)) + 1)
                    else:
                        r.raise_for_status()
                        msgs = r.json()
                        msgs.sort(key=lambda m: int(m["id"]))
                        for m in msgs:
                            last_message_id = m["id"]
                            if m.get("author", {}).get("username", "") != TARGET_USERNAME:
                                continue
                            content = (m.get("content") or "").strip()
                            if not content or not is_exit_signal(content):
                                continue
                            if not open_positions:
                                # She's talking about a trade we never entered
                                # (or already closed) -- not actionable, and
                                # not worth escalating either.
                                continue
                            msg_date = msg_date_et(m)
                            if msg_date != datetime.now(ET).date():
                                # Defense in depth: even with continuous
                                # polling above, never let a message from a
                                # prior trading day close a position opened
                                # today.
                                print(f"Ignoring stale exit signal from {msg_date} (not today): {content!r}")
                                continue

                            print(f"EXIT SIGNAL from Ashley: {content}")
                            if len(open_positions) == 1:
                                name = next(iter(open_positions))
                            else:
                                name = match_open_setup(content, open_positions, setups)
                            if name is None:
                                telegram(
                                    f"Ashley posted an exit signal (\"{content}\") but {len(open_positions)} "
                                    f"positions are open ({', '.join(open_positions.keys())}) and I can't tell "
                                    f"which one from her message -- NOT guessing. Close manually.",
                                    high_priority=True,
                                )
                                oversight_log(
                                    f"Ambiguous exit signal with {len(open_positions)} open positions: \"{content}\"",
                                    "Multiple positions open, message didn't clearly identify which -- "
                                    "escalated instead of guessing.",
                                    outcome="escalated_ambiguous",
                                )
                                continue

                            pos = open_positions.pop(name)
                            await close_position(ib, name, pos, setups, content)
                except Exception as exc:
                    print(f"Discord poll error: {exc}")

                # Real fix 2026-08-31: Ashley posted "773 C" then edited it to
                # the correct "763 C" ~3 minutes later. The poll above only
                # ever looks at messages AFTER last_message_id -- once the
                # alert itself has been consumed, nothing ever re-checks IT
                # specifically for an edit. This re-fetches that one message
                # each cycle and compares edited_timestamp; on a real change,
                # reconciles by NAME: a setup not yet fired gets updated/
                # added/removed safely; a setup that's already fired (order
                # placed, position open or not) is NEVER auto-touched --
                # that's a real, already-executed trade, and correcting it
                # automatically is a bigger, riskier action than this fix is
                # meant to take. Instead it escalates HIGH PRIORITY so a human
                # decides, exactly like the existing ambiguous-exit-signal path.
                if alert_message_id:
                    try:
                        msg = fetch_message(headers, alert_message_id)
                        if msg and msg.get("edited_timestamp") != alert_edited_ts:
                            alert_edited_ts = msg.get("edited_timestamp")
                            new_setups = setups_from_content(msg.get("content") or "")
                            old_by_name = {s["name"]: s for s in setups}
                            new_by_name = {s["name"]: s for s in new_setups}

                            for name, new_s in new_by_name.items():
                                old_s = old_by_name.get(name)
                                if old_s is None:
                                    setups.append(new_s)
                                    c = Option("SPY", today_ibkr, new_s["strike"], new_s["right"],
                                               "SMART", tradingClass="SPY")
                                    await ib.qualifyContractsAsync(c)
                                    contracts[name] = c
                                    txt = (f"Ashley EDITED her alert -- added new setup {name} @ "
                                           f"SPY {new_s['lo']}-{new_s['hi']}. Now watching it too.")
                                    print(txt); telegram(txt, high_priority=True)
                                    oversight_log(txt, "Detected via message edited_timestamp change.",
                                                  outcome="setup_added")
                                elif (old_s["lo"], old_s["hi"]) != (new_s["lo"], new_s["hi"]):
                                    if name in fired:
                                        txt = (f"\U0001F6A8 Ashley EDITED her alert for {name} (zone "
                                               f"{old_s['lo']}-{old_s['hi']} -> {new_s['lo']}-{new_s['hi']}) "
                                               f"but we ALREADY acted on the OLD version -- NOT "
                                               f"auto-correcting a live/already-placed trade. Check manually NOW.")
                                        print(txt); telegram(txt, high_priority=True)
                                        oversight_log(txt, "Edit conflict with an already-fired setup.",
                                                      outcome="edit_conflict_escalated")
                                    else:
                                        old_s["lo"], old_s["hi"] = new_s["lo"], new_s["hi"]
                                        txt = (f"Ashley EDITED her alert -- {name}'s zone changed to "
                                               f"{new_s['lo']}-{new_s['hi']}. Updated (not yet fired).")
                                        print(txt); telegram(txt, high_priority=True)
                                        oversight_log(txt, "Detected via message edited_timestamp change.",
                                                      outcome="zone_updated")

                            for name in list(old_by_name):
                                if name not in new_by_name:
                                    if name in fired:
                                        txt = (f"\U0001F6A8 Ashley's edit REMOVED setup {name} but we "
                                               f"ALREADY acted on it -- NOT touching the live/already-placed "
                                               f"trade. Check manually.")
                                        print(txt); telegram(txt, high_priority=True)
                                        oversight_log(txt, "Edit conflict: setup removed but already fired.",
                                                      outcome="edit_conflict_escalated")
                                    else:
                                        setups[:] = [s for s in setups if s["name"] != name]
                                        txt = (f"Ashley's edit REMOVED setup {name} (not yet fired) -- "
                                               f"no longer watching it.")
                                        print(txt); telegram(txt, high_priority=False)
                                        oversight_log(txt, "Detected via message edited_timestamp change.",
                                                      outcome="setup_removed")
                    except Exception as exc:
                        print(f"Edit-check error: {exc}")

            await asyncio.sleep(5)
    finally:
        try:
            ib.cancelMktData(spy)
        except Exception:
            pass
        ib.disconnect()

    return last_message_id


async def main():
    print(f"Ashley-trigger DAILY executor started {datetime.now(ET)}")
    cfg = load_config()
    global TELEGRAM_TOKEN, TELEGRAM_CHAT
    TELEGRAM_TOKEN = cfg["telegram_token"]
    TELEGRAM_CHAT = cfg["telegram_chat_id"]
    bot_token = cfg["discord_bot_token"]
    headers = {"Authorization": f"Bot {bot_token}"}

    state = load_state()
    if state.get("last_message_id") is None:
        r = requests.get(f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
                          headers=headers, params={"limit": 1}, timeout=15)
        r.raise_for_status()
        msgs = r.json()
        if msgs:
            state["last_message_id"] = msgs[0]["id"]
            save_state(state)

    while True:
        now = datetime.now(ET)
        today_iso = now.date().isoformat()

        if now.weekday() >= 5:
            await asyncio.sleep(DAY_POLL_S * 20)
            continue

        if state.get("last_processed_date") == today_iso:
            # Already ran today (or already gave up for today) -- idle until tomorrow.
            await asyncio.sleep(DAY_POLL_S * 20)
            continue

        if now < now.replace(hour=9, minute=30, second=0, microsecond=0):
            await asyncio.sleep(DAY_POLL_S)
            continue

        print(f"New trading day {today_iso} -- waiting for ashleyklieu's daily alert...")
        setups, last_message_id, alert_edited_ts = wait_for_daily_setups(headers, state)
        alert_message_id = last_message_id  # see wait_for_daily_setups docstring -- same ID
        state["last_message_id"] = last_message_id
        save_state(state)

        if not setups:
            msg = f"Ashley-trigger daily watcher: no setups posted by {WAIT_CUTOFF_ET[0]}:{WAIT_CUTOFF_ET[1]:02d} ET today -- skipping {today_iso}."
            print(msg)
            telegram(msg, high_priority=False)
            oversight_log(msg, "No qualifying Discord alert found within the daily wait window.",
                          outcome="skipped_no_alert")
            state["last_processed_date"] = today_iso
            save_state(state)
            continue

        if len(setups) > 5:
            telegram(
                f"Ashley-trigger daily watcher: today's alert parsed {len(setups)} setups "
                f"(unusually many, normal is 2-3) -- proceeding, but check the raw message if this looks wrong.",
                high_priority=True,
            )

        try:
            last_message_id = await run_daily_watch(
                setups, headers, state.get("last_message_id"), alert_message_id, alert_edited_ts)
            state["last_message_id"] = last_message_id
        except Exception as exc:
            print(f"Daily watch crashed: {exc}")
            telegram(f"Ashley-trigger daily watcher crashed today: {exc} -- check manually, no more entries today.",
                      high_priority=True)
            oversight_log(f"Daily watch crashed: {exc}", "Unhandled exception mid-day.", outcome="crashed")

        state["last_processed_date"] = today_iso
        save_state(state)


if __name__ == "__main__":
    asyncio.run(main())
