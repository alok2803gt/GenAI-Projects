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
     contract) via the favorable->mid->aggressive ladder
     (ibkr_place_leg_with_ladder_async, ibkr_0dte_common.py) proven
     2026-08-26 on Alpaca -- no flat, no-retry price. WHICH contract depends
     on when the zone triggers (CEO relayed 2026-10-14, direct from Ashley):
     before noon ET, same-day 0DTE; at/after noon, the 3rd real trading day
     out (dte_kind_for(), AFTERNOON_CUTOFF_HOUR_ET) -- she only plays 0DTE in
     the morning, and an afternoon trade is always 3DTE on her side too.
  3. EXIT: polls the same Discord channel for her follow-up messages. A
     clear, past-tense exit phrase ("took profit", "stopped out", "closed",
     "i'm out", etc.) on a setup this run currently holds triggers a real
     sell via the same ladder mechanism. Conservative on ambiguity:
     conditional/hypothetical phrasing ("if stopped...") is never treated
     as a fill trigger, and an exit signal with >1 position open and no
     clear match escalates via Telegram instead of guessing.
  4. FORCE-CLOSE SAFETY WINDOW at 15:55 ET: any OPEN POSITION WHOSE CONTRACT
     ACTUALLY EXPIRES TODAY (today's 0DTE, or a 3DTE position whose 3rd
     trading day has arrived) with no exit signal seen gets a real forced
     close attempted here, before market close; if it somehow still didn't
     clear, flags it HIGH PRIORITY. A 3DTE position NOT yet at its expiry is
     deliberately left alone here -- it is expected to carry past today's
     close, persisted in OPEN_POSITIONS_FILE (not day-scoped) and picked back
     up by reconcile_all_open_positions() on the next run, same day or a
     later one, until she signals an exit or its own expiry day arrives.
     Either way the day's loop then ends and goes back to wait for the NEXT
     trading day.

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
import time
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
WATCH_START_ET = (9, 0)          # start WATCHING Discord for her post at 9:00 ET (CEO, 2026-09-23:
                                 # she posts "sometime before 9:30", so the watcher must be up before
                                 # the open; parsing a post is harmless pre-market)
EXEC_START_ET = (9, 30)          # but place NO orders before the real open -- options do not trade
                                 # pre-market, so the entry/exit watch waits here until 9:30 ET
WAIT_CUTOFF_ET = (13, 0)         # give up waiting for today's alert past 1:00 PM ET -- loosened
                                  # from 11:00 (CEO instruction 2026-08-27) after she posted her
                                  # daily setups at 12:20 PM ET that day, past the old cutoff,
                                  # and the run skipped a real, tradeable day for it. 1:00 PM still
                                  # leaves ~3 hours before the 4:00 PM close for a 0DTE entry+exit.
DAY_POLL_S = 30                  # how often the outer daily loop checks in
CRASH_RETRY_LIMIT = 3           # same-day budget for auto-retrying after run_daily_watch() throws
                                 # (e.g. a transient TWS-relogin race) before giving up on today --
                                 # see the crash handler in main(), real incident 2026-10-12
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

# LATE-ENTRY GUARD (added 2026-10-01 after a real loss). A new position opened
# minutes before FORCE_CLOSE_ET has no time for the thesis to work -- it only
# pays the spread twice plus two commissions. On 2026-09-30 the executor bought
# the 764C at 15:54:44 and the forced close sold it at 15:55:05: a TWENTY-ONE
# SECOND hold, 0.89 -> 0.66, -$24.78 realised, of which $1.78 was commission.
# There was never a path to a profit.
#
# Expressed as minutes BEFORE the force close rather than an absolute clock
# time, so changing FORCE_CLOSE_ET cannot leave this guard stranded at a stale
# hour. 25 minutes is the CEO's call (2026-10-01), i.e. no new entries after
# 15:30 while the force close is 15:55.
MIN_MINUTES_BEFORE_FORCE_CLOSE = 25

# Trailing stop on Ashley's entries (decided 2026-10-06). Sell when the live bid
# falls this far below the highest bid seen since entry. Replayed on her 11 logged
# closed trades (09-14..10-02): 20% turned -$268 into -$28 but stopped out two
# winners; 30% gave -$56 with one winner cut. In-sample, n=11 -- verify forward.
TRAIL_STOP_PCT = 0.30

# Morning/afternoon DTE rule (CEO relayed 2026-10-14, direct from Ashley herself):
# she only plays 0DTE during morning hours; an afternoon trade is always the 3rd
# trading day out. Noon ET is the CEO's chosen cutoff. This is about WHEN A ZONE
# ACTUALLY TRIGGERS, not a separate afternoon alert -- she posts one setup list
# pre-market and the bot watches it all day, so the same zone list can produce a
# 0DTE fill at 9:40 and a 3DTE fill at 1pm depending purely on when SPY gets there.
AFTERNOON_CUTOFF_HOUR_ET = 12


def dte_kind_for(now_et: datetime) -> str:
    """"0dte" before noon ET, "3dte" at/after -- see AFTERNOON_CUTOFF_HOUR_ET."""
    return "0dte" if now_et.hour < AFTERNOON_CUTOFF_HOUR_ET else "3dte"


# Full-day NYSE closures (2026-2027; weekend holidays use the exchange's observed
# date -- Sat -> prior Fri, Sun -> next Mon). Copied verbatim from the same table
# in harami_daily_trader.py rather than imported -- this script's own convention
# (see its other state files) is to stay a self-contained single file with no
# cross-strategy imports, so a change to harami's trading calendar can never
# silently affect Ashley's. Keep both copies in sync by hand; extend before 2028.
NYSE_HOLIDAYS = {date.fromisoformat(s) for s in (
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
)}
HOLIDAY_LIST_LAST_YEAR = 2027


def _is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in NYSE_HOLIDAYS


def _add_trading_days(d: date, n: int) -> date:
    """Skips weekends and NYSE_HOLIDAYS, so the result is always a real trading
    day (within the years that table covers -- see HOLIDAY_LIST_LAST_YEAR)."""
    cur = d
    added = 0
    while added < n:
        cur += timedelta(days=1)
        if _is_trading_day(cur):
            added += 1
    return cur


def trail_stop_hit(peak_bid: float, bid: float | None, pct: float = TRAIL_STOP_PCT) -> bool:
    """True when the live bid has fallen pct below the highest bid since entry."""
    return bid is not None and bid > 0 and bid <= (1 - pct) * peak_bid

# Same pattern as ashleyklieu_alert_monitor.py -- reused verbatim so both
# scripts parse her messages identically.
ENTRY_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*([CP])\b[^0-9]{0,15}Entry:?\s*(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# REAL INCIDENT 2026-09-29: she took profit on 765C at 10:51 ("765 TP"), then
# at 10:52 marked the level done by STRIKING IT THROUGH in a NEW message
# (~~765 C | Entry: 764.2 - 764.5 ~~), and at 11:06 posted "Next two levels in
# play are here". This executor only re-parses EDITS to the original daily
# alert, so it never learned the level was retired -- and at 11:00:52 it bought
# the 765C into a setup she had closed nine minutes earlier. The exit logic
# could not save it either: at 10:51 there was no open position, so her "765 TP"
# was correctly ignored. Retirement has to be handled on the ENTRY side.
#
# Discord strikethrough is ~~text~~. Any setup named inside a struck-through
# span is treated as retired for the rest of the day.
STRIKETHROUGH_RE = re.compile(r"~~(.+?)~~", re.DOTALL)


def retired_setups_from(content: str, setups: list[dict]) -> set:
    """Names of setups Ashley has struck through in this message."""
    out = set()
    for span in STRIKETHROUGH_RE.findall(content or ""):
        for m in ENTRY_PATTERN.finditer(span):
            strike, right = m.group(1), m.group(2).upper()
            for s in setups:
                if abs(float(s["strike"]) - float(strike)) < 1e-6 and s["right"] == right:
                    out.add(s["name"])
    return out


def retired_names_needing_exit(retired: set, open_positions: dict) -> list:
    """Of the names she struck through in THIS message, which ones are
    currently open positions that need a real close -- not just a "don't
    enter" flag? Real incident 2026-09-29 (765C): 5 minutes after the fill
    she posted new setups with 765C struck through, right next to "Next
    two levels in play are here" -- as clear an "I'm done with this one"
    as the strikethrough convention gets. The retirement handling at the
    call site only ever updates `fired` for names NOT already fired (its
    whole job is stopping a dead level from being entered), so an already-
    open position's strikethrough was silently doing nothing. Strikethrough
    on an open position is unconditionally treated as her exit, same as
    the not-yet-fired case is unconditionally treated as retirement -- no
    extra phrase-matching gate, since striking through a level you still
    want to hold would defeat the whole point of the convention."""
    return [n for n in retired if n in open_positions]


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

# REAL INCIDENT 2026-10-01: she posted "2nd Play invalid, I wasn't in this trade
# and this happened too fast. SPY dumping to 760" FOUR MINUTES after our fill on
# the 762C. is_exit_signal() returned False -- it only ever recognised
# profit-taking and stop language -- so the executor logged "watching her
# messages for the exit" and waited for a signal that was never coming. The
# position rode to the 15:55 clock and had to be closed by hand.
#
# This is the single biggest money leak in the strategy. Of 15 round trips to
# 2026-10-01, the ones she actually called an exit on net +$31.80, while the six
# closed only by the 15:55 clock net -$286.18. Four of those six were positions
# she had effectively walked away from.
#
# ABANDONMENT means "she is not in this trade": there will never be a TP or a
# stop, so the only question is how long we hold an unmanaged 0DTE. Treated as
# an exit, with its own kind so the log says why we closed.
ABANDON_PHRASES = (
    "invalid", "no longer valid", "wasn't in this trade", "wasnt in this trade",
    "not in this trade", "wasn't in that trade", "wasnt in that trade",
    "didn't take this", "didnt take this", "didn't take that", "didnt take that",
    "scratch that", "disregard", "cancel that", "nvm", "never mind",
)
# Her completion notices. "First play is done ✅" (2026-10-01 09:34) is her
# saying the trade is finished -- if we hold it, we should be flat too.
DONE_PHRASES = ("play is done", "plays are done", "trade is done", "that's it for",
                "thats it for", "done for the day", "wrapping up")

# REAL BUG found 2026-10-02: at 09:52:31 she posted "I got small limit filled at
# 772.10, TP 771.3" -- an ENTRY announcement carrying the target she intends to
# exit at. TP_WORD matched, so is_exit_signal() returned True. No position was
# open so it was harmlessly ignored, but with a position open that message would
# have closed it 15 seconds after she opened hers. The mirror image of the
# abandonment bug: this one invents exits that were never called.
#
# So in a message where she is describing GETTING IN, a bare "TP" is a stated
# plan, not a fill, and only explicit past-tense exit language counts.
ENTRY_MARKERS = (
    "got filled", "limit filled", "got in at", "got in @", "i'm in", "im in",
    "entered at", "entered @", "adding", "starter", "i got small", "i got a small",
    "bought", "buying", "scaling in", "size in", "sized in",
)
# Unambiguously past-tense: safe even inside an entry announcement.
HARD_EXIT_PHRASES = (
    "tp hit", "took profit", "stopped out", "i'm out", "im out", "flat now",
    "out now", "banked", "closed my", "closed the", "locked in profit",
)

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


OPEN_POSITIONS_FILE = "ashleyklieu_open_positions.json"
# Deliberately NOT day-scoped (unlike FIRED_STATE_FILE/the old per-day trail-peak
# file it replaces) -- an afternoon (3DTE) position bought today must still be
# found here tomorrow and the day after. Holds every currently-open Ashley
# position, 0DTE or 3DTE alike, keyed by setup name, each record carrying its
# own frozen "expiry" (YYYYMMDD) so a position is always closed against the
# SAME contract it was bought on, never today's date. Built 2026-10-14 for the
# 3DTE-afternoon-trade feature; replaces the old TRAIL_STATE_FILE (peak_bid now
# just lives on each record here) and the old reconcile_open_positions(), which
# only ever checked today's 0DTE contracts and so could never find -- or even
# correctly re-close -- a position still open from a prior afternoon.


def load_open_positions() -> dict:
    try:
        with open(OPEN_POSITIONS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_open_positions(open_positions: dict) -> None:
    try:
        with open(OPEN_POSITIONS_FILE, "w") as f:
            json.dump(open_positions, f, indent=2)
    except OSError as exc:
        print(f"save_open_positions failed (non-fatal): {exc}")


async def reconcile_all_open_positions(ib, setups: list[dict], fired: set) -> dict:
    """Real IBKR position check against EVERY persisted open position -- today's
    0DTE ones AND any 3DTE position carried over from a prior afternoon -- not
    trusting local bookkeeping, which is exactly what a restart (or a new day)
    loses. For each record in OPEN_POSITIONS_FILE, rebuilds its own frozen-expiry
    contract, qualifies it, and checks real ib.positions(): a genuinely still-open
    one is restored with its REAL qty (never guessed) but this script's OWN
    fill_px/pos_id/opened_at/trail/peak_bid (more precise than re-deriving from
    IBKR's avgCost, since we wrote those ourselves at the moment of entry). One
    that's gone (closed some other way -- manually, exercised, expired) is
    dropped silently here: its real economics were already captured by
    commission_ledger's own snapshot, and no P&L is guessed.

    Mutates `setups` (adds a pseudo-setup for any restored name not already in
    today's fresh list, so every strike/right lookup elsewhere keeps working)
    and `fired` (so the entry loop never tries to re-enter something already
    open). Returns the restored open_positions dict, with "contract" NOT
    included (callers must re-derive/qualify as needed via each record's own
    strike/right/expiry) -- actually returns alongside a parallel contracts
    dict; see call site.
    """
    persisted = load_open_positions()
    if not persisted:
        return {}, {}
    to_qualify = {}
    for name, rec in persisted.items():
        to_qualify[name] = Option("SPY", rec["expiry"], rec["strike"], rec["right"],
                                   "SMART", tradingClass="SPY")
    try:
        await ib.qualifyContractsAsync(*to_qualify.values())
    except Exception as exc:
        print(f"reconcile_all_open_positions: could not qualify persisted contracts: {exc}")
        return {}, {}
    try:
        real_positions = {p.contract.conId: p for p in ib.positions() if p.position != 0}
    except Exception as exc:
        print(f"reconcile_all_open_positions: could not fetch real IBKR positions: {exc}")
        return {}, {}

    restored, restored_contracts = {}, {}
    existing_names = {s["name"] for s in setups}
    for name, rec in persisted.items():
        c = to_qualify[name]
        rp = real_positions.get(c.conId)
        if rp is None:
            print(f"reconcile_all_open_positions: {name} (exp {rec['expiry']}) no longer a real "
                  f"IBKR position -- dropping from {OPEN_POSITIONS_FILE}, not carrying forward.")
            continue
        restored[name] = {**rec, "qty": int(abs(rp.position))}
        restored_contracts[name] = c
        if name not in existing_names:
            setups.append({"name": name, "strike": rec["strike"], "right": rec["right"],
                           "lo": rec["strike"], "hi": rec["strike"]})
            existing_names.add(name)
        fired.add(name)
    return restored, restored_contracts


def telegram(msg, high_priority=True):
    try:
        from telegram_alert_gate import alert_enabled
        if not alert_enabled("ashley_signals"):
            return
    except Exception:
        pass
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


def exit_signal_kind(content: str) -> str | None:
    """Why we should close, or None to keep holding.

    Returns one of:
      "tp"        -- she took profit / hit her target
      "stop"      -- she was stopped out
      "abandoned" -- she is NOT in this trade (invalid, never filled, scratched)
      "done"      -- she has declared the play finished

    "abandoned" exists because of the 2026-10-01 762C: silence from her is
    indistinguishable from an unmanaged position, and waiting for a TP that will
    never come is how the forced-close bucket lost $286.18 across six trades.
    """
    lower = content.strip().lower()
    if lower.startswith(CONDITIONAL_PREFIXES) or " if " in lower[:40]:
        return None

    hypothetical = any(h in lower for h in HYPOTHETICAL_MARKERS)

    # Abandonment is checked FIRST and is not subject to the entry-marker
    # suppression below: "2nd Play invalid, I wasn't in this trade" is itself a
    # statement about not being filled, so an entry word appearing in it must
    # not mute it.
    #
    # Nor is it subject to the hypothetical-marker suppression -- fixed
    # 2026-10-19 after real incident 2026-10-15 (770P): she posted "Not in
    # this trade, I'd take profit now", and "i'd" tripped `hypothetical`,
    # which silently swallowed the clearly-stated, non-hypothetical
    # abandonment clause ("Not in this trade") earlier in the SAME message.
    # The hypothetical gate exists for AMBIGUOUS_EXIT_PHRASES ("take profit"/
    # "took profit"), which really do double as her casual commentary style
    # (the 2026-08-30 incident this gate was built for). Abandonment
    # language is never that -- she doesn't muse hypothetically about not
    # being in a trade -- so it is checked ungated, same as it already is
    # against the entry-marker suppression just above.
    if any(p in lower for p in ABANDON_PHRASES):
        return "abandoned"
    if any(p in lower for p in DONE_PHRASES) and not hypothetical:
        return "done"

    # She often announces an ENTRY together with the target she will exit at
    # ("I got small limit filled at 772.10, TP 771.3"). That TP is a plan, not a
    # fill, so inside an entry announcement only past-tense exit language counts.
    if any(k in lower for k in ENTRY_MARKERS):
        if any(p in lower for p in HARD_EXIT_PHRASES) and not hypothetical:
            return "stop" if "stopped" in lower else "tp"
        return None

    # Her own explicit, deliberate alert convention (confirmed directly by her,
    # 2026-08-31): sharing "TP" IS the take-profit alert.
    if TP_WORD.search(lower):
        return "tp"
    # "take profit"/"took profit" overlap with her casual commentary about
    # profit-taking philosophy -- a hypothetical marker anywhere in the message
    # means it's her opinion, not a trade she's making.
    if any(p in lower for p in AMBIGUOUS_EXIT_PHRASES):
        return None if hypothetical else "tp"
    if any(p in lower for p in EXIT_PHRASES if p not in AMBIGUOUS_EXIT_PHRASES):
        return "stop" if "stopped" in lower else "tp"
    return None


def is_exit_signal(content: str) -> bool:
    """Back-compat wrapper -- any kind of close signal."""
    return exit_signal_kind(content) is not None


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


def recover_todays_alert(headers, state, today_iso):
    """Restart recovery for a mid-day restart. Real bug found 2026-09-18/20:
    state["last_message_id"] is the alert's OWN id while the day's watch is
    running, and wait_for_daily_setups() polls `after=last_message_id` -- so it
    can never see today's already-consumed alert again. A restart mid-day (with
    a position open) would sit there waiting for a NEW alert, never reach
    run_daily_watch(), and so never reach reconcile_open_positions() -- the open
    position would go unmonitored. Fix: main() persists the alert's id + parsed
    setups + date when it first consumes it; here we re-fetch that exact message
    by id (picks up an edit made while we were down), falling back to the cached
    setups if Discord is unreachable. Returns (setups, edited_ts) or None if
    there is nothing to recover for today."""
    if state.get("alert_date") != today_iso or not state.get("alert_message_id"):
        return None
    cached = state.get("alert_setups") or []
    msg = None
    for attempt in range(4):
        msg = fetch_message(headers, state["alert_message_id"])
        if msg:
            break
        import time as _t
        _t.sleep(5)
    if msg:
        fresh = setups_from_content((msg.get("content") or "").strip())
        if fresh:
            return fresh, msg.get("edited_timestamp")
    if cached:
        print("Restart recovery: could not re-fetch today's alert from Discord -- using the cached setups.")
        return cached, state.get("alert_edited_ts")
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
    # The position's OWN actual expiry -- never date.today(). Real bug fixed
    # 2026-10-14 alongside the 3DTE-afternoon feature: this used to always
    # rebuild today's 0DTE contract here, which happened to be correct only
    # because nothing but same-day 0DTE existed yet. A 3DTE (or carried-over)
    # position closed this way would have silently tried to sell the WRONG
    # contract (today's unrelated 0DTE strike) instead of the one actually held.
    expiry = pos.get("expiry") or date.today().strftime("%Y%m%d")
    c = Option("SPY", expiry, strike, right, "SMART", tradingClass="SPY")
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


def seconds_until(hh, mm, now=None):
    """Seconds from `now` until hh:mm ET today (0 if already past)."""
    now = now or datetime.now(ET)
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return max((target - now).total_seconds(), 0.0)


async def run_daily_watch(setups, headers, last_message_id, alert_message_id=None, alert_edited_ts=None):
    # Her post can land before the open (watcher starts at WATCH_START_ET), but
    # nothing tradable exists until 9:30 -- hold the entry/exit machinery here
    # rather than connecting to IBKR and pricing options pre-market.
    wait_s = seconds_until(*EXEC_START_ET)
    if wait_s > 0:
        msg = (f"Ashley setups parsed pre-market ({[s['name'] for s in setups]}); "
               f"execution monitoring starts at {EXEC_START_ET[0]}:{EXEC_START_ET[1]:02d} ET "
               f"(in {wait_s / 60:.0f} min).")
        print(msg)
        telegram(msg, high_priority=False)
        await asyncio.sleep(wait_s)
    cfg = load_config()
    global TELEGRAM_TOKEN, TELEGRAM_CHAT
    TELEGRAM_TOKEN = cfg["telegram_token"]
    TELEGRAM_CHAT = cfg["telegram_chat_id"]

    today_ibkr  = date.today().strftime("%Y%m%d")
    today_yymmdd = date.today().strftime("%y%m%d")
    # The 3rd real trading day out, computed once from TODAY's date -- DTE counts
    # in days, not hours, so this is the same value regardless of what time this
    # afternoon a zone actually triggers (CEO/Ashley rule, 2026-10-14).
    third_ibkr   = _add_trading_days(date.today(), 3).strftime("%Y%m%d")
    third_yymmdd = _add_trading_days(date.today(), 3).strftime("%y%m%d")

    ib = IB()
    # Retry the initial connect for up to 10 min -- real failure mode seen
    # 2026-09-15: Ashley's Discord alert was found fine right at 9:30, but
    # TWS's API port wasn't listening yet for a few minutes after market
    # open, and this was the one connect() in the whole live-trading fleet
    # with no retry around it. Day Trader / SPY Weekly Condor both already
    # retry their own connect and recovered on their own that same morning;
    # this brings Ashley in line. Safe to retry freely here specifically --
    # nothing below this point has run yet, so a retry can't double an order.
    _connect_deadline = time.monotonic() + 600
    while True:
        try:
            await ib.connectAsync("127.0.0.1", 7496, clientId=994, timeout=15)
            break
        except Exception as exc:
            if time.monotonic() >= _connect_deadline:
                raise
            print(f"IBKR connect failed, retrying: {exc}")
            await asyncio.sleep(15)

    # Per-execution commission capture (added 2026-09-28). The 0DTE legs here
    # are the account's most fee-heavy: ~$0.90 per leg on ~$120 of premium is
    # roughly a 1.6% round-trip handicap, which only shows up if it is recorded.
    try:
        import commission_ledger
        commission_ledger.attach(ib, strategy="ashley_0dte")
        await commission_ledger.snapshot_async(ib, strategy="ashley_0dte")
    except Exception as exc:
        print(f"commission_ledger setup failed (non-fatal): {exc}")

    spy = Stock("SPY", "SMART", "USD")
    await ib.qualifyContractsAsync(spy)
    spy_td = ib.reqMktData(spy, "", False, False)

    # Two candidate contracts per not-yet-fired setup -- which one actually gets
    # bought is decided at trigger time by dte_kind_for(now), since a zone from
    # the SAME pre-market alert can trigger at 9:40 (0DTE) or 2pm (3DTE). Both
    # are qualified upfront here so the hot entry path never has to await a
    # qualify call. `contracts` (singular) holds the ONE real contract for any
    # name that's already open (today's reconciled 0DTE, a carried 3DTE, or one
    # that fires during this run) -- everything downstream of entry (force-close,
    # trailing stop, exits) looks it up there, never in contracts0/contracts3.
    contracts0, contracts3 = {}, {}
    for s in setups:
        contracts0[s["name"]] = Option("SPY", today_ibkr, s["strike"], s["right"],
                                        "SMART", tradingClass="SPY")
        contracts3[s["name"]] = Option("SPY", third_ibkr, s["strike"], s["right"],
                                        "SMART", tradingClass="SPY")
    try:
        await ib.qualifyContractsAsync(*contracts0.values(), *contracts3.values())
    except Exception as exc:
        # A single bad contract in the batch can fail the whole qualify call --
        # fall back to qualifying 0DTE and 3DTE separately so a problem with the
        # (less certain to be listed) 3rd-trading-day expiry can't also take
        # down today's ordinary 0DTE setups.
        print(f"Batch contract qualification failed ({exc}); retrying 0DTE and 3DTE separately.")
        await ib.qualifyContractsAsync(*contracts0.values())
        try:
            await ib.qualifyContractsAsync(*contracts3.values())
        except Exception as exc3:
            print(f"3DTE contract qualification failed: {exc3} -- afternoon entries today will be "
                  f"SKIPPED (not silently bought as 0DTE) until this resolves.")
            contracts3.clear()
    contracts = dict(contracts0)  # today's 0DTE is the default "real" contract for a fresh setup

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

    # Restart recovery, built 2026-08-31, generalized 2026-10-14 for 3DTE.
    # load_fired_today() carries forward anything already acted on today
    # (including a setup the CEO manually closed -- with no real position left
    # to find, it just stays correctly "done, don't re-enter"). It is day-scoped
    # on purpose (yesterday's 0DTE strikes don't exist today) and is ONLY about
    # "don't re-enter" -- it is not where position identity lives any more.
    #
    # reconcile_all_open_positions() independently checks REAL IBKR state
    # against OPEN_POSITIONS_FILE (NOT day-scoped) for anything still
    # genuinely open -- today's 0DTE reconciled after a same-day crash, OR a
    # 3DTE position carried over from a prior afternoon -- and restores each
    # with its real qty (from IBKR) plus this script's own fill_px/pos_id/
    # opened_at/trail/peak_bid (from the file), so exit-signal monitoring,
    # the trailing stop, and eventual force-close on its real expiry day all
    # resume correctly regardless of which day it was actually opened on.
    fired = load_fired_today()
    open_positions, restored_contracts = await reconcile_all_open_positions(ib, setups, fired)
    contracts.update(restored_contracts)
    _late_skips: set = set()
    trail_tickers: dict = {}  # live market-data tickers for trailing-stop positions
    last_open_positions_snapshot: dict | None = None  # last state written to OPEN_POSITIONS_FILE
    fired |= set(open_positions.keys())  # a real open position is definitely "fired"
    if fired or open_positions:
        save_fired_today(fired)
        save_open_positions(open_positions)
        last_open_positions_snapshot = dict(open_positions)
        txt = (f"Ashley-trigger restart recovery: already-fired today = {sorted(fired) or 'none'}, "
               f"real open positions restored = {sorted(open_positions.keys()) or 'none'}"
               + (f" (carried from a prior day: "
                  f"{sorted(n for n in open_positions if open_positions[n].get('expiry') != today_ibkr)})"
                  if any(p.get("expiry") != today_ibkr for p in open_positions.values()) else "") + ".")
        print(txt)
        telegram(txt, high_priority=False)
        oversight_log(txt, "Restart recovery from persisted fired-state + real IBKR reconciliation.")
    tick = 0
    force_close_attempted = False

    try:
        while True:
            now = datetime.now(ET)

            # Trailing stop (see TRAIL_STOP_PCT). Only positions opened after this change
            # carry "trail": True, so anything already open keeps following her exit.
            for name in list(trail_tickers):
                if name not in open_positions:
                    ib.cancelMktData(contracts[name])
                    trail_tickers.pop(name, None)
            if is_market_hours(now):
                for name, pos in list(open_positions.items()):
                    if not pos.get("trail"):
                        continue
                    if name not in trail_tickers:
                        trail_tickers[name] = ib.reqMktData(contracts[name], "", False, False)
                        continue
                    tk = trail_tickers[name]
                    bid = tk.bid if tk.bid and tk.bid > 0 else None
                    if bid is not None and bid > pos["peak_bid"]:
                        pos["peak_bid"] = bid
                    if trail_stop_hit(pos["peak_bid"], bid):
                        ib.cancelMktData(contracts[name])
                        trail_tickers.pop(name, None)
                        msg = (f"Ashley {name}: trailing stop hit -- bid ${bid} is "
                               f"{TRAIL_STOP_PCT:.0%} below peak bid ${pos['peak_bid']}. Selling.")
                        print(msg)
                        telegram(msg, high_priority=True)
                        # Real bug fixed 2026-10-16: this used to pop() the position
                        # out of open_positions UNCONDITIONALLY, before even checking
                        # whether close_position() actually got a fill. A failed sell
                        # (real incident 2026-10-15, 770P: 3 ladder steps, none filled)
                        # would then silently vanish from tracking -- no force-close
                        # at 15:55, no EOD alarm -- while the real IBKR position sat
                        # open and unmanaged. Only drop it from tracking on a
                        # CONFIRMED fill; an unfilled attempt leaves it in
                        # open_positions so force-close/the EOD check still see it.
                        ok = await close_position(ib, name, pos, setups,
                                                   f"trailing stop {TRAIL_STOP_PCT:.0%} from peak ${pos['peak_bid']}")
                        if ok:
                            open_positions.pop(name, None)

            # Persist open positions (incl. trailing peaks) whenever anything
            # changes (a peak moves, a trailed position closes), so a restart
            # resumes correctly instead of dropping state. Replaces the old
            # day-scoped TRAIL_STATE_FILE -- peak_bid now lives on each
            # OPEN_POSITIONS_FILE record, which survives across days too.
            if open_positions != last_open_positions_snapshot:
                save_open_positions(open_positions)
                last_open_positions_snapshot = dict(open_positions)

            # Real forced-close safety window (see FORCE_CLOSE_ET comment) --
            # runs once, while the market is still open, before the
            # alert-only post-close check below even has a chance to matter.
            # Scoped to positions whose expiry is ACTUALLY TODAY -- a 3DTE
            # position bought this afternoon, or one carried in from a prior
            # afternoon that isn't expiring today, must NOT be force-closed
            # here; it keeps right on being watched for her exit signal.
            expiring_today = {n: p for n, p in open_positions.items() if p.get("expiry", today_ibkr) == today_ibkr}
            if (is_market_hours(now) and expiring_today and not force_close_attempted
                    and (now.hour, now.minute) >= FORCE_CLOSE_ET):
                force_close_attempted = True
                for name, pos in list(expiring_today.items()):
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
                # Split by whether this position's contract actually expires TODAY
                # (today's 0DTE, or a carried-in position whose 3rd-trading-day
                # expiry happens to BE today) vs genuinely carrying past today's
                # close (a fresh or still-carried 3DTE position with real life
                # left). Only the former is subject to the auto-exercise alarm
                # below -- the latter is exactly the point of the 3DTE feature
                # and is expected to still be open here.
                expiring_today_open = {n: p for n, p in open_positions.items()
                                       if p.get("expiry", today_ibkr) == today_ibkr}
                carried_forward = {n: p for n, p in open_positions.items()
                                   if p.get("expiry", today_ibkr) != today_ibkr}
                if expiring_today_open:
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
                    for name, pos in list(expiring_today_open.items()):
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
                                open_positions.pop(name, None)
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
                if carried_forward:
                    txt = ("Ashley-trigger daily watcher stopping for today: carrying "
                           + ", ".join(f"{n} (exp {p['expiry']})" for n, p in carried_forward.items())
                           + " past today's close -- these are 3DTE, not expiring today, and will keep "
                             "being watched for her exit signal on a later day.")
                    print(txt)
                    telegram(txt, high_priority=False)
                    oversight_log(txt, "3DTE afternoon position(s) correctly carried past close, not alarmed.",
                                  outcome="carried_forward")
                if not expiring_today_open and not carried_forward:
                    telegram("Ashley-trigger daily watcher stopping: market closed, no open positions.",
                              high_priority=False)
                save_open_positions(open_positions)
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
                        dte_kind = dte_kind_for(now)
                        # MIN_MINUTES_BEFORE_FORCE_CLOSE only matters for a 0DTE
                        # entry (real case 2026-09-30: 21-second hold, -$24.78,
                        # from a 0DTE bought 6 seconds before the force close).
                        # An afternoon trigger buys the 3DTE contract instead,
                        # which isn't expiring today at all, so this guard
                        # (and the whole concept of "too close to force-close")
                        # simply doesn't apply to it.
                        if dte_kind == "0dte":
                            _fc = now.replace(hour=FORCE_CLOSE_ET[0], minute=FORCE_CLOSE_ET[1],
                                              second=0, microsecond=0)
                            _mins_left = (_fc - now).total_seconds() / 60
                            if _mins_left < MIN_MINUTES_BEFORE_FORCE_CLOSE:
                                if s["name"] not in _late_skips:
                                    _late_skips.add(s["name"])
                                    txt = (f"Ashley {s['name']}: zone hit at {now:%H:%M} ET but only "
                                           f"{_mins_left:.0f} min to the {FORCE_CLOSE_ET[0]}:"
                                           f"{FORCE_CLOSE_ET[1]:02d} forced close "
                                           f"(need {MIN_MINUTES_BEFORE_FORCE_CLOSE}) -- NOT entering.")
                                    print(txt)
                                    telegram(txt)
                                    oversight_log(txt,
                                                  "A position opened this close to the forced close "
                                                  "cannot work; it only pays the spread and two "
                                                  "commissions (real case 2026-09-30: 21-second hold, "
                                                  "-$24.78).",
                                                  outcome="skipped_late_entry")
                                continue

                        if dte_kind == "3dte" and s["name"] not in contracts3:
                            # 3DTE contract failed to qualify earlier -- see the
                            # qualify try/except above. Never silently fall back
                            # to buying 0DTE instead; that would be exactly the
                            # wrong-instrument mistake this whole feature exists
                            # to prevent. Skip and escalate instead.
                            if s["name"] not in _late_skips:
                                _late_skips.add(s["name"])
                                txt = (f"Ashley {s['name']}: afternoon zone hit at {now:%H:%M} ET but the "
                                       f"3DTE contract (exp {third_ibkr}) never qualified -- NOT entering "
                                       f"as 0DTE instead (that would be the wrong instrument). Check manually.")
                                print(txt)
                                telegram(txt, high_priority=True)
                                oversight_log(txt, "3DTE contract qualification failure blocked an afternoon entry.",
                                              outcome="skipped_no_3dte_contract")
                            continue

                        expiry = today_ibkr if dte_kind == "0dte" else third_ibkr
                        yymmdd = today_yymmdd if dte_kind == "0dte" else third_yymmdd
                        c = contracts0[s["name"]] if dte_kind == "0dte" else contracts3[s["name"]]
                        contracts[s["name"]] = c
                        print(f"TRIGGER: {s['name']} -- SPY {spot} in zone [{s['lo']}, {s['hi']}] "
                              f"({dte_kind}, exp {expiry})")
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

                        sym = occ_symbol(s["strike"], s["right"], yymmdd)
                        ok, fill_px = await ibkr_place_leg_with_ladder_async(
                            ib, c, "BUY", f"ashley {s['name']}", qty, bid, ask, mid)

                        fired.add(s["name"])
                        save_fired_today(fired)
                        if ok:
                            pos_id = f"ASHLEY_{s['name']}_{now.strftime('%Y%m%d_%H%M%S')}"
                            open_positions[s["name"]] = {
                                "symbol": sym, "fill_px": fill_px, "qty": qty,
                                "opened_at": now.isoformat(), "pos_id": pos_id,
                                "trail": True, "peak_bid": fill_px,
                                "expiry": expiry, "dte_kind": dte_kind,
                                "strike": s["strike"], "right": s["right"],
                            }
                            save_open_positions(open_positions)
                            last_open_positions_snapshot = dict(open_positions)
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
                            dte_note = " [3DTE -- afternoon rule]" if dte_kind == "3dte" else ""
                            msg = (f"Ashley trigger FILLED: bought {qty}x {s['name']} @ ${fill_px}"
                                   f"{dte_note} (SPY {spot}, zone [{s['lo']},{s['hi']}]). Now watching her "
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
                            if not content:
                                continue

                            # RETIREMENT (2026-09-29 incident, see STRIKETHROUGH_RE):
                            # she marks a finished level by striking it through in a
                            # NEW message, not by editing the daily alert. Retire it
                            # here so the entry monitor can never buy into a level she
                            # has already closed.
                            retired = retired_setups_from(content, setups)
                            if retired:
                                newly = {n for n in retired if n not in fired}
                                if newly:
                                    fired.update(newly)
                                    save_fired_today(fired)
                                    txt = (f"Ashley retired {', '.join(sorted(newly))} "
                                           f"(struck through in her message) -- will NOT be entered.")
                                    print(txt)
                                    telegram(txt)
                                    oversight_log(txt, "Setup retired by strikethrough.",
                                                  outcome="retired_by_strikethrough")
                                # Fixed 2026-10-19: real incident 2026-09-29 (765C) --
                                # she struck through an ALREADY-OPEN position's listing
                                # 5 minutes after the fill, and the retirement handling
                                # above (which only ever updates `fired` for NOT-yet-
                                # entered names) silently did nothing with it. A struck-
                                # through OPEN position is her exit, just as unconditionally
                                # as a struck-through pending one is her "don't enter."
                                for _name in retired_names_needing_exit(retired, open_positions):
                                    _pos = open_positions[_name]
                                    _txt = (f"Ashley struck through {_name} while it was an open "
                                            f"position -- treating as her exit, closing now.")
                                    print(_txt)
                                    telegram(_txt, high_priority=True)
                                    _ok = await close_position(
                                        ib, _name, _pos, setups,
                                        f"struck through (retired) in her message: {content!r}")
                                    if _ok:
                                        open_positions.pop(_name, None)

                            kind = exit_signal_kind(content)
                            if kind is None:
                                continue
                            # An abandoned or finished play must never be entered
                            # later in the day either -- she has told us the level
                            # is off. This is the exit-side twin of the
                            # strikethrough retirement above, and it matters even
                            # when we hold nothing right now, because the entry
                            # monitor is still watching that zone.
                            if kind in ("abandoned", "done"):
                                hint = match_open_setup(content, open_positions, setups) \
                                       if open_positions else None
                                stale = {hint} if hint else set()
                                if not stale and len(open_positions) == 1:
                                    stale = set(open_positions)
                                newly = {n for n in stale if n not in fired}
                                if newly:
                                    fired.update(newly)
                                    save_fired_today(fired)
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

                            print(f"EXIT SIGNAL from Ashley [{kind}]: {content}")
                            if kind == "abandoned":
                                telegram(f"Ashley ABANDONED the setup (\"{content}\") -- "
                                         f"closing now rather than waiting for a TP that "
                                         f"will never come.", high_priority=True)
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

                            # Real incident 2026-10-15 (770P): popped here
                            # unconditionally before, so a sell that failed to
                            # fill (3 ladder steps, none filled) silently
                            # vanished from tracking anyway -- no force-close,
                            # no EOD alarm, while the real position sat open.
                            # Only drop it on a CONFIRMED fill.
                            pos = open_positions[name]
                            ok = await close_position(ib, name, pos, setups, content)
                            if ok:
                                open_positions.pop(name, None)
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
                            edit_changed_state = True
                            old_by_name = {s["name"]: s for s in setups}
                            new_by_name = {s["name"]: s for s in new_setups}

                            for name, new_s in new_by_name.items():
                                old_s = old_by_name.get(name)
                                if old_s is None:
                                    setups.append(new_s)
                                    # Build both DTE variants, same as the upfront setups --
                                    # an edit posted mid-morning could still trigger this
                                    # afternoon, so it needs the 3DTE option ready too.
                                    c0 = Option("SPY", today_ibkr, new_s["strike"], new_s["right"],
                                                "SMART", tradingClass="SPY")
                                    c3 = Option("SPY", third_ibkr, new_s["strike"], new_s["right"],
                                                "SMART", tradingClass="SPY")
                                    try:
                                        await ib.qualifyContractsAsync(c0, c3)
                                        contracts3[name] = c3
                                    except Exception as exc:
                                        print(f"Edit-added setup {name}: 3DTE qualify failed ({exc}), "
                                              f"0DTE only -- an afternoon trigger on it will be skipped, "
                                              f"not bought as 0DTE.")
                                        await ib.qualifyContractsAsync(c0)
                                    contracts0[name] = c0
                                    contracts[name] = c0
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

                            # PERSIST the reconciled setups (fix 2026-09-25).
                            # Real gap found live: Ashley edited her alert to add
                            # 771P, this loop added it in MEMORY and traded it,
                            # but the state file still held only the original
                            # three setups. TWS drops most nights and launchd
                            # restarts this process; a restart reloads from that
                            # file, so the 771P would have returned UNKNOWN --
                            # no exit monitoring on her message and, worse, no
                            # 15:55 force-close on a physically settled 0DTE.
                            try:
                                st_now = load_state()
                                st_now["alert_setups"] = setups
                                st_now["alert_edited_ts"] = alert_edited_ts
                                save_state(st_now)
                                print(f"state persisted after edit: {[x['name'] for x in setups]}")
                            except Exception as exc:
                                telegram(f"WARNING: could not persist edited setups ({exc}) -- a restart "
                                         f"may lose track of them.", high_priority=True)
                    except Exception as exc:
                        print(f"Edit-check error: {exc}")

            await asyncio.sleep(5)
    finally:
        try:
            ib.cancelMktData(spy)
        except Exception:
            pass
        # Final account-wide commission catch-up for the session. This runs
        # after the 15:55 force-close, so the day's closing fees are included.
        try:
            import commission_ledger
            await commission_ledger.snapshot_async(ib, strategy="ashley_0dte")
        except Exception as exc:
            print(f"commission_ledger snapshot failed (non-fatal): {exc}")
        ib.disconnect()

    return last_message_id


def _update_crash_state(state: dict, today_iso: str, now_after_crash) -> bool:
    """Bumps today's same-day crash counter (resetting it on a new day) and
    returns True iff main() should give up on today entirely -- past the
    16:00 close, or the crash budget (CRASH_RETRY_LIMIT) is exhausted --
    vs False to retry via the next loop iteration's recover_todays_alert()
    path. A pure-ish mutate-and-decide helper so this is unit-testable in
    isolation, the way trail_stop_hit() already is for the trailing stop.

    Real incident 2026-10-12: run_daily_watch() crashed ~9:23-9:40 ET on a
    ConnectionRefusedError while TWS was mid-relogin. The old code
    unconditionally set state["last_processed_date"] = today_iso right after
    ANY crash, so every restart for the rest of the day saw "already
    processed today" and silently idled -- never watching again. SPY sat
    inside the day's 778C zone [777.7,778.0] for ~25 straight minutes right
    before the close and nothing ever checked for it.

    Deliberately NOT is_market_hours() for the close check -- that requires
    now >= 9:30, but the real crash above first fired ~9:23, pre-open. The
    question here is only "is there still time left today to retry", i.e.
    are we before the close -- not whether official hours have begun."""
    if state.get("crash_date") != today_iso:
        state["crash_date"] = today_iso
        state["crash_count"] = 0
    state["crash_count"] += 1
    past_close = now_after_crash >= now_after_crash.replace(hour=16, minute=0, second=0, microsecond=0)
    return past_close or state["crash_count"] >= CRASH_RETRY_LIMIT


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

        if now < now.replace(hour=WATCH_START_ET[0], minute=WATCH_START_ET[1], second=0, microsecond=0):
            await asyncio.sleep(DAY_POLL_S)
            continue

        print(f"New trading day {today_iso} -- waiting for ashleyklieu's daily alert...")
        recovered = recover_todays_alert(headers, state, today_iso)
        if recovered:
            setups, alert_edited_ts = recovered
            alert_message_id = state["alert_message_id"]
            last_message_id = state.get("last_message_id")
            print(f"Restart recovery: today's alert ({alert_message_id}) already consumed earlier -- "
                  f"resuming watch with setups {[s['name'] for s in setups]}.")
        else:
            setups, last_message_id, alert_edited_ts = wait_for_daily_setups(headers, state)
            alert_message_id = last_message_id  # see wait_for_daily_setups docstring -- same ID
            state["last_message_id"] = last_message_id
            if setups:
                # Persist what a restart needs to resume today's watch -- see
                # recover_todays_alert().
                state.update({"alert_date": today_iso, "alert_message_id": alert_message_id,
                              "alert_setups": setups, "alert_edited_ts": alert_edited_ts})
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
            state["last_processed_date"] = today_iso
            save_state(state)
        except Exception as exc:
            # Real incident 2026-10-12: run_daily_watch() crashed ~9:23-9:40 ET
            # on a ConnectionRefusedError while TWS was mid-relogin. The old code
            # unconditionally set last_processed_date = today_iso right after this
            # except block, so every restart for the REST OF THE DAY saw "already
            # processed today" at the top-of-loop check and silently idled --
            # never calling recover_todays_alert()/run_daily_watch() again. SPY
            # sat inside the day's 778C zone [777.7,778.0] for ~25 straight minutes
            # right before the 16:00 close and nothing ever checked for it.
            # Fix: only give up for the day (set last_processed_date) once we're
            # past close or have exhausted a small same-day crash budget; otherwise
            # leave last_processed_date unset so the next loop iteration re-enters
            # via recover_todays_alert() (the same mid-day-restart path that already
            # works correctly, per the 2026-10-06 10:22 restart log) and keeps trying.
            print(f"Daily watch crashed: {exc}")
            give_up = _update_crash_state(state, today_iso, datetime.now(ET))
            if give_up:
                telegram(f"Ashley-trigger daily watcher crashed today: {exc} -- check manually, no more entries today.",
                          high_priority=True)
                oversight_log(f"Daily watch crashed: {exc}", "Unhandled exception mid-day; crash budget "
                              f"exhausted or past market close -- giving up for today.", outcome="crashed")
                state["last_processed_date"] = today_iso
                save_state(state)
            else:
                telegram(f"Ashley-trigger daily watcher crashed today: {exc} -- retrying automatically "
                          f"(attempt {state['crash_count']}/{CRASH_RETRY_LIMIT}), market still open.",
                          high_priority=True)
                oversight_log(f"Daily watch crashed: {exc}", "Unhandled exception mid-day; market still open "
                              f"and crash budget remains -- auto-retrying via recover_todays_alert().",
                              outcome="crashed_retrying")
                save_state(state)
                await asyncio.sleep(DAY_POLL_S)


if __name__ == "__main__":
    asyncio.run(main())
