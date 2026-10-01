"""Housekeeping agent -- keeps the books agreeing with the broker.

WHY THIS EXISTS
---------------
Bookkeeping drift has been a recurring, expensive class of failure here, and
every instance was found by hand, late:

  2026-09-21  Stock journal rows auto-closed 3 seconds after opening, because
              IBKR reports stocks with strike=0.0/right='0' and the journal
              stores NULL. Key normalisation fixed.
  2026-09-25  Ashley edited her alert to add 771P; the executor traded it but
              never persisted it, so a restart would have dropped exit
              monitoring and the 15:55 force-close on a physically-settled 0DTE.
  2026-09-25  accumulation recorded IBKR-REJECTED orders as completed tranches,
              wedging the -5% add trigger against a phantom entry for four days.
  2026-09-29  The executor bought a level Ashley had struck through 9 minutes
              earlier (-$99), because retirement arrives in a NEW message and
              only alert EDITS were re-read.
  2026-09-29  Five still-held IDR positions marked exit_reason='orphaned' during
              TWS's nightly restart: ib.portfolio() came back PARTIAL, and the
              guard only required it to be non-empty.

None of those were strategy failures. All were the books disagreeing with
reality, and every one was caught only because somebody went looking.

DESIGN PRINCIPLE: CONSERVATIVE BY CONSTRUCTION
---------------------------------------------
The 2026-09-29 corruption was itself caused by over-eager automatic closing. So
this agent splits every finding into exactly two classes:

  SAFE_FIX  -- provably safe, idempotent and reversible, and ONLY where the
               broker is the authority and disagreement can have one meaning.
               Example: a journal row marked 'orphaned' for a position IBKR
               still holds. There is no reading of that where the row is right.

  FLAG      -- everything else. Reported and alerted, never touched. Anything
               involving closing a row, deleting state, or guessing an outcome
               lands here on purpose.

It NEVER places, modifies or cancels an order, and never marks anything closed.
It can only re-open, drop provably-unfilled records, and backfill the ledger.
Every write is preceded by a timestamped copy of the file.

    ../venv/bin/python housekeeping_agent.py             # report only (default)
    ../venv/bin/python housekeeping_agent.py --fix       # apply SAFE_FIX items
    ../venv/bin/python housekeeping_agent.py --fix --quiet
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
JOURNAL = HERE / "trade_journal.db"
BACKUPS = HERE / "housekeeping_backups"
LOG = HERE / "housekeeping_agent.log"
IBKR_PORT, CLIENT_ID = 7496, 1780

STATE_FILES = {
    "idr": HERE / "harami_trader_state.json",
    "accumulation": HERE / "accumulation_state.json",
    "ashley": HERE / "ashleyklieu_trigger_executor_state.json",
}

findings: list[dict] = []


def note(kind: str, severity: str, msg: str, fix=None) -> None:
    """severity: SAFE_FIX (auto-repairable) | FLAG (human decision)."""
    findings.append({"kind": kind, "severity": severity, "msg": msg, "fix": fix})


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    BACKUPS.mkdir(exist_ok=True)
    dst = BACKUPS / f"{path.name}.{datetime.now(ET):%Y%m%d-%H%M%S}"
    shutil.copy2(path, dst)
    return dst


def log_line(text: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"[{datetime.now(ET).isoformat()}] {text}\n")


# ─────────────────────────── broker truth ───────────────────────────

def _ledger_sell(sym: str) -> dict | None:
    """IBKR's most recent recorded SELL for a symbol, or None.

    The commission ledger is populated from IBKR's own execution reports, so a
    row here is broker fact rather than local bookkeeping -- which is what makes
    the repairs that rely on it SAFE_FIX rather than FLAG.
    """
    if not JOURNAL.exists():
        return None
    con = sqlite3.connect(JOURNAL)
    try:
        r = con.execute(
            "SELECT price, realized_pnl, trade_date FROM executions "
            "WHERE symbol=? AND side='SLD' ORDER BY trade_date DESC, rowid DESC LIMIT 1",
            (sym,)).fetchone()
    except sqlite3.OperationalError:
        r = None
    finally:
        con.close()
    return {"price": r[0], "realized": r[1] or 0.0, "date": r[2]} if r else None


def _ledger_buy_after(sym: str, date_str: str) -> bool:
    """Did IBKR report a BUY for this symbol on/after date_str?"""
    if not JOURNAL.exists() or not date_str:
        return False
    con = sqlite3.connect(JOURNAL)
    try:
        n = con.execute("SELECT COUNT(*) FROM executions WHERE symbol=? AND side='BOT' "
                        "AND trade_date >= ?", (sym, date_str)).fetchone()[0]
    except sqlite3.OperationalError:
        n = 0
    finally:
        con.close()
    return n > 0


def broker_state(ib) -> dict:
    """The authority. Everything else is compared against this."""
    stk, opt = {}, {}
    for p in ib.positions():
        if not p.position:
            continue
        c = p.contract
        if c.secType == "STK":
            stk[c.symbol] = {"qty": p.position, "avg": p.avgCost}
        else:
            opt[(c.localSymbol or "").strip()] = {
                "qty": p.position, "avg": p.avgCost, "symbol": c.symbol,
                "secType": c.secType}
    return {"stocks": stk, "options": opt}


# ─────────────────────────── checks ───────────────────────────

def check_wrongly_orphaned(bs: dict) -> None:
    """Journal rows closed as 'orphaned' for positions the broker still holds.

    This is the 2026-09-29 corruption. It has exactly one meaning -- the row is
    wrong -- so it is the clearest SAFE_FIX in the agent.
    """
    if not JOURNAL.exists():
        return
    con = sqlite3.connect(JOURNAL)
    rows = con.execute(
        "SELECT id, ticker, closed_at FROM trade_journal "
        "WHERE exit_reason='orphaned' AND exit_price IS NULL AND closed_at IS NOT NULL"
    ).fetchall()
    con.close()
    bad = [(i, t, c) for i, t, c in rows if t in bs["stocks"]]
    for i, t, c in bad:
        note("wrongly_orphaned", "SAFE_FIX",
             f"journal id {i} ({t}) marked orphaned at {str(c)[:19]} but IBKR still "
             f"holds {bs['stocks'][t]['qty']:g} shares",
             fix=("reopen_journal", i))


def check_open_journal_without_position(bs: dict) -> None:
    """Open journal rows with no broker position. FLAG -- never auto-close.

    Closing these automatically is precisely what corrupted the journal on
    2026-09-29. A missing position can mean a real exit whose record was lost,
    a partial portfolio feed, or an unfilled order.
    """
    if not JOURNAL.exists():
        return
    con = sqlite3.connect(JOURNAL)
    rows = con.execute(
        "SELECT id, ticker, opened_at, strategy_type FROM trade_journal "
        "WHERE closed_at IS NULL"
    ).fetchall()
    con.close()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    for i, t, opened, strat in rows:
        if t in bs["stocks"]:
            continue
        try:
            od = datetime.fromisoformat(opened)
            if od.tzinfo is None:
                od = od.replace(tzinfo=ET)
            if od > cutoff:
                continue            # may still be a working order
        except Exception:
            pass
        note("open_row_no_position", "FLAG",
             f"journal id {i} ({t}, {strat}) is open but IBKR holds no position "
             f"-- a real exit whose record was lost, or a stale row. NOT closing "
             f"automatically; decide and record the real exit.")


def check_closed_without_outcome() -> None:
    """Closed rows with no exit_price / pnl -- the outcome was never recorded."""
    if not JOURNAL.exists():
        return
    con = sqlite3.connect(JOURNAL)
    rows = con.execute(
        "SELECT id, ticker, closed_at, exit_reason FROM trade_journal "
        "WHERE closed_at IS NOT NULL AND (exit_price IS NULL OR pnl IS NULL)"
    ).fetchall()
    con.close()
    for i, t, closed, reason in rows:
        # The commission ledger holds IBKR's own fills. Where it has a SELL for
        # this symbol, the outcome is not a judgement call -- it is recorded
        # broker fact -- so backfilling it is a SAFE_FIX rather than a FLAG.
        sell = _ledger_sell(t)
        if sell:
            note("closed_without_outcome", "SAFE_FIX",
                 f"journal id {i} ({t}) closed with no exit_price/pnl, but the ledger "
                 f"has IBKR's fill: sold {sell['price']} realised {sell['realized']:+.2f}",
                 fix=("backfill_outcome", (i, t)))
        else:
            note("closed_without_outcome", "FLAG",
                 f"journal id {i} ({t}) closed with no exit_price/pnl and the ledger has "
                 f"no matching sell -- realised P&L is unrecoverable for this row.")


def check_state_vs_broker(bs: dict) -> None:
    """Trader state files that disagree with the broker."""
    p = STATE_FILES["idr"]
    if not p.exists():
        return
    try:
        st = json.loads(p.read_text())
    except Exception as exc:
        note("state_unreadable", "FLAG", f"{p.name} will not parse: {exc}")
        return
    pos = st.get("positions", {})
    seen: dict[str, list[str]] = {}
    for key, info in pos.items():
        if not isinstance(info, dict):
            continue
        t = info.get("ticker")
        phase = info.get("phase")
        seen.setdefault(t, []).append(key)
        held = t in bs["stocks"]
        if phase in ("open", "pending_exit") and not held:
            sell = _ledger_sell(t)
            if sell:
                note("state_claims_unheld", "SAFE_FIX",
                     f"{p.name}: {t} ({key}) is phase='{phase}' but IBKR holds none and "
                     f"the ledger shows it SOLD at {sell['price']} "
                     f"({sell['realized']:+.2f}) -- the exit completed",
                     fix=("close_state_entry", (key, sell)))
            else:
                note("state_claims_unheld", "FLAG",
                     f"{p.name}: {t} ({key}) is phase='{phase}' but IBKR holds none and "
                     f"the ledger has no sell -- cannot tell whether it exited or the "
                     f"feed was partial. Not touching it.")
        if phase == "pending_entry" and info.get("entry_price") in (None, 0) \
                and not _ledger_buy_after(t, str(info.get("placed_at") or "")[:10]):
            note("state_stale_pending_entry", "SAFE_FIX",
                 f"{p.name}: {t} ({key}) is phase='pending_entry' with no entry_price "
                 f"and the ledger has NO buy on/after {str(info.get('placed_at'))[:10]} "
                 f"-- the order never filled",
                 fix=("drop_state_entry", key))
        elif phase == "pending_entry" and held:
            note("state_pending_but_held", "FLAG",
                 f"{p.name}: {t} ({key}) is phase='pending_entry' but IBKR already "
                 f"holds {bs['stocks'][t]['qty']:g} -- a fill the state never recorded.")
    for t, keys in seen.items():
        if len(keys) > 1:
            note("state_duplicate_ticker", "FLAG",
                 f"{p.name}: {t} appears {len(keys)} times ({', '.join(keys)}) -- "
                 f"duplicate entries can double-count exposure.")
    for t in bs["stocks"]:
        if t not in seen:
            note("held_untracked", "FLAG",
                 f"IBKR holds {bs['stocks'][t]['qty']:g} {t} but no IDR state entry "
                 f"claims it -- nothing is managing its exit.")


def check_phantom_tranches() -> None:
    """accumulation tranches recorded from orders that never filled."""
    p = STATE_FILES["accumulation"]
    if not p.exists():
        return
    try:
        st = json.loads(p.read_text())
    except Exception as exc:
        note("state_unreadable", "FLAG", f"{p.name} will not parse: {exc}")
        return
    # REAL INCIDENT 2026-10-01, caused by this very check. It used to treat ANY
    # status other than "filled" as phantom and DELETE the tranche. A
    # market-on-open tranche is recorded "Submitted" and fills at the NEXT open,
    # so on 2026-09-30 it deleted the RTX and TFC tranches that had ACTUALLY
    # FILLED (the commission ledger proves it: TFC BOT 2 @ 46.50, RTX BOT @
    # 185.83). accumulation then saw zero tranches, called it "first tranche"
    # again, and bought 3 more TFC at the 10-01 open -- bypassing the -5% add
    # trigger the whole design rests on.
    #
    # The detector was written BEFORE the reconciler in plan_for() and the two
    # were never checked against each other. They now agree on one authority:
    # the commission ledger. A Submitted tranche whose fill is confirmed is
    # CORRECTED to Filled, never deleted; only an order the broker actually
    # rejected is dropped.
    for name, rec in (st.get("names") or {}).items():
        confirmed, rejected = [], []
        for t in rec.get("tranches", []):
            stt = str(t.get("status", "")).lower()
            if stt in ("filled", ""):
                continue
            if stt in ("cancelled", "inactive", "apicancelled", "rejected"):
                rejected.append(t)
            elif _ledger_buy_after(name, str(t.get("date"))):
                confirmed.append(t)
            # Submitted with no fill yet and not rejected = still working; leave it.
        if confirmed:
            note("tranche_status_stale", "SAFE_FIX",
                 f"{p.name}: {name} has {len(confirmed)} tranche(s) still marked "
                 f"'Submitted' whose fill IS confirmed in the commission ledger -- "
                 f"correcting to Filled so accumulation counts them and does not "
                 f"re-place a first tranche",
                 fix=("mark_tranches_filled", name))
        if rejected:
            note("phantom_tranche", "SAFE_FIX",
                 f"{p.name}: {name} has {len(rejected)} tranche(s) from orders the "
                 f"broker REJECTED (status "
                 f"{', '.join(sorted({str(t.get('status')) for t in rejected}))}) -- "
                 f"these wedge the add trigger against a price nothing was bought at",
                 fix=("drop_phantom_tranches", name))


def check_ashley_state(bs: dict) -> None:
    """Ashley setups vs what actually fired, and open 0DTE nobody is tracking."""
    p, fired_p = STATE_FILES["ashley"], HERE / "ashleyklieu_fired_today.json"
    if not p.exists():
        return
    try:
        st = json.loads(p.read_text())
    except Exception as exc:
        note("state_unreadable", "FLAG", f"{p.name} will not parse: {exc}")
        return
    today = datetime.now(ET).date().isoformat()
    setups = [s.get("name") for s in st.get("alert_setups", []) if isinstance(s, dict)]
    if st.get("alert_date") == today and not setups:
        note("ashley_no_setups", "FLAG",
             f"{p.name}: alert_date is today but no setups parsed.")
    try:
        fired = json.loads(fired_p.read_text()) if fired_p.exists() else {}
    except Exception:
        fired = {}
    if fired.get("date") == today:
        unknown = [f for f in fired.get("fired", []) if f not in setups]
        if unknown:
            note("ashley_fired_unknown", "FLAG",
                 f"fired-today lists {unknown} which are not in the persisted setups "
                 f"-- a restart would not know these were already done.")
    # any 0DTE option expiring today that no setup claims
    for ls, info in bs["options"].items():
        if info["secType"] != "OPT":
            continue
        exp = ls.split()[-1][:6] if ls else ""
        if exp and exp == datetime.now(ET).strftime("%y%m%d"):
            note("untracked_0dte", "FLAG",
                 f"IBKR holds {info['qty']:g} {ls} expiring TODAY. Confirm a tracker "
                 f"owns it -- an unmanaged 0DTE risks assignment on a physically "
                 f"settled contract.")


def check_ledger(ib) -> None:
    """Executions the broker reports today that the commission ledger lacks."""
    try:
        import commission_ledger
    except Exception as exc:
        note("ledger_missing", "FLAG", f"commission_ledger unavailable: {exc}")
        return
    try:
        ib.reqExecutions()
        ib.sleep(3)
        fills = ib.fills()
    except Exception as exc:
        note("ledger_check_failed", "FLAG", f"could not read executions: {exc}")
        return
    if not fills:
        return
    con = sqlite3.connect(JOURNAL)
    try:
        have = {r[0] for r in con.execute("SELECT exec_id FROM executions")}
    except sqlite3.OperationalError:
        have = set()
    con.close()
    missing = [f for f in fills if f.execution.execId not in have]
    if missing:
        note("ledger_incomplete", "SAFE_FIX",
             f"{len(missing)} broker execution(s) today are absent from the "
             f"commission ledger -- IBKR only serves the CURRENT day, so an "
             f"unrecorded session is unrecoverable",
             fix=("snapshot_ledger", None))
    # timestamps that cannot be real
    con = sqlite3.connect(JOURNAL)
    try:
        future = con.execute(
            "SELECT COUNT(*) FROM executions WHERE ts_utc > ?",
            (datetime.now(timezone.utc).isoformat(),)).fetchone()[0]
    except sqlite3.OperationalError:
        future = 0
    con.close()
    if future:
        note("ledger_bad_timestamps", "FLAG",
             f"{future} ledger row(s) have ts_utc in the FUTURE. Live-event rows are "
             f"stamped with a local-offset bug; commissions and trade_date are "
             f"correct but intraday ordering is not.")


def check_doomed_roundtrips() -> None:
    """Round trips so short they could never have worked.

    On 2026-09-30 the Ashley executor bought the 764C at 15:54:44 and the 15:55
    forced close sold it at 15:55:05 -- a 21-second hold, -$24.78, of which
    $1.78 was commission. There was no path to a profit at entry. A
    MIN_MINUTES_BEFORE_FORCE_CLOSE guard now prevents that specific case, but
    the CLASS is worth detecting generically: any entry whose exit was forced
    by a clock rather than by the thesis.

    FLAG only -- nothing to repair after the fact. The value is noticing the
    pattern early rather than finding it in a monthly P&L review.
    """
    if not JOURNAL.exists():
        return
    con = sqlite3.connect(JOURNAL)
    try:
        rows = con.execute(
            "SELECT local_symbol, trade_date, side, price, ts_utc, realized_pnl "
            "FROM executions WHERE trade_date >= date('now','-7 day') "
            "ORDER BY local_symbol, rowid").fetchall()
    except sqlite3.OperationalError:
        con.close()
        return
    con.close()
    legs: dict[str, list] = {}
    for ls, td, side, px, ts, rp in rows:
        legs.setdefault((ls, td), []).append((side, px, ts, rp))
    for (ls, td), v in legs.items():
        buys = [x for x in v if x[0] == "BOT"]
        sells = [x for x in v if x[0] == "SLD"]
        if not buys or not sells:
            continue
        try:
            t0 = datetime.fromisoformat(buys[0][2])
            t1 = datetime.fromisoformat(sells[-1][2])
            mins = abs((t1 - t0).total_seconds()) / 60
        except Exception:
            continue
        pnl = sells[-1][3] or 0
        # A SHORT hold is not itself the problem -- the 768C on 2026-09-28 was
        # held 2.7 minutes and made +$28.91, a successful scalp. The signature of
        # a DOOMED entry is short AND losing: the position was closed by a clock
        # before the thesis could resolve. Filtering on the sign avoids flagging
        # every fast winner.
        #
        # Duration is used rather than time-of-day deliberately: live_event rows
        # carry a known local-offset bug in ts_utc, so absolute times are not
        # trustworthy while DIFFERENCES between two rows still are.
        if mins <= 5 and pnl < 0:
            note("doomed_roundtrip", "FLAG",
                 f"{ls} on {td} was opened and closed within {mins:.1f} minute(s) for "
                 f"{pnl:+.2f}. Short AND losing is the signature of an entry closed by a "
                 f"clock rather than by the thesis -- check which strategy placed it and "
                 f"whether its late-entry guard is active.")


def check_valuation_plausibility() -> None:
    """Catch the failure mode found on 2026-09-30 automatically.

    The debt alias list had missed what several issuers file under, so a missing
    debt figure silently became ZERO and RTX was valued as holding net CASH when
    it carries ~$30B of net debt -- understating enterprise value and making the
    company look cheaper than it is. Two tells are checkable without re-deriving
    anything: a non-bank showing net CASH, and an accumulate-track position whose
    stored valuation predates the current model revision.
    """
    log = HERE / "valuation_decisions.jsonl"
    if not log.exists():
        note("valuation_no_audit_trail", "FLAG",
             "valuation_decisions.jsonl is missing -- accumulate/technical calls "
             "are not being recorded anywhere that survives a cache refresh.")
        return
    try:
        import valuation as V
        rev = getattr(V, "VALUATION_LOGIC_REV", "?")
    except Exception:
        rev = "?"
    rows = []
    for ln in log.read_text().splitlines():
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    latest: dict[str, dict] = {}
    for r in rows:
        latest[r.get("ticker")] = r
    BANKS = {"TFC", "SOFI", "JPM", "BAC", "WFC", "C"}
    for t, r in latest.items():
        if r.get("eligible") and r.get("logic_rev") != rev:
            note("valuation_stale_logic", "FLAG",
                 f"{t} is eligible on logic {r.get('logic_rev')} but the model is now "
                 f"{rev} -- re-screen before adding to it.")
        nd = r.get("net_debt")
        if t not in BANKS and isinstance(nd, (int, float)) and nd < 0 \
                and r.get("model") == "reverse DCF":
            note("valuation_net_cash", "FLAG",
                 f"{t} is valued with NET CASH (net_debt ${nd/1e9:.2f}B). Genuine for "
                 f"some issuers, but it is also exactly what a missed debt concept "
                 f"looks like -- verify against the filing before trusting it.")


def check_duplicate_processes() -> None:
    import subprocess
    for name in ("ashleyklieu_trigger_executor", "day_trader_agent",
                 "harami_daily_trader", "accumulation", "spy_weekly_condor_agent"):
        try:
            out = subprocess.run(["pgrep", "-f", f"{name}.py"],
                                 capture_output=True, text=True).stdout.split()
        except Exception:
            continue
        if len(out) > 1:
            note("duplicate_process", "FLAG",
                 f"{len(out)} copies of {name}.py are running (pids {', '.join(out)}) "
                 f"-- duplicates can double-place orders.")


# ─────────────────────────── repairs ───────────────────────────

def apply_fix(kind: str, arg) -> str:
    if kind == "reopen_journal":
        backup(JOURNAL)
        con = sqlite3.connect(JOURNAL)
        con.execute("UPDATE trade_journal SET closed_at=NULL, exit_reason=NULL "
                    "WHERE id=? AND exit_price IS NULL", (arg,))
        con.commit()
        con.close()
        return f"reopened journal id {arg}"
    if kind == "mark_tranches_filled":
        pth = STATE_FILES["accumulation"]
        backup(pth)
        stt = json.loads(pth.read_text())
        rec = stt["names"][arg]
        n = 0
        for t in rec["tranches"]:
            if str(t.get("status", "")).lower() not in ("filled", "") \
                    and _ledger_buy_after(arg, str(t.get("date"))):
                t["status"] = "Filled"
                t["confirmed_by"] = "housekeeping_agent (commission ledger)"
                n += 1
        pth.write_text(json.dumps(stt, indent=1))
        return f"marked {n} {arg} tranche(s) Filled from ledger confirmation"
    if kind == "drop_phantom_tranches":
        p = STATE_FILES["accumulation"]
        backup(p)
        st = json.loads(p.read_text())
        rec = st["names"][arg]
        before = len(rec["tranches"])
        rec["tranches"] = [
            t for t in rec["tranches"]
            if str(t.get("status", "")).lower() not in
               ("cancelled", "inactive", "apicancelled", "rejected")]
        p.write_text(json.dumps(st, indent=1))
        return f"dropped {before - len(rec['tranches'])} unfilled tranche(s) for {arg}"
    if kind == "backfill_outcome":
        jid, tkr = arg
        sell = _ledger_sell(tkr)
        if not sell:
            return f"no ledger sell for {tkr}; skipped"
        backup(JOURNAL)
        con = sqlite3.connect(JOURNAL)
        con.execute("UPDATE trade_journal SET exit_price=?, pnl=?, win=? "
                    "WHERE id=? AND exit_price IS NULL",
                    (sell["price"], sell["realized"],
                     1 if sell["realized"] > 0 else 0, jid))
        con.commit()
        con.close()
        return (f"backfilled id {jid} ({tkr}) exit {sell['price']} "
                f"pnl {sell['realized']:+.2f} from the ledger")
    if kind == "close_state_entry":
        key, sell = arg
        p = STATE_FILES["idr"]
        backup(p)
        st = json.loads(p.read_text())
        rec = st["positions"].get(key)
        if not rec:
            return f"{key} already gone"
        rec["phase"] = "closed"
        rec["exit_price"] = sell["price"]
        rec["pnl"] = sell["realized"]
        rec["closed_at"] = datetime.now(ET).isoformat()
        rec["closed_by"] = "housekeeping_agent (broker+ledger confirmed)"
        p.write_text(json.dumps(st, indent=1, default=str))
        return f"marked {key} closed at {sell['price']} ({sell['realized']:+.2f})"
    if kind == "drop_state_entry":
        p = STATE_FILES["idr"]
        backup(p)
        st = json.loads(p.read_text())
        if st["positions"].pop(arg, None) is None:
            return f"{arg} already gone"
        p.write_text(json.dumps(st, indent=1, default=str))
        return f"removed never-filled entry {arg}"
    if kind == "snapshot_ledger":
        import commission_ledger
        from ib_insync import IB
        ib = IB()
        ib.connect("127.0.0.1", IBKR_PORT, clientId=CLIENT_ID + 1, timeout=25)
        try:
            n = commission_ledger.snapshot(ib, strategy=None)
        finally:
            ib.disconnect()
        return f"ledger snapshot recorded {n} executions"
    return f"unknown fix {kind}"


# ─────────────────────────── main ───────────────────────────

def telegram(text: str) -> None:
    try:
        import requests
        cfg = json.loads((HERE / "scanner_config.json").read_text())
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": text}, timeout=15)
    except Exception as exc:
        log_line(f"telegram failed: {exc}")


def main() -> int:
    do_fix = "--fix" in sys.argv
    quiet = "--quiet" in sys.argv
    stamp = datetime.now(ET)
    print(f"housekeeping agent  {stamp:%Y-%m-%d %H:%M:%S ET}  "
          f"mode={'FIX' if do_fix else 'report-only'}")

    from ib_insync import IB
    ib = IB()
    try:
        ib.connect("127.0.0.1", IBKR_PORT, clientId=CLIENT_ID, timeout=25)
    except Exception as exc:
        # TWS is down nightly and at weekends; that is not a fault worth alerting.
        print(f"TWS not reachable ({type(exc).__name__}) -- skipping this run")
        log_line(f"TWS unreachable, skipped: {type(exc).__name__}")
        return 0
    try:
        bs = broker_state(ib)
        print(f"broker: {len(bs['stocks'])} stock, {len(bs['options'])} option position(s)")
        check_wrongly_orphaned(bs)
        check_open_journal_without_position(bs)
        check_closed_without_outcome()
        check_state_vs_broker(bs)
        check_phantom_tranches()
        check_ashley_state(bs)
        check_ledger(ib)
        check_doomed_roundtrips()
        check_valuation_plausibility()
        check_duplicate_processes()
    finally:
        ib.disconnect()

    safe = [f for f in findings if f["severity"] == "SAFE_FIX"]
    flags = [f for f in findings if f["severity"] == "FLAG"]

    if not findings:
        print("clean -- books agree with the broker")
        log_line("clean")
        return 0

    print(f"\n{len(safe)} auto-repairable, {len(flags)} needing a decision\n")
    for f in safe:
        print(f"  [SAFE_FIX] {f['kind']}: {f['msg']}")
    for f in flags:
        print(f"  [FLAG]     {f['kind']}: {f['msg']}")

    applied = []
    if do_fix:
        print()
        for f in safe:
            if not f["fix"]:
                continue
            try:
                res = apply_fix(*f["fix"])
                applied.append(res)
                print(f"  FIXED: {res}")
            except Exception as exc:
                print(f"  FIX FAILED ({f['kind']}): {type(exc).__name__}: {exc}")
                flags.append({"kind": "fix_failed", "severity": "FLAG",
                              "msg": f"{f['kind']} repair failed: {exc}", "fix": None})

    log_line(f"{len(safe)} safe, {len(flags)} flagged, {len(applied)} applied: "
             + " | ".join(f["kind"] for f in findings))

    if not quiet and (applied or flags):
        lines = [f"🧹 Housekeeping {stamp:%a %H:%M ET}"]
        if applied:
            lines.append(f"repaired {len(applied)}: " + "; ".join(applied[:4]))
        if flags:
            lines.append(f"needs a decision ({len(flags)}):")
            lines += [f"• {f['msg'][:180]}" for f in flags[:5]]
            if len(flags) > 5:
                lines.append(f"• …and {len(flags) - 5} more (see housekeeping_agent.log)")
        telegram("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
