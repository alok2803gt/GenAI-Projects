"""
Record a manually-completed EVC close: writes BOTH the trade_journal.db
update AND moves the position from earnings_vol_crush_state.json's
"positions" to "closed_today" -- in one step, so a manual close (buying-
power wall, order-routing quirk, stranded leg, CLOSE_INCOMPLETE with no
retry -- all real, repeated reasons this account's EVC closes have needed
manual completion) can never again skip the journal update the way CRM,
CRWD, INTU, MRVL, AFRM, and ULTA all did this week.

Root cause (found 2026-08-28, CEO question "so what is total pnl by evc
ticker"): _evc_close_position() DOES correctly write real pnl to
trade_journal.db on its own -- but every single one of this week's real
closes was done by hand (via alpaca_0dte_common.place_leg_with_ladder
scripts), bypassing that function entirely. A separate, generic orphan-
cleanup sweep later found the journal row still open for a ticker no
longer in ev["positions"] (since the position was removed via a direct
state-file edit) and marked it "orphaned" with pnl=None -- silently
discarding the real, known outcome.

Usage (as a library, from a python -c snippet after computing real fills):

    from evc_record_manual_close import record_manual_close
    record_manual_close(
        pos_id="ULTA_20260827", ticker="ULTA", entry_date="2026-08-27",
        net_credit=-0.39, close_cost=0.31, pnl=-58.0, exit_reason="manual_close_...",
    )

This does NOT place any orders -- it only records a close that has ALREADY
been executed via real Alpaca fills. Always verify real Alpaca positions
are actually flat (or down to an accepted worthless stranded leg) before
calling this.
"""
import json
import sqlite3
from datetime import date

JOURNAL_DB_PATH = "trade_journal.db"
EVC_STATE_PATH = "earnings_vol_crush_state.json"


def record_manual_close(pos_id: str, ticker: str, entry_date: str,
                         net_credit: float, pnl: float, exit_reason: str,
                         close_cost: float = None, exit_date: str = None) -> None:
    exit_date = exit_date or date.today().isoformat()
    pnl_pct = round(pnl / (net_credit * 100) * 100, 1) if net_credit else None
    win = 1 if pnl > 0 else 0

    # 1. Journal: update the existing open row if one exists (matches
    #    ticker + strategy_type + closed_at IS NULL, same lookup
    #    _evc_close_position itself uses), else insert a fresh row --
    #    covers positions that never got an initial journal insert at all
    #    (e.g. entered via a one-off manual script, not the normal scan flow).
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    row = con.execute(
        "SELECT id FROM trade_journal WHERE ticker=? AND strategy_type='earnings_vol_crush'"
        " AND closed_at IS NULL ORDER BY id DESC LIMIT 1", (ticker,)
    ).fetchone()
    if row:
        con.execute("""
            UPDATE trade_journal
            SET closed_at=?, exit_price=?, pnl=?, pnl_pct=?, win=?, exit_reason=?
            WHERE id=?
        """, (exit_date, close_cost, pnl, pnl_pct, win, exit_reason, row[0]))
        action = f"updated existing journal row id={row[0]}"
    else:
        con.execute("""
            INSERT INTO trade_journal
                (opened_at, closed_at, ticker, action, qty, entry_price, exit_price,
                 pnl, pnl_pct, win, exit_reason, strategy_type, commission, is_paper, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (entry_date, exit_date, ticker, "SELL_CONDOR", 1, net_credit, close_cost,
              pnl, pnl_pct, win, exit_reason, "earnings_vol_crush", 0.0, 0,
              "Recorded via evc_record_manual_close.py -- no prior open journal row found."))
        action = "inserted new journal row"
    con.commit()
    con.close()
    print(f"[journal] {ticker}: {action}, pnl={pnl}")

    # 2. State file: move from positions -> closed_today, if still present there.
    #    Safe to call even if it's already been removed (e.g. this is being
    #    run purely to fix a journal gap after the state file was already
    #    hand-edited) -- just skips this half in that case.
    try:
        with open(EVC_STATE_PATH) as f:
            ev = json.load(f)
    except FileNotFoundError:
        print(f"[state] {EVC_STATE_PATH} not found -- skipped")
        return
    pos = ev.get("positions", {}).pop(pos_id, None)
    if pos is None:
        print(f"[state] {pos_id} not found in open positions -- already removed, skipped")
        return
    ev.setdefault("closed_today", []).append({
        "pos_id": pos_id, "ticker": ticker, "date": pos.get("date"),
        "exit_date": exit_date, "expiry": pos.get("expiry"),
        "expected_move": pos.get("expected_move"), "short_put": pos.get("short_put"),
        "short_call": pos.get("short_call"), "net_credit": net_credit,
        "close_cost": close_cost, "pnl": pnl, "pnl_pct": pnl_pct, "win": win,
        "exit_reason": exit_reason,
    })
    with open(EVC_STATE_PATH, "w") as f:
        json.dump(ev, f)
    print(f"[state] {pos_id} moved to closed_today. NOTE: restart the backend "
          f"(kill + let the watchdog relaunch) for main.py to pick this up -- "
          f"it holds this file's contents in memory, not a live read.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pos-id", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--entry-date", required=True)
    ap.add_argument("--net-credit", type=float, required=True)
    ap.add_argument("--pnl", type=float, required=True)
    ap.add_argument("--exit-reason", required=True)
    ap.add_argument("--close-cost", type=float, default=None)
    ap.add_argument("--exit-date", default=None)
    args = ap.parse_args()
    record_manual_close(args.pos_id, args.ticker, args.entry_date, args.net_credit,
                         args.pnl, args.exit_reason, args.close_cost, args.exit_date)
