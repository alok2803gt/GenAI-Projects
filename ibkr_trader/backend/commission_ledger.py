"""Per-execution commission ledger -- what IBKR actually charged, per fill.

WHY THIS EXISTS
---------------
Asked on 2026-09-27 for "all commission charged by IBKR per trade till date",
nothing could answer it. trade_journal stores entry_price as IBKR's avgCost,
which already has the commission baked in (SOFI: 17.48 fill + 0.1763 comm =
17.656303 in the journal) -- so the fee is invisible and unrecoverable from it.
The only record was IBKR's commissionReport callbacks incidentally captured in
backend_launchd.log at INFO level, which meant coverage depended on which
process happened to be running. This module makes the capture deliberate.

Commissions are charged PER EXECUTION, not per trade -- one trade can fill in
several executions, each with its own fee -- so the grain here is the execution,
keyed by IBKR's execId. trade_journal.commission is then a rolled-up view of
this table, never the source of truth.

THE MECHANISM, AND WHY IT IS A SNAPSHOT AND NOT JUST AN EVENT
-------------------------------------------------------------
Read ib_insync/wrapper.py before changing this. Wrapper.commissionReport()
emits commissionReportEvent ONLY when permId2Trade holds the trade -- that is,
only for orders THIS client placed in THIS session. Every other execution on
the account hits the `else: # commission report is not for this client` branch
and no event ever fires. That is exactly what happened historically: the logs
held account-wide commissions because the wrapper LOGS them unconditionally,
while the event never fired for another trader's fills.

But the same method calls dataclassUpdate(fill.commissionReport, ...) for any
execId present in wrapper.fills, regardless of the event. So after
reqExecutions(), ib.fills() carries populated commissionReport objects for the
whole account. The snapshot is therefore the complete path; the event is only
the low-latency path for a trader's own fills.

IBKR's execution API serves the CURRENT TRADING DAY ONLY -- verified 2026-09-27
by querying it unfiltered and back-dated to 2026-01-01 and 2026-09-01, all of
which returned 0 fills. There is no API backfill. A day that is never
snapshotted is lost to this ledger forever, which is why snapshot_daily is
scheduled independently of any trader rather than relying on trader lifetimes.

USAGE
-----
From inside a trader that already holds an IB connection:

    import commission_ledger
    commission_ledger.attach(ib, strategy="ashley_0dte")   # live own-fill capture
    ...
    commission_ledger.snapshot(ib, strategy="ashley_0dte") # before disconnecting

Standalone (the safety net -- reads only, places nothing):

    ../venv/bin/python commission_ledger.py --snapshot
    ../venv/bin/python commission_ledger.py --report
    ../venv/bin/python commission_ledger.py --report --since 2026-09-01
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
DB = HERE / "trade_journal.db"
ET = ZoneInfo("America/New_York")

# Several processes write this ledger concurrently (each trader has its own IBKR
# connection and its own snapshot call), so every connection needs WAL and a
# real busy timeout or one of them takes "database is locked" mid-session.
_PRAGMAS = ("PRAGMA journal_mode=WAL", "PRAGMA busy_timeout=10000", "PRAGMA synchronous=NORMAL")

SCHEMA = """
CREATE TABLE IF NOT EXISTS executions (
    exec_id      TEXT PRIMARY KEY,   -- IBKR's execId: the natural dedup key.
    ts_utc       TEXT,               -- execution time, UTC (IBKR reports UTC)
    trade_date   TEXT,               -- the ET session date, for grouping
    account      TEXT,
    sec_type     TEXT,
    symbol       TEXT,
    local_symbol TEXT,
    expiry       TEXT,
    strike       REAL,
    right        TEXT,
    multiplier   INTEGER,            -- 100 for options, 1 for stock
    side         TEXT,               -- BOT / SLD, as IBKR reports it
    qty          REAL,
    price        REAL,
    notional     REAL,               -- price * multiplier * qty
    commission   REAL,
    realized_pnl REAL,               -- IBKR's own figure, already net of fees
    exchange     TEXT,
    order_id     INTEGER,
    perm_id      INTEGER,
    client_id    INTEGER,
    order_ref    TEXT,
    strategy     TEXT,               -- best-effort label from the caller
    source       TEXT,               -- live_event | snapshot | log_reconstruction
    recorded_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_exec_date   ON executions(trade_date);
CREATE INDEX IF NOT EXISTS idx_exec_symbol ON executions(symbol);
"""


def connect(db: Path = DB) -> sqlite3.Connection:
    con = sqlite3.connect(str(db), timeout=15)
    for p in _PRAGMAS:
        con.execute(p)
    return con


def ensure_schema(con: sqlite3.Connection) -> None:
    """Idempotent -- safe to call on every trader start."""
    con.executescript(SCHEMA)
    # trade_journal predates this module; add the rolled-up column in place.
    cols = {r[1] for r in con.execute("PRAGMA table_info(trade_journal)")}
    if "commission" not in cols:
        con.execute("ALTER TABLE trade_journal ADD COLUMN commission REAL")
    if "commission_note" not in cols:
        con.execute("ALTER TABLE trade_journal ADD COLUMN commission_note TEXT")
    con.commit()


# An execDetails arrives BEFORE its commissionReport, so the first write of a row
# legitimately has commission=0. A later snapshot must be able to fill that in --
# but must never blank a real value back to 0 when a stale row is re-seen. Hence
# the CASE guards rather than a plain upsert.
_UPSERT = """
INSERT INTO executions (exec_id, ts_utc, trade_date, account, sec_type, symbol,
    local_symbol, expiry, strike, right, multiplier, side, qty, price, notional,
    commission, realized_pnl, exchange, order_id, perm_id, client_id, order_ref,
    strategy, source, recorded_at)
VALUES (:exec_id, :ts_utc, :trade_date, :account, :sec_type, :symbol,
    :local_symbol, :expiry, :strike, :right, :multiplier, :side, :qty, :price,
    :notional, :commission, :realized_pnl, :exchange, :order_id, :perm_id,
    :client_id, :order_ref, :strategy, :source, :recorded_at)
ON CONFLICT(exec_id) DO UPDATE SET
    commission   = CASE WHEN COALESCE(excluded.commission, 0) <> 0
                        THEN excluded.commission ELSE executions.commission END,
    realized_pnl = CASE WHEN COALESCE(excluded.realized_pnl, 0) <> 0
                        THEN excluded.realized_pnl ELSE executions.realized_pnl END,
    strategy     = COALESCE(executions.strategy, excluded.strategy),
    local_symbol = COALESCE(executions.local_symbol, excluded.local_symbol),
    source       = CASE WHEN executions.source = 'log_reconstruction'
                        THEN excluded.source ELSE executions.source END,
    recorded_at  = excluded.recorded_at
"""


def _row_from_fill(fill, strategy: str | None, source: str) -> dict:
    """ib_insync Fill (contract, execution, commissionReport, time) -> ledger row.

    The contract comes from the Fill, which is why this path names instruments
    for free -- IBKR's bare Execution object carries no contract, the gap that
    forced average-cost fingerprinting to identify the historical trades.
    """
    c, e, cr = fill.contract, fill.execution, fill.commissionReport
    try:
        mult = int(float(c.multiplier)) if c.multiplier else 1
    except (TypeError, ValueError):
        mult = 100 if c.secType in ("OPT", "FOP") else 1
    ts = e.time
    if ts and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    comm = getattr(cr, "commission", 0.0) or 0.0
    # IBKR sends UNSET_DOUBLE (~1.8e308) when a field is absent; the wrapper
    # normalises realizedPNL/yield_ but not commission on an empty report.
    if abs(comm) > 1e100:
        comm = 0.0
    rp = getattr(cr, "realizedPNL", 0.0) or 0.0
    if abs(rp) > 1e100:
        rp = 0.0
    return {
        "exec_id": e.execId,
        "ts_utc": ts.astimezone(timezone.utc).isoformat() if ts else None,
        "trade_date": ts.astimezone(ET).date().isoformat() if ts else None,
        "account": e.acctNumber,
        "sec_type": c.secType,
        "symbol": c.symbol,
        "local_symbol": (c.localSymbol or "").strip() or None,
        "expiry": c.lastTradeDateOrContractMonth or None,
        "strike": float(c.strike) if c.strike else None,
        "right": (c.right or "").strip() or None,
        "multiplier": mult,
        "side": e.side,
        "qty": float(e.shares),
        "price": float(e.price),
        # rounded because binary floats make 1.14*100 = 113.99999999999999
        "notional": round(float(e.price) * mult * float(e.shares), 4),
        "commission": comm,
        "realized_pnl": rp,
        "exchange": e.exchange,
        "order_id": e.orderId,
        "perm_id": e.permId,
        "client_id": e.clientId,
        "order_ref": e.orderRef or None,
        "strategy": strategy,
        "source": source,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def record_fills(fills, strategy: str | None = None, source: str = "snapshot",
                 db: Path = DB) -> int:
    """Upsert an iterable of ib_insync Fills. Returns rows written."""
    rows = []
    for f in fills:
        try:
            rows.append(_row_from_fill(f, strategy, source))
        except Exception as exc:                      # never break a trader over bookkeeping
            print(f"commission_ledger: skipped a fill ({type(exc).__name__}: {exc})")
    if not rows:
        return 0
    con = connect(db)
    try:
        ensure_schema(con)
        con.executemany(_UPSERT, rows)
        con.commit()
    finally:
        con.close()
    return len(rows)


def snapshot(ib, strategy: str | None = None, db: Path = DB) -> int:
    """Pull the account's executions for the CURRENT DAY and record them.

    Account-wide, not just this client's -- see the module docstring. Returns
    the number of executions recorded. Never raises: a bookkeeping failure must
    not take down a live trader.
    """
    try:
        ib.reqExecutions()
        ib.sleep(3)                     # commissionReports trail execDetails
        n = record_fills(ib.fills(), strategy=strategy, source="snapshot", db=db)
        # Printed even when zero: silence must not be ambiguous between "ran and
        # found nothing" and "never ran at all" (which is what the first live
        # deploy on 2026-09-28 could not be distinguished from in the logs).
        print(f"commission_ledger: snapshot recorded {n} executions")
        return n
    except Exception as exc:
        print(f"commission_ledger: snapshot failed ({type(exc).__name__}: {exc})")
        return 0


async def snapshot_async(ib, strategy: str | None = None, db: Path = DB) -> int:
    """Async twin of snapshot() for the traders built on connectAsync.

    ib.sleep() drives the event loop itself, so calling the sync version from
    inside a coroutine would re-enter a running loop. Use this one there.
    """
    import asyncio
    try:
        await ib.reqExecutionsAsync()
        await asyncio.sleep(3)          # commissionReports trail execDetails
        n = record_fills(ib.fills(), strategy=strategy, source="snapshot", db=db)
        print(f"commission_ledger: snapshot recorded {n} executions")
        return n
    except Exception as exc:
        print(f"commission_ledger: snapshot_async failed ({type(exc).__name__}: {exc})")
        return 0


def attach(ib, strategy: str | None = None, db: Path = DB) -> None:
    """Subscribe to this client's own fills for immediate capture.

    Only fires for orders this client placed in this session (wrapper
    limitation, documented above), so this is a latency optimisation, NOT the
    coverage mechanism. Always pair it with snapshot().
    """
    try:
        ensure_schema_once(db)
    except Exception as exc:
        print(f"commission_ledger: could not prepare schema ({exc})")

    def on_exec(trade, fill):
        record_fills([fill], strategy=strategy, source="live_event", db=db)

    def on_comm(trade, fill, report):
        record_fills([fill], strategy=strategy, source="live_event", db=db)

    ib.execDetailsEvent += on_exec
    ib.commissionReportEvent += on_comm
    print(f"commission_ledger: attached (strategy={strategy})")


def ensure_schema_once(db: Path = DB) -> None:
    con = connect(db)
    try:
        ensure_schema(con)
    finally:
        con.close()


def commission_for_trade(symbol: str, opened_at: str | None = None,
                         closed_at: str | None = None, db: Path = DB) -> tuple[float, str]:
    """Total commission on a symbol between two timestamps -> (total, note).

    Used to populate trade_journal.commission. The note records how many
    executions it came from, so a partial roll-up is never mistaken for final.
    """
    con = connect(db)
    try:
        ensure_schema(con)
        # Compared at DATE grain, not timestamp: the rows recovered from logs
        # have no ts_utc (the logs carried only a session date), and a
        # `ts_utc >= ?` filter silently drops NULLs -- which made this return
        # $0.00 for UAL and GE while their fills sat in the ledger. Caveat of
        # the date grain: two separate round trips in the same symbol on the
        # same day would be summed together. That cannot happen for the 1-share
        # multi-day IDR trades this is used for, but check before reusing it
        # for an intraday strategy.
        q = ("SELECT COUNT(*), COALESCE(SUM(commission),0) FROM executions"
             " WHERE symbol = ?")
        args: list = [symbol]
        if opened_at:
            q += " AND COALESCE(date(ts_utc), trade_date) >= date(?)"
            args.append(_to_utc_iso(opened_at))
        if closed_at:
            q += " AND COALESCE(date(ts_utc), trade_date) <= date(?)"
            args.append(_to_utc_iso(closed_at))
        n, total = con.execute(q, args).fetchone()
    finally:
        con.close()
    return round(total or 0.0, 4), f"{n} execution(s) from commission ledger"


def _to_utc_iso(ts: str) -> str:
    """Journal timestamps are a mix of ET-offset and UTC ISO strings."""
    try:
        d = datetime.fromisoformat(ts)
    except ValueError:
        return ts
    if d.tzinfo is None:
        d = d.replace(tzinfo=ET)
    return d.astimezone(timezone.utc).isoformat()


def import_reconstructed(path: Path, db: Path = DB) -> int:
    """Load the 2026-09-14..09-25 history recovered from logs on 2026-09-27.

    Those rows were identified by average-cost fingerprinting, not by IBKR
    handing us a contract, so they are marked source='log_reconstruction' and
    any later snapshot of the same execId overwrites them.
    """
    rows = json.load(open(path))
    out = []
    for r in rows:
        name = (r.get("name") or "").strip()
        parts = name.split()
        sym = parts[0] if parts else None
        expiry = strike = right = None
        local = name or None
        if len(parts) == 2 and len(parts[1]) == 15:        # e.g. 260925P00771000
            occ = parts[1]
            expiry, right, strike = "20" + occ[:6], occ[6], int(occ[7:]) / 1000
        mult = int(r.get("mult") or 1)
        out.append({
            "exec_id": r["execId"], "ts_utc": None, "trade_date": r["date"],
            "account": "U17469343", "sec_type": "OPT" if mult == 100 else "STK",
            "symbol": sym, "local_symbol": local, "expiry": expiry,
            "strike": strike, "right": right, "multiplier": mult,
            "side": r["side"], "qty": r.get("qty") or 1, "price": r.get("price"),
            "notional": (r.get("price") or 0) * mult * (r.get("qty") or 1),
            "commission": r["comm"], "realized_pnl": r.get("realized") or 0.0,
            "exchange": r.get("exch"), "order_id": None, "perm_id": None,
            "client_id": None, "order_ref": None, "strategy": None,
            "source": "log_reconstruction",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
    con = connect(db)
    try:
        ensure_schema(con)
        con.executemany(_UPSERT, out)
        con.commit()
    finally:
        con.close()
    return len(out)


def report(since: str | None = None, db: Path = DB) -> None:
    con = connect(db)
    try:
        ensure_schema(con)
        where, args = "", []
        if since:
            where, args = " WHERE trade_date >= ?", [since]
        rows = con.execute(
            f"SELECT trade_date, side, COALESCE(local_symbol, symbol, '?') AS inst, qty, price,"
            f" notional, commission, realized_pnl, source FROM executions{where}"
            f" ORDER BY trade_date, exec_id", args).fetchall()
        if not rows:
            print("commission ledger is empty"
                  + (f" for dates >= {since}" if since else "")
                  + " -- run --snapshot on a trading day, or --import-reconstructed")
            return
        print(f"{'date':<12}{'B/S':<5}{'instrument':<24}{'qty':>4}{'price':>9}"
              f"{'notional':>10}{'comm':>7}{'%notl':>7}{'realized':>10}  src")
        for r in rows:
            d, side, inst, q, px, notl, comm, rp, src = r
            pct = (comm / notl * 100) if notl else 0.0
            print(f"{d or '?':<12}{'BUY' if side == 'BOT' else 'SELL':<5}{inst:<24}"
                  f"{q or 0:>4.0f}{px or 0:>9.2f}{notl or 0:>10.2f}{comm or 0:>7.2f}"
                  f"{pct:>6.2f}%{(rp or 0):>+10.2f}  {src}")
        tot = con.execute(f"SELECT COUNT(*), COALESCE(SUM(commission),0),"
                          f" COALESCE(SUM(notional),0) FROM executions{where}", args).fetchone()
        print(f"\n{tot[0]} executions   TOTAL COMMISSION ${tot[1]:.2f}   "
              f"on ${tot[2]:.2f} notional ({tot[1] / tot[2] * 100 if tot[2] else 0:.2f}%)")
        print("\nby date:")
        for d, n, c in con.execute(f"SELECT trade_date, COUNT(*), SUM(commission)"
                                   f" FROM executions{where} GROUP BY trade_date"
                                   f" ORDER BY trade_date", args):
            print(f"  {d or '?'}  {n:>3} execs  ${c:.2f}")
        print("\nby strategy label:")
        for s, n, c in con.execute(f"SELECT COALESCE(strategy,'(unlabelled)'), COUNT(*),"
                                   f" SUM(commission) FROM executions{where}"
                                   f" GROUP BY 1 ORDER BY 3 DESC", args):
            print(f"  {s:<28}{n:>3} execs  ${c:.2f}")
    finally:
        con.close()


def main(argv: list[str]) -> int:
    since = None
    if "--since" in argv:
        since = argv[argv.index("--since") + 1]
    if "--import-reconstructed" in argv:
        p = Path(argv[argv.index("--import-reconstructed") + 1])
        print(f"imported {import_reconstructed(p)} reconstructed executions")
        return 0
    if "--snapshot" in argv:
        from ib_insync import IB
        ib = IB()
        # clientId 1730: unused by any scheduled trader (checked 2026-09-28).
        try:
            ib.connect("127.0.0.1", 7496, clientId=1730, timeout=25)
        except (ConnectionRefusedError, OSError, TimeoutError) as exc:
            # TWS logs out over the weekend and during its nightly restart, so
            # a refused port is an EXPECTED outcome, not a failure worth a
            # traceback -- the log has to stay readable enough that a genuine
            # problem stands out. Nothing is recorded; the next run catches up,
            # as long as it is still the same trading day (the API only ever
            # serves today).
            print(f"TWS not reachable on 127.0.0.1:7496 ({type(exc).__name__}) -- "
                  f"nothing recorded this run.")
            return 0
        try:
            n = snapshot(ib, strategy=None)
            print(f"snapshot recorded {n} executions for today")
        finally:
            ib.disconnect()
        report(since)
        return 0
    if "--report" in argv or len(argv) == 0:
        report(since)
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
