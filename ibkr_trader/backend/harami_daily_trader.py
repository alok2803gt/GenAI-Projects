"""
REAL LIVE trading for the validated daily bullish-harami + downtrend +
5-day-hold rule (candlestick_pattern_research/harami_backtest.py --
5y, 112-ticker universe, 5-day hold +1.073% vs a matched downtrend
baseline of +0.486%, Welch p=0.00007). CEO decision 2026-09-11: trade it
live, matching the backtest's exact mechanics (1 share, entry at the
next trading day's open, exit at the close 5 trading days later) rather
than an options approximation the backtest never tested.

harami_scanner.py (IBKR-HaramiScanner, 4:10pm ET weekdays) does the real
signal detection and logs a `harami_scanner_alert` entry to
oversight_log.jsonl -- this script is the execution layer on top of that,
in three modes / scheduled tasks:

  --mode morning ~9:05am ET weekdays (added 2026-09-21). Pre-market recovery: re-places
                 any MOO entry order that vanished overnight (see run_morning).

  --mode entry   ~4:20pm ET weekdays, AFTER harami_scanner.py's 4:10pm
                 run. Places a Market-on-Open BUY for any signal fired
                 TODAY that passes the capital gate. Also confirms and
                 journals any position whose MOO order (placed by
                 yesterday's entry run) filled at THIS morning's open.
  --mode exit    ~3:45pm ET weekdays, before the close (MOC orders must
                 be submitted before a same-day cutoff). Places a
                 Market-on-Close SELL for any open position whose exit
                 date is today. Also confirms and journals any position
                 whose MOC order (placed by yesterday's exit run) filled
                 at YESTERDAY's close.

Order mechanics -- verified live via whatIfOrder before any real use
(2026-09-11), because getting this wrong on real money is unacceptable:
  entry = Order(action="BUY",  totalQuantity=qty, orderType="MKT", tif="OPG")
  exit  = Order(action="SELL", totalQuantity=qty, orderType="MOC", tif="DAY")
tif="MOC" is REJECTED outright (Error 10052: Invalid time in force) --
Market-on-Close is its own IBKR orderType, not a TIF value. Confirmed
live; do not "fix" this back without re-verifying first.

Capital gate: cro_cfo_capital_budget(ib, cap_pct=0.25) -- CEO decision
2026-09-11 (same override mechanism already used for GOOG Condor,
2026-08-24). The account's standard 5% per-trade cap (~$108 on the
current ~$2,170 net liq) would reject 1 share of nearly every real
signal seen so far (RTX $198, ISRG $360, LMT $530) -- this strategy
structurally needs a size-appropriate cap on this account. 25% is still
bounded underneath by the account-wide 50%-of-net-liq total risk budget
shared across every strategy (cro_cfo_capital_budget's own
TOTAL_RISK_BUDGET_PCT), so it cannot run away regardless of this
override.

REAL MONEY -- trade_journal rows are is_paper=0 (never touches the
is_paper=1 shadow convention chartexpert_auto_trader.py uses).

Trading-day arithmetic (_add_trading_days) skips weekends AND the hardcoded
NYSE_HOLIDAYS list (2026-2027 -- extend it before 2028; main() warns once
the year passes the list). Both modes do nothing on a market holiday. Exits
match `exit_date <= today`, so a missed run is retried on the next one, and
an exit never SELLs a position the account doesn't actually hold.

Usage: python harami_daily_trader.py --mode entry|exit|morning
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, date, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from ib_insync import IB, Order, Stock

sys.path.insert(0, str(Path(__file__).parent))
from alpaca_0dte_common import cro_cfo_capital_budget, safe_px  # noqa: E402

HERE = Path(__file__).parent
CFG_PATH = HERE / "scanner_config.json"
OVERSIGHT_LOG = HERE / "oversight_log.jsonl"
STATE_PATH = HERE / "harami_trader_state.json"
TRADES_LOG = HERE / "harami_trader_trades.jsonl"
JOURNAL_DB_PATH = HERE / "trade_journal.db"

ET = ZoneInfo("America/New_York")
TWS_PORT = 7496
CLIENT_ID = 1910
QTY = 1
HOLD_TRADING_DAYS = 5          # kept: the SHADOW rule we now measure against
# Exit rule changed 2026-09-24 after re-auditing the strategy. The original
# "hold exactly 5 trading days" came from a backtest that (a) compared trades
# against ANY random day rather than the same days' market, and (b) treated
# up to 29 same-day signals across the universe as independent. Correcting
# both dropped the 5-day rule's edge from +0.74pp (t=7.37) to +0.088pp
# (t=0.48) -- i.e. nothing. Of 17 exits re-tested the same honest way, the
# best was "sell at the first CLOSE above the entry price, give up after 10
# days": +0.351pp market-adjusted, 63.6% win, 3.4 days average hold, 0.104
# pp/day vs 0.081 for the 5-day rule (clustered t=2.02 -- real but NOT past
# the Bonferroni bar of 2.97 for 17 rules, so this is an unproven candidate
# being tracked forward, not a validated edge).
EXIT_RULE = "first_close_above_entry"
MAX_HOLD_TRADING_DAYS = 10     # backstop when the position never closes green
CAP_PCT = 0.25   # CEO override 2026-09-11 -- see module docstring

# Same alert format harami_scanner.py writes -- matches
# candlestick_pattern_research/harami_daily_forward_performance.py's regex.
_RE = re.compile(
    r"(?:Bullish Harami \+ downtrend|Inside Day Reversal):\s*(?P<ticker>[A-Z][A-Z.\-]*)\s*\((?P<date>\d{4}-\d{2}-\d{2})\)",
)


def now_et():
    return datetime.now(ET)


def load_cfg():
    return json.loads(CFG_PATH.read_text())


def telegram_text(cfg, text):
    try:
        from telegram_alert_gate import alert_enabled
        if not alert_enabled("harami_trader"):
            return
    except Exception:
        pass
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    try:
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      data={"chat_id": cfg["telegram_chat_id"], "text": html, "parse_mode": "HTML"},
                      timeout=10)
    except Exception as e:
        print(f"Telegram send failed: {e}")


def oversight_log(actor, category, summary, rationale="", outcome=None, pnl_impact=None):
    entry = {"time": now_et().astimezone().isoformat(), "actor": actor, "category": category,
              "summary": summary, "rationale": rationale, "outcome": outcome, "pnl_impact": pnl_impact}
    with open(OVERSIGHT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# Full-day NYSE closures (2026-2027; weekend holidays use the exchange's
# observed date -- Sat -> prior Fri, Sun -> next Mon). Hardcoded rather than
# pulling in a calendar package (pandas_market_calendars isn't installed in
# this venv). Added 2026-09-20: the old weekday-only skip could land an exit
# date on a holiday, where the MOC sell went to a closed market. Beyond the
# last year listed this silently degrades to weekday-only, so main() warns.
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
    """Skips weekends and the NYSE_HOLIDAYS above, so the result is always a
    real trading day (within the years the list covers)."""
    cur = d
    added = 0
    while added < n:
        cur += timedelta(days=1)
        if _is_trading_day(cur):
            added += 1
    return cur


def load_todays_signals(today_str):
    """Real harami_scanner_alert entries logged TODAY (date-stamped in the
    alert text itself, not the log line's own timestamp -- the scanner
    runs at 4:10pm ET so both are normally the same day, but matching on
    the alert's own date is the more correct source of truth)."""
    sigs = []
    if not OVERSIGHT_LOG.exists():
        return sigs
    with open(OVERSIGHT_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("category") != "harami_scanner_alert":
                continue
            m = _RE.search(e.get("summary", ""))
            if not m or m.group("date") != today_str:
                continue
            sigs.append({"ticker": m.group("ticker"), "date": m.group("date")})
    return sigs


def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"positions": {}}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def log_trade(record):
    with open(TRADES_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def get_spot(ib, ticker):
    try:
        stk = ib.qualifyContracts(Stock(ticker, "SMART", "USD"))[0]
        [t] = ib.reqTickers(stk)
        return safe_px(t.marketPrice()) or safe_px(t.close), stk
    except Exception:
        return None, None


def journal_insert_open(ticker, entry_price, opened_at, valuation=None, track=None):
    """valuation/track are recorded on the PERMANENT row, not just in the state
    file: trade_journal is the only record that outlives a state reset, and
    without them there is no way to audit why a name was put on the accumulate
    track months later (gap found 2026-09-30)."""
    try:
        con = sqlite3.connect(JOURNAL_DB_PATH)
        cols = {r[1] for r in con.execute("PRAGMA table_info(trade_journal)")}
        if "valuation_json" not in cols:
            con.execute("ALTER TABLE trade_journal ADD COLUMN valuation_json TEXT")
        if "track" not in cols:
            con.execute("ALTER TABLE trade_journal ADD COLUMN track TEXT")
        con.execute("""INSERT INTO trade_journal
            (opened_at, ticker, action, qty, entry_price, strategy_type, is_paper, notes)
            VALUES (?, ?, 'BUY', ?, ?, 'HARAMI_DAILY', 0, ?)""",
            (opened_at, ticker, QTY, entry_price,
             "IDR daily live entry -- the original p=0.00007 was an artefact of "
             "treating same-day signals as independent; date-clustered and "
             "market-adjusted it is +0.088pp, t=0.48 (i.e. no proven edge)"))
        if valuation is not None or track is not None:
            con.execute(
                "UPDATE trade_journal SET valuation_json=?, track=? WHERE id=?",
                (json.dumps(valuation, default=str) if valuation else None, track,
                 con.execute("SELECT MAX(id) FROM trade_journal").fetchone()[0]))
        con.commit()
        con.close()
    except Exception as e:
        print(f"  journal_insert_open failed: {e}")


def journal_close(ticker, closed_at, exit_price, pnl, exit_reason):
    try:
        con = sqlite3.connect(JOURNAL_DB_PATH)
        row = con.execute("""SELECT id, opened_at FROM trade_journal
                             WHERE ticker=? AND strategy_type='HARAMI_DAILY' AND closed_at IS NULL
                             ORDER BY id DESC LIMIT 1""", (ticker,)).fetchone()
        if row is None:
            print(f"  journal_close: no open {ticker} row to close")
            con.close()
            return
        # Roll up BOTH legs' real commissions from the per-execution ledger.
        # pnl above is price-based; this column is what the round trip actually
        # cost in fees, which entry_price hides by being IBKR's avgCost.
        comm, note = None, None
        try:
            import commission_ledger
            comm, note = commission_ledger.commission_for_trade(
                ticker, opened_at=row[1], closed_at=closed_at)
        except Exception as exc:
            note = f"commission lookup failed: {type(exc).__name__}"
        con.execute("""UPDATE trade_journal SET
            closed_at=?, exit_price=?, pnl=?, win=?, exit_reason=?,
            commission=?, commission_note=?
            WHERE id = ?""",
            (closed_at, exit_price, pnl, 1 if pnl > 0 else 0, exit_reason, comm, note, row[0]))
        con.commit()
        con.close()
    except Exception as e:
        print(f"  journal_close failed: {e}")


def _portfolio_qty(ib, conid):
    """Real, current position size for a conid, straight from the account
    (ib.portfolio() is a live server-side snapshot, unlike ib.trades()).
    Returns 0.0 if no position exists."""
    for p in ib.portfolio():
        if p.contract.conId == conid:
            return safe_px(p.position) or 0.0
    return 0.0


def _fill_info(ib, order_id, conid=None, expect="buy"):
    """Real fill status/price for a previously-placed order, by orderId.
    Returns (filled: bool, avg_price: float|None).

    Found 2026-09-18 (real GE/RCL positions stuck at phase=pending_entry
    indefinitely -- and since run_exit() only ever acts on phase=="open",
    this silently meant their 5-day hold could NEVER auto-exit): ib.trades()
    is a SESSION-LOCAL cache, populated only by activity the CURRENT
    connection has personally seen. This script opens a fresh IB() connection
    on every single run (see main()), so an order placed by yesterday's
    process is invisible to today's ib.trades() even though it genuinely
    filled -- confirmed live: a brand-new connection's ib.trades() didn't
    contain GE's order_id at all, despite GE being a real, confirmed
    portfolio position. Both ib.reqCompletedOrders() and ib.reqExecutions()
    (even with an explicit historical date filter) were ALSO checked live
    and only returned same-day data on this account -- neither is a working
    cross-session substitute here.

    Fix: when the session-local order lookup can't confirm a fill (either
    because it's genuinely not filled yet, OR because it's from a prior
    session), fall back to asking the account directly whether a real
    position now exists (conid required for this fallback). This is the
    same ground-truth check used to diagnose this bug in the first place --
    it's slower (one more query) but reflects real state regardless of
    which session placed the order."""
    for tr in ib.trades():
        if tr.order.orderId == order_id:
            st = tr.orderStatus
            if st.status == "Filled" and safe_px(st.avgFillPrice):
                return True, safe_px(st.avgFillPrice)
            if st.status == "Filled":
                return True, None
            # Order IS visible this session but not (yet) filled -- trust
            # that over the portfolio fallback; don't fall through.
            if conid is None:
                return False, None
            break

    if conid is None:
        return False, None
    qty = _portfolio_qty(ib, conid)
    if expect == "buy" and qty > 0:
        for p in ib.portfolio():
            if p.contract.conId == conid:
                return True, safe_px(p.averageCost)
        return True, None
    if expect == "sell" and qty == 0:
        return True, None  # confirmed closed; caller supplies its own price fallback
    return False, None


# ── entry mode ────────────────────────────────────────────────────────────
def run_entry(ib, cfg):
    et = now_et()
    today_str = et.strftime("%Y-%m-%d")
    state = load_state()

    # 1. Confirm any position placed by a prior entry run whose MOO order
    #    should have already filled at an earlier open.
    for key, pos in list(state["positions"].items()):
        if pos.get("phase") != "pending_entry":
            continue
        filled, px = _fill_info(ib, pos["order_id"], conid=pos.get("conid"), expect="buy")
        if not filled:
            age_days = (date.today() - date.fromisoformat(pos["signal_date"])).days
            if age_days >= 2:
                telegram_text(cfg, f"⚠️ harami_daily_trader: {pos['ticker']} MOO entry from "
                                    f"{pos['signal_date']} still not filled after {age_days}d -- check IBKR.")
            continue
        entry_price = px or pos.get("expected_spot")
        fill_date_str = et.strftime("%Y-%m-%d")
        exit_date = _add_trading_days(date.fromisoformat(fill_date_str), HOLD_TRADING_DAYS)
        # Every filled IDR order is screened on FILED fundamentals and tagged
        # with a track (added 2026-09-25):
        #   technical  -> the ordinary exit rule (first close above entry, 10d cap)
        #   accumulate -> the name stands on its own numbers, so it moves to
        #                 accumulation.py's tranche plan and a VALUATION exit
        # Any screen failure returns technical, so a data outage can never turn
        # into an unintended long-term hold.
        track, why = "technical", "not screened"
        try:
            import accumulation
            ok, why, summary = accumulation.screen(pos["ticker"])
            track = "accumulate" if ok else "technical"
            pos["valuation"] = summary
        except Exception as exc:
            why = f"screen error: {type(exc).__name__}"
        pos.update({"phase": "open", "entry_price": entry_price, "entry_fill_date": fill_date_str,
                    "exit_date": exit_date.isoformat(), "track": track, "track_reason": why})
        journal_insert_open(pos["ticker"], entry_price, et.isoformat(),
                            valuation=pos.get("valuation"), track=track)
        telegram_text(cfg, f"✅ <b>IDR LIVE ENTRY FILLED: {pos['ticker']}</b>\n"
                            f"Bought {QTY}x @ ${entry_price:.2f} (MOO).\n"
                            f"Track: <b>{track.upper()}</b> -- {why}\n"
                            + (f"Exit: first close above ${entry_price:.2f}, {MAX_HOLD_TRADING_DAYS}-day backstop."
                               if track == "technical" else
                               "Exit: valuation-based; accumulation.py manages tranches."))
        oversight_log("trader", "harami_live_entry_filled",
                      f"{pos['ticker']}: MOO entry filled @ ${entry_price:.2f}. Exit planned {exit_date}.")
    save_state(state)

    # 2. Place new entries for today's fresh signals.
    signals = load_todays_signals(today_str)
    if not signals:
        print(f"No fresh harami_scanner_alert signals today ({today_str}).")
    for sig in signals:
        key = f"{sig['ticker']}_{sig['date']}"
        if key in state["positions"]:
            continue   # already handled -- idempotent against a retried run

        # A REPEAT signal on a name already held is allowed to trade (CEO policy
        # 2026-09-25) -- the pattern firing again is fresh information, and the
        # state key is per signal DATE so the positions stay separate.
        # ONE exception: a name on the accumulate track is already being bought
        # in planned tranches with a valuation exit. Opening a second, parallel
        # 10-day technical position in the same stock would put two exit rules
        # on one holding and double the intended weight, so the repeat signal is
        # routed to accumulation.py as a tranche trigger instead.
        prior = [q for q in state["positions"].values()
                 if q.get("ticker") == sig["ticker"] and q.get("phase") in ("open", "pending_entry")]
        if prior and any(q.get("track") == "accumulate" for q in prior):
            telegram_text(cfg, f"IDR: <b>{sig['ticker']} signalled again</b> -- already on the ACCUMULATE "
                                f"track, so no separate technical position. accumulation.py will size the "
                                f"next tranche (it runs 15:30 ET).")
            oversight_log("trader", "harami_repeat_signal_accumulate",
                          f"{sig['ticker']}: repeat signal on an accumulate-track holding; "
                          f"routed to the tranche plan rather than a parallel technical trade.")
            continue
        if prior:
            print(f"{sig['ticker']}: repeat signal while {len(prior)} position(s) already open -- "
                  f"trading it (CEO policy 2026-09-25).")
        spot, contract = get_spot(ib, sig["ticker"])
        if spot is None:
            telegram_text(cfg, f"harami_daily_trader: {sig['ticker']} signal fired but no live quote -- skipped.")
            oversight_log("trader", "harami_live_skipped", f"{sig['ticker']}: no live quote available.")
            continue

        budget = cro_cfo_capital_budget(ib, cap_pct=CAP_PCT)
        cost = spot * QTY
        if cost > budget["per_strategy_cap"] or cost > budget["headroom"]:
            reason = (f"exceeds {CAP_PCT:.0%} per-trade cap (${budget['per_strategy_cap']:.0f})"
                      if cost > budget["per_strategy_cap"]
                      else f"exceeds remaining portfolio headroom (${budget['headroom']:.0f})")
            telegram_text(cfg, f"harami_daily_trader: <b>{sig['ticker']} SKIPPED</b> -- 1 share ~${cost:.0f} {reason}.")
            oversight_log("trader", "harami_live_skipped",
                          f"{sig['ticker']}: 1 share ~${cost:.0f} {reason}.",
                          rationale=f"cap_pct={CAP_PCT}, per_strategy_cap=${budget['per_strategy_cap']:.2f}, "
                                    f"headroom=${budget['headroom']:.2f}")
            continue

        # EARNINGS GATE (added 2026-09-30). This is MARKET-ON-OPEN, so anything
        # reporting before that open fills straight into the gap. The IDR
        # pipeline had NO earnings awareness until today, while the breakout and
        # daytrader scanners both had blackouts -- backwards, since IDR is the
        # strategy that HOLDS for days. Found on the day it mattered: MU reports
        # 2026-09-30 postmarket with a 6.5pct expected move and MU is in the IDR
        # panel. Fails CLOSED -- an unavailable earnings feed blocks the entry
        # rather than buying blind into a binary event.
        try:
            from earnings_guard import reports_before_next_open
            _blocked, _ewhy = reports_before_next_open(sig["ticker"])
        except Exception as _exc:
            _blocked, _ewhy = True, f"earnings guard failed ({type(_exc).__name__})"
        if _blocked:
            print(f"  SKIP {sig['ticker']}: {_ewhy}")
            telegram_text(cfg, f"IDR entry skipped: <b>{sig['ticker']}</b> -- {_ewhy}")
            oversight_log("trader", "harami_live_skipped_earnings",
                          f"{sig['ticker']}: entry skipped -- {_ewhy}.",
                          rationale="MOO fills at the next open; a report before then is "
                                    "an unhedged overnight bet on a binary event.")
            continue

        order = Order(action="BUY", totalQuantity=QTY, orderType="MKT", tif="OPG", transmit=True)
        trade = ib.placeOrder(contract, order)
        ib.sleep(2)
        state["positions"][key] = {
            "ticker": sig["ticker"], "signal_date": sig["date"], "conid": contract.conId,
            "order_id": trade.order.orderId, "phase": "pending_entry",
            "placed_at": et.isoformat(), "qty": QTY, "expected_spot": spot,
        }
        save_state(state)
        telegram_text(cfg, f"\U0001F7E2 <b>HARAMI LIVE ENTRY PLACED: {sig['ticker']}</b>\n"
                            f"Market-on-Open BUY {QTY}x, fills at tomorrow's open (spot now ~${spot:.2f}, "
                            f"cap ${budget['per_strategy_cap']:.0f}, headroom ${budget['headroom']:.0f}).")
        oversight_log("trader", "harami_live_entry_placed",
                      f"{sig['ticker']}: MOO BUY {QTY}x placed, order {trade.order.orderId}, spot ~${spot:.2f}.",
                      rationale="Real backtest, 5d hold, p=0.00007. CEO decision 2026-09-11 to trade live, "
                                "1 share matching backtest mechanics exactly, cap_pct=0.25 override.")


# ── exit mode ─────────────────────────────────────────────────────────────
def _last_price(ib, ticker):
    """Live last/close for the exit test."""
    try:
        c = Stock(ticker, "SMART", "USD")
        q = ib.qualifyContracts(c)
        td = ib.reqMktData(q[0], "", False, False)
        ib.sleep(3)
        px = td.last or td.close or td.marketPrice()
        ib.cancelMktData(q[0])
        return float(px) if px and px == px else None
    except Exception as exc:
        print(f"_last_price {ticker}: {exc}")
        return None


def _trading_days_between(a: date, b: date) -> int:
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if _is_trading_day(d):
            n += 1
    return n


def shadow_log(pos, last, note):
    """Record what the retired 5-day rule would have done, so the two exits can
    be compared on the SAME live trades (forward, out of sample)."""
    rec = {"time": datetime.now(ET).isoformat(), "ticker": pos.get("ticker"),
           "entry_date": pos.get("entry_fill_date"), "entry_price": pos.get("entry_price"),
           "price_now": last, "planned_5d_exit": pos.get("exit_date"), "note": note}
    with open(Path(__file__).resolve().parent / "harami_exit_shadow.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def run_exit(ib, cfg):
    et = now_et()
    today_str = et.strftime("%Y-%m-%d")
    state = load_state()

    # 1. Confirm any MOC sell placed by a prior exit run.
    for key, pos in list(state["positions"].items()):
        if pos.get("phase") != "pending_exit":
            continue
        filled, px = _fill_info(ib, pos["exit_order_id"], conid=pos.get("conid"), expect="sell")
        if not filled:
            telegram_text(cfg, f"⚠️ harami_daily_trader: {pos['ticker']} MOC exit not yet confirmed filled -- check IBKR.")
            continue
        if px is None:
            # Portfolio-fallback path confirms the position is gone (so it did
            # fill) but can't report a sell price -- averageCost only reflects
            # what's LEFT, and there's nothing left. Use a live quote instead
            # of silently defaulting to entry_price, which would mask the real
            # P&L as exactly $0 rather than admit it's an approximation.
            live_spot, _ = get_spot(ib, pos["ticker"])
            px = live_spot
        exit_price = px or pos["entry_price"]
        pnl = round((exit_price - pos["entry_price"]) * pos["qty"], 2)
        journal_close(pos["ticker"], et.isoformat(), exit_price, pnl, "5d_hold_complete")
        log_trade({**pos, "exit_price": exit_price, "pnl": pnl, "closed_at": et.isoformat()})
        telegram_text(cfg, f"\U0001F4B0 <b>HARAMI LIVE EXIT FILLED: {pos['ticker']}</b>\n"
                            f"Sold {pos['qty']}x @ ${exit_price:.2f} (entry ${pos['entry_price']:.2f}, MOC).\n"
                            f"P&amp;L: ${pnl:+.2f}")
        oversight_log("trader", "harami_live_exit_filled",
                      f"{pos['ticker']}: MOC exit filled @ ${exit_price:.2f}, pnl ${pnl:+.2f}.",
                      pnl_impact=pnl)
        del state["positions"][key]
    save_state(state)

    # 2. Place MOC exits for any position whose planned exit date is today OR
    #    already past (`<=`, not `==`: found 2026-09-20 -- with an equality
    #    match, one missed run -- Mac asleep, TWS down, a holiday -- meant the
    #    position was never exited, ever).
    for key, pos in list(state["positions"].items()):
        if pos.get("phase") != "open" or not pos.get("exit_date"):
            continue

        # --- first-close-above-entry rule -------------------------------------
        # Runs at ~15:45, so "the close" is approximated by the price now; the
        # order itself is MOC, which is what the backtest assumed.
        exit_now, why = False, None
        if pos.get("track") == "accumulate":
            # managed by accumulation.py (tranches + valuation exit), not by a clock
            shadow_log(pos, _last_price(ib, pos.get("ticker")),
                       "accumulate track: technical exit skipped")
            continue
        if EXIT_RULE == "first_close_above_entry":
            last = _last_price(ib, pos.get("ticker"))
            entry = pos.get("entry_price")
            fill_date = pos.get("entry_fill_date") or pos.get("signal_date")
            held_days = _trading_days_between(date.fromisoformat(fill_date), date.today()) if fill_date else 0
            if last and entry and last > entry:
                exit_now, why = True, f"first close above entry (${last:.2f} > ${entry:.2f})"
            elif held_days >= MAX_HOLD_TRADING_DAYS:
                exit_now, why = True, f"{MAX_HOLD_TRADING_DAYS}-day backstop, never closed green"
            # what the OLD rule would have done today -- logged, not traded
            if pos["exit_date"] <= today_str and not exit_now:
                shadow_log(pos, last, "5d rule would exit today; new rule holds")
            if not exit_now:
                continue
        else:
            if pos["exit_date"] > today_str:
                continue
        overdue = pos["exit_date"] < today_str

        # Never SELL something the account doesn't hold -- a MOC sell against
        # a flat position opens a naked short (the same failure that produced
        # the margin rejections on Ashley's 2026-09-18 forced close). Leave
        # the state as-is so this alerts every run until a human resolves it.
        held = _portfolio_qty(ib, pos.get("conid"))
        if held < pos["qty"]:
            telegram_text(cfg, f"⚠️ harami_daily_trader: {pos['ticker']} exit due ({pos['exit_date']}) but IBKR "
                                f"shows only {held:g} held (expected {pos['qty']}) -- NOT selling. Check IBKR.")
            oversight_log("trader", "harami_live_exit_skipped",
                          f"{pos['ticker']}: exit due {pos['exit_date']}, portfolio qty {held:g} < {pos['qty']}; no order placed.")
            continue

        contract = Stock(pos["ticker"], "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            telegram_text(cfg, f"harami_daily_trader: {pos['ticker']} exit due today but could not qualify contract -- check IBKR.")
            continue
        order = Order(action="SELL", totalQuantity=pos["qty"], orderType="MOC", tif="DAY", transmit=True)
        trade = ib.placeOrder(qualified[0], order)
        ib.sleep(2)
        # Definitive rejection -> keep phase "open" so the next run retries,
        # instead of parking it in pending_exit where nothing re-places it.
        if trade.orderStatus.status in ("Cancelled", "ApiCancelled", "Inactive"):
            telegram_text(cfg, f"\U0001F6A8 harami_daily_trader: {pos['ticker']} MOC exit REJECTED by IBKR "
                                f"(status {trade.orderStatus.status}) -- position still held, will retry next run. Check IBKR.")
            oversight_log("trader", "harami_live_exit_rejected",
                          f"{pos['ticker']}: MOC SELL order {trade.order.orderId} status {trade.orderStatus.status}; phase left open for retry.")
            continue
        pos["phase"] = "pending_exit"
        pos["exit_order_id"] = trade.order.orderId
        save_state(state)
        telegram_text(cfg, f"\U0001F7E1 <b>HARAMI LIVE EXIT PLACED: {pos['ticker']}</b>\n"
                            f"Market-on-Close SELL {pos['qty']}x, entry was ${pos['entry_price']:.2f} "
                            + (f"(OVERDUE -- was due {pos['exit_date']})." if overdue
                               else f"({HOLD_TRADING_DAYS}-trading-day hold complete)."))
        oversight_log("trader", "harami_live_exit_placed",
                      f"{pos['ticker']}: MOC SELL placed, order {trade.order.orderId}, entry ${pos['entry_price']:.2f}.")


# ── morning recovery mode ────────────────────────────────────────────────
MORNING_CUTOFF = dtime(9, 25)   # Nasdaq's opening cross stops taking MOO orders ~9:28 ET


def run_morning(ib, cfg):
    """Pre-market recovery for MOO entries that never reached the open.

    Found 2026-09-21: of the three MOO buys placed Friday 4:20pm (SOFI, TFC,
    UAL) only TFC was still live at IBKR on Monday morning; the same thing
    happened to PYPL the week before. Every NASDAQ-listed name (PYPL, SOFI,
    UAL) lost its order and every NYSE-listed one (GE, RCL, TFC) kept it --
    6 for 6. Mechanism NOT proven (suspected: NASDAQ drops opening-cross
    orders entered after hours); the script also mutes IBKR's error events and
    only waits 2s after placing, so it never saw whatever happened. Rather than
    depend on the theory, this runs at ~9:05am ET: any pending_entry whose
    order is no longer open AND that holds no position is re-placed for TODAY's
    open -- but only if the signal is from the previous trading day (never chase
    an entry days late) and it still passes the capital gate. Stale ghosts are
    dropped. Never re-places if an open BUY for that contract, or a position,
    already exists -- a duplicate real buy is the failure to avoid here."""
    et = now_et()
    today = et.date()
    if et.time() >= MORNING_CUTOFF:
        print(f"{et:%H:%M} ET is past the {MORNING_CUTOFF:%H:%M} opening-auction cutoff -- morning recovery does nothing.")
        return
    state = load_state()
    open_buy_conids = {t.contract.conId for t in ib.reqAllOpenOrders() if t.order.action == "BUY"}
    for key, pos in list(state["positions"].items()):
        if pos.get("phase") != "pending_entry":
            continue
        conid = pos.get("conid")
        if conid in open_buy_conids:
            print(f"{pos['ticker']}: MOO order still live at IBKR -- nothing to do.")
            continue
        if _portfolio_qty(ib, conid) > 0:
            print(f"{pos['ticker']}: already held -- the evening entry run will confirm it.")
            continue

        # Order gone AND no position: it was dropped somewhere between Friday and now.
        signal_d = date.fromisoformat(pos["signal_date"])
        if _add_trading_days(signal_d, 1) != today:
            del state["positions"][key]
            save_state(state)
            telegram_text(cfg, f"harami_daily_trader: {pos['ticker']} entry from {pos['signal_date']} never filled and its "
                                f"order is gone -- too old to chase, dropped from tracking. (No position, no open order.)")
            oversight_log("trader", "harami_live_entry_abandoned",
                          f"{pos['ticker']}: signal {pos['signal_date']} stale, order gone, no position -- dropped.")
            continue

        spot, contract = get_spot(ib, pos["ticker"])
        budget = cro_cfo_capital_budget(ib, cap_pct=CAP_PCT)
        if spot is None or spot * QTY > budget["per_strategy_cap"] or spot * QTY > budget["headroom"]:
            why = "no live quote" if spot is None else f"1 share ~${spot * QTY:.0f} fails the capital gate"
            del state["positions"][key]
            save_state(state)
            telegram_text(cfg, f"harami_daily_trader: {pos['ticker']} entry order was lost and could not be re-placed ({why}) -- dropped.")
            oversight_log("trader", "harami_live_entry_abandoned", f"{pos['ticker']}: re-place blocked: {why}.")
            continue

        # EARNINGS GATE (added 2026-09-30). This is MARKET-ON-OPEN, so anything
        # reporting before that open fills straight into the gap. The IDR
        # pipeline had NO earnings awareness until today, while the breakout and
        # daytrader scanners both had blackouts -- backwards, since IDR is the
        # strategy that HOLDS for days. Found on the day it mattered: MU reports
        # 2026-09-30 postmarket with a 6.5pct expected move and MU is in the IDR
        # panel. Fails CLOSED -- an unavailable earnings feed blocks the entry
        # rather than buying blind into a binary event.
        try:
            from earnings_guard import reports_before_next_open
            _blocked, _ewhy = reports_before_next_open(pos["ticker"])
        except Exception as _exc:
            _blocked, _ewhy = True, f"earnings guard failed ({type(_exc).__name__})"
        if _blocked:
            print(f"  SKIP {pos['ticker']}: {_ewhy}")
            telegram_text(cfg, f"IDR entry skipped: <b>{pos['ticker']}</b> -- {_ewhy}")
            oversight_log("trader", "harami_live_skipped_earnings",
                          f"{pos['ticker']}: entry skipped -- {_ewhy}.",
                          rationale="MOO fills at the next open; a report before then is "
                                    "an unhedged overnight bet on a binary event.")
            continue

        order = Order(action="BUY", totalQuantity=QTY, orderType="MKT", tif="OPG", transmit=True)
        trade = ib.placeOrder(contract, order)
        ib.sleep(3)
        status = trade.orderStatus.status
        if status in ("Cancelled", "ApiCancelled", "Inactive"):
            telegram_text(cfg, f"\U0001F6A8 harami_daily_trader: re-placed {pos['ticker']} MOO buy was REJECTED (status {status}) -- "
                                f"no entry today. Check IBKR.")
            oversight_log("trader", "harami_live_entry_rejected", f"{pos['ticker']}: morning re-place status {status}.")
            continue
        pos.update({"order_id": trade.order.orderId, "placed_at": et.isoformat(), "expected_spot": spot, "replaced": True})
        save_state(state)
        telegram_text(cfg, f"\U0001F501 <b>HARAMI ENTRY RE-PLACED: {pos['ticker']}</b>\n"
                            f"Friday's MOO order was lost; placed a new Market-on-Open BUY {QTY}x for this morning's open "
                            f"(spot now ~${spot:.2f}, status {status}).")
        oversight_log("trader", "harami_live_entry_replaced",
                      f"{pos['ticker']}: MOO BUY {QTY}x re-placed pre-market, order {trade.order.orderId}, status {status}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["entry", "exit", "morning"], required=True)
    args = ap.parse_args()

    et = now_et()
    if et.weekday() >= 5:
        print("Weekend, nothing to do.")
        return
    if et.year > HOLIDAY_LIST_LAST_YEAR:
        print(f"WARNING: NYSE_HOLIDAYS only covers through {HOLIDAY_LIST_LAST_YEAR} -- "
              f"exit dates fall back to weekday-only. Extend the list.")
    elif not _is_trading_day(et.date()):
        print(f"Market holiday ({et.date()}), nothing to do.")
        return

    cfg = load_cfg()
    ib = IB(); ib.errorEvent += lambda *a: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    ib.reqMarketDataType(1)
    # Record what IBKR actually charged, per execution (added 2026-09-28).
    # trade_journal.entry_price is IBKR's avgCost and already has the
    # commission inside it, so the fee is not recoverable from the journal.
    try:
        import commission_ledger
        commission_ledger.attach(ib, strategy="idr_daily")
    except Exception as exc:
        print(f"commission_ledger attach failed (non-fatal): {exc}")
    try:
        if args.mode == "entry":
            run_entry(ib, cfg)
        elif args.mode == "morning":
            run_morning(ib, cfg)
        else:
            run_exit(ib, cfg)
    finally:
        # Account-wide catch-up before dropping the connection: the event path
        # only fires for THIS client's own orders (see commission_ledger docs).
        try:
            import commission_ledger
            commission_ledger.snapshot(ib, strategy="idr_daily")
        except Exception as exc:
            print(f"commission_ledger snapshot failed (non-fatal): {exc}")
        ib.disconnect()
    print(f"harami_daily_trader --mode {args.mode} done.")


if __name__ == "__main__":
    main()
