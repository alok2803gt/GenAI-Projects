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
in two modes / two scheduled tasks:

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

Trading-day arithmetic (_add_trading_days) is a plain weekday skip, not a
real market-holiday calendar -- can be off by a day around a holiday.
Flagged rather than silently assumed exact, same disclosure discipline
this account's other approximations (macro-calendar fallback lists,
Black-Scholes premium estimates) already use.

Usage: python harami_daily_trader.py --mode entry|exit
"""
import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, date
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
HOLD_TRADING_DAYS = 5
CAP_PCT = 0.25   # CEO override 2026-09-11 -- see module docstring

# Same alert format harami_scanner.py writes -- matches
# candlestick_pattern_research/harami_daily_forward_performance.py's regex.
_RE = re.compile(
    r"Bullish Harami \+ downtrend:\s*(?P<ticker>[A-Z][A-Z.\-]*)\s*\((?P<date>\d{4}-\d{2}-\d{2})\)",
)


def now_et():
    return datetime.now(ET)


def load_cfg():
    return json.loads(CFG_PATH.read_text())


def telegram_text(cfg, text):
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


def _add_trading_days(d: date, n: int) -> date:
    """Plain weekday-skip -- NOT a real market-holiday calendar. Can be off
    by a day around a holiday; see module docstring."""
    cur = d
    added = 0
    while added < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
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


def journal_insert_open(ticker, entry_price, opened_at):
    try:
        con = sqlite3.connect(JOURNAL_DB_PATH)
        con.execute("""INSERT INTO trade_journal
            (opened_at, ticker, action, qty, entry_price, strategy_type, is_paper, notes)
            VALUES (?, ?, 'BUY', ?, ?, 'HARAMI_DAILY', 0, ?)""",
            (opened_at, ticker, QTY, entry_price,
             "harami daily live entry -- backtested 5d hold, p=0.00007"))
        con.commit()
        con.close()
    except Exception as e:
        print(f"  journal_insert_open failed: {e}")


def journal_close(ticker, closed_at, exit_price, pnl, exit_reason):
    try:
        con = sqlite3.connect(JOURNAL_DB_PATH)
        con.execute("""UPDATE trade_journal SET
            closed_at=?, exit_price=?, pnl=?, win=?, exit_reason=?
            WHERE id = (SELECT id FROM trade_journal
                        WHERE ticker=? AND strategy_type='HARAMI_DAILY' AND closed_at IS NULL
                        ORDER BY id DESC LIMIT 1)""",
            (closed_at, exit_price, pnl, 1 if pnl > 0 else 0, exit_reason, ticker))
        con.commit()
        con.close()
    except Exception as e:
        print(f"  journal_close failed: {e}")


def _fill_info(ib, order_id):
    """Real fill status/price for a previously-placed order, by orderId.
    Returns (filled: bool, avg_price: float|None)."""
    for tr in ib.trades():
        if tr.order.orderId == order_id:
            st = tr.orderStatus
            if st.status == "Filled" and safe_px(st.avgFillPrice):
                return True, safe_px(st.avgFillPrice)
            return st.status == "Filled", None
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
        filled, px = _fill_info(ib, pos["order_id"])
        if not filled:
            age_days = (date.today() - date.fromisoformat(pos["signal_date"])).days
            if age_days >= 2:
                telegram_text(cfg, f"⚠️ harami_daily_trader: {pos['ticker']} MOO entry from "
                                    f"{pos['signal_date']} still not filled after {age_days}d -- check IBKR.")
            continue
        entry_price = px or pos.get("expected_spot")
        fill_date_str = et.strftime("%Y-%m-%d")
        exit_date = _add_trading_days(date.fromisoformat(fill_date_str), HOLD_TRADING_DAYS)
        pos.update({"phase": "open", "entry_price": entry_price, "entry_fill_date": fill_date_str,
                    "exit_date": exit_date.isoformat()})
        journal_insert_open(pos["ticker"], entry_price, et.isoformat())
        telegram_text(cfg, f"✅ <b>HARAMI LIVE ENTRY FILLED: {pos['ticker']}</b>\n"
                            f"Bought {QTY}x @ ${entry_price:.2f} (MOO). Exit planned {exit_date.isoformat()} "
                            f"(close, {HOLD_TRADING_DAYS} trading days).")
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
def run_exit(ib, cfg):
    et = now_et()
    today_str = et.strftime("%Y-%m-%d")
    state = load_state()

    # 1. Confirm any MOC sell placed by a prior exit run.
    for key, pos in list(state["positions"].items()):
        if pos.get("phase") != "pending_exit":
            continue
        filled, px = _fill_info(ib, pos["exit_order_id"])
        if not filled:
            telegram_text(cfg, f"⚠️ harami_daily_trader: {pos['ticker']} MOC exit not yet confirmed filled -- check IBKR.")
            continue
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

    # 2. Place MOC exits for any position whose planned exit date is today.
    for key, pos in list(state["positions"].items()):
        if pos.get("phase") != "open" or pos.get("exit_date") != today_str:
            continue
        contract = Stock(pos["ticker"], "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            telegram_text(cfg, f"harami_daily_trader: {pos['ticker']} exit due today but could not qualify contract -- check IBKR.")
            continue
        order = Order(action="SELL", totalQuantity=pos["qty"], orderType="MOC", tif="DAY", transmit=True)
        trade = ib.placeOrder(qualified[0], order)
        ib.sleep(2)
        pos["phase"] = "pending_exit"
        pos["exit_order_id"] = trade.order.orderId
        save_state(state)
        telegram_text(cfg, f"\U0001F7E1 <b>HARAMI LIVE EXIT PLACED: {pos['ticker']}</b>\n"
                            f"Market-on-Close SELL {pos['qty']}x, entry was ${pos['entry_price']:.2f} "
                            f"({HOLD_TRADING_DAYS}-trading-day hold complete).")
        oversight_log("trader", "harami_live_exit_placed",
                      f"{pos['ticker']}: MOC SELL placed, order {trade.order.orderId}, entry ${pos['entry_price']:.2f}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["entry", "exit"], required=True)
    args = ap.parse_args()

    et = now_et()
    if et.weekday() >= 5:
        print("Weekend, nothing to do.")
        return

    cfg = load_cfg()
    ib = IB(); ib.errorEvent += lambda *a: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    ib.reqMarketDataType(1)
    try:
        if args.mode == "entry":
            run_entry(ib, cfg)
        else:
            run_exit(ib, cfg)
    finally:
        ib.disconnect()
    print(f"harami_daily_trader --mode {args.mode} done.")


if __name__ == "__main__":
    main()
