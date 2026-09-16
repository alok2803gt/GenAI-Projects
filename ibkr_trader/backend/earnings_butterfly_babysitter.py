"""
Babysitter for the 3 manually-placed ORCL/ADBE earnings butterflies
(2026-09-10 entries, 2026-09-11 expiry), registered in Manual Trader.
One-off intraday job for today's specific trade -- not a permanent
strategy (naturally goes inert once these 3 positions are closed/gone;
safe to leave the scheduled task registered, it just no-ops).

REWRITTEN 2026-09-11 08:45 ET after a real near-miss: the first version
compared live_pnl against a floor that's satisfied whenever a position is
merely close to flat -- which is exactly what stale PRE-MARKET marks look
like (yesterday's close, not today's reality). Run once at 8:35am with
$0 real edge behind it, it fired real close orders (BUY-to-close +
SELL-to-close limits) on the ORCL put fly and ADBE call fly. Caught before
any fill (options don't really trade pre-9:30; all 6 orders sat at
Submitted/Inactive with zero fills), cancelled via DELETE /orders/{id},
and the two positions -- stuck at phase="closing" because the close
coroutine's 120s wrapper timeout outran its own two 60s fill-waits --
were unstuck by a new _mt_reconcile_sync() case 0 in main.py (reverts
closing->open when all legs are still exactly intact at IBKR with zero
working orders). No real money was lost or gained by the mistake itself.

Two corrections baked into this version:
  1. HARD GATE: does nothing before 09:35 ET. Pre-market/early marks are
     not a valid basis for any decision here.
  2. The "salvage" positions no longer use a pnl-floor comparison at all.
     They use a direct, real per-leg bid/ask read to compute today's
     actual net closing value of the spread, matching the plan's real
     wording ("if bid worth anything, close for scrap; if $0.00, let it
     expire") instead of an ambiguous proxy for it.
  3. Requires 2 consecutive fresh (post-09:35) confirmations before
     firing any close, except the ORCL call fly's outright profit-target
     hit, which is unambiguous enough to act on a single fresh read.

Manual Trader already has a REAL, live, continuously-running monitor
(_mt_monitor_coro in main.py) that checks profit_target_usd/stop_loss_usd/
hard_close_time every tick and closes for real -- that's the proven
execution layer, reused here via POST /manual-trader/positions/{id}/close
rather than duplicated. Its hard_close_time=15:55 stays the final backstop
under everything here: if this script never sees a clean signal, MT
closes all 3 positions at 15:55 regardless.

Plan (per CEO conversation, 2026-09-11 morning):
  ORCL call fly (160/167.5/175C, net debit $65.48):
    - Take profit at >=3x (pnl >= ~$130).
    - Protect gains: once a real high-water-mark profit is seen, close if
      it gives back to <=40% of that peak.
    - Cut losses: if ORCL has clearly failed the ~160.6 breakeven (spot
      < 159.50) and the position is underwater, close rather than hold
      into a coin-flip close.
  ORCL put fly (148/140/132P, net debit $139.52) -- likely dead, ORCL ran
  up hard on earnings:
    - If real net closing value (via live bid/ask) is worth more than a
      few dollars, close it for scrap.
    - Otherwise leave it -- MT's own 15:55 hard-close is the backstop.
  ADBE call fly (252.5/262.5/272.5C, net debit $116.52) -- likely a loss:
    - Same real-bid salvage check as the ORCL put fly.

Runs every ~10 min, 9:30-16:00 ET, weekdays, via IBKR-EarningsButterflyBabysitter.
"""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

HERE = Path(__file__).parent
CFG_PATH = HERE / "scanner_config.json"
STATE_PATH = HERE / "earnings_butterfly_babysitter_state.json"
OVERSIGHT_LOG = HERE / "oversight_log.jsonl"
BACKEND_URL = "http://localhost:8000"

ET = ZoneInfo("America/New_York")
ACT_NOT_BEFORE = "09:35"   # hard gate -- no decisions on pre-market/early marks
SALVAGE_MIN_CREDIT = 5.0   # real $ value below which scrap isn't worth a commission

POSITIONS = {
    "ORCL_custom_20260910_153531": {
        "kind": "live", "ticker": "ORCL", "net_entry": -65.48,
        "profit_target": 130.0, "protect_gains_floor_pct": 0.40, "protect_gains_min_peak": 40.0,
        "breakeven_spot": 159.50, "cut_loss_pnl": -20.0,
    },
    "ORCL_custom_20260910_153534": {
        "kind": "salvage", "ticker": "ORCL",
        "legs": [
            {"local_symbol": "ORCL  260911P00148000", "strike": 148.0, "right": "P", "action": "BUY", "qty": 1},
            {"local_symbol": "ORCL  260911P00140000", "strike": 140.0, "right": "P", "action": "SELL", "qty": 2},
            {"local_symbol": "ORCL  260911P00132000", "strike": 132.0, "right": "P", "action": "BUY", "qty": 1},
        ],
        "expiry": "20260911",
    },
    "ADBE_custom_20260910_160122": {
        "kind": "salvage", "ticker": "ADBE",
        "legs": [
            {"local_symbol": "ADBE  260911C00252500", "strike": 252.5, "right": "C", "action": "BUY", "qty": 1},
            {"local_symbol": "ADBE  260911C00262500", "strike": 262.5, "right": "C", "action": "SELL", "qty": 2},
            {"local_symbol": "ADBE  260911C00272500", "strike": 272.5, "right": "C", "action": "BUY", "qty": 1},
        ],
        "expiry": "20260911",
    },
}


def now_et():
    return datetime.now(ET)


def load_cfg():
    return json.loads(CFG_PATH.read_text())


def telegram_text(cfg, text):
    import re
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


def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"high_water_mark": {}, "confirm_count": {}}


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2))


def get_mt_positions():
    r = requests.get(f"{BACKEND_URL}/manual-trader/positions", timeout=15)
    r.raise_for_status()
    return r.json().get("positions", {})


def safe_px(v):
    try:
        f = float(v)
        return f if f > 0 and f == f else None  # f == f is False for NaN
    except (TypeError, ValueError):
        return None


def get_spot(ib, ticker):
    from ib_insync import Stock
    stk = ib.qualifyContracts(Stock(ticker, "SMART", "USD"))[0]
    [t] = ib.reqTickers(stk)
    return safe_px(t.marketPrice()) or safe_px(t.close)


def get_salvage_net_credit(ib, cfgd):
    """Real net $ credit to close this spread right now, using live bid/ask
    per leg (not a stale MT live_pnl figure). Positive = worth closing for."""
    from ib_insync import Option
    contracts = [Option(cfgd["ticker"], cfgd["expiry"], leg["strike"], leg["right"], "SMART", "100", "USD")
                 for leg in cfgd["legs"]]
    qualified = ib.qualifyContracts(*contracts)
    if len(qualified) != len(cfgd["legs"]):
        return None
    tickers = ib.reqTickers(*qualified)
    net = 0.0
    for leg, t in zip(cfgd["legs"], tickers):
        bid, ask = safe_px(t.bid), safe_px(t.ask)
        if bid is None and ask is None:
            last = safe_px(t.last) or safe_px(t.close)
            if last is None:
                return None  # no usable price for this leg -- don't guess
            bid = ask = last
        elif bid is None:
            bid = ask
        elif ask is None:
            ask = bid
        # Closing a long (BUY entry) = SELL now -> use bid (conservative).
        # Closing a short (SELL entry) = BUY now -> use ask (conservative).
        px = bid if leg["action"] == "BUY" else ask
        sign = 1 if leg["action"] == "BUY" else -1  # selling a long = +credit, buying back a short = -cost
        net += sign * px * leg["qty"] * 100
    return round(net, 2)


def close_position(cfg, pos_id, name, reason, live_pnl):
    try:
        r = requests.post(f"{BACKEND_URL}/manual-trader/positions/{pos_id}/close", timeout=30)
        ok = r.status_code == 200
    except Exception as e:
        ok = False
        reason += f" (close call failed: {e})"
    tag = "CLOSED" if ok else "CLOSE FAILED"
    telegram_text(cfg, f"\U0001FA82 <b>Earnings butterfly babysitter: {tag} {name}</b>\n"
                        f"Reason: {reason}\nLive P&L at decision: ${live_pnl:+.2f}")
    oversight_log("trader", "earnings_butterfly_babysitter_close" if ok else "earnings_butterfly_babysitter_close_failed",
                  f"{name}: {tag} -- {reason}. live_pnl=${live_pnl:+.2f}",
                  rationale="Following the 2026-09-11 morning plan for the ORCL/ADBE earnings butterflies.",
                  pnl_impact=live_pnl if ok else None)
    return ok


def main():
    et = now_et()
    if et.weekday() >= 5:
        print("Weekend, nothing to do.")
        return
    if et.strftime("%H:%M") < ACT_NOT_BEFORE:
        print(f"Before {ACT_NOT_BEFORE} ET gate -- marks aren't trustworthy yet, doing nothing.")
        return

    cfg = load_cfg()
    state = load_state()
    hwm = state.setdefault("high_water_mark", {})
    confirm = state.setdefault("confirm_count", {})

    mt_positions = get_mt_positions()
    open_tracked = {pid: cfgd for pid, cfgd in POSITIONS.items() if pid in mt_positions and mt_positions[pid].get("phase") == "open"}
    if not open_tracked:
        print("No tracked earnings-butterfly positions still open -- nothing to do.")
        return

    ib = None
    if any(cfgd["kind"] in ("live", "salvage") for cfgd in open_tracked.values()):
        from ib_insync import IB
        ib = IB(); ib.errorEvent += lambda *a: None
        ib.connect("127.0.0.1", 7496, clientId=1944, timeout=20)
        ib.reqMarketDataType(1)

    try:
        spot_cache = {}
        for pid, cfgd in open_tracked.items():
            pos = mt_positions[pid]
            name = pos["name"]
            live_pnl = pos.get("live_pnl")
            if live_pnl is None:
                print(f"  [{pid}] no live_pnl yet (quotes not flowing) -- skipping this cycle")
                confirm[pid] = 0
                continue

            if cfgd["kind"] == "live":
                prev_hwm = hwm.get(pid, live_pnl)
                new_hwm = max(prev_hwm, live_pnl)
                hwm[pid] = new_hwm

                if live_pnl >= cfgd["profit_target"]:
                    close_position(cfg, pid, name, f"profit target hit (>= ${cfgd['profit_target']:.0f}, 3x+)", live_pnl)
                    confirm[pid] = 0
                    continue

                fire_reason = None
                if new_hwm >= cfgd["protect_gains_min_peak"] and live_pnl <= new_hwm * cfgd["protect_gains_floor_pct"]:
                    fire_reason = (f"protecting gains -- peaked at ${new_hwm:+.2f}, now back to ${live_pnl:+.2f} "
                                   f"(<= {cfgd['protect_gains_floor_pct']:.0%} of peak)")
                else:
                    if cfgd["ticker"] not in spot_cache:
                        spot_cache[cfgd["ticker"]] = get_spot(ib, cfgd["ticker"])
                    spot = spot_cache[cfgd["ticker"]]
                    if spot is not None and spot < cfgd["breakeven_spot"] and live_pnl < cfgd["cut_loss_pnl"]:
                        fire_reason = f"breakeven failed -- {cfgd['ticker']} ${spot:.2f} < ${cfgd['breakeven_spot']:.2f}, pnl ${live_pnl:+.2f}"

                if fire_reason:
                    confirm[pid] = confirm.get(pid, 0) + 1
                    if confirm[pid] >= 2:
                        close_position(cfg, pid, name, fire_reason + " (confirmed 2x)", live_pnl)
                        confirm[pid] = 0
                    else:
                        print(f"  [{pid}] {fire_reason} -- 1st confirmation, will act if it repeats next cycle")
                else:
                    confirm[pid] = 0
                    print(f"  [{pid}] holding: live_pnl=${live_pnl:+.2f} hwm=${new_hwm:+.2f}")

            else:  # "salvage" -- real bid/ask based, not a pnl-floor guess
                net_credit = get_salvage_net_credit(ib, cfgd)
                if net_credit is None:
                    print(f"  [{pid}] no usable quotes for salvage check this cycle -- skipping")
                    confirm[pid] = 0
                    continue
                if net_credit > SALVAGE_MIN_CREDIT:
                    confirm[pid] = confirm.get(pid, 0) + 1
                    if confirm[pid] >= 2:
                        close_position(cfg, pid, name,
                                      f"real salvage value confirmed 2x (net closing credit ${net_credit:+.2f} "
                                      f"via live bid/ask > ${SALVAGE_MIN_CREDIT:.0f} floor)", live_pnl)
                        confirm[pid] = 0
                    else:
                        print(f"  [{pid}] real closing credit ${net_credit:+.2f} -- 1st confirmation")
                else:
                    confirm[pid] = 0
                    print(f"  [{pid}] near-worthless (net closing credit ${net_credit:+.2f}) -- leaving for MT's 15:55 hard-close")

        state["confirm_count"] = confirm
        save_state(state)
    finally:
        if ib is not None:
            ib.disconnect()


if __name__ == "__main__":
    main()
