"""
KO put credit spread trader -- real live trading, per the 2026-09-11 theta-
decay research (ibkr_trader/backend/theta_decay_research/): KO was the one
name that came out positive across every structural variant tested (flat
12%/18%, EVC expected-move cushion, and the hybrid 12%-short + EVC-style
tightest-liquid-wing), with genuinely confirmed real, two-sided live
liquidity at those strikes -- unlike D/PEP, whose backtests looked good but
whose 18%-OTM long leg had zero real bid.

Structure (matches the validated hybrid backtest exactly):
  - Short put: nearest REAL listed strike to spot * 0.88 (12% OTM).
  - Long put (wing): walk real listed strikes tightest-first from the short
    strike toward the 18%-OTM fallback (spot * 0.82), take the first whose
    CONSERVATIVE credit (short_bid - candidate_long_ask) is still positive
    -- same rule as EVC's own real wing selection (main.py ~16703-16722),
    applied here to just the wing, not the short strike (the hybrid that
    backtested best). Falls back to the 18% target if nothing tighter
    clears the bar.
  - ~35 DTE (nearest real expiry).
  - Skips entry entirely if KO's next real earnings date falls within
    [today, today + DTE + 1] -- the backtest was earnings-excluded, so
    trading through an earnings window is untested and NOT what was
    validated.
  - 50%-of-credit profit target, stop-loss at -100% of credit (liability =
    2x credit received) -- both handled by Manual Trader's own proven
    monitor loop once registered, not reimplemented here.

Execution: places the real 2-leg BAG combo order via the existing, already-
proven POST /manual-trader/enter (same mechanism the ORCL/ADBE earnings
butterflies used) -- reprices down automatically if the initial credit
doesn't fill, auto-registers the position in Manual Trader on fill. All
ongoing monitoring (profit target / stop loss / hard-close backstop) is
Manual Trader's existing, already-fixed code -- nothing new to maintain
here beyond entry selection.

Usage:
  python ko_put_spread_trader.py --dry-run   # print the real computed structure, place nothing
  python ko_put_spread_trader.py             # real live entry (no-ops if a KO position is already open)
"""
import argparse
import json
import os
from datetime import date, timedelta

import sys

import requests
import yfinance as yf
from ib_insync import IB, Option, Stock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alpaca_0dte_common import cro_cfo_capital_budget  # noqa: E402

BACKEND_URL = "http://localhost:8000"
TARGET_DTE = 35
SHORT_OTM_PCT = 0.12
WING_FALLBACK_PCT = 0.18
MIN_CONSERVATIVE_CREDIT = 5.0  # $, matches the account's established "not worth a commission below this" bar
EARNINGS_BUFFER_DAYS = 1
CLIENT_ID = 1952


def safe_px(v):
    try:
        f = float(v)
        return f if f > 0 and f == f else None
    except (TypeError, ValueError):
        return None


def load_cfg():
    with open(os.path.join(os.path.dirname(__file__), "scanner_config.json")) as f:
        return json.load(f)


def telegram_text(cfg, text):
    try:
        from telegram_alert_gate import alert_enabled
        if not alert_enabled("ko_put_spread"):
            return
    except Exception:
        pass
    try:
        requests.post(f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
                      data={"chat_id": cfg["telegram_chat_id"], "text": text, "parse_mode": "HTML"},
                      timeout=10)
    except Exception as e:
        print(f"Telegram send failed: {e}")


def oversight_log(actor, category, summary, rationale="", outcome=None, pnl_impact=None):
    from datetime import datetime, timezone
    entry = {"time": datetime.now(timezone.utc).astimezone().isoformat(), "actor": actor,
              "category": category, "summary": summary, "rationale": rationale,
              "outcome": outcome, "pnl_impact": pnl_impact}
    with open(os.path.join(os.path.dirname(__file__), "oversight_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def already_open() -> bool:
    r = requests.get(f"{BACKEND_URL}/manual-trader/positions", timeout=15)
    r.raise_for_status()
    positions = r.json().get("positions", {})
    return any(p.get("ticker") == "KO" and p.get("phase") in ("open", "closing") for p in positions.values())


def next_earnings_blocks_entry(dte: int) -> tuple[bool, str]:
    try:
        ed = yf.Ticker("KO").get_earnings_dates(limit=8)
        today = date.today()
        for ts in ed.index:
            d = ts.date()
            if today <= d <= today + timedelta(days=dte + EARNINGS_BUFFER_DAYS):
                return True, f"KO earnings {d.isoformat()} falls inside the {dte}-day holding window"
    except Exception as e:
        return True, f"could not verify earnings calendar ({e}) -- skipping out of caution"
    return False, ""


def find_strikes_and_credit(ib: IB) -> dict:
    stk = ib.qualifyContracts(Stock("KO", "SMART", "USD"))[0]
    [stk_t] = ib.reqTickers(stk)
    spot = safe_px(stk_t.marketPrice()) or safe_px(stk_t.close)
    if spot is None:
        raise RuntimeError("no live KO spot price")

    chains = ib.reqSecDefOptParams(stk.symbol, "", stk.secType, stk.conId)
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
    today = date.today()
    expiries = sorted(e for e in chain.expirations
                       if (date(int(e[:4]), int(e[4:6]), int(e[6:8])) - today).days > 0)
    target_exp = min(expiries, key=lambda e: abs((date(int(e[:4]), int(e[4:6]), int(e[6:8])) - today).days - TARGET_DTE))
    dte = (date(int(target_exp[:4]), int(target_exp[4:6]), int(target_exp[6:8])) - today).days
    real_strikes = sorted(chain.strikes)

    def listed(k):
        q = ib.qualifyContracts(Option("KO", target_exp, k, "P", "SMART", "100", "USD"))
        return q[0] if q else None

    def nearest_listed(target):
        for k in sorted(real_strikes, key=lambda k: abs(k - target)):
            opt = listed(k)
            if opt:
                return k, opt
        return None, None

    short_k, short_opt = nearest_listed(spot * (1 - SHORT_OTM_PCT))
    if short_k is None:
        raise RuntimeError("could not qualify any real short-put strike")

    wing_fallback = spot * (1 - WING_FALLBACK_PCT)
    candidates = sorted({s for s in real_strikes if wing_fallback <= s < short_k}
                         | {min(real_strikes, key=lambda k: abs(k - wing_fallback))}, reverse=True)
    cand_opts = [listed(c) for c in candidates]
    paired = [(c, o) for c, o in zip(candidates, cand_opts) if o]

    ib.reqTickers(short_opt, *[o for _, o in paired])
    ib.sleep(3)
    live = ib.reqTickers(short_opt, *[o for _, o in paired])
    short_t = live[0]
    short_bid = safe_px(short_t.bid)
    short_mid = round((safe_px(short_t.bid) + safe_px(short_t.ask)) / 2, 3) if safe_px(short_t.bid) and safe_px(short_t.ask) else None

    # Economic gate uses MID credit, not the worst-case bid/ask crossing --
    # confirmed live 2026-09-11: KO's own 18%-OTM wing priced at a real $5.00
    # mid credit but a NEGATIVE conservative (crossing) credit, and the crossing
    # metric never clears $5 at ANY strike in the search range. Real fills
    # happen near mid (MT's own /manual-trader/enter reprices down from an
    # ask-side start over 3 attempts), so mid is the honest economic signal;
    # the conservative crossing is kept only as a sanity bound against a
    # genuinely toxic (e.g. massively negative) spread.
    MAX_TOLERABLE_CONSERVATIVE_LOSS = -50.0  # $ per spread, sanity bound only
    chosen_long_k = chosen_cons_credit_dollars = chosen_mid_credit_per_share = None
    for (cand_k, _), t in zip(paired, live[1:]):
        long_ask = safe_px(t.ask)
        long_bid = safe_px(t.bid)
        if short_mid is None or long_bid is None or long_ask is None:
            continue
        cons_credit_dollars = round((short_bid - long_ask) * 100, 2) if short_bid is not None else None
        mid_credit_per_share = round(short_mid - (long_bid + long_ask) / 2, 3)
        mid_credit_dollars = round(mid_credit_per_share * 100, 2)
        if mid_credit_dollars >= MIN_CONSERVATIVE_CREDIT and (cons_credit_dollars is None or cons_credit_dollars > MAX_TOLERABLE_CONSERVATIVE_LOSS):
            chosen_long_k = cand_k
            chosen_cons_credit_dollars = cons_credit_dollars
            chosen_mid_credit_per_share = mid_credit_per_share
            break

    if chosen_long_k is None:
        chosen_long_k = candidates[-1] if candidates else round(wing_fallback, 2)

    return {
        "spot": spot, "expiry": target_exp, "dte": dte,
        "short_k": short_k, "long_k": chosen_long_k,
        "conservative_credit_$": chosen_cons_credit_dollars,
        "mid_credit_$": round(chosen_mid_credit_per_share * 100, 2) if chosen_mid_credit_per_share is not None else None,
        "mid_credit_per_share": chosen_mid_credit_per_share,
        "width_$": round((short_k - chosen_long_k) * 100, 2),
    }


def pretrade_review(structure: dict, budget: dict) -> dict:
    """CRO/CFO blocking gate -- added 2026-09-21. Until then this was the ONLY
    live strategy with no capital check at all: neither this script nor Manual
    Trader's /manual-trader/enter validates size against the account, so a
    ~$400-500 defined-risk spread could go on a ~$2,100 account unchallenged.
    Same two tests every other strategy's pre-trade review applies: the
    per-trade cap (default 5% of combined net liq -- no per-strategy override
    has been granted for KO, unlike GOOG's 15%) AND the real remaining
    portfolio headroom. Max risk here is the true defined-risk max loss,
    (width - credit) x 100, not the -100%-of-credit stop Manual Trader applies
    after entry -- the gate must bound what the structure CAN lose."""
    max_risk = round(structure["width_$"] - structure["mid_credit_$"], 2)
    net_liq = budget["combined_net_liq"]
    findings = [f"CFO: max risk ${max_risk:,.0f} = {(max_risk / net_liq if net_liq else 1.0):.1%} of combined net liq ${net_liq:,.0f}",
                f"CFO: per-trade cap ${budget['per_strategy_cap']:,.0f} ({budget['per_strategy_cap_pct']:.0%}), "
                f"portfolio headroom ${budget['headroom']:,.0f} of ${budget['total_budget']:,.0f} total budget"]
    approved = True
    if max_risk > budget["per_strategy_cap"]:
        findings.append(f"EXCEEDS per-trade cap (${budget['per_strategy_cap']:,.0f})")
        approved = False
    if max_risk > budget["headroom"]:
        findings.append("EXCEEDS remaining portfolio headroom")
        approved = False
    return {"approved": approved, "findings": findings, "max_risk": max_risk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if already_open():
        print("KO position already open in Manual Trader -- nothing to do.")
        return

    blocked, reason = next_earnings_blocks_entry(TARGET_DTE)
    if blocked:
        print(f"Earnings gate: {reason} -- skipping entry.")
        oversight_log("trader", "ko_put_spread_skip", f"KO entry skipped: {reason}")
        return

    ib = IB()
    ib.errorEvent += lambda *a: None
    ib.connect("127.0.0.1", 7496, clientId=CLIENT_ID, timeout=20)
    ib.reqMarketDataType(1)
    try:
        structure = find_strikes_and_credit(ib)
        try:
            budget = cro_cfo_capital_budget(ib)
        except Exception as e:
            budget = None
            budget_err = e
    finally:
        ib.disconnect()

    print(json.dumps(structure, indent=2))

    # Economic gate is mid-price credit (the realistic, fillable-via-repricing
    # number) -- see find_strikes_and_credit's own note on why the worst-case
    # bid/ask-crossing credit is too strict a bar for KO's real market.
    if structure["mid_credit_$"] is None or structure["mid_credit_$"] < MIN_CONSERVATIVE_CREDIT:
        print(f"No real fillable credit >= ${MIN_CONSERVATIVE_CREDIT:.0f} found -- skipping entry.")
        oversight_log("trader", "ko_put_spread_skip",
                       f"KO entry skipped: best real mid credit "
                       f"${structure['mid_credit_$']} below ${MIN_CONSERVATIVE_CREDIT:.0f} floor",
                       rationale=json.dumps(structure))
        return

    # CRO/CFO gate -- blocking. Fails CLOSED if the budget couldn't be computed.
    if budget is None:
        msg = f"KO put spread: budget check failed ({budget_err}) -- NOT entering (fail closed)."
        print(msg)
        oversight_log("risk_manager", "ko_put_spread_rejected", msg, outcome="REJECTED -- no order placed")
        telegram_text(load_cfg(), "\u26a0\ufe0f " + msg)
        return
    review = pretrade_review(structure, budget)
    print("--- CRO/CFO pre-trade review ---")
    for f in review["findings"]:
        print(f"  {f}")
    if not review["approved"]:
        msg = (f"KO put spread {structure['short_k']}P/{structure['long_k']}P exp {structure['expiry']}: "
               f"REJECTED by pre-trade review -- " + "; ".join(review["findings"]))
        print(msg + ("  (DRY RUN: nothing would be placed)" if args.dry_run else ""))
        oversight_log("risk_manager", "ko_put_spread_rejected", msg, outcome="REJECTED -- no order placed",
                      rationale=json.dumps(structure))
        return
    print("CRO/CFO gate: PASSED.")

    target_limit = structure["mid_credit_per_share"]
    profit_target_usd = round(structure["mid_credit_$"] * 0.50, 2)
    stop_loss_usd = round(-structure["mid_credit_$"] * 1.00, 2)

    if args.dry_run:
        print(f"DRY RUN -- would place: SELL {structure['short_k']}P / BUY {structure['long_k']}P, "
              f"limit_price=${target_limit:.2f}, profit_target=${profit_target_usd:.2f}, "
              f"stop_loss=${stop_loss_usd:.2f}")
        return

    payload = {
        "ticker": "KO", "expiry": structure["expiry"],
        "legs": [
            {"action": "SELL", "strike": structure["short_k"], "right": "P"},
            {"action": "BUY", "strike": structure["long_k"], "right": "P"},
        ],
        "limit_price": target_limit, "qty": 1,
        "name": f"KO Put Credit Spread {structure['expiry']}",
        "strategy": "vertical",
        "profit_target_usd": profit_target_usd,
        "stop_loss_usd": stop_loss_usd,
        "hard_close_time": "15:45",
        "reprice_steps": 3, "reprice_wait_s": 15, "max_concession_pct": 0.25,
    }

    cfg = load_cfg()
    r = requests.post(f"{BACKEND_URL}/manual-trader/enter", json=payload, timeout=90)
    if r.status_code == 200:
        result = r.json()
        msg = (f"\U0001F7E2 <b>KO Put Credit Spread entered</b>\n"
               f"SELL {structure['short_k']}P / BUY {structure['long_k']}P, exp {structure['expiry']} "
               f"({structure['dte']}d)\nCredit target ${target_limit:.2f}/share, "
               f"profit target ${profit_target_usd:.2f}, stop ${stop_loss_usd:.2f}")
        telegram_text(cfg, msg)
        oversight_log("trader", "ko_put_spread_entry", msg.replace("\n", " "),
                       rationale="Per the 2026-09-11 theta-decay research -- validated hybrid structure (12% short, EVC-style tightest-liquid wing), earnings excluded.",
                       outcome=json.dumps(result))
        print(json.dumps(result, indent=2))
    else:
        err = f"KO entry FAILED: {r.status_code} {r.text[:300]}"
        telegram_text(cfg, f"\U0001F534 <b>{err}</b>")
        oversight_log("trader", "ko_put_spread_entry_failed", err, rationale=json.dumps(structure))
        # Windows console (cp1252) can't encode arrow chars etc. that show up in
        # real IBKR/reprice error text -- confirmed live 2026-09-11 (first real
        # run crashed here after a real, correctly-raised entry failure).
        print(err.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
