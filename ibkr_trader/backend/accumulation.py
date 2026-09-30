"""
Valuation-gated accumulation for Inside Day Reversal names.

The IDR signal by itself has no measured edge (market-adjusted, date-clustered
excess +0.088pp, t=0.48 -- see the 2026-09-24 audit), and averaging down into
a no-edge technical trade only scales variance: dca_backtest.py showed the mean
loss when wrong going from -6.4% to -10.4% for a 6% gain in capital efficiency.

So this does something different. It treats an IDR signal as a *timing* input
and lets FUNDAMENTALS decide whether a name deserves capital at all. A name
qualifies only if filed numbers say it is cheap on its own merits -- the tests
RTX and TFC pass today and BA, RIVN, SOFI, RCL, NKE fail:

  operating companies                      lenders
    normalized FCF > 0                       ROE >= 80% of cost of equity
    implied growth <= delivered growth       price <= 1.0x residual-income value
    net debt <= 5x normalized FCF
    capex <= 25% of revenue (not a
      perpetual capital sink)

Qualifying names get a PLANNED accumulation: a target weight, equal tranches,
minimum spacing between adds, and a valuation exit -- not a 10-day clock. Names
that fail stay on the ordinary IDR exit rule. Nothing is averaged down on a
name that is merely falling.

    ../venv/bin/python accumulation.py              # report eligibility + the plan
    ../venv/bin/python accumulation.py --execute    # place the next due tranche
"""
import argparse
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

import valuation as V

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
STATE = HERE / "accumulation_state.json"
TARGET_WEIGHT = 0.10        # of net liquidation, per qualifying name
# CEO exception 2026-09-29: a percentage-of-net-liq target cannot tranche a
# high-priced name on a small account -- RTX at $187/share is 13% of a $1,448
# net liq, so one share already overshot the 10% target and the model correctly
# refused to ever accumulate it. A flat dollar target per name is used instead
# when set, which makes RTX tranche-able (2 shares = $374 of the $420 target).
#
# CONCENTRATION THIS IMPLIES, stated plainly: $420 is ~29% of a $1,448 net liq
# per name, and MAX_NAMES=3 would be ~87% of the account in three averaged-down
# IDR names. Available funds ($548) cannot actually fund three, so in practice
# the per-run cash check will gate it -- but the intent is materially more
# concentrated than the 10% design. The valuation screen is the thing keeping
# that honest: only names whose delivered cash generation supports the price
# get here at all.
TARGET_DOLLARS_OVERRIDE = 420.0   # set to None to go back to TARGET_WEIGHT
TRANCHES = 3
ADD_TRIGGERS = [-5.0, -10.0]   # % below the first tranche's price
MIN_SPACING_DAYS = 5
MAX_NAMES = 3               # portfolio-level cap on concurrent accumulations
EXIT_WHEN_IMPLIED_EXCEEDS_DELIVERED_BY = 0.05


def eligibility(o):
    """(eligible, reason) from a valuation.analyse() result."""
    t = o.get("ticker")
    if o.get("error"):
        return False, o["error"]
    if o.get("model") == "residual income":
        roe, coe, pf = o.get("roe"), o.get("coe"), o.get("price_to_fair")
        if roe is None or pf is None:
            return False, "lender: no usable ROE/book"
        if roe < 0.8 * coe:
            return False, f"lender: ROE {roe:.1%} below 80% of {coe:.1%} cost of equity"
        if pf > 1.0:
            return False, f"lender: {pf:.2f}x residual-income value (not cheap)"
        return True, f"lender: ROE {roe:.1%} vs {coe:.1%} COE, {pf:.2f}x fair value"
    if o.get("model") == "cash runway":
        return False, f"burns cash (${o.get('burn', 0)/1e9:.1f}B/yr) -- no valuation support"
    base = o.get("base_fcf")
    ig, gh = o.get("implied_g"), o.get("fcf_cagr")
    if not base or base <= 0:
        return False, "no positive normalized free cash flow"
    if not isinstance(ig, float) or abs(ig) > 5:
        return False, "price implies implausible growth (cannot solve)"
    if gh is None:
        return False, "no clean free-cash-flow history to judge against"
    if ig > gh:
        return False, f"price needs {ig:+.1%}/yr vs {gh:+.1%} delivered"
    if o.get("net_debt") and base and o["net_debt"] > 5 * base:
        return False, f"net debt ${o['net_debt']/1e9:.1f}B is over 5x free cash flow"
    if o.get("capex_pct_rev") and o["capex_pct_rev"] > 0.25:
        return False, f"capex {o['capex_pct_rev']:.0%} of revenue -- perpetual capital sink"
    return True, f"needs {ig:+.1%}/yr vs {gh:+.1%} delivered, {o['wacc']:.1%} WACC"


CACHE = HERE / "valuation_cache.json"
# APPEND-ONLY audit trail. valuation_cache.json is OVERWRITTEN on every refresh,
# so after a refresh there is no way to reconstruct the numbers that justified a
# past decision -- a gap made concrete on 2026-09-30, when force-refreshing the
# cache destroyed the (buggy, debt-understating) figures that had justified RTX
# and TFC eligibility for the previous five days. Every evaluation is now also
# appended here with the logic revision that produced it, and nothing is ever
# rewritten.
DECISION_LOG = HERE / "valuation_decisions.jsonl"
CACHE_DAYS = 7          # fundamentals move quarterly; refetching per run is wasteful and rate-limited


def log_decision(ticker, eligible, reason, summary, cached):
    """Append one immutable row. Never raises -- an audit-trail failure must not
    stop a trade being screened."""
    try:
        row = {"ts": datetime.now(ET).isoformat(), "ticker": ticker,
               "eligible": bool(eligible), "reason": reason,
               "logic_rev": getattr(V, "VALUATION_LOGIC_REV", "?"),
               "cached": bool(cached)}
        row.update({k: v for k, v in (summary or {}).items()})
        with open(DECISION_LOG, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception as exc:
        print(f"  (valuation decision log failed: {type(exc).__name__}: {exc})")


def screen(ticker, force=False):
    """(eligible, reason, summary) for one ticker, cached for CACHE_DAYS.

    Used by harami_daily_trader.py at ENTRY time so every IDR order is tagged
    with a track: 'accumulate' for names whose filed fundamentals stand on their
    own, 'technical' for everything else (which keeps the ordinary exit rule).
    Any failure returns the technical track -- a data outage must never turn
    into an unintended long-term hold."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    hit = cache.get(ticker)
    if hit and not force:
        age = (date.today() - date.fromisoformat(hit["as_of"])).days
        if age <= CACHE_DAYS:
            r = hit["reason"] + f" (cached {age}d)"
            log_decision(ticker, hit["eligible"], r, hit.get("summary", {}), cached=True)
            return hit["eligible"], r, hit.get("summary", {})
    try:
        ciks = {v["ticker"]: str(v["cik_str"]).zfill(10)
                for v in requests.get("https://www.sec.gov/files/company_tickers.json",
                                      headers=V.UA, timeout=30).json().values()}
        if ticker not in ciks:
            return False, "no SEC CIK", {}
        o = V.analyse(ticker, ciks[ticker])
        ok, why = eligibility(o)
        summary = {k: o.get(k) for k in ("model", "fcf_norm", "implied_g", "fcf_cagr", "wacc",
                                         "roe", "coe", "price_to_fair", "net_debt", "ev")}
        cache[ticker] = {"as_of": str(date.today()), "eligible": bool(ok), "reason": why, "summary": summary}
        CACHE.write_text(json.dumps(cache, indent=1, default=str))
        log_decision(ticker, ok, why, summary, cached=False)
        return bool(ok), why, summary
    except Exception as exc:
        why = f"screen failed ({type(exc).__name__}) -- defaulting to technical track"
        log_decision(ticker, False, why, {}, cached=False)
        return False, why, {}


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"names": {}}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1, default=str))


def account_values(ib):
    v = {x.tag: float(x.value) for x in ib.accountValues()
         if x.currency in ("USD", "") and x.tag in ("NetLiquidation", "TotalCashValue", "AvailableFunds")}
    return v.get("NetLiquidation", 0.0), v.get("AvailableFunds", 0.0)


def last_price(ib, ticker):
    from ib_insync import Stock
    c = Stock(ticker, "SMART", "USD")
    q = ib.qualifyContracts(c)
    td = ib.reqMktData(q[0], "", False, False)
    ib.sleep(3)
    px = td.last or td.close or td.marketPrice()
    ib.cancelMktData(q[0])
    return float(px) if px and px == px else None


def recent_signal(ticker, days=3):
    """Did the IDR scanner fire for this ticker in the last `days` calendar days?
    A repeat signal is fresh information (CEO policy 2026-09-25), so it triggers
    the next tranche even when the -5%/-10% price trigger has not been hit --
    spacing and the target weight still apply."""
    import re
    rx = re.compile(r"(?:Bullish Harami \+ downtrend|Inside Day Reversal):\s*([A-Z][A-Z.\-]*)\s*\((\d{4}-\d{2}-\d{2})\)")
    log = HERE / "oversight_log.jsonl"
    if not log.exists():
        return None
    cutoff = date.today() - __import__("datetime").timedelta(days=days)
    hit = None
    for line in log.read_text(encoding="utf-8").splitlines():
        if "harami_scanner_alert" not in line:
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        m = rx.search(e.get("summary", ""))
        if m and m.group(1) == ticker and date.fromisoformat(m.group(2)) >= cutoff:
            hit = m.group(2)
    return hit


def _tranche_filled(ticker, tranche) -> bool:
    """Did this recorded tranche actually fill? Checked against the commission
    ledger, which is populated from IBKR's own execution reports."""
    import sqlite3
    db = HERE / "trade_journal.db"
    if not db.exists():
        return False
    try:
        con = sqlite3.connect(db)
        n = con.execute(
            "SELECT COUNT(*) FROM executions WHERE symbol=? AND side='BOT' "
            "AND trade_date >= ?", (ticker, str(tranche.get("date")))).fetchone()[0]
        con.close()
        return n > 0
    except Exception:
        return False


def plan_for(ticker, o, net_liq, price, st, held_value=0.0):
    """What the next action is for one qualifying name.

    THREE REAL BUGS FIXED 2026-09-29, all found together after the CEO asked why
    this had never bought anything. It had placed nothing since 2026-09-25:

      1. orderType was "MOO", which IBKR rejects (Error 321, "Invalid order type
         was entered"). Market-on-open is orderType="MKT" with tif="OPG".
      2. The cancelled orders were still appended to state as completed
         tranches, so plan_for read first_px off an order that never filled and
         then sat waiting for a -5% drop from a phantom entry. `done` now counts
         FILLED tranches only.
      3. Sizing was blind to the position it is supposed to be averaging into.
         This function never saw the existing IDR holding, and `max(shares, 1)`
         forced at least one share regardless of price -- so for RTX (tranche
         budget $48, share price $187) it wanted 1 share, 3.9x the tranche, on
         top of an existing $187 holding against a $145 target: 258% of target
         weight in a name the strategy only wanted 10% of. held_value is now
         passed in and the remaining room to target is respected.
    """
    held = st["names"].get(ticker, {"tranches": []})
    # Only FILLED tranches count -- see bug 2 above. But a market-on-open order
    # is recorded as "Submitted" and fills at the NEXT open, and nothing used to
    # go back and mark it Filled. Counting only "filled" then made the tranche
    # invisible forever, so this would BUY AGAIN every run until the cash or the
    # target ran out (bug 4, found 2026-09-30 immediately after the first
    # successful placement). A Submitted tranche from a PREVIOUS day whose fill
    # is confirmed in the commission ledger is therefore counted, and the record
    # is corrected in place.
    done = []
    for t in held["tranches"]:
        stt = str(t.get("status", "")).lower()
        if stt == "filled":
            done.append(t)
            continue
        if stt in ("submitted", "presubmitted", "pendingsubmit"):
            if str(t.get("date")) == str(date.today()):
                done.append(t)          # queued today; treat as committed
            elif _tranche_filled(ticker, t):
                t["status"] = "Filled"  # corrected in place; caller saves state
                done.append(t)
    target_dollars = (TARGET_DOLLARS_OVERRIDE if TARGET_DOLLARS_OVERRIDE
                      else net_liq * TARGET_WEIGHT)
    tranche_dollars = target_dollars / TRANCHES
    room = target_dollars - held_value
    if len(done) >= TRANCHES:
        return {"action": "hold", "why": f"all {TRANCHES} tranches filled"}
    if price and room < price:
        return {"action": "hold",
                "why": (f"already ${held_value:,.0f} vs ${target_dollars:,.0f} target "
                        f"({held_value / target_dollars * 100:.0f}%) -- no room for 1 share "
                        f"at ${price:,.2f}")}
    # Normal case: fill the tranche budget, never exceeding the room to target.
    shares = int(min(tranche_dollars, room) // price) if price else 0
    # Stretch to ONE share only when a single share alone exceeds the tranche
    # budget -- that is the whole point of the flat-dollar override (RTX at $187
    # against a $140 tranche). Without this guard the stretch also let a CHEAP
    # name take 1.33x its tranche (TFC wanted 4 shares / $186 vs a $140 budget).
    if shares < 1 and price and price <= min(tranche_dollars * 1.5, room):
        shares = 1
    if shares < 1:
        return {"action": "hold",
                "why": (f"1 share at ${price:,.2f} exceeds the ${tranche_dollars:,.0f} "
                        f"tranche budget (target ${target_dollars:,.0f}, held ${held_value:,.0f})")}
    if not done:
        return {"action": "buy", "tranche": 1, "shares": shares,
                "why": (f"first tranche, target ${target_dollars:,.0f} "
                        f"({'flat CEO override' if TARGET_DOLLARS_OVERRIDE else format(TARGET_WEIGHT, '.0%') + ' of net liq'})"
                        f", already held ${held_value:,.0f}")}
    first_px = done[0]["price"]
    drop = (price / first_px - 1) * 100
    trigger = ADD_TRIGGERS[len(done) - 1] if len(done) - 1 < len(ADD_TRIGGERS) else None
    if trigger is None:
        return {"action": "hold", "why": "no further trigger defined"}
    last_day = date.fromisoformat(done[-1]["date"])
    spacing = (date.today() - last_day).days
    sig = recent_signal(ticker)
    if drop > trigger and not sig:
        return {"action": "wait", "why": f"{drop:+.1f}% vs trigger {trigger:+.1f}%, no fresh signal"}
    if spacing < MIN_SPACING_DAYS:
        return {"action": "wait", "why": f"only {spacing}d since last tranche (need {MIN_SPACING_DAYS})"}
    why = (f"{drop:+.1f}% below first tranche, trigger {trigger:+.1f}%" if drop <= trigger
           else f"repeat IDR signal {sig} (price trigger not needed)")
    return {"action": "buy", "tranche": len(done) + 1, "shares": max(shares, 1), "why": why}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="place the next due tranche (default: report only)")
    ap.add_argument("tickers", nargs="*", help="default: current IDR positions + today's signals")
    args = ap.parse_args()
    from ib_insync import IB, Stock, Order

    tickers = args.tickers
    if not tickers:
        st_idr = json.loads((HERE / "harami_trader_state.json").read_text())
        tickers = sorted({p["ticker"] for p in st_idr["positions"].values()
                          if p.get("phase") in ("open", "pending_entry")})
    ciks = {v["ticker"]: str(v["cik_str"]).zfill(10)
            for v in requests.get("https://www.sec.gov/files/company_tickers.json",
                                  headers=V.UA, timeout=30).json().values()}
    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=1690, timeout=20)
    # Per-execution commission capture (added 2026-09-28) -- tranche adds are
    # 1-share buys where IBKR's 1%-of-value cap often binds instead of the
    # $1.00 minimum, so the real fee per add is worth recording, not assuming.
    try:
        import commission_ledger
        commission_ledger.attach(ib, strategy="idr_accumulation")
    except Exception as exc:
        print(f"commission_ledger attach failed (non-fatal): {exc}")
    net_liq, avail = account_values(ib)
    st = load_state()
    _tgt = (f"${TARGET_DOLLARS_OVERRIDE:,.0f}/name (flat override, "
            f"{TARGET_DOLLARS_OVERRIDE / net_liq * 100:.0f}% of net liq)"
            if TARGET_DOLLARS_OVERRIDE else f"{TARGET_WEIGHT:.0%} of net liq")
    print(f"net liq ${net_liq:,.2f} | available ${avail:,.2f} | target {_tgt} "
          f"({TRANCHES} tranches) | max {MAX_NAMES} names\n")
    print(f"{'sym':<6}{'eligible':>10}   reason")
    eligible = []
    for t in tickers:
        if t not in ciks:
            print(f"{t:<6}{'no':>10}   no SEC CIK")
            continue
        o = V.analyse(t, ciks[t])
        ok, why = eligibility(o)
        print(f"{t:<6}{('YES' if ok else 'no'):>10}   {why}")
        if ok:
            eligible.append((t, o))

    if not eligible:
        print("\nno name qualifies for accumulation today -- all stay on the ordinary IDR exit rule")
        ib.disconnect()
        return
    print(f"\naccumulation plan ({len(eligible)} qualifying):")
    for t, o in eligible[:MAX_NAMES]:
        px = last_price(ib, t)
        if not px:
            print(f"  {t}: no price")
            continue
        # Real market value of what we ALREADY hold in this name (bug 3).
        held_value = sum(abs(pi.marketValue) for pi in ib.portfolio()
                         if pi.contract.symbol == t and pi.contract.secType == "STK")
        p = plan_for(t, o, net_liq, px, st, held_value=held_value)
        cost = p.get("shares", 0) * px
        print(f"  {t:<5} ${px:>8.2f}  {p['action'].upper():<5} "
              f"{(str(p.get('shares', '')) + ' sh ($' + format(cost, ',.0f') + ')') if p['action']=='buy' else '':<22}"
              f"{p['why']}")
        if p["action"] == "buy" and cost > avail:
            print(f"        SKIP: needs ${cost:,.0f} but only ${avail:,.0f} available")
            continue
        if p["action"] == "buy" and args.execute:
            c = ib.qualifyContracts(Stock(t, "SMART", "USD"))[0]
            # Market-on-open is orderType="MKT" with tif="OPG" -- "MOO" is not a
            # valid IBKR order type and was rejected with Error 321 every run
            # since 2026-09-25 (bug 1).
            order = Order(action="BUY", totalQuantity=p["shares"], orderType="MKT",
                          tif="OPG", transmit=True)
            tr = ib.placeOrder(c, order)
            ib.sleep(3)
            status = tr.orderStatus.status
            if status in ("Cancelled", "Inactive", "ApiCancelled"):
                # Do NOT record a rejected order as a tranche -- that is what
                # wedged RTX and TFC for four days (bug 2).
                print(f"        REJECTED: {t} order {status} -- not recorded as a tranche")
                continue
            st["names"].setdefault(t, {"tranches": []})["tranches"].append(
                {"date": str(date.today()), "price": px, "shares": p["shares"],
                 "order_id": order.orderId, "status": status})
            save_state(st)
            print(f"        PLACED market-on-open BUY {p['shares']} {t} (status {status})")
            avail -= cost
    if not args.execute:
        print("\nreport only -- rerun with --execute to place the next due tranche")
    print("\nexit discipline for accumulated names: review when the price implies growth more than "
          f"{EXIT_WHEN_IMPLIED_EXCEEDS_DELIVERED_BY:.0%} above delivered, or the next 10-K turns FCF negative.")
    try:
        import commission_ledger
        commission_ledger.snapshot(ib, strategy="idr_accumulation")
    except Exception as exc:
        print(f"commission_ledger snapshot failed (non-fatal): {exc}")
    ib.disconnect()


if __name__ == "__main__":
    main()
