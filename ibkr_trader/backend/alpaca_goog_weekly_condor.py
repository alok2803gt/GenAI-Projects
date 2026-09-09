"""
GOOG weekly iron condor, premium collection -- same proven sequential-leg
execution pattern as the SPY/QQQ/IWM 0DTE butterflies / Ashley signal-follow.
Pricing/quotes AND execution are both on IBKR (migrated from Alpaca
2026-09-07: Alpaca's real account balance dropped to $0 buying power,
fully consumed by an unrelated existing position -- no longer usable for
any strategy). MLEG combo orders are known to reject on this account for
equity options (task 2026-08-12-003, the reason this strategy went to
Alpaca in the first place) -- the workaround, sequential single-leg
orders, works identically on IBKR, so this was never actually an
IBKR-specific limitation.

Parameters (otm_pct=5%, width_pct=1.5%) come from a real, earnings-week-
excluded backtest run 2026-08-18 (weekly_condor_backtest.py, 3 years of
daily GOOG history, Black-Scholes premium off 20-day realized vol):
5% OTM / ~1.5% width -> 85.8% win rate, +5.3% avg return/week, ~$512/contract
modeled max risk. GOOG's real listed strikes near the money are $2.50 apart
(confirmed live 2026-08-18), so width_pct=0.015 is a TARGET -- the script
snaps to the nearest actually-listed strike, which lands on a real $5.00
width (~1.47% of spot), not exactly 1.5%. AAPL's equivalent backtest showed
this edge going NEGATIVE at a tighter ~1% width (tail losses eating the
credit) -- do not shrink width_pct below what's been validated without
rerunning the backtest at the new number.

Backtest was Monday-open entries only. Running this off-cycle (not Monday,
or targeting an expiry that isn't a standard 4-5-DTE weekly from today)
deviates from what was actually tested -- the script does not auto-compute
"next Friday" for this reason; --expiry is a required, explicit argument
so the caller states what they're actually entering, matching this
account's established discipline (alpaca_earnings_condor.py, alpaca_mu_pcs.py).

No auto-monitor / early profit-take loop -- the backtest determined win/loss
via real Friday close vs strikes with no early exit modeled, so this script
holds to expiry, matching what was actually validated. Position is visible
via /goog-condor/status same as every other tracked strategy; a human (or a
future close script) exits it before/at expiry.

Usage:
  python alpaca_goog_weekly_condor.py --expiry 2026-08-28 --client-id 1602 [--dry-run]
"""
import argparse
import json
import sys

import requests
from ib_insync import IB, Stock, Option

from alpaca_0dte_common import (
    load_config, get_quotes_batch, register_position, now_et,
    cro_cfo_capital_budget, safe_px,
)
from ibkr_0dte_common import ibkr_place_condor_sequential, occ_symbol

TICKER = "GOOG"
TWS_PORT = 7496
OTM_PCT_DEFAULT = 0.05
WIDTH_PCT_DEFAULT = 0.015
MIN_CONSERVATIVE_CREDIT_DEFAULT = 0.30
DECISIONS_FILE = "goog_condor_decisions.json"


def telegram(text, high_priority=False):
    cfg = load_config()
    prefix = "\U0001F6A8 " if high_priority else ""
    try:
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": prefix + text},
            timeout=10,
        )
    except Exception as e:
        print(f"telegram send failed: {e}")


def oversight_log(category, summary, rationale="", outcome=None, pnl_impact=None):
    entry = {
        "time": now_et().isoformat(), "actor": "trader", "category": category,
        "summary": summary, "rationale": rationale, "outcome": outcome, "pnl_impact": pnl_impact,
    }
    with open("oversight_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def goog_condor_log(action, detail):
    entry = {"time": now_et().strftime("%Y-%m-%d %H:%M:%S ET"), "action": action, "detail": detail}
    try:
        with open(DECISIONS_FILE) as f:
            decisions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        decisions = []
    decisions.append(entry)
    decisions = decisions[-200:]
    with open(DECISIONS_FILE, "w") as f:
        json.dump(decisions, f, indent=2)


def pretrade_review(strikes, conservative_credit, max_risk, spot, budget):
    """Real-time CRO/CFO pre-trade review -- BLOCKING gate before any order
    is placed. This script never had one (task 2026-08-20-006 flagged it as
    still-open since the day the pattern was built for EVC/Day Trader/
    SPY 0DTE) -- added 2026-08-24 before resuming Monday-cycle entries.

    CRO: what-if P&L scenarios across a range of real moves, using this
         condor's actual strikes/credit.
    CFO: this trade's size vs the 5% per-trade cap AND real remaining
         portfolio headroom (cro_cfo_capital_budget) -- IBKR-only net liq
         since this script has no Alpaca dependency (2026-09-07); other
         strategies still executing on Alpaca get a genuinely combined
         view from the same shared function.
    """
    findings = []
    approved = True

    short_put, long_put = strikes["short_put"], strikes["long_put"]
    short_call, long_call = strikes["short_call"], strikes["long_call"]

    scenarios = []
    for pct in (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10):
        px = round(spot * (1 + pct), 2)
        put_intrinsic = max(0.0, short_put - px) - max(0.0, long_put - px)
        call_intrinsic = max(0.0, px - short_call) - max(0.0, px - long_call)
        pnl = round(conservative_credit * 100 - (put_intrinsic + call_intrinsic) * 100, 2)
        scenarios.append({"move": f"{pct:+.0%}", "price": px, "pnl": pnl})
    findings.append("scenarios: " + "  ".join(f"{s['move']}(${s['price']:.0f}):{s['pnl']:+.0f}" for s in scenarios))

    pct_of_acct = (max_risk / budget["combined_net_liq"]) if budget["combined_net_liq"] else 1.0
    findings.append(f"CFO: max risk ${max_risk:,.0f} = {pct_of_acct:.1%} of combined net liq ${budget['combined_net_liq']:,.0f}")
    if max_risk > budget["per_strategy_cap"]:
        findings.append(f"EXCEEDS per-trade cap (${budget['per_strategy_cap']:,.0f} = {budget['per_strategy_cap_pct']:.0%} of net liq)")
        approved = False

    findings.append(f"CFO: portfolio headroom ${budget['headroom']:,.0f} of ${budget['total_budget']:,.0f} total budget")
    if max_risk > budget["headroom"]:
        findings.append("EXCEEDS remaining portfolio headroom")
        approved = False

    return {"approved": approved, "findings": findings, "scenarios": scenarios}


def nearest(strikes, target):
    return min(strikes, key=lambda s: abs(s - target))


def pick_strikes(ib, expiry_ibkr, spot, otm_pct, width_pct):
    """Real listed GOOG strikes for expiry, snapped to nearest-available
    OTM target and nearest-available width."""
    stk = Stock(TICKER, "SMART", "USD")
    ib.qualifyContracts(stk)
    chains = ib.reqSecDefOptParams(stk.symbol, "", stk.secType, stk.conId)
    smart = [c for c in chains if c.exchange == "SMART"]
    if not smart or expiry_ibkr not in smart[0].expirations:
        return None, f"expiry {expiry_ibkr} not found in SMART chain"
    strikes = sorted(smart[0].strikes)

    raw_width = spot * width_pct
    short_put = nearest(strikes, spot * (1 - otm_pct))
    put_candidates = [s for s in strikes if s < short_put]
    long_put = nearest(put_candidates, short_put - raw_width) if put_candidates else None

    short_call = nearest(strikes, spot * (1 + otm_pct))
    call_candidates = [s for s in strikes if s > short_call]
    long_call = nearest(call_candidates, short_call + raw_width) if call_candidates else None

    if long_put is None or long_call is None or long_put >= short_put or long_call <= short_call:
        return None, f"could not build a valid spread from listed strikes near spot={spot}"
    return {"short_put": short_put, "long_put": long_put, "short_call": short_call, "long_call": long_call}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expiry", required=True, help="YYYY-MM-DD, the Friday weekly expiry to enter")
    ap.add_argument("--otm-pct", type=float, default=OTM_PCT_DEFAULT)
    ap.add_argument("--width-pct", type=float, default=WIDTH_PCT_DEFAULT)
    ap.add_argument("--min-credit", type=float, default=MIN_CONSERVATIVE_CREDIT_DEFAULT)
    ap.add_argument("--qty", type=int, default=1)
    ap.add_argument("--client-id", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true", help="price and log only, never submit an order")
    args = ap.parse_args()

    expiry_ibkr = args.expiry.replace("-", "")
    print(f"=== GOOG weekly condor: exp {args.expiry}  otm={args.otm_pct*100:.1f}%  "
          f"width_target={args.width_pct*100:.2f}%  dry_run={args.dry_run} ===")

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=args.client_id, timeout=20)
    print("Connected to IBKR.")

    try:
        stk = Stock(TICKER, "SMART", "USD")
        ib.qualifyContracts(stk)
        td = ib.reqMktData(stk, "", False, False)
        ib.sleep(3)
        bid, ask = safe_px(td.bid), safe_px(td.ask)
        # Real bug found 2026-09-07 (testing on a market holiday exposed it):
        # `td.marketPrice() or td.last or td.close` doesn't correctly skip a
        # NaN value -- NaN is truthy in Python, so a NaN marketPrice() short-
        # circuited the whole fallback chain instead of falling through to
        # last/close. safe_px() (already used for exactly this reason
        # elsewhere in this codebase, e.g. etf_spot()) explicitly excludes
        # NaN. Pre-existing in the Alpaca-era version of this script too --
        # just never hit before since this only runs when bid/ask are
        # normally present (real market hours).
        spot = (bid + ask) / 2 if bid and ask else (safe_px(td.marketPrice()) or safe_px(td.last) or safe_px(td.close))
        ib.cancelMktData(stk)
        if not spot:
            print("ERROR: no live GOOG spot available -- aborting.")
            goog_condor_log("PRICING_ABORT", "no live spot")
            sys.exit(1)
        print(f"spot: ${spot:.2f}")

        strikes, err = pick_strikes(ib, expiry_ibkr, spot, args.otm_pct, args.width_pct)
        if err:
            print(f"ERROR: {err}")
            goog_condor_log("STRIKES_MISSING", err)
            sys.exit(1)
        print(f"strikes: {strikes}  (put width ${strikes['short_put']-strikes['long_put']:.2f}, "
              f"call width ${strikes['long_call']-strikes['short_call']:.2f})")

        legs = {
            "long_put":   Option(TICKER, expiry_ibkr, strikes["long_put"],   "P", "SMART"),
            "short_put":  Option(TICKER, expiry_ibkr, strikes["short_put"],  "P", "SMART"),
            "short_call": Option(TICKER, expiry_ibkr, strikes["short_call"], "C", "SMART"),
            "long_call":  Option(TICKER, expiry_ibkr, strikes["long_call"],  "C", "SMART"),
        }
        quotes = get_quotes_batch(ib, legs)
        for name, q in quotes.items():
            print(f"  {name} {getattr(legs[name], 'strike', '')}: bid={q['bid']} ask={q['ask']}")
        if any(not (q["bid"] and q["ask"]) for q in quotes.values()):
            print("ERROR: missing live bid/ask on one or more legs -- aborting.")
            goog_condor_log("PRICING_ABORT", "missing live bid/ask on one or more legs")
            sys.exit(1)

        conservative_credit = round(
            (quotes["short_put"]["bid"] + quotes["short_call"]["bid"])
            - (quotes["long_put"]["ask"] + quotes["long_call"]["ask"]), 2
        )
        # max(), not min() -- real bug found 2026-08-26 in EVC's own review
        # (main.py's _evc_pretrade_review, later also fixed proactively in
        # _spy_pretrade_review): an iron condor's true max loss is bounded
        # by whichever wing actually gets breached, unknown in advance, so
        # using the narrower wing's width silently understates real
        # worst-case risk. This script's pick_strikes() targets symmetric
        # otm_pct/width_pct on both sides, but real strike-grid rounding can
        # still leave put_width != call_width in practice -- confirmed this
        # was still min() here 2026-08-27 while re-checking max_risk
        # correctness on the live GOOG_weekly_condor_20260824_095213
        # position (that one happened to have equal 5.0-wide puts/calls, so
        # it was unaffected, but IBKR-GOOGCondorMonday's next scheduled run
        # would have re-hit this on any asymmetric snap). Fixed to max().
        max_risk = round(max(strikes["short_put"] - strikes["long_put"],
                              strikes["long_call"] - strikes["short_call"]) * 100 * args.qty, 2)
        print(f"conservative credit (worst-fill): ${conservative_credit:.2f}  threshold: ${args.min_credit:.2f}")
        print(f"max_risk: ${max_risk:.0f}")

        # 15% per-trade cap for GOOG weekly condor specifically -- CEO decision
        # 2026-08-24, above the account-wide 5% default (this strategy's
        # validated structure runs ~$500 max risk/contract). Does not affect
        # any other strategy's pre-trade review, which still defaults to 5%.
        # IBKR-only now (2026-09-07): no Alpaca dependency left at all, not
        # even read-only -- `client` omitted (cro_cfo_capital_budget's
        # optional param, see its own docstring).
        budget = cro_cfo_capital_budget(ib, cap_pct=0.15)
        review = pretrade_review(strikes, conservative_credit, max_risk, spot, budget)
        print("--- CRO/CFO pre-trade review ---")
        for f in review["findings"]:
            print(f"  {f}")
        if not review["approved"]:
            msg = f"GOOG weekly condor: REJECTED by pre-trade review — {'; '.join(review['findings'])}"
            print(msg)
            oversight_log("pretrade_review", msg, outcome="REJECTED — no order placed")
            goog_condor_log("PRETRADE_REJECTED", msg)
            sys.exit(0)

        entry_limits = {}
        for name in ("long_put", "long_call"):
            q = quotes[name]
            entry_limits[name] = round(q["ask"] - (q["ask"] - q["mid"]) * 0.40, 2)
        for name in ("short_put", "short_call"):
            q = quotes[name]
            entry_limits[name] = round(q["bid"] + (q["mid"] - q["bid"]) * 0.40, 2)
        print("entry limits:", entry_limits)

        if conservative_credit < args.min_credit:
            msg = (f"GOOG weekly condor: NO FIRE. strikes={strikes} conservative_credit=${conservative_credit:.2f} "
                   f"< ${args.min_credit:.2f} threshold. No order placed.")
            print(msg)
            goog_condor_log("NO_FIRE", msg)
            sys.exit(0)

        if args.dry_run:
            msg = (f"GOOG weekly condor DRY RUN: strikes={strikes} conservative_credit=${conservative_credit:.2f} "
                   f"max_risk=${max_risk:.0f} -- clears gate, no order submitted (dry run).")
            print(msg)
            goog_condor_log("DRY_RUN_CLEAR", msg)
            ib.disconnect()
            sys.exit(0)

        # ── Execution: IBKR sequential legs (migrated from Alpaca 2026-09-07)
        # Reuses the SAME `ib` connection pricing already used -- previously
        # disconnected here since execution went to Alpaca and didn't need
        # it; now order placement needs a live IBKR session too, so this
        # stays connected through the fire step instead of opening a second
        # connection. `legs` (built above for quoting) are already
        # qualified real Option contracts -- reused directly as the order
        # contracts, no separate symbol lookup needed.
        syms = {k: occ_symbol(TICKER, strikes[k], legs[k].right, args.expiry.replace("-", "")[2:])
                for k in legs}
        telegram(f"GOOG weekly condor: FIRING. {strikes['long_put']}/{strikes['short_put']}P.."
                 f"{strikes['short_call']}/{strikes['long_call']}C exp {args.expiry}, "
                 f"conservative_credit=${conservative_credit:.2f}, qty={args.qty}, max_risk=${max_risk:.0f}")

        ok, fills, state = ibkr_place_condor_sequential(ib, legs, entry_limits, args.qty)
        if not ok:
            msg = f"GOOG weekly condor: ENTRY INCOMPLETE (state={state}, fills={fills}). Check IBKR positions manually NOW."
            print(msg)
            telegram(msg, high_priority=True)
            oversight_log("execution_issue", msg, outcome="PAUSED -- possible naked leg, needs manual review")
            goog_condor_log("ENTRY_INCOMPLETE", msg)
            sys.exit(1)
    finally:
        ib.disconnect()
        print("Disconnected from IBKR.")

    net_entry_credit = round((fills["short_put"] + fills["short_call"]) - (fills["long_put"] + fills["long_call"]), 2)
    entry_time = now_et()
    pos_id = f"GOOG_weekly_condor_{entry_time.strftime('%Y%m%d_%H%M%S')}"
    register_position(
        pos_id, TICKER, "iron_condor_weekly",
        [{"leg": k, "strike": strikes[k], "symbol": syms[k], "fill": fills[k]} for k in syms],
        net_entry_credit, args.qty, max_risk, None, None, entry_time.isoformat(),
        notes=f"Weekly premium collection, exp {args.expiry}. Backtest (2026-08-18, earnings-excluded, "
              f"3yr): 5% OTM/~1.5% width -> 85.8% win, +5.3% avg return/week. Holds to expiry (no early "
              f"profit-take modeled or automated).",
    )
    msg = f"GOOG weekly condor: ENTERED. pos_id={pos_id} net_entry_credit=${net_entry_credit:.2f} (${net_entry_credit*args.qty*100:.0f} total) max_risk=${max_risk:.0f}"
    print(msg)
    telegram(msg)
    oversight_log("position_opened", msg, outcome="entered, holds to expiry")
    goog_condor_log("ENTERED", msg)
    print(json.dumps({"pos_id": pos_id, "strikes": strikes, "net_entry_credit": net_entry_credit, "fills": fills}, indent=2))


if __name__ == "__main__":
    main()
