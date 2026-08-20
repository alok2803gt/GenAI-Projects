"""
GOOG weekly iron condor, premium collection -- same proven sequential-leg
Alpaca execution as SPY 0DTE / MU PCS / the LOW+TJX earnings condors.
Pricing/quotes from IBKR (this account's data source of record), execution
on Alpaca (2026-08-11 architecture decision, avoids IBKR PDT gate).

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
    load_config, alpaca_client, get_quote, get_alpaca_symbols,
    place_condor_sequential, register_position, now_et,
)

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
        bid, ask = td.bid, td.ask
        spot = (bid + ask) / 2 if bid and ask and bid > 0 and ask > 0 else (td.marketPrice() or td.last or td.close)
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
        quotes = {name: get_quote(ib, c) for name, c in legs.items()}
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
        max_risk = round(min(strikes["short_put"] - strikes["long_put"],
                              strikes["long_call"] - strikes["short_call"]) * 100 * args.qty, 2)
        print(f"conservative credit (worst-fill): ${conservative_credit:.2f}  threshold: ${args.min_credit:.2f}")
        print(f"max_risk: ${max_risk:.0f}")

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
            sys.exit(0)
    finally:
        ib.disconnect()
        print("Disconnected from IBKR (pricing done).")

    cfg = load_config()
    client = alpaca_client(cfg)
    expiry_alp = args.expiry
    put_by_strike, call_by_strike = get_alpaca_symbols(
        client, TICKER, expiry_alp,
        strikes["long_put"] - 1, strikes["short_put"] + 1,
        strikes["short_call"] - 1, strikes["long_call"] + 1,
    )
    missing = [k for k in (strikes["short_put"], strikes["long_put"]) if k not in put_by_strike] + \
              [k for k in (strikes["short_call"], strikes["long_call"]) if k not in call_by_strike]
    if missing:
        msg = f"GOOG weekly condor: strikes not listed on Alpaca: {missing} -- no order placed."
        print(msg)
        telegram(msg, high_priority=True)
        goog_condor_log("STRIKES_MISSING", msg)
        sys.exit(1)

    syms = {
        "short_put": put_by_strike[strikes["short_put"]], "long_put": put_by_strike[strikes["long_put"]],
        "short_call": call_by_strike[strikes["short_call"]], "long_call": call_by_strike[strikes["long_call"]],
    }
    telegram(f"GOOG weekly condor: FIRING. {strikes['long_put']}/{strikes['short_put']}P.."
             f"{strikes['short_call']}/{strikes['long_call']}C exp {args.expiry}, "
             f"conservative_credit=${conservative_credit:.2f}, qty={args.qty}, max_risk=${max_risk:.0f}")

    ok, fills, state = place_condor_sequential(client, syms, entry_limits, args.qty)
    if not ok:
        msg = f"GOOG weekly condor: ENTRY INCOMPLETE (state={state}, fills={fills}). Check Alpaca positions manually NOW."
        print(msg)
        telegram(msg, high_priority=True)
        oversight_log("execution_issue", msg, outcome="PAUSED -- possible naked leg, needs manual review")
        goog_condor_log("ENTRY_INCOMPLETE", msg)
        sys.exit(1)

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
