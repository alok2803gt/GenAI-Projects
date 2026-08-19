"""
Live/paper-adjacent 0DTE iron condor trader for SPY and SPX, via Alpaca
execution + IBKR pricing -- implements the parameters validated in the
2026-08-13 real-price backtests (spy_0dte_backtest_real.py,
spx_narrow_width_backtest.py), NOT the earlier Black-Scholes-modeled
version (later found to have understated real premiums badly).

VALIDATED PARAMS USED HERE (see oversight_log.jsonl 2026-08-13 for the
full backtest results these come from):
  SPY: 09:45 ET entry, 1.0% OTM shorts, $3-ish wide (0.4% of spot),
       no stop-loss, 50% profit target, hard close 15:45.
       Backtest: 38 real trades, 97.4% win rate, +$168 total, max_risk ~$300/contract.
  SPX: 09:45 ET entry, 1.0% OTM shorts, $5 FIXED wide (real SPXW strike
       increment), no stop-loss, 50% profit target, hard close 15:45.
       Backtest: 38 real trades, 100% win rate, +$1,368 total, max_risk $500/contract.

THIS IS BRAND NEW, NEVER-LIVE-TESTED CODE running against a real account.
Default qty=1 (one contract) for both -- do not raise without a fresh
explicit sizing conversation, per this account's standing rule that new/
unvalidated logic gets a small explicit-budget test, not full sizing.

Caveats carried over from the backtest, still true live:
  - Backtest used last-TRADE prices, not bid/ask -- real fills (especially
    a 4-leg close under stress) will likely be worse than backtested.
  - One ~2-month calm-to-moderate VIX regime (14-22) validated this,
    not a full market-cycle test.
  - No stop-loss means the real embedded tail risk (0DTE negative gamma)
    is deliberately NOT capped by an exit rule -- it's capped only by the
    condor's own defined-risk structure (max loss = width - credit).

Usage:
  python alpaca_0dte_trader.py --ticker spy --dry-run     # price it, don't place
  python alpaca_0dte_trader.py --ticker spy --fire        # place + monitor live
  python alpaca_0dte_trader.py --ticker spx --fire
"""
import argparse
import sys
import time
from datetime import date, datetime

from ib_insync import IB

from alpaca_0dte_common import (
    load_config, alpaca_client, get_quote, spx_option, spy_option, spx_spot, spy_spot,
    target_px, place_condor_sequential, close_condor_sequential, get_alpaca_symbols,
    register_position, close_position, now_et,
)

TWS_PORT = 7496
CLIENT_ID = {"spy": 1560, "spx": 1561}

CONFIG = {
    "spy": {
        "otm_pct": 0.010, "width_is_dollars": False, "width": 0.004,
        "strike_round": 1, "alpaca_underlying": "SPY",
    },
    "spx": {
        "otm_pct": 0.010, "width_is_dollars": True, "width": 5.0,
        "strike_round": 5, "alpaca_underlying": "SPX",
    },
}
PROFIT_TARGET_PCT = 0.50
HARD_CLOSE_TIME = "15:45"
QTY = 1
MONITOR_INTERVAL_S = 60
FILL_CHECK_WINDOW_S = 20


def round_to(x, step):
    return round(x / step) * step


def price_legs(ib, ticker, cfg, expiry_ibkr):
    if ticker == "spy":
        S0 = spy_spot(ib)
    else:
        S0 = spx_spot(ib)
    if not S0:
        print("ERROR: no live spot quote -- market likely closed. Aborting.")
        sys.exit(1)
    print(f"{ticker.upper()} spot: ${S0:.2f}")

    short_put_k = round_to(S0 * (1 - cfg["otm_pct"]), cfg["strike_round"])
    short_call_k = round_to(S0 * (1 + cfg["otm_pct"]), cfg["strike_round"])
    if cfg["width_is_dollars"]:
        long_put_k = short_put_k - cfg["width"]
        long_call_k = short_call_k + cfg["width"]
    else:
        long_put_k = round_to(S0 * (1 - cfg["otm_pct"] - cfg["width"]), cfg["strike_round"])
        long_call_k = round_to(S0 * (1 + cfg["otm_pct"] + cfg["width"]), cfg["strike_round"])

    print(f"Target condor: {long_put_k}P / {short_put_k}P  ...  {short_call_k}C / {long_call_k}C")

    opt_fn = spx_option if ticker == "spx" else spy_option
    legs = {
        "short_put": opt_fn(expiry_ibkr, short_put_k, "P"),
        "long_put": opt_fn(expiry_ibkr, long_put_k, "P"),
        "short_call": opt_fn(expiry_ibkr, short_call_k, "C"),
        "long_call": opt_fn(expiry_ibkr, long_call_k, "C"),
    }
    quotes = {name: get_quote(ib, c) for name, c in legs.items()}
    for name, q in quotes.items():
        print(f"  {name}: bid={q['bid']} ask={q['ask']}")
    if any(not (q["bid"] and q["ask"]) for q in quotes.values()):
        print("ERROR: missing live bid/ask on one or more legs -- aborting.")
        sys.exit(1)

    conservative_credit = round(
        (quotes["short_put"]["bid"] + quotes["short_call"]["bid"])
        - (quotes["long_put"]["ask"] + quotes["long_call"]["ask"]), 2
    )
    entry_limits = {name: target_px(q, name.startswith("short")) for name, q in quotes.items()}
    strikes = {"short_put": short_put_k, "long_put": long_put_k,
               "short_call": short_call_k, "long_call": long_call_k}
    max_risk = round((short_put_k - long_put_k) * 100, 2)
    return strikes, quotes, entry_limits, conservative_credit, max_risk


def condor_mark(ib, ticker, expiry_ibkr, strikes):
    """Current mark-to-market value of the SHORT condor (what it costs to close)."""
    opt_fn = spx_option if ticker == "spx" else spy_option
    legs = {
        "short_put": opt_fn(expiry_ibkr, strikes["short_put"], "P"),
        "long_put": opt_fn(expiry_ibkr, strikes["long_put"], "P"),
        "short_call": opt_fn(expiry_ibkr, strikes["short_call"], "C"),
        "long_call": opt_fn(expiry_ibkr, strikes["long_call"], "C"),
    }
    quotes = {name: get_quote(ib, c) for name, c in legs.items()}
    if any(not q["mid"] for q in quotes.values()):
        return None, quotes
    value = (quotes["short_put"]["mid"] - quotes["long_put"]["mid"]) + \
            (quotes["short_call"]["mid"] - quotes["long_call"]["mid"])
    return value, quotes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True, choices=["spy", "spx"])
    ap.add_argument("--fire", action="store_true", help="place real orders; omit for dry-run pricing only")
    ap.add_argument("--qty", type=int, default=QTY)
    args = ap.parse_args()
    ticker = args.ticker
    cfg = CONFIG[ticker]
    cfg_obj = load_config()

    today_ibkr = date.today().strftime("%Y%m%d")
    today_alp = date.today().strftime("%Y-%m-%d")

    print(f"=== {ticker.upper()} 0DTE condor -- {'FIRE' if args.fire else 'DRY RUN'} -- expiry {today_alp} ===")

    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID[ticker], timeout=20)
    print("Connected to IBKR.")

    try:
        strikes, quotes, entry_limits, conservative_credit, max_risk = price_legs(ib, ticker, cfg, today_ibkr)
        print(f"conservative credit (worst-fill): ${conservative_credit:.2f}  "
              f"(=${conservative_credit*100:.0f}/contract)")
        print(f"max_risk: ${max_risk:.0f}/contract")
        if conservative_credit <= 0:
            print("ERROR: no positive conservative credit -- aborting.")
            sys.exit(1)
    finally:
        ib.disconnect()
        print("Disconnected from IBKR (pricing done).")

    if not args.fire:
        print("\nDRY RUN -- no orders placed. Re-run with --fire to place live orders.")
        return

    client = alpaca_client(cfg_obj)
    put_by_strike, call_by_strike = get_alpaca_symbols(
        client, cfg["alpaca_underlying"], today_alp,
        strikes["long_put"] - 1, strikes["short_put"] + 1,
        strikes["short_call"] - 1, strikes["long_call"] + 1,
    )
    missing = [k for k in (strikes["short_put"], strikes["long_put"]) if k not in put_by_strike] + \
              [k for k in (strikes["short_call"], strikes["long_call"]) if k not in call_by_strike]
    if missing:
        print(f"ERROR: strikes not listed on Alpaca: {missing}")
        sys.exit(1)
    syms = {
        "short_put": put_by_strike[strikes["short_put"]],
        "long_put": put_by_strike[strikes["long_put"]],
        "short_call": call_by_strike[strikes["short_call"]],
        "long_call": call_by_strike[strikes["long_call"]],
    }
    print(f"Alpaca symbols: {syms}")

    ok, fills, state = place_condor_sequential(client, syms, entry_limits, args.qty)
    if not ok:
        print(f"\nENTRY INCOMPLETE ({state}) -- check Alpaca positions manually before doing anything else.")
        sys.exit(1)

    net_entry_credit = round(
        (fills["short_put"] + fills["short_call"]) - (fills["long_put"] + fills["long_call"]), 2
    )
    entry_time = now_et()
    pos_id = f"{ticker.upper()}_0dte_condor_{entry_time.strftime('%Y%m%d_%H%M%S')}"
    profit_target_usd = round(net_entry_credit * args.qty * 100 * PROFIT_TARGET_PCT, 2)
    register_position(
        pos_id, ticker.upper(), "iron_condor_0dte",
        [{"leg": k, "strike": strikes[k], "symbol": syms[k], "fill": fills[k]} for k in syms],
        net_entry_credit, args.qty, max_risk * args.qty, profit_target_usd, HARD_CLOSE_TIME,
        entry_time.isoformat(),
        notes=f"Live test of 2026-08-13 real-price backtest finding. otm={cfg['otm_pct']}, "
              f"width={cfg['width']}{'usd' if cfg['width_is_dollars'] else 'pct'}, no stop-loss, "
              f"{PROFIT_TARGET_PCT*100:.0f}% profit target, hard close {HARD_CLOSE_TIME}.",
    )
    print(f"\nENTERED. pos_id={pos_id}  net_entry_credit=${net_entry_credit:.2f} "
          f"(${net_entry_credit*args.qty*100:.0f} total)  profit_target=${profit_target_usd:.2f}")

    # ── Monitor until profit target or hard close (no stop-loss, per validated params) ──
    print(f"\nMonitoring every {MONITOR_INTERVAL_S}s until profit target or {HARD_CLOSE_TIME} hard close...")
    hard_close_dt = datetime.strptime(f"{entry_time.strftime('%Y-%m-%d')} {HARD_CLOSE_TIME}", "%Y-%m-%d %H:%M")
    hard_close_dt = hard_close_dt.replace(tzinfo=entry_time.tzinfo)

    ib2 = IB()
    ib2.errorEvent += lambda reqId, code, msg, contract: None
    ib2.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID[ticker] + 100, timeout=20)
    try:
        while True:
            now = now_et()
            value, mon_quotes = condor_mark(ib2, ticker, today_ibkr, strikes)
            if value is None:
                print(f"[{now.strftime('%H:%M:%S')}] quote gap, will retry next cycle")
            else:
                live_pnl = (net_entry_credit - value) * args.qty * 100
                print(f"[{now.strftime('%H:%M:%S')}] condor_value=${value:.2f}  "
                      f"live_pnl=${live_pnl:+.2f}  target=${profit_target_usd:.2f}")
                if live_pnl >= profit_target_usd:
                    print("\nPROFIT TARGET HIT -- closing.")
                    close_limits = {k: target_px(mon_quotes[k], not k.startswith("short")) for k in mon_quotes}
                    ok, close_fills = close_condor_sequential(client, syms, close_limits, args.qty)
                    close_position(pos_id, "profit_target_hit", live_pnl)
                    print(f"Closed. ok={ok} fills={close_fills}")
                    return
            if now >= hard_close_dt:
                print(f"\n{HARD_CLOSE_TIME} HARD CLOSE -- closing regardless of P&L.")
                _, mon_quotes = condor_mark(ib2, ticker, today_ibkr, strikes)
                if mon_quotes:
                    close_limits = {k: target_px(mon_quotes[k], not k.startswith("short")) for k in mon_quotes}
                else:
                    close_limits = {k: entry_limits[k] for k in entry_limits}
                ok, close_fills = close_condor_sequential(client, syms, close_limits, args.qty)
                final_value, _ = condor_mark(ib2, ticker, today_ibkr, strikes)
                final_pnl = (net_entry_credit - final_value) * args.qty * 100 if final_value else None
                close_position(pos_id, "hard_close", final_pnl)
                print(f"Closed. ok={ok} fills={close_fills} final_pnl={final_pnl}")
                return
            time.sleep(MONITOR_INTERVAL_S)
    finally:
        ib2.disconnect()


if __name__ == "__main__":
    main()
