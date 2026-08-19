"""
NFLX bottom-watch monitor. Checks two concrete, CIO-defined signals every
run; if either fires, MODELS a narrow defined-risk put credit spread with
live IBKR pricing and real Alpaca contract symbols, and alerts immediately
-- does NOT place the order automatically (recurring auto-fire got blocked
by the permission system as too large a standing commitment; CEO chose
alert-and-confirm instead). Once alerted, run with --fire to actually
place it (fresh pricing at that moment, not the stale alert-time numbers).

Context (2026-08-12 CIO/CRO/CFO consult, oversight_log.jsonl): NFLX at
$73.93, 200sma $89.52 (-17.4%), 50sma $75.00, RSI-14 67.2 -- bounced ~9% off
a 52-week low of $67.60 but still under its longer-term downtrend. Not a
clean entry as-is (RSI already hot off the bottom). CIO's own criteria for
a real entry: "either a real breakout confirmation above the 200-day, or a
fresh pullback that resets RSI."

Two signal tiers (either one fires the trade):
  SIGNAL A (earlier, lower-confidence): RSI-14 cools to <=45 from today's
    67.2 elevated level, AND price holds above the 52wk low ($67.60, with
    a small buffer) -- i.e. the bounce forms a higher low instead of
    breaking down. Base-building confirmation.
  SIGNAL B (later, higher-confidence): price closes above the 200-day SMA.
    Real trend-reversal confirmation, not just a bounce.

Trade if triggered: put credit spread, short ~11% OTM, width=2 (narrow --
CFO's sizing call: a 5-wide spread was ~half of IBKR's available cash in
one position, too concentrated; 2-wide keeps max risk to roughly $150-200,
proportionate to this account's real size). Placed as two sequential
single-leg Alpaca orders (long first, then short) -- MLEG combo orders
reject on this account for an unresolved reason (task 2026-08-12-003);
this is the proven working path.

State file (nflx_monitor_state.json) tracks whether this has already fired
this cycle, so repeated cron runs don't place the trade more than once.

Usage: python nflx_bottom_monitor.py
"""
import json
import sys
import time
from datetime import datetime, timezone

import yfinance as yf
from ib_insync import IB, Option, Stock
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest

TICKER        = "NFLX"
RSI_RESET_MAX = 45.0     # Signal A: RSI must cool to this or below
LOW_BUFFER    = 0.98     # Signal A: price must stay above 52wk_low * this (2% buffer)
SHORT_OTM_PCT = 0.11     # ~11% OTM short, matches todays consult
WIDTH         = 2.0      # narrow width per CFO sizing (not the wider 5pt version)
QTY           = 1
TWS_PORT      = 7496
CLIENT_ID     = 1553
STATE_FILE    = "nflx_monitor_state.json"

with open("scanner_config.json") as f:
    cfg = json.load(f)


def telegram(msg: str):
    import requests
    try:
        requests.post(
            f"https://api.telegram.org/bot{cfg['telegram_token']}/sendMessage",
            json={"chat_id": cfg["telegram_chat_id"], "text": msg, "parse_mode": "HTML"},
            timeout=8,
        )
    except Exception as e:
        print(f"Telegram send failed: {e}")


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"fired": False, "fired_at": None, "checks": 0}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def check_signals() -> dict:
    hist = yf.Ticker(TICKER).history(period="1y")
    # yfinance adds a placeholder row for the still-in-progress trading day
    # with a NaN close pre-market -- iloc[-1] grabbed that NaN directly while
    # low_52wk's .min() silently ignored it, so price/sma/rsi came back nan
    # while low_52wk looked fine. Drop incomplete rows first so this is
    # correct regardless of when the check runs.
    hist = hist.dropna(subset=["Close"])
    price = float(hist["Close"].iloc[-1])
    sma200 = float(hist["Close"].rolling(200).mean().iloc[-1])
    sma50 = float(hist["Close"].rolling(50).mean().iloc[-1])
    low_52wk = float(hist["Close"].min())

    delta = hist["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = float(100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1]))

    signal_a = rsi <= RSI_RESET_MAX and price >= low_52wk * LOW_BUFFER
    signal_b = price > sma200

    return {
        "price": round(price, 2), "sma200": round(sma200, 2), "sma50": round(sma50, 2),
        "rsi14": round(rsi, 1), "low_52wk": round(low_52wk, 2),
        "signal_a": signal_a, "signal_b": signal_b,
        "triggered": signal_a or signal_b,
        "which": "A (RSI reset, held above low)" if signal_a else ("B (200sma breakout)" if signal_b else None),
    }


def safe_px(v):
    if v is None:
        return None
    try:
        f = float(v)
        return f if f > 0 and f == f else None
    except (TypeError, ValueError):
        return None


def get_quote(ib: IB, contract) -> dict:
    ib.qualifyContracts(contract)
    if not contract.conId:
        raise ValueError(f"could not qualify {contract}")
    td = ib.reqMktData(contract, "", False, False)
    ib.sleep(3)
    bid, ask = safe_px(td.bid), safe_px(td.ask)
    ib.cancelMktData(contract)
    return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2 if bid and ask else None}


def model_trade(spot: float) -> dict:
    """Live IBKR pricing + real Alpaca contract symbols -- computes the exact
    trade that would be placed, but does NOT submit any order. Read-only."""
    ib = IB()
    ib.errorEvent += lambda reqId, code, msg, contract: None
    ib.connect("127.0.0.1", TWS_PORT, clientId=CLIENT_ID, timeout=20)
    try:
        from datetime import date as _date
        chains = ib.reqSecDefOptParams(TICKER, "", "STK",
                                        ib.qualifyContracts(Stock(TICKER, "SMART", "USD"))[0].conId)
        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
        today = _date.today()
        expiry = None
        for e in sorted(chain.expirations):
            dte = (datetime.strptime(e, "%Y%m%d").date() - today).days
            if 25 <= dte <= 45:
                expiry = e
                break
        expiry = expiry or sorted(chain.expirations)[2]
        expiry_alp = f"{expiry[:4]}-{expiry[4:6]}-{expiry[6:8]}"

        real_strikes = sorted(chain.strikes)
        target_short = spot * (1 - SHORT_OTM_PCT)
        short_k = min(real_strikes, key=lambda s: abs(s - target_short))
        target_long = short_k - WIDTH
        long_k = min([s for s in real_strikes if s < short_k], key=lambda s: abs(s - target_long))

        long_q = get_quote(ib, Option(TICKER, expiry, long_k, "P", "SMART"))
        short_q = get_quote(ib, Option(TICKER, expiry, short_k, "P", "SMART"))
        if not (long_q["bid"] and long_q["ask"] and short_q["bid"] and short_q["ask"]):
            return {"error": "no live bid/ask on one or both legs"}
        long_limit = round(long_q["ask"] - (long_q["ask"] - long_q["mid"]) * 0.40, 2)
        short_limit = round(short_q["bid"] + (short_q["mid"] - short_q["bid"]) * 0.40, 2)
    finally:
        ib.disconnect()

    client = TradingClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
                            paper=False, url_override=cfg["alpaca_base_url"])
    contracts = client.get_option_contracts(GetOptionContractsRequest(
        underlying_symbols=[TICKER], expiration_date=expiry_alp, type=ContractType.PUT,
        strike_price_gte=str(long_k - 1), strike_price_lte=str(short_k + 1),
    )).option_contracts
    by_strike = {float(c.strike_price): c for c in contracts}
    if short_k not in by_strike or long_k not in by_strike:
        return {"error": f"strikes not listed on Alpaca: {short_k}/{long_k}"}
    long_sym, short_sym = by_strike[long_k].symbol, by_strike[short_k].symbol

    credit = round(short_limit - long_limit, 2)
    return {
        "expiry": expiry_alp, "short_strike": short_k, "long_strike": long_k,
        "long_symbol": long_sym, "short_symbol": short_sym,
        "long_limit": long_limit, "short_limit": short_limit,
        "target_credit": credit, "width": WIDTH,
        "max_risk_per_contract": round(WIDTH * 100 - credit * 100, 2),
    }


def fire_trade(modeled: dict) -> dict:
    """Actually submit the sequential-leg orders for an already-modeled trade.
    Only call this after explicit CEO confirmation -- re-models with fresh
    prices first since modeled numbers may be stale by confirmation time."""
    long_sym, short_sym = modeled["long_symbol"], modeled["short_symbol"]
    long_limit, short_limit = modeled["long_limit"], modeled["short_limit"]

    client = TradingClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
                            paper=False, url_override=cfg["alpaca_base_url"])

    long_order = client.submit_order(LimitOrderRequest(
        symbol=long_sym, qty=QTY, side=OrderSide.BUY,
        type=OrderType.LIMIT, time_in_force=TimeInForce.DAY, limit_price=long_limit,
    ))
    filled = False
    for _ in range(20):
        time.sleep(1)
        chk = client.get_order_by_id(long_order.id)
        if str(chk.status) == "OrderStatus.FILLED":
            filled = True
            break
        if str(chk.status) in ("OrderStatus.REJECTED", "OrderStatus.CANCELED"):
            return {"error": f"long leg {chk.status}, aborted before short leg"}
    if not filled:
        try:
            client.cancel_order_by_id(long_order.id)
        except Exception:
            pass
        return {"error": "long leg did not fill in time, aborted before short leg"}

    short_order = client.submit_order(LimitOrderRequest(
        symbol=short_sym, qty=QTY, side=OrderSide.SELL,
        type=OrderType.LIMIT, time_in_force=TimeInForce.DAY, limit_price=short_limit,
    ))
    short_filled = False
    for _ in range(20):
        time.sleep(1)
        chk = client.get_order_by_id(short_order.id)
        if str(chk.status) == "OrderStatus.FILLED":
            short_filled = True
            break
        if str(chk.status) in ("OrderStatus.REJECTED", "OrderStatus.CANCELED"):
            break

    return {
        "expiry": expiry_alp, "short_strike": short_k, "long_strike": long_k,
        "long_filled": filled, "short_filled": short_filled,
        "long_order_id": str(long_order.id), "short_order_id": str(short_order.id),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire", action="store_true",
                     help="Actually place the last-modeled trade (re-prices fresh first). "
                          "Use only after explicit CEO confirmation of an alerted signal.")
    args = ap.parse_args()

    state = load_state()

    if args.fire:
        if not state.get("awaiting_confirmation"):
            print("Nothing awaiting confirmation -- run without --fire first to check for a signal.")
            return
        sig = check_signals()   # fresh read, not the stale alert-time one
        print(f"Re-pricing and firing NFLX put credit spread (signal was {state['signal']['which']})...")
        modeled = model_trade(sig["price"])
        if "error" in modeled:
            telegram(f"⚠️ <b>NFLX fire attempt failed to re-price:</b> {modeled['error']}")
            print(f"ERROR: {modeled['error']}")
            return
        try:
            result = fire_trade(modeled)
        except Exception as e:
            result = {"error": f"unhandled exception: {e}"}
        state["awaiting_confirmation"] = not (("error" not in result) and result.get("long_filled"))
        state["fired"] = "error" not in result and result.get("long_filled", False)
        state["trade_result"] = result
        save_state(state)
        if "error" in result:
            telegram(f"⚠️ <b>NFLX fire failed:</b> {result['error']}")
            print(f"ERROR: {result['error']}")
        else:
            telegram(
                f"✅ <b>NFLX trade placed (confirmed)</b>\n"
                f"SELL {result['short_strike']}P / BUY {result['long_strike']}P, {result['expiry']}\n"
                f"Long filled: {result['long_filled']}  Short filled: {result['short_filled']}"
            )
            print(json.dumps(result, indent=2))
        return

    if state.get("awaiting_confirmation"):
        print(f"Already alerted at {state.get('alerted_at')}, awaiting CEO confirmation "
              f"(run with --fire once confirmed) -- not re-checking or re-alerting.")
        return
    if state.get("fired"):
        print(f"Already fired at {state['fired_at']} -- done.")
        return

    sig = check_signals()
    state["checks"] = state.get("checks", 0) + 1
    print(f"[{datetime.now(timezone.utc).isoformat()}] check #{state['checks']}")
    print(f"  price=${sig['price']} sma200=${sig['sma200']} sma50=${sig['sma50']} "
          f"rsi14={sig['rsi14']} 52wk_low=${sig['low_52wk']}")
    print(f"  Signal A (RSI<={RSI_RESET_MAX} + held low): {sig['signal_a']}")
    print(f"  Signal B (price > 200sma): {sig['signal_b']}")

    if not sig["triggered"]:
        save_state(state)
        print("No signal yet.")
        return

    print(f"*** TRIGGERED: Signal {sig['which']} *** Modeling trade (not placing)...")
    try:
        modeled = model_trade(sig["price"])
    except Exception as e:
        modeled = {"error": f"unhandled exception: {e}"}

    state["signal"] = sig
    state["modeled_trade"] = modeled
    if "error" not in modeled:
        state["awaiting_confirmation"] = True
        state["alerted_at"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    if "error" in modeled:
        telegram(f"⚠️ <b>NFLX signal fired ({sig['which']}) but could not model a trade:</b> "
                 f"{modeled['error']}\nWill retry on next check.")
        print(f"ERROR modeling trade: {modeled['error']}")
    else:
        telegram(
            f"🎯 <b>NFLX bottom signal fired: {sig['which']}</b>\n"
            f"Price ${sig['price']}, RSI {sig['rsi14']}, 200sma ${sig['sma200']}\n\n"
            f"Modeled trade, ready to confirm:\n"
            f"SELL 1x ${modeled['short_strike']}P / BUY 1x ${modeled['long_strike']}P, {modeled['expiry']}\n"
            f"Target credit ${modeled['target_credit']}  |  Max risk ${modeled['max_risk_per_contract']}\n\n"
            f"Say the word and this fires with fresh pricing."
        )
        print(json.dumps(modeled, indent=2))


if __name__ == "__main__":
    main()
