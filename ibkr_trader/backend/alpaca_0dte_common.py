"""
Shared execution + registry code for the SPY/SPX 0DTE iron condor live
tests (task from 2026-08-13 real-price backtest validation). Used by
alpaca_0dte_trader.py for both tickers -- the two strategies differ only
in their validated parameters (see TICKER_CONFIG), not in mechanics.

Execution pattern: 4 SEQUENTIAL single-leg orders, not one MLEG combo --
MLEG combo orders reject on this account for an unresolved reason (task
2026-08-12-003), proven workaround is sequential legs (see
alpaca_cohr_spread.py, the first live 4-leg condor placed this way).
Order: BUY both long legs first (defined-risk, no margin concern), THEN
SELL both short legs (now each is covered). On close: BUY BACK shorts
first (removes risk), THEN SELL longs -- mirrors this account's own
established rule (feedback_ibkr_spread_execution memory: "two-phase
closes, cover shorts first").

Pricing/quotes come from IBKR (this account's data source of record,
2026-08-11 architecture decision); Alpaca is execution-only.

Registry: alpaca_0dte_positions.json -- durable, append-only-ish record
so a portfolio-oversight sweep can actually see these positions exist.
This directly addresses task 2026-08-12-005 (Alpaca trades were
completely invisible to every existing tracking system) for this new
strategy going forward.
"""
import json
import time
from datetime import datetime, timezone, timedelta

from ib_insync import IB, Option, Stock, Index
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest

ET = timezone(timedelta(hours=-4))
REGISTRY_FILE = "alpaca_0dte_positions.json"
FILL_WAIT_S = 20


def now_et():
    return datetime.now(ET)


def load_config():
    with open("scanner_config.json") as f:
        return json.load(f)


def alpaca_client(cfg):
    return TradingClient(cfg["alpaca_api_key"], cfg["alpaca_secret_key"],
                          paper=False, url_override=cfg["alpaca_base_url"])


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


def spx_option(expiry_ibkr, strike, right):
    c = Option("SPX", expiry_ibkr, strike, right, "SMART", "100", "USD")
    c.tradingClass = "SPXW"   # PM-settled weekly (0DTE) -- required by IBKR, proven pattern (main.py)
    return c


def spy_option(expiry_ibkr, strike, right):
    return Option("SPY", expiry_ibkr, strike, right, "SMART")


def spx_spot(ib):
    idx = Index("SPX", "CBOE")
    ib.qualifyContracts(idx)
    td = ib.reqMktData(idx, "", False, False)
    ib.sleep(3)
    px = safe_px(td.last) or safe_px(td.close) or safe_px(td.marketPrice())
    ib.cancelMktData(idx)
    return px


def spy_spot(ib):
    """Spot price only (used to compute target strikes, not for order
    limits) -- falls back to last/close/marketPrice() same as spx_spot(),
    since bid/ask can legitimately come back NaN on a live SPY stock quote
    even when last-trade data is fine (confirmed 2026-08-13 dry-run: real
    bid=nan/ask=nan while last=776.35 was populated). Deliberately NOT
    applied to option-leg quotes (get_quote()) -- those need a real
    bid/ask for limit pricing, where a stale last-trade fallback would be
    the wrong kind of robustness."""
    c = Stock("SPY", "SMART", "USD")
    ib.qualifyContracts(c)
    td = ib.reqMktData(c, "", False, False)
    ib.sleep(3)
    bid, ask = safe_px(td.bid), safe_px(td.ask)
    mid = (bid + ask) / 2 if bid and ask else None
    px = mid or bid or ask or safe_px(td.last) or safe_px(td.close) or safe_px(td.marketPrice())
    ib.cancelMktData(c)
    return px


def target_px(q, is_short):
    mid = q["mid"]
    return round((mid - (mid - q["bid"]) * 0.40) if is_short else (mid + (q["ask"] - mid) * 0.40), 2)


def place_leg(client, symbol, side, limit_price, label, qty=1):
    print(f"  {label}: {side.value} {qty}x {symbol} @ ${limit_price}...")
    order = client.submit_order(LimitOrderRequest(
        symbol=symbol, qty=qty, side=side,
        type=OrderType.LIMIT, time_in_force=TimeInForce.DAY, limit_price=limit_price,
    ))
    for _ in range(FILL_WAIT_S):
        time.sleep(1)
        chk = client.get_order_by_id(order.id)
        if str(chk.status) == "OrderStatus.FILLED":
            print(f"    FILLED @ ${chk.filled_avg_price}")
            return True, float(chk.filled_avg_price)
        if str(chk.status) in ("OrderStatus.REJECTED", "OrderStatus.CANCELED"):
            print(f"    {chk.status}")
            return False, None
    try:
        client.cancel_order_by_id(order.id)
    except Exception:
        pass
    print("    did not fill in time -- cancelled")
    return False, None


def place_condor_sequential(client, syms, limits, qty=1):
    """syms/limits: dict with keys long_put, long_call, short_put, short_call.
    Returns (ok: bool, fills: dict of leg->fill_price or None, state: str
    describing how far it got, for safe error handling by the caller)."""
    fills = {}
    print("\n--- Leg 1/4: BUY long put ---")
    ok, px = place_leg(client, syms["long_put"], OrderSide.BUY, limits["long_put"], "long put", qty)
    fills["long_put"] = px
    if not ok:
        return False, fills, "long_put_failed_nothing_on"

    print("--- Leg 2/4: BUY long call ---")
    ok, px = place_leg(client, syms["long_call"], OrderSide.BUY, limits["long_call"], "long call", qty)
    fills["long_call"] = px
    if not ok:
        return False, fills, "long_call_failed_long_put_naked_long"

    print("--- Leg 3/4: SELL short put (covered by long put) ---")
    ok, px = place_leg(client, syms["short_put"], OrderSide.SELL, limits["short_put"], "short put", qty)
    fills["short_put"] = px
    if not ok:
        return False, fills, "short_put_failed_both_longs_uncovered"

    print("--- Leg 4/4: SELL short call (covered by long call) ---")
    ok, px = place_leg(client, syms["short_call"], OrderSide.SELL, limits["short_call"], "short call", qty)
    fills["short_call"] = px
    if not ok:
        return False, fills, "short_call_failed_3_of_4_on"

    return True, fills, "all_4_filled"


def close_condor_sequential(client, syms, limits, qty=1):
    """Two-phase close: BUY BACK shorts first (removes risk), THEN SELL longs.
    Matches this account's standing rule (feedback_ibkr_spread_execution)."""
    fills = {}
    print("\n--- Close 1/4: BUY TO CLOSE short call ---")
    ok, px = place_leg(client, syms["short_call"], OrderSide.BUY, limits["short_call"], "close short call", qty)
    fills["short_call"] = px

    print("--- Close 2/4: BUY TO CLOSE short put ---")
    ok2, px = place_leg(client, syms["short_put"], OrderSide.BUY, limits["short_put"], "close short put", qty)
    fills["short_put"] = px

    print("--- Close 3/4: SELL TO CLOSE long call ---")
    ok3, px = place_leg(client, syms["long_call"], OrderSide.SELL, limits["long_call"], "close long call", qty)
    fills["long_call"] = px

    print("--- Close 4/4: SELL TO CLOSE long put ---")
    ok4, px = place_leg(client, syms["long_put"], OrderSide.SELL, limits["long_put"], "close long put", qty)
    fills["long_put"] = px

    return all([ok, ok2, ok3, ok4]), fills


def get_alpaca_symbols(client, ticker, expiry_alp, put_lo, put_hi, call_lo, call_hi):
    """ticker: 'SPY' or 'SPX' (SPX query surfaces real SPXW-rooted contracts --
    confirmed 2026-08-13, querying underlying_symbols=['SPXW'] returns 0 results,
    must query 'SPX' even though returned contracts use the SPXW ticker root)."""
    puts = client.get_option_contracts(GetOptionContractsRequest(
        underlying_symbols=[ticker], expiration_date=expiry_alp, type=ContractType.PUT,
        strike_price_gte=str(put_lo), strike_price_lte=str(put_hi),
    )).option_contracts
    calls = client.get_option_contracts(GetOptionContractsRequest(
        underlying_symbols=[ticker], expiration_date=expiry_alp, type=ContractType.CALL,
        strike_price_gte=str(call_lo), strike_price_lte=str(call_hi),
    )).option_contracts
    put_by_strike = {float(c.strike_price): c.symbol for c in puts}
    call_by_strike = {float(c.strike_price): c.symbol for c in calls}
    return put_by_strike, call_by_strike


# ── Registry ──────────────────────────────────────────────────────────
def load_registry():
    try:
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"positions": {}, "closed": []}


def save_registry(reg):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(reg, f, indent=2)


def register_position(pos_id, ticker, strategy, legs, net_entry_credit, qty, max_risk,
                       profit_target_usd, hard_close_time, entry_time, notes=""):
    reg = load_registry()
    reg["positions"][pos_id] = {
        "pos_id": pos_id, "ticker": ticker, "strategy": strategy, "legs": legs,
        "net_entry_credit": net_entry_credit, "qty": qty, "max_risk": max_risk,
        "profit_target_usd": profit_target_usd, "hard_close_time": hard_close_time,
        "phase": "open", "entry_time": entry_time, "notes": notes,
        "close_reason": None, "close_pnl": None, "closed_at": None,
    }
    save_registry(reg)
    return reg["positions"][pos_id]


def close_position(pos_id, close_reason, close_pnl):
    reg = load_registry()
    if pos_id in reg["positions"]:
        pos = reg["positions"].pop(pos_id)
        pos["phase"] = "closed"
        pos["close_reason"] = close_reason
        pos["close_pnl"] = close_pnl
        pos["closed_at"] = now_et().isoformat()
        reg["closed"].append(pos)
        save_registry(reg)
        return pos
    return None
