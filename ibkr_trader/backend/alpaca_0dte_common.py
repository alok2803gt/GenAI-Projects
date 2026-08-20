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
from alpaca.trading.enums import ContractType, OrderClass, OrderSide, OrderType, TimeInForce
from alpaca.trading.requests import (
    GetOptionContractsRequest, GetOrdersRequest, LimitOrderRequest, MarketOrderRequest,
    TrailingStopOrderRequest, TakeProfitRequest, StopLossRequest,
)
from alpaca.trading.enums import QueryOrderStatus

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


TOTAL_RISK_BUDGET_PCT = 0.50   # max fraction of combined net liq at risk across ALL strategies at once
PER_STRATEGY_CAP_PCT  = 0.05   # max fraction any SINGLE strategy may commit to one new trade
                                 # -- aligned to 5% 2026-08-20 to match EVC's own pre-trade
                                 # review and Rule 5 (config.max_position_pct), which were
                                 # both already 5%. Was 0.15 at launch, inconsistent with EVC
                                 # by design oversight -- reconciled after CEO caught the gap.


def cro_cfo_capital_budget(ib: IB, client) -> dict:
    """Cross-strategy capital allocation view (CRO/CFO), shared by every
    strategy's pre-trade review. Computes combined (IBKR + Alpaca) net liq,
    total capital currently committed across every open option position on
    Alpaca (the account's real execution venue since 2026-08-11) plus IBKR's
    own gross position value, and the resulting headroom against a total
    portfolio risk budget. Built 2026-08-20 (task 2026-08-20-006) -- until
    now every strategy's own capital config (csp_capital, position_size,
    etc.) was set independently, with nothing checking the SUM against the
    real account. "Capital committed" per Alpaca option leg is
    |qty * avg_entry_price * 100| -- a consistent proxy for capital at risk
    across heterogeneous strategies without needing each one's own bespoke
    max_risk formula.

    ib: a connected IB() instance (sync ib_insync calls, matches this
        module's existing style). client: an Alpaca TradingClient.
    """
    ibkr_net_liq = 0.0
    ibkr_deployed = 0.0
    try:
        if ib and ib.isConnected():
            vals = {v.tag: v.value for v in ib.accountValues()}
            ibkr_net_liq = float(vals.get("NetLiquidation", 0.0))
            ibkr_deployed = abs(float(vals.get("GrossPositionValue", 0.0)))
    except Exception:
        pass

    alpaca_equity = 0.0
    total_deployed_alpaca = 0.0
    try:
        acct = client.get_account()
        alpaca_equity = float(acct.equity)
        for p in client.get_all_positions():
            if getattr(p, "asset_class", None) and "OPTION" in str(p.asset_class):
                total_deployed_alpaca += abs(float(p.qty) * float(p.avg_entry_price) * 100)
    except Exception:
        pass

    combined_net_liq = ibkr_net_liq + alpaca_equity
    total_deployed = ibkr_deployed + total_deployed_alpaca
    total_budget = combined_net_liq * TOTAL_RISK_BUDGET_PCT
    headroom = max(0.0, total_budget - total_deployed)
    per_strategy_cap = combined_net_liq * PER_STRATEGY_CAP_PCT

    return {
        "combined_net_liq": round(combined_net_liq, 2),
        "ibkr_net_liq": round(ibkr_net_liq, 2),
        "alpaca_equity": round(alpaca_equity, 2),
        "total_deployed": round(total_deployed, 2),
        "total_budget": round(total_budget, 2),
        "headroom": round(headroom, 2),
        "per_strategy_cap": round(per_strategy_cap, 2),
        "total_risk_budget_pct": TOTAL_RISK_BUDGET_PCT,
        "per_strategy_cap_pct": PER_STRATEGY_CAP_PCT,
    }


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
    try:
        order = client.submit_order(LimitOrderRequest(
            symbol=symbol, qty=qty, side=side,
            type=OrderType.LIMIT, time_in_force=TimeInForce.DAY, limit_price=limit_price,
        ))
    except Exception as exc:
        # A raw API-level rejection (e.g. insufficient buying power) must be
        # treated exactly like a REJECTED order, not allowed to propagate --
        # an uncaught exception here used to skip straight past every
        # ENTRY_INCOMPLETE/partial-fill safety check in the caller (real
        # incident: DE 2026-08-19, see task 2026-08-19-005).
        print(f"    SUBMIT FAILED: {exc}")
        return False, None
    for _ in range(FILL_WAIT_S):
        time.sleep(1)
        try:
            chk = client.get_order_by_id(order.id)
        except Exception as exc:
            print(f"    status check failed: {exc}")
            continue
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


def price_ladder(side, bid, ask, mid):
    """3-step limit-price ladder: [favorable, mid, aggressive].
    favorable = 75% of the way from mid toward the best side (ask for SELL,
    bid for BUY) -- tries for meaningfully better than mid first.
    mid = the standard fallback price.
    aggressive = crosses to the opposite side -- guarantees a fill.
    Falls back to whatever single side exists if there's no two-sided market
    (mid is None)."""
    if mid is None:
        fallback = ask if side == OrderSide.SELL else bid
        fallback = fallback or 0.01
        return [round(fallback, 2)] * 3
    if side == OrderSide.SELL:
        favorable = round(mid + (ask - mid) * 0.75, 2)
        aggressive = bid if bid else round(mid * 0.5, 2)
    else:  # BUY (to close a short)
        favorable = round(mid - (mid - bid) * 0.75, 2) if bid else round(mid * 0.75, 2)
        aggressive = ask
    return [favorable, mid, aggressive]


def place_leg_with_ladder(client, symbol, side, label, qty, bid, ask, mid):
    """Like place_leg(), but walks a 3-step price ladder (favorable -> mid ->
    aggressive) instead of firing one flat price -- maximizes realized P&L
    on a close while still guaranteeing a fill if the favorable/mid steps
    don't get taken. Each step gets FILL_WAIT_S to fill before moving on.
    Validated 2026-08-20 (DE close): the favorable step filled at $10.10 vs
    an $8.55 mid, a real, meaningful improvement -- use this instead of a
    single flat place_leg() call whenever the goal is maximizing the close
    price, not just guaranteeing speed."""
    ladder = price_ladder(side, bid, ask, mid)
    step_labels = ["favorable", "mid", "aggressive"]
    for step_i, px in enumerate(ladder):
        print(f"  [{label}] step {step_i+1} ({step_labels[step_i]}): {side.value} {qty}x {symbol} @ ${px}")
        ok, fill_px = place_leg(client, symbol, side, px, label, qty)
        if ok:
            return True, fill_px
        if step_i < len(ladder) - 1:
            print(f"    not filled at step {step_i+1}, moving to next price")
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


# ── Single-stock bracket / trailing-stop primitives ──────────────────────
# Added for the Day Trader Alpaca migration (2026-08-19) -- these are the
# single-leg-equity analog of the condor-execution functions above. Real,
# verified field names against the installed alpaca-py SDK (checked
# 2026-08-19: Order.legs holds bracket child orders, OrderClass.BRACKET/
# TakeProfitRequest/StopLossRequest/TrailingStopOrderRequest all confirmed
# present) -- NOT yet live-fire-verified end-to-end (market closed at
# build time). Verify with a small real order before trusting this in an
# actively-trading loop.

def submit_bracket_buy(client, symbol, qty, limit_price, stop_price, target_price):
    """One order submission: BUY qty @ limit_price, with a native OCO
    exit pair (stop_loss + take_profit) that activates once the entry
    fills -- Alpaca's equivalent of IBKR's ocaGroup STP+LMT pair, but as
    ONE order instead of three. Returns the Alpaca Order object; its
    .legs (once populated) are the child stop/target orders."""
    req = LimitOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.BUY,
        type=OrderType.LIMIT, time_in_force=TimeInForce.DAY, limit_price=limit_price,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=round(target_price, 2)),
        stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
    )
    return client.submit_order(req)


def submit_entry_buy(client, symbol, qty, limit_price):
    """Plain entry BUY, no bracket -- used in trailing-stop mode, where the
    protective order can't be attached at entry time (it needs the real
    fill price to anchor the trail) and gets placed separately once filled
    via submit_trailing_stop_sell()."""
    req = LimitOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.BUY,
        type=OrderType.LIMIT, time_in_force=TimeInForce.DAY, limit_price=limit_price,
    )
    return client.submit_order(req)


def submit_trailing_stop_sell(client, symbol, qty, trail_percent):
    """Native server-side trailing stop -- Alpaca ratchets it, same as
    IBKR's TRAIL order type. Placed AFTER the entry fills (needs qty to
    match the real filled shares)."""
    req = TrailingStopOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.SELL,
        type=OrderType.TRAILING_STOP, time_in_force=TimeInForce.DAY, trail_percent=trail_percent,
    )
    return client.submit_order(req)


def submit_market_sell(client, symbol, qty):
    """Marketable exit -- EOD force-close or any other certainty-over-price close."""
    req = MarketOrderRequest(
        symbol=symbol, qty=qty, side=OrderSide.SELL,
        type=OrderType.MARKET, time_in_force=TimeInForce.DAY,
    )
    return client.submit_order(req)


def get_order(client, order_id):
    """Fresh order status by id -- for fill-detection polling (Alpaca has
    no batch-fetch-all-my-recent-trades call as cheap as IBKR's
    ib.trades()/ib.fills(), so this is called per-order-of-interest)."""
    return client.get_order_by_id(order_id)


def has_open_position(client, symbol):
    """Returns the Position object if one exists, else None. Used for
    exit-fill detection on bracket/trailing-stop-protected positions:
    once the position disappears, the broker-side protective order fired
    -- simpler and more robust than trying to track/match individual
    bracket child-leg order ids."""
    try:
        return client.get_open_position(symbol)
    except Exception:
        return None


def get_last_closing_fill(client, symbol):
    """Most recent FILLED sell order for symbol, or None. Used right after
    has_open_position() reports the position gone, to recover the real
    exit fill price/time for P&L."""
    orders = client.get_orders(GetOrdersRequest(
        status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=10, direction="desc",
    ))
    for o in orders:
        if str(o.side).lower().endswith("sell") and o.filled_avg_price and float(o.filled_qty or 0) > 0:
            return o
    return None


def close_stock_position_market(client, symbol):
    """Cancels any open orders on the position (bracket/trailing-stop legs
    included) AND submits a market order to flatten it -- Alpaca's own
    DELETE /positions/{symbol} semantics, one call instead of manually
    orchestrating cancel-then-sell. Returns the closing Order, or None if
    no position exists (already flat)."""
    try:
        return client.close_position(symbol)
    except Exception as exc:
        if "position does not exist" in str(exc).lower() or "404" in str(exc):
            return None
        raise
