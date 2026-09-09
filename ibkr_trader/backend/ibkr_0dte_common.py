"""
IBKR-native execution primitives for strategies that previously placed
orders on Alpaca (Ashley, Butterflies, GOOG Condor) -- built 2026-09-07
after Alpaca's real account balance dropped to $0 buying power (equity
$75, fully consumed by an existing NKE position), making Alpaca no longer
usable for any of them. Mirrors alpaca_0dte_common.py's function shapes
(place_leg / price_ladder / place_leg_with_ladder / place_*_sequential /
close_*_sequential / has_open_position / get_last_closing_fill) so the
call sites in each strategy file need minimal changes -- same sequencing,
same two-phase-close discipline, same price-ladder logic, just IBKR order
objects instead of Alpaca ones.

Sequencing rule (feedback_ibkr_spread_execution memory, applied here for
the first time to real option spreads on IBKR rather than Alpaca): BUY
longs first (defined-risk, no margin concern), THEN sell shorts (now
covered). On close: BUY BACK shorts first (removes risk), THEN SELL longs.
Deliberately sequential, single-leg orders throughout -- BAG/combo orders
are known to reject on this account for equity options (same root issue
that pushed these strategies to Alpaca in the first place; Alpaca's own
MLEG combo orders reject too, so this was never actually an IBKR-specific
limitation -- sequential legs is the proven workaround everywhere).

Registry (alpaca_0dte_positions.json via register_position/close_position)
and the CRO/CFO capital-budget view (cro_cfo_capital_budget) are both
broker-agnostic already -- reused as-is from alpaca_0dte_common.py, not
duplicated here. cro_cfo_capital_budget's ibkr_deployed figure comes from
IBKR's own GrossPositionValue account value, so it automatically reflects
these positions once they're placed here -- no change needed there.
"""
import asyncio
import re
import time

from ib_insync import LimitOrder, MarketOrder, Option

FILL_WAIT_S = 20
_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def occ_symbol(ticker, strike, right, yymmdd):
    """Standard OCC option symbol -- ROOT + YYMMDD + C/P + 8-digit strike,
    no padding (e.g. SPY260907C00591000). Broker-agnostic -- used purely as
    a durable, parseable registry label (see parse_occ_symbol below), not
    for order routing (orders go by qualified Option contract, not this
    string). Same compact format this account's Alpaca-era code already
    used, kept identical on purpose so every existing downstream consumer
    that parses a registry symbol (butterfly babysitters, etc.) keeps
    working unchanged after the IBKR migration."""
    strike_str = f"{int(round(strike * 1000)):08d}"
    return f"{ticker.upper()}{yymmdd}{right.upper()}{strike_str}"


def parse_occ_symbol(sym):
    """Inverse of occ_symbol(). Returns (ticker, expiry_ibkr 'YYYYMMDD',
    strike, right). Regex-based, not fixed string offsets -- works for any
    root length (SPY/QQQ/IWM are 3 chars, GOOG is 4) unlike the original
    ad hoc sym[-15:-9]/sym[-9] slicing this replaces, which only happened
    to work for 3-char roots."""
    m = _OCC_RE.match(sym)
    if not m:
        raise ValueError(f"not a valid OCC symbol: {sym!r}")
    ticker, yymmdd, right, strike_str = m.groups()
    return ticker, "20" + yymmdd, int(strike_str) / 1000.0, right


def occ_contract(ticker, expiry_ibkr, strike, right):
    """Option contract for ticker's own tradingClass on SMART -- matches
    this account's established pattern for QQQ/IWM (both carry a sparse
    ambiguous secondary tradingClass otherwise, see butterfly_babysitter_
    common.py's module docstring) and is a safe no-op for SPY/GOOG, whose
    default tradingClass already equals the ticker itself. NOT yet
    qualified -- caller must still call ib.qualifyContracts(...)."""
    return Option(ticker, expiry_ibkr, strike, right, "SMART", tradingClass=ticker)
IB_STILL_WORKING = {"PendingSubmit", "PreSubmitted", "Submitted", "ApiPending"}
IB_DONE_NOT_FILLED = {"Cancelled", "ApiCancelled", "Inactive", "Expired"}


def ibkr_place_leg(ib, contract, action, limit_price, label, qty=1, fill_wait_s=FILL_WAIT_S):
    """action: 'BUY' or 'SELL'. contract must already be qualified
    (ib.qualifyContracts already called). Same fill/cancel/settle-wait
    shape as alpaca_0dte_common.place_leg(), including the race-on-cancel
    handling (order can fill in the gap between our timeout check and the
    cancel request actually landing)."""
    px = round(limit_price, 2)
    print(f"  {label}: {action} {qty}x {contract.localSymbol or contract.symbol} @ ${px}...")
    order = LimitOrder(action, qty, px, tif="DAY")
    try:
        trade = ib.placeOrder(contract, order)
    except Exception as exc:
        print(f"    SUBMIT FAILED: {exc}")
        return False, None

    for _ in range(fill_wait_s):
        ib.sleep(1)
        status = trade.orderStatus.status
        if status == "Filled":
            fill_px = trade.orderStatus.avgFillPrice
            print(f"    FILLED @ ${fill_px}")
            return True, float(fill_px)
        if status in IB_DONE_NOT_FILLED:
            print(f"    {status}")
            return False, None

    try:
        ib.cancelOrder(order)
    except Exception:
        pass
    for _ in range(5):
        ib.sleep(1)
        status = trade.orderStatus.status
        if status == "Filled":
            fill_px = trade.orderStatus.avgFillPrice
            print(f"    actually FILLED @ ${fill_px} (raced the cancel)")
            return True, float(fill_px)
        if status in IB_DONE_NOT_FILLED:
            break
    print("    did not fill in time -- cancelled (settled)")
    return False, None


async def ibkr_place_leg_async(ib, contract, action, limit_price, label, qty=1, fill_wait_s=FILL_WAIT_S):
    """Async-context sibling of ibkr_place_leg() -- for scripts that use
    ib.connectAsync()/asyncio throughout (ashleyklieu_trigger_executor.py),
    where calling the sync ib.sleep() from inside a running event loop
    raises 'This event loop is already running'. Uses `await
    asyncio.sleep()` instead, which yields to the same loop ib_insync's own
    connection runs on -- fill/status updates still arrive normally.
    Otherwise identical to ibkr_place_leg()."""
    px = round(limit_price, 2)
    print(f"  {label}: {action} {qty}x {contract.localSymbol or contract.symbol} @ ${px}...")
    order = LimitOrder(action, qty, px, tif="DAY")
    try:
        trade = ib.placeOrder(contract, order)
    except Exception as exc:
        print(f"    SUBMIT FAILED: {exc}")
        return False, None

    for _ in range(fill_wait_s):
        await asyncio.sleep(1)
        status = trade.orderStatus.status
        if status == "Filled":
            fill_px = trade.orderStatus.avgFillPrice
            print(f"    FILLED @ ${fill_px}")
            return True, float(fill_px)
        if status in IB_DONE_NOT_FILLED:
            print(f"    {status}")
            return False, None

    try:
        ib.cancelOrder(order)
    except Exception:
        pass
    for _ in range(5):
        await asyncio.sleep(1)
        status = trade.orderStatus.status
        if status == "Filled":
            fill_px = trade.orderStatus.avgFillPrice
            print(f"    actually FILLED @ ${fill_px} (raced the cancel)")
            return True, float(fill_px)
        if status in IB_DONE_NOT_FILLED:
            break
    print("    did not fill in time -- cancelled (settled)")
    return False, None


async def ibkr_place_leg_with_ladder_async(ib, contract, action, label, qty, bid, ask, mid):
    ladder = ibkr_price_ladder(action, bid, ask, mid)
    step_labels = ["favorable", "mid", "aggressive"]
    for step_i, px in enumerate(ladder):
        print(f"  [{label}] step {step_i+1} ({step_labels[step_i]}): {action} {qty}x "
              f"{contract.localSymbol or contract.symbol} @ ${px}")
        ok, fill_px = await ibkr_place_leg_async(ib, contract, action, px, label, qty)
        if ok:
            return True, fill_px
        if step_i < len(ladder) - 1:
            print(f"    not filled at step {step_i+1}, moving to next price")
    return False, None


def ibkr_price_ladder(action, bid, ask, mid):
    """Same 3-step [favorable, mid, aggressive] ladder as
    alpaca_0dte_common.price_ladder(), action as 'BUY'/'SELL' string."""
    if mid is None:
        fallback = ask if action == "SELL" else bid
        fallback = fallback or 0.01
        return [round(fallback, 2)] * 3
    mid = round(mid, 2)
    if action == "SELL":
        favorable = round(mid + (ask - mid) * 0.75, 2)
        aggressive = round(bid, 2) if bid else round(mid * 0.5, 2)
    else:  # BUY (to close a short, or to open a long)
        favorable = round(mid - (mid - bid) * 0.75, 2) if bid else round(mid * 0.75, 2)
        aggressive = round(ask, 2)
    return [favorable, mid, aggressive]


def ibkr_place_leg_with_ladder(ib, contract, action, label, qty, bid, ask, mid):
    ladder = ibkr_price_ladder(action, bid, ask, mid)
    step_labels = ["favorable", "mid", "aggressive"]
    for step_i, px in enumerate(ladder):
        print(f"  [{label}] step {step_i+1} ({step_labels[step_i]}): {action} {qty}x "
              f"{contract.localSymbol or contract.symbol} @ ${px}")
        ok, fill_px = ibkr_place_leg(ib, contract, action, px, label, qty)
        if ok:
            return True, fill_px
        if step_i < len(ladder) - 1:
            print(f"    not filled at step {step_i+1}, moving to next price")
    return False, None


def _leg(ib, contracts, limits, name, action, label, qty):
    val = limits[name]
    if isinstance(val, (tuple, list)):
        bid, ask, mid = val
        return ibkr_place_leg_with_ladder(ib, contracts[name], action, label, qty, bid, ask, mid)
    return ibkr_place_leg(ib, contracts[name], action, val, label, qty)


def ibkr_place_condor_sequential(ib, contracts, limits, qty=1):
    """contracts/limits keys: long_put, long_call, short_put, short_call.
    contracts values must already be qualified Option contracts.
    limits[leg]: flat price (float) or (bid, ask, mid) tuple -> ladder.
    Returns (ok, fills dict, state string) -- same shape as
    alpaca_0dte_common.place_condor_sequential()."""
    fills = {}

    print("\n--- Leg 1/4: BUY long put ---")
    ok, px = _leg(ib, contracts, limits, "long_put", "BUY", "long put", qty)
    fills["long_put"] = px
    if not ok:
        return False, fills, "long_put_failed_nothing_on"

    print("--- Leg 2/4: BUY long call ---")
    ok, px = _leg(ib, contracts, limits, "long_call", "BUY", "long call", qty)
    fills["long_call"] = px
    if not ok:
        return False, fills, "long_call_failed_long_put_naked_long"

    print("--- Leg 3/4: SELL short put (covered by long put) ---")
    ok, px = _leg(ib, contracts, limits, "short_put", "SELL", "short put", qty)
    fills["short_put"] = px
    if not ok:
        return False, fills, "short_put_failed_both_longs_uncovered"

    print("--- Leg 4/4: SELL short call (covered by long call) ---")
    ok, px = _leg(ib, contracts, limits, "short_call", "SELL", "short call", qty)
    fills["short_call"] = px
    if not ok:
        return False, fills, "short_call_failed_3_of_4_on"

    return True, fills, "all_4_filled"


def ibkr_close_condor_sequential(ib, contracts, limits, qty=1):
    """Two-phase close: BUY BACK shorts first, THEN SELL longs."""
    fills = {}

    print("\n--- Close 1/4: BUY TO CLOSE short call ---")
    ok, px = _leg(ib, contracts, limits, "short_call", "BUY", "close short call", qty)
    fills["short_call"] = px

    print("--- Close 2/4: BUY TO CLOSE short put ---")
    ok2, px = _leg(ib, contracts, limits, "short_put", "BUY", "close short put", qty)
    fills["short_put"] = px

    print("--- Close 3/4: SELL TO CLOSE long call ---")
    ok3, px = _leg(ib, contracts, limits, "long_call", "SELL", "close long call", qty)
    fills["long_call"] = px

    print("--- Close 4/4: SELL TO CLOSE long put ---")
    ok4, px = _leg(ib, contracts, limits, "long_put", "SELL", "close long put", qty)
    fills["long_put"] = px

    return all([ok, ok2, ok3, ok4]), fills


async def _leg_async(ib, contracts, limits, name, action, label, qty):
    val = limits[name]
    if isinstance(val, (tuple, list)):
        bid, ask, mid = val
        return await ibkr_place_leg_with_ladder_async(ib, contracts[name], action, label, qty, bid, ask, mid)
    return await ibkr_place_leg_async(ib, contracts[name], action, val, label, qty)


async def ibkr_place_condor_sequential_async(ib, contracts, limits, qty=1):
    """Async-context sibling of ibkr_place_condor_sequential() -- for
    callers using ib.connectAsync()/asyncio throughout (main.py's strategy
    coroutines), where the sync version's ib.sleep() would raise 'This
    event loop is already running'. Otherwise identical."""
    fills = {}

    print("\n--- Leg 1/4: BUY long put ---")
    ok, px = await _leg_async(ib, contracts, limits, "long_put", "BUY", "long put", qty)
    fills["long_put"] = px
    if not ok:
        return False, fills, "long_put_failed_nothing_on"

    print("--- Leg 2/4: BUY long call ---")
    ok, px = await _leg_async(ib, contracts, limits, "long_call", "BUY", "long call", qty)
    fills["long_call"] = px
    if not ok:
        return False, fills, "long_call_failed_long_put_naked_long"

    print("--- Leg 3/4: SELL short put (covered by long put) ---")
    ok, px = await _leg_async(ib, contracts, limits, "short_put", "SELL", "short put", qty)
    fills["short_put"] = px
    if not ok:
        return False, fills, "short_put_failed_both_longs_uncovered"

    print("--- Leg 4/4: SELL short call (covered by long call) ---")
    ok, px = await _leg_async(ib, contracts, limits, "short_call", "SELL", "short call", qty)
    fills["short_call"] = px
    if not ok:
        return False, fills, "short_call_failed_3_of_4_on"

    return True, fills, "all_4_filled"


async def ibkr_close_condor_sequential_async(ib, contracts, limits, qty=1):
    """Async-context sibling of ibkr_close_condor_sequential()."""
    fills = {}

    print("\n--- Close 1/4: BUY TO CLOSE short call ---")
    ok, px = await _leg_async(ib, contracts, limits, "short_call", "BUY", "close short call", qty)
    fills["short_call"] = px

    print("--- Close 2/4: BUY TO CLOSE short put ---")
    ok2, px = await _leg_async(ib, contracts, limits, "short_put", "BUY", "close short put", qty)
    fills["short_put"] = px

    print("--- Close 3/4: SELL TO CLOSE long call ---")
    ok3, px = await _leg_async(ib, contracts, limits, "long_call", "SELL", "close long call", qty)
    fills["long_call"] = px

    print("--- Close 4/4: SELL TO CLOSE long put ---")
    ok4, px = await _leg_async(ib, contracts, limits, "long_put", "SELL", "close long put", qty)
    fills["long_put"] = px

    return all([ok, ok2, ok3, ok4]), fills


def ibkr_place_butterfly_sequential(ib, contracts, limits, qty=1):
    """3-leg long butterfly. contracts/limits keys: wing_lo (buy 1x),
    wing_hi (buy 1x), body (sell 2x). Body is one order with qty=2*qty --
    no ratio-order type needed, IBKR accepts any plain integer quantity on
    a single-leg order."""
    fills = {}

    print("\n--- Leg 1/3: BUY wing_lo ---")
    ok, px = _leg(ib, contracts, limits, "wing_lo", "BUY", "wing_lo", qty)
    fills["wing_lo"] = px
    if not ok:
        return False, fills, "wing_lo_failed_nothing_on"

    print("--- Leg 2/3: BUY wing_hi ---")
    ok, px = _leg(ib, contracts, limits, "wing_hi", "BUY", "wing_hi", qty)
    fills["wing_hi"] = px
    if not ok:
        return False, fills, "wing_hi_failed_wing_lo_naked_long"

    print("--- Leg 3/3: SELL body 2x (covered by both wings) ---")
    ok, px = _leg(ib, contracts, limits, "body", "SELL", "body", qty * 2)
    fills["body"] = px
    if not ok:
        return False, fills, "body_failed_both_wings_uncovered_long"

    return True, fills, "all_3_filled"


def ibkr_close_butterfly_sequential(ib, contracts, limits, qty=1):
    """Two-phase close: BUY BACK the short body first, THEN sell both wings."""
    fills = {}

    print("\n--- Close 1/3: BUY TO CLOSE body 2x ---")
    ok, px = _leg(ib, contracts, limits, "body", "BUY", "close body", qty * 2)
    fills["body"] = px

    print("--- Close 2/3: SELL TO CLOSE wing_lo ---")
    ok2, px = _leg(ib, contracts, limits, "wing_lo", "SELL", "close wing_lo", qty)
    fills["wing_lo"] = px

    print("--- Close 3/3: SELL TO CLOSE wing_hi ---")
    ok3, px = _leg(ib, contracts, limits, "wing_hi", "SELL", "close wing_hi", qty)
    fills["wing_hi"] = px

    return all([ok, ok2, ok3]), fills


def ibkr_has_open_position(ib, contract) -> bool:
    """True if a real, nonzero IBKR position exists for this exact
    contract (matched by conId -- contract must already be qualified).
    Mirrors alpaca_0dte_common.has_open_position()'s role: exit-fill
    detection by checking whether the position itself is gone, rather than
    tracking individual order ids."""
    for p in ib.positions():
        if p.contract.conId == contract.conId and p.position != 0:
            return True
    return False


def ibkr_get_last_closing_fill(ib, contract):
    """Most recent SELL (side == 'SLD') fill for this exact contract, or
    None. Used right after ibkr_has_open_position() reports the position
    gone, to recover the real exit fill price for P&L -- IBKR's ib.fills()
    is a cheap, already-cached local call (unlike Alpaca's, no extra API
    round trip needed)."""
    matches = [f for f in ib.fills()
               if f.contract.conId == contract.conId and f.execution.side == "SLD"]
    if not matches:
        return None
    return max(matches, key=lambda f: f.execution.time)
