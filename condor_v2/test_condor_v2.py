"""Offline tests -- no IBKR, no orders, no network. Run: ./.venv/bin/python test_condor_v2.py"""
import math
import tempfile
from pathlib import Path

import condor_v2 as C

REAL = {n: getattr(C, n) for n in ("close_condor_legs", "place_leg_ladder", "place_condor", "condor_value")}
SPOT, IV = 660.0, 0.12
CFG = dict(C.DEFAULTS, ticker="SPY", qty=1)


class FakeIB:
    def sleep(self, s):
        pass


def fake_quotes(ib, contracts):
    out = {}
    for name, c in contracts.items():
        mid = max(C.bs_price(SPOT, c.strike, C.minutes_to_close(), IV, c.right), 0.01)
        out[name] = {"bid": round(mid - 0.005, 3), "ask": round(mid + 0.005, 3), "mid": round(mid, 3)}
    return out


def setup():
    C.get_quotes = fake_quotes
    C.spot = lambda ib, t: SPOT
    tmp = Path(tempfile.mkdtemp())
    C.STATE_FILE, C.LOG_FILE, C.PAUSE_FLAG = tmp / "state.json", tmp / "log.jsonl", tmp / "PAUSED.flag"


setup()


def test_implied_vol_roundtrip():
    px = C.bs_price(660, 660, 300, 0.12, "C")
    assert abs(C.implied_vol(px, 660, 660, 300, "C") - 0.12) < 1e-3


def test_expected_move_and_strikes():
    setup()
    u, iv = C.expected_move(FakeIB(), CFG, "20260923", SPOT)
    assert abs(iv - IV) < 5e-3, iv
    assert abs(u - SPOT * IV * math.sqrt(C.minutes_to_close() / C.MIN_PER_YEAR)) < 0.01
    plan, err = C.build_condor(FakeIB(), CFG, "20260923", SPOT, u)
    assert err is None, err
    s = plan["strikes"]
    assert s["short_put"] == round(SPOT - 0.5 * u) and s["short_call"] == round(SPOT + 0.5 * u)
    assert s["long_put"] == s["short_put"] - 5 and s["long_call"] == s["short_call"] + 5
    assert plan["conservative"] < plan["credit_mid"]


def test_thin_credit_skipped():
    setup()
    cfg = dict(CFG, min_credit=99.0)
    plan, err = C.build_condor(FakeIB(), cfg, "20260923", SPOT, 3.0)
    assert plan is None and "credit too thin" in err


def test_zero_bid_wing_is_usable():
    setup()
    C.get_quotes = lambda ib, cs: {n: ({"bid": 0.0, "ask": 0.01, "mid": 0.005} if "long" in n
                                       else {"bid": 0.40, "ask": 0.41, "mid": 0.405}) for n in cs}
    plan, err = C.build_condor(FakeIB(), CFG, "20260923", SPOT, 2.0)
    assert err is None, f"a zero-bid wing must not void the condor: {err}"
    assert plan["credit_mid"] > 0
    C.get_quotes = fake_quotes


def test_price_ladder_directions():
    buy = C.price_ladder("BUY", 1.00, 1.10, 1.05)
    sell = C.price_ladder("SELL", 1.00, 1.10, 1.05)
    assert buy[0] < buy[1] < buy[2], buy          # buy: cheap -> mid -> through the ask
    assert sell[0] > sell[1] > sell[2], sell      # sell: rich -> mid -> through the bid
    assert buy[2] == 1.12 and sell[2] == 0.98, "last step must cross the quote by CROSS to fill"
    ubuy = C.price_ladder("BUY", 1.00, 1.10, 1.05, urgent=True)
    usell = C.price_ladder("SELL", 1.00, 1.10, 1.05, urgent=True)
    assert all(p > 1.10 for p in ubuy) and all(p < 1.00 for p in usell), "stop exits must cross from step 1"
    assert C.price_ladder("SELL", 0.01, 0.02, 0.015, urgent=True)[-1] == 0.01, "never price below a penny"


def test_entry_records_slot_and_stop():
    setup()
    state = {"session": "t", "open": [], "closed": [], "realized": 0.0}
    C.place_condor = lambda ib, cfg, cs, lim, qty: (True, {n: lim[n][2] for n in cs}, "all_4_filled")
    C.enter(FakeIB(), CFG, "20260923", state, False, "10:01")
    pos = state["open"][0]
    assert pos["slot"] == "10:01", "scheduled slot must be stored for restart safety"
    assert abs(pos["stop_loss_usd"] - 1.0 * pos["credit"] * 100) < 1e-9, "stop must be 1x credit (2026-10-13)"


def test_stop_closes_and_books_pnl():
    setup()
    state = {"session": "t", "open": [], "closed": [], "realized": 0.0}
    C.place_condor = lambda ib, cfg, cs, lim, qty: (True, {n: lim[n][2] for n in cs}, "all_4_filled")
    C.enter(FakeIB(), CFG, "20260923", state, False, "10:01")
    pos = state["open"][0]
    closed = {}
    C.close_condor_legs = lambda ib, cfg, cs, lim, qty: (closed.update(n=len(cs)) or (True, {n: 0.0 for n in cs}))
    C.condor_value = lambda ib, cfg, e, st: (pos["credit"] * 2, {n: None for n in pos["strikes"]},
                                             {n: {"bid": 1, "ask": 1.01, "mid": 1.005} for n in pos["strikes"]})
    C.close_condor(FakeIB(), CFG, "20260923", pos, state, "stop 0.5x", False)
    assert closed["n"] == 4 and not state["open"] and state["closed"][0]["close_reason"] == "stop 0.5x"
    assert state["realized"] > 0, "exit at 0 should book the full credit"


def test_near_money_closes_only_threatened_side():
    setup()
    calls = []
    C.place_leg_ladder = lambda ib, cfg, c, action, label, qty, b, a, m, **kw: calls.append((action, label)) or (True, m)
    pos = {"id": "x", "qty": 1, "strikes": {"short_put": 655, "long_put": 650, "short_call": 665, "long_call": 670}}
    C.close_near_money_side(FakeIB(), CFG, "20260923", pos, 665.1, {"open": []}, False)
    assert [a for a, _ in calls] == ["BUY", "SELL"] and all("call" in l for _, l in calls), calls
    calls.clear()
    C.close_near_money_side(FakeIB(), CFG, "20260923", pos, 660.0, {"open": []}, False)
    assert not calls, "closed a side that was not near the money"


def test_partial_entry_unwinds_and_does_not_pause():
    setup()
    calls = []
    C.close_condor_legs = REAL["close_condor_legs"]
    C.place_leg_ladder = lambda ib, cfg, c, action, label, qty, b, a, m, **kw: calls.append((action, label)) or (True, m)
    C.place_condor = lambda ib, cfg, cs, lim, qty: (False, {"long_put": 0.10, "long_call": 0.08,
                                                           "short_put": 0.90, "short_call": None}, "short_call_failed")
    state = {"session": "t", "open": [], "closed": [], "realized": 0.0}
    C.enter(FakeIB(), CFG, "20260923", state, False, "10:01")
    assert not C.PAUSE_FLAG.exists(), "a partial entry that unwinds cleanly must not pause the day"
    assert not state["open"] and len(state["unwound"]) == 1
    assert calls[0] == ("BUY", "close short put"), f"the filled short must be bought back first: {calls}"
    assert sorted(a for a, _ in calls[1:]) == ["SELL", "SELL"], calls


def test_stop_close_keeps_wing_and_retries():
    setup()
    C.close_condor_legs = REAL["close_condor_legs"]
    fail_short_call = [True]
    calls = []

    def ladder(ib, cfg, c, action, label, qty, b, a, m, urgent=False):
        assert urgent, f"stop exit '{label}' must be priced to fill immediately"
        calls.append(label)
        if label == "close short call" and fail_short_call[0]:
            return False, None
        return True, 0.50

    C.place_leg_ladder = ladder
    strikes = {"short_put": 655, "long_put": 650, "short_call": 665, "long_call": 670}
    C.condor_value = lambda ib, cfg, e, st: (3.0, {n: object() for n in st},
                                             {n: {"bid": 0.49, "ask": 0.51, "mid": 0.50} for n in st})
    pos = {"id": "p", "qty": 1, "credit": 1.40, "strikes": strikes, "stop_loss_usd": 70}
    state = {"session": "t", "open": [pos], "closed": [], "realized": 0.0}
    C.close_condor(FakeIB(), CFG, "20260923", pos, state, "stop 0.5x", False)
    assert "close long call" not in calls, "sold the call wing while its short was still open"
    assert state["open"] == [pos] and set(pos["close_fills"]) == {"short_put", "long_put"}
    fail_short_call[0] = False
    calls.clear()
    C.close_condor(FakeIB(), CFG, "20260923", pos, state, "stop 0.5x", False)
    assert calls == ["close short call", "close long call"], f"retry must only touch the open legs: {calls}"
    assert not state["open"] and state["closed"][0]["close_reason"] == "stop 0.5x"
    C.condor_value = REAL["condor_value"]


def test_entry_requotes_each_leg():
    setup()
    seen = []
    C.get_quotes = lambda ib, cs: {n: {"bid": 0.50, "ask": 0.52, "mid": 0.51} for n in cs}
    C.place_leg_ladder = lambda ib, cfg, c, action, label, qty, b, a, m, **kw: seen.append((label, b)) or (True, m)
    stale = {n: (9.0, 9.2, 9.1) for n in C.LEGS}
    ok, fills, _ = REAL["place_condor"](FakeIB(), CFG, {n: object() for n in C.LEGS}, stale, 1)
    assert ok and all(b == 0.50 for _, b in seen), f"legs must use fresh quotes, not the plan's: {seen}"
    C.get_quotes = fake_quotes


def recording_ladder(calls):
    return lambda ib, cfg, c, action, label, qty, b, a, m, **kw: calls.append((action, label, qty)) or (True, m)


def test_near_money_closes_itm_short_beyond_dist():
    # 2026-10-06: a short put $0.79 in the money was skipped by the old |spot - strike| rule and assigned
    setup()
    calls = []
    C.place_leg_ladder = recording_ladder(calls)
    pos = {"id": "x", "qty": 1, "strikes": {"short_put": 661, "long_put": 656, "short_call": 663, "long_call": 668}}
    C.close_near_money_side(FakeIB(), CFG, "20260923", pos, 660.21, {"open": []}, False)
    assert [(a, l) for a, l, _ in calls] == [("BUY", "close short put"), ("SELL", "close long put")], calls
    assert set(pos["legs_gone"]) == {"short_put", "long_put"}, "the call side ($2.79 out of the money) must stay open"


def seven_condors():
    return {"open": [{"id": f"p{i}", "qty": 1, "credit": 0.80,
                      "strikes": {"short_put": 660, "long_put": 655, "short_call": 665, "long_call": 670}}
                     for i in range(7)], "closed": [], "realized": 0.0}


def test_reconcile_marks_shorts_ibkr_closed_and_pass_buys_only_the_rest():
    # 2026-10-06 15:42: IBKR bought back 6 of 7 short 779P; the 15:58 close then re-bought four,
    # leaving the account long. Now the account is reconciled first and only the real short is bought.
    setup()
    calls, real_held = [], C.held_options
    C.place_leg_ladder = recording_ladder(calls)
    C.held_options = lambda ib, cfg, e: {(660.0, "P"): -1, (655.0, "P"): 7, (665.0, "C"): -7, (670.0, "C"): 7}
    state = seven_condors()
    C.reconcile_broker(FakeIB(), CFG, "20260923", state, False)
    assert [a for a, _, _ in calls] == ["SELL"] * 6, f"no buys; six orphaned wings sold: {calls}"
    assert sum("short_put" in p.get("legs_gone", []) for p in state["open"]) == 6
    calls.clear()
    for pos in list(state["open"]):
        C.close_near_money_side(FakeIB(), CFG, "20260923", pos, 660.10, state, False, 0.25)
    assert [a for a, _, _ in calls] == ["BUY", "SELL"], f"only the one real short is bought back: {calls}"
    C.held_options = real_held


def test_reconcile_ignores_a_wing_netting_another_short():
    # condor B's put wing sits on condor A's short strike: the account nets them to zero there
    setup()
    calls, real_held = [], C.held_options
    C.place_leg_ladder = recording_ladder(calls)
    a = {"id": "a", "qty": 1, "credit": 0.8, "strikes": {"short_put": 660, "long_put": 655, "short_call": 668, "long_call": 673}}
    b = {"id": "b", "qty": 1, "credit": 0.8, "strikes": {"short_put": 665, "long_put": 660, "short_call": 670, "long_call": 675}}
    state = {"open": [a, b], "closed": [], "realized": 0.0}
    C.held_options = lambda ib, cfg, e: {(655.0, "P"): 1, (665.0, "P"): -1, (668.0, "C"): -1, (673.0, "C"): 1,
                                         (670.0, "C"): -1, (675.0, "C"): 1}  # 660P nets to 0: A short + B wing
    C.reconcile_broker(FakeIB(), CFG, "20260923", state, False)
    assert not calls and not any(p.get("legs_gone") for p in state["open"]), calls
    C.held_options = real_held


def test_reconcile_finishes_a_stop_close_ibkr_interrupted():
    setup()
    calls, real_held = [], C.held_options
    C.close_condor_legs = REAL["close_condor_legs"]
    C.condor_value = REAL["condor_value"]
    C.place_leg_ladder = lambda ib, cfg, c, action, label, qty, b, a, m, **kw: calls.append((action, label)) or (True, 0.40)
    pos = {"id": "s", "qty": 1, "credit": 1.0, "stop_loss_usd": 50, "close_reason": "stop 0.5x",
           "close_fills": {"short_call": 0.9, "long_call": 0.1},
           "strikes": {"short_put": 655, "long_put": 650, "short_call": 665, "long_call": 670}}
    state = {"session": "t", "open": [pos], "closed": [], "realized": 0.0}
    C.held_options = lambda ib, cfg, e: {(650.0, "P"): 1}  # IBKR bought the 655P back mid stop-close
    C.reconcile_broker(FakeIB(), CFG, "20260923", state, False)
    assert "short_put" in pos["close_fills"] and not calls
    C.close_condor(FakeIB(), CFG, "20260923", pos, state, "stop 0.5x", False)
    assert calls == [("SELL", "close long put")], f"the short must not be bought again: {calls}"
    assert not state["open"] and state["closed"][0]["close_reason"] == "stop 0.5x"
    C.held_options = real_held


def test_stop_buys_the_threatened_short_first():
    # 2026-10-08: the stop bought the cheap short call first while the short put ran away
    setup()
    calls = []
    C.place_leg_ladder = lambda ib, cfg, c, action, label, qty, b, a, m, **kw: calls.append(label) or (True, m)
    limits = {"short_call": (0.09, 0.11, 0.10), "short_put": (1.48, 1.52, 1.50),
              "long_call": (0.01, 0.02, 0.015), "long_put": (0.20, 0.22, 0.21)}
    REAL["close_condor_legs"](FakeIB(), CFG, {n: object() for n in limits}, limits, 1)
    assert calls[0] == "close short put", f"the dearer (threatened) short goes first: {calls}"


def test_urgent_ladder_requotes_every_step_with_short_waits():
    setup()
    asks, placed = iter([1.20, 1.40]), []
    real_leg = C.place_leg
    C.get_quotes = lambda ib, cs: {"leg": (lambda a: {"bid": a - 0.02, "ask": a, "mid": a - 0.01})(next(asks))}
    C.place_leg = lambda ib, cfg, c, action, px, label, qty, wait=None: placed.append((px, wait)) or (False, None)
    ok, _ = REAL["place_leg_ladder"](FakeIB(), CFG, object(), "BUY", "close short put", 1, 1.00, 1.02, 1.01, urgent=True)
    assert not ok and [p for p, _ in placed] == [1.04, 1.25, 1.75], f"fresh ask each step, last crosses 25%: {placed}"
    assert all(w == CFG["urgent_fill_wait_seconds"] for _, w in placed), placed
    C.place_leg, C.get_quotes = real_leg, fake_quotes


def test_urgent_last_step_crosses_at_least_25_percent():
    assert C.price_ladder("BUY", 0.98, 1.00, 0.99, urgent=True)[-1] == 1.25
    assert C.price_ladder("SELL", 1.00, 1.02, 1.01, urgent=True)[-1] == 0.75
    assert C.price_ladder("BUY", 0.18, 0.20, 0.19, urgent=True)[-1] == 0.30, "cheap legs still cross $0.10"


def test_early_pass_is_off_by_default():
    assert not C.DEFAULTS["near_money_early"], "IBKR's liquidation is reconciled, not pre-empted (2026-10-08)"
    assert C.DEFAULTS["reconcile_from"] == "14:45"


def test_early_pass_closes_both_sides_and_books_position():
    setup()
    calls = []
    C.place_leg_ladder = recording_ladder(calls)
    pos = {"id": "n", "qty": 1, "credit": 0.90,
           "strikes": {"short_put": 660, "long_put": 655, "short_call": 661, "long_call": 666}}
    state = {"open": [pos], "closed": [], "realized": 0.0}
    C.close_near_money_side(FakeIB(), CFG, "20260923", pos, 660.5, state, False, 1.0)
    assert len(calls) == 4 and not state["open"], f"both shorts within $1: whole condor closes ({calls})"
    assert state["closed"][0]["close_reason"] == "near-money close"
    assert abs(state["realized"] - (0.90 - pos["closed_cost"]) * 100) < 1e-6


def test_stop_after_early_pass_closes_only_remaining_side():
    setup()
    calls = []
    C.close_condor_legs = REAL["close_condor_legs"]
    C.condor_value = REAL["condor_value"]
    C.place_leg_ladder = lambda ib, cfg, c, action, label, qty, b, a, m, **kw: calls.append(label) or (True, 0.50)
    pos = {"id": "s", "qty": 1, "credit": 1.00, "stop_loss_usd": 50, "legs_closed": ["put"], "closed_cost": 0.30,
           "strikes": {"short_put": 655, "long_put": 650, "short_call": 665, "long_call": 670}}
    state = {"session": "t", "open": [pos], "closed": [], "realized": 0.0}
    C.close_condor(FakeIB(), CFG, "20260923", pos, state, "stop 0.5x", False)
    assert sorted(calls) == ["close long call", "close short call"], f"the closed put side must be left alone: {calls}"
    # call side: bought at 0.50, sold at 0.50 -> 0.00; plus 0.30 already paid on the put side
    assert not state["open"] and abs(state["closed"][0]["pnl"] - (1.00 - 0.30) * 100) < 1e-6


def test_held_options_reads_the_account():
    class P:
        def __init__(self, sym, k, r, q, exp="20260923", sec="OPT"):
            self.contract = type("Ct", (), {"secType": sec, "symbol": sym, "strike": k, "right": r,
                                            "lastTradeDateOrContractMonth": exp})()
            self.position = q

    class IBP(FakeIB):
        def positions(self):
            return [P("SPY", 779, "P", -1), P("SPY", 779, "P", 4), P("SPY", 774, "P", 3),
                    P("QQQ", 750, "P", -2), P("SPY", 779, "P", 9, exp="20260924"), P("SPY", 0, "", 100, sec="STK")]

    held = C.held_options(IBP(), CFG, "20260923")
    assert held == {(779.0, "P"): 3, (774.0, "P"): 3}, held


def test_quote_recorder_band_and_cap():
    import quote_recorder as Q
    keys = Q.wanted(777.4, band=10)
    assert len(keys) == 42 and (767.0, "P") in keys and (787.0, "C") in keys and (766.0, "C") not in keys
    kept = Q.trim(keys | Q.wanted(790.2, band=10), 790.2, 40)
    assert len(kept) == 40 and (790.0, "C") in kept and (767.0, "P") not in kept, "the cap drops the strikes farthest from spot"


def test_config_defaults_are_paper():
    cfg = C.DEFAULTS
    assert cfg["tws_port"] == 7497, "default port must be TWS PAPER"
    assert cfg["qty"] == 1 and len(cfg["entry_times"]) == 8 and cfg["entry_times"][0] == "11:01"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nall offline tests passed")
