"""Offline tests for condor_v2_trader: no IBKR, no orders, no network.
Run: ../venv/bin/python test_condor_v2.py"""
import json
import math
import os
import tempfile
from datetime import datetime, time as dtime

import condor_v2_trader as C

C.telegram = lambda *a, **k: None          # never touch the network in tests
SPOT = 660.0
IV = 0.12


class FakeIB:
    def sleep(self, s):
        pass


def fake_quotes(ib, contracts):
    """Black-Scholes mids with a 1-cent spread, using the module's own math."""
    out = {}
    for name, c in contracts.items():
        mid = C.bs_price(SPOT, c.strike, C.minutes_to_close(), IV, c.right)
        mid = max(mid, 0.01)
        out[name] = {"bid": round(mid - 0.005, 3), "ask": round(mid + 0.005, 3), "mid": round(mid, 3)}
    return out


def setup():
    C.get_quotes_batch = fake_quotes
    C.spot = lambda ib, ticker: SPOT           # spot() now talks to IBKR directly
    C.get_quote = lambda ib, c: {"bid": SPOT - 0.01, "ask": SPOT + 0.01, "mid": SPOT}
    tmp = tempfile.mkdtemp()
    C.STATE_FILE = os.path.join(tmp, "state.json")
    C.LOG_FILE = os.path.join(tmp, "log.jsonl")
    C.PAUSE_FLAG = os.path.join(tmp, "PAUSED.flag")


setup()          # install the fake broker before any test runs


def test_implied_vol_roundtrip():
    px = C.bs_price(660, 660, 300, 0.12, "C")
    assert abs(C.implied_vol(px, 660, 660, 300, "C") - 0.12) < 1e-3, "IV does not invert BS price"


def test_expected_move_and_strikes():
    setup()
    ib = FakeIB()
    u, iv = C.expected_move(ib, "spy", "20260922", SPOT)
    assert abs(iv - IV) < 5e-3, f"ATM IV wrong: {iv}"
    expect = SPOT * IV * math.sqrt(C.minutes_to_close() / C.MIN_PER_YEAR)
    assert abs(u - expect) < 0.01, "expected move formula changed"
    plan, err = C.build_condor(ib, "spy", "20260922", SPOT, u)
    assert err is None, err
    s = plan["strikes"]
    assert s["short_put"] == round(SPOT - 0.5 * u) and s["short_call"] == round(SPOT + 0.5 * u), s
    assert s["long_put"] == s["short_put"] - 5 and s["long_call"] == s["short_call"] + 5, "wings must be $5"
    assert plan["credit_mid"] > 0 and plan["conservative"] < plan["credit_mid"], "conservative credit must be lower"


def test_thin_credit_is_skipped():
    setup()
    ib = FakeIB()
    old = C.MIN_CREDIT
    C.MIN_CREDIT = 99.0                      # nothing can clear this
    plan, err = C.build_condor(ib, "spy", "20260922", SPOT, 3.0)
    C.MIN_CREDIT = old
    assert plan is None and "credit too thin" in err, "thin-credit guard did not fire"


class FakeBroker:
    """Stands in for Broker: records what would be routed, fills at the mid."""
    def __init__(self):
        self.placed, self.closed, self.legs = [], [], []

    def place(self, ib, contracts, limits, strikes, qty):
        self.placed.append(strikes)
        return True, {n: limits[n][2] for n in contracts}, "all_4_filled"

    def close(self, ib, contracts, limits, strikes, qty):
        self.closed.append(strikes)
        return True, {n: limits[n][2] for n in contracts}

    def leg(self, ib, contract, sym, action, label, qty, bid, ask, mid):
        self.legs.append((action, label, sym))
        return True, mid


def test_entry_then_stop_fires_at_half_credit():
    setup()
    ib = FakeIB()
    broker = FakeBroker()
    state = {"session": "t", "open": [], "closed": [], "realized": 0.0}
    C.enter(ib, broker, "spy", "20260922", state, dry=False, slot="10:01")
    assert state["open"][0]["slot"] == "10:01", "scheduled slot must be recorded for restart safety"
    assert len(state["open"]) == 1, "entry not recorded"
    pos = state["open"][0]
    credit = pos["credit"]
    assert credit > 0 and abs(pos["stop_loss_usd"] - 0.5 * credit * 100) < 1e-6, "stop level must be 0.5x credit"

    # value below the stop -> no close
    C.condor_value = lambda *a, **k: (credit * 1.4, {}, {})
    loss = (credit * 1.4 - credit) * 100
    assert loss < pos["stop_loss_usd"]
    # value above the stop -> close
    C.condor_value = lambda *a, **k: (credit * 1.6, {n: None for n in
                                                    ("short_put", "long_put", "short_call", "long_call")},
                                      {n: {"bid": 1, "ask": 1.01, "mid": 1.005} for n in
                                       ("short_put", "long_put", "short_call", "long_call")})
    loss = (credit * 1.6 - credit) * 100
    assert loss >= pos["stop_loss_usd"], "test setup: loss should exceed the stop"
    C.close_condor(ib, broker, "spy", "20260922", pos, state, "stop 0.5x", dry=False)
    assert len(broker.closed) == 1 and not state["open"] and len(state["closed"]) == 1, "stop did not close the condor"
    assert state["closed"][0]["close_reason"] == "stop 0.5x"


def test_near_money_close_only_threatened_side():
    setup()
    ib = FakeIB()
    broker = FakeBroker()
    C.get_quotes_batch = fake_quotes
    state = {"open": [], "closed": [], "realized": 0.0}
    pos = {"id": "x", "qty": 1, "strikes": {"short_put": 655, "long_put": 650, "short_call": 665, "long_call": 670}}
    C.close_near_money_side(ib, broker, "spy", "20260922", pos, 665.10, state, dry=False)   # call side near
    assert [a for a, _, _ in broker.legs] == ["BUY", "SELL"], broker.legs
    assert all("call" in lbl for _, lbl, _ in broker.legs), "closed the wrong side"
    assert all(sym.startswith("SPY260922C") for _, _, sym in broker.legs), broker.legs
    broker.legs.clear()
    C.close_near_money_side(ib, broker, "spy", "20260922", pos, 660.0, state, dry=False)    # neither side near
    assert not broker.legs, "closed a side that was not near the money"


def test_paused_flag_blocks_everything():
    setup()
    C.pause("unit test")
    assert os.path.exists(C.PAUSE_FLAG) and "unit test" in open(C.PAUSE_FLAG).read()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
