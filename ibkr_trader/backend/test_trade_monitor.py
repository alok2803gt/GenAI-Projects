"""Unit tests for trade_monitor.py -- specifically, that it WOULD have
caught the real 2026-10-15 770P incident (registry said open, Ashley's own
live tracking had silently dropped it), using fake/mocked positions rather
than a real live trade.

Run: ../venv/bin/python test_trade_monitor.py
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import trade_monitor as tm

ET = ZoneInfo("America/New_York")
FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def fake_position(symbol, expiry, right, strike, qty):
    contract = SimpleNamespace(secType="OPT", symbol=symbol,
                                lastTradeDateOrContractMonth=expiry, right=right, strike=strike)
    return SimpleNamespace(contract=contract, position=qty)


def test_reproduces_the_real_770p_incident():
    """Registry says 770P is open for ashley_signal; Ashley's own live file
    is empty (exactly 2026-10-15's actual state after the bug). The drift
    check must flag it, and mark it HIGH PRIORITY since it expires today."""
    today = datetime.now(ET).strftime("%Y%m%d")
    key = ("SPY", today, "P", 770.0)
    registry_keys = {key: ("ASHLEY_770P_20261015_110457", "ashley_signal")}

    orig_file = tm.HERE
    tmpdir = tempfile.mkdtemp()
    tm.HERE = type(tm.HERE)(tmpdir)
    try:
        (tm.HERE / "ashleyklieu_open_positions.json").write_text("{}")
        findings = tm.check_live_tracking_drift(registry_keys, {}, dry_run=True)
        check("the exact 2026-10-15 scenario is flagged by the drift check", len(findings) == 1,
              f"findings={findings}")
        if findings:
            _, pos_id, strategy, expires_today = findings[0]
            check("flagged with the right pos_id", pos_id == "ASHLEY_770P_20261015_110457")
            check("flagged as expiring today (same-day expiry in this test)", expires_today is True)
    finally:
        tm.HERE = orig_file


def test_live_tracking_present_means_no_flag():
    """The mirror case: registry says open, and Ashley's live file DOES
    have it (the normal, healthy state) -- must NOT be flagged."""
    today = datetime.now(ET).strftime("%Y%m%d")
    key = ("SPY", today, "P", 770.0)
    registry_keys = {key: ("ASHLEY_770P_X", "ashley_signal")}

    orig_file = tm.HERE
    tmpdir = tempfile.mkdtemp()
    tm.HERE = type(tm.HERE)(tmpdir)
    try:
        live = {"770P": {"strike": 770.0, "right": "P", "expiry": today, "qty": 1}}
        (tm.HERE / "ashleyklieu_open_positions.json").write_text(json.dumps(live))
        findings = tm.check_live_tracking_drift(registry_keys, {}, dry_run=True)
        check("a genuinely-tracked position is never flagged", len(findings) == 0, f"findings={findings}")
    finally:
        tm.HERE = orig_file


def test_strategy_with_no_registered_tracker_is_skipped():
    """A strategy not in LIVE_TRACKERS (e.g. a butterfly babysitter that
    re-reads the registry fresh every run, with no second copy to drift
    from) must never be flagged by this check -- it structurally can't
    have this specific bug."""
    key = ("IWM", "20261020", "C", 230.0)
    registry_keys = {key: ("IWM_FLY_X", "iwm_butterfly")}
    findings = tm.check_live_tracking_drift(registry_keys, {}, dry_run=True)
    check("an un-registered strategy (no LIVE_TRACKERS entry) is silently skipped",
          len(findings) == 0, f"findings={findings}")


def test_universal_orphan_sweep_flags_a_true_stranger():
    today = datetime.now(ET).strftime("%Y%m%d")
    key = ("QQQ", today, "C", 500.0)
    live = {key: fake_position("QQQ", today, "C", 500.0, 1)}
    registry_keys = {}  # nothing registered at all

    orig_file = tm.HERE
    tmpdir = tempfile.mkdtemp()
    tm.HERE = type(tm.HERE)(tmpdir)
    try:
        # No state files exist in this empty tmpdir at all.
        findings = tm.check_universal_orphans(live, registry_keys, {}, dry_run=True)
        check("a position in no registry and no state file is flagged as an orphan",
              len(findings) == 1, f"findings={findings}")
        if findings:
            _, _, expires_today = findings[0]
            check("flagged as expiring today", expires_today is True)
    finally:
        tm.HERE = orig_file


def test_universal_orphan_sweep_ignores_registered_positions():
    today = datetime.now(ET).strftime("%Y%m%d")
    key = ("SPY", today, "P", 770.0)
    live = {key: fake_position("SPY", today, "P", 770.0, 1)}
    registry_keys = {key: ("ASHLEY_770P_X", "ashley_signal")}
    findings = tm.check_universal_orphans(live, registry_keys, {}, dry_run=True)
    check("a position the registry already claims is never flagged as an orphan",
          len(findings) == 0, f"findings={findings}")


def test_registry_open_keys_skips_multi_leg():
    """Multi-leg spreads (butterflies, condors) must be skipped, not
    silently mismatched under a made-up single-leg key -- see the
    docstring on registry_open_keys()."""
    orig_registry = tm.load_registry
    try:
        tm.load_registry = lambda: {"positions": {
            "SPREAD_1": {"phase": "open", "strategy": "iwm_butterfly",
                         "legs": [{"symbol": "IWM261020C00230000"}, {"symbol": "IWM261020C00235000"}]},
        }}
        keys = tm.registry_open_keys()
        check("a multi-leg open position produces no key at all", len(keys) == 0, f"keys={keys}")
    finally:
        tm.load_registry = orig_registry


if __name__ == "__main__":
    test_reproduces_the_real_770p_incident()
    test_live_tracking_present_means_no_flag()
    test_strategy_with_no_registered_tracker_is_skipped()
    test_universal_orphan_sweep_flags_a_true_stranger()
    test_universal_orphan_sweep_ignores_registered_positions()
    test_registry_open_keys_skips_multi_leg()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED'}: {FAILURES}")
    import sys
    sys.exit(1 if FAILURES else 0)
