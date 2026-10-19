"""Unit tests for the 2026-10-14 morning/afternoon DTE feature (CEO relayed,
direct from Ashley: 0DTE mornings only, afternoon trades are always the 3rd
trading day out). Covers the pure helpers: dte_kind_for(), the trading-day
calendar (_is_trading_day/_add_trading_days), and the OPEN_POSITIONS_FILE
round trip. The IBKR-dependent reconciliation (reconcile_all_open_positions)
and the live entry/force-close/EOD logic inside run_daily_watch() are NOT
covered here -- verify those the first time a real afternoon trigger and a
real overnight carry actually happen, the same caveat the original
reconcile_open_positions() docstring carried.

Run: ../venv/bin/python test_ashley_3dte.py
"""
from __future__ import annotations

import os
import tempfile
from datetime import date, datetime

import ashleyklieu_trigger_executor as m

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def et(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=m.ET)


def test_dte_kind_boundary():
    check("11:59 ET is 0dte", m.dte_kind_for(et(2026, 10, 14, 11, 59)) == "0dte")
    check("12:00 ET (the cutoff itself) is 3dte", m.dte_kind_for(et(2026, 10, 14, 12, 0)) == "3dte")
    check("09:30 ET (the open) is 0dte", m.dte_kind_for(et(2026, 10, 14, 9, 30)) == "0dte")
    check("15:59 ET is 3dte", m.dte_kind_for(et(2026, 10, 14, 15, 59)) == "3dte")


def test_add_trading_days_skips_weekend():
    # Tue -> Wed, Thu, Fri
    check("Tue 10-13 + 3 trading days = Fri 10-16",
          m._add_trading_days(date(2026, 10, 13), 3) == date(2026, 10, 16))
    # Thu -> Fri, (skip Sat/Sun), Mon, Tue
    check("Thu 10-15 + 3 trading days = Tue 10-20 (skips the weekend)",
          m._add_trading_days(date(2026, 10, 15), 3) == date(2026, 10, 20))
    # Fri -> Mon, Tue, Wed
    check("Fri 10-16 + 3 trading days = Wed 10-21 (skips the weekend)",
          m._add_trading_days(date(2026, 10, 16), 3) == date(2026, 10, 21))


def test_add_trading_days_skips_a_real_holiday():
    # 2026-11-26 (Thanksgiving) is in NYSE_HOLIDAYS. Tue 11-24 + 3 trading days
    # would naively land on Fri 11-27 by calendar count alone, but Thu 11-26 is
    # a holiday, so it must land on Mon 11-30 instead (Wed 25, Fri 27, Mon 30).
    check("holiday is correctly skipped (Tue 11-24 + 3td = Mon 11-30, not Fri 11-27)",
          m._add_trading_days(date(2026, 11, 24), 3) == date(2026, 11, 30))


def test_is_trading_day():
    check("a known NYSE_HOLIDAYS date is not a trading day",
          m._is_trading_day(date(2026, 12, 25)) is False)
    check("an ordinary Tuesday is a trading day",
          m._is_trading_day(date(2026, 10, 13)) is True)
    check("a Saturday is not a trading day",
          m._is_trading_day(date(2026, 10, 17)) is False)


def test_open_positions_file_round_trip():
    orig_file = m.OPEN_POSITIONS_FILE
    tmpdir = tempfile.mkdtemp()
    m.OPEN_POSITIONS_FILE = os.path.join(tmpdir, "test_open_positions.json")
    try:
        check("loading a missing file returns {}", m.load_open_positions() == {})
        sample = {
            "778P": {"strike": 778.0, "right": "P", "expiry": "20261017", "qty": 1,
                      "fill_px": 0.86, "pos_id": "ASHLEY_778P_20261014_130000",
                      "opened_at": "2026-10-14T13:00:00-04:00", "trail": True, "peak_bid": 0.9,
                      "dte_kind": "3dte"},
        }
        m.save_open_positions(sample)
        loaded = m.load_open_positions()
        check("round-tripped record matches exactly", loaded == sample, f"got {loaded}")
    finally:
        m.OPEN_POSITIONS_FILE = orig_file


def test_close_position_uses_frozen_expiry_not_today():
    # close_position() must use pos["expiry"], never date.today() -- otherwise
    # a carried 3DTE position closed on a LATER day would try to sell the
    # wrong (today's unrelated 0DTE) contract. We can't call the real async
    # function without IBKR, but we can check the source doesn't regress to
    # the old unconditional date.today() pattern for building `c`.
    import inspect
    src = inspect.getsource(m.close_position)
    check('close_position() reads pos.get("expiry") for its contract',
          'pos.get("expiry")' in src)
    check('close_position() no longer unconditionally uses date.today() for the contract expiry',
          'c = Option("SPY", today_ibkr' not in src)


if __name__ == "__main__":
    test_dte_kind_boundary()
    test_add_trading_days_skips_weekend()
    test_add_trading_days_skips_a_real_holiday()
    test_is_trading_day()
    test_open_positions_file_round_trip()
    test_close_position_uses_frozen_expiry_not_today()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED'}: {FAILURES}")
    import sys
    sys.exit(1 if FAILURES else 0)
