"""Unit tests for ibkr_position_reconciler.worthless_expiry_pnl() -- the fix
for two real incidents (SPY 770P, 2026-10-05 and 2026-10-15) where a
position that expired worthless left close_pnl=None forever, silently
excluding a real, confidently-knowable loss from every downstream pnl stat.

Run: ../venv/bin/python test_worthless_expiry_pnl.py
"""
from __future__ import annotations

from types import SimpleNamespace

import ibkr_position_reconciler as rec

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


class FakeIB:
    """Stubs just the two ib_insync calls worthless_expiry_pnl() makes."""
    def __init__(self, bars):
        self._bars = bars

    def qualifyContracts(self, *args):
        pass

    def reqHistoricalData(self, *args, **kwargs):
        return self._bars


def bar(date_str, close):
    return SimpleNamespace(date=date_str, close=close)


def test_otm_put_confirmed_worthless():
    # Real 2026-10-15 case: SPY closed 771.74, strike 770 put -> OTM.
    ib = FakeIB([bar("20261015", 771.74)])
    pnl = rec.worthless_expiry_pnl(ib, "SPY", "20261015", 770.0, "P", 1.35, 1)
    check("OTM put (2026-10-15, SPY 770P @ 1.35) -> confident -$135.00 loss",
          pnl == -135.0, f"got {pnl}")


def test_otm_put_confirmed_worthless_second_incident():
    # Real 2026-10-05 case: SPY closed 774.83, strike 770 put -> OTM.
    ib = FakeIB([bar("20261005", 774.83)])
    pnl = rec.worthless_expiry_pnl(ib, "SPY", "20261005", 770.0, "P", 1.22, 1)
    check("OTM put (2026-10-05, SPY 770P @ 1.22) -> confident -$122.00 loss",
          pnl == -122.0, f"got {pnl}")


def test_otm_call_confirmed_worthless():
    ib = FakeIB([bar("20261010", 765.00)])
    pnl = rec.worthless_expiry_pnl(ib, "SPY", "20261010", 770.0, "C", 0.50, 2)
    check("OTM call (close 765 < strike 770) -> confident loss, qty=2 scales it",
          pnl == -100.0, f"got {pnl}")


def test_itm_put_never_guessed():
    # Close is BELOW the put strike -- it finished ITM, real exercise
    # economics, must stay None rather than guess.
    ib = FakeIB([bar("20261015", 765.00)])
    pnl = rec.worthless_expiry_pnl(ib, "SPY", "20261015", 770.0, "P", 1.35, 1)
    check("ITM put is never guessed at -- stays None", pnl is None, f"got {pnl}")


def test_itm_call_never_guessed():
    ib = FakeIB([bar("20261015", 780.00)])
    pnl = rec.worthless_expiry_pnl(ib, "SPY", "20261015", 770.0, "C", 1.35, 1)
    check("ITM call is never guessed at -- stays None", pnl is None, f"got {pnl}")


def test_future_expiry_not_touched():
    ib = FakeIB([bar("20991231", 771.74)])
    pnl = rec.worthless_expiry_pnl(ib, "SPY", "20991231", 770.0, "P", 1.35, 1)
    check("an expiry that hasn't happened yet is never touched", pnl is None, f"got {pnl}")


def test_no_matching_bar_returns_none():
    ib = FakeIB([bar("20261014", 771.74)])  # wrong date in the returned bars
    pnl = rec.worthless_expiry_pnl(ib, "SPY", "20261015", 770.0, "P", 1.35, 1)
    check("no bar matching the exact expiry date -> None, not a wrong guess",
          pnl is None, f"got {pnl}")


def test_fetch_exception_returns_none():
    class ExplodingIB(FakeIB):
        def reqHistoricalData(self, *a, **k):
            raise RuntimeError("no market data")
    pnl = rec.worthless_expiry_pnl(ExplodingIB([]), "SPY", "20261015", 770.0, "P", 1.35, 1)
    check("a data-fetch failure returns None, doesn't crash the reconciler", pnl is None, f"got {pnl}")


if __name__ == "__main__":
    test_otm_put_confirmed_worthless()
    test_otm_put_confirmed_worthless_second_incident()
    test_otm_call_confirmed_worthless()
    test_itm_put_never_guessed()
    test_itm_call_never_guessed()
    test_future_expiry_not_touched()
    test_no_matching_bar_returns_none()
    test_fetch_exception_returns_none()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED'}: {FAILURES}")
    import sys
    sys.exit(1 if FAILURES else 0)
