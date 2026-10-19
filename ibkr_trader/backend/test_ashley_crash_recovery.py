"""Unit tests for _update_crash_state() -- the 2026-10-13 fix for the
2026-10-12 incident (a mid-day crash permanently idled Ashley for the rest
of the day; see the function's own docstring in ashleyklieu_trigger_executor.py).

Run: ../venv/bin/python test_ashley_crash_recovery.py
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from ashleyklieu_trigger_executor import _update_crash_state, CRASH_RETRY_LIMIT

ET = ZoneInfo("America/New_York")

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def t(hour, minute):
    return datetime(2026, 10, 13, hour, minute, tzinfo=ET)


def test_first_crash_midday_retries():
    state = {}
    give_up = _update_crash_state(state, "2026-10-13", t(9, 23))
    check("a single pre-open crash (9:23, the real 2026-10-12 timing) does NOT give up",
          give_up is False, f"give_up={give_up}")
    check("crash_count is now 1", state["crash_count"] == 1, str(state))


def test_real_incident_exact_reproduction():
    # This is the literal case that got silently dropped on 2026-10-12: a
    # crash during the TWS relogin race, then a human-driven restart at
    # 09:58/09:59 should be able to resume rather than idle all day.
    state = {}
    give_up = _update_crash_state(state, "2026-10-12", t(9, 40))
    check("the real 9:40 crash does not give up (this is exactly what used to fail)",
          give_up is False)


def test_crash_after_close_gives_up():
    state = {}
    give_up = _update_crash_state(state, "2026-10-13", t(16, 5))
    check("a crash after the 16:00 close gives up immediately", give_up is True)


def test_crash_at_exactly_close_gives_up():
    state = {}
    give_up = _update_crash_state(state, "2026-10-13", t(16, 0))
    check("a crash at exactly 16:00 gives up (boundary is inclusive)", give_up is True)


def test_budget_exhausted_gives_up_even_midday():
    state = {"crash_date": "2026-10-13", "crash_count": CRASH_RETRY_LIMIT - 1}
    give_up = _update_crash_state(state, "2026-10-13", t(11, 0))
    check(f"the {CRASH_RETRY_LIMIT}th same-day crash gives up even mid-day (no infinite crash loop)",
          give_up is True, f"crash_count={state['crash_count']}")


def test_budget_resets_on_a_new_day():
    state = {"crash_date": "2026-10-12", "crash_count": CRASH_RETRY_LIMIT}
    give_up = _update_crash_state(state, "2026-10-13", t(9, 5))
    check("a new day's first crash retries even if yesterday exhausted its budget",
          give_up is False, f"state={state}")
    check("crash_count reset to 1 for the new day", state["crash_count"] == 1, str(state))


def test_repeated_crashes_same_day_count_up_then_give_up():
    state = {}
    results = []
    for _ in range(CRASH_RETRY_LIMIT + 1):
        results.append(_update_crash_state(state, "2026-10-13", t(10, 0)))
    check(f"first {CRASH_RETRY_LIMIT - 1} crashes retry, exactly the {CRASH_RETRY_LIMIT}th+ give up",
          results == [False] * (CRASH_RETRY_LIMIT - 1) + [True] * (len(results) - (CRASH_RETRY_LIMIT - 1)),
          str(results))


if __name__ == "__main__":
    test_first_crash_midday_retries()
    test_real_incident_exact_reproduction()
    test_crash_after_close_gives_up()
    test_crash_at_exactly_close_gives_up()
    test_budget_exhausted_gives_up_even_midday()
    test_budget_resets_on_a_new_day()
    test_repeated_crashes_same_day_count_up_then_give_up()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED'}: {FAILURES}")
    import sys
    sys.exit(1 if FAILURES else 0)
