"""Unit tests for the Ashley trailing stop (TRAIL_STOP_PCT).

Run: ../venv/bin/python test_ashley_trail.py

Note (2026-10-14): this file used to also cover TRAIL_STATE_FILE/
save_trail_peaks()/load_trail_peaks() round-tripping. Those were removed as
part of the 3DTE-afternoon feature -- a trailing position's peak_bid now
lives on its own record in OPEN_POSITIONS_FILE (not day-scoped, since a 3DTE
position must keep its trailing stop across days), which is covered in
test_ashley_3dte.py's test_open_positions_file_round_trip() instead. Also,
these were written as pytest tests (bare `assert`, a `tmp_path` fixture) but
pytest was never actually installed in this venv -- `python test_ashley_trail.py`
silently ran nothing and reported nothing, rather than failing loudly. Rewritten
here to the same plain-runner pattern as the other test_ashley_*.py files so a
broken assertion is actually seen.
"""
import ashleyklieu_trigger_executor as ax

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_trail_fires_exactly_at_30_percent_below_peak():
    check("30% below peak fires (at the line)", ax.trail_stop_hit(1.00, 0.70) is True)
    check("below 30% fires (through the line)", ax.trail_stop_hit(1.00, 0.69) is True)
    check("inside 30% does not fire", ax.trail_stop_hit(1.00, 0.71) is False)


def test_trail_ignores_missing_or_zero_bid():
    check("a None bid never fires", ax.trail_stop_hit(1.00, None) is False)
    check("a zero bid never fires", ax.trail_stop_hit(1.00, 0.0) is False)


def test_trail_pct_is_30_percent():
    check("TRAIL_STOP_PCT is 0.30", ax.TRAIL_STOP_PCT == 0.30)


def test_trail_uses_passed_pct():
    check("a looser pct override doesn't fire yet", ax.trail_stop_hit(1.00, 0.85, pct=0.20) is False)
    check("a looser pct override fires once crossed", ax.trail_stop_hit(1.00, 0.79, pct=0.20) is True)


if __name__ == "__main__":
    test_trail_fires_exactly_at_30_percent_below_peak()
    test_trail_ignores_missing_or_zero_bid()
    test_trail_pct_is_30_percent()
    test_trail_uses_passed_pct()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED'}: {FAILURES}")
    import sys
    sys.exit(1 if FAILURES else 0)
