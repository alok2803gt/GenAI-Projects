"""Unit tests for the two exit-signal gaps found 2026-10-19 while analyzing
Ashley's real trade history (how many times did she send a real exit vs.
go silent -- 4 of the 9 "no automated exit" trades turned out to be real
signals the code failed to act on):

1. Strikethrough on an ALREADY-OPEN position (real incident 2026-09-29,
   765C) must close it, not just skip retiring it (retirement only ever
   applied to not-yet-fired names).
2. The hypothetical-marker gate must not suppress a real, non-hypothetical
   abandonment phrase just because the SAME message also contains "i'd"
   elsewhere (real incident 2026-10-15, 770P: "Not in this trade, I'd take
   profit now").

Critically also re-tests the ORIGINAL case the hypothetical gate was built
to protect (2026-08-30: "I'd take profit already now", pure casual
commentary with NO abandonment phrase) to confirm fix #2 didn't just trade
one bug for its mirror image.

Run: ../venv/bin/python test_ashley_exit_signals.py
"""
import ashleyklieu_trigger_executor as ax

FAILURES = []


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


# ---- Fix #2: abandonment must not be gated by the hypothetical marker ----

def test_real_incident_770p_now_recognized():
    kind = ax.exit_signal_kind("Not in this trade, I'd take profit now")
    check('real 2026-10-15 770P message now returns "abandoned"', kind == "abandoned", f"got {kind!r}")


def test_original_hypothetical_incident_still_suppressed():
    # The 2026-08-30 incident this gate exists for: pure casual commentary
    # about her profit-taking philosophy, no abandonment language at all.
    # Must still return None -- fixing #2 must not break this.
    kind = ax.exit_signal_kind("SPY taking off, but I'd take profit already now")
    check('original 2026-08-30 casual-commentary case is still correctly ignored',
          kind is None, f"got {kind!r}")


def test_plain_abandonment_still_works():
    kind = ax.exit_signal_kind("2nd Play invalid, I wasn't in this trade and this happened too fast")
    check("the original 2026-10-01 762C abandonment case still works",
          kind == "abandoned", f"got {kind!r}")


def test_abandonment_with_other_hypothetical_marker():
    kind = ax.exit_signal_kind("wasn't in this trade, I would have taken profit though")
    check('"I would" variant of the hypothetical marker also no longer suppresses abandonment',
          kind == "abandoned", f"got {kind!r}")


def test_done_phrase_hypothetical_gate_unchanged():
    # Confirm fix #2 was scoped ONLY to ABANDON_PHRASES -- DONE_PHRASES'
    # existing hypothetical gate is untouched (no evidence it's broken).
    kind = ax.exit_signal_kind("I'd say the play is done whenever it hits that level")
    check("DONE_PHRASES hypothetical gate is deliberately left as-is",
          kind is None, f"got {kind!r}")


# ---- Fix #1: strikethrough on an open position must trigger a real exit ----

def test_strikethrough_on_open_position_flags_for_exit():
    retired = {"765C", "768P"}
    open_positions = {"765C": {"fill_px": 1.04, "qty": 1}}  # 765C is open, 768P never fired
    names = ax.retired_names_needing_exit(retired, open_positions)
    check("an open position struck through is flagged for a real close",
          names == ["765C"], f"got {names}")


def test_strikethrough_on_pending_setup_not_flagged():
    retired = {"768P"}  # not yet fired, no open position
    open_positions = {}
    names = ax.retired_names_needing_exit(retired, open_positions)
    check("a struck-through setup that was never open is not flagged for a close",
          names == [], f"got {names}")


def test_strikethrough_real_incident_765c():
    # Real 2026-09-29 message content (abbreviated): she struck through the
    # already-open 765C while posting the next two levels.
    content = "768 P\nEntry 768.2 - 768.5\n\n~~765 C\nEntry: 764.2 - 764.5 ~~\n\n762 C\nEntry 762 - 762.3"
    setups = [{"name": "765C", "strike": 765.0, "right": "C", "lo": 764.2, "hi": 764.5}]
    retired = ax.retired_setups_from(content, setups)
    check("real 2026-09-29 message text correctly parses 765C as struck through",
          retired == {"765C"}, f"got {retired}")
    names = ax.retired_names_needing_exit(retired, {"765C": {"fill_px": 1.04, "qty": 1}})
    check("...and, now open, is correctly flagged for a real close",
          names == ["765C"], f"got {names}")


if __name__ == "__main__":
    test_real_incident_770p_now_recognized()
    test_original_hypothetical_incident_still_suppressed()
    test_plain_abandonment_still_works()
    test_abandonment_with_other_hypothetical_marker()
    test_done_phrase_hypothetical_gate_unchanged()
    test_strikethrough_on_open_position_flags_for_exit()
    test_strikethrough_on_pending_setup_not_flagged()
    test_strikethrough_real_incident_765c()
    print(f"\n{'ALL PASSED' if not FAILURES else f'{len(FAILURES)} FAILED'}: {FAILURES}")
    import sys
    sys.exit(1 if FAILURES else 0)
