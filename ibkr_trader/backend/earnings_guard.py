"""Is a name about to report earnings? Shared gate for the IDR pipeline.

WHY (2026-09-30)
----------------
The IDR pipeline had ZERO earnings awareness -- no reference to earnings in
harami_daily_trader.py, either scanner, or accumulation.py -- while the breakout
and daytrader scanners both carry blackouts. That is backwards: IDR is the one
strategy that HOLDS for days, so an earnings print inside its window is an
unhedged overnight bet on a binary event.

The concrete exposure, found on the day it mattered: MU reports after the close
on 2026-09-30 with a 6.45% expected move, MU is in the IDR panel, and the IDR
scanner runs at 16:12 with entry at 16:20 placing MARKET-ON-OPEN for the NEXT
open. A signal on MU tonight would therefore have been filled straight into the
post-earnings gap, on a 1-share position with no stop.

Data: Unusual Whales /api/earnings/{ticker}, which carries report_date,
report_time (premarket / postmarket) and the options-implied expected move. An
API with a key does not break the way the scraped BLS and Federal Reserve pages
behind macro_calendar did -- all of which had been failing silently for days.

FAIL-SAFE DIRECTION MATTERS. If the check cannot run, this returns
"blocked, reason=unknown" for the pre-fill window rather than "clear". An
unavailable data source must not silently turn into permission to buy into
earnings. That is the opposite of the choice accumulation.screen() makes -- there
a failure defaults to the TECHNICAL track, because the risk there is an
unintended long-term hold, and the conservative direction is different.

    ../venv/bin/python earnings_guard.py AAPL MU RTX TFC
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ET = ZoneInfo("America/New_York")
CACHE = HERE / "earnings_cache.json"
CACHE_HOURS = 12          # report dates move rarely; 12h keeps it fresh enough


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _save_cache(c: dict) -> None:
    try:
        CACHE.write_text(json.dumps(c, indent=1, default=str))
    except Exception:
        pass


def next_report(ticker: str, force: bool = False) -> dict | None:
    """{date, time, expected_move_perc} for the next report, or None.

    None means "no upcoming report found" -- distinguish that from a FAILURE,
    which raises. Callers must treat the two differently.
    """
    cache = _load_cache()
    hit = cache.get(ticker)
    if hit and not force:
        try:
            age = datetime.now(ET) - datetime.fromisoformat(hit["cached_at"])
            if age < timedelta(hours=CACHE_HOURS):
                return hit.get("next")
        except Exception:
            pass
    import unusual_whales_client as uw
    c = uw.UnusualWhalesClient()
    rows = c._get(f"/api/earnings/{ticker}", {})
    rows = rows.get("data", rows) if isinstance(rows, dict) else rows
    today = date.today()
    best = None
    for x in rows or []:
        rd = str(x.get("report_date") or "")[:10]
        if len(rd) != 10:
            continue
        try:
            d = date.fromisoformat(rd)
        except Exception:
            continue
        if d < today:
            continue
        # earliest report on/after today
        if best is None or d < date.fromisoformat(best["date"]):
            em = x.get("expected_move_perc")
            try:
                em = float(em) * 100 if em is not None else None
            except Exception:
                em = None
            best = {"date": rd, "time": str(x.get("report_time") or "").lower(),
                    "expected_move_pct": em}
    cache[ticker] = {"cached_at": datetime.now(ET).isoformat(), "next": best}
    _save_cache(cache)
    return best


def reports_before_next_open(ticker: str) -> tuple[bool, str]:
    """Will this name report BEFORE a market-on-open order placed now fills?

    That is the acute case for the IDR entry path, which submits MOO after the
    close for the next open: anything reporting tonight postmarket, or tomorrow
    premarket, lands before the fill.

    Returns (blocked, reason). On any failure -> (True, "...unavailable...").
    """
    try:
        nxt = next_report(ticker)
    except Exception as exc:
        return True, (f"earnings check unavailable ({type(exc).__name__}) -- "
                      f"blocking rather than buying blind into a possible report")
    if not nxt:
        return False, ""
    today = date.today()
    try:
        d = date.fromisoformat(nxt["date"])
    except Exception:
        return True, "earnings date unparseable -- blocking"
    em = (f", {nxt['expected_move_pct']:.1f}% expected move"
          if nxt.get("expected_move_pct") else "")
    # reports tonight after the close
    if d == today and "post" in nxt["time"]:
        return True, f"reports TONIGHT postmarket ({nxt['date']}{em})"
    # reports tomorrow before the open
    nxt_day = today + timedelta(days=1)
    while nxt_day.weekday() >= 5:
        nxt_day += timedelta(days=1)
    if d == nxt_day and "pre" in nxt["time"]:
        return True, f"reports {nxt['date']} PREMARKET{em} -- before a MOO fill"
    if d == today and "pre" in nxt["time"]:
        return False, ""            # already reported this morning
    return False, ""


def reports_within(ticker: str, days: int) -> tuple[bool, str]:
    """Does the next report fall within `days` calendar days? For hold-window
    checks rather than the fill-window check above."""
    try:
        nxt = next_report(ticker)
    except Exception as exc:
        return True, f"earnings check unavailable ({type(exc).__name__}) -- blocking"
    if not nxt:
        return False, ""
    try:
        d = date.fromisoformat(nxt["date"])
    except Exception:
        return True, "earnings date unparseable -- blocking"
    delta = (d - date.today()).days
    if 0 <= delta <= days:
        em = (f", {nxt['expected_move_pct']:.1f}% expected move"
              if nxt.get("expected_move_pct") else "")
        return True, f"reports in {delta}d on {nxt['date']} ({nxt['time']}{em})"
    return False, ""


if __name__ == "__main__":
    import sys
    for t in (sys.argv[1:] or ["MU", "RTX", "TFC", "AAPL"]):
        try:
            n = next_report(t, force=True)
        except Exception as exc:
            print(f"  {t:<6} FAILED {type(exc).__name__}: {exc}")
            continue
        b1, r1 = reports_before_next_open(t)
        b2, r2 = reports_within(t, 10)
        print(f"  {t:<6} next={n}")
        print(f"         before-next-open: {'BLOCK' if b1 else 'clear'}  {r1}")
        print(f"         within 10 days  : {'BLOCK' if b2 else 'clear'}  {r2}")
