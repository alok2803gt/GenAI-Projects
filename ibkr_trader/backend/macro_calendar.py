"""
Real, auto-refreshing macro-event calendar: FOMC rate decisions, NFP
(Employment Situation), CPI, and PPI release dates. Fetches live from the
Federal Reserve's own site and BLS's own schedule pages, with hardcoded/
calculated fallbacks if a fetch fails. Cached once per real calendar day.

Extracted 2026-09-04 from main.py's SPX 0DTE strategy (where this already
existed and was already live, gating SPX 0DTE's entry on all 4 event
types) into its own module so the SAME real calendar can gate/caution
OTHER live strategies (SPY/QQQ/IWM 0DTE butterflies, Day Trader, Ashley
signal-follow) without each reimplementing or duplicating the fetch logic.
main.py now imports from here instead of defining these inline -- no
behavior change for SPX 0DTE, just a shared source of truth.

Usage for a new caller:
    from macro_calendar import is_macro_day
    skip, reason = is_macro_day()   # today, by default
    if skip:
        ...  # block or caution, depending on the caller's own risk posture
"""
from datetime import date

_FOMC_DATES_FALLBACK: dict[int, list[str]] = {
    2025: [
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    ],
    2026: [
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    ],
}

# Real BLS 2026 CPI/PPI release dates (source: fedratecalc.com/us-economic-
# calendar, cross-checked against BLS's own 8:30am ET release convention,
# verified 2026-09-04) -- added as a fallback because bls.gov's own CPI/PPI
# schedule pages started returning 403 Forbidden to this scraper around the
# same time (bot-blocking, not a parsing bug) with NO existing fallback,
# meaning CPI/PPI protection was silently dead until this was added. Only
# near-term months are populated; extend as later months are published
# (BLS publishes ~6-12 months ahead) -- same maintenance pattern as
# _FOMC_DATES_FALLBACK above.
_CPI_DATES_FALLBACK: dict[int, list[str]] = {
    2026: ["2026-09-11", "2026-10-14", "2026-11-10"],
}
_PPI_DATES_FALLBACK: dict[int, list[str]] = {
    2026: ["2026-09-10", "2026-10-15", "2026-11-13"],
}

_macro_cache: dict = {
    "dates":          {},      # populated by _refresh_macro_calendar()
    "last_refreshed": None,    # date object
    "fomc_source":    None,    # "online (federalreserve.gov)" | "fallback (hardcoded)"
    "nfp_source":     None,    # "online (bls.gov)" | "calculated (first Friday)"
    "cpi_source":     None,    # "online (bls.gov)" | "none"
    "ppi_source":     None,    # "online (bls.gov)" | "none"
}


def _nfp_dates(year: int) -> list[str]:
    """First Friday of each month = NFP release day (calculated fallback).
    Holiday shifts (e.g. Jan 1 on Friday) push NFP to the following Friday --
    the online BLS fetch handles those edge cases correctly.
    """
    result = []
    for month in range(1, 13):
        for day in range(1, 8):
            d = date(year, month, day)
            if d.weekday() == 4:   # Friday
                result.append(d.isoformat())
                break
    return result


def _fetch_fomc_dates_online(year: int) -> list[str] | None:
    """Scrape FOMC announcement dates from federalreserve.gov for the given year.
    Returns list of ISO date strings, or None if fetch/parse fails.
    """
    import re
    import logging
    log = logging.getLogger(__name__)
    try:
        import requests as _req
        r = _req.get(
            "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; IBKRTrader/1.0)"},
        )
        r.raise_for_status()
        html = r.text

        _MONTH_NUM = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12,
        }

        # Isolate the section for the target year.
        # The page uses <h4>YYYY</h4> or similar markers.
        year_match = re.search(
            rf'(?s)(?:{year})(.+?)(?:{year + 1}|$)', html
        )
        if not year_match:
            log.warning("FOMC online: year %d section not found in page", year)
            return None
        section = year_match.group(1)

        dates: list[str] = []
        # Match patterns like "January 28-29" or "January 29" (single day)
        # The page sometimes includes asterisks for unscheduled / tentative
        for m in re.finditer(
            r'(January|February|March|April|May|June|July|August'
            r'|September|October|November|December)[^0-9]{0,10}(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?',
            section,
        ):
            month_name = m.group(1)
            start_day  = int(m.group(2))
            end_day    = int(m.group(3)) if m.group(3) else start_day
            try:
                ann_date = date(year, _MONTH_NUM[month_name], end_day)
                dates.append(ann_date.isoformat())
            except ValueError:
                pass   # bad day number — skip

        if not dates:
            log.warning("FOMC online: parsed 0 dates for %d — HTML structure may have changed", year)
            return None

        log.info("FOMC online: fetched %d announcement dates for %d from federalreserve.gov",
                 len(dates), year)
        return sorted(set(dates))

    except Exception as exc:
        log.warning("FOMC online fetch failed: %s", exc)
        return None


def _fetch_bls_dates_online(year: int, url: str, label: str) -> list[str] | None:
    """Generic BLS schedule page scraper (CPI, PPI, NFP/empsit, etc.).
    All BLS release schedule pages use the same date format:
      "Wednesday, January 15, 2025" or "Thursday, January 16, 2025"
    """
    import re
    import logging
    log = logging.getLogger(__name__)
    _MONTH_NUM = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    try:
        import requests as _req
        r = _req.get(url, timeout=10,
                     headers={"User-Agent": "Mozilla/5.0 (compatible; IBKRTrader/1.0)"})
        r.raise_for_status()
        dates: list[str] = []
        for m in re.finditer(
            r'(?:Monday|Tuesday|Wednesday|Thursday|Friday),\s+'
            r'(January|February|March|April|May|June|July|August'
            r'|September|October|November|December)\s+(\d{1,2}),\s+(\d{4})',
            r.text,
        ):
            mon, day, yr = m.group(1), int(m.group(2)), int(m.group(3))
            if yr == year and mon in _MONTH_NUM:
                try:
                    dates.append(date(yr, _MONTH_NUM[mon], day).isoformat())
                except ValueError:
                    pass
        if not dates:
            log.warning("%s online: 0 dates for %d — BLS page structure may have changed", label, year)
            return None
        log.info("%s online: fetched %d dates for %d", label, len(dates), year)
        return sorted(set(dates))
    except Exception as exc:
        log.warning("%s online fetch failed: %s", label, exc)
        return None


def _refresh_macro_calendar() -> None:
    """Fetch FOMC, NFP, CPI, and PPI dates online; merge with fallbacks.
    Runs at most once per calendar day (cached). Safe to call frequently.
    """
    import logging
    log = logging.getLogger(__name__)
    today = date.today()
    if _macro_cache["last_refreshed"] == today:
        return   # already fresh

    year   = today.year
    merged: dict[str, str] = {}

    # ── FOMC ──────────────────────────────────────────────────────────────────
    fomc_online = _fetch_fomc_dates_online(year)
    if fomc_online:
        fomc_dates                  = fomc_online
        _macro_cache["fomc_source"] = "online (federalreserve.gov)"
    else:
        fomc_dates                  = _FOMC_DATES_FALLBACK.get(year, [])
        _macro_cache["fomc_source"] = "fallback (hardcoded)"
        if not fomc_dates:
            log.warning("macro calendar: no FOMC fallback dates for %d — update _FOMC_DATES_FALLBACK", year)
    for d in fomc_dates:
        merged[d] = "FOMC rate decision"

    # ── NFP ───────────────────────────────────────────────────────────────────
    nfp_online = _fetch_bls_dates_online(
        year, "https://www.bls.gov/schedule/news_release/empsit.htm", "NFP")
    if nfp_online:
        nfp_dates                  = nfp_online
        _macro_cache["nfp_source"] = "online (bls.gov)"
    else:
        nfp_dates                  = _nfp_dates(year)
        _macro_cache["nfp_source"] = "calculated (first Friday)"
    for d in nfp_dates:
        merged.setdefault(d, "NFP release")

    # ── CPI ───────────────────────────────────────────────────────────────────
    cpi_dates = _fetch_bls_dates_online(
        year, "https://www.bls.gov/schedule/news_release/cpi.htm", "CPI")
    if cpi_dates:
        _macro_cache["cpi_source"] = "online (bls.gov)"
    else:
        cpi_dates = _CPI_DATES_FALLBACK.get(year, [])
        _macro_cache["cpi_source"] = "fallback (hardcoded)" if cpi_dates else "none (no fallback for this year)"
    for d in cpi_dates:
        merged.setdefault(d, "CPI inflation report")

    # ── PPI ───────────────────────────────────────────────────────────────────
    ppi_dates = _fetch_bls_dates_online(
        year, "https://www.bls.gov/schedule/news_release/ppi.htm", "PPI")
    if ppi_dates:
        _macro_cache["ppi_source"] = "online (bls.gov)"
    else:
        ppi_dates = _PPI_DATES_FALLBACK.get(year, [])
        _macro_cache["ppi_source"] = "fallback (hardcoded)" if ppi_dates else "none (no fallback for this year)"
    for d in ppi_dates:
        merged.setdefault(d, "PPI inflation report")

    _macro_cache["dates"]          = merged
    _macro_cache["last_refreshed"] = today
    log.info(
        "Macro calendar refreshed: %d skip dates for %d  "
        "(FOMC=%s, NFP=%s, CPI=%s, PPI=%s)",
        len(merged), year,
        _macro_cache["fomc_source"], _macro_cache["nfp_source"],
        _macro_cache["cpi_source"],  _macro_cache["ppi_source"],
    )


def _spx_macro_skip_dates(year: int | None = None) -> dict[str, str]:
    """Return {date_iso: reason} for all FOMC + NFP (+ CPI/PPI when the
    online fetch succeeds) days. Triggers a daily online refresh on first
    call of the day. For years other than current, returns hardcoded +
    calculated data (no CPI/PPI fallback -- those have no reliable
    calculated substitute, unlike NFP's "first Friday" rule).

    Name kept as-is (originally SPX-0DTE-specific) since main.py's SPX
    0DTE code already calls it by this name in several places -- this is
    now the shared calendar every gated strategy reads from, not just SPX.
    """
    _refresh_macro_calendar()
    if year is None or year == date.today().year:
        return _macro_cache["dates"]
    # Non-current year: use fallback + calculated
    result: dict[str, str] = {}
    for d in _FOMC_DATES_FALLBACK.get(year, []):
        result[d] = "FOMC rate decision"
    for d in _nfp_dates(year):
        result.setdefault(d, "NFP release")
    return result


def is_macro_day(d: date | None = None) -> tuple[bool, str]:
    """Convenience wrapper for callers that just want a yes/no + reason for
    ONE date (default: today) -- e.g. an entry gate in a standalone script
    that doesn't need the full {date: reason} calendar dict. Returns
    (True, reason) if d is a real FOMC/NFP/CPI/PPI day, else (False, "").
    """
    d = d or date.today()
    days = _spx_macro_skip_dates(d.year)
    reason = days.get(d.isoformat())
    return (reason is not None, reason or "")
