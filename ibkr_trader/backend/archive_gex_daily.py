"""
Daily GEX/VEX archiver -- runs gex-vex-calculator and appends the
resulting snapshot to a durable historical log.

Why this exists: gex_vex_cache.json (the calculator's own output) is a
single current-day snapshot that gets overwritten daily -- there is no
history. True historical dealer GEX isn't purchasable anywhere accessible
to this account (confirmed 2026-08-13 -- Massive/Polygon's "daily open
interest" is current-state only, not a historical series; CBOE DataShop
has no public self-serve pricing). The only honest path to real historical
GEX is to start archiving today's real live snapshots going forward, so
in a few months there's a genuine (if slow-built) historical record to
validate a GEX-based regime filter against -- see oversight_log.jsonl
2026-08-13 for the full context on why this is the fallback.

Caveat inherited from the calculator itself: 25-45 DTE near-term expiry,
not 0DTE-specific -- this is a general dealer-positioning regime signal,
not a true 0DTE gamma snapshot. Still the best available real signal.

Scope widened 2026-09-16 (CEO decision) from SPY/QQQ-only to the full
~112-ticker curated universe (calc_gex_vex.py's own UNIVERSE, same list
as safe-income-screener's) -- found that Safe Income Trader screens that
full universe by default and was only getting gamma-wall context for
SPY/QQQ, "n/a" for everything else.

Real cost of the wider scope, measured directly (not estimated): 15
tickers took 16m35s live option-chain lookups (~66s/ticker) -- a full
112-ticker run is ~2hrs, not the 20-40min originally assumed. calc_gex_vex.py
was fixed the same day to write its cache incrementally after every
ticker (was write-once-at-the-end, meaning any timeout/crash/interruption
across that 2hrs used to lose 100% of the run). timeout below is sized
generously above the real ~2hr figure; if it's ever hit anyway, real
partial progress already made it into gex_vex_cache.json regardless.

Usage: python archive_gex_daily.py [--tickers SPY,QQQ]
No --tickers = full universe (the new default). Appends one line per run
to gex_vex_history.jsonl (gitignored, runtime data).
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ET = timezone(timedelta(hours=-4))
_SKILL_DIR = Path.home() / ".claude" / "skills" / "gex-vex-calculator"
CALC_SCRIPT = str(_SKILL_DIR / "calc_gex_vex.py")
CACHE_FILE = str(_SKILL_DIR / "gex_vex_cache.json")
HISTORY_FILE = "gex_vex_history.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None, help="comma-separated subset; default = full universe")
    args = ap.parse_args()

    cmd = [sys.executable, CALC_SCRIPT]
    if args.tickers:
        cmd += ["--tickers", args.tickers]
    print(f"Running gex-vex-calculator for {args.tickers or 'the full universe'}...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10800,  # 3hrs -- real full-universe cost is ~2hrs
        )
    except subprocess.TimeoutExpired as exc:
        # Not fatal to the archive itself anymore: calc_gex_vex.py now saves
        # its cache incrementally after every ticker (fixed 2026-09-16), so
        # whatever completed before the timeout already made it into
        # gex_vex_cache.json. Fall through and archive that real partial
        # result instead of losing the run entirely.
        print(f"calc_gex_vex.py exceeded the {exc.timeout:.0f}s timeout -- continuing with "
              f"whatever it saved before being cut off (incremental writes, not lost).")
    else:
        if result.returncode != 0:
            print("ERROR running calc_gex_vex.py:")
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            sys.exit(1)
        print(result.stdout[-1500:])

    with open(CACHE_FILE) as f:
        cache = json.load(f)

    entry = {
        "archived_at": datetime.now(ET).isoformat(),
        "computed_at": cache.get("computed_at"),
        "computed_at_date": cache.get("computed_at_date"),
        "band": cache.get("band"),
        "tickers": {
            t: {k: v for k, v in data.items() if k != "strikes"}
            for t, data in cache.get("tickers", {}).items()
        },
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Archived {len(entry['tickers'])} tickers to {HISTORY_FILE} "
          f"(as of {entry['computed_at']})")


if __name__ == "__main__":
    main()
