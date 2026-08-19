"""
Daily GEX/VEX archiver -- runs gex-vex-calculator for our 0DTE-relevant
tickers and appends the resulting snapshot to a durable historical log.

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

Usage: python archive_gex_daily.py [--tickers SPY,QQQ]
Appends one line per run to gex_vex_history.jsonl (gitignored, runtime data).
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta

ET = timezone(timedelta(hours=-4))
CALC_SCRIPT = r"C:\Users\AlokD\.claude\skills\gex-vex-calculator\calc_gex_vex.py"
CACHE_FILE = r"C:\Users\AlokD\.claude\skills\gex-vex-calculator\gex_vex_cache.json"
HISTORY_FILE = "gex_vex_history.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="SPY,QQQ")
    args = ap.parse_args()

    print(f"Running gex-vex-calculator for {args.tickers}...")
    result = subprocess.run(
        [sys.executable, CALC_SCRIPT, "--tickers", args.tickers],
        capture_output=True, text=True, timeout=600,
    )
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
