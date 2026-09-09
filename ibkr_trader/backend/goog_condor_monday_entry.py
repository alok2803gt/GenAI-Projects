"""
Weekly wrapper for alpaca_goog_weekly_condor.py, for unattended Monday
scheduling (CEO decision 2026-08-24: automate the Monday-cycle entry
going forward).

alpaca_goog_weekly_condor.py deliberately requires an explicit --expiry
argument and does NOT auto-compute "next Friday" itself -- that was a
real, deliberate safety choice (its own docstring: running off-cycle
"deviates from what was actually tested"). The 2026-08-19 entry proved
why: it was placed on a Wednesday, off the validated Monday-only cycle,
and needed cleanup 2026-08-24. Automating this does NOT mean removing
that safety property -- it means this wrapper computes the correct
expiry deterministically (today + 4 days, since a Monday + 4 = Friday)
and passes it explicitly, plus refuses to run at all on any day but
Monday. A human fat-fingering a date was the old risk; this removes it
rather than reintroducing it under automation.

Usage: python goog_condor_monday_entry.py [--dry-run]
"""
import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
PYTHON = r"C:\Users\AlokD\AppData\Local\Programs\Python\Python311\python.exe"
SCRIPT = BACKEND_DIR / "alpaca_goog_weekly_condor.py"
CLIENT_ID = 1680


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = date.today()
    if today.weekday() != 0:  # Monday=0
        print(f"Today ({today}, {today.strftime('%A')}) is not Monday -- "
              f"this strategy only enters on the validated Monday cycle. Skipping.")
        return

    expiry = today + timedelta(days=4)  # Monday + 4 = Friday
    print(f"Monday entry: computing expiry {expiry.isoformat()} (this week's Friday).")

    cmd = [PYTHON, str(SCRIPT), "--expiry", expiry.isoformat(), "--client-id", str(CLIENT_ID)]
    if args.dry_run:
        cmd.append("--dry-run")
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=BACKEND_DIR)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
