#!/usr/bin/env python3
"""Scheduled-task entry point: only starts condor_v2.py if TWS is confirmed on a
paper account. Never starts it against a live account. Logs its decision either way.

    python safe_start.py            # SPY instance (this folder)
    python safe_start.py qqq        # instance folder ./qqq (own config/state/log)
"""
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ib_insync import IB

HERE = Path(__file__).resolve().parent
BASE = (HERE / sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE
LOG = BASE / "safe_start_log.jsonl"


def log(msg):
    line = f"{datetime.now().isoformat()} {msg}"
    print(line)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    now = datetime.now()
    if now.weekday() >= 5:
        log("weekend -- not starting")
        return

    ib = IB()
    try:
        ib.connect("127.0.0.1", 7497, clientId=random.randint(9000, 9999), timeout=15)
    except Exception as e:
        log(f"ABORT: could not reach TWS on 127.0.0.1:7497 ({type(e).__name__}: {e})")
        return

    accounts = ib.managedAccounts()
    ib.disconnect()

    if not accounts or not accounts[0].startswith("DU"):
        log(f"ABORT: account {accounts} is not a paper account (must start with 'DU') -- "
            f"refusing to start condor_v2 to avoid trading live by accident")
        return

    log(f"OK: paper account {accounts[0]} confirmed -- launching condor_v2.py --dir {BASE}")
    subprocess.Popen(
        [str(HERE / ".venv" / "Scripts" / "python.exe"), "-u", str(HERE / "condor_v2.py"), "--dir", str(BASE)],
        cwd=str(HERE),
        stdout=open(BASE / "condor_v2_console.log", "a"),
        stderr=subprocess.STDOUT,
    )


if __name__ == "__main__":
    main()
