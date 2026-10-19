"""One-time backfill: the two historical SPY 770P positions (2026-10-05 and
2026-10-15) that expired worthless but were left with close_pnl=None in
alpaca_0dte_positions.json, silently excluding a real -$257.00 combined
loss from every downstream pnl stat (/pnl/dashboard's today_pnl,
net_realized_pnl, win_rate, etc). Root cause fixed going forward in
ibkr_position_reconciler.py (worthless_expiry_pnl()); this script applies
the SAME function, once, to the two already-"closed" records the live
reconciler will never revisit (it only scans phase="open" positions).

Run once: ../venv/bin/python backfill_worthless_770p.py
A backup (alpaca_0dte_positions.json.backup_20261016_before_worthless_backfill)
was taken by hand before this ever ran.
"""
import json

from ib_insync import IB

import ibkr_position_reconciler as rec

TARGET_POS_IDS = {
    "ASHLEY_770P_20261005_093110": ("SPY", "20261005", 770.0, "P", 1.22, 1),
    "ASHLEY_770P_20261015_110457": ("SPY", "20261015", 770.0, "P", 1.35, 1),
}


def main():
    with open("alpaca_0dte_positions.json") as f:
        reg = json.load(f)

    ib = IB()
    ib.connect("127.0.0.1", 7496, clientId=2507, timeout=20)
    try:
        changed = 0
        for pos in reg.get("closed", []):
            pos_id = pos.get("pos_id")
            if pos_id not in TARGET_POS_IDS or pos.get("close_pnl") is not None:
                continue
            ticker, expiry, strike, right, entry_fill, qty = TARGET_POS_IDS[pos_id]
            pnl = rec.worthless_expiry_pnl(ib, ticker, expiry, strike, right, entry_fill, qty)
            if pnl is None:
                print(f"SKIP {pos_id}: worthless_expiry_pnl could not confirm OTM -- left as-is.")
                continue
            old_reason = pos.get("close_reason")
            pos["close_pnl"] = pnl
            pos["close_reason"] = "auto_reconciled_worthless_expiry"
            pos["notes"] = (pos.get("notes") or "") + \
                f" [backfilled {pnl} via backfill_worthless_770p.py, 2026-10-16; was close_reason={old_reason}, close_pnl=None]"
            print(f"BACKFILLED {pos_id}: close_pnl None -> {pnl} (was {old_reason})")
            changed += 1
    finally:
        ib.disconnect()

    if changed:
        with open("alpaca_0dte_positions.json", "w") as f:
            json.dump(reg, f, indent=2)
        print(f"\nWrote {changed} backfilled record(s) to alpaca_0dte_positions.json.")
    else:
        print("\nNothing changed.")


if __name__ == "__main__":
    main()
