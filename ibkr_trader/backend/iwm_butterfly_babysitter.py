"""
Standing babysitter for the daily IWM 0DTE butterfly (IBKR-IWMButterflyWindow,
fires every weekday ~9:35 ET) -- built 2026-09-04 as the second instance of
the same babysitter pattern already deployed for QQQ, after the CEO flagged
IWM chopping sideways intraday and wanted the same GEX/dark-pool/momentum
early-exit insight coverage plus the same end-of-day "did it actually close"
safety net.

Shares all real logic with qqq_butterfly_babysitter.py via
butterfly_babysitter_common.py -- see that module's docstring for the full
per-run behavior (pre-verify insight checks -> verify all 3 legs flat ->
emergency close if not) and rationale (registry's close-fallback can mark
phase="closed" even on a partial close; "cut a loss early" stays a CEO
judgment call, never an automated action).

Building this confirmed IWM needed its OWN handling, not a copy-paste of
QQQ's: IWM's real listed strikes near the money are NOT uniformly spaced
($0.50 half-strikes mixed into a mostly-$1 grid, e.g. 291/292/292.5/293/294),
and today's real IWM butterfly (291/294/297.5) has ASYMMETRIC wings (3.0 vs
3.5 wide) -- both handled correctly by the shared module's real-chain-strike
lookup and wing_width_lo-based max-profit math, but would have been silently
wrong under QQQ's original hardcoded-$1-increment, symmetric-wing logic.
"""
import sys

sys.path.insert(0, ".")
from butterfly_babysitter_common import run_babysitter

TWS_PORT = 7496
CLIENT_ID = 1791         # emergency-close connection -- distinct range from QQQ's
                           # 1691/1692 so both babysitters can run concurrently
INSIGHT_CLIENT_ID = 1792  # intraday insight connection

if __name__ == "__main__":
    run_babysitter("IWM", TWS_PORT, CLIENT_ID, INSIGHT_CLIENT_ID)
