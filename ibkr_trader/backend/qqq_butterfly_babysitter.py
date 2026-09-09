"""
Standing babysitter for the daily QQQ 0DTE butterfly (IBKR-QQQButterflyWindow,
fires every weekday ~9:35 ET) -- built 2026-09-04 directly from a real,
confirmed gap: alpaca_0dte_butterfly_trader.py's own close-fallback path
(close_position() in alpaca_0dte_common.py) records phase="closed" in the
registry UNCONDITIONALLY, even when the actual close did not fully confirm
(ok=False) -- and its own "close did not fully confirm" warning only prints
to stdout, never reaching Telegram or oversight_log. This is exactly what
happened to QQQ's 2026-09-03 butterfly: only 2 of 3 legs actually closed via
Alpaca's real MLEG auto-close; the third leg (long 707C) plus the remaining
short 711C sat open for hours, completely unflagged, until manually found
and reconciled. Not a one-off -- QQQ fires fresh every weekday, so this is a
standing risk, not a single incident to patch and forget.

Runs every ~10min around the clock (OS-level trigger, same "fire often,
self-gate on time/day" pattern as IBKR-PortfolioOversight/CRORiskCheck) --
does real work only inside market hours, real verification work only from
VERIFY_START onward.

What it actually does each real run:
  1. Find today's real QQQ butterfly entry in alpaca_0dte_positions.json --
     checks BOTH the "positions" (still-open) and "closed" lists, since a
     "closed" entry is exactly what's NOT trustworthy without re-checking
     real Alpaca state (2026-09-03's exact failure mode).
  2. Before VERIFY_START: lightweight, informational-only mark-to-market
     log (no alerting -- avoids Telegram spam during a normal trading day),
     PLUS a real intraday GEX/dark-pool/momentum INSIGHT check at most
     every 30min while the position is open -- see
     butterfly_babysitter_common.intraday_insight_check() for the full
     methodology. This is deliberately alert-only: it recommends whether
     an early close looks worth considering, but never places or closes
     an order itself -- that's the CEO's judgment call.
  3. From VERIFY_START onward: the real check. Queries real Alpaca
     positions for all 3 known leg symbols.
       - All 3 genuinely flat: reconcile the REAL close P&L from actual
         Alpaca order fills (same method used to manually fix QQQ/IWM/SPY
         2026-09-03/04), write the verified number into the registry if it
         is missing or differs from what's currently recorded, log a normal
         (not high-priority) confirmation. Marks itself done for the day
         (state file) so repeated runs don't re-alert/re-spam.
       - Any leg still real-open: HIGH-PRIORITY Telegram immediately with
         the specific leg(s) and real current value, THEN attempts a real
         emergency close via place_leg_with_ladder (same proven
         favorable->mid->aggressive ladder every other close in this
         codebase uses) -- but sends the alert regardless of whether the
         auto-fix succeeds, so this can never again go unnoticed the way
         2026-09-03's did.

Shared logic (registry lookup, GEX/dark-pool/insight synthesis, emergency
close) lives in butterfly_babysitter_common.py -- factored out 2026-09-04
when iwm_butterfly_babysitter.py was built as the second ticker instance
and several QQQ-only assumptions (uniform $1 strike increment, symmetric
wing widths) turned out to be real correctness bugs for other tickers, not
boilerplate. This file just supplies QQQ's own ticker/client-ID config.
"""
import sys

sys.path.insert(0, ".")
from butterfly_babysitter_common import run_babysitter

TWS_PORT = 7496
CLIENT_ID = 1691         # emergency-close connection
INSIGHT_CLIENT_ID = 1692  # intraday insight connection -- distinct from CLIENT_ID
                           # so an overlapping run can't collide on the same IBKR clientId

if __name__ == "__main__":
    run_babysitter("QQQ", TWS_PORT, CLIENT_ID, INSIGHT_CLIENT_ID)
