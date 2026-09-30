# Pre-registration — SPY 0DTE iron condor held to expiry (holdout test, v1)

This plan was written and hashed **before any holdout session was loaded**. It supersedes the earlier "short 1.0-move call spread" holdout idea, which is withdrawn. This is the **only** test the holdout will be used for.

## Origin (disclosed)

The idea came from exploration round 4 on development data only (`explore_ledger.csv`, 1,406 tests in total). It was motivated by a cost argument: buying back near-worthless short options at 15:45 pays spread and commission, and gives up the last 15 minutes of decay. That argument does not depend on the data.

On development data (403 sessions) it made +$10.14 per condor per day, with t = 3.90. First half: +$9.47, t 2.66. Second half: +$10.80, t 2.84. Every one of the 19 entry times was positive, and 17 of 20 months. It did **not** clear the development Bonferroni bar of 4.12, which is why it needs this test.

## Specification (frozen — nothing may change after viewing)

| Item | Rule |
|---|---|
| Sessions | Holdout: 2025-09-22 … last session with option data. Full and early-close sessions, same filters as development |
| Entries | Signal bars at 10:00, 10:15, …, 14:30 (19 per full session); fill 1 minute later at that minute's 1-min VWAP (prints ≤ 2 min old allowed) |
| Expected move `u` | S × ATM implied vol (signal-minute prints only) × √(minutes left / (390·252)) |
| Put spread | Short leg: the printed strike nearest S − 1.0u. Long wing $3 lower. The wing must have printed at the signal minute |
| Call spread | Short leg: the printed strike nearest S + 1.0u. Long wing $3 higher. The wing must have printed at the signal minute |
| Condor | Both spreads at the same entry; each spread's credit after costs must be > 0 |
| Entry costs | QuoteModel width = max($0.01, 3% × mid), × 1.5 when ≤ 30 min left; **λ = 0.25**; $0.65 per contract per leg |
| Exit | Held to expiry. Settled at intrinsic value against SPY's final regular-session 1-min close. No exit cost or commission |
| Code | `option_returns.build_candidates` (structures `spread_P_d1.0`, `spread_C_d1.0`) plus `explore_round4.spread_variants`, at the frozen source hashes recorded in the results log |

## Primary test (one)

- **Statistic:** the session's mean condor P&L (averaged over its entry times) is one observation. The test is a one-sample t on those session means.
- **PASS** if and only if the mean is **> 0 and t ≥ 2.00**.

## Reported but not part of PASS/FAIL

These are descriptive only: results at λ ∈ {0, 0.5, 1.0}, put side vs call side, by entry time, by month, worst days, maximum drawdown, and win rate.

## Decision

- **PASS:** move to paper trading. The known live-execution gaps must be addressed before any real capital:
  - real bid/ask fills;
  - SPY's physical settlement and after-hours pin/assignment risk, or moving to cash-settled SPX;
  - position sizing.
- **FAIL:** the hypothesis is rejected. The holdout is spent, and no variant may be re-tested on it.
