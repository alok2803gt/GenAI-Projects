# Pre-registration — QQQ confirmation of SPY hold-to-expiry discoveries (v1)

This plan was written and hashed **before any QQQ hold-to-expiry structure other than the midday 1.0u/$3 condor was computed**. The QQQ `expiry_family` legs have not been built yet.

## Origin

`expiry_family.py discover` ran on SPY, 660 sessions: 62 structure × window tests, with a Bonferroni bar of t 3.35. The three structures below were chosen by these rules, fixed before seeing QQQ:
- Highest SPY t.
- Positive with t ≥ 2 in **both** SPY halves.
- Excluding the already-validated midday 1.0u/$3 condor.
- At least one defined-risk structure from each entry window.

| ID | Structure (held to expiry) | Window | SPY t (dev / holdout) |
|---|---|---|---|
| H1 | Iron condor, short strikes ±0.5u, wings $5 | late (entries 14:46–15:31) | 5.86 (4.13 / 4.25) |
| H2 | Short straddle, ATM (naked — risk flagged) | late | 5.86 (4.06 / 4.44) |
| H3 | Iron condor, short strikes ±0.5u, wings $5 | midday (entries 10:01–14:31) | 4.51 (3.98 / 2.21) |

## QQQ test

- **Code:** `expiry_family.py` unchanged, run with root QQQ. The wing is scaled as `max($1, round($5 × QQQ_open / SPY_open))`, the same rule as the replication test. λ 0.25; $0.65 per contract.
- **Settlement:** QQQ's final regular-session 1-min close.
- **Sessions:** all QQQ sessions from 2024-02-01 to 2026-09-21 with a same-day expiry.
- **Statistic:** one-sample t on session means.
- **Pass:** each structure passes if its **mean > 0 and t ≥ 2.39** (Bonferroni over 3).
- **Confirmed:** a structure is confirmed if it passes. The family result is confirmed if ≥ 2 of 3 pass.
