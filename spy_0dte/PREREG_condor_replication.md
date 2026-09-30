# Pre-registration — cross-asset replication of the expiry iron condor (v1)

This plan was written and hashed **before any QQQ or IWM price or option data was downloaded**. It tests whether the SPY result generalizes. The tested assets are fresh, never-seen data.

## Rules

- **Specification:** identical to `PREREG_condor_expiry.md` (sha256 `256a5a47…`) in every respect, with two exceptions.
- **Underlying:** QQQ and IWM, tested separately.
- **Wing width:** SPY used $3 at SPY ≈ $500–660, about 0.5% of spot. For the other assets the wing is `round($3 × X_open / SPY_open)` to the nearest listed $1 strike, with a minimum of $1. The inputs are that session's 09:30 opens, known before any entry.
- **Sessions:** every session from 2024-02-01 to 2026-09-21 on which the underlying has options **expiring that day**. Days without a same-day expiry are skipped, never replaced with another expiry.
- **Code:** the frozen SPY pipeline, parameterized only by root symbol and wing width. Source hashes are recorded in the results log.

## Test

- **Unit and statistic:** the session's mean condor P&L is one observation; the test is a one-sample t.
- **Two tests, Bonferroni:** each asset passes if its **mean > 0 and t ≥ 2.24**.
- **Outcomes:**
  - **Replication confirmed:** at least one asset passes and neither has a significantly negative mean (t ≤ −2.24).
  - **Replication failed:** neither asset passes. The SPY edge is then treated as SPY-specific and fragile, and the paper-trading plan is reconsidered.

## Reported, not part of the verdict

Pooled result across both assets; results at λ 0 / 0.5 / 1.0; put vs call side; by month; worst days; drawdown.
