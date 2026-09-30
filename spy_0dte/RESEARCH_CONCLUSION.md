# SPY 0DTE — Research Conclusion (frozen_v1)

**Status:** archived on 2026-09-22 under the pre-registered stopping rule. The locked holdout (sessions from 2025-09-22 on) remains unexamined.

## Phase 2 conclusion (development data only)

- **No robust edge.** Across the tested SPY 0DTE structures, entry times, holding periods and observable features, no robust conditional trading edge was identified.
- **Long options lose.** Their returns were negative even under frictionless modeled execution.
- **Credit spreads are indistinguishable from zero.** They showed a small gross return. In aggregate, spread profitability was not statistically distinguishable from zero, and it deteriorated under modeled execution costs.
- **Models add nothing.** Out of sample, Ridge and LightGBM did not outperform structure-specific unconditional means. Model-selected trades did not produce statistically reliable profits.
- **One structure was exploratory only.** One short-call structure showed exploratory significance at zero execution cost. This was one result among multiple structures, and it weakened under the frozen λ = 0.25 assumption.

Development is therefore stopped under the pre-registered rule. No additional feature or threshold search, and no quote-data purchase, is justified. The locked holdout remains unexamined.

## Key numbers

Inference is at the session level: overlapping candidates within one session count as one observation, with a whole-session bootstrap.

| Every candidate taken | λ = 0 mean $/contract | t | 95% CI | λ = 0.25 mean |
|---|---|---|---|---|
| All credit spreads | +1.96 | 1.64 | −0.36 … +4.21 | +0.19 |
| Short call spread, 1.0 move (exploratory) | +2.74 | 2.54 | +0.59 … +4.88 | +1.41 (t 1.29) |
| All long options | −3.59 | −1.80 | −7.00 … +0.75 | −5.04 |

| Model check | Result |
|---|---|
| Out-of-sample R² vs structure mean | Negative for Ridge and LightGBM, for both longs and spreads |
| Tail ablation (raw vs 1/99 winsorized target) | Raw target is worse in every rule; winsorizing did not create an edge |
| Best frozen trade rule at λ = 0 | Ridge, EV > $10, 1 trade/day: +$7.48/trade, t = 1.04 (75 trades) |

## What the successive controls removed

1. **Synthetic pricing.** Real traded option prices replaced Black-Scholes P&L, and most of the apparent spread profit disappeared.
2. **The regime/barrier proxy.** Direct forward option P&L replaced it as the target. The structure mean was the baseline to beat, and realized P&L had to be economically ordered by predicted EV.
3. **Execution assumptions.** Even frictionless execution (λ = 0) did not give sufficient evidence to continue.

## Addendum — last-hour test (2026-09-22, archive/last_hour_v1)

The plan was pre-registered in `PREREG_last_hour.md` and hashed before any result was computed (sha256 `72a2d28e…`). It covered four structures: an ATM long call, an ATM long put, and 0.5u short put and call spreads, $2 wide. Entries were at 15:00, 15:15 and 15:30 with exit at 15:55, on 405 full development sessions. **No structure passed**, so the last-hour hypothesis is rejected.

| Structure | λ = 0.25 mean | t | λ = 0 mean |
|---|---|---|---|
| Long call | +0.21 | 0.06 | +1.76 |
| Long put | −5.26 | −1.48 | −3.85 |
| Short put spread | −2.26 | −1.45 | −0.67 |
| Short call spread | −6.19 | −3.62 | −4.60 |

**Post-hoc observation (not a finding):** short call spreads lost money consistently in the last hour. This fits the period's afternoon upward drift, which is a market-direction effect specific to 2024–25 rather than a structural premium. It must not be treated as a discovered strategy.

## Rules for the holdout

- **It is not research data.** It must not be used to iterate on these development results.
- **One test only.** If it is ever opened, test exactly one frozen specification, fixed before viewing:
  - short 1.0-expected-move call spread, $3 wide
  - every decision bar from 10:00 to 14:30, exiting at 15:45
  - λ = 0.25
  - no change to timing, width, filters, thresholds or features after seeing it
- **Alternatively, save it.** Keep it for a genuinely new hypothesis that comes from independent reasoning or new data.
