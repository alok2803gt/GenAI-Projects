# Pre-registration — SPY 0DTE last-hour test (v1)

This plan was written and hashed **before any last-hour result was computed**. Its SHA-256 is printed in every results log. Any edit after the first run invalidates the test.

## Hypothesis

In the final hour, 0DTE option P&L is dominated by rapid theta decay and gamma. This is a different economic regime from the 10:00–14:30 window that frozen_v1 tested. The question is whether a **short-premium or long-gamma structure earns a positive, execution-robust unconditional return** in that hour.

The test is unconditional only: no ML model and no features. frozen_v1 showed that conditional models added nothing, so this checks first whether any premium exists at all.

## Data and period

- **Instrument:** SPY 0DTE options, using the same data as frozen_v1. Prices are traded 1-minute VWAP from Alpaca, used as the mid.
- **Period:** development sessions only, 2024-02-01 … 2025-09-19. The holdout (2025-09-22 onward) is **not** loaded.
- **Session filter:** full sessions only (390 minutes). Early-close days are excluded.

## Timing

| Item | Rule |
|---|---|
| Signal bars | 14:59, 15:14, 15:29 |
| Entry (fill) | 15:00, 15:15, 15:30, at that minute's VWAP. A print at most 2 minutes old is allowed |
| Exit | 15:55, at that minute's VWAP. If there is no print, use the first print up to 15:59. If there is still none, the trade is booked at its **worst case** |
| Strike selection | Uses only prints at or before the signal minute |

## Structures (exactly four)

`u` is the expected move: S × (ATM implied vol from signal-minute prints) × √(minutes left / trading-minute year).

| ID | Structure |
|---|---|
| L_C | Long call, nearest printed strike to S |
| L_P | Long put, nearest printed strike to S |
| SPS | Short put spread: short leg at the printed strike nearest S − 0.5u, long leg $2 lower |
| SCS | Short call spread: short leg at the printed strike nearest S + 0.5u, long leg $2 higher |

A spread is skipped if its wing did not print at the signal minute, or if its credit after costs is ≤ 0.

## Costs

- Bid/ask is modeled with QuoteModel: width = max($0.01, 3% × mid), × 1.5 when 30 or fewer minutes remain.
- **Primary execution level: λ = 0.25.** Commission: $0.65 per contract per leg, each way.
- Sensitivity grid: λ ∈ {0, .05, .10, .15, .20, .25, .50}.

## Inference

- **Unit:** the session. The mean of a structure's candidates within a session counts as one observation.
- **Statistics:** session-level t, and a whole-session bootstrap 95% CI (2,000 draws, seed 20260922).
- **Multiple testing:** 4 structures, Bonferroni, two-sided α = 0.05 → **|t| ≥ 2.50**.

## Pass rule (per structure; ALL must hold)

1. Session-level mean P&L at **λ = 0.25** is > 0, with **t ≥ 2.50**.
2. The whole-session bootstrap 95% CI at λ = 0.25 lies entirely above 0.
3. Break-even **λ* ≥ 0.40**.
4. The mean is positive in **both halves** of the development period, split at the median session.
5. Exits missing a real print are **≤ 2%** of that structure's candidates.

## Decision

- **If no structure passes:** the last-hour hypothesis is rejected, no further last-hour variants are tried, and the holdout stays unspent.
- **If exactly one passes:** it becomes the **single** frozen holdout candidate, with nothing changed. Opening the holdout is still a separate, explicit decision.
- **If more than one passes:** only the one with the highest t at λ = 0.25 is carried forward.

## Reported but not part of the pass rule

These are descriptive only: results by entry time, the full λ grid, win rate, and exit-price staleness.
