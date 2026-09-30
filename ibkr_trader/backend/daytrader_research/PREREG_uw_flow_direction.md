# Pre-registration — options-flow direction for the Day Trader scalp

**Written 2026-09-29, while the data pull was still running and BEFORE any
flow signal was computed or inspected.** Hashed as
`PREREG_uw_flow_direction.md.sha256`.

## The gap being closed

`direction_search.py` screened eleven signed daily-bar features on two
independent panels. Nothing survived. The arithmetic it produced defines the
target precisely (407-panel, Day Trader candidates):

| | value | share of available |
|---|---|---|
| mean \|excess\| — what perfect sign earns | 1.726pp | 100% |
| best daily-bar feature found | 0.088pp | 5.1% |
| fee hurdle at the live $148 position | 0.473pp | 27.4% |

The move is large; daily bars predict ~5% of its sign; ~27% is needed. Same-day
direction is not in daily bars, which is unsurprising — by the open, the prior
session is priced. Options flow is intraday and is the next candidate.

## Data

Unusual Whales, two calls per (ticker, session):
- `net-prem-ticks` — one row per minute, `tape_time` in UTC, PER-MINUTE (not
  cumulative) values: `net_call_premium`, `net_put_premium`, `net_delta`,
  and bid/ask-side call and put volume.
- `ohlc/5m` — 5-minute candles, giving a real entry price and close.

Scope: the top 5 Day Trader candidates by `composite_score` on each session
(`composite_score >= 75`, `atr_pct >= 2.5`), 2024-09-16 onward — the key's
history limit. ~2,500 ticker-days.

## Train / test split — fixed here, before any signal is computed

- **EXPLORATION: sessions before 2025-09-16.** The functional form of the flow
  signal (which field, which window) is chosen here and only here.
- **HOLDOUT: sessions on or after 2025-09-16.** Untouched until the single
  confirmatory test.

Splitting by DATE, not at random, because same-day candidates are one event and
a random split would leak across it.

## Hypothesis

**H1:** Net options-flow direction accumulated in the opening window predicts
the sign of the REMAINDER of the session for high-magnitude candidates.

Entry at the close of the opening window (a real, tradeable price from the 5m
candles, not the open). Exit at the session close — "out same day", the stated
intent. Long only, matching the agent.

Candidate signal forms, all to be settled on the exploration set:
- `sum(net_delta)` over the window
- `sum(net_call_premium) - sum(net_put_premium)`
- ask-minus-bid side imbalance: `(call_ask - call_bid) - (put_ask - put_bid)`
- windows: 9:30–9:35, 9:30–9:45, 9:30–10:00

Whichever is chosen, it is FROZEN before the holdout is touched, and the choice
plus its exploration-set numbers are written into the research log first.

## Metric

- `excess` = the name's entry→close return minus the equal-weighted mean of the
  SAME measure across that session's candidates. Benchmark is the candidate set
  itself, because the live decision is *which of today's candidates to buy*.
- `t` on the series of DAILY means — one event per session.

## Pass criteria — fixed before the test

1. **Edge exists:** mean excess > 0 with **t > 2.0** on the holdout.
2. **Edge is tradeable:** mean excess > **0.473pp**, the Tiered round trip at
   the live $148 position.

- Both → validated. Implement, and only then discuss sizing.
- (1) only → real but sub-fee. Report the break-even position; Day Trader stays
  disabled. This is the likely good-but-insufficient outcome and it must NOT be
  dressed up as a pass.
- (1) fails → rejected. Options flow does not supply the direction either.

## Rules of engagement

- **ONE test on the holdout.** No re-cuts, no window sweeps, no second field.
- The holdout is spent afterwards, pass or fail.
- Timing realism: the window's closing 5m candle is the entry price. No
  mid-window or open price may be substituted to improve the result.
- A failure is reported as plainly as a success.
