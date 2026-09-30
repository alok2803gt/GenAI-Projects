# Pre-registration — Day Trader revival rule

**Written 2026-09-29, BEFORE any holdout data was downloaded or inspected.**
Hash this file (`PREREG_daytrader_revival.md.sha256`) fixes its contents. Any
later edit invalidates the registration.

## Why a holdout is required

Every statistic produced on 2026-09-29 came from the 112-ticker panel in
`breakout_research/universe_5y_ohlcv.pkl`. That panel is contaminated: across
`gate_threshold_study.py`, `confirmation_gate_study.py`,
`universe_reduction_study.py` and the gap-band decomposition, well over a
hundred subsets of it have been examined. Every one landed between +0.1pp and
+0.3pp with t between 1.2 and 1.8 — the signature of a weak-or-absent effect
sliced many ways. No in-sample number from that panel may be used to justify
trading.

## Hypothesis

The Day Trader's candidate pipeline has three identified DEFECTS, each
established on the contaminated panel but each with a reason to believe
independent of it:

1. **Direction.** The scanner ranks on `|gap_pct|`, so it surfaces gap-UPS,
   which measured negative excess in every band tested (−0.014 to −0.081pp),
   while the agent is LONG ONLY. The original 500-ticker study independently
   reported gap-up averaging −0.03% vs gap-down +0.13%.
2. **Falling knives.** Gaps worse than −3% were the single worst band measured
   (−0.257pp, t=−1.64) — the "catching a falling knife" risk the scanner's own
   docstring warns a single-day average cannot capture. The usable gap-down
   region is modest gaps, not extreme ones.
3. **The sigma gate.** `std_score >= 3.7` had no support in any population
   tested (−0.143pp overall; −0.101pp within gap<=0; −0.227pp versus the
   candidates it rejected) and reduces the strategy to ~5% of sessions when
   ANDed with the ATR gate.

**H1:** After correcting all three, the rule below earns a positive
market-adjusted, date-clustered excess return on tickers never examined.

## The rule — frozen

Entry candidates on session *t*, evaluated at the open:

| Component | Value |
|---|---|
| `composite_score` | **>= 75** (unchanged; 0.55·pr(atr_pct) + 0.25·pr(\|gap\|) + 0.12·pr(\|prior_day_ret\|) + 0.08·pr(\|ret5d\|), percentile-ranked within that day's universe) |
| `atr_pct` | **>= 2.5** (unchanged) |
| `gap_pct` | **in [−3.0%, 0.0%]** — NEW; excludes gap-ups and falling knives |
| `atr_mult` | **>= 1.4** — relaxed from 1.8, measured on the last COMPLETED session |
| `std_score` | **gate REMOVED** |
| Direction | long only |
| Entry / exit | buy at the open, exit at the close, one unit per name |

All features use bars strictly before *t*, except `gap_pct`, which uses
`open[t]` and `close[t-1]` — both known at entry.

## Data — untouched at time of writing

The **410 S&P 500 tickers in `sp500_universe_cache.json` that are NOT among the
112 in `universe_5y_ohlcv.pkl`.** Same ~5-year daily window. Not yet
downloaded. Percentile ranks and the market benchmark are computed *within the
holdout universe itself*, so it is a self-contained replication rather than a
graft onto explored data.

## Metric

- `excess` = the name's same-day open→close return minus the equal-weighted
  mean open→close return of the holdout universe that same day.
- `t` computed on the series of **daily means** — one event per session.

## Pass criteria — fixed before the test

1. **Edge exists:** mean excess > 0 with **t > 2.0**.
2. **Edge is tradeable:** mean excess > **0.47pp**, the Tiered round-trip cost
   ($0.70) at the live $148 position.

- Both hold → the rule is validated; implement and size at >= $148.
- (1) only → a real but sub-fee edge. Report the break-even position and leave
  the Day Trader DISABLED until net liq supports it. Do NOT trade it.
- (1) fails → the revival is rejected. Day Trader stays disabled.

## Rules of engagement

- **ONE test.** No variants, no re-cuts, no threshold sweeps on the holdout.
  The first number produced is the result.
- The holdout is **spent** after this test, pass or fail.
- A failure is reported as plainly as a success.
