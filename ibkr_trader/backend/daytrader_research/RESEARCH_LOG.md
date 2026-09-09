# Day-Trading Strategy Comparison — Log

Goal (CEO, 2026-08-25): find genuinely different day-trading strategy
archetypes, backtest them with real data, and synthesize one final
recommendation — compared honestly against what's already live. Does NOT
modify daytrader_scanner.py, day_trader live code, or any config. All new
code lives in this directory.

## Starting point — NOT a blank slate

Before building anything, surveyed existing daytrader_*backtest*.py scripts
already in the account and found substantial real prior work already
completed and ALREADY DEPLOYED:

1. `daytrader_sizing_backtest.py` (5y, 12,243 trades, daily bars): found the
   original blind-entry/fixed 0.25% target/1.0% stop config's daily-bar
   ambiguity unresolvable (26.7% or 92.7% win rate depending on assumed
   intraday ordering) -- every position-sizing scheme tested lost
   essentially all capital either way.
2. `daytrader_intraday_backtest.py` (real 1-min bars, 15 days, 147 usable):
   resolved the ambiguity for real -- BASELINE (blind entry) = 67.3% win
   rate but avg_ret -0.16%/trade (NEGATIVE despite majority wins -- reward:
   risk too small to break even at 80% required win rate). CONFIRMATION
   (0.35% + volume) + 0.3% TRAILING STOP = 50.5% win rate, avg +0.137%/trade
   (POSITIVE), +14.92% total over the sample.
3. `daytrader_atr_confirm_backtest.py`: tested whether an ATR-relative
   confirmation threshold beats the flat 0.35% -- it does NOT (every ATR
   multiple tested had lower avg return than the flat baseline).

**Confirmed via `/day-trader/status`: this exact winning config (confirm_pct
0.35%, trailing_stop_pct 0.3%, expected_return_pct 0.1369, win_rate_est
0.505) is the account's LIVE Day Trader config right now.** This whole
project's job is checking whether anything genuinely different beats it,
not rediscovering what already exists.

## Iteration 1 (2026-08-25, ~9:25-10:05 AM ET) — real data pipeline

Built new infra reusing the LIVE scanner's real selection logic directly
(no re-derived copy -- same discipline the existing scripts already use):
`load_universe`/`compute_dt_scores`/`apply_sector_cap` imported from
daytrader_scanner.py, `download_all`/`build_ticker_frame` from
daytrader_sizing_backtest.py.

`build_candidates.py`: real run, 503-ticker S&P 500 universe, 120-day
recent window (minute-bar data is the real constraint, not daily-bar
history) -> **820 real candidate (date, ticker) selections, 82 trading
days, 100 unique tickers** -- a materially larger sample than any prior
day-trader backtest in this account (prior work used 15-day/~150-candidate
samples).

`fetch_minute_bars.py`: real IBKR minute bars via the backend's own
`/market/history/minute` endpoint (same real source/pacing convention
daytrader_intraday_backtest.py established -- ~5 req/15s server-side, real
~40min fetch for 820 pairs). Running now.

`strategy_comparison.py`: built and syntax-verified while the fetch runs.
Real bug caught before it could waste the whole fetch: guessed 3 possible
key formats for the returned bar dict instead of checking the real
endpoint source -- actual format is `f"{ticker}:{date}"` (colon-separated,
confirmed by reading main.py's /market/history/minute handler directly).
Fixed before running anything against real data.

Four strategies to compare on the SAME 820-candidate sample (same real
bars, same opportunity set -- differences reflect entry/exit mechanics
only, not different stocks/days):
  1. BASELINE -- replicates the live config exactly, run fresh on this
     sample as a fair same-period reference point.
  2. Opening Range Breakout (15min and 30min range) -- genuinely different
     archetype: breakout above a defined range, not momentum confirmation.
  3. VWAP mean-reversion (long only) -- buys weakness (a dip below VWAP),
     opposite philosophy from the other three.
  4. Gap-fade (long only) -- bets on reversion toward prior close for
     names that gapped down meaningfully, rather than continuation.

## Iteration 2 (2026-08-25, ~10:05 AM ET) — real results, final synthesis

fetch_minute_bars.py completed clean: 820/820 real minute-bar series
returned. strategy_comparison.py ran against all 820, all real bars.

| Strategy | n | Win rate | Avg return | Total |
|---|---|---|---|---|
| **BASELINE (live)** | 581 | **52.7%** | **+0.154%** | **+89.55%** |
| ORB (15min range) | 495 | 47.7% | +0.079% | +39.24% |
| ORB (30min range) | 415 | 44.6% | +0.045% | +18.50% |
| VWAP mean-reversion | 774 | 37.7% | **-0.089%** | **-69.11%** |
| Gap-fade | 294 | 27.6% | +0.070% | +20.55% |

Real, on a materially larger sample (581-774 trades vs the baseline's own
prior 109-147-trade validation) -- confirms the live strategy's edge is
real and if anything slightly STRONGER on this bigger sample (52.7%/+0.154%
here vs 50.5%/+0.137% on the original 15-day validation).

**ORB degrades monotonically with wider ranges** (15min beats 30min on
every metric) -- real, makes sense (a tighter range is a more selective,
more information-dense breakout signal). Still loses to baseline at every
width tested.

**VWAP mean-reversion is a clean, real loser** (-0.089%/trade, worst win
rate). Makes sense in hindsight: these candidates are ATR-selected,
high-composite-score MOMENTUM names -- selected because they're already
moving. Betting on mean-reversion against a population selected for the
opposite quality fights the very reason they were chosen.

**Gap-fade's positive average is NOT robust** -- investigated further
before accepting it: median return is exactly -1.00% (the stop level,
meaning the median/typical trade is a real loser), 210/294 (71%) hit the
stop, only 70/294 (24%) hit the real gap-fill target. Removing just the
single largest winner (CSGP +16.56%, 2026-07-29) collapses avg return from
+0.070% to +0.014% and total from +20.55% to +3.98%. This is an outlier-
dependent result, not a real, repeatable edge -- flagged and rejected
rather than reported as a false positive.

## FINAL SYNTHESIS

**None of the 4 alternative archetypes tested beat the account's existing,
already-live Day Trader strategy** (ATR-selected momentum breakout +
confirmation gate + 0.3% trailing stop). The honest recommendation is to
keep it as-is -- this project's real value was rigorously checking for a
better alternative and finding none, on a real, materially larger sample
than the original validation used, which itself is a useful confirmation
the existing edge holds up. No code or config changes recommended.

Real caveats stated plainly: all 4 new archetypes were tested with only
one reasonable parameterization each (single dip_pct/stop_pct for VWAP
reversion, single min_gap_down/stop_pct for gap-fade) -- unlike the
existing strategy's own multi-round validation (sizing -> intraday ambiguity
resolution -> ATR-relative confirmation test), these weren't tuned across a
real parameter grid. A more exhaustive per-archetype tuning pass could
still be done if there's appetite, but the gap to baseline (especially
VWAP reversion's clearly negative edge) is large enough that it's unlikely
grid-tuning closes it.

## Iteration 3 (2026-09-05) — RSI(1min) tested and rejected, real live/backtest gap root-caused and fixed

CEO noted ATR% alone might not be enough and asked to test adding RSI on
1-minute bars, prompted by a real observation that live Day Trader
performance (33 closed trailing_stop trades, -$9.54, 21% win rate) looked
far worse than the validated backtest. Two closed questions and one major
real fix came out of this iteration.

### RSI(1min) as a confirmation-strength filter -- rejected

`rsi_1min_confirm_backtest.py`, same real 820-candidate 1-minute-bar
dataset. Wilder-smoothed RSI(14) on 1-min closes, evaluated at the
confirming minute, tested as an ADDITIONAL gate on top of the existing
0.35%+volume confirmation:

| Variant | n | Win rate | Avg return | Total |
|---|---|---|---|---|
| BASELINE (current live gate) | 581 | 52.7% | +0.154% | +89.55% |
| RSI >= 50 | 477 | 39.8% | -0.011% | -5.12% |
| RSI >= 60 | 410 | 40.5% | -0.012% | -5.03% |
| RSI >= 70 | 207 | 37.7% | -0.018% | -3.65% |
| RSI 50-70 band | 467 | 40.9% | -0.011% | -5.10% |
| RSI < 80 (overbought excl.) | 489 | 39.9% | -0.008% | -3.97% |

Every threshold flips the strategy from profitable to a loser. Same root
cause as the already-rejected ATR-relative confirmation test
(`daytrader_atr_confirm_backtest.py`): any condition that makes entry wait
longer means buying into a move that's already run further, with less
room before the trailing stop. Not RSI-specific -- this is a momentum
strategy; delay hurts regardless of what's used to justify the delay.

### RSI(1min)<30 oversold-bounce entry -- also rejected, fragile not real

Different mechanism (replaces the entry trigger with RSI<30+volume instead
of price+volume): positive-*looking* on the full population (n=240,
RSI<30, +3.53% total) but median return is negative at every threshold
tested (20/25/30/35), the classic outlier-dependent pattern already used
to reject the Gap-fade archetype in Iteration 2. Confirmed fragile via the
same "remove top N winners" stress test: removing just the top 5 of 240
winners flips +3.53% to -0.34%; top 10 to -3.57%. Checked whether any real
subset rescues it (`rsi_oversold_breakdown.py`, sector map reused from
daytrader_scanner.load_universe(), liquidity proxy = real avg daily $
volume from the bars themselves):
  - Best single ticker (VRT, 73.3% win rate, n=15): removing its own top 5
    of 15 wins flips +1.97% to -0.26%. Even the best-looking individual
    name doesn't survive the same stress test.
  - Top-liquidity quartile (the biggest, most liquid names): -0.96% total,
    a real loser outright -- restricting to high liquidity makes it WORSE,
    not better.
  - Information Technology (the largest sector bucket, n=69) is a net
    loser (-0.61%); no sector/liquidity slice rescues the idea.

Closed, documented finding: RSI(1min) does not help Day Trader's entry
logic in any form tested (strength filter, oversold-bounce, sliced by
sector/liquidity/ticker).

### Real root cause of the live/backtest gap -- found and fixed

Investigating "why is live so much worse than backtest" (not an RSI
question) led to the actual answer via real order-history + scanner-log
forensics, not a parameter search:

1. **`watch_expiry_backtest.py`**: candidates that never confirm within the
   live 60min window are correctly-filtered losers, not missed
   opportunities (forced entry on the 239 non-confirmers: 37.7% win,
   -0.041%/trade avg, -9.87% total -- walking away from these costs
   nothing). Extending the window doesn't rescue value either (rescued
   candidates at 90/120min are still net losers). The confirmation
   mechanism itself is sound.
2. Real order-history check (Alpaca account activities) found real,
   systematic trailing-stop-order slippage beyond the nominal trigger
   price: avg -0.068%/trade across a 12-trade sample, 11 of 12 adverse.
   Real but modest -- not enough alone to explain a 52.7%->21% win-rate
   collapse.
3. **The actual answer**: real entry prices land 1-9x further into the
   move than the 0.35% trigger implies (e.g. NOW on 2026-09-03: confirmed
   needing only +0.35%, real fill landed +3.13% above the true session
   open). Traced to the scanner's own design: `SCAN_TIME_ET=(9,35)` was a
   deliberate, never-validated 5min wait, and the real 2026-09-03 log
   shows the full pipeline (483-ticker fetch + sequential 20-signal
   dispatch) didn't finish until ~9:36:50 -- ~7min after the true open.
   `scan_delay_backtest.py` (same real 820-candidate dataset, watch-start
   delayed 0-10 real minutes past the true open) quantified this
   precisely: win rate 52.7%->41.0%, avg return +0.154%->+0.017%/trade
   between 0min and 7min of delay -- the confirmation gate's edge decays
   almost 10x over exactly the delay the live pipeline actually has. This
   is the real, dominant explanation for the live/backtest gap, not stop
   width, not RSI, not exit slippage alone.

**Fix deployed 2026-09-05** (see oversight_log.jsonl,
category=daytrader_scanner_pipeline): rebuilt daytrader_scanner.py into a
two-stage pipeline -- an 8am ET premarket shortlist build scoring the full
universe on the 75%-weight non-gap inputs (atr_pct/prior_day_ret_pct/
ret5d_prior, none of which need today's session), then a fast 9:30:10 ET
at-open finalize pass that only re-checks the cached ~80-ticker shortlist
with a REAL (not pre-market-estimated -- CEO explicitly didn't trust IBKR
extended-hours quotes for thinner names) gap_pct, dispatching the top 20
in parallel instead of the old sequential loop. Falls back to the full
483-ticker scan if the shortlist is missing/stale. Real end-to-end test
(market closed, real IBKR data): premarket build 50.6s for the full
universe, fast finalize 11.6s total for the shortlist (vs. the old
design's ~90-110s) -- both fallback paths (missing/stale shortlist)
verified to correctly trigger the original full scan rather than silently
doing nothing. Pending real validation against a live trading day (next
real session: Tuesday 2026-09-08, Monday 9/7 is Labor Day).

## Intraday dispersion re-check (2026-09-08) — tested, rejected

**CEO question**: today (2026-09-08) the dispersion-combo gate correctly sat out (dispersion
percentile 0.52 < 0.75, EOD-anchored on yesterday's close) — should the scanner re-check
dispersion intraday (e.g. hourly) to catch a same-day regime shift?

**First pass (`intraday_dispersion_test.py`)**: reused the 7,343 real signal-day outcomes from
`target_combo_multiregime_rows.csv` (real 0.3% trailing-stop exits, 2021-2026). Bucketed by
whether EOD dispersion fires vs skips, and whether that SAME day's full close-to-close realized
dispersion was high. Result looked promising: of 5,340 EOD-skipped days, 1,268 (23.7%) had high
same-day dispersion, and that bucket averaged +0.119%/trade vs +0.068% for the rest — seemingly
worth building.

**Caught before acting on it**: that first pass used each day's FULL close-to-close dispersion as
the "intraday" proxy — information you would not actually have at 10:30am, only by end of day. A
real intraday re-check needs a genuinely no-lookahead test.

**Real test (`true_intraday_dispersion_test.py`)**: pulled real Alpaca 1-min bars for the full
112-ticker dispersion universe, 2021-09 to 2026-08 (`fetch_am_window_bars.py`, ~37min, 112/112
tickers), computed dispersion from ONLY the prior close -> 10:30am ET move (what you'd genuinely
know by 10:30am), ranked the same way. Result reversed:

| Bucket | n | Win rate | Avg ret/trade | Median |
|---|---|---|---|---|
| EOD gate fires (live today) | 2,003 | 55.8% | +0.177% | +0.049% |
| EOD gate skips | 5,340 | 48.1% | +0.080% | -0.017% |
| EOD skips, TRUE 10:30am dispersion high | 276 | **46.4%** | +0.122% | **-0.023%** |
| EOD skips, 10:30am dispersion also low | 5,064 | 48.2% | +0.078% | -0.016% |

Only 5.2% of skipped days (276, down from the first pass's 1,268) even qualify as "high" once
lookahead is removed — and the ones that do are **worse**, not better: lower win rate than the
rest of the skip bucket, negative median return (the positive mean is a few outlier winners
pulling up a typically-losing bucket, not a real edge). Year-by-year is all over the place (2022:
-0.003% avg on n=49; 2024: 11.1% win rate on n=9; 2026: +0.392% on n=27) — no consistent regime
signal, mostly small-sample noise.

**Conclusion: do not build an intraday dispersion re-check.** The first pass's apparent edge was
entirely a lookahead artifact of using full-day data as an "intraday" proxy. The properly
no-lookahead version shows nothing to recover. This also empirically confirms
`daytrader_scanner.py`'s own 2026-09-01 comment was right: dispersion is genuinely EOD-determined,
and today's own price action through 10:30am doesn't carry the information a real-time re-check
would need. No code changes made to the live scanner.
