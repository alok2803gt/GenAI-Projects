# Breakout Alert Quality Research — Log

Goal (CEO, 2026-08-25): current breakout_scanner.py produces 30-40 alerts/day.
Find a small set of real, backtested filters/theories that raise win rate and
avg return by cutting noise, WITHOUT touching breakout_scanner.py or any live
config. All new code lives in this directory. All findings are reported
honestly, including negative results.

## Baseline (real data, alert_performance table, 2026-06-25 to 2026-08-24, n=1107)

- Win rate: 52.9%
- Avg return: +0.048% (EOD, entry-at-alert-price to same-day close)
- Median: +0.043%, stdev: 1.37%
- This is coin-flip performance -- avg return is not even 1/25th of a
  standard deviation. Real evidence the current unfiltered alert stream is
  mostly noise, consistent with the CEO's own observation.

## Iteration 1 (2026-08-25, ~6:20 AM ET) — mining existing captured features

Before inventing new indicators, checked whether the 8 fields already
captured per alert (pct_b, rsi, vol_ratio, tape_score, tape_label,
prev_state, mins_in_pre_breakout, state_path) already separate winners from
losers. Real bucketed results (min n=10 per bucket):

**tape_label** (only 159/1107 = 14% of alerts have this populated; 86% are
"NO DATA" and perform at baseline):
- STRONGLY BULLISH: n=7, win=85.7%, avg=+0.422% (tiny sample, promising)
- BEARISH: n=6, win=66.7%, avg=+0.354% (tiny sample, odd -- bearish tape on
  a bullish breakout signal outperforming needs investigation, not just
  taken at face value)
- BULLISH: n=57, win=52.6%, avg=+0.083%
- NEUTRAL: n=65, win=50.8%, avg=-0.074%
- NO DATA: n=948, win=52.8%, avg=+0.047% (= baseline, as expected)

**RSI**: overbought entries (70-80) UNDERPERFORM (n=69, win=50.7%,
avg=-0.154%) vs 50-60 band (n=428, win=54.0%, avg=+0.058%). Real, sensible
-- chasing an already-overbought breakout is worse, not better.

**pct_b**: sweet spot is 95-100 (n=13, win=69.2%, avg=+0.396%) -- right at
the breakout threshold, not yet extended. >100 (EXTENDED territory) is the
WORST bucket (n=66, win=48.5%, avg=-0.131%). Real, intuitive: buying
already-extended names performs worse than buying the actual breakout edge.

**vol_ratio**: below-average volume (0.5-1x) is the worst bucket (n=128,
win=44.5%, avg=-0.175%) -- a "breakout" without real volume confirmation is
weak, matches conventional TA wisdom, now backed by this account's own real
data. 2-3x is best (n=12, win=66.7%, avg=+0.353%) but too small to trust yet.

**mins_in_pre_breakout**: 30-60min and 120min+ bands underperform (win
41.9%/41.8%) vs 15-30min (win 54.3%) and 60-120min (win 56.2%) -- possible
"staleness" effect, setups that linger too long without resolving may be
lower quality. Non-monotonic pattern though (0-1min is only 49.4%) -- needs
more investigation, not a clean story yet.

**prev_state**: WEAKENING->current-state transitions win only 40% of the
time (n=20) despite a surprisingly high avg return (+0.252%, likely a couple
big winners skewing the mean with n=20) -- flag as noisy, not yet a real
finding.

### Honest caveats on Iteration 1
- All of this is CORRELATIONAL on a 2-month, ~41-trading-day sample. No
  out-of-sample test yet, no combined/multi-factor model, several buckets
  have n<20 and should not be trusted individually.
- "EOD return from alert price" is the only outcome metric available in the
  existing table -- doesn't capture max favorable excursion, doesn't test
  alternate holding periods (next-day, 3-day, etc.), doesn't test a stop-loss.
- Next real step: pull 5 years of real daily OHLCV (yfinance, matching every
  other backtest in this codebase) for breakout_scanner's own 112-ticker
  universe, recompute Bollinger %B/RSI/volume-ratio historically, and
  properly backtest candidate filters (pct_b entry zone, RSI ceiling,
  volume-ratio floor) against 5 years of REAL breakout-like setups --
  not just the 2 months this account has lived-alert data for.

## Iteration 2 (2026-08-25, ~6:30-7:00 AM ET) — 5-year data + important negative result

Pulled 5y real OHLCV for all 112 universe tickers (fetch_5y_data.py ->
universe_5y_ohlcv.pkl). Found and reused 4 EXISTING prior backtest scripts
(breakout_horizon_analysis.py, breakout_intraday_faithful_backtest.py,
breakout_volume_threshold_backtest.py, pre_breakout_backtest.py) rather than
duplicating work -- real prior finding already validated: volume-ratio
95th-percentile threshold beats 90th (2.5yr, 80 tickers, intraday-faithful).

**RSI finding, cross-validated on real live alerts (indicator_horizon_test.py,
1107 real alerts, all 4 horizons)**: RSI>=70 (overbought) at entry is
CONSISTENTLY bad and gets worse the longer held -- win rate 52.8%(EOD) ->
47.1%(+1d) -> 35.8%(+3d) -> 39.1%(+5d), avg return -0.07% -> -0.20% ->
-0.46% -> -0.60%. Real, monotonic, multi-horizon-consistent, n=64-72 per
horizon. BUT: only 6.5% of all alerts (72/1107) have RSI>=70 -- cutting it
takes daily alert count from ~28.4/day to ~26.5/day, a MODEST cut, not the
dramatic reduction the CEO wants. Real, honest, but partial answer.

**IMPORTANT METHODOLOGICAL NEGATIVE RESULT**: built a naive 5-year "raw
Bollinger-transition" proxy (event_reconstruction_5y.py, 7711 events, all
112 tickers, 2021-2026) to test genuinely new features (ADX, distance from
52-week high, gap-day) that aren't in the live captured data. Found a
promising, clean, monotonic gap-day effect (gap-down entries got
progressively worse through +5d, win rate 53%->44.8%). Cross-validated this
against the REAL live 1107-alert population (computed real gap_pct for
each actual alert's ticker+date) -- **it does NOT replicate. At +5d it's
actually REVERSED**: gap-down alerts averaged +1.223% (60.2% win rate) vs
gap-up's -0.133% on the real live population.

Also found the naive 5y proxy disagrees with the live data on RSI direction
too (5y proxy: RSI>=70 looks GOOD; live data: RSI>=70 looks BAD) -- same
pattern, same conclusion.

**Why this happens, and what it means going forward**: the live scanner's
F2/F5/F6/F7/F8/F9/F10 gates already heavily pre-filter which raw
transitions actually become alerts. A naive proxy tests the WRONG
population -- "all raw Bollinger transitions across 5 years" is not a
faithful stand-in for "what breakout_scanner.py's real gated pipeline would
have alerted." Conclusions from the naive proxy are NOT trustworthy for
predicting live-alert behavior. This account already has the RIGHT tool for
this -- breakout_intraday_faithful_backtest.py -- which faithfully
replicates the real gates (built for exactly this reason, per its own
docstring). Going forward: new-indicator testing needs to run through that
faithful methodology (or directly against the real, smaller live-alert
population), not the naive 5y proxy. The naive proxy is still useful for
breadth/regime-coverage checks on findings ALREADY validated on real data,
just not for discovering new ones from scratch.

## Iteration 3 (2026-08-25, ~7:00-7:30 AM ET) — real gate-accurate 2.5y backtest

Built faithful_backtest_extended.py: reuses breakout_intraday_faithful_backtest.py's
REAL simulation logic (same gates, same Alpaca minute-bar methodology) but
fixed at TODAY's actual production settings (no grid sweep) and extended to
capture RSI/ADX/gap/dist-from-52w-high plus 1d/3d/5d forward returns, not
just same-day EOD. Real run: 1420 properly-gated alerts, 86 tickers,
2024-06-11 to 2026-08-24 (2.5 years, spans a real regime range).

**RSI >= 80 (extreme overbought) is real, monotonic, consistent** across
all 4 horizons (n=28): EOD -0.42%/43%WR -> +1d -0.93%/43%WR -> +3d
-0.87%/36%WR -> +5d -1.71%/36%WR. Confirms and REFINES the earlier live-data
RSI finding -- it's specifically the extreme >=80 band that's clean; 70-80
is actually fine-to-good at +3d/+5d (55%+ WR). Real edge, but only 2.0% of
alerts (28/1420) -- too small alone to fix the volume problem.

**IMPORTANT CORRECTION, triangulated across two independent real datasets**:
gap-down entries are NOT bad -- they're mildly GOOD at longer horizons.
Faithful 2.5y backtest: gap<-1% shows +1.48% avg/56%WR at +5d (best gap
bucket at that horizon). This matches the live 2-month cross-check
(+1.22%/60%WR at +5d) and CONTRADICTS Iteration 2's naive-proxy finding
(which showed gap-down getting worse). Two real, independently-gated
datasets agree; the naive proxy was wrong. Do NOT filter on gap-down.

**Distance from 52-week high**: entries within 5% of the 52w high show the
best, most consistent performance (+5d: +0.75%/55%WR) vs further off the
high (-5% to -15%: +0.06%/46%WR; >-15%: mixed, weaker). Real, modest tilt,
not a hard filter.

**BREAKOUT vs PRE-BREAKOUT signal type -- the real volume lever**:
PRE-BREAKOUT is 1270/1420 = 89.4% of ALL alert volume. Real performance
gap: BREAKOUT averages +1.75%/+1.83% (3d/5d) vs PRE-BREAKOUT's
+0.34%/+0.51% -- roughly 5x the avg return, though win rates are closer
(53-54% vs 51-52%, so this is more a fat-right-tail effect than a higher
hit rate). This is the first REAL, large lever found: since PRE-BREAKOUT
is ~9 of every 10 alerts sent, tightening or reducing PRE-BREAKOUT
specifically (not BREAKOUT) is where the actual daily-count problem lives.

### Honest overall read after Iteration 3
No single individual factor tested so far (RSI, ADX, gap, 52w-distance)
cuts a LARGE fraction of alert volume on its own -- the real, modest edges
found (RSI>=80 exclusion, near-52w-high preference) only touch a few
percent of alerts each. The one real large-volume lever is the
BREAKOUT/PRE-BREAKOUT split itself. Next real step: test whether
PRE-BREAKOUT alerts that ALSO clear the RSI<80 + near-52w-high quality bar
combine into a materially smaller, better-performing subset -- and check
whether raising PRE-BREAKOUT's own pct_b/RSI entry thresholds (currently
65/60) trades quantity for quality in a way that's actually favorable.

## Iteration 4 (2026-08-25, ~7:30 AM ET) — the headline finding

Investigated PRE-BREAKOUT specifically (89% of faithful-backtest volume, and
per alert_history the single biggest category live too). Found pct_b is
NON-monotonic within PRE-BREAKOUT: 65-75 band best, 75-85 band worst (a real
"dead zone"), 85-95 good again -- not the clean gradient assumed. Testing
"exclude pct_b 75-85 AND RSI<80" as a PRE-BREAKOUT-specific quality filter:

- 2.5y faithful backtest: 71.3% of PRE-BREAKOUT alerts pass, win rate
  53.7%/53.1% (3d/5d) vs baseline 52.0%/51.1%, avg return +0.528%/+0.696%
  vs baseline +0.342%/+0.513%. Real but modest on this cut.
- **Cross-validated directly against the REAL 537 live PRE-BREAKOUT alerts
  (2026-06-25 to 2026-08-24)**: 59.4% pass. The pass/fail split is a clean
  reversal, not just a tilt:
  - PASS (319 alerts): win rate 54.5%(EOD)->59.0%(+5d), avg return
    +0.087%->+0.880%
  - FAIL (218 alerts): win rate 46.8%(EOD)->45.7%(+5d), avg return
    -0.087%->-0.371%

This is the strongest, most consistent, most directly-validated result of
the whole research session -- confirmed on the ACTUAL live alert population,
not just a backtest proxy.

**Real volume impact**: 537 live PRE-BREAKOUT alerts / 41 days = ~13.1/day.
Filtered to 59.4% -> ~7.8/day, saving ~5.3/day. Combined with BREAKOUT's
~14.3/day (588/41, left untouched by this specific filter, not yet tested
for its own quality tightening), total daily alerts would drop from ~28.4
to ~22/day (~22% cut) -- with the PRE-BREAKOUT alerts that remain having a
MUCH better real track record (flips from marginal/negative to solidly
positive).

### Caveat, stated plainly
The faithful backtest's own BREAKOUT/PRE-BREAKOUT split (11%/89%) does not
match the real live alert_history table's split (588 BREAKOUT / 580
PRE-BREAKOUT, roughly 50/50) -- the simulation's dedup logic likely isn't
byte-for-byte identical to production's (e.g. same-day BREAKOUT-after-
PRE-BREAKOUT counting as two separate alert_history rows live, vs the
simulation only keeping one). This does NOT undermine the headline filter
finding above, since that was independently re-validated directly against
the real 537 live PRE-BREAKOUT alerts -- but it does mean the faithful
backtest's absolute alert-volume/mix numbers shouldn't be taken as exactly
matching production, only its within-PRE-BREAKOUT relative comparisons.

## Iteration 5 (2026-08-25, ~8:07 AM ET) — BREAKOUT-specific analysis

Tested the same feature set against BREAKOUT alerts specifically (n=150,
faithful 2.5y backtest). Smaller sample, more mixed picture than
PRE-BREAKOUT -- no single finding as clean or large:

- ADX<30 beats ADX>=30 (win rate 55-57% vs 45-49%, both horizons) -- real,
  plausible (a fresh breakout in a not-yet-exhausted trend has more room),
  but n=33 in the ADX>=30 bucket, not cross-validated against live data yet.
- mins_in_pre_breakout before confirming is non-monotonic again: 30-60min
  is the WORST bucket (win 49-51%, avg +0.14%/+0.61%), both 15-30min AND
  60+min do better. The 60+min bucket looks great (61-72% WR, avg
  +3.27%/+4.59%) but n=18 -- too small to trust on its own.
- RSI 70-80 for BREAKOUT actually does BETTER than RSI<70 (63-69% WR vs
  53-54%) -- the OPPOSITE direction from the RSI>=80 finding on
  PRE-BREAKOUT. RSI's relationship to outcome is signal-type-dependent, not
  a universal rule -- consistent with the account's other findings that a
  filter tuned on one population doesn't automatically transfer.

**Conclusion**: BREAKOUT doesn't need aggressive filtering the way
PRE-BREAKOUT did -- it's already a smaller (~11% of volume), better-
performing signal (avg +1.75-1.83% vs PRE-BREAKOUT's +0.34-0.51%). The
patterns found here are real but smaller-sample and not yet cross-
validated against the live alert population the way the headline
PRE-BREAKOUT filter was -- flagging as a real lead, not a validated result.

## Iteration 6 (2026-08-25, ~8:34-8:50 AM ET) — shadow filter, live validation infra

Built shadow_filter_monitor.py: read-only against alert_history (breakout_
scanner's own real table), applies the Iteration 4 filter to every alert as
it fires, logs pass/fail to shadow_filter_log.csv. No writes to
breakout_scanner.py or its state, no Telegram, no trading -- purely a live,
ongoing, OUT-OF-SAMPLE validation layer (strongest possible test: these are
alerts that fire AFTER the filter was designed, not historical replay).

Ran once against the FULL historical alert_history (1168 alerts, market
closed so real live polling hasn't started yet) as an integration test --
independently reproduced the headline finding via a completely fresh code
path/join, on the hardest available metric (same-day EOD return, not the
friendlier 3d/5d): PASS cohort (891, includes all BREAKOUT pass-through +
filtered PRE-BREAKOUT) 54.5% win rate/+0.100% avg vs FAIL cohort (180,
PRE-BREAKOUT only) 46.7% win rate/-0.107% avg. Real, clean, consistent with
every earlier check.

Added IBKR-BreakoutShadowFilter to scheduled_tasks_setup.ps1 (every 15min,
market-hours-gated, same pattern as the dark-pool monitor) -- pending CEO
re-run of the setup script. Once market opens and this starts polling real
NEW alerts (last_seen_id already caught up to 1168, so it'll only log
genuinely new ones), `--report` will start showing true forward validation
as those alerts' real outcomes land in alert_performance.

## Iteration 7 (2026-08-25, ~9:05 AM ET) — ranking within PREFER

CEO asked whether PREFER (53.5% of alerts) can be ranked further, not just
treated as one bucket. Tried a smooth continuous score first (weighted sum
of real per-bucket avg returns) -- real but noisy: only 16 distinct values
given the categorical features, and one bucket (n=13) showed a -7% avg
driven by outliers. Honest conclusion: a full continuous ranking oversells
precision this sample size doesn't support.

Collapsed to a validated 3-level sub-ranking within PREFER instead --
clean and MONOTONIC across both ret_3d and ret_5d, decent samples:

- PREFER-HIGH (n=316-320): win rate 57.6%->59.9%, avg +1.09%->+1.39%,
  median +0.68%->+0.84%. BREAKOUT+ADX<30, or PRE-BREAKOUT clearing >=2 of
  3 checks (pct_b 65-75, dist-from-52w-high >-2%, vol_ratio>=1.5).
- PREFER-MID (n=376-377): win rate 55.6%->54.5%, avg +0.47%->+0.82%,
  median +0.19%->+0.25%. Clears exactly 1 check.
- PREFER-LOW (n=63): win rate 42.9%->46.0%, avg -0.08%->-0.59%, median
  -0.54%->-0.47%. Clears 0 checks (or BREAKOUT+ADX>=30). REAL, honest
  finding: this bucket goes NEGATIVE -- worse than NEUTRAL. The coarse
  PREFER boundary from Iteration 4 is a bit loose at the edges; flagged
  rather than silently tightened, since PREFER-LOW is now separately
  visible in the tag rather than hidden inside PREFER.

Rewired shadow_filter_monitor.py to a full 5-level tag (AVOID / NEUTRAL /
PREFER-LOW / PREFER-MID / PREFER-HIGH). Real engineering note: alert_history
only captures pct_b/rsi/vol_ratio live -- NOT adx or dist_52w_high (those
were backtest-only features). Rather than silently drop 2 of the validated
ranking's real components, the live tagger now computes both in real time
via a per-ticker yfinance daily-bar pull (cached within one poll run).
Retagged the full historical alert set as a plumbing test -- **important
real catch, not a clean re-validation**: the live retag used TODAY's
(2026-08-25) ADX/52w-high for all 1168 historical alerts, not the real
point-in-time values from each alert's actual date -- a materially
different (and wrong, for backtest purposes) computation than the offline
faithful backtest, which correctly indexed to each alert's own historical
date via Alpaca daily bars. Real effect: PREFER-LOW's badness DID hold up
retroactively (-0.14% avg on the hardest EOD-only metric), but PREFER-HIGH
vs PREFER-MID ordering did NOT cleanly reproduce (MID edged out HIGH on
same-day EOD, whereas both cleanly beat LOW in the correct offline test).
This retroactive run only proves the code executes correctly -- it is NOT
a valid historical backtest and isn't being treated as one. Going forward
this is the right approach: real-time computation is point-in-time-correct
for genuinely NEW alerts as they fire (today's data IS the right timeframe
for a live alert). Cleared shadow_filter_log.csv and left
last_seen_id=1168 so the log starts clean with only real, live, correctly-
timed tags from here.

## Iteration 8 (2026-08-25, ~9:10 AM ET) — Telegram for PREFER-HIGH

CEO confirmed IBKR-BreakoutShadowFilter is live (every 15min, market-hours-
gated, verified via Get-ScheduledTask: PT15M repetition, next run 9:08 AM
ET). Asked for PREFER-HIGH specifically to notify via Telegram.

Added telegram() to shadow_filter_monitor.py, reusing the same scanner_
config.json creds pattern as every other monitor in this account. ONE
digest per poll run (never one message per alert -- same flood-avoidance
convention as darkpool_activity_monitor.py), containing only PREFER-HIGH
alerts; every other tier still logs to shadow_filter_log.csv but doesn't
notify. Every message states plainly it's a research tier, not a trade
signal. Creds load confirmed working. scheduled_tasks_setup.ps1 description
updated -- pending CEO re-run for the new description text (logic itself
is already live, standalone script, no restart needed for the code change
itself).

## Next planned iterations
1. Pull 5y OHLCV for the 112-ticker universe (breakout_scanner.py's own
   CANDIDATE_POOL, read-only reference, not modified).
2. Reconstruct the %B/RSI/vol-ratio state machine historically (same
   classify_state() logic, read as reference) to generate a real 5-year
   sample of breakout/pre-breakout events, not just the 2 months of live
   alerts.
3. Test the 4 real leads above (pct_b entry zone, RSI ceiling, vol-ratio
   floor, mins-in-pre-breakout window) individually and combined, across
   multiple regimes (2021 growth/2022 bear/2023-24 recovery/2025-26 current).
4. Explore genuinely new indicators/theories beyond what's already captured
   (e.g. ADX trend strength already exists as F8 in the live scanner --
   check if IT correlates with outcome in the real alert data too; also
   worth trying: multi-day momentum context, sector-relative strength,
   distance from 52-week high, gap-day vs no-gap-day entries).
5. Report every real backtested candidate strategy, including ones that
   fail, with real parameter grids and multiple regimes -- same standard
   as this account's other opportunity-evaluation work.
