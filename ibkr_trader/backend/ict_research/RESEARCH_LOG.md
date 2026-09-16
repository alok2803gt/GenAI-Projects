# Liquidity-Sweep Reversal (ICT-style) — From-Scratch Strategy

## Goal

Third genuinely different strategy tested this session (after the XGBoost Signals-model variants
in `signals_research/`, found to have no edge, and VWAP-deviation mean reversion in
`scalp_research/`). This one is structurally different from both: not a classifier, not a
volume-weighted-average-price fade — a **liquidity-sweep reversal**, one of the core setups from
ICT ("Inner Circle Trader") retail trading methodology. Concept: price briefly breaks a recent
swing high/low (triggering stops resting there — "sweeping liquidity"), then snaps back inside the
prior range. The sweep is read as the move that was actually designed to trigger stops and fill
opposing orders, not a genuine breakout — trade the reversal, not the break.

ICT methodology in full is broad and often applied somewhat discretionarily (order blocks, fair
value gaps, market structure shifts, premium/discount zones, session killzones all combined). This
tests one well-defined, objectively-codeable piece of it — the sweep-and-reversal itself — rather
than every ICT concept at once, so a result can be attributed to a specific, testable mechanism.

## Constraints (same discipline as the other two)

- Does NOT modify `main.py` or any live trading process. No capital at risk.
- Built fresh, no shared code with `signals_research/` or `scalp_research/`.
- Real 1-min Alpaca bars (reuses the same real, split-adjusted data already fetched for
  `scalp_research/` — re-fetching identical historical price data would just waste an API pull,
  this is data reuse, not strategy-logic reuse).
- Tested across a real parameter grid and multiple regimes, same as the other two.
- Honest reporting regardless of outcome.

## Strategy design

- **Swing level**: rolling N-minute lookback high/low (the level that's plausibly resting
  liquidity/stops).
- **Sweep**: a bar's high/low breaks beyond that level by at least a small buffer (avoids
  counting a marginal, noise-level poke as a real sweep).
- **Reversal confirmation**: within M bars of the sweep, price closes back on the OTHER side of
  the swept level (confirms rejection, not a genuine breakout that's still running).
- **Entry**: on reversal confirmation, in the reversal direction (sweep below support + close
  back above -> LONG; sweep above resistance + close back below -> SHORT).
- **Exit**: target = the opposite recent swing level (or a fraction of the prior range), stop =
  beyond the sweep's own extreme (if price re-breaks the sweep low/high, the "reversal" thesis is
  wrong), max holding time, forced flat at session close.
- **Parameter grid**: swing lookback window, sweep buffer size, confirmation window M, target
  fraction.
- **Cost**: same 6bps round-trip assumption as the other two strategies, for direct comparability.

## Status

Complete. See FINAL SYNTHESIS below.

## FINAL SYNTHESIS

**Recommendation: no edge found. Do not proceed to a live test on this strategy.**

All 16 parameter combinations (swing lookback x buffer x confirm window x target fraction),
across all 4 tickers and both regimes, are negative:

| Config | Avg ret/trade | Win rate | Trades |
|---|---|---|---|
| lookback=20, buffer=0.5x ATR, confirm=5, target=0.75x range (least bad) | -0.060% | 31.8% | 23,099 |
| lookback=60, buffer=0.5x ATR, confirm=3, target=0.75x range (worst) | -0.068% | 27.9% | 13,449 |

Zero of the 64 (config x ticker) cells checked were positive. Shorter swing lookback (20min)
consistently beat longer (60min) -- more, smaller-timeframe sweep setups outperformed fewer,
larger ones, though "outperformed" here still means less negative, not positive.

**A striking cross-strategy pattern worth stating plainly**: this liquidity-sweep result
(-0.060% to -0.068%/trade) lands in almost exactly the same range as VWAP mean reversion
(`scalp_research/`, -0.058% to -0.066%) and the XGBoost signal reversal rule
(`signals_research/`, -0.055% to -0.075%) -- three structurally unrelated strategies (a
classifier predicting direction, a price-vs-VWAP fade, a stop-hunt reversal), independently
designed and independently backtested, converging on nearly the same small negative edge, all
close to the 6bps round-trip cost assumption shared across the three. That convergence is itself
evidence, not a coincidence to shrug off: it's consistent with AAPL/MSFT/NVDA/SPY's short-term
(5-60 min) price action being close to a random walk once realistic costs are included -- exactly
what market efficiency predicts for four of the most liquid, heavily-analyzed names in the entire
market. A real edge, if one exists on these names at these horizons, is not sitting in public
price/volume data alone.

## Follow-up: higher-volatility universe check

Same unmodified engine, 3 focused configs, against TSLA/COIN/PLTR instead of the mega-cap set:

| Config | Overall avg ret/trade | Win rate |
|---|---|---|
| lb=20, buf=0.5, cw=5, tf=0.75 | -0.059% | 34.9% |
| lb=20, buf=0.5, cw=5, tf=0.5 | -0.059% | 40.1% |
| lb=20, buf=0.3, cw=5, tf=0.75 | -0.060% | 32.9% |

Still negative on every config, every ticker. Same conclusion as `scalp_research/`'s equivalent
check: this isn't a mega-cap-efficiency artifact, it generalizes.

---

# Full ICT Framework: Accumulation -> Consolidation -> Manipulation -> Pullback-to-POC -> Confirmation -> Entry

## Goal

Extends the sweep-reversal engine above into the fuller ICT sequence: instead of entering
immediately on reversal confirmation (the sweep-reversal test), this waits for price to pull back
to the session's volume-profile Point of Control (POC) after the initial reversal confirms, and
only enters on a rejection off the POC zone. "Accumulation" and "consolidation" are combined into
one gate (a tight-range filter relative to ATR) rather than modeled as two separate phases --
disclosed explicitly, not a hidden simplification.

## Data (real methodology deviation from the sweep-reversal test above, disclosed)

This account has no Alpaca market-data credentials, and the original 2.5-year Alpaca cache
(`scalp_research/bars_1min_cache/`) is gitignored and was never regenerated on this Mac. Data was
instead pulled fresh via IBKR (`fetch_minute_data_ibkr.py`): ~6 months of real 1-min RTH bars
(2026-03-17 -> 2026-09-14) for AAPL/MSFT/NVDA/SPY, ~48,500 bars/ticker. A materially shorter sample
than the sweep-reversal test's 2.5 years -- fewer regime cycles covered, taken as a real caveat on
how much weight this result should carry, not swept under the rug.

## Parameter grid

192 configs: lookback (20, 60) x range-tightness (1.5, 2.5, 5.0x ATR) x sweep buffer (0.3, 0.5x
ATR) x confirm window (3, 5) x pullback window (5, 10 bars) x pullback tolerance (0.15, 0.25x ATR
from POC) x target fraction (0.5, 0.75x range). Same 6bps round-trip cost assumption, same
regime-tagging (SPY realized vol, median split) as every other test in this file.

## Status

Complete. See FINAL SYNTHESIS below.

## FINAL SYNTHESIS

**Recommendation: no edge found. Do not proceed to a live test on this strategy.**

96 of the 192 configs cleared the n_trades >= 30 significance bar. **Zero of those 96 were
positive.** Best (least bad) and worst:

| Config | Avg ret/trade | Win rate | Trades |
|---|---|---|---|
| lookback=60, tightness=5.0x ATR, buf=0.3, confirm=5, pullback_window=5, pullback_tol=0.25, target=0.75x (least bad) | -0.038% | 37.4% | 107 |
| lookback=20, tightness=2.5x ATR, buf=0.5, confirm=3, pullback_window=5, pullback_tol=0.15, target=0.5x (worst) | -0.126% | 34.2% | 38 |

A pattern worth naming: the least-bad configs all cluster at `tightness=5.0` -- the loosest
consolidation filter tested, i.e. barely filtering at all (0-3% of bars flagged "consolidating").
Tighter filters (1.5x, 2.5x) that actually try to isolate real accumulation/consolidation ranges
made results *worse*, not better. That's the opposite of what the framework predicts and a real
signal that the added POC-pullback machinery isn't finding a cleaner setup -- it's just diluting
trade count until the average drifts back toward the same small negative number every other
strategy in this file lands on.

One ticker/regime slice (MSFT, this config, n=27) showed a small positive average (+0.0061%/trade)
-- noted for completeness, not treated as a finding: n=27 is below this file's own 30-trade bar,
and finding one positive slice out of 96 configs x 4 tickers x 2 regimes (768 slices) is exactly
the kind of result multiple-comparisons noise produces by chance, not evidence of a real MSFT-
specific edge.

**Net conclusion**: adding volume-profile POC computation and a pullback-confirmation step on top
of the already-negative sweep-reversal engine did not create an edge -- if anything the extra
machinery performed slightly worse at its best than the simpler sweep-reversal test's best
config (-0.038% vs -0.060%, but on a much smaller n=107 vs n=23,099, so not a strong comparison
either way). Combined with the sweep-reversal test, the VWAP mean-reversion test, and the XGBoost
classifier test all independently converging near the same small negative number, four
structurally different strategies now agree: no exploitable edge in this framework on
AAPL/MSFT/NVDA/SPY at 1-min-to-intraday horizons using price/volume data alone.

## Follow-up: Fair Value Gap (FVG) confluence

Added `require_fvg` to `poc_pullback_engine.py` (opt-in, default False so the run above is
bit-for-bit reproducible): requires a genuine 3-candle Fair Value Gap (standard ICT definition)
to have formed along the sweep -> pullback path, in the reversal direction, before an entry is
allowed. This was the one real ICT concept explicitly excluded from the original scope
(RESEARCH_LOG.md line 14-15) -- added on request to test it directly rather than leave it
untested.

Re-ran a focused 32-config slice (dropped range_tightness=1.5, which never produced a single
consolidating bar; fixed pullback_tolerance=0.25 and target_fraction=0.75, the values every
original top-10 config used) with require_fvg False and True paired for direct comparison.

**Result: FVG confluence helped, consistently, in the one region of the grid that was already
least-bad (lookback=60, tightness=5.0x ATR) -- and produced this project's first-ever positive
average return.**

| Config | fvg=False | fvg=True |
|---|---|---|
| lb60, tight5.0, buf0.3, pw5 | n=107, win 37.4%, avg -0.038% | n=54, win 46.3%, **avg +0.0148%** |
| lb60, tight5.0, buf0.5, pw5 | n=85, win 38.8%, avg -0.039% | n=45, win 44.4%, **avg +0.012%** |
| lb60, tight5.0, buf0.3, pw10 | n=122, win 38.5%, avg -0.043% | n=74, win 44.6%, avg -0.013% |
| lb60, tight5.0, buf0.5, pw10 | n=99, win 39.4%, avg -0.047% | n=65, win 43.1%, avg -0.021% |

All 4 of the lb60/tight5.0 configs improved with FVG required -- not cherry-picked from an
unrelated part of the grid, a consistent direction within the region that already looked best.
At lb20 (tighter lookback), the effect was smaller and inconsistent -- some configs improved,
none went positive, one got worse.

**Why this isn't a live-test signal yet, despite being positive:**

1. **Not distinguishable from noise at this n.** Per-trade spread on the best config was
   +0.68% (best trade) to -0.37% (worst trade) around a +0.0148% average, n=54 -- the standard
   error of that mean is on the order of several times the mean itself. This is consistent with
   zero.
2. **Entirely carried by 3 of 4 tickers, with SPY actively losing.** Ticker breakdown for the
   best config: MSFT +0.077%/trade (n=17, 64.7% win), NVDA +0.060% (n=11, 54.6% win), AAPL
   -0.018% (n=16, 43.8% win), **SPY -0.090% (n=10, 10.0% win)**. The second config's SPY slice is
   worse still: 0% win rate, n=7. A "positive average" built on SPY losing on 9 of 10 or 7 of 7
   trades is not a robust result -- it's one or two strong single-stock tickers outweighing a
   consistently bad index-ETF slice.
3. **Small sample from a 32-config search.** 2 of 32 configs landing barely positive, without
   correcting for having searched 32 of them, is weak evidence on its own -- the consistency
   across the 4 lb60/tight5.0 variants (point 1 above) is what makes this worth logging rather
   than dismissing outright, not the raw fact that 2 numbers turned positive.

**Recommendation: still no live test.** This is the most promising lead this framework has
produced, and it's real enough to be worth a proper follow-up (more history if fetchable via
IBKR in chunks, a wider ticker universe, explicit separation of SPY/index behavior from
single-stock behavior) -- but not real enough, on 45-54 trades with SPY dragging against the
mean, to act on as-is.

## Follow-up: same question, 12 tickers instead of 4 -- the positive result was noise

Ran the proper follow-up flagged above: widened from AAPL/MSFT/NVDA/SPY to 12 tickers (added
GOOGL/AMZN/TSLA/META/AMD/CRWD -- single stocks with the same liquid/momentum profile as MSFT/
NVDA, which had performed well -- plus QQQ/IWM alongside SPY, to test whether SPY's weakness was
SPY-specific or shared across index ETFs), narrowed the grid to just the one region that showed
promise (lookback=60, tightness=5.0x ATR -- the only non-degenerate setting at that lookback),
and swept pullback_window finer (3/5/7/10 instead of 5/10). 8 base configs x require_fvg
{False,True} = 16 configs, run on real IBKR 1-min bars (~6 months for the original 4, ~5 months
for the 8 new tickers, fetched 2026-09-15).

**Every one of the 16 configs is negative.** The earlier +0.0148%/+0.012% results do not survive
contact with a bigger sample -- best fvg=True config here: n=99, win 34.3%, avg **-0.0458%**.
Confirms this was a 4-ticker small-sample artifact, not a real edge.

What DOES hold up, consistently across all 8 config pairs: **FVG confluence still improves on
its own fvg=False baseline every single time** -- e.g. best config's baseline (fvg=False) was
n=191, win 33.0%, avg -0.0587%; requiring FVG moved it to n=99, win 34.3%, avg -0.0458%. Less bad
in all 8 pairs, never once worse. That direction is now a real, if small and still-negative,
finding backed by 3x more tickers than the original lead.

Also real: **single stocks consistently beat index ETFs**, with or without FVG. Best fvg=True
config split: single stocks n=77, win 37.7%, avg -0.040% vs index ETFs (SPY+QQQ+IWM) n=22, win
22.7%, avg -0.065%. FVG confluence helped single stocks in every config; it occasionally made
index ETFs *worse* (e.g. buf0.3/pw3: index win rate fell from 25.5% to 19.4% with FVG required).
Consistent with a real mechanism -- imbalances/gaps are a more meaningful concept on individual
names that can move idiosyncratically than on a diversified index product -- but still entirely
within negative territory, not a strategy.

**Final verdict on the full framework, FVG included: no edge, on any data this account can
access.** Don't pursue this further without the original Alpaca-sourced 2.5-year dataset --
smaller/shorter real data keeps producing different, contradictory shapes (this file now has
three separate reversals: sweep-reversal negative, POC-pullback flat-negative, FVG apparently-
positive-then-confirmed-negative), which is itself the finding: this framework's true signal, if
any, is too small to see clearly in anything less than years of data across dozens of names.

## Follow-up: bullish daily-trend filter on LONG setups

Added `require_bullish_trend` to `poc_pullback_engine.py` (opt-in, default False, same pattern as
`require_fvg` -- existing results unaffected). Deliberately asymmetric per request: only gates
LONG (down-sweep -> bounce) setups on whether yesterday's daily close was above yesterday's
50-day SMA of daily closes, both computed through yesterday only (no same-day lookahead). SHORT
setups are completely untouched -- no mirrored bearish-trend requirement was built. Gates at
setup INITIATION (a down-sweep is never watched at all if the trend isn't bullish yet), not at
entry.

Tested the same way as the FVG stress test: 8 base configs (lb60/tight5.0, buf 0.3/0.5, pw
3/5/7/10) x require_bullish_trend {False,True} = 16 configs, all 12 tickers.

**Result: worse in all 8 pairs, not better.** Overall avg return got more negative every single
time the gate was applied -- e.g. best baseline (buf0.5/pw3): n=191, avg -0.0587% -> gated: n=133,
avg -0.0692%. LONG-only avg return also got worse in all 8 pairs (occasionally with a slightly
*higher* win rate, e.g. buf0.5/pw3 LONG-only 34.65%->39.02%, but avg return still fell from
-0.0432% to -0.0550% -- fewer, not-better trades). SHORT-only numbers moved by only a trade or two
per config as expected (SHORT is ungated; the tiny shifts are a real, understood mechanical
side-effect -- the state machine only tracks one pending setup at a time, so suppressing some
DOWN_SWEEP watches frees a few bars for a fresh UP_SWEEP to be noticed instead, not a bug in the
gate itself).

**Conclusion: trading only with the broader daily trend does not rescue this framework -- it
makes an already-negative result worse.** Combined with the FVG finding (helps on the margin,
never flips positive) and the duration finding (scalping doesn't help either), the pattern across
every angle tested is now consistent: there is no free variable in this framework's own inputs
that turns the sign around. Whatever is wrong is more fundamental than any single filter can fix.

## Follow-up: real market structure (higher highs + higher lows) instead of an SMA

The SMA-based trend filter above isn't how ICT actually defines trend -- requested a proper
replacement: `compute_swing_structure()` in `poc_pullback_engine.py`, a standard 2-day fractal
identifying daily swing highs/lows, confirmed only once the following bars exist (no lookahead).
Bullish structure = the two most recently confirmed swing highs AND the two most recently
confirmed swing lows are both ascending -- real higher-highs-and-higher-lows, not a moving-average
proxy. Noticeably stricter than the SMA filter when checked directly (SPY: ~19% of days qualify
vs ~51% for the SMA version), since an actual ascending swing sequence is a harder bar to clear
than being above an average. LONG-only, same as the SMA filter; tested independently, not combined
with it or with FVG. Same 8-config x 12-ticker structure as every other follow-up here.

**Result: the strongest, most consistent effect of anything tested so far -- every one of the 8
pairs improved, both overall and LONG-only, several LONG-only configs landed near breakeven, and
one turned genuinely positive:**

| Config | Overall: no filter | Overall: structure gated | LONG-only: no filter | LONG-only: gated |
|---|---|---|---|---|
| buf0.5, pw3 (best) | n=191, avg -0.0587% | n=117, avg -0.0526% | n=101, avg -0.0432% | n=25, **avg +0.0316%** (44% win) |
| buf0.3, pw3 | n=267, avg -0.0776% | n=174, avg -0.0721% | n=128, avg -0.0661% | n=30, avg -0.0008% (37% win) |
| buf0.5, pw5 | n=227, avg -0.0642% | n=141, avg -0.0621% | n=120, avg -0.0509% | n=32, avg -0.0062% (41% win) |
| buf0.5, pw10 | n=259, avg -0.0668% | n=162, avg -0.0551% | n=139, avg -0.0592% | n=37, avg -0.0125% (41% win) |

All 8 LONG-only configs improved (not just these 4); SHORT-only stayed within a trade or two of
its unfiltered baseline in every config, confirming the gate is doing exactly what it's supposed
to and nothing else.

**Checked the best config's 25 LONG trades ticker-by-ticker before trusting it** (learned from the
FVG false-positive, where one ticker alone explained the whole effect): this one is genuinely more
distributed -- 10 of 12 tickers appear, and the winners (AAPL n=5 +0.178%, CRWD n=3 +0.311%, NVDA
n=1 +0.104%) aren't one or two trades propping up everything else. But **SPY is negative again**
(n=2, 0% win, -0.120% avg) -- the third straight follow-up (FVG, SMA trend, now structure) where
SPY specifically drags against the mean -- and AMZN is worse still (n=4, 0% win, -0.132% avg).

**Recommendation: still no live test, but this is the most credible lead this research effort has
produced.** More broadly distributed than the FVG result, a large and consistent effect size
across all 8 configs (not 2 of 32), and a real mechanism story (waiting for genuine trend
confirmation before taking a counter-move long). What it still lacks is scale -- n=25-37 per
config, 10-25 trades per ticker at most, over one ~6-month window. Same conclusion as everywhere
else in this file: the next real step is more history, not more filters, and this account's IBKR
data access tops out at what's already been pulled.

## Follow-up: real out-of-sample validation, 24 months of Alpaca data -- the lead did not survive

Alpaca API credentials were added to the account 2026-09-15, unlocking real history: fetched 2.5
years of 1-min bars for the same 12 tickers via `fetch_minute_data_alpaca.py` (474K-588K bars per
ticker, 2024-03-18 -> 2026-09-14, ~23s/ticker -- no pacing constraints, unlike IBKR). This finally
makes a genuine out-of-sample test possible: every result logged above (base framework, FVG, SMA
trend, HH/HL structure) was found AND checked on the exact same one 6-month IBKR window, with no
untouched data to confirm anything against.

Ran the 3 best-performing configs from the structure-filter test above against
`run_poc_structure_oos_validation.py`: real Alpaca data restricted to 2024-03-18 -> 2026-03-16,
the 24 months that do NOT overlap the IBKR window at all. No parameters were tuned on this data --
the filter's logic and every config tested were fixed before it was ever fetched.

**Result: the structure filter's positive result did not replicate.** The single config that
turned genuinely positive on 6 months of IBKR data (buf0.5/pw3, LONG-only: n=25, avg +0.0316%)
comes back solidly negative on 24 months of real Alpaca data, on a sample ~6x larger:

| Config | LONG-only, no structure gate | LONG-only, with structure gate |
|---|---|---|
| buf0.5, pw3 | n=434, win 34.3%, avg -0.0564% | n=145, win 31.7%, avg **-0.0537%** |
| buf0.5, pw7 | n=543, win 33.9%, avg -0.0657% | n=181, win 31.5%, avg **-0.0606%** |
| buf0.5, pw10 | n=578, win 33.9%, avg -0.0693% | n=191, win 31.4%, avg **-0.0649%** |

The *direction* of the effect (structure gate slightly better than no gate) technically still
holds in all 3 configs, but the magnitude shrank to near-nothing and every number stayed firmly
negative. Win rates also dropped across the board (31-34% here vs 33-44% on the original small
sample), consistent with the original 6-month IBKR window simply being a friendlier stretch for
this setup, not the filter having found something real.

**This is the most confident conclusion in this entire file, precisely because it's the first one
backed by a real sample size (434-633 trades per config) instead of a few dozen.** Every other
result here was found and validated on the same single window; this one wasn't, and it's the one
that broke. The honest read: the framework, with or without any of the filters tested (POC
pullback, FVG, SMA trend, HH/HL structure), has no real edge on these 12 tickers at these
horizons. Full stop on this line of research absent a fundamentally different idea, not just
another filter on the same mechanism -- four filters and 24 months of validation have now been
spent trying to rescue the same core setup.

## Follow-up: real ICT liquidity targets + killzone session filter, tested together

A methodology critique raised two real gaps, distinct from anything tested above: (1) every
sweep in every prior test targeted a generic rolling-intraday-lookback swing point, not the
specific pools a practitioner actually watches (prior day/week high/low, equal highs/lows), and
(2) every setup could initiate at any RTH hour, when real ICT practice restricts entries to
narrow session windows (London/NY opens, specific hourly blocks). Requested tested together, not
as another independent filter layered onto the same chain.

Built for real: `compute_prior_day_levels()` computes each session's actual Prior Day High/Low
from the prior COMPLETED session (verified exact match against real data before running anything
at scale); `apply_prior_day_liquidity()` substitutes PDH/PDL in for compute_levels()'s rolling
swing_high/swing_low everywhere downstream -- pure substitution, the consolidation gate and POC
computation (local price action, a different concept) are untouched. `apply_killzone_gate()`
restricts setup initiation to NY AM (09:30-11:00 ET) + NY PM (13:30-16:00 ET), gating BOTH
directions since session timing isn't directional. Tested as a real 4-way crossed comparison
(rolling vs PDH/PDL swing source x killzone off/on) on the FULL 2.5-year Alpaca dataset -- fair,
since neither idea was designed or tuned on any of this data, nothing to hold back.

**Result: both ideas made it worse individually, and combined they produced the worst result of
the entire research effort.**

| Swing source | Killzone | N (pw=10) | Win rate | Avg return |
|---|---|---|---|---|
| Rolling (baseline) | Off | 1,254 | 33.4% | -0.0812% |
| Rolling | On | 913 | 32.0% | -0.0829% |
| Real PDH/PDL | Off | 108 | 16.7% | -0.0924% |
| **Real PDH/PDL** | **On (both combined)** | **67** | **14.9%** | **-0.1153%** |

Killzone alone made the rolling-swing baseline slightly worse, not better. Real PDH/PDL alone
collapsed the win rate roughly in half (33% -> 17%) and dropped sample size ~12x (real liquidity
pools get swept far less often than a rolling intraday extreme recomputed every bar). Combined,
exactly as requested, produced the single worst avg-return/win-rate pairing logged anywhere in
this file.

**Why, mechanically**: sweeping the actual prior-day high/low is a materially bigger, more
decisive move than sweeping a small rolling-window local extreme. The data says this is more
often a genuine breakout/continuation past a real level than an exhaustion wick worth fading --
the opposite of the premise that targeting "real" ICT liquidity pools would produce cleaner
reversals. This engine trades reversals; real PDH/PDL sweeps apparently continue more than they
revert.

**This directly answers the "backtest isn't the full methodology" critique, and not in the
framework's favor.** The generic swing-point definition and all-hours entry window really were
real gaps versus practitioner ICT -- but closing both gaps didn't expose a hidden edge the
mechanical skeleton had been missing. It exposed that the skeleton performs worse, not better,
once pointed at the specific things ICT claims matter. Combined with stage 8's out-of-sample
reversal, there are now two independent, rigorous tests -- one on a bigger sample, one on the
actual concepts under dispute -- and both point the same direction: no edge here, mechanized or
not.

## Follow-up: cost stress-test, risk-adjusted metrics, regime check, capacity check

Five further gaps flagged in review: transaction costs untested against pessimism, no risk-
adjusted/distributional metrics (Sharpe, win/loss size ratio, drawdown), the whole 9-stage effort
being a multiple-comparisons problem, no regime segmentation, no capacity/execution realism.
Prioritized per request: cost stress-test first (cheapest, most decisive), then the rest from data
already computed rather than re-running further backtests.

**Cost stress-test (stage 7's IBKR result and stage 8's OOS result, LONG-only trades, raw returns
re-priced at a range of round-trip cost assumptions):**

| Cost | Stage 7 (IBKR, n=25) | Stage 8 OOS pw=3 (n=145) | Stage 8 OOS pw=10 (n=191) |
|---|---|---|---|
| 0bps (frictionless) | +0.0916% | +0.0063% | -0.0049% |
| 6bps (used throughout this file) | +0.0316% | -0.0537% | -0.0649% |
| 10bps | -0.0084% | -0.0937% | -0.1049% |
| 20bps | -0.1084% | -0.1937% | -0.2049% |

Stage 7's positive result flips negative between 6 and 10bps -- it was never robust to costs, a
separate and additional failure mode from the out-of-sample reversal. More importantly: **stage
8's out-of-sample result is negative (or barely positive, at pw=3) even at ZERO cost.** This
isn't a real edge that transaction costs ate -- gross of any friction at all, there is essentially
no edge to eat.

**Risk-adjusted metrics (same trades, at the 6bps assumption used throughout):**

| | Sharpe-like (mean/std) | Win/loss size ratio | Max drawdown |
|---|---|---|---|
| Stage 7 IBKR (n=25) | 0.120 | 1.78 | -1.04% |
| Stage 8 OOS (all 3 configs) | -0.29 to -0.32 | ~1.0 | -8.8% to -13.4% |

The small sample's favorable 1.78x win/loss size ratio -- the thing that made it look like a real
"let winners run" edge -- collapsed to roughly 1:1 out of sample, and drawdown scaled up 8-13x
deeper than the small sample suggested. Fails on a risk-adjusted basis independent of the raw
average return.

**Multiple comparisons**: real, and worth updating priors on, not just this branch. Stage 3->4 and
stage 7->8 are the same pattern twice -- scan many configs, find one that looks good, validate it,
watch it die. Two independent instances of that is evidence about this framework's parameter
space being dominated by noise generally, not two unrelated coincidences.

**Regime segmentation**: already computed and saved (stage 7's by_regime, combined LONG+SHORT):
HIGH_VOL -0.0533% vs LOW_VOL -0.0523% -- no meaningful regime dependency, uniformly weak across
both rather than a hidden edge in one subset.

**Capacity/execution**: CRWD's median 1-min volume (3,849 shares) is 3-7x thinner than every other
ticker in the universe, and it was one of the positive contributors to stage 7's result -- real
position sizing there would likely face more market impact than even the pessimistic cost
scenarios above modeled. Portfolio-level capital/overlapping-position constraints were never
modeled either (each ticker simulated independently) -- moot at this point given the edge doesn't
exist, but a real, disclosed gap.

**Net effect of this whole follow-up**: closes every remaining question about whether this
framework might still have a real edge hiding somewhere. It doesn't, at any cost assumption tried
including zero, on a risk-adjusted basis, across regimes, or before even considering capacity
constraints that would make things worse, not better.

## Follow-up: ES/NQ futures -- ICT's own home turf

Every prior stage tested equities/equity ETFs only. ICT methodology is most commonly taught and
practiced on ES/NQ futures specifically, given their near-24h session structure across
Asian/London/NY -- a real, disclosed gap in everything above. Fetched real 1-min ES/NQ bars via
IBKR (`fetch_minute_data_futures_ibkr.py`).

**Two real data-access constraints hit live, disclosed rather than smoothed over:**
1. IBKR's continuous-contract abstraction (ContFuture) rejects any historical request with an end
   date/time set (Error 10339) -- it can only return "now minus duration," not walk backward.
   Fixed by using the specific expiry contract instead, which does support backward chunking.
2. The June 2026 and March 2026 ES/NQ contracts had already been purged from IBKR's historical
   lookup ("No security definition found") -- only the current front-month (Sept 2026) contract
   was queryable. Actual sample: ~2.3 months (2026-07-08 -> 2026-09-15), not the originally
   planned 6, ~67,500 raw all-hours bars per symbol. A real ceiling on this account's data access,
   not a choice.

**Methodology limitation, disclosed plainly**: reused `_rth_only` (9:30-16:00 ET) unchanged --
day-session-only. Futures actually trade ~23h/day across Asian/London/NY sessions; testing that
properly would need simulate()'s day-boundary/EOD-flat logic redesigned around a real CME
settlement-to-settlement trading day, not attempted here given the added risk of a rushed
session-boundary rewrite. This means the specific overnight/London-open killzone behavior ICT
attributes much of its edge to on these instruments was NOT tested -- only whether the same
equity-tuned mechanical skeleton does anything during the NY day session on a futures product.

Grid re-opened rather than reusing equity-tuned settings: lb{20,60} x tight{2.5,5.0} x
buf{0.3,0.5} x pw{5,10} x require_bullish_structure{False,True} = 32 configs x 2 symbols (ES,
NQ), no lookback/tightness assumed to transfer from equities.

**Result: zero of 15 statistically meaningful configs positive.** Best: n=173, win 28.3%, avg
-0.0514%/trade. Same conclusion as every equity test.

Two things worth noting beyond the headline negative:
- **The HH/HL structure filter's direction of effect replicated on a completely different
  instrument class**: it improved the result in every single comparable buf/pw pair, same as the
  equity finding (stage 7) -- never flips positive, but consistently helps on the margin. Small,
  real, cross-asset evidence that this one concept isn't pure noise, even though it's nowhere near
  enough to trade.
- **Win rates here (12-28%) are meaningfully worse than every equity test (typically 30-40%)**,
  plausibly explained by the day-session-only limitation above -- this tested whether the
  mechanical skeleton works during NY hours on a futures product, not whether ICT's actual
  multi-session thesis (Asian/London sweeps feeding the NY session) has any edge, which remains
  genuinely untested.

**Verdict: still no live test, and this doesn't reopen the question.** ICT's own preferred
instrument class produces the same negative result as everything else in this file, on the
portion of its session structure that was tested. The untested overnight/global-session portion
is a real, acknowledged gap -- not a reason to expect a different answer, but the one remaining
piece of ICT's own methodology this whole effort never actually got to check.

## Follow-up: the real ~23h session -- Asian/London/NY, not just NY hours

Closed the gap stage 11 explicitly flagged. Required a real engine change, not just another
filter: `compute_levels()` gained `rth_only=False`, which skips the equity-hours filter and groups
bars into an actual CME trading session (18:00 ET one day -> 17:00 ET the next, the real
settlement-to-settlement day) instead of calendar-date or market-hours buckets. `simulate()` was
updated to use this session_date for day-boundary detection instead of raw calendar date --
necessary, not cosmetic: a session spanning midnight would otherwise get incorrectly treated as
two separate days mid-session (spurious forced-flat exit, spurious state reset). Verified directly
before running anything at scale: a bar at 23:59 ET and the bar one minute later at 00:00 both
land in the session that started the evening before, and the session correctly rolls only at the
real 17:00->16:00 ET maintenance break. Equity reproducibility re-verified unaffected (rth_only
defaults True, bit-for-bit identical n=14 check).

Tested 4 standard ICT global killzones (Asian 19:00-00:00, London 02:00-05:00, NY AM 08:00-11:00,
NY PM 13:30-16:00 ET) individually, combined, and against no restriction at all, crossed with the
structure filter. Core params fixed to stage 11's best config (lb20/tight5.0/buf0.5/pw10) -- a
focused test of the session-structure question, not a fresh grid search. Same ~2.3-month ES/NQ
sample as stage 11.

**Result: all 12 conditions negative. No exceptions.** But not uniform noise -- a real, coherent
pattern survived:

| Session | Structure filter | N | Win rate | Avg return |
|---|---|---|---|---|
| No restriction | Off / On | 1223 / 747 | 22.7% / 22.2% | -0.0605% / -0.0576% |
| Asian | Off / On | 290 / 176 | 18.6% / 14.8% | -0.0567% / -0.0594% |
| London | Off / On | 158 / 89 | 26.6% / 23.6% | -0.0504% / -0.0562% |
| **NY AM** | Off / **On** | 175 / **111** | 32.6% / **39.6%** | -0.0647% / **-0.0425%** |
| NY PM | Off / On | 135 / 82 | 25.2% / 25.6% | -0.0555% / -0.0551% |
| All 4 combined | Off / On | 758 / 458 | 24.7% / 24.5% | -0.0570% / -0.0539% |

Three real findings, not noise: (1) NY AM is clearly the best session -- highest win rate in the
entire futures effort (39.6%) and the structure filter's single biggest improvement anywhere,
consistent with stage 11's equity-hours-only test having implicitly tested this same window all
along; (2) Asian is the worst, not the best -- cuts directly against the intuitive assumption that
overnight liquidity sweeps would be the cleanest setups, and the structure filter actively hurts
there; (3) all-4-combined underperforms NY AM alone -- the 3 weaker sessions dilute the one that
actually carries signal rather than adding to it.

**Final verdict on the full ICT line of research, all 12 stages: no edge, anywhere, on any
instrument, any session, any filter, any dataset this account can access.** The session-structure
test closes the last explicitly-acknowledged gap and produces the most interesting non-result in
the whole effort -- a real, specific, coherent pattern (NY AM good, Asian bad) that still isn't
profitable. That combination -- genuine structure in the data, zero profitability -- is a more
convincing negative than uniform flatness would have been: it says the mechanism finds something
real about session timing, and that something still isn't enough to overcome the cost of trading
it.
