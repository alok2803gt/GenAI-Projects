# Sector/Sub-Sector Rotation Research

## Origin

CEO observation (2026-09-09): "there are days when hyperscaler run, then
there are days when memory stocks run, then there are days when chip
makers run" -- asked to validate whether this is a real, statistically
supported rotation pattern within the AI complex, or hindsight noise.

## Baskets (business-model grounded)

Corrected same day: STX/WDC are HDD makers, NOT memory-chip/HBM producers
-- kept as a separate "Storage" control group rather than lumped in with
real memory-chip makers (MU, SNDK), which is exactly the distinction the
CEO flagged in `sector_catalyst_scanner.py`'s existing live "Memory/Storage"
basket (`MU/WDC/STX/SNDK` all grouped together there -- worth revisiting
separately, see Open Items below).

| Basket | Tickers | Role |
|---|---|---|
| Hyperscalers | MSFT, GOOGL, AMZN, META, ORCL | AI infra BUYERS (capex spenders) |
| Chipmakers | NVDA, AMD, AVGO, MRVL, TSM | AI compute SELLERS |
| Memory | MU, SNDK | Real HBM/memory-chip makers |
| Storage (control) | STX, WDC | HDD makers -- NOT memory-chip makers |
| SemiCapEquip | AMAT, LRCX, KLAC | Tool makers, adjacent group |

Data: 5y daily OHLCV (yfinance), reusing `breakout_research/universe_5y_ohlcv.pkl`
where already cached, fetching the 4 missing tickers (TSM, STX, WDC, SNDK)
fresh. **SNDK only has ~1.5y of independent trading history** (spun off
from WDC ~Feb 2025) -- this is the binding constraint on every test
involving the Memory basket (n=382 days, not the full 5y).

## Finding 1 -- raw same-day correlation: NO zero-sum rotation

`rotation_analysis.py`. Same-day correlation between every basket pair is
**strongly positive** (+0.37 to +0.81) -- the opposite of what a true
zero-sum "money leaves A for B" rotation story would predict (that would
show negative correlation). Everything in the AI complex moves together
on a given day, just with different amplitude per basket.

Rotation/lead-lag test (pop day = basket return >=3%, Bonferroni-corrected
across 40 tests): 19/20 pairs significant SAME-day (confirms the shared-
beta finding above), but **0/20 pairs significant NEXT-day**. A big pop in
one basket does not predict another basket outperforming ITS OWN baseline
the next day. No detectable lagged catch-up at the basket level.

## Finding 2 -- relative rotation (common factor removed): REAL

`relative_rotation.py`. Raw levels are swamped by a shared "AI-theme beta"
-- the sharper test is each basket's return MINUS that day's 5-basket
equal-weight average (what's left after removing the common factor).

Same-day RELATIVE-return correlation flips substantially negative for the
key pairs:

| Pair | Raw corr | Relative corr |
|---|---|---|
| Hyperscalers vs Memory | +0.370 | **-0.710** |
| Chipmakers vs Memory | +0.639 | **-0.624** |
| Chipmakers vs Storage | +0.592 | **-0.528** |
| Hyperscalers vs Storage | +0.436 | -0.390 |
| Storage vs SemiCapEquip | +0.652 | -0.375 |
| Hyperscalers vs Chipmakers | +0.686 | +0.469 (stays positive -- these two co-move even relatively) |

**This is real, measurable structure, and it matches the CEO's observation
directly**: on days the mega-cap AI trade (Hyperscalers+Chipmakers) is
relatively hot, Memory relatively cools off, and vice versa -- almost
exactly -0.71 correlation for Hyperscalers-vs-Memory specifically. The
day-to-day sub-sector "leadership" the CEO is seeing is real, not
imagined -- it just only shows up once the shared beta is stripped out.

## Finding 3 -- but NOT tradeable on a next-day lag (yet)

Lag-1 autocorrelation of each basket's own relative return is near zero
for all 5 baskets (-0.04 to +0.11). Conditioning on "today's relative
leader" (top-tercile day) and testing next-day relative return against
that basket's own unconditional baseline: **0/5 significant** after
Bonferroni correction (p-values 0.20-0.86, no exceptions).

Honest read: the CROSS-SECTIONAL pattern (on any given day, one side of
the AI complex relatively leads while the other lags) is real and
statistically supported. But WHICH side leads on WHICH day does not
persist or mean-revert in a way this daily-bar test can detect -- it's
close to a coin flip day-to-day. This does not hand over a simple "basket
A led today, therefore trade basket B tomorrow" rule.

## Finding 2, stress-tested (`null_test_and_robustness.py`)

Finding 2's headline number (Hyperscalers vs Memory relative-return
correlation = -0.71) needed checking against three real ways it could be
an illusion rather than genuine rotation:

1. **Mechanical artifact**: with only 5 baskets, subtracting the daily
   cross-sectional average forces the 5 relative returns to sum to ~0
   every day -- that alone induces SOME negative correlation between
   differently-volatile series even with zero true relationship (a
   demeaning/compositional-data artifact). Monte Carlo null (5,000 sims,
   5 independent random series matched to each basket's real volatility,
   demeaned the same way): this mechanical effect alone produces a null
   mean of **-0.37** (std 0.044) -- so part of the raw -0.71 number IS
   inflated by the demeaning constraint itself, not pure signal. But the
   real -0.71 sits at the **0.00th percentile of that null (z=-7.77)** --
   nowhere close to something 5,000 pure-chance simulations ever produced.
   Net: real, additional negative co-movement beyond the mechanical floor,
   not an illusion of the method.
2. **Beta artifact**: Memory's beta to the common factor (1.62) really is
   much higher than Hyperscalers' (0.35) -- high-beta names mechanically
   overshoot on big shared-move days, which could masquerade as
   "rotation." Controlling for this (regressing out |common factor|
   magnitude) barely moves the number: -0.7072 vs -0.7096 raw. Survives
   -- not a beta-dispersion illusion.
3. **Outlier-driven**: Spearman (rank, outlier-robust) is -0.64 vs Pearson
   -0.71 -- close, not driven by a few extreme points. Dropping the 5
   highest-|common-factor| days only softens it to -0.69. Broad-based,
   not a few blowup days doing all the work.

All three checks confirm Finding 2 is real structure, not a statistical
artifact of the small-basket-count methodology.

## What actually drives it (`what_drives_it.py`)

Three candidate explanations tested for the validated Hyperscalers-vs-
Memory rotation:

**A. Risk-on/off sentiment (VIX) -- partially explains it.** Hyperscalers'
relative return correlates +0.29 with same-day VIX change; Memory's
correlates -0.24. The (Hyperscalers-minus-Memory) spread correlates +0.28
with VIX change. Real and coherent: on fear-spike days, money/relative
preference shifts toward mega-cap "quality" Hyperscalers and away from
smaller-cap, higher-beta Memory names -- a genuine flight-to-quality
effect. But r~0.28-0.29 is moderate, not dominant -- VIX alone doesn't
explain a -0.71 correlation. 10Y yield change shows ~zero relationship
either direction (-0.04 / +0.07) -- this isn't a rates/duration story.

**B. MU's own earnings calendar -- ruled out.** Only 2/20 of Memory's most
extreme relative-lead/lag days fall within 2 days of a real MU earnings
date, against a ~1.6/20 base rate if purely random (6 earnings dates /
382 trading days x 5-day window x 20 samples). Indistinguishable from
chance. The rotation is NOT primarily idiosyncratic single-company
earnings surprises -- it's something more continuous/pervasive, which
fits with Finding A being a daily sentiment factor rather than an event.

**C. Generic small-cap-vs-mega-cap size factor -- doesn't look like the
main story either, though the direct test here was flawed.** QQQ-vs-IWM
raw same-day correlation is +0.81 -- notably TIGHTER than Hyperscalers-
vs-Memory's raw +0.37. If this were purely a generic size-factor effect,
the AI-specific pair should co-move at least as tightly as the generic
mega/small pair, not less. (Caveat: the RELATIVE-correlation version of
this test with only 2 series -- QQQ, IWM -- is mechanically forced to
exactly -1.0 by the same demeaning-artifact logic Finding 2's stress test
exists to catch, so that half of the comparison is uninformative by
construction; only the raw-correlation comparison above is a clean read.)

**Net**: real risk-sentiment (VIX) component identified and quantified,
one hypothesis (earnings-driven) cleanly ruled out, size-factor looks
unlikely to be the primary driver -- but VIX alone leaves a substantial
unexplained residual. The most likely remaining candidate, not testable
with data on hand: memory-industry-specific supply/demand news (DRAM/NAND
contract pricing, channel checks, AI-memory-specific capacity/allocation
headlines) that moves Memory independent of both the broad market and
the hyperscaler capex narrative -- this would need a structured news/price
dataset (e.g. TrendForce-style pricing reports or a tagged headline feed)
rather than daily OHLCV to test properly.

## Round 2 -- macro/micro factors continued (`macro_micro_round2.py`)

**Ruled out, cleanly:**
- Dollar strength (UUP): correlations ~0.01-0.03 across the board. No relationship.
- Growth-vs-value factor (IWF-IWD spread): r=-0.13 vs the Hyperscalers-
  Memory spread. The rotation is genuinely distinct from the broad
  growth/value factor, not a repackaging of it.
- NVDA's own earnings calendar: 1/20 of the spread's most extreme days
  fall within 2 days of a real NVDA earnings date, vs a ~1.6/20 base rate
  if random. Same conclusion as MU's earnings test -- not event-driven.
- Broad semis cycle (SOXX, residualized against the AI-basket common
  factor): r=-0.295 vs Memory's relative return. If Memory were just
  riding the general semis cycle this should be positive -- it's
  negative, meaning Memory actually tends to diverge from SOXX (which is
  compute/logic-chip-heavy, closer to the Chipmakers side) rather than
  track it.

**The real driver, found: a global memory-industry factor, independent of
US AI sentiment.** EWY (iShares South Korea ETF -- home market of Samsung
and SK Hynix, the world's other 2 major HBM/DRAM producers) correlates
+0.498 with Memory's relative return and -0.633 with Hyperscalers'
relative return. Critically, this SURVIVES controlling for the shared
AI-basket common factor: MU's raw return and EWY's raw return, BOTH
residualized against the common factor first, still correlate +0.233.
That means MU and the Korean memory-producer market move together on
days that have NOTHING to do with the broader AI/market move that day --
real evidence of an industry-specific DRAM/HBM supply-demand cycle
(pricing, capacity, allocation news) that hits the whole global memory
industry together, separate from both the hyperscaler-capex narrative and
the broader compute-chip semis cycle.

**Caveat on tradability**: EWY is a US-listed ETF that trades during US
hours, not the actual Korean market index -- this same-day correlation is
real but has NOT yet been tested as a genuine pre-market LEADING signal
(e.g. does EWY's own overnight gap -- yesterday's US close to today's US
open, which would bake in the actual Korean overnight session -- predict
MU's full-day return). That's a distinct, not-yet-answered question from
"these two move together," and is the natural next step if this is meant
to inform same-day positioning rather than just explain the pattern after
the fact.

## Is the EWY relationship tradeable? (`ewy_leading_signal_test.py`)

Tested directly: does EWY's overnight gap (today's open vs yesterday's
close -- fully known by 9:30am ET) predict MU's INTRADAY return
(today's open -> today's close, i.e. what's left to happen AFTER you'd
see the EWY gap)? n=381 days.

1. **EWY overnight gap vs MU's own overnight gap: r=+0.72.** They already
   move together pre-market -- both react to the same overnight/global
   news roughly simultaneously. By the time EWY's gap is visible at the
   open, MU has already gapped similarly on its own.
2. **EWY overnight gap vs MU's intraday (open->close) return: r=+0.067,
   p=0.19 -- not significant.** Conditioning on EWY's top-quintile
   gap-up days (n=77) vs bottom-quintile gap-down days (n=77): neither
   group's MU intraday return differs meaningfully from MU's own
   unconditional baseline (p=0.73 and p=0.58). Once the day is underway,
   EWY's gap size carries no further predictive power over how MU trades
   the rest of that day.
3. **EWY's gap residual (whatever it contains beyond what MU's own gap
   already reflects) vs MU intraday: r=+0.050, p=0.33 -- still not
   significant.** EWY adds no incremental information beyond MU's own
   pre-market indication.

**Honest conclusion: the global-memory-industry driver is real (Round 2),
but NOT independently tradeable via watching EWY as a lagged signal.**
EWY and MU react to the same information at essentially the same time
(both gap together pre-market) rather than EWY leading MU with usable
lead time. This explains the mechanism behind the CEO's original
observation without handing over a new executable edge -- the two
findings answer different questions ("why does this happen" vs "can you
trade it") and this closes the second one honestly negative.

## Real historical options-flow data (`polygon_options_flow_backtest.py`)

UW's API only allows live/current-day queries (403 on any historical
`date` param, confirmed 2026-09-09 across `/darkpool/{ticker}` and
`/darkpool/{ticker}/price-levels`) -- and this account's own
`darkpool_activity_monitor.py` log only covers 2.5 weeks with ZERO MU hits
in that window. Neither could test whether real historical options
positioning distinguishes rotation days. Polygon's flat-files S3 bucket
(already used for real 0DTE options backtests elsewhere in this repo) has
genuine historical minute aggregates for the whole options market on any
past date -- used here to pull real MU+SNDK call/put volume for the top-10
Memory-leads days, top-10 Hyperscalers-leads (Memory-lags) days, and a
25-day random baseline, all from the actual `relative_rotation.py`
dataset.

**Volume surges on big-divergence days either direction -- real but not
deeply novel.** MU's real options volume has grown ~0.4%/day in log terms
over this window (r=0.79, p<0.0001) -- unsurprising given MU's own huge
real price appreciation over the period. Controlling for that trend:
mem_leads days run +34% above trend (t=4.14, p=0.0004), hyp_leads days
+25% above trend (t=2.69, p=0.016), both vs baseline's -27%. Real,
survives the trend adjustment, but close to a truism -- big relative
moves come with elevated trading interest regardless of direction, this
doesn't by itself explain WHY the rotation happens.

**The real, novel finding: an ASYMMETRIC options-positioning signature.**
Call skew (call volume / total volume) on hyp_leads days (Memory lagging)
drops to 0.507 vs the 25-day baseline's 0.553 -- real and significant
(t=-2.92, p=0.0067). But on mem_leads days (Memory leading), skew is
0.555 -- statistically indistinguishable from baseline (t=0.07, p=0.94).
Real defensive/hedging positioning (more puts) shows up in Memory's
options market specifically when it's underperforming the rotation --
there's no symmetric "extra bullish calls" signature when it's
outperforming. This is a genuine behavioral asymmetry, not just "options
market reacts to price," and it's the kind of signal that would need
real historical flow data (not price alone) to ever find.

## Bottom line

The CEO's observation is validated as a real phenomenon (relative
rotation between the mega-cap AI trade and Memory specifically, and to a
lesser extent Storage/SemiCapEquip), not hindsight pattern-matching -- but
it is currently a DESCRIPTIVE finding, not yet a trading signal. The
missing piece is almost certainly **what triggers a given day's relative
leader** (an earnings report, a specific guidance data point, a macro
print) rather than the rotation being self-sustaining/predictable from
price action alone -- a catalyst-conditioned version of this test (closer
to how `sector_catalyst_scanner.py` already works at the single-stock
level) is the natural next step, not a longer lookback on the same
daily-return approach.

## Open items

- `sector_catalyst_scanner.py`'s live "Memory/Storage" basket
  (`MU/WDC/STX/SNDK`, all 4 together) mixes the Memory and Storage groups
  this study kept separate. Given Finding 2 shows Memory and Storage
  aren't even the same side of the relative-rotation pattern (Memory vs
  Storage relative corr is only +0.067, near zero -- they don't move
  together once the common factor is removed), the live basket's original
  4-stock backtest may be diluted by mixing two groups with no real
  co-movement. Worth a dedicated re-backtest split into `{MU, SNDK}` vs
  `{STX, WDC}` separately -- flagged in the prior conversation, not yet
  done.
- A catalyst-conditioned version of this test (same-day relative leader
  vs whether a REAL scheduled catalyst -- earnings, Fed data, a specific
  company news item -- landed that day) is the natural next step to find
  out whether relative rotation is predictable EX-ANTE (before the catalyst
  lands) or only explainable EX-POST (after the fact).
- Equal-weight, not market-cap-weight, baskets throughout -- a cap-weighted
  version (NVDA/AVGO dominating Chipmakers, MSFT/AMZN dominating
  Hyperscalers) might show a cleaner or different pattern; not yet tested.
