# MU "Buy the Dip, Sell the Run" -- Real Findings

## Origin

CEO instruction (2026-09-09): "people are making money buying dip selling
run on MU, find out how" -- explicit instruction to keep digging until a
real, validated mechanism is found, not to stop at a shallow answer.

## Test 1 -- naive directional price timing: NOT supported

`mu_dip_run_backtest.py`. Real 5y MU daily data (2021-08-25 to
2026-08-24, price $48.88 -> $1213.56, +1130% over the window). Tested a
REAL parameter grid, not one cherry-picked setting: N-day return
thresholds (1/2/3/5-day windows x 3%/5%/8% thresholds), RSI(14)
oversold/overbought, Z-score vs SMA20 (1.0/1.5/2.0 std dev) -- each
against 5 hold periods (1/3/5/10/20 days), always vs MU's OWN
unconditional forward return at that same hold period (the baseline that
nets out the secular uptrend), Bonferroni-corrected across all 160
signal/hold/direction combinations (alpha=0.000313).

**0/80 dip-buy signals significant. 1/80 run-sell signals significant --
and it's the WRONG SIGN.** RSI>70 (overbought) at a 20-day hold showed
MU's forward return at +12.0% vs baseline's +5.5% (p<0.001) -- MU kept
RUNNING harder after becoming "overbought," the opposite of what a
mean-reversion sell signal needs. Simple technical dip/run timing is not
a real, exploitable edge on MU by any of these standard definitions.

## Test 2 -- volatility/options-premium harvesting: real signal, but not a clean strategy

Hypothesis: professionals aren't betting on price reversal at all --
they're harvesting elevated implied vol after a big move (either
direction), which mean-reverts even when price keeps trending (fully
consistent with Test 1's momentum finding).

`mu_vol_harvest_backtest.py`. Real historical MU options data via
Polygon's flat-files S3 bucket (`us_options_opra/minute_aggs_v1/`) --
this account has genuine historical access here, unlike Unusual Whales
(confirmed 403 on any historical date query for this account's tier).
For 10 real single-day dip days (<=-5%), 10 real run days (>=+5%), and 15
random baseline days (all 2025+, for comparable price scale), found the
real, actually-traded near-ATM straddle (25-50 DTE) on the trigger day,
computed real Black-Scholes-implied vol from the real observed option
prices, then tracked the SAME contracts 10 trading days later.

**Raw dollar P&L is misleading here** -- MU's price moved another ~10x
just within 2025-2026, so a straddle's dollar P&L isn't comparable across
dates without normalizing. Properly normalized (P&L as % of premium
collected) and checked both parametrically and with outlier-robust
Mann-Whitney: **no significant short-straddle P&L edge** for dip (p=0.90)
or run (p=0.37) days vs baseline. One massive real loss (2026-05-15 dip
day: MU kept falling AND vol kept rising, -113% of premium collected) by
itself would have wiped out a naive short-straddle account -- this is
real, uncapped tail risk, not a clean edge.

**But a real, narrower signal survives**: MU's implied vol crushes
significantly MORE after a dip specifically than after a normal day, even
after controlling for the (weak, not itself significant) overall IV-level
time trend across the sample: dip days average +6.4% trend-adjusted IV
crush vs baseline's -4.7% (t=2.39, p=0.028; Mann-Whitney p=0.051,
borderline). Run days show ~0% trend-adjusted crush (p=0.28) --
statistically indistinguishable from a normal day. This mirrors the exact
same DOWNSIDE-SPECIFIC asymmetry already found in
`sector_rotation_research/` (real put-buildup on Memory's underperformance
days, no symmetric call-buildup on outperformance days) -- a recurring,
real pattern in how this name's options market behaves, not a one-off.

## The honest answer

**The dominant, real explanation is survivorship in an extraordinary
trend, not timing skill.** MU is up +1130% over this window. In a move
that large, MU's own UNCONDITIONAL forward return is strongly positive at
every hold period tested (e.g. +5.46% over 20 days, no conditioning at
all) -- simply being long MU at almost any point worked out. Dip-buyers
who made money on MU mostly made money because they were long a
generational winner, not because their specific dip-timing added
provable value beyond that -- Test 1 found their entries were
statistically indistinguishable from a random entry.

**The one real, secondary mechanism found: an asymmetric IV crush after
downside moves specifically** (real, survives trend-adjustment, borderline
p=0.03-0.05) -- but it doesn't translate into a clean, reliably
profitable premium-selling STRATEGY on its own once real gamma/price-path
risk is included (one bad tail event erases the average edge in a naked
short-straddle). A defined-risk structure (iron condor/butterfly instead
of a naked straddle, or active delta-hedging) would be needed to actually
harvest this real-but-modest vol-crush tendency without the uncapped
downside that sank the naive version here -- not yet tested.

**"Selling the run" is most plausibly ordinary profit-taking/risk
management** (trimming a winner into strength), not a validated
short-selling or premium-collection edge -- Test 1 found no reversal
signal to sell against, and Test 2 found no differentiated vol-crush
advantage specifically on run days either.

## What's not yet tested

- A defined-risk structure (iron condor/put credit spread after a dip,
  call credit spread after a run) instead of a naked straddle, to see if
  the real IV-crush-after-dips signal survives with capped tail risk.
- Intraday mean reversion -- this account's Polygon subscription does
  NOT include `us_stocks_sip` (confirmed 403 on both minute and day
  aggregates), so a genuine multi-year intraday test isn't currently
  possible with data this account has access to.
- A larger sample (n=9-15 per group here is real but thin) -- the vol-
  crush finding is directionally consistent and survives real rigor
  checks, but a bigger sample would sharpen the p=0.03-0.05 range into
  something more conclusive either way.
