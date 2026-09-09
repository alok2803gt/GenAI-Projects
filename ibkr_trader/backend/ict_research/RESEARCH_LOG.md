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
