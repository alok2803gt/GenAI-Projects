# VWAP-Deviation Mean-Reversion Scalp — From-Scratch Strategy

## Goal

Separate from `signals_research/` (the XGBoost Signals-tab model, found to have no trading edge
and no basis for its CSP/LEAP scanner role). This is a genuinely different strategy, built and
tested from scratch, not a variant of that model: **intraday VWAP-deviation mean reversion** —
when price stretches too far from the session's volume-weighted average price, fade the move back
toward it. This is real, commonly-practiced short-term trading structure (not an ML classifier
guessing next-bar direction), chosen because it's structurally different from everything already
tested this session, not a re-parameterization of it.

## Constraints (same discipline as `signals_research/`)

- Does NOT modify `main.py` or any live trading process. Pure offline research, no capital at risk.
- Built entirely fresh — no imports from `signals_research/`'s model/backtest code.
- Real Alpaca 1-minute bars (finer than the 5-min data used in `signals_research/` — scalping
  means genuinely short holding periods, and 5-min bars are already too coarse for that).
- Every approximation (transaction costs, VWAP anchoring, stop/target logic) stated explicitly.
- Tested across a real parameter grid and multiple regimes, same as the prior work — one lucky
  parameter setting is not a strategy.
- Honest reporting regardless of outcome.

## Strategy design

- **VWAP**: session-anchored (resets each trading day), volume-weighted average of typical price
  ((high+low+close)/3) from the session open through the current bar.
- **Deviation band**: rolling standard deviation of (price − VWAP) over a lookback window, in
  units of that day's own price scale.
- **Entry**: price ≥ K standard deviations below VWAP → LONG (expect reversion up). Price ≥ K
  standard deviations above VWAP → SHORT (expect reversion down).
- **Exit**: reversion target (price returns to within a smaller band of VWAP, e.g. 0.3 std),
  OR a stop-loss if the deviation *grows* past a larger multiple (trend-continuation protection —
  the classic risk of fading a real breakout, not noise), OR a max holding time / forced flat at
  session close (no overnight scalp risk).
- **Parameter grid**: entry threshold K, stop-loss multiple, max holding bars — tested together,
  not just one setting.
- **Costs**: same 6bps round-trip assumption as `signals_research/`, for direct comparability.

## Status

Complete. See FINAL SYNTHESIS below.

## FINAL SYNTHESIS

**Recommendation: no edge found. Do not proceed to a live test on this strategy.**

All 24 parameter combinations (entry threshold K x stop multiple x max hold time), across all 4
tickers and both volatility regimes, are negative. Not most -- all. Best-to-worst:

| Config | Avg ret/trade | Win rate | Trades |
|---|---|---|---|
| k=1.5, stop=1.5x, hold=15min (least bad) | -0.058% | 27.5% | 17,464 |
| k=3.0, stop=2.0x, hold=60min (worst) | -0.066% | 42.4% | 9,979 |

Zero of the 96 (config x ticker) cells checked showed a positive average return -- not "mostly
negative," genuinely all of them. Same pattern as the threshold experiment in `signals_research/`:
win rate climbs toward 44% at wider stops/longer holds, but average return gets *worse*, not
better, because the losses that do happen get proportionally bigger. Tighter stops (get out fast)
consistently outperform looser ones (give it room) here -- the opposite of what "let winners run"
intuition would suggest, which is itself informative: it means the strategy isn't finding real
directional persistence to let run, just noise that a tight stop cuts before it compounds.

**Bug caught along the way, worth keeping visible**: the first version of this engine re-entered
the same losing trade on the very next bar after every stop-out (a "buy the falling knife
repeatedly" pathology), producing a 6.6% win rate with 96% of exits being stops on a smoke test.
Fixed with a re-entry cooldown (require the deviation to genuinely ease before re-arming that
side) before the real grid was run -- worth remembering for any future mean-reversion engine built
in this codebase.

**Not proceeding to any further tuning of this specific structure** -- the flatness across the
entire grid (all 24 cells within a ~0.01 percentage point band of each other) says this isn't a
parameter-selection problem, it's that VWAP-deviation mean reversion doesn't have edge on these 4
tickers at these timeframes, at any setting tested.

## Follow-up: does a different, higher-volatility universe change anything?

AAPL/MSFT/NVDA/SPY are among the most heavily-traded, closely-analyzed names in the market --
plausibly too efficient for this to ever have worked. Re-ran the same unmodified engine (3 focused
configs, not a fresh grid) against TSLA/COIN/PLTR (higher volatility, less mega-cap coverage):

| Config | Overall avg ret/trade | Win rate |
|---|---|---|
| k=1.5, stop=1.5, hold=15 | -0.060% | 33.3% |
| k=2.0, stop=1.5, hold=15 | -0.063% | 34.4% |
| k=2.5, stop=2.0, hold=30 | -0.073% | 43.2% |

Still negative, at every config, on every one of the 3 new tickers individually. If anything
slightly worse than the mega-cap result at the wider settings. **The negative finding is not
specific to mega-cap market efficiency** -- it holds on genuinely more volatile, more
retail-driven names too. That's a stronger, more general conclusion than the original result
alone would have supported.
