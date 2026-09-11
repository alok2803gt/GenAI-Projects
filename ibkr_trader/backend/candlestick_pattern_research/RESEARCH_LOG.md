# Bullish Harami -- Real Backtest and Live Scanner

## Context

Follow-on from squeeze_momentum_research/ (LazyBear's Squeeze Momentum
Indicator, six honest variants tested, none beat a real baseline). CEO
asked to check a different, classic candlestick pattern -- bullish harami
-- with the same rigor. Unlike squeeze momentum, this one held up.

## Pattern definition (standard, Nison-style)

- Prior bar: bearish (close < open), a "large" real body (>= its own
  trailing 60-day median body size -- filters out tiny/noise candles,
  matches the classic emphasis on a real, decisive down day).
- Current bar: bullish (close > open), body fully contained inside the
  prior bar's body: current_open > prior_close AND current_close < prior_open.
- Classic theory: only means something as a reversal signal in the context
  of an existing downtrend -- tested bare AND with a downtrend-context
  filter (close below both its own SMA20 and SMA50).

No lookahead: pattern completes at bar t's close, entry at bar t+1's open.
Same universe as squeeze_momentum_research/ (112 tickers, 5y daily,
breakout_research/universe_5y_ohlcv.pkl).

## Results (harami_backtest.py)

Bare pattern already beat the plain unconditional baseline at all 3 hold
periods tested (5/10/20 trading days). With the downtrend-context filter,
the lift got bigger at every hold period -- both metrics moving the same
direction is itself a good sign (a real effect showing a sensible
dose-response to the filter it should respond to, not a coincidental
one-off number).

## The critical check: matched baseline, not just "beats a random day"

Comparing harami+downtrend against a MATCHED baseline (any random day that
was ALSO already in a downtrend, no pattern required) -- the fair
apples-to-apples test, since "being oversold" alone has known
mean-reversion tendency that could otherwise get mistaken for the
pattern's own edge:

| hold | harami+downtrend mean | matched baseline mean | Welch p-value |
|---|---|---|---|
| 5d  | +1.073% (n=1858) | +0.486% (n=48298) | **0.00007** |
| 10d | +1.315% (n=1856) | +0.909% (n=48206) | 0.033 |
| 20d | +2.255% (n=1846) | +1.694% (n=47908) | 0.050 |

5-day is the strong, real result. 10/20-day are directionally consistent
but materially weaker.

## Per-ticker robustness (harami_per_ticker.py)

Same discipline as squeeze_momentum_research's per-ticker check (each
ticker vs ITS OWN matched baseline, Bonferroni correction for testing many
tickers at once, min 8 trades/ticker required):

- 0/111 tickers individually survive Bonferroni correction at any hold
  period -- same as squeeze. No single ticker's own sample is large enough
  to prove it alone.
- BUT the fraction of tickers showing ANY positive lift: 76/111 (68%) at
  5d, 64/111 (58%) at 10d, 63/111 (57%) at 20d. Binomial test on "is this
  fraction different from the 50% you'd expect under pure noise":
  - 5d:  p=0.000062 (very strong)
  - 10d: p=0.064 (marginal)
  - 20d: p=0.092 (not significant)

This is the real, meaningful difference from squeeze momentum: a broad,
statistically genuine directional tilt across most of the universe at the
5-day hold, even though no individual ticker's data alone is conclusive --
the textbook signature of a real, small, broad-based effect rather than a
few lucky names or pure noise.

## FINAL SYNTHESIS

**5-day hold, bullish harami + downtrend context, is a real, credible
finding** -- the first in this account's technical-indicator research
(squeeze momentum, 6 variants) to actually clear a properly matched
baseline with real statistical support at two independent levels
(aggregate mean-difference test AND per-ticker directional-consistency
test). 10-day and 20-day holds are directionally consistent but should be
treated as secondary, not independently actionable.

Not yet done: an out-of-sample split-period check (does it hold up testing
the first and second halves of the 5-year window separately) -- offered to
the CEO, not run yet since the scanner was prioritized first.

## Live deployment -- harami_scanner.py

Alert-only, one-shot daily (NOT a continuous poller -- the pattern only
confirms once the day's candle is final), registered as `IBKR-HaramiScanner`
weekday 4:10pm ET. Universe loaded fresh from the same backtest pickle
every run (a hand-typed first attempt at this list was already caught
wrong -- missing GLD/META, included 4 tickers never in the real universe).
Logic cross-validated against find_harami_signals() on 22 real historical
fired-days across 5 tickers -- 0 mismatches. Sends a real Telegram alert
with the actual OHLC levels and the backtested plan (enter next open, hold
5 trading days) for manual review -- places no orders, matching the CEO's
stated preference for this class of finding.

## Live forward-return scoring -- harami_daily_forward_performance.py (2026-09-10)

The scanner logs `harami_scanner_alert` to oversight_log.jsonl but never
scores its own signals. This script does: it parses every real fired
signal, replays it against real subsequent daily bars using the EXACT
entry/exit convention from harami_backtest.py (enter Open[i+1], exit
Close[i+1+hold]), and tracks 3/5/10-day-hold forward return vs two
baselines computed as-of each signal with no lookahead --
  - matched: that ticker's mean fwd return on trailing days it was also
    below both SMA20 and SMA50 (reproduces the backtest's +0.486%), and
  - plain: its unconditional fwd return over the same trailing window.
Incremental persist to harami_daily_performance_log.csv (gitignored),
recomputes only new/"pending" rows -- safe to re-run weekly, sample grows.
Math sanity-checked against hand calc (exact match). As of build: 3 live
signals (ISRG 9/9, LMT 9/10, RTX 9/10), all still mid-hold -- nothing to
conclude yet, by design.

## 15-minute-candle version -- NEGATIVE result (2026-09-10/11)

harami_15m_backtest.py: same pattern definition, same universe (112
tickers), same bare-vs-downtrend-context discipline as the daily-bar
version, run on real 1-year 15-min RTH bars (Polygon REST). "SMA20/SMA50"
here means 20/50 fifteen-minute bars, not 20/50 days.

| hold (bars) | downtrend mean | downtrend n | baseline mean | baseline n | p |
|---|---|---|---|---|---|
| 1 (15m)  | -0.008% | 11,619 | +0.004% | 725,370 | 0.053 |
| 2 (30m)  | -0.006% | 11,617 | +0.007% | 725,258 | 0.075 |
| 4 (1h)   | **-0.009%** | 11,608 | +0.015% | 725,034 | **0.016** |
| 8 (2h)   | +0.022% | 11,600 | +0.029% | 724,586 | 0.59 |
| 13 (~3.25h) | +0.047% | 11,593 | +0.048% | 724,026 | 0.98 |
| 26 (~1 RTH day) | +0.134% | 11,542 | +0.095% | 722,570 | 0.12 |

**No edge at any horizon.** This is a clean negative, not an underpowered
one -- every hold has n=11,500+ (well above the daily-bar version's
n=1,858), so a real effect this size would show up. The one nominally
significant result (1h hold, p=0.016) is in the WRONG direction: the
downtrend-context pattern actually underperforms its own matched baseline
there, not outperforms it. The ~1-day hold (26 bars) is directionally
consistent with the daily-bar finding (+0.134% vs +0.095% baseline) but
does not clear significance (p=0.12) despite the large sample -- read as
noise, not a weaker version of the real effect.

**Conclusion: the daily-bar harami+downtrend edge (RESEARCH_LOG.md above,
p=0.00007) does not translate to 15-minute intraday bars.** Whatever
produces the daily effect (overnight information processing, end-of-day
positioning, multi-session mean reversion) is specific to the daily
timeframe, not a fractal pattern that repeats at finer resolution. Do not
build a 15-min version of harami_scanner.py off this data.
