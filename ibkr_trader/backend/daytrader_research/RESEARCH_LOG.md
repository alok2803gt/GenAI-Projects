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

---

## 2026-09-29 — Day Trader gates re-derived from data; strategy DISABLED

**Trigger.** 2026-09-28 the scanner surfaced eight candidates above the score
threshold (BE 93.6, COHR 82.4, INTC 82.2, MRNA 81.5, AKAM 80.2, GEN 79.1,
DELL 78.0, ORCL 75.5) and the agent entered none. The question asked was
whether the scanner should run additional intraday passes.

**Additional passes would change nothing.** All four composite-score inputs
freeze at the open: `atr_pct` (ATR14 from completed bars), `gap_pct` (defined
once, at the open), `prior_day_ret_pct`, `ret5d_prior`. A second pass computes
identical percentile ranks on the same universe. `scan_delay_backtest.py`
separately found the confirmation edge decays ~10x from 0 to 7 minutes, so
later passes are worse, not neutral.

**The real blocker was the entry gates** (`day_trader_agent.py:1345-1350`),
a strict AND: `atr_mult >= 1.8` AND `std_score >= 3.7`, both read off the last
completed session since the 2026-09-23 forming-bar fix.

**Method** (`gate_threshold_study.py`, `confirmation_gate_study.py`): 112
tickers x 1233 sessions = 138,057 ticker-days, 2021-10..2026-09. No look-ahead,
market-adjusted vs the universe's same-day mean, date-clustered (one event per
session). Caveat: the original 500-ticker study script is not on this machine,
so percentile ranks come from a 112-name universe.

| population | rows | days | excess% | t | hit >=0.5% |
|---|---|---|---|---|---|
| all ticker-days | 138,057 | 1233 | 0.000 | — | 36.5% |
| score >= 75 | 23,502 | 1233 | +0.017 | 0.45 | 42.7% |
| + atr_mult >= 1.8 only | 932 | 491 | +0.174 | 1.23 | 43.0% |
| + std_score >= 3.7 only | 119 | 113 | −0.143 | −0.42 | 47.1% |
| + BOTH (live) | 62 | 60 | −0.293 | −0.56 | 46.8% |
| + EITHER | 989 | 520 | +0.153 | 1.13 | 43.3% |

Paired, same-day, passers minus the candidates each gate rejected:
BOTH −0.157pp (t=−0.31); EITHER +0.110pp (t=+0.81); atr_mult +0.168pp
(t=+1.21); std_score −0.227pp (t=−0.64). Nothing clears |t|>2, let alone the
|t|>3.1 Bonferroni bar for 19 comparisons.

- The **sigma gate has no support** and is directionally harmful: it fires on
  9.2% of sessions and picks worse than what it discards.
- The **AND conjunction is the worst** configuration and leaves only 4.9% of
  sessions with any passing candidate — the reason the agent is dormant.
- **Confirmation gate does not supply the sign.** Forward excess from the
  confirmed entry price is −0.004pp (t=−0.11) for score>=75, +0.208pp
  (t=1.49) with atr_mult. It also barely filters: 88.4% of candidates confirm.
  NB: "confirmed minus unconfirmed" shows +2.64pp t=+44.6 and is TAUTOLOGICAL
  (confirmation is defined by the same day's high) — kept in the script as a
  labelled warning, not a result.

**Decisive finding — cost, not thresholds.** `position_size_pct` (10) takes
precedence over `position_size` (`day_trader_agent.py:767-772`), so the live
position is 10% of net liq ~= $148. IBKR's $1.00 minimum binds, giving a fixed
$2.00 round trip = **1.35%**. The best measured edge is 0.174%. The fee is ~8x
the edge, and because the fee is fixed in dollars its percentage cost RISES as
the position shrinks.

| position | round trip | fee % |
|---|---|---|
| $148 (live) | $2.00 | 1.35% |
| $500 | $2.00 | 0.40% |
| $2,000 | $2.00 | 0.10% |

**Action taken 2026-09-29:** Day Trader `enabled=false` via `/day-trader/enable`
with zero open positions. Thresholds deliberately NOT retuned — loosening them
would only produce more trades losing 1.2% instead of 1.6%. Revisit when net
liq supports >= $2,000 positions (i.e. ~$20,000 at 10% sizing).

**One positive result worth keeping:** the composite score does lift the
>=0.5% MOVE hit rate 36.5% -> 42.7%, corroborating the original ATR-decile
finding. But the *signed* excess is +0.017pp (t=0.45) — it finds big movers,
not upward ones, and the agent is long-only. That gap, not the gates, is the
real weakness in the strategy's design.

### 2026-09-29 addendum — universe reduction tested after moving to Tiered pricing

Account switched to IBKR Tiered, cutting the per-order minimum $1.00 -> $0.35
and the hurdle at the live $148 position from 1.35% to **0.47pp**. Question:
does a tighter universe (higher score cut, fewer names/day, directional filter)
lift excess above that? `universe_reduction_study.py`, grid fixed before
looking: 5 score cuts x 5 top-N x 3 gap regimes = 75 tests, Bonferroni bar
|t| > 3.40.

**No subset clears the fee.** Best of all 75: gap DOWN < −1% with score >= 95,
**+0.319pp at t=1.74** (698 rows / 459 days) — still 0.15pp short of the 0.47pp
hurdle and less than half the corrected significance bar.

Excess does NOT rise with score: buckets 75-80 / 80-85 / 85-90 / 90-95 / 95+
give +0.037 / +0.011 / +0.010 / −0.013 / +0.167 pp. Concentration adds variance,
not expected return.

Economics even if the edge were real: break-even position is $219 (15% of net
liq); at a $300 position the best subset nets +0.086pp = **$0.26/trade**, about
140 trades/yr ~= **$36/yr**. Not worth the concentration on an unproven edge.

**Genuine finding, directionally consistent with the original 500-ticker study:**
gap regime is asymmetric and the sign is stable across every score cut.
gap DOWN < −1% is positive throughout (+0.024 to +0.319pp); gap UP > +1% is
NEGATIVE throughout (−0.014 to −0.081pp). The scanner ranks on |gap_pct| and so
deliberately surfaces both, which is actively wrong for a LONG-ONLY agent. If
the Day Trader is ever revived, restrict entries to gap-down reversion before
touching anything else. Day Trader remains disabled.

### 2026-09-29 — Day Trader revival PRE-REGISTERED and REJECTED out of sample

Intent was to revive the strategy, so the three identified defects were
corrected, the resulting rule was pre-registered and hashed BEFORE any holdout
data was downloaded, and it was tested once.

**Pre-registration:** `PREREG_daytrader_revival.md`,
sha256 `3071a656987d817f583a88aae7244b33e06f40fc32bf551b29abf289c7a572cc`,
locked 2026-09-29 09:21:42 ET. Hash re-verified at test time.

**Frozen rule:** `composite_score >= 75`, `atr_pct >= 2.5`,
`gap_pct in [-3%, 0%]` (excludes gap-ups AND falling knives), `atr_mult >= 1.4`,
**sigma gate removed**, long only, buy at open / sell at close.

**Holdout:** the 407 S&P 500 tickers NOT in the explored 112-name panel —
**497,820 ticker-days**, 2021-10-28..2026-09-28, downloaded after the hash was
taken. Percentile ranks and the market benchmark computed within the holdout
universe itself, so it is a self-contained replication. Identical feature code
(`build()` parameterised by panel path).

| | in-sample (contaminated) | **holdout** |
|---|---|---|
| excess | +0.203pp | **+0.045pp** |
| t (date-clustered) | 1.68 | **+0.72** |
| trades | 1,106 / 521 days | 3,841 / 914 days |
| hit >= 0.5% | 42.5% | 42.6% |

Both pass criteria FAILED: t = 0.72 (needed > 2.0), excess 0.045pp (needed
> 0.47pp, the Tiered fee at $148). The in-sample edge shrank ~78% out of
sample — the signature of selection noise, not a real effect.

**Correction to the earlier addendum the same day.** The claim "gap DOWN < −1%
is positive throughout (+0.024 to +0.319pp)" was accurate but misleading: the
band decomposition showed the gap-down region is NOT monotonic. Gaps worse than
−3% were the WORST band of all (−0.257pp, t=−1.64) and the only meaningfully
positive band was a modest −1% to −0.5% (+0.105pp, t=1.35). The "gap-down
reversion" story is much weaker than a cumulative cut suggested, and the
holdout then rejected it outright.

**One thing DID replicate:** the composite score's hit-rate lift. Baseline
same-day >=0.5% move is ~36.5%; the rule's candidates hit 42.6% in the holdout,
almost exactly the 42.7% seen in-sample. The score genuinely finds big movers.
It does not find UPWARD movers, which is what a long-only agent needs.

**Status: the holdout is SPENT.** No further tests were run against it, per the
pre-registration's rules of engagement. Day Trader remains disabled. Live gates
were NOT modified — the revived rule failed, so there is no evidence-backed
reason to touch them. Reviving this strategy now requires a genuinely new
hypothesis (e.g. trading the MOVE rather than the direction — a straddle-like
structure — or adding a directional signal the score does not contain), plus a
fresh holdout. It is not a tuning problem.

### 2026-09-29 — can BREAKOUT supply the Day Trader's missing direction? No.

Tested to the CEO's stated intent: not the 3d/5d swing horizon, but a same-day
momentum scalp exiting at a profit target ("out same day, breakeven or better").
`breakout_direction_study.py`, both panels, market-adjusted and date-clustered.

**The two scanners barely overlap.** Only **1.9%** (112 panel) / **1.5%** (407
panel) of Day Trader candidates had a BREAKOUT the previous session. They
select near-disjoint populations, so breakout cannot act as a filter on DT flow
at any useful volume.

**Where they do overlap, direction adds nothing.** Paired same-day,
breakout-yesterday minus no-breakout:

| exit | explored 112 | independent 407 |
|---|---|---|
| ride to close | −0.040pp (t −0.22) | +0.043pp (t +0.27) |
| scalp +0.5% | −0.051pp (t −0.49) | +0.019pp (t +0.36) |
| scalp +1.0% | +0.022pp (t +0.19) | −0.071pp (t −0.83) |

Signs flip between panels at every horizon. Noise. A softer filter
(pct_b(t-1) > 80) is likewise flat: −0.013pp and +0.038pp.

**The scalp exit itself is the bigger problem.** Capping the upside while
leaving the downside open is consistently NEGATIVE versus riding to the close:

| | ride to close | scalp +0.5% |
|---|---|---|
| explored 112 | +0.020pp (t 0.52) | **−0.037pp (t −2.97)** |
| independent 407 | +0.046pp (t 2.00) | **−0.012pp (t −1.50)** |

This is "cut the winners, keep the losers" in arithmetic form, and it is the
most consistent effect in the whole study. A +0.5% target truncates the right
tail that pays for the left. "Breakeven or better" as a same-day exit RULE
does not protect the account; on this candidate set it removes the expectancy.
(It worked on the 771P/766C option positions because those were about salvaging
a specific open trade, not a repeated entry rule.)

**The one statistically consistent finding is a NEGATIVE.** DT candidates whose
prior-session pct_b < 20 (bottom of the Bollinger band) underperform on every
exit and both panels: −0.077 (t −2.71), −0.085 (t −2.29), −0.057 (t −3.45),
−0.045 (t −2.02). Four of four negative, two past the Bonferroni bar. Buying
oversold high-volatility names at the open is reliably bad — the falling-knife
effect again. Useful as an EXCLUSION if the strategy ever runs; it does not
create a long edge.

Nothing tested reaches the 0.47pp fee hurdle. Day Trader remains disabled.

### 2026-09-29 — SEARCH for the missing direction: 11 features, 2 panels, none works

`direction_search.py`. Goal per CEO: same-day entry and exit, quick scalp. The
composite score predicts magnitude, not sign; this screened eleven SIGNED
features, all computable before the entry, for the missing direction. Quintiled
within each session, market-adjusted, date-clustered, on the DT's own candidate
population (score >= 75, atr_pct >= 2.5). Every result printed, nothing dropped.

Features: pct_b(t-1), close_qual(t-1), gap_pct, prior_day_ret, rs5, rs20,
dist_52w_high, sma20_dist, vol_ratio(t-1), range_pos_20d, overnight_streak.

**Primary filter was CONSISTENCY ACROSS TWO INDEPENDENT PANELS**, which is far
harder to fake than one t-stat. Result: **nothing survives same-sign plus
|t|>2 on both.** Five features agree in sign (prior_day_ret, rs5, rs20,
range_pos_20d, overnight_streak) but none is strong on both panels.

The nearest misses are three correlated "has been rising relative to its own
range / the market" features -- rs20, sma20_dist, range_pos_20d -- with Q5
t-stats of 2.32 / 2.41 / 2.65 on the 407 panel but only 1.20 / 1.18 / 1.12 on
the 112 panel. A weak, consistently-signed momentum tilt, not an edge.

**THE DECISIVE ARITHMETIC** (407 panel, DT candidates):

| | value | share of available |
|---|---|---|
| mean \|excess\| -- what PERFECT sign would earn | **1.726pp** | 100% |
| best Q5 excess across 11 features x 2 panels | 0.088pp | **5.1%** |
| fee hurdle at the live $148 position (Tiered) | 0.473pp | **27.4%** |

This is the real shape of the problem, and it is not "no opportunity". The move
is LARGE -- 1.73pp of dispersion sits there every day, which is why the
magnitude score works. Daily-bar features predict about **5%** of its sign.
Breaking even at $148 requires **27%**. Nothing short of a ~5x better
directional signal closes that gap at this position size.

Equivalently: the best observed signal (0.088pp) breaks even at a **$795**
position. So the requirement is a BETTER SIGNAL *and* a BIGGER POSITION, not
either alone.

**Where a real signal would have to come from.** Daily OHLC is exhausted -- 11
features, 2 panels, 5 quintiles = 110 cells and the maximum is 0.088pp. Same-day
direction is not in daily bars. Candidates that are, and what exists here:
  * opening-range behaviour (first 5-15 minutes) -- the scanner ALREADY computes
    `fetch_intraday_pct_b` on 15m bars but never sends it to the agent;
  * options flow -- Unusual Whales keys exist but the key returns 401 and needs
    regenerating (CEO action);
  * GEX/VEX positioning -- `gex_vex_history.jsonl` is already being collected;
  * 1-minute bars -- `ict_research/bars_1min_cache_*` covers only 5 tickers, so
    a universe-wide intraday study needs a real data pull first.

Day Trader stays disabled. No config changed. This is a screen: anything
pursued from it needs a fresh pre-registered holdout.

### 2026-09-29 — options flow (Unusual Whales): EXPLORATION found nothing; holdout NOT spent

UW key restored, 2,469 usable ticker-days pulled (top 5 DT candidates per
session, 2024-09-16..2026-09-28, ~2yr = the key's history limit). Two calls per
ticker-day: `net-prem-ticks` (per-minute net_call_premium / net_put_premium /
net_delta, `tape_time` UTC) and `ohlc/5m` for a real window-close entry price.

Pre-registration `PREREG_uw_flow_direction.md`, sha256
`67c5d03a444c13c43ac88e806d64e5e69c1661f4660c4a96fc3b383efa58af5d`, locked
10:54 ET before the pull finished. Split: exploration < 2025-09-16 (1,195 rows,
250 sessions), holdout >= 2025-09-16 (1,274 rows).

**Exploration, cross-sectional (3 fields x 3 windows), top tercile long:**
every |t| <= 1.14; `net_delta` NEGATIVE in all three windows (-0.139, -0.075,
-0.024pp) -- bullish flow predicting LOWER forward returns, i.e. backwards.

**Raw (non-demeaned) formulation**, tested because demeaning within only 5
names/day could erase a market-wide flow component: best is `ask_bid_imbal`
9:30-9:35 at +0.278pp, t=1.41. Still below the 0.473pp fee.

**Day-level timing** (aggregate flow -> that day's average candidate return):
correlations between -0.069 and +0.054. Essentially zero. Best t=1.44.

**SANITY CHECK — the pipeline is sound, which is what makes the null real:**

| relationship | corr |
|---|---|
| flow 9:30-9:45 vs **CONTEMPORANEOUS** 9:30->9:45 move | **+0.2900** |
| flow 9:30-9:45 vs **FORWARD** 9:45->close move | **-0.0134** |
| contemporaneous move vs forward move | +0.0073 |

Top-third flow: contemporaneous **+1.292%**, forward +0.218%.
Bottom-third: contemporaneous **-1.023%**, forward -0.092%.
Spread: contemporaneous **+2.315pp**, forward +0.310pp.

Options flow is COINCIDENT, not predictive. It explains the move that has
already happened -- strongly, corr +0.29 -- and carries almost nothing forward.
That the contemporaneous correlation is large proves timestamps, windows and
price alignment are all correct, so the forward null is a finding rather than a
plumbing failure.

**DECISION: the holdout was NOT spent.** The best exploration candidate is a
0.310pp forward spread, below the 0.473pp fee and at t~1.4. Spending a one-shot
holdout on that would waste it to confirm something the exploration set already
says is absent -- exactly what the pre-registration exists to prevent. The
holdout (1,274 rows, 2025-09-16 onward) remains SEALED for a better-motivated
hypothesis.

For the record, forward spread by source, against the 0.473pp fee at $148:
daily-bar features 0.088pp; options flow 0.310pp. Flow is ~3.5x better than
daily bars and still short of the fee, and not significant. It would become
interesting at a position size where the fee is ~0.14pp (~$500) AND with
evidence stronger than t=1.4 -- neither holds today.

### 2026-09-29 — CORRECTION: the "oversold is reliably bad" finding does not hold

Earlier today this log recorded, under the breakout-direction study, that DT
candidates with prior-session `pct_b < 20` "underperform on every exit and both
panels: -0.077 (t -2.71), -0.085 (t -2.29), -0.057 (t -3.45), -0.045 (t -2.02)
... four of four negative, two past the Bonferroni bar", and called it "the one
statistically consistent finding".

**That was wrong, and the error was mine.** Those four numbers all came from the
CAPPED-TARGET scalp exits (`scalp +0.5%`, `scalp +1.0%`). On the plain
ride-to-close exit the same population is flat:

| excluded population | 112 panel | 407 panel |
|---|---|---|
| pct_b(t-1) < 20 | -0.002 (t -0.03) | -0.001 (t -0.04) |
| gap < -3% | -0.244 (t -1.55) | **+0.192** (t +1.43) |
| gap > +1% | -0.032 (t -0.46) | **+0.032** (t +0.63) |

`pct_b < 20` is flat, not negative. `gap < -3%` and `gap > +1%` FLIP SIGN between
the two panels. The apparent "oversold is bad" effect existed only in
interaction with a target-capped exit that is itself structurally
negative-expectancy -- so it was an artefact of a broken exit rule, not an
independent directional signal. Reporting it as a robust finding was an
overstatement.

**Excluding all three changes nothing**, which is the cleanest proof they carry
no information:

| population | 112 excess / t | 407 excess / t |
|---|---|---|
| all DT candidates | +0.020 / 0.52 | +0.046 / 2.00 |
| drop pct_b < 20 | +0.009 / 0.23 | +0.048 / 1.97 |
| + drop gap < -3% | +0.027 / 0.65 | +0.044 / 1.86 |
| + drop gap > +1% | +0.021 / 0.45 | +0.041 / 1.49 |

Net of the 0.473pp fee: -0.452pp and -0.432pp. Filtering does not help.

**Standing conclusion after every test run today: there is NO reliable
directional signal for the same-day scalp.** Not in eleven daily-bar features,
not in BREAKOUT state, not in gap sign, not in Bollinger position, and not in
options flow. The single most robust directional FACT found is that options flow
is coincident, not predictive (corr +0.29 with the contemporaneous move, -0.013
with the forward move) -- i.e. at this frequency direction is already in the
price by the time any of these inputs is observable.
