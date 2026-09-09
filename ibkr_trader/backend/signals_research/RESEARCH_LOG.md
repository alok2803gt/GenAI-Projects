# Signals-Tab ML Model — Trading-Edge Backtest

## Goal

The IBKR trader frontend's "Signals" tab shows a live BUY/SELL/HOLD prediction (XGBoost
classifier, `main.py`'s `build_features()`/`train_model()`/`predict()`) for AAPL/MSFT/NVDA/SPY.
It has been display-only since it was built and has **never been backtested as a trading rule**.
The user wants to eventually trade off it. Before any execution code gets written, this directory
answers one question honestly: **does this signal have real, regime-robust edge as a trading
rule, and does the live "one shared model across all four tickers" architecture cost anything
relative to a proper per-ticker model?**

## Constraints

- **Does NOT modify `main.py`** or any live trading process. Pure offline research.
- **No live orders, no capital at risk.** All code here is standalone, reading cached historical
  data and simulating trades on paper.
- Reimplements `build_features()`/`train_model()`/`predict()` **verbatim** from `main.py`
  (same 9 `FEATURE_COLS`, same `XGBClassifier` hyperparameters, same `BUY_THRESHOLD=0.55` /
  `SELL_THRESHOLD=0.45`) — the goal is testing what's actually deployed, not inventing a better
  model.
- Data: ~2.5 years of 5-min bars for AAPL/MSFT/NVDA/SPY via Alpaca's `StockHistoricalDataClient`,
  following the one proven precedent in this codebase for multi-regime intraday history
  (`breakout_intraday_faithful_backtest.py`). IBKR's own per-day-pacing approach
  (`daytrader_research/`'s convention) only reaches 15-120 days — not enough for real regime
  coverage.
- Trade rule (per user decision, 2026-09-07): **reversal-style position tracking.** While flat or
  short, a BUY signal closes any short and opens long. While flat or long, a SELL closes any long
  and opens short. HOLD never changes position.
- Two variants tested side by side (per user decision, 2026-09-07): **as-deployed** (one shared
  model, retrained from a single reference ticker's own trailing window, applied to score all 4
  tickers) vs **per-ticker** (each ticker gets its own independently retrained model).
- Training-window grid: 5 trading days (matches live today's `HISTORY_DURATION="5 D"`), 20, and
  60 trading days — weekly walk-forward retrain cadence.
- Every approximation is stated explicitly in each script's own docstring, not just here.

## Files

- `fetch_data.py` — pulls and caches 2.5y of 5-min bars per ticker from Alpaca.
- `signal_model.py` — verbatim `build_features`/`train_model`/`predict` reimplementation.
- `backtest_engine.py` — walk-forward simulation: both variants x 3 window sizes, reversal
  position tracker, regime split.
- `run_backtest.py` — orchestrates the full grid, writes results.
- `report.html` — final visual summary (generated after results land).

## Status

Complete. See FINAL SYNTHESIS below.

## Data note — a real bug found and fixed mid-run

The first full run (2026-09-07, ~33 min) used Alpaca's default RAW (unadjusted) prices and
produced a phantom **+90% single-trade "return"** on NVDA, priced straight across NVDA's real
2024-06-07 10-for-1 stock split (~$1208 pre-split close vs ~$120 post-split price — the same
underlying value, zero real economic return, but a `close.shift()`-based feature/return
calculation reads it as a 90% price collapse). Caught by inspecting the actual trade record
before trusting the number, per this account's standing rule. Fixed by requesting
`Adjustment.ALL` (split + dividend adjusted) from Alpaca in `fetch_data.py`, confirmed the price
series is now continuous across the split date, and re-ran the entire grid from scratch (not just
the affected ticker) for full internal consistency. All numbers below are from the corrected run;
the qualitative conclusion did not change, but the corrected run is the one being reported.

## FINAL SYNTHESIS

**Recommendation: do not proceed to Phase B (live execution) on this signal, as-is.**

All 6 configurations (2 variants x 3 training-window sizes), across all 4 tickers, in both
volatility regimes, over the full 2.5-year window, lose money after realistic transaction costs
-- and have essentially **zero raw predictive edge even before costs**:

| Config | Trades | Win rate | Avg ret/trade | Implied pre-cost edge |
|---|---|---|---|---|
| shared, 5d window | 77,089 | 31.4% | -0.058% | +0.002% |
| shared, 20d window | 55,440 | 33.5% | -0.058% | +0.002% |
| shared, 60d window | 42,667 | 36.6% | -0.055% | +0.005% |
| per-ticker, 5d window | 92,215 | 29.6% | -0.057% | +0.003% |
| per-ticker, 20d window | 71,948 | 32.1% | -0.055% | +0.005% |
| per-ticker, 60d window | 49,090 | 34.7% | -0.055% | +0.005% |

("Implied pre-cost edge" = avg return + the 0.06% round-trip cost assumption -- i.e. what the
edge would be if trading were free. All six land under 0.005%, statistically indistinguishable
from zero across tens of thousands of trades.)

**Compare to Day Trader's own validated numbers** (this account's actual live momentum-breakout
strategy, real backtested before deployment): **+0.137%/trade, ~50.5% win rate.** Every
configuration tested here is not just worse than Day Trader -- it's negative where Day Trader is
positive, and its win rate is 14-20 points *below* Day Trader's, not just lower.

**What the parameter grid actually shows:**
- **Training-window size barely matters.** 60-day windows (11,000+ rows) are marginally less bad
  than 5-day windows (matching live production today) across both variants, but never come close
  to breakeven. The walk-forward CV accuracy logged at every single retrain point, across the
  entire 2.5-year run, sat in a **0.48-0.53 band regardless of window size** -- the model is not
  failing to learn from too little data; there is close to nothing in these 9 features that
  predicts next-5-min-bar direction, at any training-set size tested.
- **The shared-vs-per-ticker architecture question is resolved: it isn't the problem.** Going
  into this, the single-global-model design (`main.py:892-899`) looked like an obvious candidate
  culprit. Fixing it (per-ticker models) does not rescue the strategy -- per-ticker results are
  statistically indistinguishable from the shared-model results, sometimes marginally worse. The
  live architecture's real flaw doesn't matter here because the underlying signal has no edge to
  lose in the first place.
- **No regime helps.** HIGH_VOL and LOW_VOL (split on SPY's own rolling realized volatility) are
  both negative in every single configuration -- this isn't a strategy that "just needs the right
  market environment."
- **SPY is the worst performer in every configuration** (18-25% win rate vs. 32-40% for the other
  three) -- consistent with SPY being the most efficiently-priced, most heavily arbitraged
  instrument in the universe tested, leaving the least exploitable inefficiency for a generic
  technical-indicator classifier to find.
- **Turnover is extreme and itself informative.** 43,000-92,000 trades across 4 tickers over 2.5
  years (tens of thousands per ticker) is the reversal rule chasing the model's predicted
  probability oscillating around the 0.55/0.45 thresholds on pure noise, not tracking any real
  directional persistence. This is *why* transaction costs matter so much here even at a
  conservative 6bps round-trip assumption -- the strategy re-trades constantly on noise.

**What this does and doesn't say**: this result is specific to the exact thing tested -- these 9
features, this model class/hyperparameters, next-5-min-bar-direction as the target, and a
reversal-on-opposite-signal trade rule. It does not prove no signal could ever be built from
similar data; it says *this* signal, built *this* way, has no edge. A materially different
feature set, target horizon, or model would be a new research question, not a re-run of this one.

**Phase B is not being started.** Per the plan, live execution was explicitly gated on this
result. A negative result is being reported as such, not as a reason to lower the bar or tweak
parameters until something looks better -- that would be p-hacking a decision that already has a
clean answer.

## Follow-up: does the signal have any basis for its CSP/LEAP scanner role?

The Signals model already feeds two live scanners without ever being validated for that use: the
CSP scanner's composite score (`_stock_quality_score()`, weight 15) and the LEAP scanner's hard
filter (`if sig.get("label") == "SELL": return []`, `main.py:6816-6818`). These are a DIFFERENT
question from Phase A above -- not "does reversal-trading on this label make money" but "does the
label (or the separate quality-score heuristic) actually precede better/worse subsequent stock
performance," which is the entire statistical premise both scanner uses rest on.

**Important correction found while scoping this**: `_stock_quality_score()` does NOT consume the
XGBoost model's label or probability at all -- it's a separate hand-coded rule using only 3 of the
9 raw features (rsi, momentum, volatility) directly. So this splits into two independent
sub-questions.

### 2a. Does the XGBoost label predict forward returns? (`experiment_forward_returns.py`)

Walk-forward per-ticker (20-day window, same as Phase A), labeled every bar, measured actual
forward return at a CSP-like horizon (5 trading days) and a LEAP-like horizon (63 trading days),
bucketed by label:

| Ticker | CSP_5d: BUY vs SELL mean | LEAP_63d: BUY vs SELL mean | Consistent with "BUY > SELL"? |
|---|---|---|---|
| AAPL | 0.32% vs 0.23% | 3.62% vs 2.71% | Yes, both horizons |
| MSFT | 0.005% vs 0.15% | 1.09% vs 1.55% | **No -- reversed, both horizons** |
| NVDA | 0.70% vs 0.25% | 5.01% vs 5.36% | Yes short horizon, **no long horizon** |
| SPY | 0.20% vs 0.13% | 2.74% vs 1.83% | Yes, both horizons |

A quick 35-day/AAPL-only smoke test suggested a clean BUY-beats-SELL pattern -- exactly the kind
of thing that looks real on one ticker and falls apart on a proper check. It didn't hold: MSFT
shows the label pointing the *opposite* direction on both horizons, and NVDA flips depending on
horizon. If the label carried real information, the same relationship should show up consistently
across most of the 4 tickers, not in roughly half depending on ticker/horizon. **This is
indistinguishable from chance -- same conclusion as Phase A, reached a different way.** LEAP's
hard SELL-filter has no demonstrated statistical basis for excluding candidates.

### 2b. Does `_stock_quality_score()` predict forward returns? (`experiment_quality_score.py`)

Terciled each ticker's own quality-score distribution into LOW/MID/HIGH and compared forward
returns, same two horizons:

| Ticker | CSP_5d: LOW vs HIGH mean | LEAP_63d: LOW vs HIGH mean | HIGH actually better? |
|---|---|---|---|
| AAPL | 0.245% vs 0.273% | 3.10% vs 3.02% | Barely, one horizon only |
| MSFT | 0.110% vs 0.121% | 1.29% vs 1.45% | Barely, both horizons |
| NVDA | 0.452% vs 0.380% | 5.01% vs 4.66% | **No -- LOW beats HIGH, both horizons** |
| SPY | 0.152% vs 0.160% | 1.945% vs 1.904% | Mixed, negligible either way |

This is an even cleaner negative than 2a. Every bucket-to-bucket gap is within noise (a few
hundredths to tenths of a percentage point on returns with multi-percent standard deviations), and
where a real difference DOES show up (NVDA), the "HIGH quality" bucket does *worse*, the opposite
of what the score claims to identify. **Neither of the two things that reach into CSP/LEAP's
scoring has a demonstrated statistical basis.** This isn't "the model needs retuning" -- both
mechanisms currently amount to filtering/ranking real candidates on inputs indistinguishable from
noise.

## Follow-up: can the direct-trading signal be salvaged with a different design?

### Experiment A: longer prediction horizon (`experiment_horizon.py`)

Testing 1 (baseline)/3/6/12 bars ahead (5/15/30/60 min), per-ticker, 20-day window, same reversal
rule and cost assumption as Phase A.

| Horizon | Trades | Win rate | Avg ret/trade |
|---|---|---|---|
| 1 bar (5 min, baseline) | 71,948 | 32.1% | -0.055% |
| 3 bars (15 min) | 50,618 | 34.7% | -0.055% |
| 6 bars (30 min) | 37,925 | 37.0% | -0.052% |
| 12 bars (60 min) | 30,193 | 38.5% | -0.054% |

Win rate creeps up slightly at longer horizons, but average return stays flat and negative the
whole way -- **no horizon tested rescues this signal.** Longer horizon isn't the fix.

### Experiment B: tighter/wider probability thresholds (`experiment_thresholds.py`)

Testing (0.55,0.45) baseline / (0.60,0.40) / (0.65,0.35) / (0.70,0.30) at the original 1-bar
horizon, 20-day window:

| Thresholds | Trades | Win rate | Avg ret/trade | Worst single trade |
|---|---|---|---|---|
| 0.55 / 0.45 (baseline) | 71,948 | 32.1% | -0.055% | -9.3% |
| 0.60 / 0.40 | 35,328 | 37.5% | -0.056% | -10.9% |
| 0.65 / 0.35 | 16,618 | 42.1% | -0.054% | -9.9% |
| 0.70 / 0.30 | 7,408 | 46.9% | **-0.075%** | **-20.8%** |

Win rate climbs toward 50% as the bar gets stricter, but average return doesn't improve --
it gets *worse* at the tightest setting, because the far-fewer trades that do clear a 0.70/0.30
bar are less frequent but occasionally catastrophically wrong (a -20.8% single trade wasn't
possible at looser thresholds). **Trading only high-conviction signals doesn't fix anything --
being right more often didn't matter because the sizing of being wrong got worse.**

### Thread 1 conclusion

Neither lever tested -- longer horizon, tighter thresholds -- rescues this signal as a direct
trading rule. Combined with Phase A's finding that training-window size and shared-vs-per-ticker
architecture don't matter either, this closes out Thread 1: **this specific feature set, model,
and reversal trade rule has no salvageable edge along any dimension tested.** A genuinely
different data source or strategy structure (see `scalp_research/`) is a new question, not a
variant of this one.
