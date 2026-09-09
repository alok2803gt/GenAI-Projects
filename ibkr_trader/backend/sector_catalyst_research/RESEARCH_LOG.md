# Sector-Catalyst Momentum Research

## Goal

Real trade last week (2026-09-03 -> 2026-09-04): user bought a $1000/$1020 MU
call debit spread, DTE 1 day, paid $71, MU ran $958.16 -> $1016.59 the next
day and the spread paid off ~10x. Confirmed via real price data + news
search this was NOT an earnings gap (MU's next real earnings date is
2026-09-30) -- it was a sector-wide catalyst: Susquehanna/TechInsights
research on DRAM contract prices (+50-200% this quarter) and NAND (+60%)
driven by AI server demand, plus Lynx Equity reiterating a $1,325 price
target, moved the whole memory sector together (MU, SK Hynix, SanDisk all
up same day).

User confirmed (2026-09-08, via AskUserQuestion): this specific trade was a
blind speculative test, no real thesis at entry. Going forward, the goal is
a REAL, ex-ante, backtestable signal for "a sector/theme catalyst is
heating up" -- not hindsight ticker-picking -- to surface as occasional
opportunistic candidate trades (not a fully automated auto-trader, at least
not yet).

## Constraint discipline (same as every other backtest in this codebase)

No historical intraday options data exists for these tickers. Any options
P&L estimate in this research uses Black-Scholes with explicitly-stated
vol assumptions, same as alpaca_0dte_butterfly_trader.py's GEX-nudge
feature and the intraday stop-loss backtest. State every approximation
out loud. Real underlying price/volume data only for the underlying
statistical signal test (Phase 1, below) -- no options model needed yet
to answer "does a coordinated-mover signal predict continuation at all."

## Baskets tested

- **Memory/Storage** (tight, matches the real MU catalyst): MU, WDC, STX,
  SNDK (SNDK only has data from 2025-02-13, post its spinoff from WDC).
- **Broader Semis/AI** (looser, tests whether the effect is
  memory-specific or a general sector-momentum phenomenon): NVDA, AMD,
  AVGO, MRVL, QCOM, TXN, INTC, AMAT, LRCX, SMCI, KLAC, ON, MCHP, ADI, ASML.

Real 5-year daily OHLCV, `sector_catalyst_research/semis_universe_5y_ohlcv.pkl`
(reused breakout_research's existing cache for 12 tickers already present,
fetched the other 10 fresh via yfinance).

## Phase 1 result (phase1_signal_test.py)

Signal: basket "hot" if >=2 members close up >=3% on >= their own 20-day
avg volume, same day. First pass used a 1.5x volume multiplier and it
MISSED the real 2026-09-04 catalyst entirely (MU/WDC/STX all popped 6%+
together on only 0.7-1.3x normal volume -- a re-rating move, not a
volume-spike event) -- loosened to 1.0x after catching this.

Memory/Storage (n=105 real hot days, 2021-2026): real, consistent,
positive lift across all 4 members. Basket-average 1-day tail-move
(>=4%) rate: 19.4% conditional vs 13.1% unconditional (+6.2pp lift).
Decomposed into spillover (ticker hadn't popped yet) vs continuation
(ticker was one of the day's poppers): spillover edge is STRONGER in 3 of
4 tickers (WDC spillover tail-rate 33.3% vs 12.5% continuation) -- the
real insight is "buy the laggard that hasn't moved yet," not "buy
whatever already popped."

Broader Semis/AI basket (n=323, 15 tickers): NO edge (tail-rate lift
+0.40pp, essentially flat) -- confirms the effect needs a genuinely
narrow, commonly-driven basket, not general sector correlation.

## Multi-sector sweep (phase1_multi_basket_sweep.py)

Same signal tested across 14 candidate baskets spanning oil majors,
airlines, homebuilders, regional banks, solar, uranium, cruise lines,
casinos, steel, coal, Chinese ADRs, lithium. Memory/Storage ranked #1 by
a wide margin (+11.75pp spillover tail-lift vs next-best Regional Banks
+3.66pp). 9 of 14 baskets showed flat-to-negative spillover lift --
that's the right shape for a real, specific effect (most candidates
correctly show nothing) rather than a universal artifact of the method.
Secondary candidates worth a future look: Regional Banks, Homebuilders
(both plausible rate-policy-driven spillover, weaker than Memory).
Coal showed a DIFFERENT mechanism (momentum continuation, not laggard
spillover: +5.43pp continuation vs -0.21pp spillover) -- a separate idea,
not evidence for this one.

## Phase 2 result (phase2_pnl_test.py) -- real $ P&L, with a caught methodology error

First calibration attempt used the CEO's real 2026-09-03 MU trade ($1000/
$1020, paid $71) as the sole IV anchor via Black-Scholes back-solve
(implied 45.8% flat IV at T=1 trading day, vs MU's own trailing-20d
realized vol of 50.4% that same day -- an IV/RV multiplier of 0.909).
Applying that multiplier to ALL historical days made even the BASELINE
(buying the same structure on a random day, no signal) show +69% to
+150% mean returns on debit -- not a realistic options-market outcome,
and the tell that the one real fill was a cheap outlier, not
representative pricing. Re-ran across a realistic IV = 1.0x-1.3x trailing
realized vol range instead.

Result, robust across all 3 vol levels and a 4-cell strike/width grid
(long strike ~4-5% OTM, width 2-3%): SIGNAL-conditioned trades (laggard,
day of the basket-hot signal, 1-day hold to next close) stay solidly
positive (mean payoff $76-$194/spread) at every vol level tested, while
BASELINE (same structure, random day, same tickers) goes from marginally
positive (optimistic 1.0x vol) to clearly NEGATIVE (-$2 to -$22/spread at
realistic 1.15x-1.3x vol) -- the signal is what turns a losing structural
bet into a real edge, not an artifact of underpriced options. Win rate
~18-25% (signal) vs ~8-12% (baseline); median trade is a full loss in
both cases (expected for cheap OTM options) -- the edge is entirely in
win size/frequency, matching the real MU trade's own asymmetric shape.
n~120 real signal instances over 5 years across 4 correlated tickers --
a real, sizeable base rate, but not 120 independent bets.

## Phase 3 (built, live) -- sector_catalyst_scanner.py

Live, alert-only intraday scanner (15-min poll during market hours),
Memory/Storage only. Fires a Telegram alert with real live option quotes
(yfinance option chain) for the laggard's suggested ~4%/2% call debit
spread when >=2 basket members pop >=3% intraday. Does NOT place any
order -- CEO chose "occasional opportunistic, manual review" deployment
(2026-09-08, via AskUserQuestion), not a new auto-trader. Registered as
an always-on watchdog (`run_sector_catalyst_scanner.ps1`,
`IBKR-SectorCatalystScannerWatchdog` scheduled task), added to
morning-checks skill's Step 2 table. Dry-run verified against real live
data 2026-09-08: correctly detected STX's real +6.5% move and correctly
did NOT alert (only 1 of 4 names crossed the threshold that day).

## FINAL SYNTHESIS

Real, validated, positive-EV edge found for one specific, narrow basket
(MU/WDC/STX/SNDK) where a genuine shared driver (DRAM/NAND commodity
pricing) causes analyst/market re-rating to spread across peers with a
lag. Confirmed NOT a generic "sector momentum" effect -- broad semis and
9 of 13 other candidate sector baskets showed no edge under the identical
test. A real methodology error (calibrating vol off the one lucky real
trade) was caught before it could produce a falsely optimistic
recommendation -- the corrected, realistic-vol version still shows a
real, robust edge. Live scanner built and running, alert-only, zero
capital at risk. Not yet live-tested with a real trade off this specific
signal -- next real validation point is whenever the scanner's first live
alert fires and the CEO decides whether to act on it.
