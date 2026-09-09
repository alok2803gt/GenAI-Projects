# Squeeze Momentum Indicator [LazyBear] -- Real Backtest

## Goal

User provided LazyBear's "Squeeze Momentum Indicator [LazyBear]" Pine Script
(the standard TradingView community port of John Carter's TTM Squeeze) and
asked for a real backtest against this account's 5-year universe. The Pine
Script itself is display-only (no entries/exits) -- this defines and tests
the one real, standard way this indicator is actually traded: enter in the
direction of momentum when the squeeze fires.

Does NOT modify main.py or any live trading process -- pure offline
research, zero capital at risk, same guarantee every other *_research/
directory in this codebase already gives.

## Universe

Reused `breakout_research/universe_5y_ohlcv.pkl` (112 tickers, real daily
OHLCV, 2021-08-25 to 2026-08-24 -- ~2 weeks stale as of today but immaterial
for a 5-year backtest, not re-fetched). Same universe already used for
breakout_scanner's own backtests.

## Indicator port (faithful to the Pine Script, stated precisely)

- BB: basis=SMA(close,20), dev=2.0*stdev(close,20) [population stdev, ddof=0,
  matching Pine's stdev()], upperBB/lowerBB = basis +/- dev.
- KC: ma=SMA(close,20), true range (Wilder's, max of high-low/|high-prevclose|/
  |low-prevclose|), rangema=SMA(TR,20), upperKC/lowerKC = ma +/- rangema*1.5.
- sqzOn = lowerBB>lowerKC AND upperBB<upperKC (BB inside KC -- compressed).
- sqzOff = lowerBB<lowerKC AND upperBB>upperKC (BB outside KC -- released).
- val = linreg(close - avg(avg(highest_high_20, lowest_low_20), sma_20), 20, 0)
  -- true OLS fit (numpy polyfit) over the last 20 bars, evaluated at the
  current bar, not an approximation.

## Trading rule (this account's own addition -- not in the original script)

Classic TTM Squeeze trade: squeeze was ON the prior bar and just turned OFF
(sqzOn[t-1] and sqzOff[t]) = entry signal, direction = sign(val[t]).
Entry executed at NEXT day's open (signal known at close of day t, no
lookahead). Exit at momentum reversal (val crosses back through zero) or a
max hold, whichever comes first -- tested at max_hold in {10, 20} trading
days. Tested LONG and SHORT setups separately (not netted), since equities'
long-term upward drift makes them a genuinely different bet.

## Status

Built and running -- results to follow.
