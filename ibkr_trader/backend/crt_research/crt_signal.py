"""
Shared, reusable Candle Range Theory (CRT) signal computation -- validated
2026-09-11/12 via crt_spy_0dte_backtest.py (2y real SPY 5-min bars, robust
across a 10-25 min sweep-window neighborhood, p<0.05 at every point tested,
p=0.0014 at the 15-min window used here).

Definition (see crt_spy_0dte_backtest.py's docstring for the full backtest
methodology):
  - Reference range = prior real trading day's RTH high/low.
  - Sweep = within the first SWEEP_WINDOW_MIN minutes of today's RTH
    session, price trades beyond the reference high or low by at least
    SWEEP_MIN_TICKS.
  - Reclaim = within RECLAIM_WINDOW_MIN minutes of the sweep bar, a 5-min
    bar closes back inside the reference range.

Live usage: call get_crt_signal(ib, ticker) any time after
9:30+SWEEP_WINDOW_MIN+RECLAIM_WINDOW_MIN ET (i.e. after ~10:15 ET for the
validated 15/30 parameters) to get today's real, live CRT status. Returns
None (not "no signal") if called before enough of the session has printed
to make a real determination -- callers must not treat None as "no CRT."
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ib_insync import IB, Stock, util

SWEEP_WINDOW_MIN = 15      # validated: 9:30-9:45 ET
RECLAIM_WINDOW_MIN = 30
SWEEP_MIN_TICKS = 0.05
ET = ZoneInfo("America/New_York")


def _fetch_reference_range(ib: IB, ticker: str):
    """Prior real trading day's RTH high/low."""
    contract = ib.qualifyContracts(Stock(ticker, "SMART", "USD"))[0]
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr="5 D",
        barSizeSetting="1 day", whatToShow="TRADES", useRTH=True, formatDate=1)
    if len(bars) < 2:
        return None, None
    prior = bars[-2]  # bars[-1] is today (still forming) if called intraday
    return prior.high, prior.low


def _fetch_today_5min_bars(ib: IB, ticker: str):
    contract = ib.qualifyContracts(Stock(ticker, "SMART", "USD"))[0]
    bars = ib.reqHistoricalData(
        contract, endDateTime="", durationStr="1 D",
        barSizeSetting="5 mins", whatToShow="TRADES", useRTH=True, formatDate=1)
    return bars


def get_crt_signal(ib: IB, ticker: str = "SPY") -> dict | None:
    """Returns {'crt_day': bool, 'direction': str|None, 'ref_high': float,
    'ref_low': float, 'sweep_time': str|None} using REAL live data, or None
    if it's too early in the session to have a real determination yet."""
    now_et = datetime.now(ET)
    session_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    min_needed = SWEEP_WINDOW_MIN + RECLAIM_WINDOW_MIN
    if now_et < session_open + timedelta(minutes=min_needed):
        return None

    ref_high, ref_low = _fetch_reference_range(ib, ticker)
    if ref_high is None:
        return None

    bars = _fetch_today_5min_bars(ib, ticker)
    if not bars:
        return None
    df = util.df(bars)

    bar_minutes = 5
    sweep_bars = max(1, SWEEP_WINDOW_MIN // bar_minutes)
    reclaim_bars = max(1, RECLAIM_WINDOW_MIN // bar_minutes)
    early = df.iloc[:sweep_bars]

    swept_up = (early["high"] >= ref_high + SWEEP_MIN_TICKS).any()
    swept_down = (early["low"] <= ref_low - SWEEP_MIN_TICKS).any()

    result = {"crt_day": False, "direction": None, "ref_high": ref_high, "ref_low": ref_low, "sweep_time": None}
    if swept_up:
        idx = early[early["high"] >= ref_high + SWEEP_MIN_TICKS].index[0]
        window = df.iloc[idx:idx + reclaim_bars]
        if (window["close"] < ref_high).any():
            result.update({"crt_day": True, "direction": "bearish_reversal_expected",
                            "sweep_time": str(df.iloc[idx]["date"])})
    elif swept_down:
        idx = early[early["low"] <= ref_low - SWEEP_MIN_TICKS].index[0]
        window = df.iloc[idx:idx + reclaim_bars]
        if (window["close"] > ref_low).any():
            result.update({"crt_day": True, "direction": "bullish_reversal_expected",
                            "sweep_time": str(df.iloc[idx]["date"])})
    return result
