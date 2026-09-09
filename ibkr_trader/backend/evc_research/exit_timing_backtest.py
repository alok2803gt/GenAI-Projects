"""
Real backtest: does EVC's live 9:31-9:35 AM exit window actually beat other
candidate exit times? CEO question 2026-08-28: "do we have these backtested
to always close between 9:30 to 9:35 after earnings open?" -- answer found
by inspecting evc_condor_backtest.py / evc_condor_backtest_real_data.py:
NO, both existing backtests measure payoff against the stock's DAILY CLOSE
on the outcome day, never against any specific intraday time. The live
9:31-9:35 exit convention has never actually been tested against anything.

Methodology:
  - Same entry/outcome-date logic (BMO->prior day, AMC->same day, outcome
    = next trading day) and same round_strike()/Black-Scholes machinery as
    evc_condor_backtest.py, reused byte-for-byte where it overlaps.
  - Strikes use the account's REAL LIVE cushions as of 2026-08-27
    (put_cushion_mult=1.3, call_cushion_mult=1.0, wing_mult=1.5) -- NOT the
    older symmetric 1.0/1.0 the original backtest script used -- since the
    point is to validate the exit timing for the structure actually traded
    today, not re-litigate entry cushion selection.
  - IV estimate = realized_vol_20d * 1.45 (midpoint of the 1.3/1.6 range
    the real-data companion script already validated as reasonable
    Black-Scholes multipliers for this account's universe).
  - The ONLY new thing: instead of pricing the payoff off yfinance's daily
    close, this pulls REAL IBKR 1-minute bars for the outcome_date and
    computes the payoff at each of several candidate exit times: the open
    (9:30), 9:35 (approximating the live 9:31-9:35 window), 9:45, 10:00,
    10:30, 11:00, 12:00, and the daily close (as the existing baseline).
  - IBKR 1-min bars confirmed available to at least ~10 months back
    (checked live 2026-08-28: NVDA 1-min bars for Nov 2025, Feb 2026, and
    May 2026 all returned real, full 390-bar days) -- events are filtered
    to outcome dates within that window; older events are skipped and
    counted, not silently dropped.
"""
import asyncio
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from ib_insync import IB, Stock

sys.path.insert(0, "..")

# ── Config -- matches this account's REAL live EVC config as of 2026-08-27 ──
PUT_CUSHION_MULT = 1.3
CALL_CUSHION_MULT = 1.0
WING_MULT = 1.5
MIN_MOVE = 0.03
MAX_MOVE_BOUND = 0.12
RISK_FREE = 0.045
DIV_YIELD = 0.0
T_YEARS = 1.0 / 252.0
VOL_LOOKBACK_DAYS = 20
IV_MULT = 1.45

IBKR_MINUTE_BAR_CUTOFF = date.today() - timedelta(days=270)  # ~9mo, safely inside the confirmed 10mo window

EXIT_TIMES = ["09:30", "09:35", "09:45", "10:00", "10:30", "11:00", "12:00", "close"]

CLIENT_ID = 987


def round_strike(price: float) -> float:
    if price < 50:
        step = 1.0
    elif price < 200:
        step = 2.5
    else:
        step = 5.0
    return float(round(price / step) * step)


def bs_call_put(S, K, T, r, sigma, q=0.0):
    if S <= 0 or K <= 0:
        return 0.0, 0.0
    if T <= 0 or sigma <= 0:
        return max(0.0, S - K), max(0.0, K - S)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return max(call, 0.0), max(put, 0.0)


def payoff_at_price(price, short_put, long_put, short_call, long_call, net_credit, put_width, call_width):
    if short_put <= price <= short_call:
        return net_credit
    elif price < short_put:
        breach = short_put - price
        return net_credit - breach if price >= long_put else net_credit - (put_width - net_credit)
    else:
        breach = price - short_call
        return net_credit - breach if price <= long_call else net_credit - (call_width - net_credit)


@dataclass
class Event:
    ticker: str
    earnings_date: str
    timing: str
    entry_date: date
    outcome_date: date
    spot: float
    short_put: float
    long_put: float
    short_call: float
    long_call: float
    net_credit: float
    put_width: float
    call_width: float
    prices: dict = field(default_factory=dict)   # exit_time -> real price
    payoffs: dict = field(default_factory=dict)  # exit_time -> payoff


def build_candidates(tickers: list[str]) -> list[Event]:
    events = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5y", interval="1d", auto_adjust=False, actions=False)
            if hist is None or hist.empty:
                continue
            hist = hist[["Close"]].copy()
            hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
            hist = hist[~hist.index.duplicated(keep="last")].sort_index()
            edf = t.get_earnings_dates(limit=8)
            if edf is None or edf.empty:
                continue
        except Exception as exc:
            print(f"  [SKIP TICKER] {ticker}: {exc}")
            continue

        trading_days = list(hist.index.date)
        td_index = {d: i for i, d in enumerate(trading_days)}
        closes = hist["Close"].values
        log_ret = np.diff(np.log(closes))

        def next_trading_day(d):
            for td in trading_days:
                if td > d:
                    return td
            return None

        def prev_trading_day(d):
            prev = None
            for td in trading_days:
                if td >= d:
                    break
                prev = td
            return prev

        def trading_day_on_or_after(d):
            for td in trading_days:
                if td >= d:
                    return td
            return None

        today = date.today()
        for ts, erow in edf.iterrows():
            edate = pd.Timestamp(ts)
            edate_local = edate.date()
            if edate_local >= today:
                continue
            reported_eps = erow.get("Reported EPS", np.nan)
            if pd.isna(reported_eps):
                continue

            timing = "BMO" if edate.hour < 12 else "AMC"
            if timing == "BMO":
                base_day = edate_local if edate_local in td_index else trading_day_on_or_after(edate_local)
                if base_day is None:
                    continue
                entry_date = prev_trading_day(base_day)
            else:
                entry_date = edate_local if edate_local in td_index else prev_trading_day(
                    trading_day_on_or_after(edate_local) or edate_local)
            if entry_date is None or entry_date not in td_index:
                continue
            outcome_date = next_trading_day(entry_date)
            if outcome_date is None or outcome_date < IBKR_MINUTE_BAR_CUTOFF:
                continue

            entry_idx = td_index[entry_date]
            if entry_idx < VOL_LOOKBACK_DAYS:
                continue
            spot = float(closes[entry_idx])
            if spot <= 0:
                continue
            window = log_ret[entry_idx - VOL_LOOKBACK_DAYS: entry_idx]
            if len(window) < VOL_LOOKBACK_DAYS or np.any(np.isnan(window)):
                continue
            realized_vol = float(np.std(window, ddof=1) * math.sqrt(252))
            if realized_vol <= 0 or not math.isfinite(realized_vol):
                continue

            atm_strike = round_strike(spot)
            iv_est = realized_vol * IV_MULT
            call_atm, put_atm = bs_call_put(spot, atm_strike, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
            expected_move = call_atm + put_atm
            im_pct = expected_move / spot if spot > 0 else np.nan
            if not math.isfinite(im_pct) or im_pct < MIN_MOVE or im_pct > MAX_MOVE_BOUND:
                continue

            short_put = round_strike(spot - PUT_CUSHION_MULT * expected_move)
            short_call = round_strike(spot + CALL_CUSHION_MULT * expected_move)
            long_put = round_strike(spot - WING_MULT * expected_move)
            long_call = round_strike(spot + WING_MULT * expected_move)
            if not (long_put < short_put < short_call < long_call):
                continue

            sc_call, _ = bs_call_put(spot, short_call, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
            _, sp_put = bs_call_put(spot, short_put, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
            lc_call, _ = bs_call_put(spot, long_call, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
            _, lp_put = bs_call_put(spot, long_put, T_YEARS, RISK_FREE, iv_est, DIV_YIELD)
            net_credit = round((sp_put + sc_call) - (lp_put + lc_call), 4)
            if net_credit <= 0:
                continue
            put_width = short_put - long_put
            call_width = long_call - short_call
            if max(put_width, call_width) - net_credit <= 0:
                continue

            events.append(Event(
                ticker=ticker, earnings_date=str(edate_local), timing=timing,
                entry_date=entry_date, outcome_date=outcome_date, spot=spot,
                short_put=short_put, long_put=long_put, short_call=short_call, long_call=long_call,
                net_credit=net_credit, put_width=put_width, call_width=call_width,
            ))
    return events


async def fill_real_prices(events: list[Event]) -> None:
    ib = IB()
    await ib.connectAsync("127.0.0.1", 7496, clientId=CLIENT_ID, timeout=15)
    ok, failed = 0, 0
    for i, ev in enumerate(events):
        try:
            c = Stock(ev.ticker, "SMART", "USD")
            await ib.qualifyContractsAsync(c)
            end_dt = f"{ev.outcome_date.strftime('%Y%m%d')} 16:00:00 US/Eastern"
            bars = await ib.reqHistoricalDataAsync(
                c, endDateTime=end_dt, durationStr="1 D", barSizeSetting="1 min",
                whatToShow="TRADES", useRTH=True, keepUpToDate=False,
            )
            if not bars or len(bars) < 300:
                failed += 1
                continue
            by_time = {b.date.strftime("%H:%M"): b for b in bars}

            def price_at(hhmm):
                if hhmm in by_time:
                    return float(by_time[hhmm].open)
                # nearest bar at or after hhmm (handles halts/thin bars)
                target = datetime.strptime(hhmm, "%H:%M").time()
                cands = [b for b in bars if b.date.time() >= target]
                return float(cands[0].open) if cands else None

            ev.prices["09:30"] = float(bars[0].open)
            for hhmm in ["09:35", "09:45", "10:00", "10:30", "11:00", "12:00"]:
                ev.prices[hhmm] = price_at(hhmm)
            ev.prices["close"] = float(bars[-1].close)

            for exit_t, px in ev.prices.items():
                if px is None:
                    continue
                ev.payoffs[exit_t] = payoff_at_price(
                    px, ev.short_put, ev.long_put, ev.short_call, ev.long_call,
                    ev.net_credit, ev.put_width, ev.call_width,
                )
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  [SKIP BARS] {ev.ticker} {ev.outcome_date}: {exc}")
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{len(events)} processed (ok={ok} failed={failed})")
        await asyncio.sleep(1.2)   # IBKR pacing
    ib.disconnect()
    print(f"Real minute-bar fetch done: {ok} ok, {failed} failed/skipped")


def summarize(events: list[Event]) -> pd.DataFrame:
    rows = []
    for exit_t in EXIT_TIMES:
        payoffs = [ev.payoffs[exit_t] for ev in events if exit_t in ev.payoffs]
        credits = [ev.net_credit for ev in events if exit_t in ev.payoffs]
        spots = [ev.spot for ev in events if exit_t in ev.payoffs]
        if not payoffs:
            rows.append({"exit_time": exit_t, "n": 0})
            continue
        wins = [p for p in payoffs if p > 0]
        pct_credit = [p / c for p, c in zip(payoffs, credits) if c]
        pct_spot = [p / s for p, s in zip(payoffs, spots) if s]
        rows.append({
            "exit_time": exit_t,
            "n": len(payoffs),
            "win_rate_pct": round(len(wins) / len(payoffs) * 100, 1),
            "avg_payoff": round(sum(payoffs) / len(payoffs), 4),
            "total_payoff": round(sum(payoffs), 2),
            "avg_payoff_pct_credit": round(sum(pct_credit) / len(pct_credit) * 100, 1) if pct_credit else None,
            "avg_payoff_pct_spot": round(sum(pct_spot) / len(pct_spot) * 100, 3) if pct_spot else None,
            "worst": round(min(payoffs), 2),
            "best": round(max(payoffs), 2),
        })
    return pd.DataFrame(rows)


async def main():
    from importlib import import_module
    import re
    main_src = open("../main.py", encoding="utf-8").read()
    m = re.search(r"CANDIDATE_POOL: List\[str\] = \[(.*?)\]\n", main_src, re.DOTALL)
    all_tickers = re.findall(r'"([A-Z\.]+)"', m.group(1))
    # Representative, tractable sample -- every 3rd ticker keeps sector mix
    # roughly intact (the pool is grouped by sector) while cutting runtime.
    tickers = all_tickers[::3]
    print(f"Universe: {len(all_tickers)} total, sampling {len(tickers)} (every 3rd)")

    print("Building candidate events (yfinance: earnings dates + daily history)...")
    events = build_candidates(tickers)
    print(f"{len(events)} qualifying events within IBKR's confirmed minute-bar window "
          f"(outcome_date >= {IBKR_MINUTE_BAR_CUTOFF})")

    if not events:
        print("No events -- nothing to backtest.")
        return

    print("Fetching REAL IBKR 1-minute bars for each event's outcome date...")
    await fill_real_prices(events)

    df_events = pd.DataFrame([{
        "ticker": e.ticker, "earnings_date": e.earnings_date, "timing": e.timing,
        "outcome_date": str(e.outcome_date), "spot": e.spot, "net_credit": e.net_credit,
        **{f"price_{k}": v for k, v in e.prices.items()},
        **{f"payoff_{k}": v for k, v in e.payoffs.items()},
    } for e in events])
    df_events.to_csv("exit_timing_backtest_events.csv", index=False)

    summary = summarize(events)
    summary.to_csv("exit_timing_backtest_summary.csv", index=False)
    print("\n=== EVC exit-timing backtest: real IBKR 1-min bars, all times same event set ===")
    print(summary.to_string(index=False))
    print("\nSaved exit_timing_backtest_events.csv and exit_timing_backtest_summary.csv")


if __name__ == "__main__":
    asyncio.run(main())
