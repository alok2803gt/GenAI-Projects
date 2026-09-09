"""
NFLX put-credit-spread signal backtest: existing mean-reversion signals
(A/B, as actually implemented in nflx_bottom_monitor.py) vs. three
trend-following candidate signals, over real historical NFLX daily data.

WHY THIS EXISTS
----------------
CEO complaint (2026-08-18): NFLX rallied from the low $70s to ~$80 over
about a week and neither of nflx_bottom_monitor.py's two signals fired --
RSI stayed elevated the whole way up (never "reset" to <=45) and price
never closed back above the ~$89-90 200-day SMA. The account watched the
move happen and caught none of it. This script asks: would a genuine
TREND-FOLLOWING entry signal (as opposed to the existing mean-reversion
signals) have caught moves like that historically, with a defensible win
rate, using the SAME underlying trade structure the account already runs?

TRADE STRUCTURE (mirrors nflx_bottom_monitor.py exactly -- read that file
for the live version of this logic)
----------------------------------------------------------------------
Bullish put credit spread: SELL a put ~11% OTM (SHORT_OTM_PCT=0.11),
BUY a put WIDTH=2.0 points further out of the money. QTY=1 contract
(100 multiplier), matching the live monitor's sizing. Max risk per
contract = width*100 - credit*100 (defined-risk, capped).

SIGNALS TESTED
--------------
Baseline (existing, replicated EXACTLY from nflx_bottom_monitor.py's
check_signals(), not the more aspirational docstring description of it):
  A: rsi14 <= 45  AND  close >= rolling_52wk_low * 0.98
  B: close > rolling_sma200

Trend-following candidates (new, none of these exist in production):
  TF1: SMA20 basing breakout -- close crosses above its 20-day SMA today,
       after closing below that SMA on each of the prior 5 trading days
       (basing, not single-day noise).
  TF2: Donchian(20) breakout -- close makes a new 20-trading-day closing
       high (today's close > max close of the preceding 20 days).
  TF3: SMA(10/30) crossover -- 10-day SMA crosses above the 30-day SMA
       ("golden cross, lite").

For every signal, a historical "entry" is only counted on the FIRST day
the condition transitions from false->true (edge detection) AND only if
no trade already opened under that same signal is still within its
modeled DTE window (cooldown = previous entry's modeled expiry date).
This mirrors nflx_bottom_monitor.py's own state-file behavior (it does
not re-alert while "awaiting_confirmation" or already "fired") and keeps
the backtest to one open position per signal at a time -- appropriate
for a ~$4,200 account trading QTY=1.

METHODOLOGY AND APPROXIMATIONS (state these plainly, do not hand-wave)
------------------------------------------------------------------
1. No real historical NFLX options-chain data exists, so premiums are a
   Black-Scholes ESTIMATE, exactly the pattern already used and disclosed
   in evc_condor_backtest.py: IV proxy = trailing 20-day realized
   (close-to-close, annualized) volatility * a multiplier. Tested at
   BOTH 1.0x and 1.3x (IV_MULTS) as an explicit sensitivity range, not a
   single guess. Real listed IV would very likely carry some skew/term
   premium over realized vol, so 1.0x is a conservative floor and 1.3x a
   more realistic estimate; treat 1.0x results as a lower bound on credit
   (and therefore a lower bound on win rate, since less credit = less
   room before the strike is breached at expiry).
2. Modeled DTE is fixed at 37 calendar days forward (midpoint of the
   25-45 DTE range nflx_bottom_monitor.py's live expiry search targets).
   This is NOT swept as a sensitivity range -- only IV_MULT is, per the
   task -- so a different DTE choice could move results and isn't tested
   here.
3. Strikes: short strike = spot*(1-0.11), rounded to the nearest real-
   world-plausible strike increment via round_strike() (reused from
   evc_condor_backtest.py's tiered convention: $1 under $50, $2.50 from
   $50-200, $5 above). Long strike = short_strike - WIDTH exactly (not
   independently rounded), so every trade's max-risk sizing stays a true
   2.0-point spread, matching the live monitor's sizing intent. Actual
   listed strikes on the day (which the live monitor picks from a real
   IBKR chain) may differ modestly from this.
4. Outcome is evaluated AT THE MODELED EXPIRY DATE ONLY (the nearest
   trading day on/after entry_date + 37 calendar days) -- i.e. European-
   style, not path-dependent. No early-assignment or intraday-breach
   modeling. Payoff is graduated (not purely binary) using the same
   piecewise-intrinsic-value formula evc_condor_backtest.py already uses
   for its condor: full credit if price finishes at/above the short
   strike, credit-minus-breach if it finishes between the strikes,
   capped at credit-minus-width (the true max loss of a defined-risk
   spread) if it finishes at/below the long strike.
5. Entry price = the signal day's CLOSE (matches what the live monitor
   would see checking EOD data once per day); no slippage, commissions,
   or bid/ask spread modeled -- same simplification evc_condor_backtest.py
   uses.
6. Signals near the end of the available data window (within ~37 days of
   the last trading day) have no real forward path to evaluate and are
   EXCLUDED from results, flagged as such -- not silently dropped or
   scored zero.
7. Underlying data: yfinance daily history for NFLX, same call pattern
   nflx_bottom_monitor.py itself uses (no explicit auto_adjust override),
   so the price series driving this backtest is exactly what the live
   monitor would have seen on each of those historical days.

Run: python nflx_trend_backtest.py
Output:
  nflx_trend_backtest_results.csv  (one row per modeled trade)
  nflx_trend_backtest_summary.csv  (per signal x iv_mult, overall + by year)
Read-only. Does not import nflx_bottom_monitor.py, does not touch any
live config, state file, or place any order.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Config -- trade structure mirrors nflx_bottom_monitor.py exactly
# ----------------------------------------------------------------------------
TICKER = "NFLX"
FETCH_START = "2019-01-01"   # >=2y buffer before the 2021 analysis window for
                              # SMA200 / 52wk-low / RSI warm-up
RSI_RESET_MAX = 45.0
LOW_BUFFER = 0.98
SHORT_OTM_PCT = 0.11
WIDTH = 2.0
QTY = 1
RISK_FREE = 0.045            # matches this account's gex-vex-calculator / evc convention
VOL_LOOKBACK_DAYS = 20
DTE_DAYS = 37                 # fixed, calendar days -- see approximation #2 above
IV_MULTS = [1.0, 1.3]
SMA20_BASING_DAYS = 5
DONCHIAN_WINDOW = 20
SMA_SHORT, SMA_LONG = 10, 30
ANALYSIS_START_YEAR = 2021   # report window: cover 2021 crash-into-2026 regimes


def round_strike(price: float) -> float:
    """Reused verbatim (tiering) from evc_condor_backtest.py's round_strike()
    -- a generic approximation for real listed strike increments since we
    have no historical NFLX options chain to pick real strikes from."""
    if price < 50:
        step = 1.0
    elif price < 200:
        step = 2.5
    else:
        step = 5.0
    return float(round(price / step) * step)


def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put premium."""
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return max(put, 0.0)


@dataclass
class Trade:
    signal: str
    iv_mult: float
    entry_date: str
    year: int
    entry_spot: float
    realized_vol_20d: float
    iv_estimate: float
    short_strike: float
    long_strike: float
    credit: float
    max_risk_dollars: float
    breakeven_win_rate_trade: float
    expiry_target_date: str
    expiry_actual_date: str
    expiry_price: float
    payoff_per_share: float
    pnl_dollars: float          # payoff_per_share * 100 * QTY
    win: bool
    status: str = "ok"
    reason: str = ""


def fetch_data() -> pd.DataFrame:
    hist = yf.Ticker(TICKER).history(start=FETCH_START)
    # same NaN-placeholder-row guard as nflx_bottom_monitor.py's check_signals()
    hist = hist.dropna(subset=["Close"])
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    hist = hist[~hist.index.duplicated(keep="last")].sort_index()
    return hist[["Close"]].copy()


def build_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    close = df["Close"]

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + gain / loss)

    df["sma200"] = close.rolling(200).mean()
    df["low_52wk"] = close.rolling(252).min()
    df["sma20"] = close.rolling(20).mean()
    df["sma_short"] = close.rolling(SMA_SHORT).mean()
    df["sma_long"] = close.rolling(SMA_LONG).mean()
    df["donchian_prior20"] = close.shift(1).rolling(DONCHIAN_WINDOW).max()

    # Signal A / B -- exact replica of nflx_bottom_monitor.py's check_signals()
    df["sig_A_raw"] = (df["rsi14"] <= RSI_RESET_MAX) & (close >= df["low_52wk"] * LOW_BUFFER)
    df["sig_B_raw"] = close > df["sma200"]

    # TF1: SMA20 basing breakout
    below20 = close < df["sma20"]
    basing_ok = below20.shift(1).rolling(SMA20_BASING_DAYS).sum() == SMA20_BASING_DAYS
    df["sig_TF1_raw"] = (close > df["sma20"]) & below20.shift(1).fillna(False).astype(bool) & basing_ok.fillna(False)

    # TF2: Donchian(20) new-high breakout
    df["sig_TF2_raw"] = close > df["donchian_prior20"]

    # TF3: SMA(10/30) golden crossover
    df["sig_TF3_raw"] = df["sma_short"] > df["sma_long"]

    # realized vol (20d trailing log-return std, annualized) -- same construction
    # as evc_condor_backtest.py's realized_vol_20d
    log_ret = np.log(close / close.shift(1))
    df["realized_vol_20d"] = log_ret.rolling(VOL_LOOKBACK_DAYS).std(ddof=1) * math.sqrt(252)

    return df


def find_entries(df: pd.DataFrame, raw_col: str, edge_required: bool) -> list:
    """Sequential walk with edge-detection + cooldown (no new entry for a
    signal while the previous trade under that signal hasn't reached its
    modeled expiry yet). edge_required=True means only count the FIRST day
    of a true-condition run (rising edge); all 5 signals use this so
    persistently-true conditions (e.g. a long stretch above the 200sma, or
    consecutive new-high days) don't spawn overlapping trades every day."""
    entries = []
    prev_raw = False
    open_until = None  # date; new entry blocked while today's date <= open_until
    for dt, row in df.iterrows():
        raw = bool(row[raw_col]) if pd.notna(row[raw_col]) else False
        is_edge = raw and not prev_raw
        fires = raw if not edge_required else is_edge
        if fires and (open_until is None or dt.date() > open_until):
            entries.append(dt)
            open_until = (dt + timedelta(days=DTE_DAYS)).date()
        prev_raw = raw
    return entries


def model_trade(df: pd.DataFrame, signal: str, entry_dt: pd.Timestamp, iv_mult: float) -> Trade:
    close = df["Close"]
    entry_spot = float(close.loc[entry_dt])
    rv = float(df.loc[entry_dt, "realized_vol_20d"])
    year = entry_dt.year

    base = dict(signal=signal, iv_mult=iv_mult, entry_date=str(entry_dt.date()), year=year,
                entry_spot=round(entry_spot, 2), realized_vol_20d=np.nan, iv_estimate=np.nan,
                short_strike=np.nan, long_strike=np.nan, credit=np.nan, max_risk_dollars=np.nan,
                breakeven_win_rate_trade=np.nan, expiry_target_date="", expiry_actual_date="",
                expiry_price=np.nan, payoff_per_share=np.nan, pnl_dollars=np.nan, win=False)

    if not math.isfinite(rv) or rv <= 0:
        return Trade(status="skip", reason="bad_vol_window", **base)

    target_expiry = entry_dt + timedelta(days=DTE_DAYS)
    fwd = close.loc[close.index > entry_dt]
    fwd_on_or_after = fwd.loc[fwd.index >= target_expiry]
    if fwd_on_or_after.empty:
        base["expiry_target_date"] = str(target_expiry.date())
        return Trade(status="skip", reason="insufficient_forward_data", **base)
    expiry_dt = fwd_on_or_after.index[0]
    expiry_price = float(fwd_on_or_after.iloc[0])

    iv_est = rv * iv_mult
    short_k = round_strike(entry_spot * (1 - SHORT_OTM_PCT))
    long_k = short_k - WIDTH
    T = DTE_DAYS / 365.0

    short_put_prem = bs_put(entry_spot, short_k, T, RISK_FREE, iv_est)
    long_put_prem = bs_put(entry_spot, long_k, T, RISK_FREE, iv_est)
    credit = round(short_put_prem - long_put_prem, 4)

    if credit <= 0:
        base.update(realized_vol_20d=round(rv, 4), iv_estimate=round(iv_est, 4),
                     short_strike=short_k, long_strike=long_k,
                     expiry_target_date=str(target_expiry.date()), expiry_actual_date=str(expiry_dt.date()))
        return Trade(status="skip", reason="non_positive_credit", **base)

    max_risk = WIDTH - credit  # per share; dollars = *100
    if max_risk <= 0:
        base.update(realized_vol_20d=round(rv, 4), iv_estimate=round(iv_est, 4),
                     short_strike=short_k, long_strike=long_k, credit=credit,
                     expiry_target_date=str(target_expiry.date()), expiry_actual_date=str(expiry_dt.date()))
        return Trade(status="skip", reason="non_positive_risk", **base)

    # payoff at expiry, graduated, capped -- same piecewise structure as
    # evc_condor_backtest.py's condor payoff, applied to a single put spread
    if expiry_price >= short_k:
        payoff = credit
    elif expiry_price <= long_k:
        payoff = credit - WIDTH
    else:
        payoff = credit - (short_k - expiry_price)

    pnl_dollars = round(payoff * 100 * QTY, 2)

    return Trade(
        signal=signal, iv_mult=iv_mult, entry_date=str(entry_dt.date()), year=year,
        entry_spot=round(entry_spot, 2), realized_vol_20d=round(rv, 4), iv_estimate=round(iv_est, 4),
        short_strike=short_k, long_strike=long_k, credit=credit,
        max_risk_dollars=round(max_risk * 100, 2),
        breakeven_win_rate_trade=round(max_risk / (max_risk + credit), 4),
        expiry_target_date=str(target_expiry.date()), expiry_actual_date=str(expiry_dt.date()),
        expiry_price=round(expiry_price, 2), payoff_per_share=round(payoff, 4),
        pnl_dollars=pnl_dollars, win=bool(payoff > 0), status="ok", reason="",
    )


SIGNAL_DEFS = [
    ("A_rsi_reset_baseline", "sig_A_raw", "mean-reversion (existing, baseline)"),
    ("B_sma200_breakout_baseline", "sig_B_raw", "mean-reversion (existing, baseline)"),
    ("TF1_sma20_basing_breakout", "sig_TF1_raw", "trend-following (candidate)"),
    ("TF2_donchian20_high", "sig_TF2_raw", "trend-following (candidate)"),
    ("TF3_sma10_30_crossover", "sig_TF3_raw", "trend-following (candidate)"),
]


def main():
    print(f"Fetching {TICKER} history from {FETCH_START}...")
    hist = fetch_data()
    print(f"  {len(hist)} trading days, {hist.index[0].date()} .. {hist.index[-1].date()}")
    df = build_indicators(hist)

    all_trades: list[Trade] = []
    for signal_name, raw_col, _kind in SIGNAL_DEFS:
        entries_all = find_entries(df, raw_col, edge_required=True)
        # restrict to the analysis window (still uses full pre-window history
        # for indicator warm-up + cooldown continuity, only reporting entries
        # from ANALYSIS_START_YEAR forward)
        entries = [e for e in entries_all if e.year >= ANALYSIS_START_YEAR]
        print(f"{signal_name}: {len(entries)} historical entries in {ANALYSIS_START_YEAR}+ "
              f"({len(entries_all)} total incl. warm-up years)")
        for entry_dt in entries:
            for iv_mult in IV_MULTS:
                t = model_trade(df, signal_name, entry_dt, iv_mult)
                all_trades.append(t)

    rows = [t.__dict__ for t in all_trades]
    res = pd.DataFrame(rows)
    out_path = "nflx_trend_backtest_results.csv"
    res.to_csv(out_path, index=False)
    print(f"\nSaved {len(res)} trade-rows to {out_path}")

    excluded = res[res["status"] != "ok"]
    if len(excluded):
        print(f"\nExcluded (not scored): {len(excluded)} rows")
        print(excluded.groupby(["signal", "reason"]).size())

    ok = res[res["status"] == "ok"].copy()

    # ------------------------------------------------------------------
    # Summary: overall (per signal x iv_mult) + by year
    # ------------------------------------------------------------------
    summary_rows = []
    for signal_name, _raw, kind in SIGNAL_DEFS:
        for iv_mult in IV_MULTS:
            sub = ok[(ok["signal"] == signal_name) & (ok["iv_mult"] == iv_mult)]
            n = len(sub)
            if n == 0:
                summary_rows.append(dict(signal=signal_name, kind=kind, iv_mult=iv_mult,
                                          year="ALL", n=0))
                continue
            summary_rows.append(dict(
                signal=signal_name, kind=kind, iv_mult=iv_mult, year="ALL",
                n=n, win_rate=round(sub["win"].mean(), 4),
                avg_pnl_dollars=round(sub["pnl_dollars"].mean(), 2),
                total_pnl_dollars=round(sub["pnl_dollars"].sum(), 2),
                worst_loss_dollars=round(sub["pnl_dollars"].min(), 2),
                avg_credit=round(sub["credit"].mean(), 4),
                avg_breakeven_win_rate_needed=round(sub["breakeven_win_rate_trade"].mean(), 4),
            ))
            for yr in sorted(sub["year"].unique()):
                ysub = sub[sub["year"] == yr]
                summary_rows.append(dict(
                    signal=signal_name, kind=kind, iv_mult=iv_mult, year=str(yr),
                    n=len(ysub), win_rate=round(ysub["win"].mean(), 4),
                    avg_pnl_dollars=round(ysub["pnl_dollars"].mean(), 2),
                    total_pnl_dollars=round(ysub["pnl_dollars"].sum(), 2),
                    worst_loss_dollars=round(ysub["pnl_dollars"].min(), 2),
                    avg_credit=round(ysub["credit"].mean(), 4),
                    avg_breakeven_win_rate_needed=round(ysub["breakeven_win_rate_trade"].mean(), 4),
                ))

    summary_df = pd.DataFrame(summary_rows)
    summary_out = "nflx_trend_backtest_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved summary to {summary_out}")

    print("\n" + "=" * 100)
    print(f"OVERALL RESULTS ({ANALYSIS_START_YEAR}-{df.index[-1].year})")
    print("=" * 100)
    overall = summary_df[summary_df["year"] == "ALL"]
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_columns", None):
        print(overall.to_string(index=False))

    print("\n" + "=" * 100)
    print("BY YEAR")
    print("=" * 100)
    by_year = summary_df[summary_df["year"] != "ALL"].sort_values(["signal", "iv_mult", "year"])
    with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_columns", None):
        print(by_year.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
