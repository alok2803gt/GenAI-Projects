"""
Real test: would an intraday mark-to-market stop-loss have improved on the
currently-deployed hold-to-hard-close design (wing_step=4, no early exit --
alpaca_0dte_butterfly_trader.py's own "no early profit target (MVP)")?

Motivation: 2026-09-08 CEO review of QQQ's real 3-trade live history (1W/2L,
-$206 net) found 2 of 3 losses traced to the underlying moving beyond the
wing band, not slippage or a fixable bug (see oversight_log.jsonl). A
pre-entry filter (trailing ATR%, tighter near-S/R threshold) was tested
first and found NOT to work (near-zero/unreliable correlation with which
days blow through the wings -- see chat record 2026-09-08). This script
tests the other candidate: exit EARLY once a trade is clearly failing,
instead of always holding the full debit to the 15:47 hard-close.

METHODOLOGY AND ITS APPROXIMATIONS (stated explicitly, per this account's
standing backtest discipline -- no historical intraday OPTIONS data exists
for this account's tickers, same constraint already documented in
alpaca_0dte_butterfly_trader.py's own docstring for the GEX-nudge feature):

1. Real data used: the exact same 45 trading days x 3 tickers x wing_step=4
   rows already in butterfly_gex_sr_enriched_v2.csv (entry_debit, k1/k2/k3,
   morning_spot, close_spot, long_payoff -- all real, already-backtested
   figures from butterfly_0dte_backtest_v2.py), PLUS real 1-min intraday
   underlying bars pulled fresh from Alpaca for those same 45 days
   (butterfly_intraday_fetch.py -> butterfly_intraday_cache/*.pkl).
2. Modeled, NOT real: intraday OPTION prices. Priced via Black-Scholes on
   each leg (all calls, matching the live structure), with a SINGLE
   constant IV per trade-day, back-solved (via scipy.optimize.brentq) so
   that the BS-modeled butterfly value AT ENTRY (9:35 ET, real T-to-close)
   exactly equals that day's REAL recorded entry_debit. This anchors every
   day's repricing curve to something real rather than an assumed vol
   level, but still assumes flat IV through the day (no real smile/skew or
   vol-of-vol dynamics) -- a real approximation.
3. Time-to-expiry: modeled as (minutes remaining until 16:00 ET) / 390
   trading minutes per day / 252 trading days/year -- a fraction-of-day
   convention, not real dividend/holiday-adjusted calendar time. This
   scaling is internally consistent for the relative decay used to
   evaluate a stop rule, but is not a claim of absolute BS pricing
   accuracy.
4. To avoid compounding model error: a simulated trade that does NOT
   trigger any stop rule keeps its REAL recorded long_payoff (from real
   close_spot, not the model) as its terminal outcome -- the model is only
   used to decide WHEN/WHETHER an early exit would have fired, and to
   price that one early exit. The majority of each day's payoff data stays
   100% real.
5. Checkpoints: every 15 real minutes from 9:35 to 15:45 ET, using the
   actual 1-min bar closest to each checkpoint (real price, no lookahead:
   only bars up to and including the checkpoint time are used).

Stop rule tested: exit the whole butterfly at the first checkpoint where
the modeled mark-to-market loss reaches >= stop_pct of the entry debit
paid (e.g., stop_pct=0.65 means "exit once down 65% of what was paid").
Tested at several stop_pct thresholds, each ticker separately and combined,
compared against the real, already-known hold-to-close baseline.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

HERE = Path(__file__).parent
CACHE = HERE / "butterfly_intraday_cache"

ET = "America/New_York"
SESSION_MINUTES = 390.0   # 9:30-16:00 ET
TRADING_DAYS_PER_YEAR = 252.0
R = 0.0  # risk-free rate, negligible over a same-day 0DTE window


def bs_call(spot, strike, t_years, vol, r=R):
    if t_years <= 1e-8 or vol <= 1e-6:
        return max(spot - strike, 0.0)
    d1 = (np.log(spot / strike) + (r + 0.5 * vol ** 2) * t_years) / (vol * np.sqrt(t_years))
    d2 = d1 - vol * np.sqrt(t_years)
    return spot * norm.cdf(d1) - strike * np.exp(-r * t_years) * norm.cdf(d2)


def butterfly_value(spot, k1, k2, k3, t_years, vol):
    return (bs_call(spot, k1, t_years, vol) + bs_call(spot, k3, t_years, vol)
            - 2 * bs_call(spot, k2, t_years, vol))


def solve_iv(target_value, spot, k1, k2, k3, t_years):
    """Back out the flat IV that makes the modeled entry value match the
    REAL recorded entry_debit. Butterfly value is not monotonic in vol in
    general, but for a near-ATM 0DTE butterfly it's decreasing in vol
    (more vol -> more value smeared into the wings, less concentrated at
    the body) over the practically relevant range, so a bracket search
    across a wide vol range is reliable here."""
    def f(vol):
        return butterfly_value(spot, k1, k2, k3, t_years, vol) - target_value
    lo, hi = 0.02, 5.0
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return None  # couldn't bracket -- skip this trade rather than guess
    return brentq(f, lo, hi, xtol=1e-5)


def t_years_remaining(ts):
    close_today = ts.normalize() + pd.Timedelta(hours=16)
    minutes_left = max((close_today - ts).total_seconds() / 60.0, 0.0)
    return (minutes_left / SESSION_MINUTES) / TRADING_DAYS_PER_YEAR


def load_bars(ticker):
    df = pd.read_pickle(CACHE / f"{ticker}.pkl")
    df = df.between_time("09:30", "16:00")
    return df


def main():
    rows = pd.read_csv(HERE / "butterfly_gex_sr_enriched_v2.csv", parse_dates=["entry_date"])
    rows = rows[(rows["wing_step"] == 4) & (rows["status"] == "ok")].copy()

    bars = {t: load_bars(t) for t in ["SPY", "QQQ", "IWM"]}

    checkpoints_min = list(range(0, int(SESSION_MINUTES - 15) + 1, 15))  # 9:30+5 .. 15:45, every 15min
    stop_pcts = [0.50, 0.65, 0.80]

    results = {sp: [] for sp in stop_pcts}
    skipped = 0

    for _, row in rows.iterrows():
        ticker = row["ticker"]
        day_naive = row["entry_date"].normalize()
        day = day_naive.tz_localize(ET)
        k1, k2, k3 = row["k1"], row["k2"], row["k3"]
        entry_debit = row["entry_debit"]
        real_payoff = row["long_payoff"]

        day_bars = bars[ticker]
        day_bars = day_bars[(day_bars.index >= day + pd.Timedelta(hours=9, minutes=30))
                             & (day_bars.index <= day + pd.Timedelta(hours=16))]
        if day_bars.empty:
            skipped += 1
            continue

        entry_ts = day + pd.Timedelta(hours=9, minutes=35)
        entry_bar = day_bars[day_bars.index <= entry_ts]
        if entry_bar.empty:
            skipped += 1
            continue
        entry_spot = entry_bar["close"].iloc[-1]
        t0 = t_years_remaining(entry_ts)

        iv = solve_iv(entry_debit, entry_spot, k1, k2, k3, t0)
        if iv is None:
            skipped += 1
            continue

        # Walk checkpoints, find first stop-trigger time per threshold
        triggered_at = {sp: None for sp in stop_pcts}
        for mins in checkpoints_min:
            ts = day + pd.Timedelta(hours=9, minutes=30 + mins)
            if ts <= entry_ts:
                continue
            snap = day_bars[day_bars.index <= ts]
            if snap.empty:
                continue
            spot_t = snap["close"].iloc[-1]
            t_rem = t_years_remaining(ts)
            val_t = butterfly_value(spot_t, k1, k2, k3, t_rem, iv)
            loss_frac = (entry_debit - val_t) / entry_debit if entry_debit > 0 else 0.0
            for sp in stop_pcts:
                if triggered_at[sp] is None and loss_frac >= sp:
                    triggered_at[sp] = (ts, val_t)

        for sp in stop_pcts:
            if triggered_at[sp] is not None:
                _, val_t = triggered_at[sp]
                payoff = val_t - entry_debit
            else:
                payoff = real_payoff  # never triggered -- keep the REAL hold-to-close outcome
            results[sp].append({
                "ticker": ticker, "entry_date": day_naive, "payoff": payoff,
                "stopped": triggered_at[sp] is not None,
                "real_payoff": real_payoff,
            })

    print(f"Processed {len(rows)} real trade-days, skipped {skipped} (missing bars / unbracketable IV)")
    print()

    baseline = rows.copy()
    baseline["full_max_loss"] = baseline["long_payoff"] <= -0.99 * baseline["entry_debit"]
    print("=== BASELINE (real, hold-to-close, already deployed) ===")
    for scope, sub in [("ALL", baseline), ("QQQ", baseline[baseline.ticker == "QQQ"])]:
        print(f"  {scope:4s} n={len(sub):3d}  win={100*(sub.long_win).mean():5.1f}%  "
              f"mean=${100*sub.long_payoff.mean():+6.2f}/contract  "
              f"full_max_loss={100*sub.full_max_loss.mean():5.1f}%  "
              f"worst=${100*sub.long_payoff.min():+7.2f}")
    print()

    for sp in stop_pcts:
        df = pd.DataFrame(results[sp])
        df["win"] = df["payoff"] > 0
        df["full_max_loss"] = df["payoff"] <= -0.99 * rows.set_index(["ticker", "entry_date"]).loc[
            list(zip(df.ticker, df.entry_date)), "entry_debit"].values
        print(f"=== STOP at -{int(sp*100)}% of debit ===")
        for scope, sub in [("ALL", df), ("QQQ", df[df.ticker == "QQQ"])]:
            n_stopped = sub["stopped"].sum()
            print(f"  {scope:4s} n={len(sub):3d}  win={100*sub.win.mean():5.1f}%  "
                  f"mean=${100*sub.payoff.mean():+6.2f}/contract  "
                  f"full_max_loss={100*sub.full_max_loss.mean():5.1f}%  "
                  f"worst=${100*sub.payoff.min():+7.2f}  "
                  f"(stopped early on {n_stopped}/{len(sub)} days)")
        print()

    out = HERE / "butterfly_intraday_stoploss_results.csv"
    all_rows = []
    for sp in stop_pcts:
        for r in results[sp]:
            all_rows.append({**r, "stop_pct": sp})
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"Saved detail rows -> {out}")


if __name__ == "__main__":
    main()
