"""
Follow-up to butterfly_intraday_stoploss_test.py: that test only checked a
LOSS-side exit (flat close at -X% of debit, else hold to 15:47 no matter
how good it looks intraday -- matching the live script's current "no early
profit target (MVP)" design). This tests whether ALSO adding a PROFIT-side
early exit (take-profit) helps, on top of / instead of the stop-loss.

Same real data, same modeling approach and stated approximations as
butterfly_intraday_stoploss_test.py (read that docstring first) -- real
underlying price paths (Alpaca 1-min bars), Black-Scholes repricing with a
single flat IV per day back-solved to match the REAL recorded entry_debit,
15-min checkpoints, real hold-to-close payoff kept for any trade that never
triggers either rule.

Take-profit is defined symmetrically to the stop-loss: exit once the
modeled mark-to-market GAIN reaches >= take_pct of the entry debit paid
(e.g., take_pct=1.0 = "exit once up 100% of what was paid", i.e. position
value has roughly doubled). At each checkpoint, whichever of stop-loss /
take-profit triggers FIRST (chronologically) determines the early exit;
if neither triggers, real hold-to-close payoff is used, same as before.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

HERE = Path(__file__).parent
CACHE = HERE / "butterfly_intraday_cache"

ET = "America/New_York"
SESSION_MINUTES = 390.0
TRADING_DAYS_PER_YEAR = 252.0
R = 0.0


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
    def f(vol):
        return butterfly_value(spot, k1, k2, k3, t_years, vol) - target_value
    lo, hi = 0.02, 5.0
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-5)


def t_years_remaining(ts):
    close_today = ts.normalize() + pd.Timedelta(hours=16)
    minutes_left = max((close_today - ts).total_seconds() / 60.0, 0.0)
    return (minutes_left / SESSION_MINUTES) / TRADING_DAYS_PER_YEAR


def load_bars(ticker):
    df = pd.read_pickle(CACHE / f"{ticker}.pkl")
    return df.between_time("09:30", "16:00")


def main():
    rows = pd.read_csv(HERE / "butterfly_gex_sr_enriched_v2.csv", parse_dates=["entry_date"])
    rows = rows[(rows["wing_step"] == 4) & (rows["status"] == "ok")].copy()
    bars = {t: load_bars(t) for t in ["SPY", "QQQ", "IWM"]}
    checkpoints_min = list(range(0, int(SESSION_MINUTES - 15) + 1, 15))

    stop_pcts = [None, 0.50]
    take_pcts = [None, 0.50, 1.00, 1.50, 2.00]

    combos = [(sp, tp) for sp in stop_pcts for tp in take_pcts]
    results = {c: [] for c in combos}
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

        # Walk checkpoints once, record modeled value at each -- then evaluate
        # every stop/take combo against this single real price path.
        path = []
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
            path.append(val_t)

        for (sp, tp) in combos:
            exit_val = None
            for val_t in path:
                pnl_frac = (val_t - entry_debit) / entry_debit if entry_debit > 0 else 0.0
                if sp is not None and pnl_frac <= -sp:
                    exit_val = val_t
                    break
                if tp is not None and pnl_frac >= tp:
                    exit_val = val_t
                    break
            payoff = (exit_val - entry_debit) if exit_val is not None else real_payoff
            results[(sp, tp)].append({
                "ticker": ticker, "entry_date": day_naive, "payoff": payoff,
                "stopped_early": exit_val is not None,
            })

    print(f"Processed {len(rows)} real trade-days, skipped {skipped}\n")

    def label(sp, tp):
        s = f"stop-50%" if sp else "no-stop"
        t = f"take+{int(tp*100)}%" if tp else "no-take"
        return f"{s} / {t}"

    for (sp, tp) in combos:
        df = pd.DataFrame(results[(sp, tp)])
        df["win"] = df["payoff"] > 0
        print(f"=== {label(sp, tp)} ===")
        for scope in ["ALL", "QQQ", "SPY", "IWM"]:
            sub = df if scope == "ALL" else df[df.ticker == scope]
            if sub.empty:
                continue
            n_early = sub["stopped_early"].sum()
            print(f"  {scope:4s} n={len(sub):3d}  win={100*sub.win.mean():5.1f}%  "
                  f"mean=${100*sub.payoff.mean():+6.2f}/contract  worst=${100*sub.payoff.min():+7.2f}  "
                  f"(exited early on {n_early}/{len(sub)})")
        print()


if __name__ == "__main__":
    main()
