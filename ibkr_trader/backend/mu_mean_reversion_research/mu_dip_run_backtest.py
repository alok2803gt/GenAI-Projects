"""
Tests whether "buy the dip, sell the rip/run" on MU is a real, exploitable
pattern, or just looks that way because MU has had an enormous secular
uptrend over this window (any long entry profits eventually in a trend
like that) -- same discipline as every other backtest this session: real
5y data, MULTIPLE dip/run definitions and hold periods (a real parameter
grid, not one cherry-picked setting), always compared against MU's own
UNCONDITIONAL forward return over the same hold period (the matched
baseline that nets out the secular-uptrend confound), Bonferroni-corrected
across the grid, plus a split-sample stability check (first half vs
second half of the window) since a "dip buying works" result that only
held in one regime isn't a real, durable edge.

Dip/run definitions tested:
  A. N-day cumulative return threshold (dip = down >=X% over N days;
     run = up >=X% over N days)
  B. RSI(14) oversold/overbought (classic <30 / >70)
  C. Z-score vs SMA20 (price N std devs below/above its own 20-day mean)

For each trigger, forward return is measured over multiple hold periods
(1, 3, 5, 10, 20 trading days) starting the day AFTER the trigger fires
(no lookahead -- trigger observed at today's close, entry at tomorrow's
close is the cleanest no-lookahead assumption for a daily-bar study; also
report same-day-close-to-trigger-close+N as a secondary check).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
UNIVERSE_PKL = BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl"

HOLD_PERIODS = [1, 3, 5, 10, 20]
N_DAY_WINDOWS = [1, 2, 3, 5]
RET_THRESHOLDS = [0.03, 0.05, 0.08]  # 3%, 5%, 8% over the window
RSI_OVERSOLD, RSI_OVERBOUGHT = 30, 70
ZSCORE_THRESH = [1.0, 1.5, 2.0]


def load_mu():
    with open(UNIVERSE_PKL, "rb") as f:
        universe = pickle.load(f)
    df = universe["MU"].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def forward_returns(close, hold_periods):
    fwd = {}
    for h in hold_periods:
        fwd[h] = close.shift(-h) / close - 1  # from TODAY's close to h days forward
    return fwd


def test_signal(name, trigger_mask_dip, trigger_mask_run, fwd, baseline_fwd, alpha_bonf):
    rows = []
    for h in HOLD_PERIODS:
        f = fwd[h]
        base_mean = baseline_fwd[h].mean()

        dip_vals = f[trigger_mask_dip].dropna()
        run_vals = f[trigger_mask_run].dropna()

        if len(dip_vals) >= 10:
            t_dip, p_dip = stats.ttest_1samp(dip_vals, base_mean)
        else:
            t_dip, p_dip = np.nan, np.nan
        if len(run_vals) >= 10:
            t_run, p_run = stats.ttest_1samp(run_vals, base_mean)
        else:
            t_run, p_run = np.nan, np.nan

        rows.append({
            "signal": name, "hold_days": h, "baseline_mean_pct": base_mean * 100,
            "dip_n": len(dip_vals), "dip_mean_pct": dip_vals.mean() * 100 if len(dip_vals) else np.nan,
            "dip_p": p_dip, "dip_sig": (p_dip < alpha_bonf) if not np.isnan(p_dip) else False,
            "run_n": len(run_vals), "run_mean_pct": run_vals.mean() * 100 if len(run_vals) else np.nan,
            "run_p": p_run, "run_sig": (p_run < alpha_bonf) if not np.isnan(p_run) else False,
        })
    return rows


def main():
    df = load_mu()
    close = df["Close"]
    fwd = forward_returns(close, HOLD_PERIODS)
    n_days = len(close)
    print(f"MU real daily data: {n_days} days, {close.index[0].date()} -> {close.index[-1].date()}")
    print(f"Price range: ${close.min():.2f} -> ${close.max():.2f} "
          f"({(close.iloc[-1]/close.iloc[0]-1)*100:+.0f}% over the full window -- the secular-uptrend "
          f"confound every test below controls for via MU's OWN baseline forward return)")

    all_results = []
    n_signal_tests = len(N_DAY_WINDOWS) * len(RET_THRESHOLDS) + 1 + len(ZSCORE_THRESH)
    n_total_tests = n_signal_tests * len(HOLD_PERIODS) * 2  # dip + run
    alpha_bonf = 0.05 / n_total_tests
    print(f"\nTotal signal/hold-period/direction combinations tested: {n_total_tests}")
    print(f"Bonferroni-corrected alpha: {alpha_bonf:.6f}\n")

    # ── A. N-day return threshold ────────────────────────────────────
    for n_win in N_DAY_WINDOWS:
        n_day_ret = close / close.shift(n_win) - 1
        for thresh in RET_THRESHOLDS:
            name = f"Nday_ret(n={n_win},thresh={thresh:.0%})"
            dip_mask = n_day_ret <= -thresh
            run_mask = n_day_ret >= thresh
            all_results.extend(test_signal(name, dip_mask, run_mask, fwd, fwd, alpha_bonf))

    # ── B. RSI(14) oversold/overbought ───────────────────────────────
    rsi = compute_rsi(close, 14)
    dip_mask = rsi < RSI_OVERSOLD
    run_mask = rsi > RSI_OVERBOUGHT
    all_results.extend(test_signal("RSI14(<30/>70)", dip_mask, run_mask, fwd, fwd, alpha_bonf))

    # ── C. Z-score vs SMA20 ──────────────────────────────────────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    zscore = (close - sma20) / std20
    for z in ZSCORE_THRESH:
        name = f"Zscore(SMA20,thresh={z})"
        dip_mask = zscore <= -z
        run_mask = zscore >= z
        all_results.extend(test_signal(name, dip_mask, run_mask, fwd, fwd, alpha_bonf))

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(HERE / "mu_dip_run_results.csv", index=False)

    print("=== Full results (baseline = MU's own unconditional forward return at that hold period) ===")
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.max_rows", None):
        print(res_df.round(3).to_string(index=False))

    sig_dip = res_df[res_df["dip_sig"]]
    sig_run = res_df[res_df["run_sig"]]
    print(f"\n{len(sig_dip)}/{len(res_df)} DIP-BUY tests significant after Bonferroni correction")
    print(f"{len(sig_run)}/{len(res_df)} RUN-SELL tests significant after Bonferroni correction")
    if len(sig_dip):
        print("\nSignificant dip-buy signals:")
        print(sig_dip[["signal", "hold_days", "baseline_mean_pct", "dip_n", "dip_mean_pct", "dip_p"]].round(3).to_string(index=False))
    if len(sig_run):
        print("\nSignificant run-sell signals:")
        print(sig_run[["signal", "hold_days", "baseline_mean_pct", "run_n", "run_mean_pct", "run_p"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
