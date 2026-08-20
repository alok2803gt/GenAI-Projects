"""
Intraday-faithful replay of breakout_scanner.py's real firing logic, built to
close the 3 gaps identified in the daily-bar version
(breakout_volume_threshold_backtest.py):
  1. Volume: real time-of-day partial-volume projection, not final realized volume.
  2. Price timing: evaluated at each 3-min scan point intraday, not just at close.
  3. Outcome: same-day alert-to-close return, not next-day close-to-close.

Also replicates gates the daily-bar version missed entirely (found by reading
the real firing code end-to-end):
  - classify_state() 7-state Bollinger machine + per-ticker daily state ledger,
    reset each new trading day (breakout_scanner.py: _ticker_states.clear()).
  - "First scan of the day" silently initializes state -- never alerts.
  - F9: BREAKOUT must arrive via PRE-BREAKOUT (prev_state == PRE-BREAKOUT, or
    day-state currently PRE-BREAKOUT) unless this is literally the ticker's
    first transition of the day.
  - F10: BREAKOUT-via-PRE-BO must have spent >= min_pre_breakout_mins (15) in
    PRE-BREAKOUT first.
  - F1 backend regime gate: NEW alerts blocked when VIX >= 25 or SPY is below
    its own 200-day SMA (real historical VIX/SPY used here, not live).
  - Post-15:45 ET suppression.
  - Dedup: one alert per ticker per signal-level per day; never downgrade
    BREAKOUT -> PRE-BREAKOUT same day.

Data: Alpaca historical minute bars (real, ~1y, free with existing keys) for
intraday price/volume path; Alpaca daily bars (~3y, for rolling-indicator
lookback + the 252-day vol-ratio percentile warmup); yfinance VIX/SPY daily
for the F1 gate (same source already used elsewhere in this codebase).

Universe: watchlist.json (80 tickers, confirmed real scanner universe).

Outcome = alert_price (real intraday price at the moment all gates first
clear) vs that SAME trading day's real close -- exactly matching
alert_performance.eod_return_pct's definition.
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date as date_cls
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ET = ZoneInfo("America/New_York")

with open("scanner_config.json") as f:
    CFG = json.load(f)
with open("watchlist.json") as f:
    TICKERS = sorted(json.load(f).keys())

MINUTE_YEARS = 2.5     # widened from the first faithful run's 1.0y -- more
                        # samples at the high-percentile end specifically,
                        # where n was already thin (95th had only 62 BREAKOUT
                        # alerts in 1y). Confirmed Alpaca's free minute-bar
                        # history reaches back this far.
DAILY_YEARS  = 4.5      # keep the 252d rolling-percentile warmup fully clear
                        # of the widened minute window's start

PCT_B_BREAKOUT_MIN = 95
PCT_B_PRE_MIN = 65
RSI_PRE_MIN = 60
MIN_PRE_BREAKOUT_MINS = 15
# Extended past 95th (where the prior run's trend was still climbing) to find
# where it actually stops helping. Capped at 0.99: the percentile is computed
# from a 252-day trailing window (matches production's own real cap), so
# beyond ~99th the "threshold" is close to the single most extreme value ever
# observed in that window -- a fragile, near-overfit statistic, not a real
# tunable knob. Not going further than that on principle, not convenience.
PERCENTILES_TO_TEST = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
MULTIPLIERS_TO_TEST = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]

print(f"Universe: {len(TICKERS)} tickers")

# ── Pull real data ──────────────────────────────────────────────────────
client = StockHistoricalDataClient(CFG["alpaca_api_key"], CFG["alpaca_secret_key"])

end = datetime.now(ET) - timedelta(hours=1)   # free-tier Alpaca blocks querying "recent" SIP data
minute_start = end - timedelta(days=int(MINUTE_YEARS * 365))
daily_start  = end - timedelta(days=int(DAILY_YEARS * 365))

print(f"Pulling {MINUTE_YEARS}y minute bars for {len(TICKERS)} tickers (this takes a while)...")
minute_bars = {}
for i, tk in enumerate(TICKERS):
    try:
        req = StockBarsRequest(symbol_or_symbols=[tk], timeframe=TimeFrame.Minute,
                                start=minute_start, end=end)
        df = client.get_stock_bars(req).df
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[tk]
        df.index = df.index.tz_convert(ET)
        minute_bars[tk] = df
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(TICKERS)} minute pulls done")
    except Exception as exc:
        print(f"  {tk}: minute pull failed ({exc})")

print(f"Pulling {DAILY_YEARS}y daily bars...")
daily_bars = {}
for tk in TICKERS:
    try:
        req = StockBarsRequest(symbol_or_symbols=[tk], timeframe=TimeFrame.Day,
                                start=daily_start, end=end)
        df = client.get_stock_bars(req).df
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[tk]
        df.index = df.index.tz_convert(ET).normalize()
        daily_bars[tk] = df
    except Exception as exc:
        print(f"  {tk}: daily pull failed ({exc})")

print("Pulling VIX + SPY daily history for the F1 regime gate...")
vix = yf.download("^VIX", period="3y", interval="1d", auto_adjust=True, progress=False)["Close"]
spy = yf.download("SPY", period="3y", interval="1d", auto_adjust=True, progress=False)["Close"]
if isinstance(vix, pd.DataFrame): vix = vix.iloc[:, 0]
if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]
spy_sma200 = spy.rolling(200).mean()
spy_above_sma200 = (spy > spy_sma200)
vix.index = vix.index.tz_localize(None)
spy_above_sma200.index = spy_above_sma200.index.tz_localize(None)


def regime_ok(day: date_cls) -> bool:
    """F1: VIX >= 25 or SPY below its own SMA200 blocks new alerts that day."""
    ts = pd.Timestamp(day)
    v = vix.asof(ts)
    s = spy_above_sma200.asof(ts)
    if pd.isna(v) or v >= 25:
        return False
    if pd.isna(s) or not bool(s):
        return False
    return True


# ── Per-ticker simulation ───────────────────────────────────────────────
records = []
SCAN_INTERVAL_MIN = 3
CUTOFF_HOUR, CUTOFF_MIN = 15, 45

for tk_i, tk in enumerate(TICKERS):
    if tk not in minute_bars or tk not in daily_bars:
        continue
    mbars = minute_bars[tk]
    dbars = daily_bars[tk]
    if mbars.empty or len(dbars) < 260:
        continue

    dclose = dbars["close"]
    dvol   = dbars["volume"]

    # ── Precompute once per ticker (not once per day -- was the real
    # bottleneck): daily-close EWM gain/loss state, rolling vol-avg, rolling
    # vol-ratio percentiles, all shifted so day T only ever sees T-1 and
    # earlier. Also pre-group minute bars by day once instead of re-scanning
    # the full array on every day iteration. ──
    d_delta = dclose.diff()
    d_gain_ema_full = d_delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    d_loss_ema_full = (-d_delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d_vol20avg_full = dvol.rolling(20).mean()
    d_ratio_full = (dvol / d_vol20avg_full.shift(1))

    mbars_rth = mbars.between_time("09:30", "15:45")
    day_groups = {ts: g for ts, g in mbars_rth.groupby(mbars_rth.index.normalize())}
    dclose_days = dclose.index.normalize()

    trading_days = sorted(day_groups.keys())
    for day_ts in trading_days:
        day = day_ts.date()
        day_min = day_groups[day_ts]
        if day_min.empty:
            continue

        # searchsorted on the precomputed daily arrays -- O(log n), no rescan
        pos = int(np.searchsorted(dclose_days.values, np.datetime64(day_ts), side="left"))
        if pos < 252 or pos < 21:
            continue

        c19  = dclose.values[pos-19:pos]
        c49  = dclose.values[pos-49:pos]
        cTm1 = dclose.values[pos-1]

        prior_gain_ema = float(d_gain_ema_full.iloc[pos-1])
        prior_loss_ema = float(d_loss_ema_full.iloc[pos-1])
        avg_vol_20 = float(d_vol20avg_full.iloc[pos-1])

        ratio_hist = d_ratio_full.iloc[max(0, pos-252):pos].dropna()
        if len(ratio_hist) < 20:
            continue
        th = {p: float(ratio_hist.quantile(p)) for p in PERCENTILES_TO_TEST}

        scan_times = day_min.index[::SCAN_INTERVAL_MIN]  # ~every 3rd minute bar
        if len(scan_times) == 0:
            continue

        # Vectorized p_m-dependent indicators across all scan points today
        p = day_min.loc[scan_times, "close"].values.astype(float)
        cum_vol = day_min["volume"].cumsum().loc[scan_times].values.astype(float)

        sma20 = (c19.sum() + p) / 20.0
        var20 = ((c19**2).sum() + p**2) / 20.0 - sma20**2
        std20 = np.sqrt(np.clip(var20, 0, None))
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        band_w = np.where((upper - lower) > 0, upper - lower, np.nan)
        pct_b = (p - lower) / band_w * 100

        sma50 = (c49.sum() + p) / 50.0
        above_sma20 = p > sma20
        above_sma50 = p > sma50

        delta_today = p - cTm1
        gain_today = np.clip(delta_today, 0, None)
        loss_today = np.clip(-delta_today, 0, None)
        alpha = 1/14
        gain_ema = alpha * gain_today + (1 - alpha) * prior_gain_ema
        loss_ema = alpha * loss_today + (1 - alpha) * prior_loss_ema
        rs = gain_ema / np.clip(loss_ema, 1e-9, None)
        rsi = 100 - 100 / (1 + rs)

        minutes_elapsed = np.array([(t - t.normalize().replace(hour=9, minute=30)).total_seconds() / 60
                                     for t in scan_times])
        scale = np.where((minutes_elapsed >= 5) & (minutes_elapsed < 390),
                          np.minimum(3.0, 390 / np.clip(minutes_elapsed, 1e-9, None)), 1.0)
        vol_ratio = (cum_vol * scale) / avg_vol_20 if avg_vol_20 > 0 else np.full_like(p, np.nan)

        day_close = float(dclose[dclose.index.normalize() == day_ts].iloc[0]) if \
            (dclose.index.normalize() == day_ts).any() else float(day_min["close"].iloc[-1])

        reg_ok = regime_ok(day)
        n = len(scan_times)
        valid = ~(pd.isna(pct_b) | pd.isna(rsi) | pd.isna(vol_ratio))

        # ── Phase 1: state machine, computed ONCE (independent of percentile/
        # multiplier -- classify_state and F9/F10 eligibility depend only on
        # price, never on volume) ──
        state_arr      = np.empty(n, dtype=object)
        prev_state_arr = np.empty(n, dtype=object)
        pre_bo_since   = np.full(n, -1, dtype=int)
        is_first_scan  = np.zeros(n, dtype=bool)

        state = None
        pbsm  = -1
        first_scan = True
        for i in range(n):
            if not valid[i]:
                state_arr[i] = state
                prev_state_arr[i] = state
                pre_bo_since[i] = pbsm
                continue
            if pct_b[i] > 100: new_state = "EXTENDED"
            elif pct_b[i] >= 95: new_state = "BREAKOUT"
            elif pct_b[i] >= 75: new_state = "PRE-BREAKOUT"
            elif pct_b[i] >= 40: new_state = "NEUTRAL"
            elif pct_b[i] >= 25: new_state = "WEAKENING"
            elif pct_b[i] >= 0:  new_state = "PRE-BREAKDOWN"
            else: new_state = "BREAKDOWN"

            prev_state_arr[i] = state
            if first_scan:
                state = new_state
                if new_state == "PRE-BREAKOUT":
                    pbsm = i
                first_scan = False
                is_first_scan[i] = True
                state_arr[i] = state
                pre_bo_since[i] = pbsm
                continue
            if new_state == "PRE-BREAKOUT" and state != "PRE-BREAKOUT" and pbsm == -1:
                pbsm = i
            if new_state not in ("PRE-BREAKOUT", "BREAKOUT", "EXTENDED"):
                pbsm = -1
            state = new_state
            state_arr[i] = state
            pre_bo_since[i] = pbsm

        hh = np.array([t.hour for t in scan_times])
        mm = np.array([t.minute for t in scan_times])
        past_cutoff = (hh > CUTOFF_HOUR) | ((hh == CUTOFF_HOUR) & (mm >= CUTOFF_MIN))
        bullish = above_sma20 & above_sma50
        mins_in_pre = np.where(pre_bo_since >= 0, np.arange(n) - pre_bo_since, -1)

        breakout_eligible = valid & ~is_first_scan & ~past_cutoff & (pct_b > PCT_B_BREAKOUT_MIN) & \
            (prev_state_arr == "PRE-BREAKOUT") & \
            ((pre_bo_since == -1) | (mins_in_pre >= MIN_PRE_BREAKOUT_MINS))
        pre_eligible = valid & ~is_first_scan & ~past_cutoff & \
            (pct_b >= PCT_B_PRE_MIN) & (pct_b <= PCT_B_BREAKOUT_MIN) & (rsi >= RSI_PRE_MIN) & bullish

        # ── Phase 2: vectorized volume-gate + dedup check per combo tested ──
        combos = [(pl, 0.75) for pl in PERCENTILES_TO_TEST] + \
                 [(0.90, m) for m in MULTIPLIERS_TO_TEST if m != 0.75]
        for pctl, mult in combos:
            vth = th[pctl]
            bo_fires  = breakout_eligible & (vol_ratio >= vth)
            pre_fires = pre_eligible & (vol_ratio >= vth * mult)

            bo_idx  = np.flatnonzero(bo_fires)
            pre_idx = np.flatnonzero(pre_fires)
            first_bo  = bo_idx[0] if len(bo_idx) else None
            first_pre = pre_idx[0] if len(pre_idx) else None

            # Dedup matching production: whichever fires first wins that slot;
            # BREAKOUT before PRE-BREAKOUT suppresses the PRE-BREAKOUT alert
            # (no downgrade); PRE-BREAKOUT before BREAKOUT allows the later
            # BREAKOUT escalation.
            fires = []
            if first_pre is not None and (first_bo is None or first_pre < first_bo):
                fires.append(("PRE-BREAKOUT", first_pre))
            if first_bo is not None:
                fires.append(("BREAKOUT", first_bo))

            for sig, i in fires:
                if not reg_ok:
                    continue  # F1 regime gate -- these are all "new entries" (first-of-day)
                ret = (day_close - p[i]) / p[i] * 100
                records.append({
                    "ticker": tk, "day": str(day), "signal": sig,
                    "percentile": pctl, "multiplier": mult,
                    "alert_price": float(p[i]), "day_close": day_close,
                    "eod_return_pct": ret, "is_win": ret > 0,
                })
    if (tk_i + 1) % 10 == 0:
        print(f"  simulated {tk_i+1}/{len(TICKERS)} tickers, {len(records)} alerts so far")

print(f"\nTotal simulated alerts: {len(records)}")
recs = pd.DataFrame(records, columns=["ticker", "day", "signal", "percentile", "multiplier",
                                       "alert_price", "day_close", "eod_return_pct", "is_win"])
recs.to_csv("breakout_intraday_faithful_backtest_rows.csv", index=False)
if recs.empty:
    print("No alerts simulated -- stopping before aggregation.")
    raise SystemExit(0)


def stats(df):
    n = len(df)
    if n == 0:
        return (0, None, None)
    return (n, float((df["is_win"]).mean() * 100), float(df["eod_return_pct"].mean()))


print("\n" + "="*70)
print("ASSUMPTION 1 (intraday-faithful): does the 90th percentile matter?")
print("(multiplier held at production 0.75)")
print("="*70)
for p in PERCENTILES_TO_TEST:
    sub = recs[(recs["percentile"] == p) & (recs["multiplier"] == 0.75)]
    for sig in ("BREAKOUT", "PRE-BREAKOUT"):
        n, wr, ar = stats(sub[sub["signal"] == sig])
        if n:
            print(f"  pctl={p:.2f}  {sig:<13} n={n:<5} win_rate={wr:5.1f}%  avg_same_day_ret={ar:+.3f}%")
        else:
            print(f"  pctl={p:.2f}  {sig:<13} n=0")

print("\n" + "="*70)
print("ASSUMPTION 2 (intraday-faithful): does the 0.75 multiplier matter?")
print("(percentile held at production 0.90, PRE-BREAKOUT only)")
print("="*70)
for m in MULTIPLIERS_TO_TEST:
    sub = recs[(recs["percentile"] == 0.90) & (recs["multiplier"] == m) & (recs["signal"] == "PRE-BREAKOUT")]
    n, wr, ar = stats(sub)
    if n:
        print(f"  mult={m:.2f}  PRE-BREAKOUT  n={n:<5} win_rate={wr:5.1f}%  avg_same_day_ret={ar:+.3f}%")
    else:
        print(f"  mult={m:.2f}  PRE-BREAKOUT  n=0")

print("\nDone. Full row-level data: breakout_intraday_faithful_backtest_rows.csv")
