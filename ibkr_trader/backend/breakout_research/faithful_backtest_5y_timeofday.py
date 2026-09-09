"""
Re-runs faithful_backtest_extended.py's real gated intraday simulation
(same methodology: 7-state Bollinger machine, F9/F10 confirmation, F1
VIX/SPY regime gate, post-15:45 suppression, dedup -- unchanged) over the
FULL available 5-year Alpaca minute-bar window instead of 2.5 years, and
fixes a real units-mismatch bug found 2026-08-31: the live production
alert_history/alert_performance table's "vol_ratio" field is NOT the same
metric as this script's time-scaled vol_ratio -- it's an unscaled same-day
cumulative-volume fraction, which is mechanically small right after the
open regardless of actual relative volume (confirmed: live BREAKOUT rows
with vol_ratio<0.1 fire at a median of 6 minutes after the open; the
correctly time-scaled vol_ratio computed here never drops below 1.05
across 1,420 gated alerts in the 2.5y version). So "vol_ratio<0.1" in any
live-alert-derived combination actually means "fired within roughly the
first 10-15 minutes of the session" -- this script captures the real
mins_after_open at each fire so that combination can be tested as it
actually behaves, instead of via the mismatched vol_ratio field.

Also adds real sub-day forward returns (30/60/120 min after fire, via the
same minute bars already pulled) alongside the existing 1d/3d/5d closes,
reusing the intraday-path methodology from the 37-event live sample
(2026-08-31) at full 5-year scale.

Read-only against Alpaca/yfinance APIs and watchlist.json/scanner_config.json
(never touches breakout_scanner.py or any live file).
"""
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

ET = ZoneInfo("America/New_York")
BACKEND_DIR = r"C:\Projects\GenAI-Projects\ibkr_trader\backend"

with open(f"{BACKEND_DIR}\\scanner_config.json") as f:
    CFG = json.load(f)
with open(f"{BACKEND_DIR}\\watchlist.json") as f:
    TICKERS = sorted(json.load(f).keys())

VOL_PERCENTILE_OVERRIDES = {"SMCI": 0.90, "PLTR": 0.90}
VOL_THRESHOLD_PCT = 0.75
MIN_PRE_BREAKOUT_MINS = 15
PCT_B_BREAKOUT_MIN = 95
PCT_B_PRE_MIN = 65
RSI_PRE_MIN = 60

MINUTE_YEARS = 5.0   # confirmed 2026-08-31: Alpaca has real minute history back
DAILY_YEARS = 5.4    # to 2021-09-02 for the full universe, incl. SMCI/PLTR/RIVN/HOOD/COIN

print(f"Universe: {len(TICKERS)} tickers, {MINUTE_YEARS}y minute / {DAILY_YEARS}y daily window")

client = StockHistoricalDataClient(CFG["alpaca_api_key"], CFG["alpaca_secret_key"])
end = datetime.now(ET) - timedelta(hours=1)
minute_start = end - timedelta(days=int(MINUTE_YEARS * 365))
daily_start = end - timedelta(days=int(DAILY_YEARS * 365))

print(f"Pulling {MINUTE_YEARS}y minute bars for {len(TICKERS)} tickers...")
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
            print(f"  {i+1}/{len(TICKERS)} minute pulls done", flush=True)
    except Exception as exc:
        print(f"  {tk}: minute pull failed ({exc})", flush=True)

print(f"Pulling {DAILY_YEARS}y daily bars...", flush=True)
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
        print(f"  {tk}: daily pull failed ({exc})", flush=True)

print("Pulling VIX + SPY for F1 regime gate...", flush=True)
vix = yf.download("^VIX", period="5y", interval="1d", auto_adjust=True, progress=False)["Close"]
spy = yf.download("SPY", period="5y", interval="1d", auto_adjust=True, progress=False)["Close"]
if isinstance(vix, pd.DataFrame): vix = vix.iloc[:, 0]
if isinstance(spy, pd.DataFrame): spy = spy.iloc[:, 0]
spy_sma200 = spy.rolling(200).mean()
spy_above_sma200 = (spy > spy_sma200)
vix.index = vix.index.tz_localize(None)
spy_above_sma200.index = spy_above_sma200.index.tz_localize(None)


def regime_ok(day):
    ts = pd.Timestamp(day)
    v = vix.asof(ts)
    s = spy_above_sma200.asof(ts)
    if pd.isna(v) or v >= 25:
        return False
    if pd.isna(s) or not bool(s):
        return False
    return True


def compute_adx(high, low, close, period=14):
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


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

    mclose_full = mbars["close"]

    dclose = dbars["close"]
    dhigh = dbars["high"]
    dlow = dbars["low"]
    dopen = dbars["open"]
    dvol = dbars["volume"]

    d_delta = dclose.diff()
    d_gain_ema_full = d_delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    d_loss_ema_full = (-d_delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d_vol20avg_full = dvol.rolling(20).mean()
    d_ratio_full = (dvol / d_vol20avg_full.shift(1))
    d_adx_full = compute_adx(dhigh.values, dlow.values, dclose.values, 14)
    d_roll252high_full = dclose.rolling(252, min_periods=100).max()

    mbars_rth = mbars.between_time("09:30", "15:45")
    day_groups = {ts: g for ts, g in mbars_rth.groupby(mbars_rth.index.normalize())}
    dclose_days = dclose.index.normalize()
    trading_days = sorted(day_groups.keys())

    vol_pctl_ticker = VOL_PERCENTILE_OVERRIDES.get(tk, 0.95)

    for day_ts in trading_days:
        day = day_ts.date()
        day_min = day_groups[day_ts]
        if day_min.empty:
            continue

        pos = int(np.searchsorted(dclose_days.values, np.datetime64(day_ts), side="left"))
        if pos < 252 or pos < 21:
            continue

        c19 = dclose.values[pos - 19:pos]
        c49 = dclose.values[pos - 49:pos]
        cTm1 = dclose.values[pos - 1]
        openT = float(dopen.values[pos]) if pos < len(dopen) else float(day_min["open"].iloc[0])
        gap_pct = (openT - cTm1) / cTm1 * 100 if cTm1 else np.nan
        adx_val = float(d_adx_full.iloc[pos - 1]) if pos >= 1 else np.nan
        roll_high = float(d_roll252high_full.iloc[pos - 1]) if pos >= 1 else np.nan
        dist_52w_high = (cTm1 - roll_high) / roll_high * 100 if roll_high else np.nan

        prior_gain_ema = float(d_gain_ema_full.iloc[pos - 1])
        prior_loss_ema = float(d_loss_ema_full.iloc[pos - 1])
        avg_vol_20 = float(d_vol20avg_full.iloc[pos - 1])

        ratio_hist = d_ratio_full.iloc[max(0, pos - 252):pos].dropna()
        if len(ratio_hist) < 20:
            continue
        vth = float(ratio_hist.quantile(vol_pctl_ticker))

        scan_times = day_min.index[::SCAN_INTERVAL_MIN]
        if len(scan_times) == 0:
            continue

        p = day_min.loc[scan_times, "close"].values.astype(float)
        cum_vol = day_min["volume"].cumsum().loc[scan_times].values.astype(float)

        sma20 = (c19.sum() + p) / 20.0
        var20 = ((c19 ** 2).sum() + p ** 2) / 20.0 - sma20 ** 2
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
        alpha = 1 / 14
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

        state_arr = np.empty(n, dtype=object)
        prev_state_arr = np.empty(n, dtype=object)
        pre_bo_since = np.full(n, -1, dtype=int)
        is_first_scan = np.zeros(n, dtype=bool)

        state = None
        pbsm = -1
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
            elif pct_b[i] >= 0: new_state = "PRE-BREAKDOWN"
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

        bo_fires = breakout_eligible & (vol_ratio >= vth)
        pre_fires = pre_eligible & (vol_ratio >= vth * VOL_THRESHOLD_PCT)

        bo_idx = np.flatnonzero(bo_fires)
        pre_idx = np.flatnonzero(pre_fires)
        first_bo = bo_idx[0] if len(bo_idx) else None
        first_pre = pre_idx[0] if len(pre_idx) else None

        fires = []
        if first_pre is not None and (first_bo is None or first_pre < first_bo):
            fires.append(("PRE-BREAKOUT", first_pre))
        if first_bo is not None:
            fires.append(("BREAKOUT", first_bo))

        for sig, i in fires:
            if not reg_ok:
                continue
            t_fire = scan_times[i]
            ret_eod = (day_close - p[i]) / p[i] * 100
            fwd = {}
            for nd in (1, 3, 5):
                fpos = pos + nd
                fwd[f"ret_{nd}d"] = ((float(dclose.values[fpos]) - p[i]) / p[i] * 100
                                      if fpos < len(dclose.values) else None)
            intraday = {}
            for horizon_min in (30, 60, 120):
                target = t_fire + pd.Timedelta(minutes=horizon_min)
                try:
                    px = mclose_full.asof(target)
                    if pd.isna(px) or target.date() != t_fire.date():
                        intraday[f"ret_{horizon_min}m"] = None
                    else:
                        intraday[f"ret_{horizon_min}m"] = (float(px) - p[i]) / p[i] * 100
                except Exception:
                    intraday[f"ret_{horizon_min}m"] = None
            records.append({
                "ticker": tk, "day": str(day), "signal": sig,
                "alert_price": float(p[i]), "day_close": day_close,
                "eod_return_pct": ret_eod, "is_win": ret_eod > 0,
                "rsi": float(rsi[i]), "pct_b": float(pct_b[i]), "vol_ratio": float(vol_ratio[i]),
                "mins_after_open": float(minutes_elapsed[i]),
                "adx": adx_val, "gap_pct": gap_pct, "dist_52w_high": dist_52w_high,
                "mins_in_pre_breakout": int(mins_in_pre[i]) if mins_in_pre[i] >= 0 else None,
                **fwd, **intraday,
            })
    if (tk_i + 1) % 10 == 0:
        print(f"  simulated {tk_i+1}/{len(TICKERS)} tickers, {len(records)} real gated alerts so far", flush=True)

print(f"\nTotal simulated REAL gated alerts (production settings, {MINUTE_YEARS}y window): {len(records)}")
recs = pd.DataFrame(records)
recs.to_csv(r"C:\Projects\GenAI-Projects\ibkr_trader\backend\breakout_research\faithful_5y_rows.csv", index=False)
print("Saved to faithful_5y_rows.csv")

print("\n=== date range achieved ===")
if len(recs):
    print(recs["day"].min(), "to", recs["day"].max())
    print(recs["signal"].value_counts())
