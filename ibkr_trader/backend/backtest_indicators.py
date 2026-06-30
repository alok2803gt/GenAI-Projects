"""
backtest_indicators.py  —  Breakout signal backtester (2-3 years)
-----------------------------------------------------------------
Downloads 2.5 years of daily OHLCV for the full scanner universe,
replays the breakout-signal logic on each historical day, computes
1-day / 3-day / 5-day forward returns for every simulated signal,
then runs correlation + quartile analysis to rank which technical
indicators actually predict positive returns.

Usage:
    python backtest_indicators.py               # full 2.5-year run
    python backtest_indicators.py --years 3     # extend to 3 years
    python backtest_indicators.py --tickers AAPL MSFT NVDA  # spot-check
    python backtest_indicators.py --no-save     # skip CSV output
"""

import argparse
import os
import sys
import warnings
from datetime import date, timedelta

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError with box chars in comments)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Universe (mirrors breakout_scanner.py CURATED_TICKERS) ──────────────────
ALL_TICKERS: list[str] = sorted(set([
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "GLD", "TLT", "ARKK",
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX",
    "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SMCI",
    "CRM", "NOW", "ADBE", "ORCL", "SNOW", "PANW", "CRWD", "ZS", "DDOG", "NET",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "V", "MA", "AXP", "TFC",
    "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO", "DHR", "ISRG", "VRTX", "GILD", "BMY",
    "HD", "MCD", "SBUX", "NKE", "LOW", "TGT", "COST", "BKNG", "LULU",
    "PG", "KO", "PEP", "WMT",
    "XOM", "CVX", "COP", "SLB", "MPC", "VLO", "OXY",
    "BA", "GE", "CAT", "HON", "RTX", "LMT", "FDX", "UPS", "DE", "UAL",
    "DIS", "CMCSA", "VZ", "T",
    "COIN", "PLTR", "UBER", "RIVN", "ROKU", "HOOD", "SOFI", "PYPL", "XYZ", "IBM",
    "RBLX", "RCL", "ABNB",
]))

# Sector ETFs — downloaded alongside tickers to build market-context indicators
SECTOR_ETFS: list[str] = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLC", "XLB", "XLRE", "XLU",
]

# Maps each ticker → its sector ETF for sector-relative-strength indicator
SECTOR_MAP: dict[str, str] = {
    **dict.fromkeys(["AAPL","MSFT","NVDA","AMD","INTC","QCOM","AVGO","TXN","MU",
                     "AMAT","LRCX","KLAC","MRVL","SMCI","CRM","NOW","ADBE","ORCL",
                     "SNOW","PANW","CRWD","ZS","DDOG","NET","IBM","COIN","PLTR","PYPL","XYZ"], "XLK"),
    **dict.fromkeys(["JPM","BAC","WFC","GS","MS","C","BLK","SCHW","V","MA","AXP","TFC","HOOD","SOFI"], "XLF"),
    **dict.fromkeys(["JNJ","UNH","LLY","PFE","ABBV","MRK","TMO","DHR","ISRG","VRTX","GILD","BMY"], "XLV"),
    **dict.fromkeys(["HD","MCD","SBUX","NKE","LOW","TGT","COST","BKNG","LULU",
                     "TSLA","AMZN","RIVN","RBLX","RCL","ABNB"], "XLY"),
    **dict.fromkeys(["PG","KO","PEP","WMT"], "XLP"),
    **dict.fromkeys(["XOM","CVX","COP","SLB","MPC","VLO","OXY"], "XLE"),
    **dict.fromkeys(["BA","GE","CAT","HON","RTX","LMT","FDX","UPS","DE","UAL","UBER"], "XLI"),
    **dict.fromkeys(["DIS","CMCSA","VZ","T","GOOGL","META","NFLX","ROKU"], "XLC"),
}

# Scanner thresholds (match scanner_config.json)
BREAKOUT_PCT_B_MIN     = 95.0
PRE_BREAKOUT_PCT_B_MIN = 75.0
VOL_RATIO_MIN          = 0.75
PRE_BREAKOUT_RSI_MIN   = 60.0

# Indicator lookback periods
BB_PERIOD   = 20
RSI_PERIOD  = 14
ADX_PERIOD  = 14
ATR_PERIOD  = 14
VOL_PERIOD  = 21   # volume avg
MIN_HISTORY = 60   # minimum bars before computing indicators

# ── Indicator math ───────────────────────────────────────────────────────────

def wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's EMA smoothing (used by RSI, ATR, ADX)."""
    out = np.full(len(arr), np.nan)
    # first non-nan start
    valid = np.where(~np.isnan(arr))[0]
    if len(valid) < period:
        return out
    start = valid[0]
    if start + period > len(arr):
        return out
    out[start + period - 1] = np.nanmean(arr[start : start + period])
    for i in range(start + period, len(arr)):
        if not np.isnan(arr[i]):
            out[i] = out[i - 1] * (period - 1) / period + arr[i] / period
    return out


def compute_ticker_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators for a single ticker's OHLCV DataFrame.
    Input columns: Open, High, Low, Close, Volume (DatetimeIndex).
    Returns the same DataFrame with indicator columns appended.
    """
    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    volume = df["Volume"].values.astype(float)
    n      = len(close)

    # ── Bollinger Bands (%B, bandwidth) ──────────────────────────────────────
    pct_b     = np.full(n, np.nan)
    bb_bwidth = np.full(n, np.nan)
    for i in range(BB_PERIOD - 1, n):
        sl = close[i - BB_PERIOD + 1 : i + 1]
        mid = sl.mean()
        std = sl.std(ddof=0)
        if std > 0:
            upper = mid + 2 * std
            lower = mid - 2 * std
            pct_b[i]     = (close[i] - lower) / (upper - lower) * 100
            bb_bwidth[i] = (upper - lower) / mid * 100   # bandwidth as % of mid

    # ── RSI (Wilder EMA method) ───────────────────────────────────────────────
    rsi_vals = np.full(n, np.nan)
    delta = np.diff(close, prepend=np.nan)
    gains  = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = wilder_smooth(gains,  RSI_PERIOD)
    avg_loss = wilder_smooth(losses, RSI_PERIOD)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi_vals = np.where(~np.isnan(avg_gain), 100 - 100 / (1 + rs), np.nan)

    # ── ATR and ADX ──────────────────────────────────────────────────────────
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low,
                            np.abs(high - prev_close),
                            np.abs(low  - prev_close)])

    prev_high = np.roll(high, 1); prev_high[0] = high[0]
    prev_low  = np.roll(low,  1); prev_low[0]  = low[0]
    up_move   = high - prev_high
    down_move = prev_low - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = minus_dm[0] = 0.0  # no movement on first bar

    atr_arr   = wilder_smooth(tr,       ATR_PERIOD)
    plus_di   = wilder_smooth(plus_dm,  ADX_PERIOD)
    minus_di  = wilder_smooth(minus_dm, ADX_PERIOD)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di_pct  = np.where(atr_arr > 0, 100 * plus_di  / atr_arr, np.nan)
        minus_di_pct = np.where(atr_arr > 0, 100 * minus_di / atr_arr, np.nan)
        di_sum  = plus_di_pct + minus_di_pct
        dx      = np.where(di_sum > 0, 100 * np.abs(plus_di_pct - minus_di_pct) / di_sum, 0.0)
    adx_arr = wilder_smooth(dx, ADX_PERIOD)

    # ATR as % of price
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_pct = np.where(close > 0, atr_arr / close * 100, np.nan)

    # ── Volume ratio (current vol / 21-day avg of prior days) ────────────────
    vol_ratio = np.full(n, np.nan)
    for i in range(VOL_PERIOD, n):
        avg_vol = volume[i - VOL_PERIOD : i].mean()
        vol_ratio[i] = volume[i] / avg_vol if avg_vol > 0 else np.nan

    # ── Projected intraday volume (daily bars: volume IS the full day) ────────
    # vol_ratio already represents full-day ratio for daily bars

    # ── Volume trend: slope of last 5 days vol vs avg (accumulation signal) ──
    vol_trend = np.full(n, np.nan)
    for i in range(5, n):
        vols_5 = volume[i - 4 : i + 1]  # last 5 days
        avg_5  = vols_5.mean()
        avg_20_start = max(0, i - 20)
        avg_20 = volume[avg_20_start : i].mean()
        vol_trend[i] = avg_5 / avg_20 if avg_20 > 0 else 1.0

    # ── Distance from 52-week high (%): negative means above ──────────────────
    dist_52wk = np.full(n, np.nan)
    look52 = 252
    for i in range(look52, n):
        high_52 = high[i - look52 : i].max()
        dist_52wk[i] = (close[i] / high_52 - 1) * 100  # 0 = at high, -10 = 10% below

    # ── BB tightness: avg daily range of last 5d vs 20d ATR ──────────────────
    # Low ratio = tight base before breakout
    tightness = np.full(n, np.nan)
    for i in range(20, n):
        range_5  = (high[i - 4 : i + 1] - low[i - 4 : i + 1]).mean()
        range_20 = (high[i - 19 : i + 1] - low[i - 19 : i + 1]).mean()
        tightness[i] = range_5 / range_20 if range_20 > 0 else 1.0

    # ── Close position in day's range (0=at low, 100=at high) ────────────────
    day_range  = high - low
    close_pos  = np.where(day_range > 0, (close - low) / day_range * 100, 50.0)

    # ── SMA 20/50/200 ─────────────────────────────────────────────────────────
    sma20  = pd.Series(close).rolling(20,  min_periods=20 ).mean().values
    sma50  = pd.Series(close).rolling(50,  min_periods=50 ).mean().values
    sma200 = pd.Series(close).rolling(200, min_periods=200).mean().values

    # ── Prior 5-day return (momentum entering signal) ─────────────────────────
    ret_5d_prior = np.full(n, np.nan)
    ret_5d_prior[5:] = (close[5:] / close[:-5] - 1) * 100

    # ── 20-day return (regime context) ───────────────────────────────────────
    ret_20d = np.full(n, np.nan)
    ret_20d[20:] = (close[20:] / close[:-20] - 1) * 100

    # ── Assemble ──────────────────────────────────────────────────────────────
    result = df.copy()
    result["pct_b"]        = pct_b
    result["rsi"]          = rsi_vals
    result["vol_ratio"]    = vol_ratio
    result["adx"]          = adx_arr
    result["atr_pct"]      = atr_pct
    result["bb_bwidth"]    = bb_bwidth
    result["close_pos"]    = close_pos
    result["dist_52wk"]    = dist_52wk
    result["tightness"]    = tightness
    result["vol_trend"]    = vol_trend
    result["ret_5d_prior"] = ret_5d_prior
    result["ret_20d"]      = ret_20d
    result["sma20"]        = sma20
    result["sma50"]        = sma50
    result["sma200"]       = sma200
    return result


def compute_pct_b_series(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Fast %B for sector ETF context columns."""
    pct_b = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        sl  = close[i - period + 1 : i + 1]
        mid = sl.mean()
        std = sl.std(ddof=0)
        if std > 0:
            upper      = mid + 2 * std
            lower      = mid - 2 * std
            pct_b[i]   = (close[i] - lower) / (upper - lower) * 100
    return pct_b


# ── Signal simulation ────────────────────────────────────────────────────────

def classify_signal(row: pd.Series) -> str | None:
    """Replicate scanner signal logic from breakout_scanner.py classify_signal()."""
    pct_b     = row["pct_b"]
    rsi_val   = row["rsi"]
    vol_ratio = row["vol_ratio"]
    above_sma20 = row["Close"] > row["sma20"]
    above_sma50 = row["Close"] > row["sma50"]

    if pd.isna(pct_b) or pd.isna(vol_ratio):
        return None
    if vol_ratio < VOL_RATIO_MIN:
        return None
    if not (above_sma20 and above_sma50):
        return None

    if pct_b >= BREAKOUT_PCT_B_MIN:
        return "BREAKOUT"
    if (pct_b >= PRE_BREAKOUT_PCT_B_MIN
            and not pd.isna(rsi_val)
            and rsi_val >= PRE_BREAKOUT_RSI_MIN):
        return "PRE-BREAKOUT"
    return None


# ── Main backtest ────────────────────────────────────────────────────────────

def run_backtest(tickers: list[str], years: float = 2.5) -> pd.DataFrame:
    end_date   = date.today()
    start_date = end_date - timedelta(days=int(years * 365) + 60)  # +60 for warmup
    print(f"\n{'='*65}")
    print(f"  BREAKOUT BACKTEST  |  {len(tickers)} tickers  |  {years} years")
    print(f"  {start_date}  ->  {end_date}")
    print(f"{'='*65}\n")

    # ── Download all data in one batch ────────────────────────────────────────
    all_download = sorted(set(tickers) | set(SECTOR_ETFS))
    print(f"Downloading {len(all_download)} tickers from yfinance...")
    raw = yf.download(
        all_download,
        start=str(start_date),
        end=str(end_date),
        interval="1d",
        auto_adjust=True,
        progress=True,
        group_by="ticker",
        threads=True,
    )
    print(f"Download complete.\n")

    def get_ohlcv(ticker: str) -> pd.DataFrame | None:
        try:
            if len(all_download) == 1:
                df = raw[["Open","High","Low","Close","Volume"]].copy()
            else:
                df = raw[ticker][["Open","High","Low","Close","Volume"]].copy()
            df = df.dropna(subset=["Close"])
            return df if len(df) >= MIN_HISTORY else None
        except Exception:
            return None

    # ── Pre-compute sector ETF %B series ─────────────────────────────────────
    sector_pct_b: dict[str, pd.Series] = {}
    for etf in SECTOR_ETFS:
        df_etf = get_ohlcv(etf)
        if df_etf is not None:
            pb = compute_pct_b_series(df_etf["Close"].values)
            sector_pct_b[etf] = pd.Series(pb, index=df_etf.index)

    spy_pct_b_series = sector_pct_b.get("SPY")

    # ── Process each ticker ───────────────────────────────────────────────────
    all_signals: list[dict] = []
    skipped = []

    for i, ticker in enumerate(tickers):
        df = get_ohlcv(ticker)
        if df is None:
            skipped.append(ticker)
            continue

        try:
            df = compute_ticker_indicators(df)
        except Exception as e:
            skipped.append(ticker)
            continue

        close_arr = df["Close"].values
        dates_arr = df.index

        # SPY %B aligned to this ticker's date index
        if spy_pct_b_series is not None:
            df["spy_pct_b"] = spy_pct_b_series.reindex(df.index)
        else:
            df["spy_pct_b"] = np.nan

        # Sector ETF %B
        sector_etf = SECTOR_MAP.get(ticker, "SPY")
        if sector_etf in sector_pct_b:
            df["sector_pct_b"] = sector_pct_b[sector_etf].reindex(df.index)
        else:
            df["sector_pct_b"] = np.nan

        # Simulate signals: skip first MIN_HISTORY bars (warmup)
        warmup_date = dates_arr[MIN_HISTORY]
        signal_rows = df[df.index >= warmup_date]

        for idx_pos, (dt, row) in enumerate(signal_rows.iterrows()):
            sig = classify_signal(row)
            if sig is None:
                continue

            # Find position in the full df for forward-return lookups
            pos = df.index.get_loc(dt)

            # Forward returns: 1d, 3d, 5d (next close / signal-day close - 1)
            signal_close = close_arr[pos]
            ret_1d = ret_3d = ret_5d = np.nan
            if pos + 1 < len(close_arr):
                ret_1d = (close_arr[pos + 1] / signal_close - 1) * 100
            if pos + 3 < len(close_arr):
                ret_3d = (close_arr[pos + 3] / signal_close - 1) * 100
            if pos + 5 < len(close_arr):
                ret_5d = (close_arr[pos + 5] / signal_close - 1) * 100

            all_signals.append({
                "date":         dt.strftime("%Y-%m-%d"),
                "ticker":       ticker,
                "signal_type":  sig,
                # forward returns
                "ret_1d":       round(ret_1d, 4) if not np.isnan(ret_1d) else np.nan,
                "ret_3d":       round(ret_3d, 4) if not np.isnan(ret_3d) else np.nan,
                "ret_5d":       round(ret_5d, 4) if not np.isnan(ret_5d) else np.nan,
                # signal-day indicators
                "pct_b":        round(float(row["pct_b"]), 2),
                "rsi":          round(float(row["rsi"]),   2) if not pd.isna(row["rsi"]) else np.nan,
                "vol_ratio":    round(float(row["vol_ratio"]), 3),
                "adx":          round(float(row["adx"]),   2) if not pd.isna(row["adx"]) else np.nan,
                "atr_pct":      round(float(row["atr_pct"]), 3) if not pd.isna(row["atr_pct"]) else np.nan,
                "bb_bwidth":    round(float(row["bb_bwidth"]), 3) if not pd.isna(row["bb_bwidth"]) else np.nan,
                "close_pos":    round(float(row["close_pos"]), 1),
                "dist_52wk":    round(float(row["dist_52wk"]), 2) if not pd.isna(row["dist_52wk"]) else np.nan,
                "tightness":    round(float(row["tightness"]), 3) if not pd.isna(row["tightness"]) else np.nan,
                "vol_trend":    round(float(row["vol_trend"]), 3) if not pd.isna(row["vol_trend"]) else np.nan,
                "ret_5d_prior": round(float(row["ret_5d_prior"]), 3) if not pd.isna(row["ret_5d_prior"]) else np.nan,
                "ret_20d":      round(float(row["ret_20d"]), 3) if not pd.isna(row["ret_20d"]) else np.nan,
                "spy_pct_b":    round(float(row["spy_pct_b"]), 2) if not pd.isna(row["spy_pct_b"]) else np.nan,
                "sector_pct_b": round(float(row["sector_pct_b"]), 2) if not pd.isna(row["sector_pct_b"]) else np.nan,
            })

        if (i + 1) % 20 == 0 or (i + 1) == len(tickers):
            print(f"  [{i+1:3d}/{len(tickers)}] processed — {len(all_signals):,} signals so far")

    if skipped:
        print(f"\nSkipped (insufficient data): {', '.join(skipped)}")

    df_signals = pd.DataFrame(all_signals)
    print(f"\nTotal simulated signals: {len(df_signals):,}  across {df_signals['ticker'].nunique()} tickers")
    return df_signals


# ── Analysis ─────────────────────────────────────────────────────────────────

INDICATORS = [
    ("pct_b",        "%B",                  "Higher = deeper into breakout zone"),
    ("rsi",          "RSI-14",              "Sweet spot vs overbought"),
    ("vol_ratio",    "Volume Ratio",        "Today vs 21-day avg"),
    ("adx",          "ADX-14",              "Trend strength (>25 = trending)"),
    ("atr_pct",      "ATR % of Price",      "Volatility; lower = tighter stock"),
    ("bb_bwidth",    "BB Bandwidth %",      "Band width; lower = squeeze/tight base"),
    ("close_pos",    "Close Position %",    "Day's close in H-L range (100=at high)"),
    ("dist_52wk",    "Dist from 52wk High %","0=at high, negative=below high"),
    ("tightness",    "5d/20d Range Ratio",  "Low = tight base before breakout"),
    ("vol_trend",    "5d Vol Trend",        "5d avg vol vs 20d; >1 = accumulating"),
    ("ret_5d_prior", "Prior 5d Return %",   "Momentum entering signal day"),
    ("spy_pct_b",    "SPY %B",              "Market breadth tailwind"),
    ("sector_pct_b", "Sector ETF %B",       "Sector tailwind for this signal"),
]


def quartile_analysis(df: pd.DataFrame, indicator: str, target: str = "ret_1d") -> pd.DataFrame:
    """Return win-rate and avg return by quartile for one indicator."""
    sub = df.dropna(subset=[indicator, target]).copy()
    if len(sub) < 40:
        return pd.DataFrame()
    try:
        sub["q"] = pd.qcut(sub[indicator], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    except ValueError:
        return pd.DataFrame()

    result = sub.groupby("q", observed=True).agg(
        count     = (target, "count"),
        avg_ret   = (target, "mean"),
        win_rate  = (target, lambda x: (x > 0).mean() * 100),
        med_ret   = (target, "median"),
    ).reset_index()
    result.columns = ["quartile", "count", "avg_ret", "win_rate", "median_ret"]
    return result


def run_analysis(df: pd.DataFrame, save_dir: str | None = None) -> None:
    if len(df) < 50:
        print("Not enough signals for meaningful analysis.")
        return

    for target_col, target_label in [("ret_1d", "1-day"), ("ret_3d", "3-day"), ("ret_5d", "5-day")]:
        sub = df.dropna(subset=[target_col])
        wins = (sub[target_col] > 0).mean() * 100
        avg  = sub[target_col].mean()
        med  = sub[target_col].median()
        print(f"  {target_label} returns: n={len(sub):,}  avg={avg:+.2f}%  median={med:+.2f}%  win_rate={wins:.1f}%")

    # ── Correlation table ─────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  INDICATOR CORRELATIONS  (Pearson r  |  Spearman ρ)")
    print(f"  {'Indicator':<22}  {'1d r':>7}  {'1d ρ':>7}  {'3d r':>7}  {'3d ρ':>7}  {'5d r':>7}  {'n':>6}")
    print(f"{'-'*65}")

    corr_rows = []
    for col, label, _ in INDICATORS:
        row = {"indicator": label}
        for tc in ["ret_1d", "ret_3d", "ret_5d"]:
            sub = df.dropna(subset=[col, tc])
            if len(sub) < 30:
                row[f"r_{tc}"] = row[f"rho_{tc}"] = np.nan
                continue
            r   = sub[col].corr(sub[tc], method="pearson")
            rho = sub[col].corr(sub[tc], method="spearman")
            row[f"r_{tc}"]   = r
            row[f"rho_{tc}"] = rho
        row["n"] = len(df.dropna(subset=[col]))
        corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows)
    for _, row in corr_df.iterrows():
        r1  = f"{row['r_ret_1d']:+.3f}"   if not pd.isna(row.get("r_ret_1d"))   else "  n/a "
        rh1 = f"{row['rho_ret_1d']:+.3f}" if not pd.isna(row.get("rho_ret_1d")) else "  n/a "
        r3  = f"{row['r_ret_3d']:+.3f}"   if not pd.isna(row.get("r_ret_3d"))   else "  n/a "
        rh3 = f"{row['rho_ret_3d']:+.3f}" if not pd.isna(row.get("rho_ret_3d")) else "  n/a "
        r5  = f"{row['r_ret_5d']:+.3f}"   if not pd.isna(row.get("r_ret_5d"))   else "  n/a "
        n   = int(row["n"]) if not pd.isna(row["n"]) else 0
        print(f"  {row['indicator']:<22}  {r1:>7}  {rh1:>7}  {r3:>7}  {rh3:>7}  {r5:>7}  {n:>6,}")

    # ── Quartile deep-dives for top indicators ────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  QUARTILE ANALYSIS  (1-day forward return by indicator quartile)")
    print(f"{'-'*65}")

    quartile_results = {}
    for col, label, desc in INDICATORS:
        qt = quartile_analysis(df, col, "ret_1d")
        if qt.empty:
            continue
        quartile_results[col] = qt
        print(f"\n  {label}  —  {desc}")
        print(f"  {'Quartile':<14}  {'n':>5}  {'Win%':>6}  {'Avg Ret':>8}  {'Median':>8}")
        for _, r in qt.iterrows():
            flag = " ◀ best" if r["avg_ret"] == qt["avg_ret"].max() else ""
            flag = " ◀ worst" if r["avg_ret"] == qt["avg_ret"].min() else flag
            print(f"  {r['quartile']:<14}  {r['count']:>5,}  {r['win_rate']:>5.1f}%  {r['avg_ret']:>+7.2f}%  {r['median_ret']:>+7.2f}%{flag}")

    # ── Signal-type breakdown ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  BY SIGNAL TYPE")
    print(f"{'-'*65}")
    for sig, grp in df.groupby("signal_type"):
        valid = grp.dropna(subset=["ret_1d"])
        if len(valid) < 5:
            continue
        wr  = (valid["ret_1d"] > 0).mean() * 100
        avg = valid["ret_1d"].mean()
        print(f"  {sig:<16}  n={len(valid):>5,}  win_rate={wr:.1f}%  avg_1d={avg:+.2f}%")

    # ── Winner profile: what does a typical winning signal look like? ─────────
    print(f"\n{'─'*65}")
    print(f"  WINNER PROFILE  (median indicator values: wins vs losses)")
    print(f"{'-'*65}")
    wins_df   = df[df["ret_1d"] > 0]
    losses_df = df[df["ret_1d"] <= 0]
    print(f"  {'Indicator':<22}  {'Wins median':>12}  {'Losses median':>13}  {'Diff':>8}")
    for col, label, _ in INDICATORS:
        wm = wins_df[col].median()
        lm = losses_df[col].median()
        if pd.isna(wm) or pd.isna(lm):
            continue
        diff = wm - lm
        flag = "  *** HIGH SIGNAL" if abs(diff / max(abs(lm), 1e-6)) > 0.10 else ""
        print(f"  {label:<22}  {wm:>12.2f}  {lm:>13.2f}  {diff:>+8.2f}{flag}")

    # ── Top-decile filter: what threshold maximizes win rate? ─────────────────
    print(f"\n{'─'*65}")
    print(f"  TOP-DECILE FILTER THRESHOLDS  (90th pctile = highest-conviction signals)")
    print(f"{'-'*65}")
    top_decile_filters = {}
    for col, label, _ in INDICATORS:
        sub = df.dropna(subset=[col, "ret_1d"])
        if len(sub) < 50:
            continue
        p90 = sub[col].quantile(0.90)
        p10 = sub[col].quantile(0.10)
        # test both directions: high values and low values
        top10_high = sub[sub[col] >= p90]
        top10_low  = sub[sub[col] <= p10]
        if len(top10_high) >= 10:
            wr_h = (top10_high["ret_1d"] > 0).mean() * 100
            avg_h = top10_high["ret_1d"].mean()
        else:
            wr_h = avg_h = np.nan
        if len(top10_low) >= 10:
            wr_l = (top10_low["ret_1d"] > 0).mean() * 100
            avg_l = top10_low["ret_1d"].mean()
        else:
            wr_l = avg_l = np.nan
        baseline_wr  = (sub["ret_1d"] > 0).mean() * 100
        baseline_avg = sub["ret_1d"].mean()

        best_wr = max(v for v in [wr_h, wr_l] if not np.isnan(v)) if not (np.isnan(wr_h) and np.isnan(wr_l)) else np.nan
        if not np.isnan(best_wr) and best_wr > baseline_wr + 3:
            direction = f">= {p90:.1f}" if (not np.isnan(wr_h) and wr_h == best_wr) else f"<= {p10:.1f}"
            top_decile_filters[col] = {"threshold": direction, "win_rate": best_wr}
            print(f"  {label:<22}  {direction:>12}  win_rate={best_wr:.1f}%  (baseline {baseline_wr:.1f}%  +{best_wr-baseline_wr:.1f}pp)")

    # ── Recommended scanner config changes ────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  RECOMMENDED CONFIG CHANGES")
    print(f"{'='*65}")
    for col, label, _ in INDICATORS:
        if col in top_decile_filters:
            f = top_decile_filters[col]
            print(f"  {label:<22}  filter {f['threshold']:>14}  → win_rate {f['win_rate']:.1f}%")
    if not top_decile_filters:
        print("  No single indicator top-decile improves win rate by > 3pp.")
        print("  Suggests composite scoring or combination filters needed.")

    # ── Save results ──────────────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        signals_path = os.path.join(save_dir, "backtest_signals.csv")
        corr_path    = os.path.join(save_dir, "backtest_correlations.csv")
        df.to_csv(signals_path, index=False)
        corr_df.to_csv(corr_path, index=False)
        print(f"\n  Saved: {signals_path}")
        print(f"  Saved: {corr_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Breakout indicator backtest")
    parser.add_argument("--years",    type=float, default=2.5,       help="Years of history (default 2.5)")
    parser.add_argument("--tickers",  nargs="+",  default=None,      help="Subset of tickers (default: all)")
    parser.add_argument("--no-save",  action="store_true",           help="Skip saving CSV output")
    parser.add_argument("--out-dir",  default="backtest_results",    help="Output directory for CSVs")
    args = parser.parse_args()

    tickers = args.tickers if args.tickers else ALL_TICKERS
    tickers = [t.upper() for t in tickers]

    df_signals = run_backtest(tickers, years=args.years)

    if len(df_signals) == 0:
        print("No signals generated — check data download or thresholds.")
        sys.exit(1)

    print(f"\n{'─'*65}")
    print(f"  RESULTS OVERVIEW")
    print(f"{'-'*65}")
    run_analysis(df_signals, save_dir=None if args.no_save else args.out_dir)

    print(f"\n{'='*65}")
    print("  Backtest complete.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
