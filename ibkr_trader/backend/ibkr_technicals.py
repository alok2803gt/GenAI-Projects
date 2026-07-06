"""
ibkr_technicals.py — GET /technicals/{ticker}
Mounted automatically by main.py if import succeeds.

Computes: RSI-14, MACD(12,26,9), Bollinger Bands(20,2), SMA-20/50/200,
          volume breakout, swing S/R levels, pivot points, Fibonacci retracement.
Uses the shared ib_insync instance from main.py via lazy import.
"""

import asyncio
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter()


# ── Shared IB instance (lazy import to avoid circular) ────────────────────────

def _get_ib():
    from main import state, _run_in_streaming_loop  # noqa: PLC0415
    ib = state.get("ib")
    if not ib or not ib.isConnected():
        raise HTTPException(503, "IBKR not connected")
    return ib, _run_in_streaming_loop


# ── Pure numpy indicator math ──────────────────────────────────────────────────

def _sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(n - 1, len(arr)):
        out[i] = arr[i - n + 1: i + 1].mean()
    return out


def _ema(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    k = 2 / (n + 1)
    start = next((i for i, v in enumerate(arr) if not np.isnan(v)), None)
    if start is None or start + n > len(arr):
        return out
    out[start + n - 1] = arr[start: start + n].mean()
    for i in range(start + n, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def _rsi(closes: np.ndarray, n: int = 14) -> float:
    delta  = np.diff(closes[-n * 2:])
    gains  = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    ag = gains[-n:].mean()
    al = losses[-n:].mean()
    if al == 0:
        return 100.0
    return round(100 - 100 / (1 + ag / al), 2)


def _macd(closes: np.ndarray, fast=12, slow=26, signal=9):
    ef   = _ema(closes, fast)
    es   = _ema(closes, slow)
    ml   = ef - es
    sl   = _ema(ml, signal)
    hist = ml - sl
    return round(float(ml[-1]), 4), round(float(sl[-1]), 4), round(float(hist[-1]), 4)


def _bollinger(closes: np.ndarray, n: int = 20, k: float = 2.0):
    mid  = _sma(closes, n)
    stds = np.array([closes[i - n + 1: i + 1].std() for i in range(n - 1, len(closes))])
    upper   = mid[n - 1:] + k * stds
    lower   = mid[n - 1:] - k * stds
    b_upper = round(float(upper[-1]), 2)
    b_mid   = round(float(mid[-1]),   2)
    b_lower = round(float(lower[-1]), 2)
    price   = closes[-1]
    pct_b   = round((price - b_lower) / (b_upper - b_lower) * 100, 1) if b_upper != b_lower else 50.0
    return b_upper, b_mid, b_lower, pct_b


def _volume_breakout(volumes: np.ndarray, closes: np.ndarray, lookback: int = 20) -> dict:
    avg_vol        = volumes[-lookback - 1: -1].mean()
    vol_ratio      = round(float(volumes[-1] / avg_vol), 2) if avg_vol > 0 else 1.0
    highest_close  = closes[-lookback - 1: -1].max()
    price_breakout = bool(closes[-1] > highest_close)
    confirmed      = price_breakout and vol_ratio >= 1.5
    signal         = "BREAKOUT" if confirmed else ("WATCH" if price_breakout else "NORMAL")
    return {
        "volume_ratio":   vol_ratio,
        "avg_volume_20d": int(avg_vol),
        "today_volume":   int(volumes[-1]),
        "price_breakout": price_breakout,
        "confirmed":      confirmed,
        "signal":         signal,
    }


# ── Support & Resistance ───────────────────────────────────────────────────────

def _swing_levels(highs: np.ndarray, lows: np.ndarray, price: float,
                  n_confirm: int = 5, lookback: int = 60,
                  tolerance: float = 0.003) -> dict:
    """
    Swing highs = local maxima confirmed by n_confirm bars on each side.
    Swing lows  = local minima confirmed the same way.
    Nearby levels within tolerance (0.3%) are clustered into one.
    Returns up to 5 nearest resistance levels above price and 5 support below.
    """
    h = highs[-lookback:]
    l = lows[-lookback:]
    n = len(h)

    raw_highs, raw_lows = [], []
    for i in range(n_confirm, n - n_confirm):
        if all(h[i] >= h[i - j] for j in range(1, n_confirm + 1)) and \
           all(h[i] >= h[i + j] for j in range(1, n_confirm + 1)):
            raw_highs.append(float(h[i]))
        if all(l[i] <= l[i - j] for j in range(1, n_confirm + 1)) and \
           all(l[i] <= l[i + j] for j in range(1, n_confirm + 1)):
            raw_lows.append(float(l[i]))

    def cluster(levels: list) -> list:
        if not levels:
            return []
        out, group = [], [sorted(levels)[0]]
        for v in sorted(levels)[1:]:
            if (v - group[0]) / group[0] <= tolerance:
                group.append(v)
            else:
                out.append(round(sum(group) / len(group), 2))
                group = [v]
        out.append(round(sum(group) / len(group), 2))
        return out

    resistance = sorted(v for v in cluster(raw_highs) if v > price)[:5]
    support    = sorted((v for v in cluster(raw_lows) if v < price), reverse=True)[:5]
    return {"resistance": resistance, "support": support}


def _pivot_points(prev_high: float, prev_low: float, prev_close: float) -> dict:
    """Classic floor-trader pivot points derived from prior session H/L/C."""
    pp = (prev_high + prev_low + prev_close) / 3
    r  = prev_high - prev_low
    return {
        "pp": round(pp,                              2),
        "r1": round(2 * pp - prev_low,              2),
        "r2": round(pp + r,                          2),
        "r3": round(prev_high + 2 * (pp - prev_low), 2),
        "s1": round(2 * pp - prev_high,             2),
        "s2": round(pp - r,                          2),
        "s3": round(prev_low - 2 * (prev_high - pp), 2),
    }


def _fibonacci(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
               lookback: int = 60) -> dict:
    """
    Fibonacci retracement from the most recent significant swing (last lookback bars).
    Uptrend  (price >= midpoint): shows pullback support levels from swing low to high.
    Downtrend (price <  midpoint): shows bounce resistance levels from swing high to low.
    """
    h          = highs[-lookback:]
    l          = lows[-lookback:]
    swing_high = float(h.max())
    swing_low  = float(l.min())
    price      = float(closes[-1])
    diff       = swing_high - swing_low
    midpoint   = (swing_high + swing_low) / 2
    ratios     = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    if price >= midpoint:
        direction = "uptrend"
        levels = {f"{r:.3f}": round(swing_high - diff * r, 2) for r in ratios}
    else:
        direction = "downtrend"
        levels = {f"{r:.3f}": round(swing_low + diff * r, 2) for r in ratios}

    return {
        "swing_high": round(swing_high, 2),
        "swing_low":  round(swing_low,  2),
        "direction":  direction,
        "levels":     levels,
    }


# ── Overall signal score ───────────────────────────────────────────────────────

def _overall(trend, rsi_zone, macd_bias, breakout_signal) -> str:
    score = 0
    if trend == "bullish":            score += 2
    elif trend == "bearish":          score -= 2
    if macd_bias == "bullish":        score += 1
    else:                             score -= 1
    if rsi_zone == "oversold":        score += 1
    elif rsi_zone == "overbought":    score -= 1
    if breakout_signal == "BREAKOUT": score += 2
    if score >= 3:  return "strong_buy"
    if score >= 1:  return "buy"
    if score <= -3: return "strong_sell"
    if score <= -1: return "sell"
    return "neutral"


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("/technicals/{ticker}")
async def get_technicals(ticker: str):
    ticker = ticker.upper().strip()
    ib, run_in_loop = _get_ib()

    from ib_insync import Stock, util  # noqa: PLC0415

    async def _fetch(ib_):
        contract = Stock(ticker, "SMART", "USD")
        await ib_.qualifyContractsAsync(contract)
        if not contract.conId:
            raise ValueError(f"Cannot qualify {ticker}")
        return await ib_.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )

    try:
        loop = asyncio.get_event_loop()
        bars = await loop.run_in_executor(
            None,
            lambda: run_in_loop(_fetch(ib), timeout=30),
        )
    except Exception as exc:
        raise HTTPException(500, f"IBKR data fetch failed: {exc}")

    if not bars or len(bars) < 60:
        raise HTTPException(404, f"Insufficient data for {ticker} ({len(bars) if bars else 0} bars)")

    df      = util.df(bars)
    closes  = df["close"].values.astype(float)
    highs   = df["high"].values.astype(float)
    lows    = df["low"].values.astype(float)
    volumes = df["volume"].values.astype(float)

    price   = round(float(closes[-1]), 2)
    prev_px = round(float(closes[-2]), 2)
    chg_pct = round((price - prev_px) / prev_px * 100, 2)

    sma20  = round(float(_sma(closes, 20)[-1]),  2)
    sma50  = round(float(_sma(closes, 50)[-1]),  2)
    sma200 = round(float(_sma(closes, 200)[-1]), 2)
    rsi14  = _rsi(closes)
    macd_v, macd_sig, macd_hist = _macd(closes)
    bb_upper, bb_mid, bb_lower, pct_b = _bollinger(closes)
    vb     = _volume_breakout(volumes, closes)

    trend      = ("bullish" if price > sma20 > sma50 > sma200
                  else "bearish" if price < sma20 < sma50 < sma200
                  else "mixed")
    rsi_zone   = "overbought" if rsi14 > 70 else "oversold" if rsi14 < 30 else "neutral"
    macd_bias  = "bullish" if macd_hist > 0 else "bearish"

    swing  = _swing_levels(highs, lows, price)
    pivots = _pivot_points(float(highs[-2]), float(lows[-2]), float(closes[-2]))
    fib    = _fibonacci(highs, lows, closes)

    return JSONResponse({
        "ticker":     ticker,
        "price":      price,
        "change_pct": chg_pct,
        "bars_used":  len(bars),
        "trend":      trend,
        "indicators": {
            "rsi14":        rsi14,
            "rsi_zone":     rsi_zone,
            "macd":         macd_v,
            "macd_signal":  macd_sig,
            "macd_hist":    macd_hist,
            "macd_bias":    macd_bias,
            "sma20":        sma20,
            "sma50":        sma50,
            "sma200":       sma200,
            "above_sma20":  bool(price > sma20),
            "above_sma50":  bool(price > sma50),
            "above_sma200": bool(price > sma200),
            "bb_upper":     bb_upper,
            "bb_mid":       bb_mid,
            "bb_lower":     bb_lower,
            "pct_b":        pct_b,
        },
        "breakout": vb,
        "support_resistance": {
            "swing":     swing,
            "pivots":    pivots,
            "fibonacci": fib,
        },
        "summary": {
            "trend":           trend,
            "rsi_zone":        rsi_zone,
            "macd_bias":       macd_bias,
            "breakout_signal": vb["signal"],
            "overall":         _overall(trend, rsi_zone, macd_bias, vb["signal"]),
        },
    })
