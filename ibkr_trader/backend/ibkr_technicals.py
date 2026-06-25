"""
ibkr_technicals.py  —  add to your ibkr_trader FastAPI app
------------------------------------------------------------
Exposes GET /technicals/{ticker} which:
  1. Pulls 200 bars of daily OHLCV from IBKR via ib_insync
  2. Computes RSI-14, MACD(12,26,9), Bollinger Bands(20,2),
     SMA-20/50/200, volume ratio, and breakout signal
  3. Returns JSON consumed by the scanner widget

Mount in main.py:
    from ibkr_technicals import router as technicals_router
    app.include_router(technicals_router)

CORS is already handled by your existing CORSMiddleware.
"""

import asyncio
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ib_insync import IB, Stock, util

router = APIRouter()

# ── reuse the shared IB instance from your app if available ──
# from app import ib   ← uncomment if you expose a global `ib`
# For standalone use this creates its own connection:
_ib: IB | None = None

async def get_ib() -> IB:
    global _ib
    if _ib is None or not _ib.isConnected():
        _ib = IB()
        await _ib.connectAsync("127.0.0.1", 7497, clientId=12)
    return _ib


# ── indicator math (pure numpy, no pandas required) ──────────

def sma(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(n - 1, len(arr)):
        out[i] = arr[i - n + 1 : i + 1].mean()
    return out


def ema(arr: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    k = 2 / (n + 1)
    start = next((i for i, v in enumerate(arr) if not np.isnan(v)), None)
    if start is None:
        return out
    out[start + n - 1] = arr[start : start + n].mean()
    for i in range(start + n, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out


def rsi(closes: np.ndarray, n: int = 14) -> float:
    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = gains[-n:].mean()
    avg_loss = losses[-n:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


def macd(closes: np.ndarray, fast=12, slow=26, signal=9):
    e_fast = ema(closes, fast)
    e_slow = ema(closes, slow)
    macd_line = e_fast - e_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return (
        round(float(macd_line[-1]), 4),
        round(float(signal_line[-1]), 4),
        round(float(histogram[-1]), 4),
    )


def bollinger(closes: np.ndarray, n: int = 20, k: float = 2.0):
    mid = sma(closes, n)
    std = np.array([closes[i - n + 1 : i + 1].std() for i in range(n - 1, len(closes))])
    upper = mid[n - 1 :] + k * std
    lower = mid[n - 1 :] - k * std
    price = closes[-1]
    b_upper = round(float(upper[-1]), 2)
    b_mid   = round(float(mid[-1]), 2)
    b_lower = round(float(lower[-1]), 2)
    pct_b   = round((price - b_lower) / (b_upper - b_lower) * 100, 1) if b_upper != b_lower else 50.0
    return b_upper, b_mid, b_lower, pct_b


def volume_breakout(volumes: np.ndarray, closes: np.ndarray, lookback: int = 20) -> dict:
    """
    Volume-confirmed breakout logic:
      - volume_ratio  : today's vol vs 20-day avg
      - price_breakout: close > 20-day highest close (excluding today)
      - confirmed     : breakout AND volume_ratio >= 1.5
      - signal        : BREAKOUT | WATCH | NORMAL
    """
    avg_vol = volumes[-lookback - 1 : -1].mean()
    vol_ratio = round(float(volumes[-1] / avg_vol), 2) if avg_vol > 0 else 1.0
    highest_close = closes[-lookback - 1 : -1].max()
    price_breakout = bool(closes[-1] > highest_close)
    confirmed = price_breakout and vol_ratio >= 1.5
    signal = "BREAKOUT" if confirmed else ("WATCH" if price_breakout else "NORMAL")
    return {
        "volume_ratio": vol_ratio,
        "avg_volume_20d": int(avg_vol),
        "today_volume": int(volumes[-1]),
        "price_breakout": price_breakout,
        "confirmed": confirmed,
        "signal": signal,
    }


# ── route ─────────────────────────────────────────────────────

@router.get("/technicals/{ticker}")
async def get_technicals(ticker: str):
    ticker = ticker.upper().strip()
    try:
        ib = await get_ib()
        contract = Stock(ticker, "SMART", "USD")
        await ib.qualifyContractsAsync(contract)

        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="1 Y",
            barSizeSetting="1 day",
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if len(bars) < 60:
            raise HTTPException(status_code=404, detail=f"Not enough data for {ticker}")

        df = util.df(bars)
        closes  = df["close"].values.astype(float)
        volumes = df["volume"].values.astype(float)

        price   = round(float(closes[-1]), 2)
        prev    = round(float(closes[-2]), 2)
        chg_pct = round((price - prev) / prev * 100, 2)

        sma20  = round(float(sma(closes, 20)[-1]), 2)
        sma50  = round(float(sma(closes, 50)[-1]), 2)
        sma200 = round(float(sma(closes, 200)[-1]), 2)

        rsi14 = rsi(closes)
        macd_val, macd_sig, macd_hist = macd(closes)
        bb_upper, bb_mid, bb_lower, pct_b = bollinger(closes)
        vb = volume_breakout(volumes, closes)

        # trend bias from MA stack
        if price > sma20 > sma50 > sma200:
            trend = "bullish"
        elif price < sma20 < sma50 < sma200:
            trend = "bearish"
        else:
            trend = "mixed"

        # RSI zone
        rsi_zone = "overbought" if rsi14 > 70 else "oversold" if rsi14 < 30 else "neutral"

        # MACD signal
        macd_signal = "bullish" if macd_hist > 0 else "bearish"

        return JSONResponse({
            "ticker": ticker,
            "price": price,
            "change_pct": chg_pct,
            "trend": trend,
            "indicators": {
                "rsi14": rsi14,
                "rsi_zone": rsi_zone,
                "macd": macd_val,
                "macd_signal_line": macd_sig,
                "macd_histogram": macd_hist,
                "macd_bias": macd_signal,
                "sma20": sma20,
                "sma50": sma50,
                "sma200": sma200,
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "bb_lower": bb_lower,
                "pct_b": pct_b,
            },
            "breakout": vb,
            "summary": {
                "trend": trend,
                "rsi_zone": rsi_zone,
                "macd_bias": macd_signal,
                "breakout_signal": vb["signal"],
                "overall": _overall(trend, rsi_zone, macd_signal, vb["signal"]),
            },
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _overall(trend, rsi_zone, macd_bias, breakout) -> str:
    score = 0
    if trend == "bullish": score += 2
    elif trend == "bearish": score -= 2
    if macd_bias == "bullish": score += 1
    else: score -= 1
    if rsi_zone == "oversold": score += 1
    elif rsi_zone == "overbought": score -= 1
    if breakout == "BREAKOUT": score += 2
    if score >= 3: return "strong_buy"
    if score >= 1: return "buy"
    if score <= -3: return "strong_sell"
    if score <= -1: return "sell"
    return "neutral"
