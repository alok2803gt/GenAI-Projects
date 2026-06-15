"""
XGBoost IBKR Trader — Backend  (v3 — Live Streaming + CSP/LEAP Scanner)

New in v3:
  • GET /csp/scan       — ranked Cash-Secured Put candidates (4 %/wk target)
  • GET /leaps/scan     — ranked LEAP call candidates (strong directional plays)
  • GET /options/quote  — real-time single-contract quote with Greeks
  • GET /csp/universe   — view / manage scan universe
  • POST /csp/universe/add
"""

import asyncio
import concurrent.futures
import logging
import math
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from ib_insync import IB, Option, Stock, util
from pydantic import BaseModel
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
import os

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ibkr_trader")

# ── Config — bar streaming ──────────────────────────────────────────────────
TWS_HOST = "127.0.0.1"
TWS_PORT = 7496          # 7497=TWS paper | 7496=TWS live | 4002=IB Gateway paper
TWS_CLIENT_ID = 10
MODEL_PATH = "model.joblib"
BAR_SIZE = "5 mins"
HISTORY_DURATION = "5 D"
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45

FEATURE_COLS = [
    "rsi", "sma5", "sma14", "momentum",
    "vol_ratio", "body_pct", "upper_wick", "lower_wick", "volatility"
]

# ── Config — CSP scanner ───────────────────────────────────────────────────
CSP_MIN_RETURN_PCT  = 4.0    # Weekly premium / strike  ≥ 4 %
CSP_MAX_DELTA       = 0.20   # Absolute delta (far-OTM safety)
CSP_MIN_OI          = 50     # Open interest floor
CSP_MAX_SPREAD_PCT  = 0.15   # Bid-ask spread as fraction of mid
SCAN_CACHE_TTL      = 300    # Seconds before scan cache expires

# ── Config — external validation (yfinance) ────────────────────────────────
EARNINGS_BLOCK_DAYS = 7      # Skip CSP/LEAP entirely if earnings this close
EARNINGS_WARN_DAYS  = 14     # Add warning flag if earnings within this window
IV_RANK_MIN_CSP     = 25     # Flag CSP rows below this IV rank (thin premium env)
EARNINGS_CACHE_TTL  = 21600  # 6 h — earnings dates don't change intraday
IV_RANK_CACHE_TTL   = 3600   # 1 h
REGIME_CACHE_TTL    = 300    # 5 min

# ── Config — LEAP scanner ──────────────────────────────────────────────────
LEAP_MIN_DTE   = 180   # ≥ 6 months
LEAP_MAX_DTE   = 540   # ≤ 18 months
LEAP_MIN_DELTA = 0.45  # Near-ATM; enough directional exposure
LEAP_MAX_DELTA = 0.75  # Not deep ITM (expensive)
LEAP_MAX_IV    = 0.70  # Don't overpay for implied vol

# ── Scan universe (user can extend via API) ────────────────────────────────
CSP_UNIVERSE: List[str] = [
    # Mega-cap tech — liquid options, strong balance sheets
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    # Financials
    "JPM", "GS", "V", "MA",
    # ETFs — wide market, very liquid
    "SPY", "QQQ",
    # Semis
    "AMD", "MU", "AVGO",
    # Consumer / Healthcare
    "COST", "HD", "LLY", "UNH",
]

# ── Global state ───────────────────────────────────────────────────────────
state: Dict = {
    # Bar streaming
    "ib": None,
    "connected": False,
    "model": None,
    "model_accuracy": None,
    "bars": {},
    "signals": {},
    "last_update": {},
    "subscriptions": {},
    "error": None,
    # Scanner
    "streaming_loop": None,   # event loop of the streaming thread
    "scan_cache": {
        "csp": None, "leaps": None, "ts": None,
    },
    # External validation cache (yfinance)
    "ext_cache": {
        "earnings": {},   # ticker → {"days": int|None, "ts": datetime}
        "iv_rank":  {},   # ticker → {"iv": float|None, "rank": float, "rv_lo": float|None, "rv_hi": float|None, "ts": datetime}
        "regime":   None, # full market regime dict
    },
    # OPRA subscription status — set once at startup, re-checked on reconnect
    "opra_active": None,   # None = not yet checked, True/False thereafter
}

TICKERS: List[str] = ["AAPL", "MSFT", "NVDA", "SPY"]
_tickers_lock = threading.Lock()

# ── Feature engineering ────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["close", "open", "high", "low", "volume"]:
        df[col] = df[col].astype(float)

    df["sma5"]      = df["close"].rolling(5).mean()
    df["sma14"]     = df["close"].rolling(14).mean()
    df["momentum"]  = (df["close"] - df["sma14"]) / (df["sma14"] + 1e-9)
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(14).mean() + 1e-9)

    rng = (df["high"] - df["low"]).replace(0, 1e-6)
    df["body_pct"]   = (df["close"] - df["open"]).abs() / rng
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]        = 100 - 100 / (1 + gain / (loss + 1e-9))
    df["volatility"] = df["close"].pct_change().rolling(14).std()
    df["target"]     = (df["close"].shift(-1) > df["close"]).astype(int)

    return df.dropna()


# ── XGBoost model ──────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    df = build_features(df)
    if len(df) < 60:
        raise ValueError(f"Need ≥60 rows, got {len(df)}")

    X, y = df[FEATURE_COLS].values, df["target"].values
    tscv = TimeSeriesSplit(n_splits=3)
    accs, model = [], None
    for train_idx, val_idx in tscv.split(X):
        m = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0, random_state=42,
        )
        m.fit(X[train_idx], y[train_idx])
        accs.append(accuracy_score(y[val_idx], m.predict(X[val_idx])))
        model = m

    acc = float(np.mean(accs))
    log.info(f"Model trained  accuracy={acc:.3f}  rows={len(df)}")
    return model, acc


def predict(model, df: pd.DataFrame) -> dict:
    df = build_features(df)
    if df.empty:
        return {"label": "HOLD", "prob": 0.5, "confidence": 0.0}

    last  = df[FEATURE_COLS].iloc[[-1]].values
    prob  = float(model.predict_proba(last)[0][1])
    conf  = abs(prob - 0.5) * 2
    label = "BUY" if prob > BUY_THRESHOLD else "SELL" if prob < SELL_THRESHOLD else "HOLD"
    feat  = df[FEATURE_COLS].iloc[-1].to_dict()

    return {
        "label": label,
        "prob": round(prob, 4),
        "confidence": round(conf, 4),
        "features": {k: round(float(v), 4) for k, v in feat.items()},
        "close": round(float(df["close"].iloc[-1]), 4),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ── Live bar callback ──────────────────────────────────────────────────────
def _bars_to_df(bars) -> pd.DataFrame:
    df = util.df(bars)[["date", "open", "high", "low", "close", "volume"]]
    df.rename(columns={"date": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    return df


def _session_label(ts: pd.Timestamp) -> str:
    try:
        et = ts.tz_convert("America/New_York") if ts.tzinfo else ts
        h = et.hour + et.minute / 60
        if 4.0 <= h < 9.5:   return "PRE"
        if 9.5 <= h < 16.0:  return "RTH"
        if 16.0 <= h < 20.0: return "POST"
        return "CLOSED"
    except Exception:
        return "UNKNOWN"


def on_bar_update(ticker: str, bars, has_new_bar: bool) -> None:
    try:
        df = _bars_to_df(bars)
        if df.empty:
            return

        bars_list = df.tail(80).to_dict(orient="records")
        for b in bars_list:
            t = b.get("time")
            if hasattr(t, "isoformat"):
                b["time"] = t.isoformat()
        state["bars"][ticker] = bars_list

        if state["model"] is None and len(df) >= 60:
            try:
                model, acc = train_model(df)
                state["model"] = model
                state["model_accuracy"] = acc
                joblib.dump(model, MODEL_PATH)
            except Exception as e:
                log.warning(f"Training failed for {ticker}: {e}")

        if state["model"] is not None:
            sig = predict(state["model"], df)
            sig["ticker"] = ticker
            sig["session"] = _session_label(df["time"].iloc[-1])
            state["signals"][ticker] = sig
            state["last_update"][ticker] = datetime.utcnow().isoformat()
            if has_new_bar:
                log.info(
                    f"NEW BAR  {ticker}  {sig['label']}  "
                    f"prob={sig['prob']}  close={sig['close']}  "
                    f"session={sig['session']}"
                )

    except Exception as e:
        log.warning(f"on_bar_update error [{ticker}]: {e}", exc_info=True)


# ── Streaming subscription ─────────────────────────────────────────────────
async def subscribe_ticker(ib: IB, ticker: str) -> None:
    contract = Stock(ticker, "SMART", "USD")
    await ib.qualifyContractsAsync(contract)
    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime="",
        durationStr=HISTORY_DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow="TRADES",
        useRTH=False,
        formatDate=1,
        keepUpToDate=True,
    )
    state["subscriptions"][ticker] = bars
    bars.updateEvent += lambda b, h: on_bar_update(ticker, b, h)
    on_bar_update(ticker, bars, False)
    log.info(f"Streaming  {ticker}  ({len(bars)} bars seeded)")


async def _subscribe_pending(ib: IB, known: set) -> set:
    with _tickers_lock:
        current = set(TICKERS)
    for ticker in current - known:
        try:
            await subscribe_ticker(ib, ticker)
        except Exception as e:
            log.warning(f"Subscribe failed [{ticker}]: {e}")
    return current


# ── Background streaming loop ──────────────────────────────────────────────
async def streaming_loop_async() -> None:
    while True:
        try:
            ib = IB()
            ctx = {"known": set()}

            def _on_ib_error(reqId, errorCode, errorString, contract):
                if errorCode == 1102:
                    ctx["known"] = set()
                    state["subscriptions"].clear()
                    log.info("IBKR reconnected (1102) — re-subscribing all tickers within 10 s")

            ib.errorEvent += _on_ib_error

            await ib.connectAsync(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID, timeout=15)
            log.info(f"Connected to IBKR  {TWS_HOST}:{TWS_PORT}")
            state["ib"] = ib
            state["connected"] = True
            state["error"] = None

            ctx["known"] = await _subscribe_pending(ib, ctx["known"])

            # Check OPRA data subscription once per connection
            state["opra_active"] = await _check_opra_subscription(ib)

            while ib.isConnected():
                ctx["known"] = await _subscribe_pending(ib, ctx["known"])
                await asyncio.sleep(10)

            state["connected"] = False
            log.warning("IBKR disconnected — retrying in 15 s")

        except Exception as e:
            state["connected"] = False
            state["error"] = str(e)
            log.error(f"IBKR error: {e}  — retrying in 15 s")
            await asyncio.sleep(15)


def streaming_loop() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    state["streaming_loop"] = loop
    try:
        loop.run_until_complete(streaming_loop_async())
    finally:
        loop.close()
        state["streaming_loop"] = None


# ── CSP / LEAP scanner helpers ─────────────────────────────────────────────

def _next_expiry(weeks_out: int = 0) -> str:
    """Next Friday (or n Fridays out) in YYYYMMDD format."""
    today = date.today()
    days_to_friday = (4 - today.weekday()) % 7 or 7
    friday = today + timedelta(days=days_to_friday + weeks_out * 7)
    return friday.strftime("%Y%m%d")


def _stock_quality_score(ticker: str) -> float:
    """
    0–1 composite from live signal features.
    High score = strong uptrend, not overbought, positive momentum.
    """
    sig = state["signals"].get(ticker)
    if not sig or "features" not in sig:
        return 0.5
    f = sig["features"]
    score = 0.0
    rsi = f.get("rsi", 50)
    # RSI 45–65: ideal for selling puts (trending up, not extended)
    if 45 <= rsi <= 65:
        score += 0.40
    elif 35 <= rsi < 45 or 65 < rsi <= 72:
        score += 0.20
    # Positive momentum: price above SMA14
    mom = f.get("momentum", 0)
    if mom > 0.002:
        score += 0.35
    elif mom > 0:
        score += 0.20
    # Low realized volatility: stable stock less likely to gap through strike
    vol = f.get("volatility", 0.01)
    if vol < 0.005:
        score += 0.25
    elif vol < 0.010:
        score += 0.15
    return round(min(score, 1.0), 3)


def _expected_weekly_move(stock_price: float, iv: float) -> float:
    """1-sigma expected weekly move based on IV (annualized)."""
    return stock_price * iv * math.sqrt(7 / 365)


def _assignment_risk(delta: float, otm_pct: float) -> str:
    abs_d = abs(delta)
    if abs_d < 0.08 and otm_pct > 12:
        return "VERY LOW"
    if abs_d < 0.12 and otm_pct > 8:
        return "LOW"
    if abs_d < 0.17:
        return "LOW-MEDIUM"
    return "MEDIUM"


def _score_csp(row: dict) -> float:
    """
    Composite rank score — higher is better.
    Uses realistic fill return (bid + 40% spread) as the primary return metric.
    """
    s = 0.0
    # 1) Realistic weekly return above 4% target (capped at 3×)
    s += min(row["weekly_return_pct"] / CSP_MIN_RETURN_PCT, 3.0) * 25
    # 2) Distance from assignment
    s += (CSP_MAX_DELTA - abs(row["delta"])) / CSP_MAX_DELTA * 20
    # 3) Liquidity (OI + volume + spread combined)
    s += (row.get("liquidity_score", 50) / 100) * 20
    # 4) Stock quality (technical trend strength)
    s += row.get("stock_quality", 0.5) * 20
    # 5) IV rank — elevated IV = better premium selling environment
    s += (row.get("iv_rank", 50) / 100) * 15
    return round(s, 2)


# ── External validation helpers (yfinance) ────────────────────────────────

async def _earnings_days_out(ticker: str) -> Optional[int]:
    """
    Days until next earnings announcement, or None if none found within 60 days.
    Cached 6 h per ticker.  Never blocks the scan on error — returns None.
    """
    cache = state["ext_cache"]["earnings"]
    now = datetime.utcnow()
    if ticker in cache and (now - cache[ticker]["ts"]).total_seconds() < EARNINGS_CACHE_TTL:
        return cache[ticker]["days"]

    loop = asyncio.get_event_loop()
    try:
        def _fetch() -> Optional[int]:
            t = yf.Ticker(ticker)
            try:
                cal = t.calendar
            except Exception:
                return None
            if cal is None:
                return None
            today = date.today()
            # yfinance >= 0.2 returns dict; older versions return DataFrame
            if isinstance(cal, dict):
                raw = cal.get("Earnings Date", [])
                if not raw:
                    return None
                d = pd.to_datetime(raw[0]).date()
            elif hasattr(cal, "loc"):
                try:
                    d = pd.to_datetime(cal.loc["Earnings Date"].iloc[0]).date()
                except Exception:
                    return None
            else:
                return None
            days = (d - today).days
            return days if 0 <= days <= 60 else None

        days = await loop.run_in_executor(None, _fetch)
    except Exception as e:
        log.debug(f"Earnings fetch [{ticker}]: {e}")
        days = None

    cache[ticker] = {"days": days, "ts": now}
    return days


async def _iv_rank_for_ticker(ticker: str) -> dict:
    """
    IV Rank proxy: where the nearest-expiry ATM put IV sits within the
    ticker's 52-week realized vol range (0 = historic low, 100 = historic high).

    Returns {"iv": float|None, "rank": float, "rv_lo": float|None, "rv_hi": float|None}
    Falls back to rank=50 (neutral) on any error.  Cached 1 h per ticker.
    """
    cache = state["ext_cache"]["iv_rank"]
    now = datetime.utcnow()
    if ticker in cache and (now - cache[ticker]["ts"]).total_seconds() < IV_RANK_CACHE_TTL:
        return {k: v for k, v in cache[ticker].items() if k != "ts"}

    loop = asyncio.get_event_loop()
    try:
        def _fetch():
            t = yf.Ticker(ticker)
            # Current stock price
            fi = t.fast_info
            price = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
            if not price:
                return None, None, None

            # ATM put IV from the nearest available expiry
            current_iv = None
            try:
                exps = t.options
                if exps:
                    chain = t.option_chain(exps[0])
                    puts = chain.puts.copy()
                    puts["dist"] = (puts["strike"] - price).abs()
                    atm = puts.nsmallest(1, "dist")
                    if not atm.empty:
                        current_iv = float(atm["impliedVolatility"].iloc[0])
            except Exception:
                pass

            # 52-week rolling-30d realized vol range
            rv_lo, rv_hi = None, None
            try:
                hist = t.history(period="1y")
                if len(hist) >= 30:
                    rv = hist["Close"].pct_change().rolling(30).std() * math.sqrt(252)
                    rv = rv.dropna()
                    if not rv.empty:
                        rv_lo = float(rv.min())
                        rv_hi = float(rv.max())
            except Exception:
                pass

            return current_iv, rv_lo, rv_hi

        current_iv, rv_lo, rv_hi = await loop.run_in_executor(None, _fetch)
    except Exception as e:
        log.debug(f"IV rank fetch [{ticker}]: {e}")
        current_iv, rv_lo, rv_hi = None, None, None

    if current_iv and rv_lo is not None and rv_hi is not None and rv_hi > rv_lo:
        rank = round(min(max((current_iv - rv_lo) / (rv_hi - rv_lo) * 100, 0), 100), 1)
    else:
        rank = 50.0

    result = {
        "iv":    round(current_iv * 100, 1) if current_iv else None,
        "rank":  rank,
        "rv_lo": round(rv_lo * 100, 1) if rv_lo else None,
        "rv_hi": round(rv_hi * 100, 1) if rv_hi else None,
        "ts":    now,
    }
    cache[ticker] = result
    return {k: v for k, v in result.items() if k != "ts"}


async def _market_regime() -> dict:
    """
    SPY vs its 50-day SMA + current VIX → BULL / NEUTRAL / BEAR.
    Cached 5 min.  Falls back to UNKNOWN on network error.
    """
    cached = state["ext_cache"]["regime"]
    now = datetime.utcnow()
    if cached and (now - cached["ts"]).total_seconds() < REGIME_CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "ts"}

    loop = asyncio.get_event_loop()
    try:
        def _fetch():
            spy_hist = yf.Ticker("SPY").history(period="3mo")["Close"]
            vix_hist = yf.Ticker("^VIX").history(period="5d")["Close"]
            spy_close = float(spy_hist.iloc[-1])
            spy_sma50 = float(spy_hist.rolling(50).mean().iloc[-1])
            vix       = float(vix_hist.iloc[-1])
            return spy_close, spy_sma50, vix

        spy_close, spy_sma50, vix = await loop.run_in_executor(None, _fetch)
        spy_vs_50d = round((spy_close - spy_sma50) / spy_sma50 * 100, 2)

        if spy_close >= spy_sma50 and vix < 20:
            regime = "BULL"
        elif spy_close >= spy_sma50 and vix < 25:
            regime = "NEUTRAL"
        else:
            regime = "BEAR"

        result = {
            "regime":         regime,
            "spy_close":      round(spy_close, 2),
            "spy_sma50":      round(spy_sma50, 2),
            "spy_vs_50d_pct": spy_vs_50d,
            "vix":            round(vix, 2),
            "ts":             now,
        }
    except Exception as e:
        log.warning(f"Market regime fetch failed: {e}")
        result = {
            "regime": "UNKNOWN",
            "spy_close": None, "spy_sma50": None,
            "spy_vs_50d_pct": None, "vix": None,
            "ts": now,
        }

    state["ext_cache"]["regime"] = result
    return {k: v for k, v in result.items() if k != "ts"}


def _build_warnings(earnings_days: Optional[int], iv_rank: float, mode: str = "csp") -> List[str]:
    """Human-readable warning tags attached to each candidate row."""
    warnings: List[str] = []
    if earnings_days is not None and earnings_days <= EARNINGS_WARN_DAYS:
        warnings.append(f"Earnings in {earnings_days}d")
    if mode == "csp" and iv_rank < IV_RANK_MIN_CSP:
        warnings.append(f"Low IV rank ({iv_rank:.0f})")
    if mode == "leap" and iv_rank > 75:
        warnings.append(f"High IV rank ({iv_rank:.0f}) — expensive premium")
    return warnings


async def _check_opra_subscription(ib: IB) -> bool:
    """
    Probe whether real-time OPRA options data is active on this account.
    Tests a SPY ATM put snapshot — NaN bid/ask means delayed or no subscription.
    """
    try:
        spy_price = await _get_stock_price(ib, "SPY")
        if spy_price <= 0:
            spy_price = 740.0
        strike = float(round(spy_price / 5) * 5)
        expiry = _next_expiry(0)
        contract = Option("SPY", expiry, strike, "P", "SMART")
        await ib.qualifyContractsAsync(contract)
        if not contract.conId:
            return False
        [td] = await ib.reqTickersAsync(contract)
        bid = td.bid if td.bid and not math.isnan(td.bid) else None
        ask = td.ask if td.ask and not math.isnan(td.ask) else None
        active = bid is not None and ask is not None and bid > 0 and ask > 0
        log.info(f"OPRA subscription: {'ACTIVE' if active else 'NOT SUBSCRIBED / DELAYED'}")
        return active
    except Exception as e:
        log.warning(f"OPRA check failed: {e}")
        return False


def _liquidity_score(oi: int, vol: int, spread_pct: float) -> float:
    """
    0-100 composite liquidity score.
    OI   (40 pts): depth — capped at 1 000 contracts
    Vol  (30 pts): today's activity — capped at 100 contracts
    Spread (30 pts): tighter = better execution
    """
    oi_pts  = min(oi  / 1_000, 1.0) * 40
    vol_pts = min(vol / 100,   1.0) * 30
    spr_pts = max(0.0, 1.0 - (spread_pct / 100) / CSP_MAX_SPREAD_PCT) * 30
    return round(oi_pts + vol_pts + spr_pts, 1)


async def _get_stock_price(ib: IB, ticker: str) -> float:
    """Live price: prefer streaming bar cache (always fresh), else IBKR snapshot."""
    # Streaming bars are the freshest source — use them if available
    bars = state["bars"].get(ticker)
    if bars:
        return float(bars[-1].close)

    # Fallback: cached signal close (may be up to one bar old)
    sig = state["signals"].get(ticker)
    if sig:
        return float(sig["close"])

    # Last resort: one-shot snapshot
    contract = Stock(ticker, "SMART", "USD")
    await ib.qualifyContractsAsync(contract)
    [t] = await ib.reqTickersAsync(contract)
    price = t.marketPrice()
    if not price or math.isnan(price):
        price = t.close or 0.0
    return float(price)


async def _option_quotes(ib: IB, contracts: list) -> list:
    """
    Snapshot market data with Greeks for a batch of option contracts.
    Qualifies, requests, and returns Ticker objects (bid/ask/modelGreeks).
    """
    await ib.qualifyContractsAsync(*contracts)
    valid = [c for c in contracts if c.conId]
    if not valid:
        return []
    return await ib.reqTickersAsync(*valid)


# ── CSP scan ───────────────────────────────────────────────────────────────
async def scan_csp(
    ib: IB,
    min_return: float = CSP_MIN_RETURN_PCT,
    max_delta: float = CSP_MAX_DELTA,
) -> dict:
    """
    For every ticker in CSP_UNIVERSE:
      1. Earnings gate — skip if earnings within EARNINGS_BLOCK_DAYS
      2. Fetch current price (streaming cache → IBKR snapshot)
      3. Get option chain + IV rank concurrently
      4. Select put strikes 5–28 % OTM on the nearest weekly expiry
      5. Snapshot quotes + Greeks
      6. Filter: return ≥ min_return, |delta| ≤ max_delta, OI ≥ 50
      7. Score (includes IV rank bonus) and return sorted list
    Returns {"candidates": [...], "regime": {...}}
    """
    expiry0 = _next_expiry(0)
    expiry1 = _next_expiry(1)
    sem = asyncio.Semaphore(5)

    # Market regime + ticker scans run concurrently
    async def _scan_ticker(ticker: str) -> List[dict]:
        async with sem:
            # ── Earnings gate (fastest disqualifier) ──────────────────────
            earnings_days = await _earnings_days_out(ticker)
            if earnings_days is not None and earnings_days <= EARNINGS_BLOCK_DAYS:
                log.info(f"CSP [{ticker}]: blocked — earnings in {earnings_days}d")
                return []

            try:
                stock_price = await _get_stock_price(ib, ticker)
                if stock_price <= 0:
                    log.debug(f"CSP [{ticker}]: no price — skipping")
                    return []

                stock = Stock(ticker, "SMART", "USD")
                await ib.qualifyContractsAsync(stock)

                # Fetch option chain and IV rank in parallel
                chains, iv_info = await asyncio.gather(
                    ib.reqSecDefOptParamsAsync(ticker, "", "STK", stock.conId),
                    _iv_rank_for_ticker(ticker),
                )
                if not chains:
                    log.debug(f"CSP [{ticker}]: no option chain — skipping")
                    return []

                chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
                expiry = expiry0 if expiry0 in chain.expirations else (
                    expiry1 if expiry1 in chain.expirations else None
                )
                if not expiry:
                    log.debug(f"CSP [{ticker}]: no weekly expiry available")
                    return []

                dte = (datetime.strptime(expiry, "%Y%m%d").date() - date.today()).days

                put_strikes = sorted([
                    s for s in chain.strikes
                    if 0.72 * stock_price <= s <= 0.95 * stock_price
                ], reverse=True)[:8]
                if not put_strikes:
                    return []

                contracts = [Option(ticker, expiry, s, "P", "SMART") for s in put_strikes]
                tickers_data = await _option_quotes(ib, contracts)

                rows: List[dict] = []
                for td in tickers_data:
                    bid = td.bid if (td.bid and td.bid > 0) else 0.0
                    ask = td.ask if (td.ask and td.ask > 0) else 0.0
                    if bid <= 0 or ask <= 0:
                        continue

                    mid = (bid + ask) / 2.0
                    greeks = td.modelGreeks
                    if not greeks or greeks.delta is None or greeks.impliedVol is None:
                        continue

                    delta = greeks.delta
                    iv    = greeks.impliedVol
                    theta = greeks.theta or 0.0

                    # ── Greek sanity check (bad OPRA data will fail this) ──
                    if delta >= 0:          continue   # put delta must be negative
                    if iv < 0.05 or iv > 3.0: continue  # IV out of sane range

                    strike = td.contract.strike
                    otm_pct = (stock_price - strike) / stock_price * 100
                    spread_pct = (ask - bid) / mid * 100
                    oi  = getattr(td, 'openInterest', None) or 0
                    vol = getattr(td, 'volume', None) or 0

                    # ── Fill estimates (selling a put: you receive near the BID) ──
                    # Conservative: bid (worst case — market order)
                    # Realistic:    bid + 40% of spread (limit order near mid)
                    fill_conservative = bid
                    fill_realistic    = bid + (mid - bid) * 0.40

                    weekly_return_bid      = fill_conservative / strike * 100
                    weekly_return_pct      = fill_realistic    / strike * 100   # primary metric
                    weekly_return_mid      = mid               / strike * 100   # optimistic
                    annualized_return      = weekly_return_pct * 52

                    # Max loss = assigned at strike, offset by premium received
                    max_loss_per_contract  = round((strike - fill_realistic) * 100, 2)
                    # Return on capital at risk (annualised)
                    return_on_risk = round(
                        fill_realistic / max(strike - fill_realistic, 0.01) * 52 * 100, 1
                    )

                    # ── Liquidity ─────────────────────────────────────────
                    liq = _liquidity_score(oi, vol, spread_pct)

                    # ── Filter gates ──────────────────────────────────────
                    if weekly_return_pct < min_return:        continue
                    if abs(delta) > max_delta:                continue
                    if oi < CSP_MIN_OI:                       continue
                    if spread_pct / 100 > CSP_MAX_SPREAD_PCT: continue

                    exp_move  = _expected_weekly_move(stock_price, iv)
                    sigma_otm = (stock_price - strike) / exp_move if exp_move > 0 else 0

                    # ── IV cross-validation (IBKR OPRA vs yfinance) ───────
                    warnings = _build_warnings(earnings_days, iv_info["rank"], "csp")
                    if iv_info["iv"] is not None and iv > 0:
                        divergence = abs(iv * 100 - iv_info["iv"]) / max(iv_info["iv"], 1)
                        if divergence > 0.35:
                            warnings.append(
                                f"IV mismatch: IBKR {iv*100:.0f}% vs YF {iv_info['iv']:.0f}%"
                            )
                    if not state["opra_active"]:
                        warnings.append("OPRA not subscribed — quotes may be delayed")

                    sq = _stock_quality_score(ticker)
                    row = {
                        "ticker":                  ticker,
                        "expiry":                  expiry,
                        "dte":                     dte,
                        "strike":                  strike,
                        "stock_price":             round(stock_price, 2),
                        "otm_pct":                 round(otm_pct, 2),
                        "sigma_otm":               round(sigma_otm, 2),
                        "bid":                     round(bid, 2),
                        "ask":                     round(ask, 2),
                        "mid":                     round(mid, 2),
                        # Return trio: conservative / realistic / optimistic
                        "weekly_return_bid":        round(weekly_return_bid, 2),
                        "weekly_return_pct":        round(weekly_return_pct, 2),
                        "weekly_return_mid":        round(weekly_return_mid, 2),
                        "annualized_return":        round(annualized_return, 1),
                        # Risk metrics
                        "max_loss_per_contract":   max_loss_per_contract,
                        "return_on_risk_ann":      return_on_risk,
                        "breakeven":               round(strike - fill_realistic, 2),
                        # Greeks
                        "delta":                   round(delta, 4),
                        "iv_pct":                  round(iv * 100, 1),
                        "theta_daily":             round(theta, 4),
                        # Liquidity
                        "open_interest":           oi,
                        "volume":                  vol,
                        "spread_pct":              round(spread_pct, 2),
                        "liquidity_score":         liq,
                        # Quality / classification
                        "stock_quality":           sq,
                        "assignment_risk":         _assignment_risk(delta, otm_pct),
                        "xgb_signal":              state["signals"].get(ticker, {}).get("label", "N/A"),
                        "cash_required":           round(strike * 100, 2),
                        "premium_collected":       round(fill_realistic * 100, 2),
                        # External validation
                        "earnings_days_out":       earnings_days,
                        "iv_rank":                 iv_info["rank"],
                        "iv_yf":                   iv_info["iv"],
                        "warnings":                warnings,
                    }
                    row["score"] = _score_csp(row)
                    rows.append(row)

                return rows

            except Exception as e:
                log.warning(f"CSP scan error [{ticker}]: {e}")
                return []

    regime, *ticker_results = await asyncio.gather(
        _market_regime(),
        *[_scan_ticker(t) for t in CSP_UNIVERSE],
    )
    candidates = [row for ticker_rows in ticker_results for row in ticker_rows]
    return {
        "candidates": sorted(candidates, key=lambda x: x["score"], reverse=True),
        "regime":     regime,
    }


# ── LEAP scan ──────────────────────────────────────────────────────────────
async def scan_leaps(ib: IB) -> dict:
    """
    For every ticker in CSP_UNIVERSE find LEAP calls:
      • Earnings gate — block if earnings within EARNINGS_BLOCK_DAYS (IV inflated pre-earnings)
      • Expiry 6–18 months out (mid-window ≈ 12 months)
      • Delta 0.45–0.75 (near-ATM to 10 % OTM)
      • IV ≤ 70 % (don't overpay)
      • Only on tickers with BUY/HOLD XGB signal and positive momentum
      • Low IV rank is favourable for LEAP buyers (cheaper premium)
    Returns {"candidates": [...], "regime": {...}}
    """
    today = date.today()
    sem = asyncio.Semaphore(5)

    async def _scan_ticker(ticker: str) -> List[dict]:
        async with sem:
            # ── Earnings gate — IV inflated before earnings, will crush after ──
            earnings_days = await _earnings_days_out(ticker)
            if earnings_days is not None and earnings_days <= EARNINGS_BLOCK_DAYS:
                log.info(f"LEAP [{ticker}]: blocked — earnings in {earnings_days}d (IV inflated)")
                return []

            try:
                sig = state["signals"].get(ticker, {})
                if sig.get("label") == "SELL":
                    return []
                sq = _stock_quality_score(ticker)
                if sq < 0.3:
                    return []

                stock_price = await _get_stock_price(ib, ticker)
                if stock_price <= 0:
                    return []

                stock = Stock(ticker, "SMART", "USD")
                await ib.qualifyContractsAsync(stock)

                # Fetch option chain and IV rank concurrently
                chains, iv_info = await asyncio.gather(
                    ib.reqSecDefOptParamsAsync(ticker, "", "STK", stock.conId),
                    _iv_rank_for_ticker(ticker),
                )
                if not chains:
                    return []

                chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

                leap_expiries = []
                for exp in chain.expirations:
                    try:
                        exp_date = datetime.strptime(exp, "%Y%m%d").date()
                        dte = (exp_date - today).days
                        if LEAP_MIN_DTE <= dte <= LEAP_MAX_DTE:
                            leap_expiries.append((dte, exp))
                    except ValueError:
                        continue

                if not leap_expiries:
                    return []

                leap_expiries.sort()
                mid_idx = len(leap_expiries) // 2
                dte, chosen_expiry = leap_expiries[mid_idx]

                call_strikes = sorted([
                    s for s in chain.strikes
                    if stock_price * 0.88 <= s <= stock_price * 1.12
                ])[:8]
                if not call_strikes:
                    return []

                contracts = [Option(ticker, chosen_expiry, s, "C", "SMART") for s in call_strikes]
                tickers_data = await _option_quotes(ib, contracts)

                rows: List[dict] = []
                for td in tickers_data:
                    bid = td.bid if (td.bid and td.bid > 0) else 0.0
                    ask = td.ask if (td.ask and td.ask > 0) else 0.0
                    if bid <= 0 or ask <= 0:
                        continue

                    mid = (bid + ask) / 2.0
                    greeks = td.modelGreeks
                    if not greeks or greeks.delta is None or greeks.impliedVol is None:
                        continue

                    delta = greeks.delta
                    iv    = greeks.impliedVol
                    theta = greeks.theta or 0.0
                    vega  = greeks.vega  or 0.0

                    # ── Greek sanity check ────────────────────────────────
                    if delta <= 0:             continue   # call delta must be positive
                    if iv < 0.05 or iv > 3.0:  continue

                    if not (LEAP_MIN_DELTA <= delta <= LEAP_MAX_DELTA):
                        continue
                    if iv > LEAP_MAX_IV:
                        continue

                    strike = td.contract.strike
                    itm_otm_pct = (stock_price - strike) / stock_price * 100
                    spread_pct  = (ask - bid) / mid * 100
                    oi  = getattr(td, 'openInterest', None) or 0
                    vol = getattr(td, 'volume', None) or 0

                    # ── Fill estimates (buying a call: you PAY near the ASK) ──
                    # Conservative: ask (worst case — market order)
                    # Realistic:    ask - 40% of spread (limit order near mid)
                    fill_conservative = ask
                    fill_realistic    = ask - (ask - mid) * 0.40

                    cost_conservative      = round(fill_conservative * 100, 2)
                    cost_per_contract      = round(fill_realistic    * 100, 2)  # primary
                    cost_mid               = round(mid               * 100, 2)  # optimistic

                    breakeven              = strike + fill_realistic
                    breakeven_move_pct     = (breakeven - stock_price) / stock_price * 100

                    # Max loss = total premium paid (realistic fill)
                    max_loss_per_contract  = cost_per_contract
                    # Weekly theta decay
                    theta_weekly           = round(theta * 7, 4)

                    liq = _liquidity_score(oi, vol, spread_pct)

                    # ── IV cross-validation ───────────────────────────────
                    warnings = _build_warnings(earnings_days, iv_info["rank"], "leap")
                    if iv_info["iv"] is not None and iv > 0:
                        divergence = abs(iv * 100 - iv_info["iv"]) / max(iv_info["iv"], 1)
                        if divergence > 0.35:
                            warnings.append(
                                f"IV mismatch: IBKR {iv*100:.0f}% vs YF {iv_info['iv']:.0f}%"
                            )
                    if not state["opra_active"]:
                        warnings.append("OPRA not subscribed — quotes may be delayed")

                    # Low iv_rank = cheaper for LEAP buyers
                    leap_iv_bonus = (1 - iv_info["rank"] / 100) * 15

                    row = {
                        "ticker":              ticker,
                        "expiry":              chosen_expiry,
                        "dte":                 dte,
                        "strike":              strike,
                        "stock_price":         round(stock_price, 2),
                        "itm_otm_pct":         round(itm_otm_pct, 2),
                        "breakeven":           round(breakeven, 2),
                        "breakeven_move_pct":  round(breakeven_move_pct, 2),
                        "bid":                 round(bid, 2),
                        "ask":                 round(ask, 2),
                        "mid":                 round(mid, 2),
                        # Cost trio: conservative / realistic / optimistic
                        "cost_conservative":   cost_conservative,
                        "cost_per_contract":   cost_per_contract,
                        "cost_mid":            cost_mid,
                        # Risk
                        "max_loss_per_contract": max_loss_per_contract,
                        # Greeks
                        "delta":               round(delta, 4),
                        "iv_pct":              round(iv * 100, 1),
                        "theta_daily":         round(theta, 4),
                        "theta_weekly":        theta_weekly,
                        "vega":                round(vega, 4),
                        # Liquidity
                        "spread_pct":          round(spread_pct, 2),
                        "open_interest":       oi,
                        "volume":              vol,
                        "liquidity_score":     liq,
                        # Quality
                        "stock_quality":       sq,
                        "xgb_signal":          sig.get("label", "N/A"),
                        "xgb_prob":            sig.get("prob", None),
                        # External validation
                        "earnings_days_out":   earnings_days,
                        "iv_rank":             iv_info["rank"],
                        "iv_yf":               iv_info["iv"],
                        "warnings":            warnings,
                    }
                    row["score"] = round(
                        sq * 55
                        + (1 - abs(delta - 0.60)) * 35
                        + leap_iv_bonus
                        + (liq / 100) * 10,
                        2
                    )
                    rows.append(row)

                return rows

            except Exception as e:
                log.warning(f"LEAP scan error [{ticker}]: {e}")
                return []

    regime, *ticker_results = await asyncio.gather(
        _market_regime(),
        *[_scan_ticker(t) for t in CSP_UNIVERSE],
    )
    candidates = [row for ticker_rows in ticker_results for row in ticker_rows]
    return {
        "candidates": sorted(candidates, key=lambda x: x["score"], reverse=True),
        "regime":     regime,
    }


# ── Real-time single option quote ──────────────────────────────────────────
async def _live_option_quote(
    ib: IB, ticker: str, expiry: str, strike: float, right: str
) -> dict:
    """Snapshot quote with full Greeks for one specific contract."""
    contract = Option(ticker, expiry, strike, right.upper(), "SMART")
    await ib.qualifyContractsAsync(contract)
    if not contract.conId:
        raise ValueError(f"Could not qualify {ticker} {expiry} {strike} {right}")

    [td] = await ib.reqTickersAsync(contract)
    greeks = td.modelGreeks

    # Validate we got usable data — IBKR error 354 leaves bid/ask as NaN
    bid = td.bid if td.bid and not math.isnan(td.bid) else None
    ask = td.ask if td.ask and not math.isnan(td.ask) else None
    if bid is None and ask is None and greeks is None:
        raise ValueError(
            f"No market data for {ticker} {expiry} {strike} {right} — "
            "option market data subscription may be required in TWS"
        )

    # Use streaming price cache — avoids 2 extra IBKR round-trips
    stock_price = await _get_stock_price(ib, ticker)

    return {
        "ticker":        ticker,
        "expiry":        expiry,
        "strike":        strike,
        "right":         right.upper(),
        "bid":           bid,
        "ask":           ask,
        "mid":           round((bid + ask) / 2, 4) if bid and ask else None,
        "last":          td.last if not math.isnan(td.last or float('nan')) else None,
        "volume":        getattr(td, 'volume', None),
        "open_interest": getattr(td, 'openInterest', None),
        "stock_price":   round(stock_price, 2),
        "delta":         greeks.delta       if greeks else None,
        "gamma":         greeks.gamma       if greeks else None,
        "theta":       greeks.theta       if greeks else None,
        "vega":        greeks.vega        if greeks else None,
        "iv_pct":      round(greeks.impliedVol * 100, 2) if greeks and greeks.impliedVol else None,
        "timestamp":   datetime.utcnow().isoformat() + "Z",
    }


# ── Bridge: run async scanner in streaming thread's event loop ─────────────
def _run_in_streaming_loop(coro, timeout: int = 180):
    """
    Submit a coroutine to the streaming thread's asyncio event loop and
    block until it completes.  Returns the result or raises on error/timeout.
    """
    loop: Optional[asyncio.AbstractEventLoop] = state.get("streaming_loop")
    if not loop or loop.is_closed():
        raise RuntimeError("Streaming event loop not available")

    future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise TimeoutError(f"Scanner timed out after {timeout}s")


# ── FastAPI app ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        state["model"] = joblib.load(MODEL_PATH)
        log.info("Loaded cached model from disk")

    t = threading.Thread(target=streaming_loop, daemon=True)
    t.start()
    log.info("Live streaming thread started")
    yield
    if state["ib"] and state["ib"].isConnected():
        state["ib"].disconnect()


app = FastAPI(title="XGBoost IBKR Trader — v3 CSP/LEAP", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────
class AddTickerRequest(BaseModel):
    ticker: str


# ── Bar streaming endpoints ────────────────────────────────────────────────
@app.get("/status")
def get_status():
    return {
        "connected":      state["connected"],
        "opra_active":    state["opra_active"],
        "error":          state["error"],
        "model_accuracy": state["model_accuracy"],
        "tickers":        list(state["signals"].keys()),
        "subscriptions":  list(state["subscriptions"].keys()),
        "last_updates":   state["last_update"],
    }


@app.get("/signal/{ticker}")
def get_signal(ticker: str):
    ticker = ticker.upper()
    if ticker not in state["signals"]:
        raise HTTPException(404, f"{ticker} not subscribed. POST /add_ticker first.")
    return state["signals"][ticker]


@app.get("/signals")
def get_all_signals():
    return state["signals"]


@app.get("/bars/{ticker}")
def get_bars(ticker: str, limit: int = 80):
    ticker = ticker.upper()
    if ticker not in state["bars"]:
        raise HTTPException(404, f"No bars for {ticker}")
    return state["bars"][ticker][-limit:]


@app.post("/add_ticker")
def add_ticker(req: AddTickerRequest):
    ticker = req.ticker.upper()
    with _tickers_lock:
        if ticker not in TICKERS:
            TICKERS.append(ticker)
            log.info(f"Queued {ticker} — streaming loop will subscribe within 10 s")
    return {"ok": True, "ticker": ticker}


@app.post("/retrain/{ticker}")
def retrain(ticker: str):
    ticker = ticker.upper()
    if ticker not in state["bars"]:
        raise HTTPException(404, f"No data for {ticker}")
    df = pd.DataFrame(state["bars"][ticker])
    try:
        model, acc = train_model(df)
        state["model"] = model
        state["model_accuracy"] = acc
        joblib.dump(model, MODEL_PATH)
        return {"ok": True, "accuracy": acc}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/health")
def health():
    return {"status": "ok"}


# ── CSP endpoints ──────────────────────────────────────────────────────────
@app.get("/market-regime")
async def market_regime_endpoint():
    """Current market regime: SPY vs 50-day SMA + VIX level."""
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(_market_regime(), timeout=20),
        )
    except (TimeoutError, RuntimeError) as e:
        raise HTTPException(503, str(e))
    return result


@app.get("/csp/scan")
async def csp_scan(
    min_return: float = Query(CSP_MIN_RETURN_PCT, description="Min weekly premium/strike %"),
    max_delta:  float = Query(CSP_MAX_DELTA,       description="Max absolute delta (0.05–0.25)"),
    refresh:    bool  = Query(False,               description="Force refresh even if cache is warm"),
):
    """
    Scan the CSP universe for cash-secured put opportunities.
    Includes earnings gate, IV rank, and market regime from yfinance.
    Results cached 5 minutes; pass refresh=true to bypass.
    """
    _require_connection()

    cache = state["scan_cache"]
    if not refresh and cache["csp"] is not None and cache["ts"]:
        age = (datetime.utcnow() - cache["ts"]).total_seconds()
        if age < SCAN_CACHE_TTL:
            return {
                "cached":        True,
                "age_seconds":   int(age),
                "count":         len(cache["csp"]),
                "candidates":    cache["csp"],
                "market_regime": cache.get("regime"),
            }

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(
                scan_csp(state["ib"], min_return, max_delta), timeout=180
            ),
        )
    except TimeoutError:
        raise HTTPException(504, "CSP scan timed out — reduce universe or try later")
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    candidates = result["candidates"]
    regime     = result["regime"]
    now = datetime.utcnow()
    cache["csp"]    = candidates
    cache["regime"] = regime
    cache["ts"]     = now
    return {
        "cached":        False,
        "scanned_at":    now.isoformat() + "Z",
        "count":         len(candidates),
        "candidates":    candidates,
        "market_regime": regime,
    }


@app.get("/csp/universe")
def get_csp_universe():
    return {"universe": CSP_UNIVERSE, "count": len(CSP_UNIVERSE)}


@app.post("/csp/universe/add")
def add_to_csp_universe(req: AddTickerRequest):
    ticker = req.ticker.upper()
    if ticker not in CSP_UNIVERSE:
        CSP_UNIVERSE.append(ticker)
        state["scan_cache"]["csp"] = None   # invalidate cache
    return {"ok": True, "universe": CSP_UNIVERSE}


# ── LEAP endpoints ─────────────────────────────────────────────────────────
@app.get("/leaps/scan")
async def leaps_scan(
    refresh: bool = Query(False, description="Force refresh even if cache is warm"),
):
    """
    Scan the CSP universe for LEAP call opportunities (6–18 month expiry).
    Includes earnings gate, IV rank (lower = better for buyers), and market regime.
    Only considers tickers with BUY/HOLD XGB signal and positive momentum.
    """
    _require_connection()

    cache = state["scan_cache"]
    if not refresh and cache["leaps"] is not None and cache["ts"]:
        age = (datetime.utcnow() - cache["ts"]).total_seconds()
        if age < SCAN_CACHE_TTL:
            return {
                "cached":        True,
                "age_seconds":   int(age),
                "count":         len(cache["leaps"]),
                "candidates":    cache["leaps"],
                "market_regime": cache.get("regime"),
            }

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(scan_leaps(state["ib"]), timeout=180),
        )
    except TimeoutError:
        raise HTTPException(504, "LEAP scan timed out — try again shortly")
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    candidates = result["candidates"]
    regime     = result["regime"]
    now = datetime.utcnow()
    cache["leaps"]  = candidates
    cache["regime"] = regime
    cache["ts"]     = now
    return {
        "cached":        False,
        "scanned_at":    now.isoformat() + "Z",
        "count":         len(candidates),
        "candidates":    candidates,
        "market_regime": regime,
    }


# ── Real-time option quote endpoint ────────────────────────────────────────
@app.get("/options/quote/{ticker}/{expiry}/{strike}/{right}")
async def option_quote(
    ticker: str,
    expiry: str,
    strike: float,
    right:  str,
):
    """
    Real-time snapshot quote + full Greeks for a single option contract.
    right = C (call) or P (put)
    expiry = YYYYMMDD
    Example: /options/quote/AAPL/20261218/200/P
    """
    _require_connection()
    if right.upper() not in ("C", "P"):
        raise HTTPException(400, "right must be C or P")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(
                _live_option_quote(state["ib"], ticker.upper(), expiry, strike, right),
                timeout=30,
            ),
        )
    except ValueError as e:
        # Unqualifiable contract → 404; no market data → 503
        msg = str(e)
        code = 503 if "subscription" in msg else 404
        raise HTTPException(code, msg)
    except (TimeoutError, RuntimeError) as e:
        raise HTTPException(503, str(e))

    return result


# ── Shared guard ───────────────────────────────────────────────────────────
def _require_connection():
    if not state["connected"] or not state["ib"]:
        raise HTTPException(503, "IBKR not connected — ensure TWS/IB Gateway is running")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
