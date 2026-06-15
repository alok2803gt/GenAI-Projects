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
    """Composite rank score — higher is better."""
    s = 0.0
    # 1) Weekly return headroom above the 4 % target (capped at 3×)
    s += min(row["weekly_return_pct"] / CSP_MIN_RETURN_PCT, 3.0) * 30
    # 2) Distance from assignment (lower delta = safer)
    s += (CSP_MAX_DELTA - abs(row["delta"])) / CSP_MAX_DELTA * 25
    # 3) Tight bid-ask spread (lower is better)
    spread_frac = row["spread_pct"] / 100
    s += max(0, (CSP_MAX_SPREAD_PCT - spread_frac) / CSP_MAX_SPREAD_PCT) * 20
    # 4) Underlying stock quality
    s += row.get("stock_quality", 0.5) * 25
    return round(s, 2)


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
) -> List[dict]:
    """
    For every ticker in CSP_UNIVERSE:
      1. Fetch current price (streaming cache, then snapshot)
      2. Get option chain structure
      3. Select put strikes 5–28 % OTM on the nearest weekly expiry
      4. Snapshot quotes + Greeks
      5. Filter: return ≥ min_return, |delta| ≤ max_delta, OI ≥ 50
      6. Score and return sorted list
    All tickers run concurrently (up to 5 at a time) to minimise scan latency.
    """
    expiry0 = _next_expiry(0)   # This Friday
    expiry1 = _next_expiry(1)   # Next Friday (fallback)
    sem = asyncio.Semaphore(5)  # Max 5 concurrent IBKR request streams

    async def _scan_ticker(ticker: str) -> List[dict]:
        async with sem:
            try:
                stock_price = await _get_stock_price(ib, ticker)
                if stock_price <= 0:
                    log.debug(f"CSP [{ticker}]: no price — skipping")
                    return []

                stock = Stock(ticker, "SMART", "USD")
                await ib.qualifyContractsAsync(stock)

                chains = await ib.reqSecDefOptParamsAsync(ticker, "", "STK", stock.conId)
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

                    strike = td.contract.strike
                    otm_pct = (stock_price - strike) / stock_price * 100
                    weekly_return_pct = mid / strike * 100
                    spread_pct = (ask - bid) / mid * 100
                    oi  = td.openInterest or 0
                    vol = td.volume or 0

                    if weekly_return_pct < min_return:        continue
                    if abs(delta) > max_delta:                continue
                    if oi < CSP_MIN_OI:                       continue
                    if spread_pct / 100 > CSP_MAX_SPREAD_PCT: continue

                    exp_move = _expected_weekly_move(stock_price, iv)
                    sigma_otm = (stock_price - strike) / exp_move if exp_move > 0 else 0

                    sq = _stock_quality_score(ticker)
                    row = {
                        "ticker":             ticker,
                        "expiry":             expiry,
                        "dte":                dte,
                        "strike":             strike,
                        "stock_price":        round(stock_price, 2),
                        "otm_pct":            round(otm_pct, 2),
                        "sigma_otm":          round(sigma_otm, 2),
                        "bid":                round(bid, 2),
                        "ask":                round(ask, 2),
                        "mid":                round(mid, 2),
                        "weekly_return_pct":  round(weekly_return_pct, 2),
                        "annualized_return":  round(weekly_return_pct * 52, 1),
                        "delta":              round(delta, 4),
                        "iv_pct":             round(iv * 100, 1),
                        "theta_daily":        round(theta, 4),
                        "open_interest":      oi,
                        "volume":             vol,
                        "spread_pct":         round(spread_pct, 2),
                        "stock_quality":      sq,
                        "assignment_risk":    _assignment_risk(delta, otm_pct),
                        "xgb_signal":         state["signals"].get(ticker, {}).get("label", "N/A"),
                        "cash_required":      round(strike * 100, 2),
                        "premium_collected":  round(mid * 100, 2),
                    }
                    row["score"] = _score_csp(row)
                    rows.append(row)

                return rows

            except Exception as e:
                log.warning(f"CSP scan error [{ticker}]: {e}")
                return []

    results = await asyncio.gather(*[_scan_ticker(t) for t in CSP_UNIVERSE])
    candidates = [row for ticker_rows in results for row in ticker_rows]
    return sorted(candidates, key=lambda x: x["score"], reverse=True)


# ── LEAP scan ──────────────────────────────────────────────────────────────
async def scan_leaps(ib: IB) -> List[dict]:
    """
    For every ticker in CSP_UNIVERSE find LEAP calls:
      • Expiry 6–18 months out
      • Delta 0.45–0.75 (near-ATM to 10 % OTM)
      • IV ≤ 70 % (don't overpay)
      • Only on tickers with BUY/HOLD XGB signal and positive momentum
    """
    today = date.today()
    sem = asyncio.Semaphore(5)

    async def _scan_ticker(ticker: str) -> List[dict]:
        async with sem:
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

                chains = await ib.reqSecDefOptParamsAsync(ticker, "", "STK", stock.conId)
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

                    if not (LEAP_MIN_DELTA <= delta <= LEAP_MAX_DELTA):
                        continue
                    if iv > LEAP_MAX_IV:
                        continue

                    strike = td.contract.strike
                    itm_otm_pct = (stock_price - strike) / stock_price * 100
                    cost_per_contract = round(mid * 100, 2)
                    spread_pct = (ask - bid) / mid * 100
                    breakeven = strike + mid

                    row = {
                        "ticker":             ticker,
                        "expiry":             chosen_expiry,
                        "dte":                dte,
                        "strike":             strike,
                        "stock_price":        round(stock_price, 2),
                        "itm_otm_pct":        round(itm_otm_pct, 2),
                        "breakeven":          round(breakeven, 2),
                        "breakeven_move_pct": round((breakeven - stock_price) / stock_price * 100, 2),
                        "bid":                round(bid, 2),
                        "ask":                round(ask, 2),
                        "mid":                round(mid, 2),
                        "cost_per_contract":  cost_per_contract,
                        "delta":              round(delta, 4),
                        "iv_pct":             round(iv * 100, 1),
                        "theta_daily":        round(theta, 4),
                        "vega":               round(vega, 4),
                        "spread_pct":         round(spread_pct, 2),
                        "stock_quality":      sq,
                        "xgb_signal":         sig.get("label", "N/A"),
                        "xgb_prob":           sig.get("prob", None),
                    }
                    row["score"] = round(sq * 60 + (1 - abs(delta - 0.60)) * 40, 2)
                    rows.append(row)

                return rows

            except Exception as e:
                log.warning(f"LEAP scan error [{ticker}]: {e}")
                return []

    results = await asyncio.gather(*[_scan_ticker(t) for t in CSP_UNIVERSE])
    candidates = [row for ticker_rows in results for row in ticker_rows]
    return sorted(candidates, key=lambda x: x["score"], reverse=True)


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
@app.get("/csp/scan")
async def csp_scan(
    min_return: float = Query(CSP_MIN_RETURN_PCT, description="Min weekly premium/strike %"),
    max_delta:  float = Query(CSP_MAX_DELTA,       description="Max absolute delta (0.05–0.25)"),
    refresh:    bool  = Query(False,               description="Force refresh even if cache is warm"),
):
    """
    Scan the CSP universe for cash-secured put opportunities.
    Results are cached for 5 minutes; pass refresh=true to bypass.
    """
    _require_connection()

    cache = state["scan_cache"]
    if not refresh and cache["csp"] is not None and cache["ts"]:
        age = (datetime.utcnow() - cache["ts"]).total_seconds()
        if age < SCAN_CACHE_TTL:
            return {
                "cached": True,
                "age_seconds": int(age),
                "count": len(cache["csp"]),
                "candidates": cache["csp"],
            }

    try:
        candidates = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(
                scan_csp(state["ib"], min_return, max_delta), timeout=180
            ),
        )
    except TimeoutError:
        raise HTTPException(504, "CSP scan timed out — reduce universe or try later")
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    now = datetime.utcnow()
    cache["csp"] = candidates
    cache["ts"]  = now
    return {
        "cached": False,
        "scanned_at": now.isoformat() + "Z",
        "count": len(candidates),
        "candidates": candidates,
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
    Only considers tickers with BUY/HOLD XGB signal and positive momentum.
    """
    _require_connection()

    cache = state["scan_cache"]
    if not refresh and cache["leaps"] is not None and cache["ts"]:
        age = (datetime.utcnow() - cache["ts"]).total_seconds()
        if age < SCAN_CACHE_TTL:
            return {
                "cached": True,
                "age_seconds": int(age),
                "count": len(cache["leaps"]),
                "candidates": cache["leaps"],
            }

    try:
        candidates = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(scan_leaps(state["ib"]), timeout=180),
        )
    except TimeoutError:
        raise HTTPException(504, "LEAP scan timed out — try again shortly")
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    now = datetime.utcnow()
    cache["leaps"] = candidates
    cache["ts"]    = now
    return {
        "cached": False,
        "scanned_at": now.isoformat() + "Z",
        "count": len(candidates),
        "candidates": candidates,
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
