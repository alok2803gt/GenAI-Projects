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
import json
import logging
import math
import sqlite3
import threading
import traceback
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from ib_insync import IB, LimitOrder, Option, Stock, util
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
TWS_PORT = 7497          # 7497=TWS paper | 7496=TWS live | 4002=IB Gateway paper
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
CSP_MIN_RETURN_PCT  = 0.75   # Weekly premium / strike  ≥ 0.75 % (realistic for VIX 15-20)
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

# ── Config — IV history & technical indicators ─────────────────────────────
IV_HISTORY_PATH     = "iv_history.json"
IV_HISTORY_MIN_PTS  = 20    # samples needed before percentile rank is reliable
JOURNAL_DB_PATH     = "trade_journal.db"
JOURNAL_MIN_TRADES  = 20    # trades needed before learned model is reliable
RETRAIN_EVERY       = 5     # retrain model every N new closed trades
AT_STATE_PATH       = "autotrader_state.json"  # persisted across restarts

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

# Fallback — original hardcoded universe (never mutated)
_DEFAULT_UNIVERSE: List[str] = list(CSP_UNIVERSE)

# Candidate pool screened nightly → top 20-30 replace CSP_UNIVERSE dynamically
CANDIDATE_POOL: List[str] = [
    # ── Mega-cap tech ────────────────────────────────────────────────────
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "ORCL", "IBM", "CSCO",
    # ── Semiconductors ───────────────────────────────────────────────────
    "AMD", "INTC", "QCOM", "TXN", "AVGO", "MU", "AMAT", "LRCX", "KLAC",
    "MRVL", "SMCI", "ON", "MPWR",
    # ── Software / Cloud / AI ────────────────────────────────────────────
    "NOW", "CRM", "ADBE", "INTU", "SNOW", "PLTR", "UBER", "ABNB",
    "NET", "DDOG", "ZS", "CRWD", "PANW",
    # ── Financials ───────────────────────────────────────────────────────
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "V", "MA",
    "SCHW", "COF", "USB", "PNC", "TFC", "SPGI", "MCO",
    # ── Healthcare ───────────────────────────────────────────────────────
    "UNH", "JNJ", "PFE", "MRK", "ABBV", "BMY", "AMGN", "GILD",
    "LLY", "TMO", "DHR", "ELV", "HUM", "CI", "ISRG", "VRTX", "REGN",
    # ── Energy ───────────────────────────────────────────────────────────
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "PSX", "MPC", "VLO",
    # ── Consumer Discretionary ───────────────────────────────────────────
    "WMT", "COST", "TGT", "HD", "LOW", "MCD", "SBUX", "NKE",
    "BKNG", "LULU", "CMG",
    # ── Consumer Staples ─────────────────────────────────────────────────
    "PG", "KO", "PEP", "PM", "MO", "MDLZ", "CL",
    # ── Industrials / Defense ────────────────────────────────────────────
    "CAT", "DE", "BA", "HON", "GE", "LMT", "RTX", "NOC", "GD",
    "UPS", "FDX", "ETN", "EMR",
    # ── Comm / Media ─────────────────────────────────────────────────────
    "NFLX", "DIS", "CMCSA", "T", "VZ", "ROKU", "SPOT",
    # ── Real Estate / Infrastructure ─────────────────────────────────────
    "AMT", "PLD", "EQIX",
    # ── ETFs ─────────────────────────────────────────────────────────────
    "SPY", "QQQ", "IWM", "GLD", "XLE", "XLF", "XLK", "XLV", "ARKK",
    # ── High-vol / options-active ─────────────────────────────────────────
    "COIN", "HOOD", "SOFI", "RBLX", "MARA", "RIOT",
    "SHOP", "SQ", "PYPL", "SNAP", "PINS",
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
        "csp": None, "leaps": None, "0dte": None, "earnings_iv": None, "ts": None,
    },
    # External validation cache (yfinance)
    "ext_cache": {
        "earnings": {},   # ticker → {"days": int|None, "ts": datetime}
        "iv_rank":  {},   # ticker → {"iv": float|None, "rank": float, "rv_lo": float|None, "rv_hi": float|None, "ts": datetime}
        "regime":   None, # full market regime dict
    },
    # OPRA subscription status — set once at startup, re-checked on reconnect
    "opra_active": None,   # None = not yet checked, True/False thereafter
    # IV history — persisted daily snapshots for true percentile rank
    "iv_history": {},      # ticker → [{"date":"YYYY-MM-DD","iv":float}, ...]
    # Dynamic universe screening
    "universe_scores":        [],   # [{ticker, score, price, avg_vol, rsi14, ...}, ...]
    "universe_last_screened": None, # ISO timestamp of last screen run
    # Auto-trader state
    "autotrader": {
        "enabled": False,
        "config": {
            "max_positions":     5,
            "profit_target_pct": 0.65,
            "stop_loss_mult":    5.0,
            "scan_types":        ["csp"],
            "csp_capital":       20000.0,
            "leap_capital":      5000.0,
            # Kelly criterion
            "use_kelly":         True,
            "total_capital":     100000.0,
            "assumed_win_rate":  0.85,
            # Trailing exit
            "trailing_exit":     True,
            # Auto-hedge
            "auto_hedge":        False,
            "hedge_threshold":   100.0,
        },
        "positions":          {},     # contract_key → entry metadata
        "log":                [],     # [{time, action, detail}, …] last 200
        "last_run":           None,
        "premium_collected":  0.0,    # total realized CSP profit
        "leap_budget":        0.0,    # 50 % of premium_collected → LEAP fund
    },
    # Reconnect request (set by /reconnect endpoint, read by streaming loop)
    "reconnect_port": None,
    # Continuous learning
    "model_learned":   None,   # XGBoost trained on real trade outcomes
    "model_version":   0,
    "trades_since_retrain": 0,
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
                elif errorCode in (2104, 2106, 2158, 2119):
                    pass  # benign farm/connectivity notifications
                elif 100 <= errorCode < 10000:
                    sym = getattr(contract, "symbol", "") if contract else ""
                    msg = f"IBKR error {errorCode} reqId={reqId} {sym}: {errorString}"
                    log.warning(msg)
                    if errorCode in (201, 202, 203, 321, 10147, 10148):
                        _at_log("IBKR-ERR", msg)

            ib.errorEvent += _on_ib_error

            port = state.get("reconnect_port") or TWS_PORT
            state["reconnect_port"] = None          # consume the request
            await ib.connectAsync(TWS_HOST, port, clientId=TWS_CLIENT_ID, timeout=15)
            log.info(f"Connected to IBKR  {TWS_HOST}:{port}")
            state["ib"] = ib
            state["connected"] = True
            state["error"] = None

            # Cancel any Inactive orders left over from previous sessions.
            # Inactive orders are stuck in TWS (not sent to exchange) and can
            # trigger IBKR's duplicate-order suppression on subsequent placements.
            await asyncio.sleep(1)   # give TWS a moment to send us existing orders
            stale = [t for t in ib.openTrades() if t.orderStatus.status == "Inactive"]
            for t in stale:
                try:
                    ib.cancelOrder(t.order)
                except Exception:
                    pass
            if stale:
                log.info(f"Cancelled {len(stale)} stale Inactive orders from previous session")
                _at_log("SYSTEM", f"Cancelled {len(stale)} stale Inactive orders on reconnect")

            ctx["known"] = await _subscribe_pending(ib, ctx["known"])

            # Wait briefly so the options data farm is fully ready before probing.
            # Without this, reqTickersAsync races with bar subscriptions and returns nan.
            await asyncio.sleep(3)
            state["opra_active"] = await _check_opra_subscription(ib)
            # Retry once — occasionally the first snapshot arrives before OPRA feed is warm
            if not state["opra_active"]:
                await asyncio.sleep(3)
                state["opra_active"] = await _check_opra_subscription(ib)

            # Heartbeat: keep usopt farm alive + re-check OPRA every 4 min.
            # Without this, the options data farm idles and the first scan
            # after a quiet period gets NaN on every contract.
            _heartbeat_tick = 0
            while ib.isConnected():
                ctx["known"] = await _subscribe_pending(ib, ctx["known"])
                _heartbeat_tick += 1
                if _heartbeat_tick % 24 == 0:   # every 24 × 10 s = 4 min
                    state["opra_active"] = await _check_opra_subscription(ib)
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

# Risk-free rate approximation (US 3-month T-bill)
_RF_RATE = 0.045

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _safe_float(val, default=None):
    """Convert IBKR/yfinance values to float, returning default for None/NaN/Inf."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _json_safe(obj):
    """Recursively replace NaN/Inf floats with None so the response serializes cleanly."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

def _bs_delta(S: float, K: float, T: float, sigma: float, is_put: bool) -> float:
    """Black-Scholes delta. T in years."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (_RF_RATE + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return (_norm_cdf(d1) - 1.0) if is_put else _norm_cdf(d1)

def _bs_theta(S: float, K: float, T: float, sigma: float, is_put: bool) -> float:
    """Black-Scholes theta per calendar day."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (_RF_RATE + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    phi = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    if is_put:
        th = -S * phi * sigma / (2 * math.sqrt(T)) + _RF_RATE * K * math.exp(-_RF_RATE * T) * _norm_cdf(-d2)
    else:
        th = -S * phi * sigma / (2 * math.sqrt(T)) - _RF_RATE * K * math.exp(-_RF_RATE * T) * _norm_cdf(d2)
    return th / 365.0


def _next_expiry(weeks_out: int = 0) -> str:
    """Next Friday (or n Fridays out) in YYYYMMDD format."""
    today = date.today()
    days_to_friday = (4 - today.weekday()) % 7 or 7
    friday = today + timedelta(days=days_to_friday + weeks_out * 7)
    return friday.strftime("%Y%m%d")


async def _fetch_opra_chain(
    ib: IB,
    ticker: str,
    right: str,
    stock_price: float,
    dte_min: int,
    dte_max: int,
    otm_lo_pct: float = 3.0,
    otm_hi_pct: float = 30.0,
    max_strikes: int = 10,
) -> tuple:
    """
    Fetch live IBKR/OPRA options data for a ticker via reqSecDefOptParamsAsync
    + reqTickersAsync.  Returns (list[Ticker], expiry_YYYYMMDD, dte_int).
    Raises ValueError if chain unavailable — caller should fall back to yfinance.
    """
    stock = Stock(ticker, "SMART", "USD")
    await ib.qualifyContractsAsync(stock)
    if not stock.conId:
        raise ValueError(f"qualify failed: {ticker}")

    chains = await ib.reqSecDefOptParamsAsync(ticker, "", "STK", stock.conId)
    if not chains:
        raise ValueError(f"no chain params: {ticker}")
    chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

    today_d = date.today()
    chosen_exp, chosen_dte = None, 0
    for exp in sorted(chain.expirations):
        try:
            exp_date = datetime.strptime(exp, "%Y%m%d").date()
            dte = (exp_date - today_d).days
            if dte_min <= dte <= dte_max:
                chosen_exp, chosen_dte = exp, dte
                break
        except ValueError:
            continue
    if not chosen_exp:
        raise ValueError(f"no expiry in {dte_min}-{dte_max} DTE for {ticker}")

    lo = stock_price * (1 - otm_hi_pct / 100)
    hi = stock_price * (1 - otm_lo_pct / 100) if right == "P" else stock_price * (1 + otm_hi_pct / 100)
    strikes = sorted(
        [s for s in chain.strikes if lo <= s <= hi],
        key=lambda s: abs(s - stock_price) if right == "C" else -s,
    )[:max_strikes]
    if not strikes:
        raise ValueError(f"no strikes in OTM range for {ticker} {right}")

    contracts = [Option(ticker, chosen_exp, s, right, "SMART") for s in strikes]
    await ib.qualifyContractsAsync(*contracts)
    valid = [c for c in contracts if c.conId]
    if not valid:
        raise ValueError(f"no qualified contracts for {ticker} {right}")

    tickers_data = await ib.reqTickersAsync(*valid)
    return tickers_data, chosen_exp, chosen_dte


async def _institutional_signals(ticker: str, stock_price: float) -> dict:
    """
    Derive max pain, gamma wall and put/call ratios from the full yfinance chain.
    OI is a daily figure so 15-min staleness is acceptable for these aggregate signals.
    """
    def _calc():
        try:
            t    = yf.Ticker(ticker)
            exps = t.options or []
            if not exps:
                return {}
            chain    = t.option_chain(exps[0])
            puts_df  = chain.puts
            calls_df = chain.calls

            tot_put_oi  = float(puts_df["openInterest"].fillna(0).sum())
            tot_call_oi = float(calls_df["openInterest"].fillna(0).sum())
            tot_put_vol = float(puts_df["volume"].fillna(0).sum())
            tot_call_vol= float(calls_df["volume"].fillna(0).sum())

            pc_oi_ratio  = round(tot_put_oi  / max(tot_call_oi,  1), 2)
            pc_vol_ratio = round(tot_put_vol / max(tot_call_vol, 1), 2)

            # Max pain: price at which total option pain (ITM value) is minimised
            all_strikes = sorted(set(
                list(puts_df["strike"].values) + list(calls_df["strike"].values)
            ))
            min_pain, max_pain_s = None, stock_price
            for s in all_strikes:
                pr = puts_df[puts_df["strike"] == s]
                cr = calls_df[calls_df["strike"] == s]
                pain = (
                    (float(pr["openInterest"].iloc[0]) * max(0, s - stock_price) if not pr.empty else 0) +
                    (float(cr["openInterest"].iloc[0]) * max(0, stock_price - s) if not cr.empty else 0)
                )
                if min_pain is None or pain < min_pain:
                    min_pain, max_pain_s = pain, s

            # Gamma wall: strike with highest combined OI (where MM hedging is heaviest)
            all_oi: dict = {}
            for _, r in puts_df.iterrows():
                all_oi[r["strike"]] = all_oi.get(r["strike"], 0) + (r["openInterest"] or 0)
            for _, r in calls_df.iterrows():
                all_oi[r["strike"]] = all_oi.get(r["strike"], 0) + (r["openInterest"] or 0)
            gamma_wall = max(all_oi, key=all_oi.get) if all_oi else stock_price

            return {
                "max_pain":    round(float(max_pain_s), 2),
                "gamma_wall":  round(float(gamma_wall), 2),
                "pc_oi_ratio": pc_oi_ratio,
                "pc_vol_ratio":pc_vol_ratio,
            }
        except Exception as e:
            log.debug("Institutional signals failed %s: %s", ticker, e)
            return {}

    return await asyncio.get_event_loop().run_in_executor(None, _calc)


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
    Uses realistic fill return (bid + 40% spread) as primary return metric.
    Institutional OPRA signals adjust the score up or down.
    """
    s = 0.0
    # 1) Realistic weekly return above min target (capped at 3×)          max 75
    s += min(row["weekly_return_pct"] / CSP_MIN_RETURN_PCT, 3.0) * 25
    # 2) Delta safety — distance from assignment                           max 20
    s += (CSP_MAX_DELTA - abs(row["delta"])) / CSP_MAX_DELTA * 20
    # 3) Liquidity (OI + volume + spread)                                  max 20
    s += (row.get("liquidity_score", 50) / 100) * 20
    # 4) Stock quality (technical trend strength)                          max 20
    s += row.get("stock_quality", 0.5) * 20
    # 5) IV rank — high IV = better premium for sellers                    max 15
    s += (row.get("iv_rank", 50) / 100) * 15
    # 6) OPRA institutional signals (bonus / penalty)
    # Strike above max pain → market makers incentivised to keep price here +8
    if row.get("max_pain") and row["strike"] >= row["max_pain"]:
        s += 8
    # Put/call volume ratio < 0.7 → call-dominant flow today (bullish)    +6
    pc_vol = row.get("pc_vol_ratio", 1.0)
    if pc_vol < 0.7:
        s += 6
    elif pc_vol > 1.5:
        s -= 6   # heavy put buying — bearish flow
    # No unusual put activity at this specific strike                      +4
    if row.get("vol_oi_ratio", 1.0) < 1.0:
        s += 4
    elif row.get("vol_oi_ratio", 1.0) > 2.0:
        s -= 8   # volume >> OI: someone is making a directional put bet
    # Ask-heavy order book → buyers aggressively lifting offers (bullish)  +4
    flow = row.get("flow_flag", "BALANCED")
    if flow == "ASK HEAVY":
        s += 4
    elif flow == "BID HEAVY":
        s -= 4
    return round(s, 2)


# ── IV history (persistent percentile rank) ───────────────────────────────

def _load_iv_history() -> dict:
    try:
        if os.path.exists(IV_HISTORY_PATH):
            with open(IV_HISTORY_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_iv_history() -> None:
    try:
        with open(IV_HISTORY_PATH, "w") as f:
            json.dump(state["iv_history"], f)
    except Exception as e:
        log.debug(f"IV history save failed: {e}")


# ── Auto-trader state persistence ──────────────────────────────────────────

def _at_save_state() -> None:
    """Persist auto-trader positions + config to disk so a restart is safe."""
    at = state["autotrader"]
    payload = {
        "enabled":           at["enabled"],
        "config":            at["config"],
        "positions":         at["positions"],
        "premium_collected": at["premium_collected"],
        "leap_budget":       at["leap_budget"],
        "model_version":     state.get("model_version", 0),
    }
    try:
        with open(AT_STATE_PATH, "w") as f:
            json.dump(payload, f, indent=2, default=str)
    except Exception as e:
        log.warning(f"Auto-trader state save failed: {e}")


def _at_load_state() -> None:
    """Reload auto-trader state from disk on startup."""
    if not os.path.exists(AT_STATE_PATH):
        return
    try:
        with open(AT_STATE_PATH, "r") as f:
            saved = json.load(f)
        at = state["autotrader"]
        at["enabled"]           = saved.get("enabled", False)
        at["premium_collected"] = saved.get("premium_collected", 0.0)
        at["leap_budget"]       = saved.get("leap_budget", 0.0)
        # Merge saved config over defaults (preserves any missing keys)
        at["config"].update(saved.get("config", {}))
        # Restore positions — the monitor loop will reconcile against live portfolio
        at["positions"] = saved.get("positions", {})
        state["model_version"] = saved.get("model_version", 0)
        log.info(
            f"Auto-trader state restored: enabled={at['enabled']}, "
            f"positions={len(at['positions'])}, model_v{state['model_version']}"
        )
    except Exception as e:
        log.warning(f"Auto-trader state load failed (starting fresh): {e}")

def _record_iv(ticker: str, iv_frac: float) -> None:
    """Store today's IV; keep last 252 trading-day snapshots (≈1 year)."""
    today_str = date.today().isoformat()
    hist = state["iv_history"].setdefault(ticker, [])
    hist[:] = [h for h in hist if h.get("date") != today_str]
    hist.append({"date": today_str, "iv": round(iv_frac, 4)})
    state["iv_history"][ticker] = hist[-252:]

def _iv_percentile(ticker: str, current_iv_frac: float) -> Optional[float]:
    """0-100 percentile rank of current IV within stored history."""
    hist = state["iv_history"].get(ticker, [])
    ivs = [h["iv"] for h in hist if h.get("iv") is not None]
    if len(ivs) < IV_HISTORY_MIN_PTS:
        return None
    return round(sum(1 for v in ivs if v <= current_iv_frac) / len(ivs) * 100, 1)


# ── Technical indicators (RSI-14 + MACD 12/26/9 + SMA crossovers) ─────────

def _compute_indicators_from_closes(closes: pd.Series) -> dict:
    empty = {"rsi14": None, "macd": None, "macd_signal": None,
             "macd_hist": None, "above_sma20": None, "above_sma50": None}
    if len(closes) < 26:
        return empty
    def _safe(s):
        v = s.iloc[-1]
        return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)

    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + gain / (loss + 1e-9))

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    msig  = macd.ewm(span=9, adjust=False).mean()
    mhist = macd - msig

    sma20 = closes.rolling(20).mean()
    sma50 = closes.rolling(50).mean() if len(closes) >= 50 else pd.Series([float("nan")] * len(closes), index=closes.index)

    last = float(closes.iloc[-1])
    s20  = _safe(sma20)
    s50  = _safe(sma50)
    return {
        "rsi14":       round(_safe(rsi)  or 50, 1),
        "macd":        round(_safe(macd) or 0,  4),
        "macd_signal": round(_safe(msig) or 0,  4),
        "macd_hist":   round(_safe(mhist) or 0, 4),
        "above_sma20": bool(last > s20) if s20 is not None else None,
        "above_sma50": bool(last > s50) if s50 is not None else None,
    }


async def _tech_indicators(ticker: str) -> dict:
    """RSI-14 + MACD from streaming bars (live 5-min) or yfinance daily fallback."""
    empty = {"rsi14": None, "macd": None, "macd_signal": None,
             "macd_hist": None, "above_sma20": None, "above_sma50": None}
    bars = state["bars"].get(ticker)
    if bars and len(bars) >= 30:
        closes = pd.Series([b["close"] for b in bars], dtype=float)
        return _compute_indicators_from_closes(closes)
    loop = asyncio.get_event_loop()
    try:
        def _fetch():
            hist = yf.Ticker(ticker).history(period="3mo")
            if hist.empty or len(hist) < 26:
                return None
            return hist["Close"].astype(float)
        closes = await loop.run_in_executor(None, _fetch)
        if closes is not None:
            return _compute_indicators_from_closes(closes)
    except Exception as e:
        log.debug(f"Tech indicators [{ticker}]: {e}")
    return empty


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

    # Override with true percentile rank once enough history accumulates.
    # Falls back to the rolling-RV proxy above until IV_HISTORY_MIN_PTS days stored.
    if current_iv:
        _record_iv(ticker, current_iv)
        _save_iv_history()
        hist_rank = _iv_percentile(ticker, current_iv)
        if hist_rank is not None:
            rank = hist_rank

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
    # IV rank thresholds are enforced as direct numeric filters in _filter_csp_recommended
    # and _filter_leap_recommended — no need to double-block via warnings here.
    return warnings


async def _check_opra_subscription(ib: IB) -> bool:
    """
    Probe whether real-time OPRA options data is active on this account.
    Do NOT call reqMarketDataType(1) — it triggers ARCA equity TOP/ALL requests
    which require a separate subscription and poison the options snapshot.
    """
    try:
        spy_price = await _get_stock_price(ib, "SPY")
        if spy_price <= 0:
            spy_price = 740.0
        strike = float(round(spy_price / 5) * 5)

        # Try current-week expiry first; fall back to next week if unqualified
        contract = None
        for weeks_out in (0, 1):
            expiry = _next_expiry(weeks_out)
            c = Option("SPY", expiry, strike, "P", "SMART")
            await ib.qualifyContractsAsync(c)
            if c.conId:
                contract = c
                break

        if contract is None:
            log.warning("OPRA check: could not qualify SPY put contract")
            return False

        log.info(f"OPRA check: SPY {expiry} {strike}P  conId={contract.conId}  spot=${spy_price:.2f}")

        [td] = await ib.reqTickersAsync(contract)

        raw_bid = td.bid
        raw_ask = td.ask
        log.info(
            f"OPRA check raw: bid={raw_bid}  ask={raw_ask}  "
            f"last={getattr(td,'last',None)}  close={getattr(td,'close',None)}  greeks={td.modelGreeks}"
        )

        def _valid(v) -> bool:
            return v is not None and not math.isnan(v) and v > 0

        active = _valid(raw_bid) and _valid(raw_ask)
        log.info(f"OPRA subscription: {'ACTIVE' if active else 'NOT SUBSCRIBED / DELAYED'}")
        return active
    except Exception as e:
        log.warning(f"OPRA check failed: {e}", exc_info=True)
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


# ── Dynamic universe screener ─────────────────────────────────────────────

async def _screen_universe(top_n: int = 25) -> Optional[List[str]]:
    """
    Screen CANDIDATE_POOL (~155 tickers) with yfinance bulk download.
    Criteria: price $20-$800, 30-day ADV >500K shares, above SMA-50,
              RSI-14 in 40-65 range, IV rank >25% (real iv_history or HV proxy).
    Returns top_n tickers by composite score, or None on failure.
    """
    log.info("Universe screen: scoring %d candidates → top %d", len(CANDIDATE_POOL), top_n)
    try:
        def _bulk_dl():
            return yf.download(
                CANDIDATE_POOL,
                period="1y",
                interval="1d",
                group_by="ticker",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, _bulk_dl)

        scored = []
        for ticker in CANDIDATE_POOL:
            try:
                # Extract close & volume series for this ticker
                if isinstance(df.columns, pd.MultiIndex):
                    if ticker not in df.columns.get_level_values(0):
                        continue
                    closes = df[ticker]["Close"].dropna()
                    vol_series = df[ticker].get("Volume", pd.Series(dtype=float)).dropna()
                else:
                    closes = df["Close"].dropna()
                    vol_series = df.get("Volume", pd.Series(dtype=float)).dropna()

                if len(closes) < 21:
                    continue

                price = float(closes.iloc[-1])
                if not (20.0 <= price <= 800.0):
                    continue

                # 30-day average daily volume
                avg_vol = float(vol_series.tail(30).mean()) if len(vol_series) >= 5 else 0.0
                if avg_vol < 500_000:
                    continue

                # SMA-50 (need ≥50 bars; if fewer, treat as unknown)
                above_sma50: Optional[bool] = None
                if len(closes) >= 50:
                    sma50 = float(closes.rolling(50).mean().iloc[-1])
                    above_sma50 = bool(price > sma50)

                # RSI-14
                delta = closes.diff()
                gain = delta.clip(lower=0).rolling(14).mean()
                loss = (-delta.clip(upper=0)).rolling(14).mean()
                rs_val = gain.iloc[-1] / (loss.iloc[-1] if loss.iloc[-1] != 0 else float("nan"))
                rsi = float(100 - 100 / (1 + rs_val)) if not np.isnan(rs_val) else 50.0

                # IV rank: prefer persisted iv_history (real IV); fall back to
                # 30d rolling HV rank computed from the downloaded 1-year series.
                # HV rank tracks IV rank directionally and is always available
                # without extra API calls. Upgrades to real IV once 20 daily
                # iv_history entries accumulate.
                iv_rank: Optional[int] = None
                hv_proxy = False
                hist = state.get("iv_history", {}).get(ticker, [])
                if len(hist) >= 20:
                    ivs = [h["iv"] for h in hist]
                    iv_range = max(ivs) - min(ivs)
                    if iv_range > 0:
                        iv_rank = int((ivs[-1] - min(ivs)) / iv_range * 100)
                else:
                    try:
                        rets = closes.pct_change().dropna()
                        hv_series = rets.rolling(30).std().dropna() * math.sqrt(252)
                        if len(hv_series) >= 20:
                            hv_cur = float(hv_series.iloc[-1])
                            hv_lo  = float(hv_series.min())
                            hv_hi  = float(hv_series.max())
                            if hv_hi > hv_lo:
                                iv_rank = int(round(min(max(
                                    (hv_cur - hv_lo) / (hv_hi - hv_lo) * 100, 0), 100)))
                                hv_proxy = True
                    except Exception:
                        pass

                # ── Composite score ──────────────────────────────────────
                score = 0.0

                # Momentum: being above SMA-50 is the single strongest signal
                if above_sma50 is True:
                    score += 25.0
                elif above_sma50 is None:
                    score += 8.0  # not enough history; neutral

                # RSI sweet-spot (not over-extended, not oversold dump)
                if 40.0 <= rsi <= 65.0:
                    score += 25.0
                elif 35.0 <= rsi < 40.0 or 65.0 < rsi <= 70.0:
                    score += 12.0
                elif rsi < 30.0 or rsi > 80.0:
                    score -= 10.0  # extreme — skip unless IV compensates

                # IV rank: higher = better premium for CSP sellers
                if iv_rank is not None:
                    if iv_rank >= 25:
                        score += min(iv_rank / 100.0, 0.80) * 25.0
                else:
                    score += 8.0  # unknown — neutral

                # Volume quality bonus
                if avg_vol >= 2_000_000:
                    score += 15.0
                elif avg_vol >= 1_000_000:
                    score += 10.0
                else:
                    score += 5.0

                # Continuity bonus — tickers already in the active universe
                if ticker in CSP_UNIVERSE:
                    score += 5.0

                scored.append({
                    "ticker":      ticker,
                    "score":       round(score, 1),
                    "price":       round(price, 2),
                    "avg_vol_30d": int(avg_vol),
                    "rsi14":       round(rsi, 1),
                    "above_sma50": above_sma50,
                    "iv_rank":     iv_rank,
                    "hv_proxy":    hv_proxy,
                })
            except Exception as _e:
                log.debug("Screen [%s]: %s", ticker, _e)

        if len(scored) < 5:
            log.warning("Universe screen: only %d tickers passed filters — keeping default", len(scored))
            return None

        scored.sort(key=lambda x: x["score"], reverse=True)
        state["universe_scores"] = scored
        state["universe_last_screened"] = datetime.now().isoformat()

        tickers = [t["ticker"] for t in scored[:top_n]]
        log.info("Universe screen complete (%d scored): %s", len(scored), ", ".join(tickers))
        return tickers

    except Exception as exc:
        log.error("Universe screen failed (%s) — keeping existing universe", exc)
        return None


async def _universe_scheduler() -> None:
    """Background task: refresh universe daily at 08:30 ET on weekdays."""
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    while True:
        try:
            now = datetime.now(ET)
            # Next 08:30 ET
            target = now.replace(hour=8, minute=30, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)
            # Skip weekends
            while target.weekday() >= 5:
                target += timedelta(days=1)
            wait = (target - now).total_seconds()
            log.info("Universe scheduler: next screen in %.0f s (at %s ET)", wait, target.strftime("%Y-%m-%d %H:%M"))
            await asyncio.sleep(wait)

            tickers = await _screen_universe()
            if tickers:
                CSP_UNIVERSE.clear()
                CSP_UNIVERSE.extend(tickers)
                state["scan_cache"]["csp"]   = None   # invalidate scan caches
                state["scan_cache"]["leaps"] = None
                log.info("Universe refreshed: %d tickers", len(CSP_UNIVERSE))
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error("Universe scheduler error: %s", exc)
            await asyncio.sleep(3600)  # back off 1 h on unexpected failure


# ── Recommendation filters (mirrors frontend JS) ──────────────────────────

def _filter_csp_recommended(candidates: list) -> list:
    # above_sma50 must be True or None (None = not enough history, give benefit of doubt)
    clean = [r for r in candidates
             if len(r.get("warnings", [])) == 0
             and r["liquidity_score"] >= 50
             and r["iv_rank"] >= 30
             and r["score"] >= 70
             and r.get("above_sma50") is not False          # reject stocks in downtrend
             and (r["earnings_days_out"] is None or r["earnings_days_out"] > 21)]
    # Augment with learned model score if available
    for r in clean:
        ls = _learned_score(r)
        r["learned_score"] = ls
        r["_sort_key"] = ls if ls is not None else r["score"]
    seen: set = set()
    out: list = []
    for r in sorted(clean, key=lambda x: x["_sort_key"], reverse=True):
        if r["ticker"] not in seen:
            seen.add(r["ticker"])
            out.append(r)
    return out


def _filter_leap_recommended(candidates: list) -> list:
    clean = [r for r in candidates
             if len(r.get("warnings", [])) == 0
             and r["liquidity_score"] >= 60
             and r.get("iv_rank", 50) <= 75]   # avoid buying when IV is too elevated
    for r in clean:
        ls = _learned_score(r)
        r["learned_score"] = ls
        r["_sort_key"] = ls if ls is not None else r["score"]
    seen: set = set()
    out: list = []
    for r in sorted(clean, key=lambda x: x["_sort_key"], reverse=True):
        if r["ticker"] not in seen:
            seen.add(r["ticker"])
            out.append(r)
    return out


# ── Auto-trader helpers ────────────────────────────────────────────────────

def _is_market_open() -> bool:
    """
    Returns True only during US options market hours.
    Options trade Mon-Fri 9:30 AM – 4:15 PM Eastern Time.
    Does not account for holidays — IBKR will simply reject those orders gracefully.
    """
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=15, second=0, microsecond=0)
    return market_open <= now_et <= market_close


def _at_log(action: str, detail: str) -> None:
    entry = {"time": datetime.utcnow().strftime("%H:%M:%S UTC"), "action": action, "detail": detail}
    at = state["autotrader"]
    at["log"].append(entry)
    at["log"] = at["log"][-200:]
    log.info("[AutoTrader] %s: %s", action, detail)


def _at_contract_key(c) -> str:
    return (f"{c.symbol}_{getattr(c,'right','')}"
            f"{getattr(c,'strike','')}{getattr(c,'lastTradeDateOrContractMonth','')}")


def _kelly_qty(cfg: dict, strike: float, t_type: str, mid_price: float = 0.0,
               regime: str = "BULL") -> int:
    """Half-Kelly position sizing, scaled by market regime."""
    p  = float(cfg.get("assumed_win_rate", 0.85))
    pt = float(cfg.get("profit_target_pct", 0.65))
    sl = float(cfg.get("stop_loss_mult", 5.0))
    b  = pt / sl if sl > 0 else pt / 5.0
    kelly = (p * (b + 1) - 1) / b if b > 0 else 0.0
    frac  = max(0.02, kelly * 0.5)             # half-Kelly
    # Regime penalty: reduce size in uncertain or falling market
    regime_scale = {"BULL": 1.0, "NEUTRAL": 0.6, "BEAR": 0.35, "UNKNOWN": 0.5}.get(regime, 0.5)
    total = float(cfg.get("total_capital", 100000.0))
    capital = total * frac * regime_scale
    if t_type == "csp":
        return max(1, int(capital / (strike * 100)))
    else:
        m = mid_price if mid_price > 0 else 5.0
        return max(1, int(capital / (m * 100)))


def _bs_put_price(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes put price (risk-free rate = 0)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * _norm_cdf(-d2) - S * _norm_cdf(-d1)


async def _autotrader_monitor_coro(ib: IB) -> None:
    at           = state["autotrader"]
    cfg          = at["config"]
    use_trailing = cfg.get("trailing_exit", True)
    if not at["positions"]:
        return
    for item in ib.portfolio():
        key = _at_contract_key(item.contract)
        if key not in at["positions"]:
            continue
        info       = at["positions"][key]
        upnl       = float(item.unrealizedPNL or 0)
        max_profit = info.get("max_profit", 0)
        if max_profit <= 0:
            continue

        if use_trailing:
            # Update high-water mark
            hw = max(info.get("high_water", 0.0), upnl)
            info["high_water"] = hw
            hw_pct = hw / max_profit
            # Tiered trailing floor: lock in more as profit builds
            if hw_pct >= 0.75:
                floor = 0.60 * max_profit
            elif hw_pct >= 0.50:
                floor = 0.35 * max_profit
            elif hw_pct >= 0.25:
                floor = 0.15 * max_profit
            else:
                floor = -cfg["stop_loss_mult"] * max_profit
            if upnl <= floor:
                reason = (f"trailing stop: floor=${floor:.0f}, uPnL=${upnl:.0f}, "
                          f"HW={hw_pct*100:.0f}% of max")
                _at_log("CLOSE", f"{key}: {reason}")
                info["exit_reason"] = "trailing_stop"
                await _autotrader_close_coro(ib, item, info, key)
        else:
            if upnl >= cfg["profit_target_pct"] * max_profit:
                _at_log("CLOSE", f"{key}: profit target hit (${upnl:.0f})")
                info["exit_reason"] = "profit_target"
                await _autotrader_close_coro(ib, item, info, key)
            elif upnl <= -cfg["stop_loss_mult"] * max_profit:
                _at_log("CLOSE", f"{key}: stop-loss hit (${upnl:.0f})")
                info["exit_reason"] = "stop_loss"
                await _autotrader_close_coro(ib, item, info, key)


async def _autotrader_close_coro(ib: IB, item, info: dict, key: str) -> None:
    c            = item.contract
    close_action = "BUY" if info["action"] == "SELL" else "SELL"
    qty          = max(1, abs(int(item.position)))
    contract     = Option(
        symbol=c.symbol,
        lastTradeDateOrContractMonth=c.lastTradeDateOrContractMonth,
        strike=float(c.strike),
        right=c.right,
        exchange="SMART",
        currency="USD",
        multiplier="100",
    )
    await ib.qualifyContractsAsync(contract)
    ticker_q = ib.reqMktData(contract, "", False, False)
    await asyncio.sleep(2)
    bid = float(ticker_q.bid or 0)
    ask = float(ticker_q.ask or 0)
    ib.cancelMktData(contract)
    if close_action == "BUY":
        lmt = round((ask + 0.01) if ask > 0 else (bid * 1.10 if bid > 0 else 0.50), 2)
    else:
        lmt = round((bid - 0.01) if bid > 0 else (ask * 0.90 if ask > 0 else 0.10), 2)
    # Capture live IV at exit for learning
    live_iv_exit = None
    try:
        if ticker_q.modelGreeks:
            live_iv_exit = round(float(ticker_q.modelGreeks.impliedVol or 0) * 100, 2)
    except Exception:
        pass

    trade = ib.placeOrder(contract, LimitOrder(close_action, qty, lmt))
    await asyncio.sleep(1)
    state["autotrader"]["positions"].pop(key, None)

    upnl_now = float(item.unrealizedPNL or 0)

    # Record exit in trade journal
    j_id = info.get("journal_id")
    if j_id:
        # Determine exit reason from context (info carries last reason from monitor)
        exit_reason = info.get("exit_reason", "manual")
        _journal_record_exit(j_id, lmt, upnl_now, exit_reason, live_iv_exit)

    # Track premium collected for LEAP budget
    if upnl_now > 0 and info.get("action") == "SELL":
        at = state["autotrader"]
        at["premium_collected"] = round(at.get("premium_collected", 0.0) + upnl_now, 2)
        at["leap_budget"]       = round(at["premium_collected"] * 0.50, 2)
        _at_log("BUDGET", f"+${upnl_now:.0f} → total=${at['premium_collected']:.0f} | LEAP=${at['leap_budget']:.0f}")
    _at_log("CLOSED", f"{close_action} {qty}x {c.symbol} @ ${lmt:.2f} "
                      f"pnl=${upnl_now:+.0f} (order #{trade.order.orderId})")
    _at_save_state()


async def _autotrader_scan_and_trade_coro(ib: IB) -> None:
    at       = state["autotrader"]
    cfg      = at["config"]

    # ── Market regime gate ────────────────────────────────────────────────────
    regime_data = await _market_regime()
    regime      = regime_data.get("regime", "UNKNOWN")
    vix         = regime_data.get("vix") or 0.0

    if regime == "BEAR" and "csp" in cfg["scan_types"]:
        _at_log("REGIME", f"BEAR market (VIX={vix:.1f}) — pausing CSP new entries to protect capital")
        return
    if vix >= 35:
        _at_log("REGIME", f"VIX={vix:.1f} (extreme fear) — pausing all new entries")
        return
    if regime == "NEUTRAL":
        _at_log("REGIME", f"NEUTRAL market (VIX={vix:.1f}) — allowing entries at reduced Kelly (0.6x)")

    # ── Position slot check ───────────────────────────────────────────────────
    # Dedup uses the positions DICT (not live portfolio) so that Inactive orders
    # (accepted by TWS but not transmitted to exchange) still prevent re-placement.
    # Using ib.portfolio() alone was the bug: Inactive orders never appear in
    # the portfolio, so every scan cycle treated all slots as empty and placed
    # 5 fresh duplicate orders.
    dict_t   = {k.split("_")[0] for k in at["positions"]}
    # Also include any portfolio positions not yet in our dict (edge case after manual trades)
    portf_t  = {item.contract.symbol for item in ib.portfolio()}
    active_t = dict_t | portf_t
    # In NEUTRAL, cap at half of max_positions to stay conservative
    max_slots = cfg["max_positions"] if regime == "BULL" else max(1, cfg["max_positions"] // 2)
    # Slot count is based on the positions dict length so Inactive orders consume slots
    slots = max_slots - len(at["positions"])
    if slots <= 0:
        _at_log("SCAN", f"Max positions reached ({len(at['positions'])}/{cfg['max_positions']}) — skipping scan")
        return
    _at_log("SCAN", f"Scanning for up to {slots} new positions (regime={regime})…")

    candidates: list = []
    if "csp" in cfg["scan_types"]:
        try:
            r = await scan_csp(ib, CSP_MIN_RETURN_PCT, CSP_MAX_DELTA)
            for c in _filter_csp_recommended(r["candidates"]):
                c["_type"] = "csp"
                c["_regime"] = regime
                candidates.append(c)
        except Exception as exc:
            _at_log("ERROR", f"CSP scan failed: {exc}")
    if "leap" in cfg["scan_types"]:
        try:
            r = await scan_leaps(ib)
            for c in _filter_leap_recommended(r["candidates"]):
                c["_type"] = "leap"
                c["_regime"] = regime
                candidates.append(c)
        except Exception as exc:
            _at_log("ERROR", f"LEAP scan failed: {exc}")
    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = [c for c in candidates if c["ticker"] not in active_t]
    _at_log("SCAN", f"Found {len(candidates)} qualifying candidates after dedup")
    placed = 0
    for row in candidates[:slots]:
        try:
            await _autotrader_place_coro(ib, row, cfg, regime=regime)
            active_t.add(row["ticker"])
            placed += 1
        except Exception as exc:
            _at_log("ERROR", f"Place {row['ticker']}: {exc}")
    _at_log("SCAN", f"Placed {placed} orders this cycle")
    # Auto-hedge if delta exposure exceeds threshold
    if cfg.get("auto_hedge", False):
        try:
            await _autotrader_hedge_coro(ib)
        except Exception as exc:
            _at_log("ERROR", f"Hedge failed: {exc}")


async def _autotrader_place_coro(ib: IB, row: dict, cfg: dict, regime: str = "BULL") -> None:
    ticker = row["ticker"]
    expiry = row["expiry"]
    strike = float(row["strike"])
    t      = row.get("_type", "csp")
    right  = "P" if t == "csp" else "C"
    action = "SELL" if t == "csp" else "BUY"
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    mid = (bid + ask) / 2 if ask > 0 else 5.0
    if cfg.get("use_kelly", True):
        qty = _kelly_qty(cfg, strike, t, mid, regime=regime)
    elif t == "csp":
        qty = max(1, int(float(cfg.get("csp_capital", 20000)) / (strike * 100)))
    else:
        qty = max(1, int(float(cfg.get("leap_capital", 5000)) / (mid * 100)))
    contract = Option(ticker, expiry, strike, right, "SMART")
    await ib.qualifyContractsAsync(contract)
    if not contract.conId:
        raise ValueError(f"Could not qualify {ticker} {expiry} {strike}{right}")
    # ── Live IBKR/OPRA price for limit order (replaces stale yfinance bid) ──
    live_iv_entry = None
    lmt           = None
    lmt_src       = "none"
    _flow_abort   = None   # set inside try; raised outside so except doesn't swallow it
    try:
        # 100=option volume, 101=option OI, 106=implied vol, 13=OI (standard)
        tq = ib.reqMktData(contract, "106,13,100,101", False, False)
        await asyncio.sleep(2.0)   # allow snapshot to settle
        ibkr_bid = tq.bid if tq.bid and not math.isnan(tq.bid) and tq.bid > 0 else None
        ibkr_ask = tq.ask if tq.ask and not math.isnan(tq.ask) and tq.ask > 0 else None
        if ibkr_bid and ibkr_ask:
            ibkr_mid = (ibkr_bid + ibkr_ask) / 2
            lmt = round(
                (ibkr_bid + (ibkr_mid - ibkr_bid) * 0.40) if action == "SELL"
                else (ibkr_ask - (ibkr_ask - ibkr_mid) * 0.40),
                2,
            )
            lmt_src = "ibkr"
        if tq.modelGreeks:
            live_iv_entry = round(float(tq.modelGreeks.impliedVol or 0) * 100, 2)

        # ── Live flow-signal gate: re-validate order-book pressure at execution ──
        # The scan snapshot can be 1-5 min old; flow can flip on news or block trades.
        # We store the abort reason rather than raising here — a ValueError raised inside
        # this try block would be caught by the except below and silently swallowed.
        live_bid_sz = int(_safe_float(getattr(tq, "bidSize", None), 0))
        live_ask_sz = int(_safe_float(getattr(tq, "askSize", None), 0))
        live_vol    = int(_safe_float(getattr(tq, "volume", None), 0))
        live_oi     = int(_safe_float(getattr(tq, "openInterest", None), 0))
        if live_bid_sz > 0 or live_ask_sz > 0:   # IBKR has live order-book data
            if   live_ask_sz > live_bid_sz * 2: live_flow = "ASK HEAVY"
            elif live_bid_sz > live_ask_sz * 2: live_flow = "BID HEAVY"
            else:                               live_flow = "BALANCED"
            live_voi = round(live_vol / max(live_oi, 1), 1) if live_oi > 0 else 0.0
            if t == "csp" and live_flow == "BID HEAVY":
                _flow_abort = (
                    f"Flow reversed to BID HEAVY at execution (was {row.get('flow_flag','?')} at scan) "
                    f"— aborting {ticker} ${strike}P to avoid selling into bearish put pressure"
                )
            elif t == "csp" and live_voi > 2.0 and live_flow != "ASK HEAVY":
                _flow_abort = (
                    f"Unusual put activity at execution (vol/OI {live_voi}x, flow={live_flow}) "
                    f"— informed buyer likely entered since scan. Aborting {ticker} ${strike}P"
                )
            elif t == "leap" and live_flow == "BID HEAVY":
                _flow_abort = (
                    f"Call order book BID HEAVY at execution — sellers dumping calls. "
                    f"Aborting {ticker} ${strike}C LEAP"
                )
            else:
                log.info(
                    "Flow gate [%s %s $%s%s]: bid_sz=%d ask_sz=%d flow=%s vol/OI=%.1f — PASS",
                    ticker, t.upper(), strike, right, live_bid_sz, live_ask_sz, live_flow, live_voi,
                )
        else:
            log.info("Flow gate [%s %s $%s%s]: no order-book data (market closed?) — skipping", ticker, t.upper(), strike, right)

        ib.cancelMktData(contract)
    except Exception as e:
        log.warning("IBKR price snapshot failed for %s %s $%s%s: %s", ticker, expiry, strike, right, e)

    # Raise flow abort AFTER the try/except so it propagates to the caller
    if _flow_abort:
        _at_log("SKIP", f"Flow gate: {_flow_abort}")
        raise ValueError(_flow_abort)

    # Fall back to yfinance if IBKR snapshot gave no data
    if not lmt or lmt <= 0:
        exp_str = f"{expiry[:4]}-{expiry[4:6]}-{expiry[6:8]}"
        def _yf_fallback():
            try:
                df = (yf.Ticker(ticker).option_chain(exp_str).calls if right == "C"
                      else yf.Ticker(ticker).option_chain(exp_str).puts)
                r  = df[df["strike"] == strike]
                if r.empty: return None
                b, a = float(r["bid"].iloc[0]), float(r["ask"].iloc[0])
                m = (b + a) / 2
                return round((b + (m-b)*0.40) if action=="SELL" else (a-(a-m)*0.40), 2)
            except Exception:
                return None
        lmt = await asyncio.get_event_loop().run_in_executor(None, _yf_fallback)
        lmt_src = "yfinance"
        log.info("Auto-trader: used yfinance fallback price for %s %s $%s%s", ticker, expiry, strike, right)

    if not lmt or lmt <= 0:
        raise ValueError(f"Could not price {ticker} {expiry} ${strike}{right}")

    trade      = ib.placeOrder(contract, LimitOrder(action, qty, lmt))
    await asyncio.sleep(2)
    order_status  = trade.orderStatus.status
    why_held      = trade.orderStatus.whyHeld or ""
    max_profit = lmt * qty * 100
    key        = _at_contract_key(contract)
    if order_status == "Inactive":
        _at_log("WARN", f"Order #{trade.order.orderId} immediately Inactive — whyHeld={why_held!r}. "
                        "Check TWS permissions or duplicate order suppression.")

    # Record entry in journal
    exp_d  = datetime.strptime(expiry[:8], "%Y%m%d").date()
    dte    = (exp_d - date.today()).days
    entry_info = {
        "ticker":             ticker,  "expiry":           expiry,
        "strike":             strike,  "right":            right,
        "action":             action,  "qty":              qty,
        "entry_price":        lmt,     "max_profit":       round(max_profit, 2),
        "iv_rank":            row.get("iv_rank"),
        "score":              row.get("score", 0),
        "liquidity_score":    row.get("liquidity_score"),
        "weekly_return_pct":  row.get("weekly_return_pct"),
        "rsi14":              row.get("rsi14"),
        "macd_hist":          row.get("macd_hist"),
        "earnings_days_out":  row.get("earnings_days_out"),
        "dte":                dte,
        "spot_price":         row.get("spot") or row.get("stock_price"),
        "market_regime":      state["ext_cache"].get("regime", {}).get("regime"),
        "live_iv":            live_iv_entry,
        "strategy_type":      row.get("_type", "csp"),
    }
    journal_id = _journal_insert_entry(entry_info)

    state["autotrader"]["positions"][key] = {
        "ticker":      ticker,    "expiry":    expiry,
        "strike":      strike,    "right":     right,
        "action":      action,    "qty":       qty,
        "entry_price": lmt,       "max_profit":round(max_profit, 2),
        "order_id":    trade.order.orderId,
        "placed_at":   datetime.utcnow().strftime("%H:%M:%S UTC"),
        "score":       row.get("score", 0),
        "journal_id":  journal_id,
        "live_iv":     live_iv_entry,
    }
    _at_log("TRADE", f"{action} {qty}x {ticker} {right}{strike} {expiry} @ ${lmt:.2f} "
                     f"[price_src={lmt_src}] (order #{trade.order.orderId}, status={order_status}, journal #{journal_id})")
    _at_save_state()


async def _autotrader_hedge_coro(ib: IB) -> None:
    """Buy a SPY protective option when net portfolio delta exceeds threshold."""
    cfg       = state["autotrader"]["config"]
    threshold = float(cfg.get("hedge_threshold", 100.0))

    # ── Portfolio delta via live IBKR Greeks (batch reqTickersAsync) ──────────
    portfolio = ib.portfolio()
    opt_contracts = [
        i.contract for i in portfolio
        if getattr(i.contract, "secType", "") in ("OPT", "FOP")
        and "FORECASTX" not in (getattr(i.contract, "exchange", ""), getattr(i.contract, "primaryExch", ""))
    ]
    live_delta: dict = {}   # conId → float
    if opt_contracts:
        try:
            await ib.qualifyContractsAsync(*opt_contracts)
            valid = [c for c in opt_contracts if c.conId]
            if valid:
                tds = await ib.reqTickersAsync(*valid)
                for td in tds:
                    if td.contract and td.modelGreeks:
                        d = _safe_float(td.modelGreeks.delta)
                        if d is not None:
                            live_delta[td.contract.conId] = d
        except Exception as e:
            log.warning("Hedge: portfolio Greeks snapshot failed (%s) — using BS fallback", e)

    net_delta = 0.0
    for item in portfolio:
        c   = item.contract
        pos = float(item.position)
        sec = getattr(c, "secType", "")
        if sec == "STK":
            net_delta += pos
        elif sec in ("OPT", "FOP"):
            try:
                cid  = c.conId
                mult = float(getattr(c, "multiplier", 100) or 100)
                if cid and cid in live_delta:
                    d = live_delta[cid]
                else:
                    # BS fallback — uses cached yfinance IV
                    K      = float(c.strike)
                    exp_d  = datetime.strptime(c.lastTradeDateOrContractMonth[:8], "%Y%m%d").date()
                    T      = max((exp_d - date.today()).days / 365, 0.001)
                    S      = await _get_stock_price(ib, c.symbol) if sec == "OPT" else (
                        float(state["bars"]["SPY"][-1]["close"]) * 10
                        if state["bars"].get("SPY") else 5500.0
                    )
                    iv_inf = state["ext_cache"]["iv_rank"].get(c.symbol, {})
                    sigma  = iv_inf.get("iv") or 0.25
                    d      = _bs_delta(S, K, T, sigma, is_put=(c.right.upper() == "P"))
                net_delta += (d or 0) * pos * mult
            except Exception:
                pass

    _at_log("HEDGE", f"Net portfolio delta = {net_delta:+.1f}  (threshold ±{threshold:.0f},"
            f" IBKR Greeks for {len(live_delta)}/{len(opt_contracts)} positions)")
    if abs(net_delta) < threshold:
        return

    hedge_right  = "P" if net_delta > 0 else "C"
    hedge_action = "BUY"

    # ── SPY hedge contract pricing via IBKR/OPRA ──────────────────────────────
    spy_spot = await _get_stock_price(ib, "SPY")
    if spy_spot <= 0:
        _at_log("HEDGE", "Could not get live SPY price")
        return

    strike, lmt, expiry_k = None, None, None
    try:
        tgt_k = spy_spot * (0.98 if hedge_right == "P" else 1.02)
        tds, chosen_exp, _ = await _fetch_opra_chain(
            ib, "SPY", hedge_right, spy_spot,
            dte_min=3, dte_max=14,
            otm_lo_pct=0.0, otm_hi_pct=4.0, max_strikes=6,
        )
        best_td, best_dist = None, float("inf")
        for td in tds:
            K2  = _safe_float(td.contract.strike)
            bid = _safe_float(td.bid, 0.0)
            ask = _safe_float(td.ask, 0.0)
            if K2 is None or ask <= 0:
                continue
            dist = abs(K2 - tgt_k)
            if dist < best_dist:
                best_dist, best_td = dist, td
        if best_td:
            strike   = best_td.contract.strike
            expiry_k = chosen_exp
            ask_live = _safe_float(best_td.ask, 0.0)
            bid_live = _safe_float(best_td.bid, 0.0)
            mid_live = (bid_live + ask_live) / 2 if ask_live > 0 else 0
            # Pay slightly above ask to ensure fill (hedge is protective, not premium-seeking)
            lmt = round(ask_live + (mid_live - ask_live) * 0.20, 2) if ask_live > 0 else None
    except ValueError as e:
        log.warning("Hedge: OPRA chain unavailable (%s) — trying yfinance", e)

    # ── yfinance fallback for hedge pricing ───────────────────────────────────
    if not lmt or lmt <= 0:
        def _yf_hedge():
            spy_tk = yf.Ticker("SPY")
            expiries = spy_tk.options or []
            exp_str = next(
                (e for e in sorted(expiries)
                 if (datetime.strptime(e, "%Y-%m-%d").date() - date.today()).days >= 3),
                None,
            )
            if not exp_str:
                return None, None, None
            df    = spy_tk.option_chain(exp_str).puts if hedge_right == "P" else spy_tk.option_chain(exp_str).calls
            tgt_k = spy_spot * (0.98 if hedge_right == "P" else 1.02)
            df = df.copy(); df["_d"] = abs(df["strike"] - tgt_k)
            row = df.sort_values("_d").iloc[0]
            ask_yf = float(row.get("ask", 0) or 0)
            if ask_yf <= 0:
                return None, None, None
            return float(row["strike"]), round(ask_yf * 1.01, 2), exp_str.replace("-", "")
        strike, lmt, expiry_k = await asyncio.get_event_loop().run_in_executor(None, _yf_hedge)

    if not lmt or lmt <= 0 or not strike or not expiry_k:
        _at_log("HEDGE", "No valid SPY quote for hedge (IBKR + yfinance both failed)")
        return

    contract = Option("SPY", expiry_k, float(strike), hedge_right, "SMART")
    await ib.qualifyContractsAsync(contract)
    if not contract.conId:
        _at_log("HEDGE", "Could not qualify SPY hedge contract")
        return

    trade = ib.placeOrder(contract, LimitOrder(hedge_action, 1, lmt))
    await asyncio.sleep(1)
    _at_log("HEDGE", (f"Placed SPY {hedge_right}{strike} {expiry_k} hedge @ ${lmt:.2f} "
                      f"(portfolio delta={net_delta:+.0f}, order #{trade.order.orderId})"))


async def _autotrader_background() -> None:
    await asyncio.sleep(15)          # let server finish starting
    while True:
        await asyncio.sleep(300)     # 5-minute cadence
        if not state["autotrader"]["enabled"]:
            continue
        if not state.get("connected") or not state.get("ib"):
            _at_log("WARN", "Not connected — skipping cycle")
            continue
        ib   = state["ib"]
        loop = asyncio.get_event_loop()
        market_open = _is_market_open()
        try:
            # Monitor runs always — catches trailing stops / profit targets on any open positions
            await loop.run_in_executor(
                None,
                lambda: _run_in_streaming_loop(_autotrader_monitor_coro(ib), timeout=30),
            )
            # New entries only during market hours — prevents queued overnight orders
            if market_open:
                await loop.run_in_executor(
                    None,
                    lambda: _run_in_streaming_loop(_autotrader_scan_and_trade_coro(ib), timeout=180),
                )
            else:
                from zoneinfo import ZoneInfo
                now_et = datetime.now(ZoneInfo("America/New_York"))
                _at_log("MARKET", f"Market closed ({now_et.strftime('%a %H:%M ET')}) — monitoring only, no new entries")
            state["autotrader"]["last_run"] = datetime.utcnow().isoformat() + "Z"
        except Exception as exc:
            _at_log("ERROR", f"Cycle error: {exc}")
            log.error("AutoTrader cycle error: %s", exc, exc_info=True)


# ── Trade Journal (SQLite) ─────────────────────────────────────────────────

_JOURNAL_FEATURE_COLS = [
    "iv_rank", "score", "liquidity_score", "weekly_return_pct",
    "rsi14", "macd_hist", "earnings_days", "dte",
]


def _journal_init() -> None:
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS trade_journal (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            opened_at         TEXT,
            closed_at         TEXT,
            ticker            TEXT,
            expiry            TEXT,
            strike            REAL,
            right             TEXT,
            action            TEXT,
            qty               INTEGER,
            entry_price       REAL,
            exit_price        REAL,
            iv_rank           REAL,
            score             INTEGER,
            liquidity_score   REAL,
            weekly_return_pct REAL,
            rsi14             REAL,
            macd_hist         REAL,
            earnings_days     INTEGER,
            dte               INTEGER,
            spot_price        REAL,
            market_regime     TEXT,
            live_iv_entry     REAL,
            exit_reason       TEXT,
            pnl               REAL,
            pnl_pct           REAL,
            win               INTEGER,
            max_profit        REAL,
            strategy_type     TEXT,
            model_version     INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_log (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at   TEXT,
            n_trades     INTEGER,
            win_rate     REAL,
            cv_accuracy  REAL,
            importances  TEXT,
            notes        TEXT
        )
    """)
    con.commit()
    con.close()
    log.info("Trade journal initialised at %s", JOURNAL_DB_PATH)


def _journal_insert_entry(info: dict) -> int:
    """Record a new auto-trader entry. Returns the DB row id."""
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    cur = con.execute("""
        INSERT INTO trade_journal
            (opened_at, ticker, expiry, strike, right, action, qty, entry_price,
             iv_rank, score, liquidity_score, weekly_return_pct, rsi14, macd_hist,
             earnings_days, dte, spot_price, market_regime, live_iv_entry,
             max_profit, strategy_type, model_version)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat(),
        info.get("ticker"), info.get("expiry"),
        info.get("strike"), info.get("right"),
        info.get("action"), info.get("qty"),
        info.get("entry_price"),
        info.get("iv_rank"),     info.get("score"),
        info.get("liquidity_score"), info.get("weekly_return_pct"),
        info.get("rsi14"),       info.get("macd_hist"),
        info.get("earnings_days_out"), info.get("dte"),
        info.get("spot_price"),  info.get("market_regime"),
        info.get("live_iv"),     info.get("max_profit"),
        info.get("strategy_type", "csp"), state.get("model_version", 0),
    ))
    row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id


def _journal_record_exit(
    journal_id: int,
    exit_price: float,
    pnl: float,
    exit_reason: str,
    live_iv_exit: Optional[float] = None,
) -> None:
    """Close out a journal row with exit data and trigger learning."""
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    row = con.execute(
        "SELECT max_profit FROM trade_journal WHERE id=?", (journal_id,)
    ).fetchone()
    max_profit = row[0] if row and row[0] else None
    pnl_pct    = round(pnl / max_profit * 100, 1) if max_profit else None
    win        = 1 if pnl > 0 else 0
    con.execute("""
        UPDATE trade_journal
        SET closed_at=?, exit_price=?, pnl=?, pnl_pct=?, win=?, exit_reason=?
        WHERE id=?
    """, (datetime.utcnow().isoformat(), exit_price, round(pnl, 2),
          pnl_pct, win, exit_reason, journal_id))
    con.commit()
    con.close()

    # Update Kelly from real fill data
    _update_kelly_from_journal()

    # Trigger periodic model retraining
    state["trades_since_retrain"] = state.get("trades_since_retrain", 0) + 1
    if state["trades_since_retrain"] >= RETRAIN_EVERY:
        state["trades_since_retrain"] = 0
        try:
            _retrain_from_journal()
        except Exception as exc:
            log.warning("Auto retrain failed: %s", exc)


def _update_kelly_from_journal() -> None:
    """EMA-blend actual win rate from journal into auto-trader's assumed_win_rate."""
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    rows = con.execute(
        "SELECT win FROM trade_journal WHERE closed_at IS NOT NULL AND win IS NOT NULL"
    ).fetchall()
    con.close()
    if len(rows) < 10:
        return
    actual_wr = sum(r[0] for r in rows) / len(rows)
    old       = state["autotrader"]["config"].get("assumed_win_rate", 0.85)
    new_wr    = round(0.70 * old + 0.30 * actual_wr, 4)  # slow EMA
    state["autotrader"]["config"]["assumed_win_rate"] = new_wr
    _at_log("LEARN", f"Kelly win rate {old:.1%} → {new_wr:.1%} (actual {actual_wr:.1%} over {len(rows)} trades)")


def _retrain_from_journal() -> dict:
    """Retrain XGBoost score model from completed journal trades."""
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    cols_sql = ", ".join(_JOURNAL_FEATURE_COLS) + ", win"
    rows = con.execute(
        f"SELECT {cols_sql} FROM trade_journal WHERE closed_at IS NOT NULL AND win IS NOT NULL"
    ).fetchall()
    con.close()
    if len(rows) < JOURNAL_MIN_TRADES:
        return {"error": f"Need {JOURNAL_MIN_TRADES}+ trades (have {len(rows)})"}

    df  = pd.DataFrame(rows, columns=_JOURNAL_FEATURE_COLS + ["win"])
    X   = df[_JOURNAL_FEATURE_COLS].fillna(df[_JOURNAL_FEATURE_COLS].median())
    y   = df["win"].astype(int)

    from sklearn.model_selection import cross_val_score
    model = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42,
    )
    cv = cross_val_score(model, X, y, cv=min(5, max(2, len(df) // 4)),
                         scoring="accuracy")
    model.fit(X, y)

    importances = {c: round(float(v), 4)
                   for c, v in zip(_JOURNAL_FEATURE_COLS, model.feature_importances_)}
    state["model_learned"]  = model
    state["model_version"]  = state.get("model_version", 0) + 1

    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    con.execute("""
        INSERT INTO model_log (trained_at, n_trades, win_rate, cv_accuracy, importances, notes)
        VALUES (?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat(), len(df),
        round(float(y.mean()), 4), round(float(cv.mean()), 4),
        json.dumps(importances),
        f"auto v{state['model_version']}",
    ))
    con.commit()
    con.close()

    top = max(importances, key=importances.get)
    _at_log("LEARN", (f"Model v{state['model_version']} trained: {len(df)} trades, "
                       f"win={y.mean():.1%}, CV={cv.mean():.1%}, top={top}"))
    return {"version": state["model_version"], "n_trades": len(df),
            "win_rate": round(float(y.mean()) * 100, 1),
            "cv_accuracy": round(float(cv.mean()) * 100, 1),
            "importances": importances}


def _learned_score(features: dict) -> Optional[float]:
    """Score a candidate with the learned model (0–100). None if model not ready."""
    model = state.get("model_learned")
    if model is None:
        return None
    X = [[features.get(col) or 0.0 for col in _JOURNAL_FEATURE_COLS]]
    try:
        prob = float(model.predict_proba(X)[0][1])
        return round(prob * 100, 1)
    except Exception:
        return None


async def _get_stock_price(ib: IB, ticker: str) -> float:
    """Live price: prefer streaming bar cache (always fresh), else IBKR snapshot."""
    # Streaming bars are the freshest source — use them if available
    bars = state["bars"].get(ticker)
    if bars:
        return float(bars[-1]["close"])

    # Fallback: cached signal close (may be up to one bar old)
    sig = state["signals"].get(ticker)
    if sig:
        return float(sig["close"])

    # Last resort: yfinance (avoids IBKR equity subscription requirement)
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1d", interval="1m")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


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
            earnings_days = await _earnings_days_out(ticker)
            if earnings_days is not None and earnings_days <= EARNINGS_BLOCK_DAYS:
                log.info(f"CSP [{ticker}]: blocked — earnings in {earnings_days}d")
                return []

            try:
                stock_price, iv_info, tech = await asyncio.gather(
                    _get_stock_price(ib, ticker),
                    _iv_rank_for_ticker(ticker),
                    _tech_indicators(ticker),
                )
                if stock_price <= 0:
                    log.debug(f"CSP [{ticker}]: no price")
                    return []

                # ── Fetch live OPRA data + institutional signals concurrently ──
                sq = _stock_quality_score(ticker)
                tds, expiry_ibkr, dte, inst = None, None, 0, {}
                opra_ok = False
                try:
                    (tds, expiry_ibkr, dte), inst = await asyncio.gather(
                        _fetch_opra_chain(ib, ticker, "P", stock_price, 5, 35),
                        _institutional_signals(ticker, stock_price),
                    )
                    opra_ok = True
                except ValueError as e:
                    log.warning("CSP [%s]: OPRA chain unavailable (%s) — will try yfinance", ticker, e)
                    inst = await _institutional_signals(ticker, stock_price)

                data_src   = ("OPRA-LIVE" if state.get("opra_active") else "OPRA-DELAYED") if opra_ok else "yfinance"
                T          = dte / 365.0
                today_d    = date.today()
                max_pain   = inst.get("max_pain")
                gamma_wall = inst.get("gamma_wall")
                pc_oi_r    = inst.get("pc_oi_ratio", 1.0)
                pc_vol_r   = inst.get("pc_vol_ratio", 1.0)
                rows: list = []

                for td in (tds or []):
                    try:
                        K   = _safe_float(td.contract.strike)
                        bid = _safe_float(td.bid, 0.0)
                        ask = _safe_float(td.ask, 0.0)
                        if K is None or bid <= 0 or ask <= 0:
                            continue

                        greeks = td.modelGreeks
                        if not greeks:
                            continue
                        iv    = _safe_float(greeks.impliedVol)
                        delta = _safe_float(greeks.delta)
                        theta = _safe_float(greeks.theta, 0.0)
                        if iv is None or delta is None:
                            continue
                        if iv < 0.05 or iv > 3.0 or delta >= 0:
                            continue
                        if abs(delta) > max_delta:
                            continue

                        mid        = (bid + ask) / 2.0
                        spread_pct = (ask - bid) / mid * 100
                        otm_pct    = (stock_price - K) / stock_price * 100
                        if otm_pct < 3 or otm_pct > 30:
                            continue

                        oi  = int(_safe_float(td.openInterest, 0))
                        vol = int(_safe_float(td.volume, 0))
                        if oi < CSP_MIN_OI:
                            continue
                        if spread_pct / 100 > CSP_MAX_SPREAD_PCT:
                            continue

                        # OPRA order-book pressure
                        bid_sz  = int(_safe_float(td.bidSize, 0))
                        ask_sz  = int(_safe_float(td.askSize, 0))
                        if   ask_sz > bid_sz * 2: flow_flag = "ASK HEAVY"   # buyers lifting ask → bullish
                        elif bid_sz > ask_sz * 2: flow_flag = "BID HEAVY"   # sellers hitting bid → bearish
                        else:                     flow_flag = "BALANCED"

                        vol_oi_ratio = round(vol / max(oi, 1), 2)

                        fill_cons  = bid
                        fill_real  = bid + (mid - bid) * 0.40
                        wk_ret_bid = fill_cons / K * 100
                        wk_ret_pct = fill_real / K * 100
                        wk_ret_mid = mid       / K * 100
                        if wk_ret_pct < min_return:
                            continue

                        ann_ret  = wk_ret_pct * 52
                        max_loss = round((K - fill_real) * 100, 2)
                        ror      = round(fill_real / max(K - fill_real, 0.01) * 52 * 100, 1)
                        liq      = _liquidity_score(oi, vol, spread_pct)
                        exp_move = _expected_weekly_move(stock_price, iv)
                        sigma_otm= (stock_price - K) / exp_move if exp_move > 0 else 0
                        warnings = _build_warnings(earnings_days, iv_info["rank"], "csp")
                        if pc_vol_r > 1.5:
                            warnings.append(f"Heavy put flow (P/C vol {pc_vol_r:.1f}x) — bearish sentiment")
                        if vol_oi_ratio > 2.0:
                            warnings.append(f"Unusual put activity at ${K:.0f} (vol/OI {vol_oi_ratio:.1f}x)")
                        row = {
                            "ticker":                ticker,
                            "expiry":                expiry_ibkr,
                            "dte":                   dte,
                            "strike":                K,
                            "stock_price":           round(stock_price, 2),
                            "otm_pct":               round(otm_pct, 2),
                            "sigma_otm":             round(sigma_otm, 2),
                            "bid":                   round(bid, 2),
                            "ask":                   round(ask, 2),
                            "mid":                   round(mid, 2),
                            "bid_size":              bid_sz,
                            "ask_size":              ask_sz,
                            "flow_flag":             flow_flag,
                            "weekly_return_bid":     round(wk_ret_bid, 2),
                            "weekly_return_pct":     round(wk_ret_pct, 2),
                            "weekly_return_mid":     round(wk_ret_mid, 2),
                            "annualized_return":     round(ann_ret, 1),
                            "max_loss_per_contract": max_loss,
                            "return_on_risk_ann":    ror,
                            "breakeven":             round(K - fill_real, 2),
                            "delta":                 round(delta, 4),
                            "iv_pct":                round(iv * 100, 1),
                            "theta_daily":           round(theta, 4),
                            "open_interest":         oi,
                            "volume":                vol,
                            "vol_oi_ratio":          vol_oi_ratio,
                            "spread_pct":            round(spread_pct, 2),
                            "liquidity_score":       liq,
                            "stock_quality":         sq,
                            "assignment_risk":       _assignment_risk(delta, otm_pct),
                            "xgb_signal":            state["signals"].get(ticker, {}).get("label", "N/A"),
                            "cash_required":         round(K * 100, 2),
                            "premium_collected":     round(fill_real * 100, 2),
                            "suggested_price":       round(fill_real, 2),
                            "earnings_days_out":     earnings_days,
                            "iv_rank":               iv_info["rank"],
                            "iv_yf":                 iv_info["iv"],
                            "rsi14":                 tech.get("rsi14"),
                            "macd_hist":             tech.get("macd_hist"),
                            "above_sma20":           tech.get("above_sma20"),
                            "above_sma50":           tech.get("above_sma50"),
                            # Institutional / OPRA signals
                            "max_pain":              max_pain,
                            "gamma_wall":            gamma_wall,
                            "above_max_pain":        bool(max_pain and K >= max_pain),
                            "pc_oi_ratio":           pc_oi_r,
                            "pc_vol_ratio":          pc_vol_r,
                            "data_source":           data_src,
                            "warnings":              warnings,
                        }
                        row["score"] = _score_csp(row)
                        # Sanitize any NaN/Inf floats before JSON serialization
                        row = {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v) for k, v in row.items()}
                        rows.append(row)
                    except Exception as exc:
                        log.debug("CSP row error %s $%s: %s", ticker, td.contract.strike if td.contract else "?", exc)
                        continue

                log.info("CSP [%s]: %d %s candidates  price=$%.2f  max_pain=$%s  P/C_vol=%.2f",
                         ticker, len(rows), data_src, stock_price,
                         f"{max_pain:.0f}" if max_pain else "n/a", pc_vol_r)

                # ── yfinance fallback: market closed or IBKR returned no bids ──
                if not rows:
                    log.info("CSP [%s]: no IBKR bids — falling back to yfinance", ticker)
                    data_src = "yfinance"
                    def _yf_puts():
                        t = yf.Ticker(ticker)
                        out = []
                        for exp_str in (t.options or []):
                            try:
                                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                            except ValueError:
                                continue
                            d = (exp_date - today_d).days
                            if d < 5:   continue
                            if d > 35:  break
                            T2 = d / 365.0
                            efmt = exp_date.strftime("%Y%m%d")
                            try:
                                puts = t.option_chain(exp_str).puts
                            except Exception:
                                continue
                            for _, r2 in puts.iterrows():
                                try:
                                    K2  = float(r2["strike"])
                                    b2  = float(r2.get("bid") or 0)
                                    a2  = float(r2.get("ask") or 0)
                                    iv2 = float(r2.get("impliedVolatility") or 0)
                                    oi2 = int(r2.get("openInterest") or 0)
                                    vl2 = int(r2.get("volume") or 0)
                                    if b2<=0 or a2<=0 or iv2<0.05 or iv2>3.0: continue
                                    mid2  = (b2+a2)/2; sp2=(a2-b2)/mid2*100
                                    otp   = (stock_price-K2)/stock_price*100
                                    if otp<3 or otp>30: continue
                                    dlt2  = _bs_delta(stock_price,K2,T2,iv2,is_put=True)
                                    if math.isnan(dlt2) or dlt2>=0 or abs(dlt2)>max_delta: continue
                                    if oi2<CSP_MIN_OI or sp2/100>CSP_MAX_SPREAD_PCT: continue
                                    tht2  = _bs_theta(stock_price,K2,T2,iv2,is_put=True)
                                    fr2   = b2+(mid2-b2)*0.40
                                    wret  = fr2/K2*100
                                    if wret<min_return: continue
                                    liq2  = _liquidity_score(oi2,vl2,sp2)
                                    em2   = _expected_weekly_move(stock_price,iv2)
                                    sig2  = (stock_price-K2)/em2 if em2>0 else 0
                                    warn2 = _build_warnings(earnings_days, iv_info["rank"], "csp")
                                    row2  = {
                                        "ticker":ticker, "expiry":efmt, "dte":d,
                                        "strike":K2, "stock_price":round(stock_price,2),
                                        "otm_pct":round(otp,2), "sigma_otm":round(sig2,2),
                                        "bid":round(b2,2), "ask":round(a2,2), "mid":round(mid2,2),
                                        "bid_size":0, "ask_size":0, "flow_flag":"N/A",
                                        "weekly_return_bid":round(b2/K2*100,2),
                                        "weekly_return_pct":round(wret,2),
                                        "weekly_return_mid":round(mid2/K2*100,2),
                                        "annualized_return":round(wret*52,1),
                                        "max_loss_per_contract":round((K2-fr2)*100,2),
                                        "return_on_risk_ann":round(fr2/max(K2-fr2,0.01)*52*100,1),
                                        "breakeven":round(K2-fr2,2),
                                        "delta":round(dlt2,4), "iv_pct":round(iv2*100,1),
                                        "theta_daily":round(tht2,4),
                                        "open_interest":oi2, "volume":vl2,
                                        "vol_oi_ratio":round(vl2/max(oi2,1),2),
                                        "spread_pct":round(sp2,2), "liquidity_score":liq2,
                                        "stock_quality":sq,
                                        "assignment_risk":_assignment_risk(dlt2,otp),
                                        "xgb_signal":state["signals"].get(ticker,{}).get("label","N/A"),
                                        "cash_required":round(K2*100,2),
                                        "premium_collected":round(fr2*100,2),
                                        "suggested_price":round(fr2,2),
                                        "earnings_days_out":earnings_days,
                                        "iv_rank":iv_info["rank"], "iv_yf":iv_info["iv"],
                                        "rsi14":tech.get("rsi14"), "macd_hist":tech.get("macd_hist"),
                                        "above_sma20":tech.get("above_sma20"),
                                        "above_sma50":tech.get("above_sma50"),
                                        "max_pain":max_pain, "gamma_wall":inst.get("gamma_wall"),
                                        "above_max_pain":bool(max_pain and K2>=max_pain),
                                        "pc_oi_ratio":inst.get("pc_oi_ratio",1.0),
                                        "pc_vol_ratio":inst.get("pc_vol_ratio",1.0),
                                        "data_source":"yfinance",
                                        "warnings":warn2,
                                    }
                                    row2["score"] = _score_csp(row2)
                                    out.append(row2)
                                except Exception: continue
                            if out: break
                        return out
                    rows = await asyncio.get_event_loop().run_in_executor(None, _yf_puts)
                    log.info("CSP [%s]: %d yfinance fallback candidates", ticker, len(rows))
                return rows

            except BaseException as e:
                log.warning("CSP scan error [%s]: %s\n%s", ticker, e, traceback.format_exc())
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
            earnings_days = await _earnings_days_out(ticker)
            if earnings_days is not None and earnings_days <= EARNINGS_BLOCK_DAYS:
                log.info(f"LEAP [{ticker}]: blocked — earnings in {earnings_days}d")
                return []

            try:
                sig = state["signals"].get(ticker, {})
                if sig.get("label") == "SELL":
                    return []
                sq = _stock_quality_score(ticker)
                if sq < 0.3:
                    return []

                stock_price, iv_info, tech = await asyncio.gather(
                    _get_stock_price(ib, ticker),
                    _iv_rank_for_ticker(ticker),
                    _tech_indicators(ticker),
                )
                if stock_price <= 0:
                    return []

                # ── Fetch live OPRA LEAP calls + institutional signals ──
                tds, expiry_ibkr, dte, inst = None, None, 0, {}
                opra_ok_leap = False
                try:
                    (tds, expiry_ibkr, dte), inst = await asyncio.gather(
                        _fetch_opra_chain(
                            ib, ticker, "C", stock_price,
                            LEAP_MIN_DTE, LEAP_MAX_DTE,
                            otm_lo_pct=0, otm_hi_pct=20, max_strikes=10,
                        ),
                        _institutional_signals(ticker, stock_price),
                    )
                    opra_ok_leap = True
                except ValueError as e:
                    log.warning("LEAP [%s]: OPRA chain unavailable (%s) — will try yfinance", ticker, e)
                    inst = await _institutional_signals(ticker, stock_price)

                leap_data_src = ("OPRA-LIVE" if state.get("opra_active") else "OPRA-DELAYED") if opra_ok_leap else "yfinance"
                T          = dte / 365.0
                pc_vol_r   = inst.get("pc_vol_ratio", 1.0)
                pc_oi_r    = inst.get("pc_oi_ratio", 1.0)
                rows: list = []

                for td in (tds or []):
                    try:
                        K   = _safe_float(td.contract.strike)
                        bid = _safe_float(td.bid, 0.0)
                        ask = _safe_float(td.ask, 0.0)
                        if K is None or bid <= 0 or ask <= 0:
                            continue

                        greeks = td.modelGreeks
                        if not greeks:
                            continue
                        iv    = _safe_float(greeks.impliedVol)
                        delta = _safe_float(greeks.delta)
                        theta = _safe_float(greeks.theta, 0.0)
                        vega  = _safe_float(greeks.vega, 0.0)
                        if iv is None or delta is None:
                            continue
                        if iv < 0.05 or iv > 3.0 or delta <= 0:
                            continue
                        if iv > LEAP_MAX_IV:
                            continue
                        if not (LEAP_MIN_DELTA <= delta <= LEAP_MAX_DELTA):
                            continue

                        mid        = (bid + ask) / 2.0
                        spread_pct = (ask - bid) / mid * 100
                        oi  = int(_safe_float(td.openInterest, 0))
                        vol = int(_safe_float(td.volume, 0))

                        # OPRA order-book pressure on LEAP calls
                        bid_sz  = int(_safe_float(td.bidSize, 0))
                        ask_sz  = int(_safe_float(td.askSize, 0))
                        if   ask_sz > bid_sz * 2: flow_flag = "ASK HEAVY"   # aggressive call buyers
                        elif bid_sz > ask_sz * 2: flow_flag = "BID HEAVY"
                        else:                     flow_flag = "BALANCED"

                        vol_oi_ratio = round(vol / max(oi, 1), 2)

                        fill_cons  = ask
                        fill_real  = ask - (ask - mid) * 0.40
                        cost_cons  = round(fill_cons * 100, 2)
                        cost_real  = round(fill_real * 100, 2)
                        cost_mid_v = round(mid       * 100, 2)
                        breakeven  = K + fill_real
                        be_move    = (breakeven - stock_price) / stock_price * 100
                        liq        = _liquidity_score(oi, vol, spread_pct)
                        itm_otm_pct= (stock_price - K) / stock_price * 100

                        warnings = _build_warnings(earnings_days, iv_info["rank"], "leap")
                        # Institutional call OI >> put OI = bullish positioning → favourable
                        if pc_oi_r < 0.6:
                            pass  # heavy call OI: good for LEAP calls
                        elif pc_oi_r > 1.5:
                            warnings.append(f"Heavy put OI (P/C {pc_oi_r:.1f}x) — market bearishly positioned")
                        if vol_oi_ratio > 2.0 and flow_flag == "ASK HEAVY":
                            warnings.append(f"Unusual call buying at ${K:.0f} (vol/OI {vol_oi_ratio:.1f}x) — institutional interest")

                        leap_iv_bonus = (1 - iv_info["rank"] / 100) * 15
                        # OPRA institutional bonus for LEAP calls
                        inst_bonus = 0.0
                        if flow_flag == "ASK HEAVY":  inst_bonus += 5   # buyers are aggressive
                        if vol_oi_ratio > 2.0:        inst_bonus += 8   # unusual call activity
                        if pc_oi_r < 0.7:             inst_bonus += 5   # call-OI dominant = institutions bullish

                        row = {
                            "ticker":                ticker,
                            "expiry":                expiry_ibkr,
                            "dte":                   dte,
                            "strike":                K,
                            "stock_price":           round(stock_price, 2),
                            "itm_otm_pct":           round(itm_otm_pct, 2),
                            "breakeven":             round(breakeven, 2),
                            "breakeven_move_pct":    round(be_move, 2),
                            "bid":                   round(bid, 2),
                            "ask":                   round(ask, 2),
                            "mid":                   round(mid, 2),
                            "bid_size":              bid_sz,
                            "ask_size":              ask_sz,
                            "flow_flag":             flow_flag,
                            "cost_conservative":     cost_cons,
                            "cost_per_contract":     cost_real,
                            "cost_mid":              cost_mid_v,
                            "max_loss_per_contract": cost_real,
                            "suggested_price":       round(fill_real, 2),
                            "delta":                 round(delta, 4),
                            "iv_pct":                round(iv * 100, 1),
                            "theta_daily":           round(theta, 4),
                            "theta_weekly":          round(theta * 7, 4),
                            "vega":                  round(vega, 4),
                            "spread_pct":            round(spread_pct, 2),
                            "open_interest":         oi,
                            "volume":                vol,
                            "vol_oi_ratio":          vol_oi_ratio,
                            "liquidity_score":       liq,
                            "stock_quality":         sq,
                            "xgb_signal":            sig.get("label", "N/A"),
                            "xgb_prob":              sig.get("prob", None),
                            "earnings_days_out":     earnings_days,
                            "iv_rank":               iv_info["rank"],
                            "iv_yf":                 iv_info["iv"],
                            "rsi14":                 tech.get("rsi14"),
                            "macd_hist":             tech.get("macd_hist"),
                            "above_sma20":           tech.get("above_sma20"),
                            "above_sma50":           tech.get("above_sma50"),
                            "pc_oi_ratio":           pc_oi_r,
                            "pc_vol_ratio":          pc_vol_r,
                            "data_source":           leap_data_src,
                            "warnings":              warnings,
                        }
                        row["score"] = round(
                            sq * 55
                            + (1 - abs(delta - 0.60)) * 35
                            + leap_iv_bonus
                            + (liq / 100) * 10
                            + inst_bonus,
                            2,
                        )
                        row = {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v) for k, v in row.items()}
                        rows.append(row)
                    except Exception as exc:
                        log.debug("LEAP row error %s: %s", ticker, exc)
                        continue

                log.info("LEAP [%s]: %d %s candidates  price=$%.2f  P/C_vol=%.2f",
                         ticker, len(rows), leap_data_src, stock_price, pc_vol_r)

                # ── yfinance fallback when IBKR has no bids (market closed) ──
                if not rows:
                    log.info("LEAP [%s]: no IBKR bids — falling back to yfinance", ticker)
                    def _yf_calls():
                        t2 = yf.Ticker(ticker)
                        cands = []
                        for exp_str in (t2.options or []):
                            try:
                                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                            except ValueError:
                                continue
                            d = (exp_date - today).days
                            if LEAP_MIN_DTE <= d <= LEAP_MAX_DTE:
                                cands.append((abs(d-365), d, exp_str, exp_date))
                        if not cands: return []
                        cands.sort()
                        _, d2, exp_str2, exp_date2 = cands[0]
                        T2 = d2 / 365.0
                        efmt2 = exp_date2.strftime("%Y%m%d")
                        out = []
                        try:
                            calls = t2.option_chain(exp_str2).calls
                        except Exception:
                            return []
                        for _, rc in calls.iterrows():
                            try:
                                K2  = float(rc["strike"])
                                b2  = float(rc.get("bid") or 0)
                                a2  = float(rc.get("ask") or 0)
                                iv2 = float(rc.get("impliedVolatility") or 0)
                                oi2 = int(rc.get("openInterest") or 0)
                                vl2 = int(rc.get("volume") or 0)
                                if b2<=0 or a2<=0 or iv2<0.05 or iv2>LEAP_MAX_IV: continue
                                mid2=(b2+a2)/2; sp2=(a2-b2)/mid2*100
                                dlt2=_bs_delta(stock_price,K2,T2,iv2,is_put=False)
                                if math.isnan(dlt2) or not (LEAP_MIN_DELTA<=dlt2<=LEAP_MAX_DELTA): continue
                                tht2=_bs_theta(stock_price,K2,T2,iv2,is_put=False)
                                fr2=a2-(a2-mid2)*0.40
                                liq2=_liquidity_score(oi2,vl2,sp2)
                                be2=K2+fr2; bem=(be2-stock_price)/stock_price*100
                                itm2=(stock_price-K2)/stock_price*100
                                warn2=_build_warnings(earnings_days,iv_info["rank"],"leap")
                                ib2=(1-iv_info["rank"]/100)*15
                                rw={
                                    "ticker":ticker,"expiry":efmt2,"dte":d2,
                                    "strike":K2,"stock_price":round(stock_price,2),
                                    "itm_otm_pct":round(itm2,2),"breakeven":round(be2,2),
                                    "breakeven_move_pct":round(bem,2),
                                    "bid":round(b2,2),"ask":round(a2,2),"mid":round(mid2,2),
                                    "bid_size":0,"ask_size":0,"flow_flag":"N/A",
                                    "cost_conservative":round(a2*100,2),
                                    "cost_per_contract":round(fr2*100,2),
                                    "cost_mid":round(mid2*100,2),
                                    "max_loss_per_contract":round(fr2*100,2),
                                    "suggested_price":round(fr2,2),
                                    "delta":round(dlt2,4),"iv_pct":round(iv2*100,1),
                                    "theta_daily":round(tht2,4),"theta_weekly":round(tht2*7,4),
                                    "vega":0.0,"spread_pct":round(sp2,2),
                                    "open_interest":oi2,"volume":vl2,
                                    "vol_oi_ratio":round(vl2/max(oi2,1),2),
                                    "liquidity_score":liq2,"stock_quality":sq,
                                    "xgb_signal":sig.get("label","N/A"),"xgb_prob":sig.get("prob"),
                                    "earnings_days_out":earnings_days,
                                    "iv_rank":iv_info["rank"],"iv_yf":iv_info["iv"],
                                    "rsi14":tech.get("rsi14"),"macd_hist":tech.get("macd_hist"),
                                    "above_sma20":tech.get("above_sma20"),"above_sma50":tech.get("above_sma50"),
                                    "pc_oi_ratio":inst.get("pc_oi_ratio",1.0),
                                    "pc_vol_ratio":inst.get("pc_vol_ratio",1.0),
                                    "data_source":"yfinance","warnings":warn2,
                                }
                                rw["score"]=round(sq*55+(1-abs(dlt2-0.60))*35+ib2+(liq2/100)*10,2)
                                out.append(rw)
                            except Exception: continue
                        return out
                    rows = await asyncio.get_event_loop().run_in_executor(None, _yf_calls)
                    log.info("LEAP [%s]: %d yfinance fallback candidates", ticker, len(rows))
                return rows

            except BaseException as e:
                log.warning("LEAP scan error [%s]: %s\n%s", ticker, e, traceback.format_exc())
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


# ── 0DTE / weekly scanner ──────────────────────────────────────────────────
ZERO_DTE_UNIVERSE = ["SPY", "QQQ", "IWM"]

async def scan_0dte(ib: IB) -> dict:
    """
    Scan SPY/QQQ/IWM for 0–7 DTE put premium-collection setups.
    IBKR/OPRA is the primary source (time-critical for 0DTE fills);
    yfinance is the fallback when OPRA chain is unavailable.
    """
    today      = date.today()
    candidates = []

    for ticker in ZERO_DTE_UNIVERSE:
        try:
            spot = await _get_stock_price(ib, ticker)
            if spot <= 0:
                continue

            # ── Primary: IBKR/OPRA snapshot (real-time bid/ask + Greeks) ──
            tds, chosen_exp, dte = None, None, 0
            data_src = "yfinance"
            try:
                # Strikes in [96%, 100%] of spot — covers the 2 % OTM target
                tds, chosen_exp, dte = await _fetch_opra_chain(
                    ib, ticker, "P", spot,
                    dte_min=0, dte_max=7,
                    otm_lo_pct=0.0, otm_hi_pct=4.0, max_strikes=8,
                )
                data_src = "OPRA-LIVE" if state.get("opra_active") else "OPRA-DELAYED"
            except ValueError as e:
                log.warning("0DTE [%s]: OPRA unavailable (%s) — falling back to yfinance", ticker, e)

            best = None

            if tds:
                tgt_strike = spot * 0.98
                exp_display = f"{chosen_exp[:4]}-{chosen_exp[4:6]}-{chosen_exp[6:]}"
                for td in tds:
                    K   = _safe_float(td.contract.strike)
                    bid = _safe_float(td.bid, 0.0)
                    ask = _safe_float(td.ask, 0.0)
                    if K is None or bid < 0.25 or ask <= 0:
                        continue
                    greeks = td.modelGreeks
                    iv  = _safe_float(greeks.impliedVol) if greeks else None
                    oi  = int(_safe_float(td.openInterest, 0))
                    vol = int(_safe_float(td.volume, 0))
                    mid = (bid + ask) / 2
                    spread_pct = (ask - bid) / mid * 100 if mid > 0 else 100
                    cand = {
                        "ticker":           ticker,
                        "expiry":           chosen_exp,
                        "expiry_display":   exp_display,
                        "dte":              dte,
                        "strike":           K,
                        "bid":              round(bid, 2),
                        "ask":              round(ask, 2),
                        "mid":              round(mid, 2),
                        "iv_pct":           round(iv * 100, 1) if iv else None,
                        "oi":               oi,
                        "volume":           vol,
                        "spot":             round(spot, 2),
                        "otm_pct":          round((spot - K) / spot * 100, 1),
                        "daily_return_pct": round(mid / spot * 100, 3),
                        "liquidity_score":  _liquidity_score(oi, vol, spread_pct),
                        "data_source":      data_src,
                        "_dist":            abs(K - tgt_strike),
                    }
                    if best is None or cand["_dist"] < best["_dist"]:
                        best = cand

                if best:
                    best.pop("_dist")
                    candidates.append(best)

            # ── Fallback: yfinance (market closed or OPRA unavailable) ──
            if not best:
                def _yf_0dte():
                    tk = yf.Ticker(ticker)
                    near = sorted(
                        (datetime.strptime(e, "%Y-%m-%d").date(), e)
                        for e in (tk.options or [])
                        if 0 <= (datetime.strptime(e, "%Y-%m-%d").date() - today).days <= 7
                    )
                    if not near:
                        return None
                    exp_d, exp_str = near[0]
                    dte_yf = (exp_d - today).days
                    chain  = tk.option_chain(exp_str)
                    puts   = chain.puts.copy()
                    puts["_diff"] = abs(puts["strike"] - spot * 0.98)
                    for _, r in puts.sort_values("_diff").head(4).iterrows():
                        K2   = float(r["strike"])
                        bid2 = float(r.get("bid", 0) or 0)
                        ask2 = float(r.get("ask", 0) or 0)
                        if bid2 < 0.25 or ask2 <= 0:
                            continue
                        mid2 = (bid2 + ask2) / 2
                        oi2  = int(r.get("openInterest", 0) or 0)
                        vol2 = int(r.get("volume", 0) or 0)
                        iv2  = float(r.get("impliedVolatility", 0) or 0)
                        sp2  = (ask2 - bid2) / mid2 * 100 if mid2 > 0 else 100
                        return {
                            "ticker":           ticker,
                            "expiry":           exp_str.replace("-", ""),
                            "expiry_display":   exp_str,
                            "dte":              dte_yf,
                            "strike":           K2,
                            "bid":              round(bid2, 2),
                            "ask":              round(ask2, 2),
                            "mid":              round(mid2, 2),
                            "iv_pct":           round(iv2 * 100, 1),
                            "oi":               oi2,
                            "volume":           vol2,
                            "spot":             round(spot, 2),
                            "otm_pct":          round((spot - K2) / spot * 100, 1),
                            "daily_return_pct": round(mid2 / spot * 100, 3),
                            "liquidity_score":  _liquidity_score(oi2, vol2, sp2),
                            "data_source":      "yfinance",
                        }
                    return None

                yf_cand = await asyncio.get_event_loop().run_in_executor(None, _yf_0dte)
                if yf_cand:
                    candidates.append(yf_cand)

        except Exception as exc:
            log.warning("0DTE scan %s: %s", ticker, exc)

    candidates = [_json_safe(c) for c in candidates]
    candidates.sort(key=lambda x: x.get("daily_return_pct") or 0, reverse=True)
    return {"candidates": candidates, "count": len(candidates), "date": today.isoformat()}


# ── Earnings IV-crush scanner ───────────────────────────────────────────────

async def scan_earnings_iv(ib: IB) -> dict:
    """
    Find tickers with earnings in 2–7 days and elevated IV rank (≥ 50).
    Strategy: sell put AFTER earnings date to capture IV crush.
    """
    today      = date.today()
    candidates = []
    for ticker in CSP_UNIVERSE:
        try:
            earnings_days = await _earnings_days_out(ticker)
            if earnings_days is None or not (2 <= earnings_days <= 7):
                continue
            iv_info = await _iv_rank_for_ticker(ticker)
            iv_rank = iv_info.get("rank", 0)
            if iv_rank < 50:
                continue
            spot = await _get_stock_price(ib, ticker)
            if spot <= 0:
                continue
            tk       = yf.Ticker(ticker)
            expiries = tk.options
            # First expiry AFTER earnings
            target_exp = None
            for exp in sorted(expiries):
                d_exp = datetime.strptime(exp, "%Y-%m-%d").date()
                dte   = (d_exp - today).days
                if earnings_days + 1 <= dte <= 28:
                    target_exp = exp
                    break
            if not target_exp:
                continue
            chain = tk.option_chain(target_exp)
            puts  = chain.puts.copy()
            # 8–12 % OTM for earnings buffer
            tgt_k = spot * 0.90
            puts["_diff"] = abs(puts["strike"] - tgt_k)
            r = puts.sort_values("_diff").iloc[0]
            bid    = float(r.get("bid", 0) or 0)
            ask    = float(r.get("ask", 0) or 0)
            if bid < 0.10:
                continue
            mid         = (bid + ask) / 2
            prem_pct    = mid / spot * 100
            dte_val     = (datetime.strptime(target_exp, "%Y-%m-%d").date() - today).days
            candidates.append({
                "ticker":        ticker,
                "expiry":        target_exp.replace("-", ""),
                "expiry_display":target_exp,
                "dte":           dte_val,
                "strike":        float(r["strike"]),
                "bid":           round(bid, 2),
                "ask":           round(ask, 2),
                "mid":           round(mid, 2),
                "premium_pct":   round(prem_pct, 2),
                "iv_rank":       round(iv_rank, 1),
                "earnings_days": earnings_days,
                "spot":          round(spot, 2),
                "note":          f"Earnings in {earnings_days}d — sell after announcement",
            })
        except Exception as exc:
            log.warning("Earnings IV scan %s: %s", ticker, exc)
    candidates.sort(key=lambda x: x["iv_rank"], reverse=True)
    return {"candidates": candidates, "count": len(candidates)}


# ── Backtesting ─────────────────────────────────────────────────────────────

async def _backtest_csp_ticker(
    ticker: str, weeks: int = 26,
    profit_target_pct: float = 0.25,
    stop_loss_mult: float = 1.5,
) -> dict:
    """
    Simulate weekly 20-delta CSP on a ticker using 2-yr historical prices.
    Uses BS pricing with realized vol as IV proxy.  Returns stats + last-20 trades.
    """
    def _run() -> dict:
        tk   = yf.Ticker(ticker)
        hist = tk.history(period="2y", interval="1d")
        if hist.empty or len(hist) < 30:
            return {}
        closes = hist["Close"].values.astype(float)
        dates  = [str(d.date()) for d in hist.index]
        trades: list = []
        i = 20
        while i < len(closes) - 6 and len(trades) < weeks:
            rv_window  = closes[i - 20:i]
            log_rets   = np.log(rv_window[1:] / rv_window[:-1])
            sigma      = float(np.std(log_rets) * np.sqrt(252))
            if sigma <= 0:
                i += 5
                continue
            S     = float(closes[i])
            T     = 5 / 252
            K     = round(S * np.exp(-0.842 * sigma * math.sqrt(T)), 0)
            prem  = _bs_put_price(S, K, T, sigma)
            if prem < 0.05:
                i += 5
                continue
            tgt_buy  = prem * (1 - profit_target_pct)
            stop_buy = prem * (1 + stop_loss_mult) if stop_loss_mult > 0 else float("inf")
            exit_pnl = 0.0
            win      = False
            for j in range(i + 1, min(i + 6, len(closes))):
                S_j   = float(closes[j])
                T_j   = max((min(i + 5, len(closes) - 1) - j) / 252, 0.001)
                p_j   = _bs_put_price(S_j, K, T_j, sigma)
                gain  = (prem - p_j) * 100
                if p_j <= tgt_buy:
                    exit_pnl, win = gain, True
                    break
                if p_j >= stop_buy:
                    exit_pnl = gain
                    break
            else:
                S_exit   = float(closes[min(i + 5, len(closes) - 1)])
                exit_pnl = (prem - max(0, K - S_exit)) * 100
                win      = S_exit >= K
            trades.append({
                "date":    dates[i],
                "spot":    round(float(S), 2),
                "strike":  round(float(K), 2),
                "premium": round(float(prem), 2),
                "iv_pct":  round(float(sigma) * 100, 1),
                "exit_pnl":round(float(exit_pnl), 2),
                "win":     bool(win),       # numpy.bool_ → Python bool for JSON
            })
            i += 5
        if not trades:
            return {}
        wins     = [t for t in trades if t["win"]]
        losses   = [t for t in trades if not t["win"]]
        pnls     = np.array([t["exit_pnl"] for t in trades])
        cum      = np.cumsum(pnls)
        dd       = cum - np.maximum.accumulate(cum)
        avg_pnl  = float(np.mean(pnls))
        std_pnl  = float(np.std(pnls))
        sharpe   = round(avg_pnl / std_pnl * math.sqrt(52), 2) if std_pnl > 0 else 0.0
        return {
            "ticker":       ticker,
            "weeks":        len(trades),
            "win_rate":     round(len(wins) / len(trades) * 100, 1),
            "total_pnl":    round(float(np.sum(pnls)), 0),
            "avg_win":      round(float(np.mean([t["exit_pnl"] for t in wins])), 0) if wins else 0,
            "avg_loss":     round(float(np.mean([t["exit_pnl"] for t in losses])), 0) if losses else 0,
            "max_drawdown": round(float(np.min(dd)), 0),
            "sharpe":       sharpe,
            "trades":       trades[-20:],
        }
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(None, _run), timeout=30)
    except asyncio.TimeoutError:
        log.warning("Backtest timeout for %s", ticker)
        return {}
    except Exception as exc:
        log.warning("Backtest executor error for %s: %s", ticker, exc)
        return {}


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
    except BaseException as e:
        log.error("_run_in_streaming_loop unhandled: %s — %s", type(e).__name__, e)
        raise


# ── FastAPI app ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        state["model"] = joblib.load(MODEL_PATH)
        log.info("Loaded cached model from disk")

    state["iv_history"] = _load_iv_history()
    log.info(f"Loaded IV history for {len(state['iv_history'])} tickers")
    _journal_init()

    # Restore auto-trader positions + config from last shutdown
    _at_load_state()

    # Warm up the learned model from journal so ranking works immediately
    try:
        result = _retrain_from_journal()
        if "error" not in result:
            log.info(f"Learned model warmed up: v{result['version']}, "
                     f"win={result['win_rate']}%, CV={result['cv_accuracy']}%")
        else:
            log.info(f"Learned model not ready yet: {result['error']}")
    except Exception as _e:
        log.warning(f"Model warm-up skipped: {_e}")

    t = threading.Thread(target=streaming_loop, daemon=True)
    t.start()
    log.info("Live streaming thread started")
    asyncio.create_task(_autotrader_background())
    log.info("Auto-trader background task started")

    # Run initial universe screen in background (non-blocking)
    async def _initial_screen():
        tickers = await _screen_universe()
        if tickers:
            CSP_UNIVERSE.clear()
            CSP_UNIVERSE.extend(tickers)
            state["scan_cache"]["csp"]   = None
            state["scan_cache"]["leaps"] = None
            log.info("Startup universe ready: %d tickers", len(CSP_UNIVERSE))
        else:
            log.info("Startup universe screen failed — using default %d tickers", len(CSP_UNIVERSE))

    asyncio.create_task(_initial_screen())
    asyncio.create_task(_universe_scheduler())
    log.info("Universe screener + daily scheduler started")
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
@app.get("/opra/recheck")
async def opra_recheck():
    """Re-probe OPRA subscription status without restarting the server."""
    ib = state.get("ib")
    if not ib or not state.get("connected"):
        raise HTTPException(503, "Not connected to TWS")
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(_check_opra_subscription(ib), timeout=15),
        )
        state["opra_active"] = result
        return {"opra_active": result}
    except Exception as e:
        raise HTTPException(503, str(e))


@app.get("/opra/debug")
async def opra_debug():
    """Return raw IBKR ticker values from an SPY options snapshot — for diagnosing OPRA issues."""
    ib = state.get("ib")
    if not ib or not state.get("connected"):
        raise HTTPException(503, "Not connected to TWS")

    async def _probe(ib: IB) -> dict:
        ib.reqMarketDataType(1)
        spy_price = await _get_stock_price(ib, "SPY")
        if spy_price <= 0:
            spy_price = 740.0
        strike = float(round(spy_price / 5) * 5)
        results = []
        for weeks_out in (0, 1):
            expiry = _next_expiry(weeks_out)
            c = Option("SPY", expiry, strike, "P", "SMART")
            await ib.qualifyContractsAsync(c)
            if not c.conId:
                results.append({"expiry": expiry, "strike": strike, "conId": None, "error": "qualify failed"})
                continue
            [td] = await ib.reqTickersAsync(c)
            greeks = td.modelGreeks
            results.append({
                "expiry":   expiry,
                "strike":   strike,
                "conId":    c.conId,
                "bid":      td.bid,
                "ask":      td.ask,
                "last":     getattr(td, "last", None),
                "close":    getattr(td, "close", None),
                "volume":   getattr(td, "volume", None),
                "open_interest": getattr(td, "openInterest", None),
                "delta":    greeks.delta if greeks else None,
                "iv":       greeks.impliedVol if greeks else None,
                "greeks_present": greeks is not None,
            })
        return {"spy_price": spy_price, "contracts": results}

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(_probe(ib), timeout=20),
        )
        return result
    except Exception as e:
        raise HTTPException(503, str(e))


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
    except BaseException as e:
        tb = traceback.format_exc()
        log.error("CSP scan unhandled exception: %s\n%s", e, tb)
        raise HTTPException(500, f"{type(e).__name__}: {e}")

    if not result:
        return {"cached": False, "count": 0, "candidates": [], "market_regime": {}}

    candidates = _json_safe(result["candidates"])
    regime     = _json_safe(result["regime"])
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
    return {
        "universe":          CSP_UNIVERSE,
        "count":             len(CSP_UNIVERSE),
        "candidate_pool":    len(CANDIDATE_POOL),
        "last_screened":     state["universe_last_screened"],
        "scores":            state["universe_scores"],
    }


@app.post("/csp/universe/add")
def add_to_csp_universe(req: AddTickerRequest):
    ticker = req.ticker.upper()
    if ticker not in CSP_UNIVERSE:
        CSP_UNIVERSE.append(ticker)
        state["scan_cache"]["csp"] = None   # invalidate cache
    return {"ok": True, "universe": CSP_UNIVERSE}


@app.post("/csp/universe/refresh")
async def refresh_csp_universe(top_n: int = Query(25, ge=10, le=50)):
    """Manually trigger a universe screen. Runs _screen_universe() immediately."""
    tickers = await _screen_universe(top_n=top_n)
    if tickers:
        CSP_UNIVERSE.clear()
        CSP_UNIVERSE.extend(tickers)
        state["scan_cache"]["csp"]   = None
        state["scan_cache"]["leaps"] = None
        return {
            "ok":           True,
            "universe":     CSP_UNIVERSE,
            "count":        len(CSP_UNIVERSE),
            "last_screened": state["universe_last_screened"],
        }
    return {"ok": False, "universe": CSP_UNIVERSE, "error": "Screen returned no results — universe unchanged"}


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
    except BaseException as e:
        tb = traceback.format_exc()
        log.error("LEAP scan unhandled exception: %s\n%s", e, tb)
        raise HTTPException(500, f"{type(e).__name__}: {e}")

    if not result:
        return {"cached": False, "count": 0, "candidates": [], "market_regime": {}}

    candidates = _json_safe(result["candidates"])
    regime     = _json_safe(result["regime"])
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


# ── Portfolio delta ────────────────────────────────────────────────────────
async def _portfolio_positions(ib: IB) -> dict:
    """
    Compute net portfolio delta from all open positions.
    OPT/FOP: live IBKR modelGreeks.delta via reqTickersAsync (preferred),
             falling back to BS delta with cached yfinance IV.
    STK:     delta = 1.0 × shares.
    Adds delta_source ('IBKR' or 'BS') and iv_pct to each option row.
    """
    items   = ib.portfolio()
    today_d = date.today()
    positions    = []
    total_delta  = 0.0

    # ── Batch-fetch live Greeks for all option positions in one round-trip ──
    opt_contracts = [
        i.contract for i in items
        if getattr(i.contract, "secType", "") in ("OPT", "FOP")
        and "FORECASTX" not in (getattr(i.contract, "exchange", ""), getattr(i.contract, "primaryExch", ""))
    ]
    live_delta: dict = {}   # conId → float
    live_iv: dict    = {}   # conId → float (percent, e.g. 28.5)
    if opt_contracts:
        try:
            await ib.qualifyContractsAsync(*opt_contracts)
            valid = [c for c in opt_contracts if c.conId]
            if valid:
                tds = await ib.reqTickersAsync(*valid)
                for td in tds:
                    if not (td.contract and td.modelGreeks):
                        continue
                    cid = td.contract.conId
                    d   = _safe_float(td.modelGreeks.delta)
                    iv  = _safe_float(td.modelGreeks.impliedVol)
                    if d  is not None: live_delta[cid] = d
                    if iv is not None: live_iv[cid]    = round(iv * 100, 2)
            log.info("Portfolio Greeks: %d/%d positions have live IBKR delta",
                     len(live_delta), len(opt_contracts))
        except Exception as e:
            log.warning("Portfolio Greeks snapshot failed (%s) — using BS fallback for all options", e)

    for item in items:
        c   = item.contract
        pos = float(item.position)
        sec = getattr(c, "secType", "")

        entry = {
            "symbol":          c.symbol,
            "sec_type":        sec,
            "position":        pos,
            "avg_cost":        _safe_float(item.averageCost, 0.0),
            "market_value":    _safe_float(item.marketValue, 0.0),
            "market_price":    _safe_float(item.marketPrice),
            "unrealized_pnl":  _safe_float(item.unrealizedPNL, 0.0),
            "realized_pnl":    _safe_float(item.realizedPNL, 0.0),
            "delta":           None,
            "position_delta":  None,
        }

        if sec == "STK":
            entry["delta"]          = 1.0
            entry["position_delta"] = round(pos, 2)
            total_delta += pos

        elif sec in ("OPT", "FOP"):
            try:
                expiry_str = getattr(c, "lastTradeDateOrContractMonth", "")[:8]
                exp_date   = datetime.strptime(expiry_str, "%Y%m%d").date()
                dte        = max((exp_date - today_d).days, 0)
                K          = float(getattr(c, "strike", 0))
                right      = getattr(c, "right", "C")
                mult       = float(getattr(c, "multiplier", 100) or 100)
                cid        = c.conId

                if cid and cid in live_delta:
                    # Live IBKR model Greeks — most accurate
                    delta      = live_delta[cid]
                    iv_pct     = live_iv.get(cid)
                    delta_src  = "IBKR"
                else:
                    # Black-Scholes fallback with cached yfinance IV
                    delta_src = "BS"
                    if sec == "OPT":
                        S = await _get_stock_price(ib, c.symbol)
                    else:
                        spy_bars = state["bars"].get("SPY")
                        S = float(spy_bars[-1]["close"]) * 10 if spy_bars else 5500.0
                    T        = max(dte / 365.0, 0.001)
                    iv_cache = state["ext_cache"]["iv_rank"].get(c.symbol)
                    sigma    = (iv_cache["iv"] / 100) if iv_cache and iv_cache.get("iv") else 0.25
                    delta    = _bs_delta(S, K, T, sigma, is_put=(right == "P"))
                    iv_pct   = round(sigma * 100, 1)

                if delta is None or math.isnan(delta):
                    raise ValueError("no valid delta")

                pd_val = round(pos * delta * mult, 2)
                entry.update({
                    "delta":          round(delta, 4),
                    "position_delta": pd_val,
                    "strike":         K,
                    "expiry":         expiry_str,
                    "right":          right,
                    "multiplier":     int(mult),
                    "dte":            dte,
                    "iv_pct":         iv_pct,
                    "delta_source":   delta_src,
                })
                total_delta += pd_val
            except Exception as e:
                log.debug("Delta calc [%s %s]: %s", c.symbol, sec, e)

        positions.append(_json_safe(entry))

    return {
        "total_delta":   round(total_delta, 2),
        "positions":     positions,
        "count":         len(positions),
        "timestamp":     datetime.utcnow().isoformat() + "Z",
    }


# ── Pydantic models ────────────────────────────────────────────────────────
class AutoTraderConfigRequest(BaseModel):
    enabled:            bool      = False
    max_positions:      int       = 5
    profit_target_pct:  float     = 0.65
    stop_loss_mult:     float     = 5.0
    scan_types:         List[str] = ["csp"]
    csp_capital:        float     = 20000.0
    leap_capital:       float     = 5000.0
    # Kelly
    use_kelly:          bool      = True
    total_capital:      float     = 100000.0
    assumed_win_rate:   float     = 0.85
    # Trailing exit
    trailing_exit:      bool      = True
    # Auto-hedge
    auto_hedge:         bool      = False
    hedge_threshold:    float     = 100.0


class ReconnectRequest(BaseModel):
    port: int   # 7497 = paper, 7496 = live, 4001/4002 = IB Gateway


class OrderRequest(BaseModel):
    ticker:      str
    expiry:      str            # YYYYMMDD
    strike:      float
    right:       str            # "P" or "C"
    action:      str            # "BUY" or "SELL"
    quantity:    int   = 1
    limit_price: Optional[float] = None   # None = derive from yfinance mid


# ── Order + portfolio endpoints ────────────────────────────────────────────
@app.post("/orders/place")
async def place_order(req: OrderRequest):
    """
    Place a limit order for a single option contract.
    For CSP: action=SELL, right=P.  For LEAP: action=BUY, right=C.
    If limit_price is omitted the backend fetches the yfinance mid and applies
    the same 40% spread improvement used by the scanner.
    """
    _require_connection()
    action = req.action.upper()
    right  = req.right.upper()
    if action not in ("BUY", "SELL"):
        raise HTTPException(400, "action must be BUY or SELL")
    if right not in ("C", "P"):
        raise HTTPException(400, "right must be C or P")

    limit_price = req.limit_price
    if limit_price is not None and limit_price <= 0:
        raise HTTPException(400, f"limit_price must be > 0 (got {limit_price})")
    if limit_price is None:
        exp_str = f"{req.expiry[:4]}-{req.expiry[4:6]}-{req.expiry[6:8]}"
        loop = asyncio.get_event_loop()
        def _get_mid():
            t  = yf.Ticker(req.ticker.upper())
            try:
                chain = t.option_chain(exp_str)
                df    = chain.calls if right == "C" else chain.puts
                row   = df[df["strike"] == req.strike]
                if row.empty:
                    return None
                bid = float(row["bid"].iloc[0])
                ask = float(row["ask"].iloc[0])
                mid = (bid + ask) / 2.0
                return round((bid + (mid - bid) * 0.40) if action == "SELL"
                             else (ask - (ask - mid) * 0.40), 2)
            except Exception:
                return None
        limit_price = await loop.run_in_executor(None, _get_mid)
        if limit_price is None:
            raise HTTPException(400, "Could not determine limit price — supply limit_price explicitly")

    async def _do_place(ib: IB):
        contract = Option(req.ticker.upper(), req.expiry, req.strike, right, "SMART")
        await ib.qualifyContractsAsync(contract)
        if not contract.conId:
            raise ValueError(f"Could not qualify {req.ticker} {req.expiry} {req.strike}{right}")
        order = LimitOrder(action, req.quantity, limit_price)
        trade = ib.placeOrder(contract, order)
        await asyncio.sleep(1)
        return {
            "order_id":    trade.order.orderId,
            "status":      trade.orderStatus.status,
            "ticker":      req.ticker.upper(),
            "expiry":      req.expiry,
            "strike":      req.strike,
            "right":       right,
            "action":      action,
            "quantity":    req.quantity,
            "limit_price": limit_price,
            "contract_id": contract.conId,
        }

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(_do_place(state["ib"]), timeout=30),
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (TimeoutError, RuntimeError) as e:
        raise HTTPException(503, str(e))

    return result


@app.get("/orders")
def get_orders():
    """List all currently open orders."""
    _require_connection()
    trades = state["ib"].openTrades()
    return {
        "orders": [
            {
                "order_id":    t.order.orderId,
                "ticker":      t.contract.symbol,
                "expiry":      getattr(t.contract, "lastTradeDateOrContractMonth", ""),
                "strike":      getattr(t.contract, "strike", None),
                "right":       getattr(t.contract, "right", ""),
                "action":      t.order.action,
                "quantity":    t.order.totalQuantity,
                "limit_price": t.order.lmtPrice,
                "tif":         t.order.tif,
                "status":      t.orderStatus.status,
                "why_held":    t.orderStatus.whyHeld or "",
                "filled":      t.orderStatus.filled,
                "remaining":   t.orderStatus.remaining,
                "avg_fill":    t.orderStatus.avgFillPrice,
            }
            for t in trades
        ],
        "count": len(trades),
    }


@app.delete("/orders/{order_id}")
def cancel_order(order_id: int):
    """Cancel an open order by its orderId."""
    _require_connection()
    ib = state["ib"]
    trades = ib.openTrades()
    target = next((t for t in trades if t.order.orderId == order_id), None)
    if target is None:
        raise HTTPException(404, f"Order {order_id} not found in open orders")
    ib.cancelOrder(target.order)
    return {"ok": True, "order_id": order_id, "message": "Cancel request sent"}


@app.post("/orders/cancel-inactive")
def cancel_inactive_orders():
    """Cancel all Inactive orders (stale from previous sessions)."""
    _require_connection()
    ib = state["ib"]
    stale = [t for t in ib.openTrades() if t.orderStatus.status == "Inactive"]
    cancelled = []
    for t in stale:
        try:
            ib.cancelOrder(t.order)
            cancelled.append(t.order.orderId)
        except Exception:
            pass
    _at_log("SYSTEM", f"Manually cancelled {len(cancelled)} Inactive orders")
    return {"ok": True, "cancelled": cancelled, "count": len(cancelled)}


@app.get("/portfolio/delta")
async def portfolio_delta():
    """Net portfolio delta across all open positions, computed via Black-Scholes."""
    _require_connection()
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(_portfolio_positions(state["ib"]), timeout=30),
        )
    except (TimeoutError, RuntimeError) as e:
        raise HTTPException(503, str(e))
    return result


# ── Auto-trader endpoints ──────────────────────────────────────────────────

@app.get("/autotrader/status")
def autotrader_status():
    """Return current auto-trader state, config, positions, and log."""
    return state["autotrader"]


@app.post("/autotrader/config")
def update_autotrader_config(req: AutoTraderConfigRequest):
    """Enable/disable the auto-trader and update its config."""
    was = state["autotrader"]["enabled"]
    state["autotrader"]["enabled"] = req.enabled
    state["autotrader"]["config"].update({
        "max_positions":     req.max_positions,
        "profit_target_pct": req.profit_target_pct,
        "stop_loss_mult":    req.stop_loss_mult,
        "scan_types":        req.scan_types,
        "csp_capital":       req.csp_capital,
        "leap_capital":      req.leap_capital,
        "use_kelly":         req.use_kelly,
        "total_capital":     req.total_capital,
        "assumed_win_rate":  req.assumed_win_rate,
        "trailing_exit":     req.trailing_exit,
        "auto_hedge":        req.auto_hedge,
        "hedge_threshold":   req.hedge_threshold,
    })
    if req.enabled and not was:
        _at_log("SYSTEM", "Auto-trader ENABLED — will scan every 5 min")
    elif not req.enabled and was:
        _at_log("SYSTEM", "Auto-trader DISABLED")
    _at_save_state()
    return {"ok": True}


@app.post("/autotrader/run-now")
async def autotrader_run_now():
    """Trigger an immediate monitor + scan cycle regardless of enabled state."""
    _require_connection()
    ib   = state["ib"]
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(
            None,
            lambda: _run_in_streaming_loop(_autotrader_monitor_coro(ib), timeout=30),
        )
        await loop.run_in_executor(
            None,
            lambda: _run_in_streaming_loop(_autotrader_scan_and_trade_coro(ib), timeout=180),
        )
        state["autotrader"]["last_run"] = datetime.utcnow().isoformat() + "Z"
    except (TimeoutError, RuntimeError) as exc:
        raise HTTPException(503, str(exc))
    return {"ok": True, "log": state["autotrader"]["log"][-20:]}


@app.post("/autotrader/clear-stale-positions")
def clear_stale_positions():
    """
    Remove positions from the tracking dict whose IBKR orders are Inactive.
    This resets bloated state from repeated scan cycles where orders never filled.
    Call this before enabling the auto-trader after fixing an Inactive order issue.
    """
    _require_connection()
    ib = state["ib"]
    at = state["autotrader"]

    # All currently open order IDs (Inactive, Submitted, PreSubmitted, etc.)
    open_trades  = ib.openTrades()
    open_ids     = {t.order.orderId for t in open_trades}
    inactive_ids = {t.order.orderId for t in open_trades if t.orderStatus.status == "Inactive"}
    # Portfolio keys for filled positions (these should stay in the dict)
    portfolio_keys = {_at_contract_key(item.contract) for item in ib.portfolio()}

    removed, kept = [], []
    for key, info in list(at["positions"].items()):
        order_id = info.get("order_id")
        in_open_trades = (order_id is not None and order_id in open_ids)
        in_portfolio   = key in portfolio_keys
        is_inactive    = (order_id is not None and order_id in inactive_ids)

        if in_portfolio:
            # Real filled position — keep it, the monitor manages it
            kept.append(key)
        elif is_inactive or not in_open_trades:
            # Inactive order (stuck) or ghost (order was cancelled/expired, not in portfolio)
            at["positions"].pop(key)
            removed.append(key)
        else:
            kept.append(key)

    _at_save_state()
    msg = f"Cleared {len(removed)} stale/ghost positions; {len(kept)} active positions kept"
    _at_log("SYSTEM", msg)
    return {"ok": True, "removed": removed, "kept": kept, "message": msg}


# ── 0DTE endpoint ──────────────────────────────────────────────────────────

@app.get("/scan/0dte")
async def scan_0dte_endpoint(
    refresh: bool = Query(False, description="Bypass 5-min cache")
):
    """Scan SPY/QQQ/IWM for 0–7 DTE cash-secured put setups."""
    _require_connection()
    cache = state["scan_cache"]
    if not refresh and cache.get("0dte") is not None and cache.get("ts"):
        age = (datetime.utcnow() - cache["ts"]).total_seconds()
        if age < SCAN_CACHE_TTL:
            return {"cached": True, "age_seconds": int(age), **cache["0dte"]}
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(scan_0dte(state["ib"]), timeout=60),
        )
    except (TimeoutError, RuntimeError) as exc:
        raise HTTPException(503, str(exc))
    cache["0dte"] = result
    return {"cached": False, **result}


# ── Earnings IV play endpoint ───────────────────────────────────────────────

@app.get("/scan/earnings-iv")
async def scan_earnings_iv_endpoint(
    refresh: bool = Query(False)
):
    """Find pre-earnings IV elevation plays in the CSP universe."""
    _require_connection()
    cache = state["scan_cache"]
    if not refresh and cache.get("earnings_iv") is not None and cache.get("ts"):
        age = (datetime.utcnow() - cache["ts"]).total_seconds()
        if age < SCAN_CACHE_TTL:
            return {"cached": True, "age_seconds": int(age), **cache["earnings_iv"]}
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _run_in_streaming_loop(scan_earnings_iv(state["ib"]), timeout=120),
        )
    except (TimeoutError, RuntimeError) as exc:
        raise HTTPException(503, str(exc))
    cache["earnings_iv"] = result
    return {"cached": False, **result}


# ── Backtest endpoint ───────────────────────────────────────────────────────

@app.get("/backtest/csp")
async def backtest_csp_endpoint(
    tickers: str = Query("AMD,LLY,AAPL,NVDA,SPY", description="Comma-separated tickers"),
    weeks:   int = Query(26, description="Lookback weeks (max 104)"),
    profit_target_pct: float = Query(0.65),
    stop_loss_mult:    float = Query(5.0),
):
    """Backtest weekly 20-delta CSP on requested tickers using 2-yr historical prices."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()][:8]
    weeks = min(max(weeks, 4), 104)
    results = []
    for ticker in ticker_list:
        try:
            r = await _backtest_csp_ticker(ticker, weeks, profit_target_pct, stop_loss_mult)
            if r:
                results.append(r)
        except BaseException as exc:
            log.warning("Backtest %s: %s", ticker, exc)
    summary = {
        "avg_win_rate":     round(sum(r["win_rate"] for r in results) / len(results), 1) if results else 0,
        "avg_sharpe":       round(sum(r["sharpe"]   for r in results) / len(results), 2) if results else 0,
        "total_pnl":        round(sum(r["total_pnl"] for r in results), 0) if results else 0,
    }
    return {"results": results, "summary": summary, "tickers": ticker_list, "weeks": weeks}


# ── Live account reconnect ──────────────────────────────────────────────────

@app.post("/reconnect")
async def reconnect_endpoint(req: ReconnectRequest):
    """
    Switch TWS port. 7497 = paper, 7496 = live.
    Disconnects the current session; streaming loop auto-reconnects.
    """
    if req.port not in (7496, 7497, 4001, 4002):
        raise HTTPException(400, "Port must be 7496 (live), 7497 (paper), 4001 or 4002 (gateway)")
    state["reconnect_port"] = req.port
    if state.get("ib") and state["ib"].isConnected():
        state["ib"].disconnect()
    mode = "LIVE" if req.port == 7496 else ("paper" if req.port == 7497 else "gateway")
    _at_log("SYSTEM", f"Reconnecting to port {req.port} ({mode})")
    return {"ok": True, "port": req.port, "mode": mode,
            "message": f"Reconnecting to {mode} account on port {req.port}…"}


# ── Trade Journal endpoints ────────────────────────────────────────────────

@app.get("/journal")
def get_journal(limit: int = Query(100, description="Most recent N trades")):
    """Return completed and open trade journal entries."""
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    rows = con.execute(
        "SELECT * FROM trade_journal ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    cols = [d[0] for d in con.execute("SELECT * FROM trade_journal LIMIT 0").description or []]
    con.close()
    trades = [dict(zip(cols, r)) for r in rows]
    return {"trades": trades, "count": len(trades)}


@app.get("/journal/stats")
def get_journal_stats():
    """Win rate, avg P&L, feature importances from completed trades."""
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    rows = con.execute("""
        SELECT win, pnl, pnl_pct, exit_reason, strategy_type, iv_rank, score
        FROM trade_journal WHERE closed_at IS NOT NULL AND win IS NOT NULL
    """).fetchall()
    model_rows = con.execute(
        "SELECT * FROM model_log ORDER BY id DESC LIMIT 10"
    ).fetchall()
    model_cols = [d[0] for d in (con.execute("SELECT * FROM model_log LIMIT 0").description or [])]
    con.close()

    if not rows:
        return {"total_trades": 0, "win_rate": None, "avg_pnl": None, "model_log": []}

    wins     = [r for r in rows if r[0] == 1]
    losses   = [r for r in rows if r[0] == 0]
    pnls     = [r[1] for r in rows if r[1] is not None]
    by_reason = {}
    for r in rows:
        rr = r[3] or "unknown"
        by_reason.setdefault(rr, {"total": 0, "wins": 0})
        by_reason[rr]["total"] += 1
        if r[0] == 1:
            by_reason[rr]["wins"] += 1

    model_log = [dict(zip(model_cols, r)) for r in model_rows]

    return {
        "total_trades":    len(rows),
        "win_rate":        round(len(wins) / len(rows) * 100, 1),
        "avg_pnl":         round(sum(pnls) / len(pnls), 2) if pnls else 0,
        "total_pnl":       round(sum(pnls), 2) if pnls else 0,
        "avg_win_pnl":     round(sum(r[1] for r in wins if r[1]) / len(wins), 2) if wins else 0,
        "avg_loss_pnl":    round(sum(r[1] for r in losses if r[1]) / len(losses), 2) if losses else 0,
        "exit_breakdown":  by_reason,
        "model_version":   state.get("model_version", 0),
        "model_log":       model_log,
        "assumed_win_rate":state["autotrader"]["config"].get("assumed_win_rate"),
    }


@app.post("/journal/retrain")
def trigger_retrain():
    """Manually trigger model retraining from journal data."""
    try:
        result = _retrain_from_journal()
        return result
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ── Shared guard ───────────────────────────────────────────────────────────
def _require_connection():
    if not state["connected"] or not state["ib"]:
        raise HTTPException(503, "IBKR not connected — ensure TWS/IB Gateway is running")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
