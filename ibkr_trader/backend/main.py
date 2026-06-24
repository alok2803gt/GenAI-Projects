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
CSP_MIN_RETURN_PCT  = 1.00   # Weekly premium / strike ≥ 1.0% — Tastytrade research: 1%+ weekly
                              # return is the floor where CSP premium compensates for assignment risk.
                              # 0.75% was too low — allows entries in flat-IV environments with thin premium.
CSP_MAX_DELTA       = 0.20   # Absolute delta (far-OTM safety)
CSP_MIN_OI          = 50     # Open interest floor
CSP_MAX_SPREAD_PCT  = 0.15   # Bid-ask spread as fraction of mid
SCAN_CACHE_TTL      = 300    # Seconds before scan cache expires

# ── Config — external validation (yfinance) ────────────────────────────────
EARNINGS_BLOCK_DAYS = 14     # Skip CSP/LEAP entirely if earnings within 14 days.
                              # IV inflates 5-10 days before earnings and the spike can blow
                              # through a 2× stop on a short put regardless of strike selection.
EARNINGS_WARN_DAYS  = 30     # Warn when earnings are within 30 days — the upcoming event
                              # compresses theta and skews the risk/reward even if it won't hit.
IV_RANK_MIN_CSP     = 50     # Only enter when IV is in the top 50th percentile of its range.
                              # Tastytrade 200k-trade study: outcomes improve materially above 50%.
                              # Journal losses averaged IV rank 36.8% — premium was too thin to
                              # compensate for the downside risk at those entry points.
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
UNIVERSE_CACHE_PATH = "universe_cache.json"     # screened universe (refreshed nightly)

# ── Config — LEAP scanner ──────────────────────────────────────────────────
LEAP_MIN_DTE   = 180   # ≥ 6 months
LEAP_MAX_DTE   = 540   # ≤ 18 months
LEAP_MIN_DELTA = 0.65  # Research: deep ITM (0.80+) preferred; 0.65 min to avoid pure speculation
LEAP_MAX_DELTA = 0.85  # Cap to avoid overpaying for very deep ITM
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
            "profit_target_pct": 0.50,   # 50% of max — research-optimal (Tastytrade 200k-trade study)
            "stop_loss_mult":    2.0,    # 2× premium received (industry-standard 200% rule)
            "scan_types":        ["csp"],
            "csp_capital":       20000.0,
            "leap_capital":      5000.0,
            # Kelly criterion
            "use_kelly":         True,
            "total_capital":     100000.0,
            "assumed_win_rate":  0.85,
            # Auto-hedge
            "auto_hedge":        False,
            "hedge_threshold":   100.0,
        },
        "positions":          {},     # contract_key → entry metadata
        "stopped_out":        {},     # ticker → ISO timestamp of last stop-loss (48h cooldown)
        "log":                [],     # [{time, action, detail}, …] last 200
        "decisions":          [],     # [{ts, action, ticker, headline, body}, …] last 500
        "last_run":           None,
        "premium_collected":  0.0,    # cumulative realized CSP wins
        "leap_pnl":           0.0,    # cumulative realized LEAP P&L (can be negative)
        "leap_budget":        0.0,    # 50% of CSP income + LEAP P&L (net available)
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
            # Use reqGlobalCancel (cancels ALL orders for this account regardless of session/clientId).
            # Regular cancelOrder only works for orders placed by the current connection,
            # so cross-session Inactive orders require the global cancel.
            await asyncio.sleep(2)   # give TWS a moment to deliver existing orders
            stale = [t for t in ib.openTrades() if t.orderStatus.status == "Inactive"]
            if stale:
                ib.client.reqGlobalCancel()
                log.info(f"reqGlobalCancel sent — clearing {len(stale)} stale Inactive orders on connect")
                _at_log("SYSTEM", f"Cleared {len(stale)} stale Inactive orders on reconnect (reqGlobalCancel)")

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

    Weight rationale (Tastytrade 200k-trade study + empirical review):
      IV rank is the #1 predictor of CSP outcome. Prior scoring gave it only
      10% weight (15/150pts) while premium got 50% (75pts). That caused
      high-premium, low-IV candidates to outrank high-IV, moderate-premium
      candidates — exactly backwards from what research supports.

      New weights (max 140 base pts):
        IV rank        35 pts  — primary driver; sell expensive vol, not just any vol
        Premium return 40 pts  — still important but capped lower (premium = f(IV), avoid double-count)
        Delta safety   20 pts  — assignment risk management (unchanged)
        Liquidity      20 pts  — execution quality (unchanged)
        Stock quality  15 pts  — trend/momentum confirmation (reduced; secondary)
        Premium/risk   10 pts  — premium vs OTM distance bonus (new; rewards efficient OTM value)
      OPRA signals:   ±30 pts  — institutional flow (unchanged)
    """
    s = 0.0

    # 1) IV rank — high IV = expensive premium = better seller's edge       max 35
    #    Tastytrade: outcomes improve materially above 50th percentile.
    iv_rank = float(row.get("iv_rank", 50))
    s += (iv_rank / 100) * 35

    # 2) Realistic weekly return above min target (capped at 2× minimum)   max 40
    #    Cap reduced from 3× to 2×: premium largely reflects IV (factor 1);
    #    uncapped premium let high-IV stocks dominate even at low IV rank.
    s += min(row["weekly_return_pct"] / CSP_MIN_RETURN_PCT, 2.0) * 20

    # 3) Delta safety — distance from assignment risk                       max 20
    s += (CSP_MAX_DELTA - abs(row["delta"])) / CSP_MAX_DELTA * 20

    # 4) Liquidity (OI + volume + spread quality)                           max 20
    s += (row.get("liquidity_score", 50) / 100) * 20

    # 5) Stock quality (trend strength vs 50-day SMA, RSI, momentum)       max 15
    s += row.get("stock_quality", 0.5) * 15

    # 6) Premium efficiency: premium collected vs OTM distance              max 10
    #    Rewards puts that collect meaningful credit relative to how far OTM
    #    they are. A 10%-OTM put at 2% premium beats a 3%-OTM put at 1.5%.
    spot   = float(row.get("stock_price") or row.get("spot") or 0)
    strike = float(row.get("strike") or 0)
    if spot > 0 and strike > 0 and spot > strike:
        otm_distance = (spot - strike) / spot  # fraction of stock price
        premium_pct  = row["weekly_return_pct"] / 100
        if otm_distance > 0:
            efficiency = min(premium_pct / otm_distance, 2.0)  # cap at 2×
            s += efficiency * 5  # max 10 pts at 2× efficiency

    # 7) RSI timing bonus: oversold pullback within uptrend = ideal CSP entry  max +5 / min −3
    #    pct_b<30 is unreachable when above_sma20=True (pct_b>=50 at/above midline by math),
    #    so RSI is the correct proxy for "temporarily depressed within an uptrend."
    rsi14 = row.get("rsi14")
    if rsi14 is not None:
        if iv_rank >= 45 and 35 <= rsi14 <= 48:
            s += 5   # oversold pullback + expensive vol: ideal CSP entry timing
        elif 35 <= rsi14 <= 52:
            s += 2   # mild pullback within trend
        elif rsi14 < 35:
            s -= 3   # too oversold: momentum risk, potential gap through strike

    # BB extension penalty: pct_b > 80 = price in upper band = reversion risk to strike
    pct_b = row.get("pct_b")
    if pct_b is not None and pct_b > 80:
        s -= 4   # price extended above midline: higher assignment risk on pullback

    # 8) OPRA institutional signals (bonus / penalty)
    # Strike above max pain → MMs incentivised to keep price here          +8
    if row.get("max_pain") and row["strike"] >= row["max_pain"]:
        s += 8
    # Put/call volume ratio < 0.7 → call-dominant flow (bullish)           +6
    pc_vol = row.get("pc_vol_ratio", 1.0)
    if pc_vol < 0.7:
        s += 6
    elif pc_vol > 1.5:
        s -= 6   # heavy put buying — bearish flow
    # No unusual put activity at this strike                                +4
    if row.get("vol_oi_ratio", 1.0) < 1.0:
        s += 4
    elif row.get("vol_oi_ratio", 1.0) > 2.0:
        s -= 8   # volume >> OI: directional put bet
    # Ask-heavy order book → buyers lifting offers (bullish)                +4
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
    # Strip legacy keys that no longer exist in the codebase
    clean_config = {k: v for k, v in at["config"].items() if k != "trailing_exit"}
    payload = {
        "enabled":           at["enabled"],
        "config":            clean_config,
        "positions":         at["positions"],
        "stopped_out":       at.get("stopped_out", {}),
        "premium_collected": at["premium_collected"],
        "leap_pnl":          at.get("leap_pnl", 0.0),
        "leap_budget":       at["leap_budget"],
        "model_version":     state.get("model_version", 0),
        "decisions":         at.get("decisions", [])[-500:],
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
        at["leap_pnl"]          = saved.get("leap_pnl", 0.0)
        at["leap_budget"]       = saved.get("leap_budget", 0.0)
        # Merge saved config over defaults; strip removed keys on load
        merged = {**saved.get("config", {})}
        merged.pop("trailing_exit", None)   # removed feature
        at["config"].update(merged)
        at["positions"]   = saved.get("positions", {})
        at["stopped_out"] = saved.get("stopped_out", {})  # restore 48h cooldowns
        at["decisions"]   = saved.get("decisions", [])    # restore plain-English trade log
        state["model_version"] = saved.get("model_version", 0)
        log.info(
            f"Auto-trader state restored: enabled={at['enabled']}, "
            f"positions={len(at['positions'])}, model_v{state['model_version']}"
        )
    except Exception as e:
        log.warning(f"Auto-trader state load failed (starting fresh): {e}")

def _universe_save(tickers: list) -> None:
    """Persist screened universe to disk so restarts don't revert to hardcoded list."""
    saved_at = datetime.utcnow().isoformat() + "Z"   # Z suffix so browsers parse as UTC
    try:
        with open(UNIVERSE_CACHE_PATH, "w") as f:
            json.dump({"tickers": tickers, "saved_at": saved_at}, f)
        state["universe_last_screened"] = saved_at   # keep in-memory state current
    except Exception as e:
        log.warning("Universe cache save failed: %s", e)


def _universe_load() -> None:
    """Load screened universe from cache on startup (skip if > 7 days old)."""
    if not os.path.exists(UNIVERSE_CACHE_PATH):
        return
    try:
        with open(UNIVERSE_CACHE_PATH, "r") as f:
            data = json.load(f)
        saved_at = datetime.fromisoformat(
            (data.get("saved_at", "2000-01-01") or "2000-01-01").replace("Z", "+00:00")
        ).replace(tzinfo=None)   # strip tz for naive UTC comparison
        age_days = (datetime.utcnow() - saved_at).days
        if age_days > 7:
            log.info("Universe cache is %d days old — using hardcoded universe", age_days)
            return
        tickers = [t for t in data.get("tickers", []) if isinstance(t, str)]
        if tickers:
            CSP_UNIVERSE.clear()
            CSP_UNIVERSE.extend(tickers)
            state["universe_last_screened"] = data.get("saved_at")   # restore timestamp for UI
            log.info("Universe restored from cache (%d tickers, %dd old)", len(tickers), age_days)
    except Exception as e:
        log.warning("Universe cache load failed: %s", e)


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

def _compute_indicators_from_closes(closes: pd.Series, volumes: pd.Series | None = None) -> dict:
    empty = {"rsi14": None, "macd": None, "macd_signal": None,
             "macd_hist": None,
             "sma20": None, "sma50": None, "sma200": None,
             "above_sma20": None, "above_sma50": None, "above_sma200": None,
             "bb_upper": None, "bb_mid": None, "bb_lower": None, "pct_b": None,
             "vol_ratio": None}
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
    sma200 = closes.rolling(200).mean() if len(closes) >= 200 else None

    # Bollinger Bands (20, 2)
    bb_std    = closes.rolling(20).std()
    bb_upper_s = sma20 + 2 * bb_std
    bb_lower_s = sma20 - 2 * bb_std

    last = float(closes.iloc[-1])
    s20  = _safe(sma20)
    s50  = _safe(sma50)
    s200 = _safe(sma200) if sma200 is not None else None
    b_upper = _safe(bb_upper_s)
    b_lower = _safe(bb_lower_s)
    pct_b   = round((last - b_lower) / (b_upper - b_lower) * 100, 1) \
              if (b_upper and b_lower and b_upper != b_lower) else None

    # Volume ratio (today vs 20-day avg) + per-ticker 90th-pct threshold
    vol_ratio = None
    vol_90pct = None
    if volumes is not None and len(volumes) >= 21:
        # Rolling 20-day avg shifted by 1 so each day uses only prior data (no look-ahead)
        roll_avg   = volumes.rolling(20).mean().shift(1)
        all_ratios = (volumes / roll_avg.replace(0, float("nan"))).dropna()
        # 90th-percentile of trailing 252 days → dynamic, per-ticker breakout threshold
        recent = all_ratios.iloc[-252:] if len(all_ratios) >= 252 else all_ratios
        vol_90pct = round(float(recent.quantile(0.90)), 2) if len(recent) >= 20 else None

        avg_vol   = float(volumes.iloc[-21:-1].mean())
        today_vol = float(volumes.iloc[-1])
        vol_ratio = round(today_vol / avg_vol, 2) if avg_vol > 0 else None

    return {
        "rsi14":       round(_safe(rsi)  or 50, 1),
        "macd":        round(_safe(macd) or 0,  4),
        "macd_signal": round(_safe(msig) or 0,  4),
        "macd_hist":   round(_safe(mhist) or 0, 4),
        "sma20":        round(s20,  2) if s20  is not None else None,
        "sma50":        round(s50,  2) if s50  is not None else None,
        "sma200":       round(s200, 2) if s200 is not None else None,
        "above_sma20":  bool(last > s20)  if s20  is not None else None,
        "above_sma50":  bool(last > s50)  if s50  is not None else None,
        "above_sma200": bool(last > s200) if s200 is not None else None,
        "bb_upper":     round(b_upper, 2) if b_upper is not None else None,
        "bb_mid":       round(s20,   2)   if s20   is not None else None,
        "bb_lower":     round(b_lower, 2) if b_lower is not None else None,
        "pct_b":        pct_b,
        "vol_ratio":    vol_ratio,
        "vol_90pct":    vol_90pct,
    }


async def _tech_indicators(ticker: str) -> dict:
    """RSI-14 + MACD + BB + SMA-200 from streaming bars or yfinance daily fallback."""
    empty = {"rsi14": None, "macd": None, "macd_signal": None,
             "macd_hist": None,
             "sma20": None, "sma50": None, "sma200": None,
             "above_sma20": None, "above_sma50": None, "above_sma200": None,
             "bb_upper": None, "bb_mid": None, "bb_lower": None, "pct_b": None,
             "vol_ratio": None}
    bars = state["bars"].get(ticker)
    if bars and len(bars) >= 30:
        closes = pd.Series([b["close"] for b in bars], dtype=float)
        return _compute_indicators_from_closes(closes)
    loop = asyncio.get_event_loop()
    try:
        def _fetch():
            hist = yf.Ticker(ticker).history(period="1y")
            if hist.empty or len(hist) < 26:
                return None, None
            return hist["Close"].astype(float), hist["Volume"].astype(float)
        closes, volumes = await loop.run_in_executor(None, _fetch)
        if closes is not None:
            return _compute_indicators_from_closes(closes, volumes)
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
            spy_hist  = yf.Ticker("SPY").history(period="3mo")["Close"]
            vix_hist  = yf.Ticker("^VIX").history(period="5d")["Close"]
            spy_close = float(spy_hist.iloc[-1])
            spy_sma50 = float(spy_hist.rolling(50).mean().iloc[-1])
            spy_sma20 = float(spy_hist.rolling(20).mean().iloc[-1])
            # 5-day return to detect short-term pullbacks within a bull trend
            spy_5d    = float(spy_hist.pct_change(5).iloc[-1] * 100) if len(spy_hist) >= 6 else 0.0
            vix       = float(vix_hist.iloc[-1])
            return spy_close, spy_sma50, spy_sma20, spy_5d, vix

        spy_close, spy_sma50, spy_sma20, spy_5d, vix = await loop.run_in_executor(None, _fetch)
        spy_vs_50d = round((spy_close - spy_sma50) / spy_sma50 * 100, 2)
        spy_vs_20d = round((spy_close - spy_sma20) / spy_sma20 * 100, 2)

        # Regime logic: VIX thresholds slightly relaxed from 20/25 to 22/28
        # to avoid hard Kelly cliffs from brief VIX spikes. SMA20 added as
        # short-term trend confirmation — both MAs must agree for BULL.
        above_50d = spy_close >= spy_sma50
        above_20d = spy_close >= spy_sma20
        if above_50d and above_20d and vix < 22:
            regime = "BULL"
        elif above_50d and vix < 28:
            regime = "NEUTRAL"
        else:
            regime = "BEAR"

        result = {
            "regime":         regime,
            "spy_close":      round(spy_close, 2),
            "spy_sma50":      round(spy_sma50, 2),
            "spy_sma20":      round(spy_sma20, 2),
            "spy_vs_50d_pct": spy_vs_50d,
            "spy_vs_20d_pct": spy_vs_20d,
            "spy_5d_pct":     round(spy_5d, 2),
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
        # Row is filtered from Recommended view when earnings_days <= EARNINGS_BLOCK_DAYS * 2.
        # Label makes this visible in Show-all mode so user understands why it disappears.
        suffix = " — excl. recommended" if earnings_days <= EARNINGS_BLOCK_DAYS * 2 else ""
        warnings.append(f"Earnings in {earnings_days}d{suffix}")
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
                _universe_save(tickers)   # persist so restarts don't revert
                log.info("Universe refreshed: %d tickers", len(CSP_UNIVERSE))
        except asyncio.CancelledError:
            break
        except Exception as exc:
            log.error("Universe scheduler error: %s", exc)
            await asyncio.sleep(3600)  # back off 1 h on unexpected failure


# ── Recommendation filters (mirrors frontend JS) ──────────────────────────

def _filter_csp_recommended(candidates: list, log_diag: bool = False) -> list:
    """Filter scan candidates to recommended CSPs. Set log_diag=True to emit AT diagnostics."""
    n_total     = len(candidates)
    n_warnings  = sum(1 for r in candidates if len(r.get("warnings", [])) > 0)
    n_liq       = sum(1 for r in candidates if len(r.get("warnings", [])) == 0 and r["liquidity_score"] < 50)
    n_iv        = sum(1 for r in candidates if len(r.get("warnings", [])) == 0
                      and r["liquidity_score"] >= 50 and r["iv_rank"] < IV_RANK_MIN_CSP)
    n_score     = sum(1 for r in candidates if len(r.get("warnings", [])) == 0
                      and r["liquidity_score"] >= 50 and r["iv_rank"] >= IV_RANK_MIN_CSP
                      and r["score"] < 70)
    n_trend     = sum(1 for r in candidates if len(r.get("warnings", [])) == 0
                      and r["liquidity_score"] >= 50 and r["iv_rank"] >= IV_RANK_MIN_CSP
                      and r["score"] >= 70
                      and (r.get("above_sma50") is False or r.get("above_sma20") is False
                           or r.get("above_sma200") is False))
    n_earnings  = sum(1 for r in candidates if len(r.get("warnings", [])) == 0
                      and r["liquidity_score"] >= 50 and r["iv_rank"] >= IV_RANK_MIN_CSP
                      and r["score"] >= 70
                      and r.get("above_sma50") is not False and r.get("above_sma20") is not False
                      and r.get("above_sma200") is not False
                      and r["earnings_days_out"] is not None
                      and r["earnings_days_out"] <= EARNINGS_BLOCK_DAYS * 2)

    # SMA-20/50/200 must be True or None (None = not enough history → benefit of doubt).
    # above_sma200 is False only when we have 200 bars and the price is below — confirmed downtrend.
    clean = [r for r in candidates
             if len(r.get("warnings", [])) == 0
             and r["liquidity_score"] >= 50
             and r["iv_rank"] >= IV_RANK_MIN_CSP
             and r["score"] >= 70
             and r.get("above_sma50") is not False
             and r.get("above_sma20") is not False
             and r.get("above_sma200") is not False
             and (r["earnings_days_out"] is None or r["earnings_days_out"] > EARNINGS_BLOCK_DAYS * 2)]

    if log_diag and n_total > 0:
        _at_log("SCAN",
            f"CSP filter: {n_total} raw → {len(clean)} passed "
            f"[warnings:{n_warnings} liq:{n_liq} iv_rank:{n_iv} "
            f"score:{n_score} trend(sma20/50/200):{n_trend} earnings:{n_earnings}]")

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
    from zoneinfo import ZoneInfo
    est_now = datetime.now(ZoneInfo("America/New_York"))
    entry = {"time": est_now.strftime("%H:%M:%S ET"), "action": action, "detail": detail}
    at = state["autotrader"]
    at["log"].append(entry)
    at["log"] = at["log"][-200:]
    log.info("[AutoTrader] %s: %s", action, detail)


def _decision_log(action: str, ticker: str, headline: str, body: str) -> None:
    """Store a plain-English decision explanation in the decisions list (last 500)."""
    from zoneinfo import ZoneInfo
    est_now = datetime.now(ZoneInfo("America/New_York"))
    entry = {
        "ts":       est_now.isoformat(),
        "time":     est_now.strftime("%b %d %H:%M ET"),
        "action":   action,   # ENTER / CLOSE_PROFIT / CLOSE_STOP / CLOSE_21DTE / ROLL / ROTATE
        "ticker":   ticker,
        "headline": headline,
        "body":     body,
    }
    at = state["autotrader"]
    at.setdefault("decisions", []).append(entry)
    at["decisions"] = at["decisions"][-500:]


def _at_contract_key(c) -> str:
    return (f"{c.symbol}_{getattr(c,'right','')}"
            f"{getattr(c,'strike','')}{getattr(c,'lastTradeDateOrContractMonth','')}")


def _kelly_qty(cfg: dict, strike: float, t_type: str, mid_price: float = 0.0,
               regime: str = "BULL") -> int:
    """Half-Kelly position sizing, scaled by market regime.

    Capital base per strategy:
      CSP  → csp_capital   (the budget allocated to short-put trades)
      LEAP → leap_capital  (the budget allocated to long-call trades)

    Kelly is a within-strategy formula — it expresses what fraction of the
    strategy's own capital to risk per trade, not the whole portfolio.
    total_capital is a portfolio-level reference used for display only.
    """
    p  = float(cfg.get("assumed_win_rate", 0.85))
    pt = float(cfg.get("profit_target_pct", 0.50))
    sl = float(cfg.get("stop_loss_mult", 2.0))
    b  = pt / sl if sl > 0 else pt / 5.0
    kelly = (p * (b + 1) - 1) / b if b > 0 else 0.0
    frac  = max(0.02, kelly * 0.5)   # half-Kelly
    regime_scale = {"BULL": 1.0, "NEUTRAL": 0.6, "BEAR": 0.35, "UNKNOWN": 0.5}.get(regime, 0.5)

    if t_type == "csp":
        base = float(cfg.get("csp_capital", 20000.0))
        capital = base * frac * regime_scale
        return max(1, int(capital / (strike * 100)))
    else:  # leap
        base = float(cfg.get("leap_capital", 5000.0))
        m = mid_price if mid_price > 0 else 5.0
        capital = base * frac * regime_scale
        return max(1, int(capital / (m * 100)))


def _bs_put_price(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes put price using the global risk-free rate (_RF_RATE = 4.5%)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (_RF_RATE + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-_RF_RATE * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


async def _autotrader_monitor_coro(ib: IB) -> None:
    at  = state["autotrader"]
    cfg = at["config"]
    if not at["positions"]:
        return

    profit_target = float(cfg.get("profit_target_pct", 0.50))   # 50% of max premium
    stop_mult     = float(cfg.get("stop_loss_mult", 2.0))        # 2× premium for CSP
    today         = date.today()

    # ── Refresh live IV for all tracked positions (LEAP + CSP) ─────────────
    # Stale entry IV causes wrong stop thresholds:
    #   CSP: stop tier (2× / 3× / 4×) is chosen from live_iv vs 40/70 thresholds.
    #   LEAP: stop (40/50/60% of cost) also depends on current IV regime.
    # Batch-request all positions with one 2s sleep to minimize monitor latency.
    iv_refresh_pairs: list = []
    for _item in ib.portfolio():
        _k = _at_contract_key(_item.contract)
        if _k in at["positions"]:
            iv_refresh_pairs.append((_k, _item.contract))
    if iv_refresh_pairs:
        _iv_tks = {_k: ib.reqMktData(_c, "106", False, False)
                   for _k, _c in iv_refresh_pairs}
        await asyncio.sleep(2)
        _iv_changed = False
        for _k, _c in iv_refresh_pairs:
            _tq = _iv_tks.get(_k)
            if _tq and _tq.modelGreeks and _tq.modelGreeks.impliedVol:
                new_iv = round(float(_tq.modelGreeks.impliedVol) * 100, 1)
                old_iv = float(at["positions"][_k].get("live_iv") or 0)
                at["positions"][_k]["live_iv"] = new_iv
                if abs(new_iv - old_iv) >= 3:
                    _iv_changed = True
                    _action = at["positions"][_k].get("action", "SELL")
                    _label  = "LEAP" if _action == "BUY" else "CSP"
                    _at_log("SYSTEM",
                            f"{_k}: {_label} IV refreshed {old_iv:.0f}% → {new_iv:.0f}%")
            ib.cancelMktData(_c)
        if _iv_changed:
            _at_save_state()

    for item in ib.portfolio():
        key = _at_contract_key(item.contract)
        if key not in at["positions"]:
            continue
        info       = at["positions"][key]
        upnl       = float(item.unrealizedPNL or 0)
        max_profit = info.get("max_profit", 0)
        if max_profit <= 0:
            _at_log("WARN", f"{key}: max_profit is 0 or missing — skipping monitor (data issue?)")
            continue

        action = info.get("action", "SELL")   # SELL = CSP short put; BUY = LEAP long call

        # Compute DTE from stored expiry
        # Slice to [:8] because IBKR occasionally stores time-of-day suffix
        # (e.g. "20261218  20:00:00 EST") which breaks the %Y%m%d parse.
        dte = 0
        expiry_str = (info.get("expiry", "") or "")[:8]
        if expiry_str:
            try:
                dte = (datetime.strptime(expiry_str, "%Y%m%d").date() - today).days
            except ValueError:
                pass

        # ── 1. Profit target: 50% of max premium (CSPs only) ─────────────────────────
        # LEAPs (BUY) exit via stop-loss or 21 DTE — not a fixed profit target.
        # Applying the CSP target to LEAPs would close them at 50% of cost (e.g.
        # +$500 on a $1000 LEAP), undermining the 6-18 month directional thesis.
        if action == "SELL" and upnl >= profit_target * max_profit:
            _at_log("CLOSE", f"{key}: {profit_target*100:.0f}% profit target hit (${upnl:.0f} / max=${max_profit:.0f})")
            info["exit_reason"] = "profit_target"
            await _autotrader_close_coro(ib, item, info, key)
            continue

        # ── 2. Hard stop-loss (IV-adjusted for high-vol underlyings) ────────
        # CSP: base 2× premium; widened for high-IV options so normal vol
        #   swings don't fire the stop prematurely (NVDA/QCOM backtest showed
        #   high-vol names need more room — 5× outperformed 2× in 1-yr test).
        #   stop_loss_mult=0 disables the CSP stop entirely (hold-to-expiry).
        # LEAP: 50% of cost paid regardless of IV.
        live_iv = float(info.get("live_iv") or 0)
        if action == "SELL":
            if stop_mult <= 0:
                stop_threshold = -float("inf")   # 0=hold-to-expiry: never stop out
                eff_stop_mult  = 0.0
            elif live_iv > 70:
                eff_stop_mult = 4.0   # high IV (QCOM 108%, ON 80%) — wide stop
                stop_threshold = -eff_stop_mult * max_profit
            elif live_iv > 40:
                eff_stop_mult = 3.0   # medium-high IV (NET 61%) — moderate
                stop_threshold = -eff_stop_mult * max_profit
            else:
                eff_stop_mult = stop_mult  # normal IV — use config value (2×)
                stop_threshold = -eff_stop_mult * max_profit
        else:
            # LEAP (long call): high IV hurts long options via IV crush.
            # Tighter stop in high IV to cut before crush compounds; wider
            # stop in low IV where stock moves are cleaner signals.
            if live_iv > 70:
                eff_stop_mult = 0.40   # tight: high IV → crush risk
            elif live_iv > 40:
                eff_stop_mult = 0.50   # standard 50% stop
            else:
                eff_stop_mult = 0.60   # wider: low IV, let directional thesis develop
            stop_threshold = -eff_stop_mult * max_profit

        if upnl <= stop_threshold:
            ticker = info.get("ticker", key.split("_")[0])
            at.setdefault("stopped_out", {})[ticker] = datetime.utcnow().isoformat()
            _at_log("CLOSE",
                    f"{key}: stop-loss hit (${upnl:.0f}, {eff_stop_mult}× threshold=${stop_threshold:.0f}, IV={live_iv:.0f}%)")
            info["exit_reason"] = "stop_loss"
            await _autotrader_close_coro(ib, item, info, key)
            continue

        # ── 3. Roll at 21 DTE for CSPs not yet at profit target ──────────
        # Research (Tastytrade): gamma risk dominates in final 21 days vs remaining theta.
        # Roll: close current position, open same strike next month (30-45 DTE).
        if action == "SELL" and 0 < dte <= 21:
            if upnl >= 0:
                # Any profit at 21 DTE — take it, free the slot
                _at_log("CLOSE", f"{key}: 21 DTE — taking ${upnl:.0f} profit (gamma risk zone)")
                info["exit_reason"] = "21dte"
                await _autotrader_close_coro(ib, item, info, key)
            else:
                # At a loss but not yet at stop — roll to next month
                _at_log("ROLL", f"{key}: 21 DTE, loss=${upnl:.0f} — rolling to next cycle")
                await _autotrader_roll_coro(ib, item, info, key)
            continue

        # ── 4. Close LEAP at 21 DTE (no roll — long calls don't roll like short puts) ──
        # At 21 DTE a 12-month LEAP has shed most of its time value.  Holding through
        # expiry risks exercise mechanics, pin risk, and assignment friction.  Close now
        # to lock in whatever remains — whether a gain or a managed loss — and free the
        # slot for a fresh entry.
        if action == "BUY" and 0 < dte <= 21:
            if upnl >= 0:
                _at_log("CLOSE",
                        f"{key}: LEAP 21 DTE — closing with ${upnl:+.0f} gain "
                        f"(time value largely exhausted; avoiding exercise risk)")
            else:
                _at_log("CLOSE",
                        f"{key}: LEAP 21 DTE — closing with ${upnl:.0f} loss "
                        f"(stop not triggered; exiting before expiry to avoid worthless expiry)")
            info["exit_reason"] = "21dte"
            await _autotrader_close_coro(ib, item, info, key)
            continue

    # ── Orphaned expired position cleanup ─────────────────────────────────────
    # Options expire and disappear from ib.portfolio() on settlement day.
    # Without cleanup, expired positions stay in at["positions"] forever —
    # blocking re-entry on that ticker and distorting slot/capital accounting.
    live_keys = {_at_contract_key(item.contract) for item in ib.portfolio()}
    orphans = []
    for key, info in list(at["positions"].items()):
        expiry_str = (info.get("expiry", "") or "")[:8]
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y%m%d").date()
        except ValueError:
            continue
        if expiry_date < today and key not in live_keys:
            orphans.append(key)

    for key in orphans:
        info = at["positions"].pop(key, {})
        j_id = info.get("journal_id")
        if j_id:
            # Close the journal row so it doesn't show as permanently "open".
            # Outcome (worthless vs assigned) is unknown here — mark as orphaned
            # so it is excluded from Kelly and stats but visible in the journal.
            _journal_record_orphaned(j_id)
        _at_log("WARN",
                f"{key}: expired position removed from tracking "
                f"(expiry={info.get('expiry','?')}, "
                f"ticker={info.get('ticker','?')}, "
                f"strike={info.get('strike','?')}). "
                f"Journal entry #{j_id} closed as 'orphaned'. "
                f"Check TWS for settlement outcome.")
    if orphans:
        _at_save_state()


async def _autotrader_roll_coro(ib: IB, item, info: dict, key: str) -> None:
    """
    Roll a CSP at 21 DTE to the next monthly cycle.
    Three guards before committing:
      1. Roll count ≤ 2  — caps repeated rolls on the same losing thesis
      2. Net credit ≥ $0.10/share — new premium must exceed buyback cost
      3. Strike selection — roll DOWN if stock within 7% of strike (gamma risk zone)
    Falls back to a plain close if any guard fails.
    """
    ticker      = info.get("ticker", item.contract.symbol)
    orig_strike = float(info.get("strike", item.contract.strike))
    right       = info.get("right", "P")
    roll_count  = info.get("roll_count", 0)

    # ── Guard 1: max 2 rolls per position ────────────────────────────────────
    if roll_count >= 2:
        _at_log("ROLL", f"{ticker}: max 2 rolls reached — closing and adding to cooldown")
        info["exit_reason"] = "roll_max"
        state["autotrader"].setdefault("stopped_out", {})[ticker] = datetime.utcnow().isoformat()
        await _autotrader_close_coro(ib, item, info, key)
        return

    # ── Guard 2: earnings proximity — never roll through an earnings event ────
    # Rolling into a new expiry that straddles earnings exposes the position to
    # assignment risk + IV crush on the wrong side. Close now; re-enter after.
    earnings_days: Optional[int] = None   # captured here so it's available for the roll row below
    try:
        earnings_days = await _earnings_days_out(ticker)
        if earnings_days is not None and earnings_days <= EARNINGS_BLOCK_DAYS:
            _at_log("ROLL",
                    f"{ticker}: earnings in {earnings_days}d — closing instead of rolling "
                    f"(block={EARNINGS_BLOCK_DAYS}d, cooldown applied)")
            info["exit_reason"] = "roll_close"
            state["autotrader"].setdefault("stopped_out", {})[ticker] = datetime.utcnow().isoformat()
            await _autotrader_close_coro(ib, item, info, key)
            return
    except Exception as _earn_exc:
        _at_log("WARN", f"{ticker}: earnings check failed at roll time ({_earn_exc}) — proceeding")

    # ── Step 1: get live buyback price (current ask = what we pay to close) ──
    c = item.contract
    contract_cur = Option(
        symbol=c.symbol,
        lastTradeDateOrContractMonth=c.lastTradeDateOrContractMonth,
        strike=float(c.strike), right=c.right,
        exchange="SMART", currency="USD", multiplier="100",
    )
    await ib.qualifyContractsAsync(contract_cur)
    tq_cur = ib.reqMktData(contract_cur, "106", False, False)  # 106 = implied vol for modelGreeks
    await asyncio.sleep(2)
    buyback_ask = _safe_float(tq_cur.ask, 0)
    live_iv_roll = None
    try:
        if tq_cur.modelGreeks and tq_cur.modelGreeks.impliedVol:
            live_iv_roll = round(float(tq_cur.modelGreeks.impliedVol) * 100, 2)
    except Exception:
        pass
    live_iv_roll = live_iv_roll or info.get("live_iv")  # fall back to entry IV if snapshot unavailable
    ib.cancelMktData(contract_cur)

    if buyback_ask <= 0:
        _at_log("ROLL", f"{ticker}: no buyback price — closing")
        info["exit_reason"] = "roll_close"
        await _autotrader_close_coro(ib, item, info, key)
        return

    # ── Step 2: fetch next-month chain + pick target strike ──────────────────
    try:
        stock_price = await _get_stock_price(ib, ticker)
        if stock_price <= 0:
            _at_log("ROLL", f"{ticker}: no stock price — closing")
            info["exit_reason"] = "roll_close"
            await _autotrader_close_coro(ib, item, info, key)
            return

        # Roll-down: if strike is within 7% OTM, target ~12% OTM for safety
        otm_pct = (stock_price - orig_strike) / stock_price * 100
        if otm_pct < 7:
            target_strike = round(stock_price * 0.88, 0)
            strike_note   = f"roll-DOWN (was {otm_pct:.1f}% OTM → ~12% OTM)"
        else:
            target_strike = orig_strike
            strike_note   = f"same strike ({otm_pct:.1f}% OTM)"

        tds, expiry_new, dte_new = await _fetch_opra_chain(
            ib, ticker, right, stock_price, 30, 45,
            otm_lo_pct=0, otm_hi_pct=40, max_strikes=15,
        )
        valid = [td for td in (tds or [])
                 if _safe_float(td.contract.strike) is not None
                 and float(td.contract.strike) < stock_price   # must be OTM put
                 and _safe_float(td.bid, 0) > 0.20
                 and _safe_float(td.ask, 0) > 0]
        if not valid:
            _at_log("ROLL", f"{ticker}: no liquid OTM contracts next month — closing")
            info["exit_reason"] = "roll_close"
            await _autotrader_close_coro(ib, item, info, key)
            return

        # Guard: never roll UP — the new strike must not exceed the original.
        # Closest-to-target can overshoot above orig_strike if that expiry
        # only has higher strikes available, which increases assignment risk.
        safe_valid = [td for td in valid if float(td.contract.strike) <= orig_strike]
        if not safe_valid:
            _at_log("ROLL", f"{ticker}: no roll strikes at or below original ${orig_strike} — closing")
            info["exit_reason"] = "roll_close"
            await _autotrader_close_coro(ib, item, info, key)
            return
        best        = min(safe_valid, key=lambda td: abs(float(td.contract.strike) - target_strike))
        roll_strike = float(best.contract.strike)
        new_bid     = _safe_float(best.bid, 0)
        new_ask     = _safe_float(best.ask, 0)
        new_mid     = (new_bid + new_ask) / 2
        new_fill    = round(new_bid + (new_mid - new_bid) * 0.40, 2)  # our expected fill

        # ── Guard 2: net credit check ─────────────────────────────────────────
        # Research minimum: $0.30/share to justify roll complexity + commissions
        net_credit = round(new_fill - buyback_ask, 2)
        if net_credit < 0.30:
            _at_log("ROLL",
                    f"{ticker}: roll credit ${net_credit:.2f}/sh below $0.30 minimum "
                    f"(buyback=${buyback_ask:.2f}, new=${new_fill:.2f}) — closing + cooldown")
            info["exit_reason"] = "roll_no_credit"
            state["autotrader"].setdefault("stopped_out", {})[ticker] = datetime.utcnow().isoformat()
            await _autotrader_close_coro(ib, item, info, key)
            return

        # ── All guards passed — execute roll ──────────────────────────────────
        _at_log("ROLL",
                f"{ticker}: {strike_note} | "
                f"buyback=${buyback_ask:.2f} new=${new_fill:.2f} net=+${net_credit:.2f}/sh "
                f"| roll #{roll_count + 1}/2")
        info["exit_reason"] = "roll_close"
        await _autotrader_close_coro(ib, item, info, key)

        # Guard: if close_coro returned early (e.g. no market data at close time),
        # the old position is still in at["positions"].  Opening the new leg would
        # create two tracking entries for the same ticker — double-counting capital.
        if key in state["autotrader"]["positions"]:
            _at_log("ROLL",
                    f"{ticker}: roll aborted — close_coro did not remove old position "
                    f"(likely no market data). Old position still tracked; will retry roll "
                    f"on next monitor cycle when market data is available.")
            return

        row = {
            "ticker":           ticker,       "expiry":      expiry_new,
            "strike":           roll_strike,  "_type":       info.get("strategy_type", "csp"),
            "_regime":          state["ext_cache"].get("regime", {}).get("regime", "BULL"),
            "bid":              new_bid,      "ask":         new_ask,
            "score":            info.get("score", 0),
            "iv_rank":          info.get("iv_rank"),
            "live_iv":          live_iv_roll,  # IV at time of roll — critical for learning model
            "roll_count":       roll_count + 1,
            "spot":             stock_price,    # available from _get_stock_price above
            "earnings_days_out": earnings_days, # available from _earnings_days_out above
        }
        await _autotrader_place_coro(ib, row, state["autotrader"]["config"], regime=row["_regime"])
        _at_log("ROLL",
                f"{ticker}: rolled → {expiry_new} P{roll_strike} "
                f"({dte_new} DTE, net +${net_credit*100:.0f}/contract)")

        # ── Plain-English roll rationale ──────────────────────────────────────
        orig_strike = float(info.get("strike", 0))
        strike_note_eng = (
            f"The strike was moved down from ${orig_strike} to ${roll_strike} "
            f"because the stock declined toward the original strike."
            if roll_strike < orig_strike else
            f"The same ${roll_strike} strike was kept since the stock remains safely above it."
        )
        _decision_log(
            "ROLL", ticker,
            f"{ticker} ${orig_strike} Put rolled → ${roll_strike} Put {expiry_new} (+${net_credit*100:.0f}/contract)",
            f"**Why we rolled:** The position reached {dte_new + 21} days to expiration (21 DTE "
            f"is our threshold). Rather than closing for a small loss or a partial profit, rolling "
            f"extends the trade to a new expiry ({expiry_new}, {dte_new} DTE) and collects an "
            f"additional net credit of ${net_credit:.2f}/share (${net_credit*100:.0f}/contract). "
            f"This is roll #{roll_count + 1} of 2 maximum.\n\n"
            f"**Strike selection:** {strike_note_eng}\n\n"
            f"**Economics:** We bought back the old put at ${buyback_ask:.2f} and sold the new "
            f"put at ${new_fill:.2f}, collecting a net ${net_credit:.2f}/share credit. This credit "
            f"reduces our cost basis and improves the overall trade's break-even point."
        )

    except Exception as exc:
        _at_log("ROLL", f"{ticker}: roll failed ({exc}) — attempting plain close")
        try:
            info["exit_reason"] = "roll_close"
            await _autotrader_close_coro(ib, item, info, key)
        except Exception:
            pass


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

    # Guard: if IBKR returns no market data at all, do NOT place the order.
    # A hardcoded fallback price would produce an unfillable limit order, and
    # unconditionally removing the position from tracking would orphan it —
    # auto-trader would never manage it again.  Defer to the next monitor cycle.
    if bid <= 0 and ask <= 0:
        _at_log("WARN",
                f"{key}: close skipped — no bid/ask data (will retry next cycle). "
                f"Check IBKR market data subscription for {c.symbol}.")
        return

    if close_action == "BUY":
        lmt = round((ask + 0.01) if ask > 0 else bid * 1.10, 2)
    else:
        lmt = round((bid - 0.01) if bid > 0 else ask * 0.90, 2)

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

    # Use limit price to compute realized P&L — more accurate than the mark-based
    # unrealizedPNL which can overstate LEAP profits by ~$100 on wide spreads.
    entry_px = float(info.get("entry_price", 0) or 0)
    if entry_px > 0:
        if close_action == "SELL":   # closing a long (LEAP): sell at lmt
            upnl_now = round((lmt - entry_px) * qty * 100, 2)
        else:                        # closing a short (CSP): buy back at lmt
            upnl_now = round((entry_px - lmt) * qty * 100, 2)
    else:
        upnl_now = float(item.unrealizedPNL or 0)

    # Record exit in trade journal
    j_id = info.get("journal_id")
    if j_id:
        # Determine exit reason from context (info carries last reason from monitor)
        exit_reason = info.get("exit_reason", "manual")
        _journal_record_exit(j_id, lmt, upnl_now, exit_reason, live_iv_exit)

    # Track premium collected / LEAP P&L for the LEAP budget fund.
    # Budget = 50% of cumulative CSP wins PLUS all realised LEAP P&L (wins net losses).
    at = state["autotrader"]
    if info.get("action") == "SELL" and upnl_now > 0:
        at["premium_collected"] = round(at.get("premium_collected", 0.0) + upnl_now, 2)
        _at_log("BUDGET", f"CSP +${upnl_now:.0f} → premium_collected=${at['premium_collected']:.0f}")
    elif info.get("action") == "BUY":
        at["leap_pnl"] = round(at.get("leap_pnl", 0.0) + upnl_now, 2)
        _at_log("BUDGET", f"LEAP ${upnl_now:+.0f} → cumulative_leap_pnl=${at['leap_pnl']:.0f}")
    at["leap_budget"] = max(0.0, round(at["premium_collected"] * 0.50 + at.get("leap_pnl", 0.0), 2))
    _at_log("BUDGET", f"leap_budget=${at['leap_budget']:.0f} "
                      f"(50%×${at['premium_collected']:.0f} CSP + ${at.get('leap_pnl',0):.0f} LEAP pnl)")
    _at_log("CLOSED", f"{close_action} {qty}x {c.symbol} @ ${lmt:.2f} "
                      f"pnl=${upnl_now:+.0f} (order #{trade.order.orderId})")

    # ── Plain-English close rationale ─────────────────────────────────────────
    exit_reason = info.get("exit_reason", "manual")
    ticker_sym  = c.symbol
    strike_cl   = float(c.strike)
    expiry_cl   = (c.lastTradeDateOrContractMonth or "")[:8]
    mp          = info.get("max_profit", lmt * 100)
    ep          = info.get("entry_price", lmt)
    held_days   = ""
    try:
        from zoneinfo import ZoneInfo as _ZI
        import re as _re
        placed = info.get("placed_at", "")
        if placed and _re.search(r"\d{4}-\d{2}-\d{2}", placed):
            from datetime import date as _date
            d0 = _date.fromisoformat(_re.search(r"\d{4}-\d{2}-\d{2}", placed).group())
            held_days = f" Held {(datetime.now(_ZI('America/New_York')).date()-d0).days} days."
    except Exception:
        pass

    _reason_map = {
        "profit_target": ("CLOSE_PROFIT", "Profit Target Reached ✓",
            f"The position gained ${upnl_now:+.0f}, which equals "
            f"{round(upnl_now/mp*100) if mp else '?'}% of the ${mp:.0f} max profit. "
            f"Research shows closing at 50% of max profit captures most of the time-decay "
            f"benefit while freeing capital for fresh opportunities.{held_days}"),
        "stop_loss": ("CLOSE_STOP", "Stop-Loss Triggered ⚠",
            f"The position lost ${abs(upnl_now):.0f}, exceeding the stop-loss threshold. "
            f"{ticker_sym} moved adversely, pushing the option closer to in-the-money. "
            f"The stop-loss prevents larger losses if the move continues. "
            f"{ticker_sym} will be on a 48-hour cooldown before re-entry.{held_days}"),
        "21dte": ("CLOSE_21DTE", "21 Days to Expiration — Theta Risk Rising",
            f"The position reached 21 days until expiration. At this point, gamma risk "
            f"(sensitivity to price moves) accelerates sharply. Closing now locks in "
            f"${upnl_now:+.0f} P&L and avoids the risk of a late adverse move.{held_days}"),
        "roll_close": ("CLOSE_ROLL", "Closed Before Rolling",
            f"Position closed as part of a roll sequence or because earnings were within "
            f"14 days at roll time. Closing first allows re-entry at a better strike/expiry "
            f"without earnings risk.{held_days}"),
        "roll_max": ("CLOSE_ROLL", "Max Rolls Reached — Closing",
            f"This position reached the maximum number of rolls allowed. "
            f"Closing now with ${upnl_now:+.0f} P&L rather than extending further.{held_days}"),
        "roll_no_credit": ("CLOSE_ROLL", "No Credit Available at Roll — Closing",
            f"When attempting to roll, no expiry offered a net credit. "
            f"Closing with ${upnl_now:+.0f} P&L instead of locking in a debit.{held_days}"),
        "rotation": ("CLOSE_ROTATE", "Rotated Out — Better Opportunity Found",
            f"A higher-scoring candidate displaced this position. "
            f"Closing with ${upnl_now:+.0f} P&L to redeploy capital.{held_days}"),
        "manual": ("CLOSE_MANUAL", "Manually Closed",
            f"Position closed by manual request with ${upnl_now:+.0f} P&L.{held_days}"),
    }
    _act, _headline_suffix, _body_detail = _reason_map.get(
        exit_reason,
        ("CLOSE_MANUAL", exit_reason.replace("_", " ").title(),
         f"Position closed ({exit_reason}) with ${upnl_now:+.0f} P&L.{held_days}")
    )
    _headline = f"{ticker_sym} ${strike_cl} {'Put' if info.get('action')=='SELL' else 'Call'} — {_headline_suffix}"
    _body = (
        f"**Contract:** {ticker_sym} ${strike_cl} expiry {expiry_cl}, "
        f"entered at ${ep:.2f}, closed at ${lmt:.2f}.\n\n"
        f"**Result:** {_body_detail}"
    )
    _decision_log(_act, ticker_sym, _headline, _body)
    _at_save_state()


def _capital_state(ib: IB, cfg: dict, at: dict) -> dict:
    """Return capital headroom for new positions.

    Uses three layers:
      1. IBKR live AvailableFunds — hard floor (never deploy what we don't have)
      2. csp_capital config — per-strategy allocation ceiling
      3. CAPITAL_BUFFER_PCT reserve — dry powder for adverse moves + opportunities

    Returns a dict with:
      allocated      : csp_capital from config
      consumed       : sum of (strike × qty × 100) for all open CSP positions
      deployable     : allocated × (1 - buffer) - consumed   (capped at IBKR available)
      ibkr_available : raw AvailableFunds from IBKR (None if not connected)
      buffer_held    : dollar amount reserved as buffer
    """
    CAPITAL_BUFFER_PCT = 0.20   # keep 20% of CSP allocation as dry powder

    csp_capital   = float(cfg.get("csp_capital", 20000.0))
    total_capital = float(cfg.get("total_capital", 100000.0))
    # csp_capital is the deliberate control knob — it's what you've decided to
    # allocate to the CSP strategy regardless of total portfolio size.  Honor it
    # exactly.  Only warn (never auto-override) when it appears to be a stale
    # default: less than $25K AND total_capital is at least 5× larger.
    allocated = csp_capital
    if csp_capital <= 25000 and total_capital >= csp_capital * 5:
        _at_log("WARN",
                f"csp_capital is ${csp_capital:,.0f} — this controls total CSP deployment, "
                f"not total_capital (${total_capital:,.0f}). "
                f"Set csp_capital in Config to your intended CSP budget (e.g. $75K-$150K).")
    buffer_amt = allocated * CAPITAL_BUFFER_PCT
    max_deploy = allocated - buffer_amt   # 80% of allocation

    # Consumed = notional margin tied up in open CSP positions
    consumed = 0.0
    for info in at["positions"].values():
        if info.get("action") == "SELL":   # CSP
            consumed += float(info.get("strike", 0)) * float(info.get("qty", 1)) * 100

    deployable = max(0.0, max_deploy - consumed)

    # IBKR live floor
    ibkr_avail = None
    try:
        acct_vals = {v.tag: float(v.value) for v in ib.accountValues()
                     if v.currency == "USD" and v.tag in ("AvailableFunds", "BuyingPower")}
        ibkr_avail = acct_vals.get("AvailableFunds")
        if ibkr_avail is not None:
            deployable = min(deployable, ibkr_avail)
    except Exception:
        pass   # not connected or account values unavailable — use config-based limit only

    return {
        "allocated":      allocated,
        "consumed":       consumed,
        "deployable":     deployable,
        "buffer_held":    buffer_amt,
        "ibkr_available": ibkr_avail,
    }


def _position_remaining_value(info: dict, upnl: float, max_profit: float,
                               dte: int, profit_target: float) -> float:
    """Score an open position on how much value it still has to extract (0–100).

    Lower score = weaker hold = candidate for rotation.
    Factors: remaining upside to target × DTE time value × original entry quality.
    """
    if max_profit <= 0:
        return 0.0
    pnl_pct          = upnl / max_profit
    remaining_upside  = max(0.0, profit_target - pnl_pct)   # fraction of target still uncaptured
    upside_score      = remaining_upside / profit_target      # 0–1

    if dte <= 0:
        dte_factor = 0.05
    elif dte <= 7:
        dte_factor = 0.25
    elif dte <= 14:
        dte_factor = 0.50
    elif dte <= 21:
        dte_factor = 0.70
    else:
        dte_factor = 1.00

    entry_quality = float(info.get("score", 50)) / 100.0
    return round(upside_score * dte_factor * entry_quality * 100, 1)


def _find_rotation_target(at: dict, portfolio_items: list, candidates: list,
                           cfg: dict) -> Optional[tuple]:
    """Find a (new_candidate, key_to_close) rotation pair.

    Rules:
      - new_candidate.score must be ≥ 1.25× the remaining-value score of the worst position
      - Only close positions that are profitable OR within 21 DTE (never lock in a loss to rotate)
      - Never rotate the same ticker (pointless churn)

    Returns (candidate_row, position_key) or None.
    """
    profit_target = float(cfg.get("profit_target_pct", 0.50))
    ROTATION_THRESHOLD = 1.25   # candidate must score 25% better than incumbent

    # Build a score for each open position
    scored_positions = []
    item_by_key = {_at_contract_key(i.contract): i for i in portfolio_items}

    today = date.today()
    for key, info in at["positions"].items():
        item = item_by_key.get(key)
        if item is None:
            continue
        upnl       = float(item.unrealizedPNL or 0)
        max_profit = float(info.get("max_profit", 0))
        # Recalculate current DTE from expiry (stored DTE is at-entry and stale after days pass)
        expiry_str = (info.get("expiry", "") or "")[:8]
        try:
            dte = max(0, (datetime.strptime(expiry_str, "%Y%m%d").date() - today).days)
        except (ValueError, TypeError):
            dte = int(info.get("dte") or 45)   # fallback only if expiry unparseable

        rv_score   = _position_remaining_value(info, upnl, max_profit, dte, profit_target)
        # LEAP (BUY) positions are long-term strategic holds managed by their own
        # stop/profit/21-DTE exits.  Never rotate out of a LEAP — the thesis needs
        # months to develop and short-term score comparisons are meaningless against it.
        is_leap = info.get("action", "SELL") == "BUY"
        is_closeable = (not is_leap) and (upnl >= 0 or dte <= 21)

        scored_positions.append({
            "key":          key,
            "ticker":       info.get("ticker", ""),
            "rv_score":     rv_score,
            "upnl":         upnl,
            "dte":          dte,
            "is_closeable": is_closeable,
        })

    closeable = [p for p in scored_positions if p["is_closeable"]]
    if not closeable:
        return None

    worst = min(closeable, key=lambda p: p["rv_score"])
    active_tickers = {p["ticker"] for p in scored_positions}

    for candidate in candidates:
        if candidate["ticker"] in active_tickers:
            continue   # already hold this ticker
        candidate_score = float(candidate.get("score", 0))
        if candidate_score >= worst["rv_score"] * ROTATION_THRESHOLD:
            return (candidate, worst["key"])

    return None


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

    # ── Active ticker dedup ───────────────────────────────────────────────────
    # Use the positions DICT (not live portfolio) so Inactive orders still
    # consume slots — prevents duplicate orders that are accepted by TWS but
    # not visible in ib.portfolio() until exchange transmission.
    dict_t   = {k.split("_")[0] for k in at["positions"]}
    portf_t  = {item.contract.symbol for item in ib.portfolio()}
    active_t = dict_t | portf_t

    # ── Capital-aware headroom ────────────────────────────────────────────────
    # Dual constraint: count headroom (sanity cap) AND capital headroom.
    # Regime conservatism is handled entirely through Kelly's regime_scale
    # (BULL=1.0, NEUTRAL=0.6, BEAR=0.35) — which produces smaller positions
    # that consume less capital, naturally limiting deployment in weak regimes.
    # A separate count cap on top of Kelly is a double-penalty: NEUTRAL at 60%
    # Kelly with max_positions=5 would have been capped at 2 positions even with
    # $40K deployable and 5 quality candidates — defeating capital-aware sizing.
    max_slots   = cfg["max_positions"]
    cap         = _capital_state(ib, cfg, at)
    count_slots = max_slots - len(at["positions"])
    # Estimate the cheapest likely CSP (use $100 strike as conservative floor)
    min_csp_cost = 100 * 100   # $10,000 for a $100-strike single-contract CSP
    capital_slots = int(cap["deployable"] / min_csp_cost) if cap["deployable"] > 0 else 0
    csp_slots = min(count_slots, capital_slots)

    # LEAP candidates use a separate budget (leap_capital / leap_budget) and must NOT
    # be gated by CSP capital exhaustion.  Compute LEAP headroom independently so a
    # depleted CSP pool doesn't zero out `slots` and prevent LEAPs from reaching the
    # per-candidate check that was added in the previous fix.
    leap_deployed = sum(float(p.get("max_profit", 0)) for p in at["positions"].values()
                        if p.get("action") == "BUY")
    leap_capital_cfg = float(cfg.get("leap_capital", 5000.0))
    leap_budget_val  = at.get("leap_budget", 0.0)
    leap_avail_now   = max(0.0, max(leap_capital_cfg, leap_budget_val) - leap_deployed)
    # Conservative floor: assume cheapest LEAP is ~$1K/contract (low-price stocks)
    leap_capital_slots = min(count_slots, int(leap_avail_now / 1000)) if leap_avail_now > 0 else 0
    # Total effective slots: CSP + LEAP are additive (separate budgets, shared count cap)
    slots = min(count_slots, csp_slots + leap_capital_slots)

    _at_log("SCAN",
            f"Capital: ${cap['consumed']:,.0f} CSP deployed / ${cap['allocated']:,.0f} allocated "
            f"| ${cap['deployable']:,.0f} CSP deployable | LEAP avail=${leap_avail_now:,.0f}"
            + (f" | IBKR available=${cap['ibkr_available']:,.0f}" if cap["ibkr_available"] else ""))

    # ── Always scan — needed for rotation decisions even at full capacity ─────
    _at_log("SCAN", f"Scanning market (count_slots={count_slots}, csp_slots={csp_slots}, "
                    f"leap_slots={leap_capital_slots}, effective_slots={slots}, regime={regime})…")

    candidates: list = []
    if "csp" in cfg["scan_types"]:
        try:
            r = await scan_csp(ib, CSP_MIN_RETURN_PCT, CSP_MAX_DELTA)
            for c in _filter_csp_recommended(r["candidates"], log_diag=True):
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

    # 48h stop-loss cooldown — don't re-enter tickers recently stopped out
    stopped_out = at.get("stopped_out", {})
    _now_utc    = datetime.utcnow()
    def _in_cooldown(ticker: str) -> bool:
        ts = stopped_out.get(ticker)
        if not ts:
            return False
        try:
            return (_now_utc - datetime.fromisoformat(ts)).total_seconds() < 48 * 3600
        except ValueError:
            return False

    cooled = [c["ticker"] for c in candidates if _in_cooldown(c["ticker"])]
    if cooled:
        _at_log("SCAN", f"48h cooldown — skipping: {cooled}")
    candidates = [c for c in candidates
                  if c["ticker"] not in active_t and not _in_cooldown(c["ticker"])]
    _at_log("SCAN", f"Found {len(candidates)} qualifying candidates after dedup + cooldown")

    placed = 0
    if slots > 0:
        # ── Normal entry path ─────────────────────────────────────────────────
        for row in candidates[:slots]:
            if row.get("_type") == "leap":
                # LEAP cost = option premium paid (cost_per_contract), NOT strike × 100.
                # Checked against LEAP-specific budget (leap_capital or self-financed
                # leap_budget), not the CSP collateral pool.
                cost = float(row.get("cost_per_contract", 0))
                leap_deployed = sum(
                    float(p.get("max_profit", 0))
                    for p in at["positions"].values()
                    if p.get("action") == "BUY"
                )
                leap_capital = float(cfg.get("leap_capital", 5000.0))
                leap_budget  = at.get("leap_budget", 0.0)
                leap_avail   = max(0.0, max(leap_capital, leap_budget) - leap_deployed)
                if cost > leap_avail:
                    _at_log("SCAN", f"Skip {row['ticker']} LEAP: cost ${cost:,.0f} > "
                                    f"LEAP available ${leap_avail:,.0f} "
                                    f"(deployed=${leap_deployed:,.0f}, "
                                    f"budget=${leap_budget:,.0f}, capital=${leap_capital:,.0f})")
                    continue
            else:
                cost = float(row.get("strike", 0)) * float(row.get("qty", 1)) * 100
                if cost > cap["deployable"]:
                    _at_log("SCAN", f"Skip {row['ticker']}: position cost ${cost:,.0f} > "
                                    f"deployable ${cap['deployable']:,.0f}")
                    continue
            try:
                await _autotrader_place_coro(ib, row, cfg, regime=regime)
                active_t.add(row["ticker"])
                if row.get("_type") != "leap":
                    cap["deployable"] -= cost   # only deduct CSP capital for CSP trades
                placed += 1
            except Exception as exc:
                _at_log("ERROR", f"Place {row['ticker']}: {exc}")
    elif candidates:
        # ── At full capacity — check for rotation opportunity ─────────────────
        rotation = _find_rotation_target(at, list(ib.portfolio()), candidates, cfg)
        if rotation:
            new_row, worst_key = rotation
            worst_info = at["positions"].get(worst_key, {})
            worst_item = next(
                (i for i in ib.portfolio() if _at_contract_key(i.contract) == worst_key), None
            )
            if worst_item:
                worst_info["exit_reason"] = "rotation"
                _at_log("ROTATE",
                        f"Closing {worst_key} (rv_score low, upnl=${float(worst_item.unrealizedPNL or 0):.0f}) "
                        f"→ opening {new_row['ticker']} (score={new_row['score']:.0f})")
                await _autotrader_close_coro(ib, worst_item, worst_info, worst_key)
                try:
                    await _autotrader_place_coro(ib, new_row, cfg, regime=regime)
                    placed += 1
                except Exception as exc:
                    _at_log("ERROR", f"Rotation place {new_row['ticker']}: {exc}")
            else:
                _at_log("ROTATE", f"Rotation skipped: {worst_key} not found in live portfolio")
        else:
            _at_log("SCAN",
                    f"At capacity ({len(at['positions'])}/{max_slots}, "
                    f"deployed=${cap['consumed']:,.0f}/${cap['allocated']:,.0f}) "
                    f"— no rotation opportunity (best candidate not ≥25% better than weakest position)")
    else:
        _at_log("SCAN",
                f"At capacity ({len(at['positions'])}/{max_slots}, "
                f"deployed=${cap['consumed']:,.0f}/${cap['allocated']:,.0f}) — no new candidates found")

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
        # 100=option volume, 101=option OI, 106=implied vol
        # tick 13 (equity OI) is NOT valid for OPT and causes error 321
        tq = ib.reqMktData(contract, "106,100,101", False, False)
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
                        "Order will NOT be tracked to avoid blocking ticker re-entry. "
                        "Check TWS permissions or margin requirements.")
        return   # do not track or journal an unfilled Inactive order

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

    from zoneinfo import ZoneInfo
    state["autotrader"]["positions"][key] = {
        "ticker":      ticker,    "expiry":    expiry,
        "strike":      strike,    "right":     right,
        "action":      action,    "qty":       qty,
        "entry_price": lmt,       "max_profit":round(max_profit, 2),
        "order_id":    trade.order.orderId,
        "placed_at":   datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET"),
        "score":       row.get("score", 0),
        "journal_id":  journal_id,
        "live_iv":     live_iv_entry,
        "roll_count":  row.get("roll_count", 0),
    }
    _at_log("TRADE", f"{action} {qty}x {ticker} {right}{strike} {expiry} @ ${lmt:.2f} "
                     f"[price_src={lmt_src}] (order #{trade.order.orderId}, status={order_status}, journal #{journal_id})")

    # ── Plain-English decision rationale ──────────────────────────────────────
    iv_rank    = row.get("iv_rank") or 0
    wkly_ret   = row.get("weekly_return_pct") or 0
    delta_val  = row.get("delta") or 0
    earn_days  = row.get("earnings_days_out")
    spot_px    = row.get("stock_price") or row.get("spot") or 0
    otm_pct    = round((float(spot_px) - strike) / float(spot_px) * 100, 1) if spot_px and t == "csp" else 0
    _cfg_stop  = float(cfg.get("stop_loss_mult", 2.0))
    if t == "csp":
        stop_mult = 4.0 if (live_iv_entry or 0) > 70 else 3.0 if (live_iv_entry or 0) > 40 else _cfg_stop
        stop_loss = round(stop_mult * max_profit, 0) if _cfg_stop > 0 else None
    else:  # LEAP: IV-adjusted % of cost (matches monitor logic at _autotrader_monitor_coro)
        leap_stop_pct = 0.40 if (live_iv_entry or 0) > 70 else 0.50 if (live_iv_entry or 0) > 40 else 0.60
        stop_mult  = leap_stop_pct
        stop_loss  = round(leap_stop_pct * max_profit, 0)
    earn_note  = f"Earnings are {earn_days} days away — well outside the 14-day block window." if earn_days else "No upcoming earnings detected."

    if t == "csp":
        headline = f"Sold {qty}× {ticker} ${strike} Put — {dte} DTE, collecting ${round(lmt*100*qty,0):.0f} premium"
        body = (
            f"**Why we entered:** {ticker}'s implied volatility (IV) rank is {iv_rank:.0f}% — "
            f"{'well above' if iv_rank >= 60 else 'above'} the 50% threshold, meaning options are "
            f"unusually expensive right now, which is ideal for selling. The stock is {otm_pct:.1f}% "
            f"below the strike price, giving a {round(100*abs(delta_val),0):.0f}% probability the "
            f"put expires worthless. At ${lmt:.2f}/share premium, this trade earns {wkly_ret:.2f}%/week "
            f"— above our 1% minimum. Market regime is {regime}.\n\n"
            f"**The contract:** Sell {qty} × {ticker} ${strike} Put expiring {expiry} ({dte} DTE) "
            f"at ${lmt:.2f}, collecting ~${round(lmt*100*qty,0):.0f} total premium.\n\n"
            f"**Risk:** Maximum profit is ${round(max_profit,0):.0f} if {ticker} stays above ${strike} "
            f"at expiration. "
            + (f"Stop-loss fires if the position loses more than ${stop_loss:.0f} ({stop_mult:.0f}× the premium collected). " if stop_loss else "No stop-loss — holding to expiry. ")
            + earn_note
        )
    else:
        headline = f"Bought {qty}× {ticker} ${strike} Call (LEAP) — {dte} DTE"
        body = (
            f"**Why we entered:** Buying a LEAP call gives long-term upside exposure to {ticker} "
            f"without owning the stock. IV rank is {iv_rank:.0f}% — "
            f"{'moderate' if iv_rank < 50 else 'elevated'} — paid ${lmt:.2f}/share for this call. "
            f"Market regime is {regime}, supported by the upward trend signals.\n\n"
            f"**The contract:** Buy {qty} × {ticker} ${strike} Call expiring {expiry} ({dte} DTE) "
            f"at ${lmt:.2f}, total cost ~${round(lmt*100*qty,0):.0f}.\n\n"
            f"**Risk:** Maximum loss is the premium paid (${round(lmt*100*qty,0):.0f}) if {ticker} "
            f"stays below ${strike}. Stop-loss triggers at {int(stop_mult*100):.0f}% loss of cost "
            f"(IV-adjusted: tighter in high-IV to limit crush risk). {earn_note}"
        )
    _decision_log("ENTER", ticker, headline, body)
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
            # Monitor runs always — checks profit targets, hard stops, and 21-DTE rolls
            await loop.run_in_executor(
                None,
                lambda: _run_in_streaming_loop(_autotrader_monitor_coro(ib), timeout=30),
            )
            # New entries only during market hours — prevents queued overnight orders
            if market_open:
                await loop.run_in_executor(
                    None,
                    lambda: _run_in_streaming_loop(_autotrader_scan_and_trade_coro(ib), timeout=270),
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
            model_version     INTEGER,
            live_iv_exit      REAL
        )
    """)
    # Add live_iv_exit column to existing databases that predate this field
    try:
        con.execute("ALTER TABLE trade_journal ADD COLUMN live_iv_exit REAL")
    except Exception:
        pass   # Column already exists — ALTER TABLE fails if column is present
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
        SET closed_at=?, exit_price=?, pnl=?, pnl_pct=?, win=?, exit_reason=?, live_iv_exit=?
        WHERE id=?
    """, (datetime.utcnow().isoformat(), exit_price, round(pnl, 2),
          pnl_pct, win, exit_reason, live_iv_exit, journal_id))
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


def _journal_record_orphaned(journal_id: int) -> None:
    """Close a journal row for an expired/disappeared position whose outcome is unknown.

    Sets closed_at and exit_reason='orphaned' so the row no longer appears as open,
    but leaves pnl/win/pnl_pct NULL so it is excluded from Kelly and stats calculations.
    The 'orphaned' reason is already filtered out of get_journal_stats() and
    _update_kelly_from_journal() via _REAL_EXIT_REASONS.
    """
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    con.execute(
        "UPDATE trade_journal SET closed_at=?, exit_reason='orphaned' WHERE id=? AND closed_at IS NULL",
        (datetime.utcnow().isoformat(), journal_id),
    )
    con.commit()
    con.close()


_REAL_EXIT_REASONS = (
    "'profit_target','stop_loss','roll_close',"
    "'roll_max','roll_no_credit','21dte','manual','rotation'"
)


def _update_kelly_from_journal() -> None:
    """EMA-blend actual win rate from real autotrader exits into assumed_win_rate.

    Only counts trades closed by legitimate exit reasons. Excludes 'orphaned'
    entries created by cleanup — those aren't real trade outcomes and would
    collapse Kelly to zero. Requires at least 10 real trades before updating.
    """
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    rows = con.execute(
        f"SELECT win FROM trade_journal "
        f"WHERE closed_at IS NOT NULL AND win IS NOT NULL "
        f"  AND exit_reason IN ({_REAL_EXIT_REASONS})"
    ).fetchall()
    con.close()
    if len(rows) < 10:
        # Not enough real data yet — keep the prior unchanged
        return
    actual_wr = sum(r[0] for r in rows) / len(rows)
    old       = state["autotrader"]["config"].get("assumed_win_rate", 0.85)
    new_wr    = round(0.70 * old + 0.30 * actual_wr, 4)  # slow EMA blend
    state["autotrader"]["config"]["assumed_win_rate"] = new_wr
    _at_log("LEARN",
            f"Kelly win rate {old:.1%} → {new_wr:.1%} "
            f"(actual {actual_wr:.1%} over {len(rows)} real trades)")

def _retrain_from_journal() -> dict:
    """Retrain XGBoost score model from completed journal trades.

    Only trains on real managed exits — excludes orphaned/trailing_stop rows
    (all win=0) that would bias the model toward predicting losses.
    Same whitelist used by _update_kelly_from_journal().
    """
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    cols_sql = ", ".join(_JOURNAL_FEATURE_COLS) + ", win"
    rows = con.execute(
        f"SELECT {cols_sql} FROM trade_journal "
        f"WHERE closed_at IS NOT NULL AND win IS NOT NULL "
        f"AND exit_reason IN ({_REAL_EXIT_REASONS})"
    ).fetchall()
    con.close()
    if len(rows) < JOURNAL_MIN_TRADES:
        return {"error": f"Need {JOURNAL_MIN_TRADES}+ real exits (have {len(rows)})"}

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
                        _fetch_opra_chain(ib, ticker, "P", stock_price, 30, 50),
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
                            "above_sma200":          tech.get("above_sma200"),
                            "pct_b":                 tech.get("pct_b"),
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

    async def _scan_ticker_safe_csp(t: str) -> List[dict]:
        try:
            return await asyncio.wait_for(_scan_ticker(t), timeout=25)
        except asyncio.TimeoutError:
            log.warning("CSP [%s]: per-ticker scan timed out (>25s) — skipping", t)
            return []

    regime, *ticker_results = await asyncio.gather(
        _market_regime(),
        *[_scan_ticker_safe_csp(t) for t in CSP_UNIVERSE],
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
      • Delta 0.65–0.85 (deep ITM — matches LEAP_MIN/MAX_DELTA constants)
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
                            otm_lo_pct=0, otm_hi_pct=25, max_strikes=25,
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

    async def _scan_ticker_safe_leap(t: str) -> List[dict]:
        try:
            return await asyncio.wait_for(_scan_ticker(t), timeout=25)
        except asyncio.TimeoutError:
            log.warning("LEAP [%s]: per-ticker scan timed out (>25s) — skipping", t)
            return []

    regime, *ticker_results = await asyncio.gather(
        _market_regime(),
        *[_scan_ticker_safe_leap(t) for t in CSP_UNIVERSE],
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

    # Restore screened universe from cache (avoids reverting to hardcoded 19 on restart)
    _universe_load()

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
            _universe_save(tickers)
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


@app.get("/technicals/{ticker}")
async def get_technicals(ticker: str):
    """
    Comprehensive technical analysis using 1Y daily OHLCV from IBKR.
    Returns RSI-14, MACD, Bollinger Bands, SMA-20/50/200, volume breakout,
    and an overall composite signal.  Requires active IBKR connection.
    """
    _require_connection()
    ticker = ticker.upper().strip()
    ib = state["ib"]
    try:
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
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"IBKR data error: {e}")

    if len(bars) < 60:
        raise HTTPException(status_code=404, detail=f"Not enough history for {ticker} ({len(bars)} bars)")

    df      = util.df(bars)
    closes  = df["close"].astype(float)
    volumes = df["volume"].astype(float)

    ind = _compute_indicators_from_closes(closes, volumes)

    price   = round(float(closes.iloc[-1]), 2)
    prev    = round(float(closes.iloc[-2]), 2)
    chg_pct = round((price - prev) / prev * 100, 2) if prev else 0.0

    # Trend classification from MA stack
    s20  = ind.get("above_sma20")
    s50  = ind.get("above_sma50")
    s200 = ind.get("above_sma200")
    if s20 and s50 and s200:
        trend = "bullish"
    elif not s20 and not s50 and (s200 is False):
        trend = "bearish"
    else:
        trend = "mixed"

    rsi14    = ind.get("rsi14") or 50
    rsi_zone = "overbought" if rsi14 > 70 else "oversold" if rsi14 < 30 else "neutral"
    macd_bias = "bullish" if (ind.get("macd_hist") or 0) > 0 else "bearish"

    # Volume breakout — use per-ticker 90th-pct threshold; fall back to 1.5× if not enough history
    vol_ratio = ind.get("vol_ratio") or 1.0
    vol_90pct = ind.get("vol_90pct") or 1.5
    pct_b_val = ind.get("pct_b")
    price_breakout = bool(pct_b_val is not None and pct_b_val > 95)
    confirmed = price_breakout and vol_ratio >= vol_90pct
    breakout_signal = "BREAKOUT" if confirmed else ("WATCH" if price_breakout else "NORMAL")

    # Composite overall score (ported from ibkr_technicals._overall)
    score = 0
    if trend == "bullish": score += 2
    elif trend == "bearish": score -= 2
    if macd_bias == "bullish": score += 1
    else: score -= 1
    if rsi_zone == "oversold": score += 1
    elif rsi_zone == "overbought": score -= 1
    if confirmed: score += 2
    if score >= 3: overall = "strong_buy"
    elif score >= 1: overall = "buy"
    elif score <= -3: overall = "strong_sell"
    elif score <= -1: overall = "sell"
    else: overall = "neutral"

    # IV rank — async, cached 1h per ticker via _iv_rank_for_ticker
    try:
        iv_data = await _iv_rank_for_ticker(ticker)
    except Exception:
        iv_data = {"iv": None, "rank": None, "rv_lo": None, "rv_hi": None}

    return {
        "ticker":     ticker,
        "price":      price,
        "change_pct": chg_pct,
        "trend":      trend,
        "bars_used":  len(bars),
        "indicators": {
            "rsi14":        ind.get("rsi14"),
            "rsi_zone":     rsi_zone,
            "macd":         ind.get("macd"),
            "macd_signal":  ind.get("macd_signal"),
            "macd_hist":    ind.get("macd_hist"),
            "macd_bias":    macd_bias,
            "sma20":        ind.get("sma20"),
            "sma50":        ind.get("sma50"),
            "sma200":       ind.get("sma200"),
            "above_sma20":  ind.get("above_sma20"),
            "above_sma50":  ind.get("above_sma50"),
            "above_sma200": ind.get("above_sma200"),
            "bb_upper":     ind.get("bb_upper"),
            "bb_mid":       ind.get("bb_mid"),
            "bb_lower":     ind.get("bb_lower"),
            "pct_b":        ind.get("pct_b"),
            "vol_ratio":    vol_ratio,
            "vol_90pct":    round(vol_90pct, 2),
        },
        "breakout": {
            "signal":        breakout_signal,
            "vol_ratio":     round(vol_ratio, 2),
            "vol_90pct":     round(vol_90pct, 2),
            "price_breakout": price_breakout,
            "confirmed":     confirmed,
        },
        "summary": {
            "trend":           trend,
            "rsi_zone":        rsi_zone,
            "macd_bias":       macd_bias,
            "breakout_signal": breakout_signal,
            "overall":         overall,
        },
        "iv_rank": iv_data,
    }


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

    # Compute filter diagnostics for the API response
    recommended = _filter_csp_recommended(candidates)
    n = len(candidates)
    filter_summary = {
        "total_raw":       n,
        "recommended":     len(recommended),
        "rejected_warn":   sum(1 for r in candidates if len(r.get("warnings", [])) > 0),
        "rejected_liq":    sum(1 for r in candidates if not r.get("warnings") and r["liquidity_score"] < 50),
        "rejected_iv":     sum(1 for r in candidates if not r.get("warnings") and r["liquidity_score"] >= 50
                               and r["iv_rank"] < IV_RANK_MIN_CSP),
        "rejected_score":  sum(1 for r in candidates if not r.get("warnings") and r["liquidity_score"] >= 50
                               and r["iv_rank"] >= IV_RANK_MIN_CSP and r["score"] < 70),
        "rejected_trend":  sum(1 for r in candidates if not r.get("warnings") and r["liquidity_score"] >= 50
                               and r["iv_rank"] >= IV_RANK_MIN_CSP and r["score"] >= 70
                               and (r.get("above_sma50") is False or r.get("above_sma20") is False
                                    or r.get("above_sma200") is False)),
        "rejected_earn":   sum(1 for r in candidates if not r.get("warnings") and r["liquidity_score"] >= 50
                               and r["iv_rank"] >= IV_RANK_MIN_CSP and r["score"] >= 70
                               and r.get("above_sma50") is not False and r.get("above_sma20") is not False
                               and r.get("above_sma200") is not False
                               and r["earnings_days_out"] is not None
                               and r["earnings_days_out"] <= EARNINGS_BLOCK_DAYS * 2),
    }

    cache["csp"]    = candidates
    cache["regime"] = regime
    cache["ts"]     = now
    return {
        "cached":          False,
        "scanned_at":      now.isoformat() + "Z",
        "count":           len(candidates),
        "candidates":      candidates,
        "market_regime":   regime,
        "filter_summary":  filter_summary,
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
        state["scan_cache"]["csp"] = None
    return {"ok": True, "universe": CSP_UNIVERSE}


@app.post("/csp/universe/remove")
def remove_from_csp_universe(req: AddTickerRequest):
    ticker = req.ticker.upper()
    if ticker in CSP_UNIVERSE:
        CSP_UNIVERSE.remove(ticker)
        state["scan_cache"]["csp"] = None
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
        _universe_save(tickers)
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
    profit_target_pct:  float     = 0.50
    stop_loss_mult:     float     = 2.0
    scan_types:         List[str] = ["csp"]
    csp_capital:        float     = 20000.0
    leap_capital:       float     = 5000.0
    use_kelly:          bool      = True
    total_capital:      float     = 100000.0
    # assumed_win_rate is intentionally excluded — it is system-managed via
    # _update_kelly_from_journal and should not be overwritten by the UI.
    auto_hedge:         bool      = False
    hedge_threshold:    float     = 100.0


class ReconnectRequest(BaseModel):
    port: int   # 7497 = paper, 7496 = live, 4001/4002 = IB Gateway


class ClosePositionRequest(BaseModel):
    key: str   # contract key from at["positions"]


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
async def cancel_inactive_orders():
    """Cancel all Inactive orders using reqGlobalCancel (handles cross-session orders)."""
    _require_connection()
    ib = state["ib"]

    async def _do_cancel(ib: IB):
        stale_ids = [t.order.orderId for t in ib.openTrades() if t.orderStatus.status == "Inactive"]
        ib.client.reqGlobalCancel()
        _at_log("SYSTEM", f"reqGlobalCancel sent — cancelling {len(stale_ids)} Inactive orders: {stale_ids}")
        return stale_ids

    loop = asyncio.get_event_loop()
    stale_ids = await loop.run_in_executor(
        None,
        lambda: _run_in_streaming_loop(_do_cancel(ib), timeout=10),
    )
    return {"ok": True, "cancelled": stale_ids, "count": len(stale_ids), "method": "reqGlobalCancel"}


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


# ── Account summary endpoint ───────────────────────────────────────────────

@app.get("/account/summary")
def account_summary():
    """
    Real-time account cash, equity, margin, and account number from IBKR.
    ib_insync subscribes account values automatically on connect, so
    ib.accountValues() is always up to date without an extra async call.
    """
    _require_connection()
    ib = state["ib"]

    # managedAccounts() returns list of account strings from the handshake
    accounts   = ib.managedAccounts()
    account_id = accounts[0] if accounts else "unknown"

    want_map = {
        "TotalCashValue":     "total_cash",
        "AvailableFunds":     "available_funds",
        "NetLiquidation":     "net_liquidation",
        "GrossPositionValue": "gross_position_value",
        "InitialMarginReq":   "initial_margin_req",
        "UnrealizedPnL":      "unrealized_pnl",
        "RealizedPnL":        "realized_pnl",
        "BuyingPower":        "buying_power",
    }

    result: dict = {v: 0.0 for v in want_map.values()}
    result["account"]  = account_id
    result["currency"] = "USD"

    for av in ib.accountValues(account_id):
        key = want_map.get(av.tag)
        if key and av.currency == "USD":
            try:
                result[key] = round(float(av.value), 2)
            except (ValueError, TypeError):
                pass

    return result


# ── Auto-trader endpoints ──────────────────────────────────────────────────

@app.get("/autotrader/status")
def autotrader_status():
    """Return current auto-trader state, config, positions, and log."""
    at = state["autotrader"]
    clean_cfg = {k: v for k, v in at["config"].items() if k != "trailing_exit"}

    # Deep-copy positions so we can safely add live P&L without mutating state
    positions_enriched = {k: dict(v) for k, v in at["positions"].items()}

    # Enrich tracked positions with live P&L; collect untracked portfolio options
    ib = state.get("ib")
    untracked_positions = []
    if ib and state.get("connected"):
        try:
            for item in ib.portfolio():
                c = item.contract
                if getattr(c, "secType", "") != "OPT":
                    continue
                k = _at_contract_key(c)
                live_pnl   = round(float(item.unrealizedPNL or 0), 2)
                live_price = round(float(item.marketPrice or 0), 4)
                if k in positions_enriched:
                    positions_enriched[k]["live_pnl"]   = live_pnl
                    positions_enriched[k]["live_price"]  = live_price
                else:
                    # Option in portfolio but not tracked by auto-trader (manual trade)
                    untracked_positions.append({
                        "ticker":     getattr(c, "symbol", ""),
                        "strike":     float(getattr(c, "strike", 0)),
                        "right":      getattr(c, "right", ""),
                        "expiry":     (getattr(c, "lastTradeDateOrContractMonth", "") or "")[:8],
                        "action":     "BUY" if float(item.position or 0) > 0 else "SELL",
                        "qty":        abs(int(item.position or 0)),
                        "live_pnl":   live_pnl,
                        "live_price": live_price,
                        "market_value": round(float(item.marketValue or 0), 2),
                        "avg_cost":   round(float(item.averageCost or 0), 4),
                    })
        except Exception:
            pass

    return {
        "enabled":              at["enabled"],
        "config":               clean_cfg,
        "positions":            positions_enriched,
        "untracked_positions":  untracked_positions,
        "stopped_out":          dict(at.get("stopped_out", {})),
        "log":                  list(at["log"]),
        "last_run":             at.get("last_run"),
        "premium_collected":    at.get("premium_collected", 0.0),
        "leap_pnl":             at.get("leap_pnl", 0.0),
        "leap_budget":          at.get("leap_budget", 0.0),
    }


@app.post("/autotrader/config")
def update_autotrader_config(req: AutoTraderConfigRequest):
    """Enable/disable the auto-trader and update its config."""
    was = state["autotrader"]["enabled"]
    state["autotrader"]["enabled"] = req.enabled
    cfg = state["autotrader"]["config"]
    cfg.pop("trailing_exit", None)  # clean any stale key on every save
    cfg.update({
        "max_positions":     req.max_positions,
        "profit_target_pct": req.profit_target_pct,
        "stop_loss_mult":    req.stop_loss_mult,
        "scan_types":        req.scan_types,
        "csp_capital":       req.csp_capital,
        "leap_capital":      req.leap_capital,
        "use_kelly":         req.use_kelly,
        "total_capital":     req.total_capital,
        # assumed_win_rate is NOT updated here — managed by _update_kelly_from_journal
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
            lambda: _run_in_streaming_loop(_autotrader_scan_and_trade_coro(ib), timeout=270),
        )
        state["autotrader"]["last_run"] = datetime.utcnow().isoformat() + "Z"
    except (TimeoutError, RuntimeError) as exc:
        raise HTTPException(503, str(exc))
    return {"ok": True, "log": state["autotrader"]["log"][-20:]}


@app.post("/autotrader/clear-cooldown")
def clear_cooldown(req: AddTickerRequest):
    """Remove a ticker from the 48h stop-loss cooldown, allowing immediate re-entry."""
    ticker = req.ticker.upper()
    at = state["autotrader"]
    removed = ticker in at.get("stopped_out", {})
    at.setdefault("stopped_out", {}).pop(ticker, None)
    _at_save_state()
    _at_log("SYSTEM", f"Cooldown cleared for {ticker} by user request")
    return {"ok": True, "removed": removed}


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


@app.post("/autotrader/close-position")
async def close_position_manual(req: ClosePositionRequest):
    """Manually close a specific tracked position by contract key."""
    _require_connection()
    ib  = state["ib"]
    at  = state["autotrader"]
    key = req.key
    if key not in at["positions"]:
        raise HTTPException(status_code=404, detail=f"Position '{key}' not tracked")

    info = dict(at["positions"][key])   # copy — close_coro will pop it
    info["exit_reason"] = "manual"

    # Find matching portfolio item
    portfolio  = ib.portfolio()
    matching   = next(
        (item for item in portfolio if _at_contract_key(item.contract) == key),
        None,
    )
    if not matching:
        # Ghost position — remove from tracking only
        at["positions"].pop(key, None)
        _at_save_state()
        _at_log("SYSTEM", f"Ghost position {key} removed from tracking (not in portfolio)")
        return {"ok": True, "note": "Removed ghost position from tracking"}

    _at_log("SYSTEM", f"Manual close requested for {key}")
    await _autotrader_close_coro(ib, matching, info, key)
    return {"ok": True}


@app.get("/autotrader/decisions")
def get_autotrader_decisions(limit: int = Query(100, ge=1, le=500)):
    """Return plain-English trade decision log (most recent first)."""
    decisions = state["autotrader"].get("decisions", [])
    return {"decisions": list(reversed(decisions[-limit:]))}


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


# ── Breakout backtest ───────────────────────────────────────────────────────

def _backtest_breakout_one(ticker: str) -> Optional[dict]:
    """Walk-forward breakout backtest for one ticker using yfinance 2-yr daily OHLCV.

    Signal: pct_b > 95 (price at/above upper BB) AND vol_ratio >= rolling vol_90pct.
    vol_90pct is recomputed each day from trailing 252 days of vol ratios → no look-ahead.
    Measures 5d / 10d / 20d forward returns from each signal day.
    """
    try:
        df = yf.Ticker(ticker).history(period="2y")
    except Exception as e:
        log.debug("Breakout backtest yf fetch %s: %s", ticker, e)
        return None
    if len(df) < 252:
        return None

    closes  = df["Close"].astype(float)
    volumes = df["Volume"].astype(float)

    # Bollinger Bands (20, 2) — %B position
    sma20    = closes.rolling(20).mean()
    bb_std   = closes.rolling(20).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    pct_b    = ((closes - bb_lower) / (bb_upper - bb_lower).replace(0, float("nan"))) * 100

    # Volume ratio: today vs prior 20-day avg (shift=1 — no look-ahead)
    roll_avg  = volumes.rolling(20).mean().shift(1)
    vol_ratio = volumes / roll_avg.replace(0, float("nan"))

    # Walk-forward 90th-pct threshold: rolling 252-day quantile of vol_ratio history
    vol_90pct_series = vol_ratio.rolling(252, min_periods=60).quantile(0.90)

    # Signal mask — require both vol_ratio and vol_90pct to be valid
    signal_mask = (
        (pct_b > 95)
        & (vol_ratio >= vol_90pct_series)
        & vol_ratio.notna()
        & vol_90pct_series.notna()
    )

    signal_indices = [i for i, v in enumerate(signal_mask) if v]
    n_signals = len(signal_indices)

    avg_threshold = round(float(vol_90pct_series.dropna().mean()), 2) \
        if not vol_90pct_series.dropna().empty else None

    if n_signals == 0:
        return {
            "ticker": ticker, "signals": 0,
            "avg_vol_90pct": avg_threshold,
            "wins_5d": 0, "wins_10d": 0, "wins_20d": 0,
            "win_rate_5d": None, "win_rate_10d": None, "win_rate_20d": None,
            "avg_ret_5d": None, "avg_ret_10d": None, "avg_ret_20d": None,
            "recent_signals": [],
        }

    rets_5d, rets_10d, rets_20d = [], [], []
    n = len(closes)
    for i in signal_indices:
        p0 = closes.iloc[i]
        if p0 <= 0:
            continue
        if i + 5  < n: rets_5d.append( (closes.iloc[i+5]  / p0 - 1) * 100)
        if i + 10 < n: rets_10d.append((closes.iloc[i+10] / p0 - 1) * 100)
        if i + 20 < n: rets_20d.append((closes.iloc[i+20] / p0 - 1) * 100)

    def _wr(rets):  return round(sum(1 for r in rets if r > 0) / len(rets) * 100, 1) if rets else None
    def _avg(rets): return round(sum(rets) / len(rets), 2) if rets else None

    # Last 5 signal dates for display
    recent = []
    for i in signal_indices[-5:]:
        d = df.index[i]
        recent.append(str(d.date()) if hasattr(d, "date") else str(d)[:10])

    return {
        "ticker":         ticker,
        "signals":        n_signals,
        "avg_vol_90pct":  avg_threshold,
        "wins_5d":        sum(1 for r in rets_5d  if r > 0),
        "wins_10d":       sum(1 for r in rets_10d if r > 0),
        "wins_20d":       sum(1 for r in rets_20d if r > 0),
        "win_rate_5d":    _wr(rets_5d),
        "win_rate_10d":   _wr(rets_10d),
        "win_rate_20d":   _wr(rets_20d),
        "avg_ret_5d":     _avg(rets_5d),
        "avg_ret_10d":    _avg(rets_10d),
        "avg_ret_20d":    _avg(rets_20d),
        "recent_signals": recent,
    }


def _run_breakout_backtest() -> dict:
    tickers = list(CSP_UNIVERSE)
    rows = []
    for ticker in tickers:
        try:
            r = _backtest_breakout_one(ticker)
            if r:
                rows.append(r)
        except Exception as exc:
            log.warning("Breakout backtest %s: %s", ticker, exc)

    with_signals = [r for r in rows if r["signals"] > 0]
    total_signals = sum(r["signals"] for r in rows)
    total_wins_5d  = sum(r["wins_5d"]  for r in rows)
    total_wins_10d = sum(r["wins_10d"] for r in rows)
    total_wins_20d = sum(r["wins_20d"] for r in rows)

    def _safe_avg(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 2) if v else None

    summary = {
        "total_signals":  total_signals,
        "win_rate_5d":    round(total_wins_5d  / total_signals * 100, 1) if total_signals else None,
        "win_rate_10d":   round(total_wins_10d / total_signals * 100, 1) if total_signals else None,
        "win_rate_20d":   round(total_wins_20d / total_signals * 100, 1) if total_signals else None,
        "avg_ret_5d":     _safe_avg([r["avg_ret_5d"]  for r in with_signals]),
        "avg_ret_10d":    _safe_avg([r["avg_ret_10d"] for r in with_signals]),
        "avg_ret_20d":    _safe_avg([r["avg_ret_20d"] for r in with_signals]),
        "tickers_tested": len(rows),
    }
    return {"tickers": tickers, "results": rows, "summary": summary}


@app.get("/backtest/breakout")
async def backtest_breakout_endpoint():
    """Walk-forward breakout backtest on all current universe tickers.

    Uses yfinance 2-yr daily OHLCV — no IBKR connection required.
    Per-ticker vol_90pct threshold computed from rolling 252-day history (no look-ahead).
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_breakout_backtest)
    return result


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
    total = con.execute("SELECT COUNT(*) FROM trade_journal").fetchone()[0]
    rows  = con.execute(
        "SELECT * FROM trade_journal ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    cols = [d[0] for d in con.execute("SELECT * FROM trade_journal LIMIT 0").description or []]
    con.close()
    trades = [dict(zip(cols, r)) for r in rows]
    return {"trades": trades, "count": len(trades), "total": total}


@app.get("/journal/stats")
def get_journal_stats():
    """Win rate, avg P&L, feature importances from completed trades."""
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    rows = con.execute(
        "SELECT win, pnl, pnl_pct, exit_reason, strategy_type, iv_rank, score "
        "FROM trade_journal "
        "WHERE closed_at IS NOT NULL AND win IS NOT NULL "
        f"AND exit_reason IN ({_REAL_EXIT_REASONS})"
    ).fetchall()
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

    by_strategy: dict = {}
    for r in rows:
        st = (r[4] or "csp").lower()
        by_strategy.setdefault(st, {"total": 0, "wins": 0, "pnl": 0.0})
        by_strategy[st]["total"] += 1
        if r[0] == 1:
            by_strategy[st]["wins"] += 1
        if r[1] is not None:
            by_strategy[st]["pnl"] = round(by_strategy[st]["pnl"] + r[1], 2)

    model_log = [dict(zip(model_cols, r)) for r in model_rows]

    win_pnls  = [r[1] for r in wins   if r[1] is not None]
    loss_pnls = [r[1] for r in losses if r[1] is not None]
    return {
        "total_trades":    len(rows),
        "win_rate":        round(len(wins) / len(rows) * 100, 1),
        "avg_pnl":         round(sum(pnls) / len(pnls), 2) if pnls else 0,
        "total_pnl":       round(sum(pnls), 2) if pnls else 0,
        "avg_win_pnl":     round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0,
        "avg_loss_pnl":    round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0,
        "exit_breakdown":  by_reason,
        "by_strategy":     by_strategy,
        "model_version":   state.get("model_version", 0),
        "model_log":       model_log,
        "assumed_win_rate":state["autotrader"]["config"].get("assumed_win_rate"),
    }


@app.get("/pnl/dashboard")
def pnl_dashboard():
    """Unified P&L dashboard: account metrics, per-position data, chart series."""
    from collections import defaultdict

    acct: dict = {}
    try:
        acct = account_summary()
    except Exception:
        pass

    con      = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    all_rows = con.execute("SELECT * FROM trade_journal ORDER BY id").fetchall()
    desc     = con.execute("SELECT * FROM trade_journal LIMIT 0").description or []
    cols     = [d[0] for d in desc]
    con.close()

    trades        = [dict(zip(cols, r)) for r in all_rows]
    open_trades   = [t for t in trades if t.get("closed_at") is None]
    closed_trades = sorted(
        [t for t in trades if t.get("closed_at") is not None],
        key=lambda t: t["closed_at"], reverse=True,
    )

    daily: dict = defaultdict(float)
    for t in closed_trades:
        day = (t.get("closed_at") or "")[:10]
        if day and t.get("pnl") is not None:
            daily[day] += t["pnl"]
    cum      = 0.0
    daily_pnl = []
    for day, pnl in sorted(daily.items()):
        cum += pnl
        daily_pnl.append({"date": day, "pnl": round(pnl, 2), "cumulative": round(cum, 2)})

    _REAL_EXIT_SET = {
        "profit_target", "stop_loss", "roll_close",
        "roll_max", "roll_no_credit", "21dte", "manual", "rotation",
    }
    real_closed   = [t for t in closed_trades if t.get("exit_reason") in _REAL_EXIT_SET]
    closed_pnls   = [t["pnl"] for t in real_closed if t.get("pnl") is not None]
    wins          = [t for t in real_closed if t.get("win") == 1]
    losses        = [t for t in real_closed if t.get("win") == 0]
    today_str     = date.today().isoformat()
    today_pnl     = sum(
        t["pnl"] for t in real_closed
        if (t.get("closed_at") or "")[:10] == today_str and t.get("pnl") is not None
    )

    exit_breakdown: dict = {}
    for t in closed_trades:
        reason = t.get("exit_reason") or "unknown"
        exit_breakdown.setdefault(reason, {"count": 0, "pnl": 0.0})
        exit_breakdown[reason]["count"] += 1
        if t.get("pnl") is not None:
            exit_breakdown[reason]["pnl"] = round(exit_breakdown[reason]["pnl"] + t["pnl"], 2)

    portfolio_items: list = []
    ib_connected = bool(state.get("ib") and state.get("connected"))
    if ib_connected:
        try:
            for item in state["ib"].portfolio():
                c = item.contract
                portfolio_items.append({
                    "ticker":         c.symbol,
                    "sec_type":       getattr(c, "secType", ""),
                    "strike":         getattr(c, "strike", None),
                    "right":          getattr(c, "right", None),
                    "expiry":         (getattr(c, "lastTradeDateOrContractMonth", "") or "")[:8] or None,
                    "position":       item.position,
                    "avg_cost":       round(float(item.averageCost or 0), 4),
                    "market_value":   round(float(item.marketValue or 0), 2),
                    "unrealized_pnl": round(float(item.unrealizedPNL or 0), 2),
                    "realized_pnl":   round(float(item.realizedPNL or 0), 2),
                })
        except Exception:
            pass

    total_realized   = round(sum(closed_pnls), 2) if closed_pnls else 0.0
    total_unrealized = round(float(acct.get("unrealized_pnl") or 0), 2)

    # ── Build open_positions: IBKR is the source of truth when connected ──────
    # Enrich each IBKR portfolio item with journal metadata (entry price, IV, etc.)
    # If not connected, fall back to recent journal entries (last 7 days).
    # Also auto-close journal entries that have no corresponding IBKR position.
    if ib_connected:
        # Build a lookup: (ticker, str(strike), right, expiry[:6]) → journal row
        def _jkey(ticker, strike, right, expiry):
            return (ticker, str(strike or ""), right or "", (expiry or "")[:6])

        jlookup = {}
        for t in open_trades:
            k = _jkey(t.get("ticker",""), t.get("strike"), t.get("right",""), t.get("expiry",""))
            jlookup[k] = t

        ibkr_keys = {
            _jkey(p["ticker"], p.get("strike"), p.get("right"), p.get("expiry"))
            for p in portfolio_items
        }

        # Auto-close open journal entries not present in IBKR portfolio
        orphan_ids = [
            t["id"] for t in open_trades
            if _jkey(t.get("ticker",""), t.get("strike"), t.get("right",""), t.get("expiry",""))
            not in ibkr_keys
        ]
        if orphan_ids:
            try:
                _con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
                ph   = ",".join("?" * len(orphan_ids))
                _con.execute(
                    f"UPDATE trade_journal SET closed_at=?, exit_reason='orphaned' "
                    f"WHERE id IN ({ph}) AND closed_at IS NULL",
                    [datetime.utcnow().isoformat()] + orphan_ids,
                )
                _con.commit()
                _con.close()
                log.info("pnl_dashboard: auto-closed %d orphaned journal entries", len(orphan_ids))
            except Exception as e:
                log.warning("pnl_dashboard orphan cleanup failed: %s", e)

        # Build a roll_count lookup from live position tracking (not stored in journal schema)
        at_positions = state["autotrader"].get("positions", {})
        rc_lookup: dict = {}  # (ticker, right, str(strike), expiry[:8]) → roll_count
        for _info in at_positions.values():
            _key = (
                _info.get("ticker", ""),
                _info.get("right", ""),
                str(_info.get("strike", "")),
                (_info.get("expiry", "") or "")[:8],
            )
            rc_lookup[_key] = _info.get("roll_count", 0)

        # Merge IBKR portfolio item with journal enrichment
        visible_open = []
        for p in portfolio_items:
            jrow = jlookup.get(_jkey(p["ticker"], p.get("strike"), p.get("right"), p.get("expiry")), {})
            rc_key = (p["ticker"], p.get("right") or "", str(p.get("strike") or ""), (p.get("expiry") or "")[:8])
            visible_open.append({
                "id":             jrow.get("id"),
                "ticker":         p["ticker"],
                "strategy_type":  jrow.get("strategy_type") or None,
                "action":         jrow.get("action", "BUY" if (p.get("position") or 0) > 0 else "SELL"),
                "strike":         p.get("strike"),
                "right":          p.get("right"),
                "expiry":         p.get("expiry"),
                "qty":            abs(p.get("position") or 0),
                "entry_price":    jrow.get("entry_price"),
                "avg_cost":       p.get("avg_cost"),
                "market_value":   p.get("market_value"),
                "unrealized_pnl": p.get("unrealized_pnl"),
                "live_iv_entry":  jrow.get("live_iv_entry"),
                "roll_count":     rc_lookup.get(rc_key, 0),
                "opened_at":      jrow.get("opened_at"),
                "dte":            jrow.get("dte"),
            })
    else:
        # Disconnected: fall back to recent journal entries (last 7 days)
        recent_cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
        visible_open  = [
            t for t in open_trades
            if (t.get("opened_at") or "") >= recent_cutoff
        ]

    return {
        "account": acct,
        "stats": {
            "total_trades":         len(real_closed),
            "open_count":           len(portfolio_items) if ib_connected else len(open_trades),
            "win_rate":             round(len(wins) / len(real_closed) * 100, 1) if real_closed else None,
            "total_realized_pnl":   total_realized,
            "total_unrealized_pnl": total_unrealized,
            "total_pnl":            round(total_realized + total_unrealized, 2),
            "today_pnl":            round(today_pnl, 2),
            "avg_win":              round(sum(t["pnl"] for t in wins   if t.get("pnl")) / len(wins),   2) if wins   else 0,
            "avg_loss":             round(sum(t["pnl"] for t in losses if t.get("pnl")) / len(losses), 2) if losses else 0,
            "best_trade":           round(max(closed_pnls), 2) if closed_pnls else 0,
            "worst_trade":          round(min(closed_pnls), 2) if closed_pnls else 0,
        },
        "open_positions": visible_open[-20:],
        "closed_trades":  real_closed[:30],
        "daily_pnl":      daily_pnl,
        "portfolio":      portfolio_items,
        "exit_breakdown": exit_breakdown,
    }


@app.post("/journal/cleanup")
def journal_cleanup():
    """
    Mark phantom open journal entries as closed (exit_reason='orphaned').

    An entry is orphaned when closed_at IS NULL but is not tracked in the
    current autotrader positions dict AND was opened more than 24 hours ago.
    Entries opened within the last 24 h are left alone — they may still be
    working orders waiting to fill.
    """
    active_journal_ids: set = {
        info.get("journal_id")
        for info in state["autotrader"].get("positions", {}).values()
        if info.get("journal_id") is not None
    }
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    # Fetch all open entries older than 24 h
    rows = con.execute(
        "SELECT id FROM trade_journal WHERE closed_at IS NULL AND opened_at < ?",
        (cutoff,),
    ).fetchall()
    orphan_ids = [r[0] for r in rows if r[0] not in active_journal_ids]
    if orphan_ids:
        con.execute(
            f"UPDATE trade_journal SET closed_at=?, exit_reason='orphaned' "
            f"WHERE id IN ({','.join('?' * len(orphan_ids))}) AND closed_at IS NULL",
            [datetime.utcnow().isoformat()] + orphan_ids,
        )
        con.commit()
    con.close()
    log.info("Journal cleanup: marked %d orphaned entries as closed", len(orphan_ids))
    return {
        "ok": True,
        "orphaned": len(orphan_ids),
        "active_tracked": len(active_journal_ids),
        "message": f"Closed {len(orphan_ids)} phantom open entries (exit_reason=orphaned, pnl left null — excluded from stats)",
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
