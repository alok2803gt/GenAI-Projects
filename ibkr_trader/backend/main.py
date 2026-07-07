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
import requests
import uvicorn
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ib_insync import IB, Index, LimitOrder, Option, Order, Stock, util
from pydantic import BaseModel
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import joblib
import os
try:
    from ibkr_technicals import router as technicals_router
    _technicals_router_ok = True
except Exception as _tech_err:
    _technicals_router_ok = False
    technicals_router = None  # loaded later with graceful skip

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

# ── Tape reader — client-ID pool (IDs 20-29 reserved for live tape WebSockets) ─
_TAPE_CLIENT_ID_POOL: set = set(range(20, 30))
_tape_pool_lock = threading.Lock()

def _acquire_tape_cid() -> Optional[int]:
    with _tape_pool_lock:
        if not _TAPE_CLIENT_ID_POOL:
            return None
        cid = min(_TAPE_CLIENT_ID_POOL)
        _TAPE_CLIENT_ID_POOL.discard(cid)
        return cid

def _release_tape_cid(cid: int) -> None:
    with _tape_pool_lock:
        _TAPE_CLIENT_ID_POOL.add(cid)

def _tape_fmt_vol(v: int) -> str:
    if v >= 1_000_000: return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:     return f"{v / 1_000:.1f}K"
    return str(v)


FEATURE_COLS = [
    "rsi", "sma5", "sma14", "momentum",
    "vol_ratio", "body_pct", "upper_wick", "lower_wick", "volatility"
]

# ── Tape sentiment — CVD scoring ───────────────────────────────────────────
TAPE_SENTIMENT_MAX_TICKERS = 20   # cap on reqMktData subscriptions for tape
TAPE_STALENESS_SECS        = 1800 # score treated as 0.0 (neutral) when older than 30 min
TAPE_BLOCK_MIN_SHARES      = 5000 # CVD-callback block threshold for real-time capture

# ── Market index banner ────────────────────────────────────────────────────
INDEX_CONFIG = [
    {"sym": "SPY",  "name": "S&P 500"},
    {"sym": "QQQ",  "name": "Nasdaq"},
    {"sym": "DIA",  "name": "Dow"},
    {"sym": "IWM",  "name": "Russell"},
    {"sym": "VIX",  "name": "VIX"},
]
# These four ETFs are always subscribed (before universe/watchlist candidates)
# so the index banner has live IBKR prices from market open.
INDEX_ETF_TICKERS  = ["SPY", "QQQ", "DIA", "IWM"]
_index_cache: dict = {"data": None, "ts": 0.0}
INDEX_CACHE_TTL    = 14400  # 4 hours — prev_close doesn't change intraday; IBKR live prices overlay on every call


class _FlowAbort(Exception):
    """Raised when the live order-book flow gate intentionally aborts a trade.
    Distinct from ValueError so callers can log it as SKIP rather than ERROR."""


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
IV_RANK_MIN_CSP     = 35     # Lowered from 50: journal losses averaged 36.8% but in low-VIX
                              # bull markets the universe is empty at 50+. 35 keeps us above the
                              # historical loss zone while allowing more trades.
IV_RANK_MIN_CSP_ETF = 20     # ETFs (SPY/QQQ/IWM etc.) have no earnings binary risk; their IV
                              # is structurally lower than single-name, so a lower bar is correct.
_CSP_ETF_TICKERS    = frozenset({"SPY", "QQQ", "IWM", "GLD", "XLE", "XLF", "XLK", "XLV", "ARKK"})
EARNINGS_CACHE_TTL  = 21600  # 6 h — earnings dates don't change intraday
IV_RANK_CACHE_TTL   = 3600   # 1 h
REGIME_CACHE_TTL    = 300    # 5 min

# ── Config — IV history & technical indicators ─────────────────────────────
IV_HISTORY_PATH     = "iv_history.json"
IV_HISTORY_MIN_PTS  = 20    # samples needed before percentile rank is reliable
JOURNAL_DB_PATH     = "trade_journal.db"
TAPE_DB_PATH        = "tape_data.db"
JOURNAL_MIN_TRADES  = 20    # trades needed before learned model is reliable
RETRAIN_EVERY       = 5     # retrain model every N new closed trades
AT_STATE_PATH       = "autotrader_state.json"  # persisted across restarts
ST_STATE_PATH       = "stock_state.json"        # stock trader state (persisted)
DT_STATE_PATH       = "day_trader_state.json"   # day trader state (persisted)
SPX_STATE_PATH      = "spx_0dte_state.json"     # SPX 0DTE trader state (persisted)
UNIVERSE_CACHE_PATH = "universe_cache.json"     # screened universe (refreshed nightly)
WATCHLIST_PATH      = "watchlist.json"          # breakout scanner watchlist

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

# ── Stock sector mapping for rotation sector guard ─────────────────────────
STOCK_SECTOR_MAP: Dict[str, str] = {
    # Technology
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AMD":"Technology",
    "INTC":"Technology","QCOM":"Technology","CRM":"Technology","NOW":"Technology",
    "ADBE":"Technology","ORCL":"Technology","SNOW":"Technology","PANW":"Technology",
    "CRWD":"Technology","ZS":"Technology","DDOG":"Technology","NET":"Technology",
    "AVGO":"Technology","TXN":"Technology","MU":"Technology","AMAT":"Technology",
    "LRCX":"Technology","KLAC":"Technology","MRVL":"Technology","PLTR":"Technology",
    "IBM":"Technology","CSCO":"Technology","INTU":"Technology","SMCI":"Technology",
    "ON":"Technology","MPWR":"Technology",
    # Consumer Discretionary
    "NKE":"Consumer Discretionary","HD":"Consumer Discretionary",
    "SBUX":"Consumer Discretionary","LOW":"Consumer Discretionary",
    "TGT":"Consumer Discretionary","MCD":"Consumer Discretionary",
    "BKNG":"Consumer Discretionary","LULU":"Consumer Discretionary",
    "RIVN":"Consumer Discretionary","RBLX":"Consumer Discretionary",
    "UBER":"Consumer Discretionary","ABNB":"Consumer Discretionary",
    "RCL":"Consumer Discretionary","ROKU":"Consumer Discretionary",
    "TSLA":"Consumer Discretionary","AMZN":"Consumer Discretionary",
    "CMG":"Consumer Discretionary","SHOP":"Consumer Discretionary",
    # Healthcare
    "JNJ":"Healthcare","UNH":"Healthcare","LLY":"Healthcare","PFE":"Healthcare",
    "ABBV":"Healthcare","MRK":"Healthcare","TMO":"Healthcare","DHR":"Healthcare",
    "ISRG":"Healthcare","VRTX":"Healthcare","GILD":"Healthcare","BMY":"Healthcare",
    "AMGN":"Healthcare","ELV":"Healthcare","HUM":"Healthcare","CI":"Healthcare",
    "REGN":"Healthcare",
    # Financials
    "JPM":"Financials","BAC":"Financials","WFC":"Financials","GS":"Financials",
    "MS":"Financials","C":"Financials","BLK":"Financials","SCHW":"Financials",
    "V":"Financials","MA":"Financials","AXP":"Financials","TFC":"Financials",
    "COIN":"Financials","HOOD":"Financials","SOFI":"Financials","PYPL":"Financials",
    "COF":"Financials","USB":"Financials","PNC":"Financials","SPGI":"Financials",
    "MCO":"Financials","SQ":"Financials",
    # Consumer Staples
    "PG":"Consumer Staples","KO":"Consumer Staples","PEP":"Consumer Staples",
    "WMT":"Consumer Staples","COST":"Consumer Staples","PM":"Consumer Staples",
    "MO":"Consumer Staples","MDLZ":"Consumer Staples","CL":"Consumer Staples",
    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy",
    "MPC":"Energy","VLO":"Energy","OXY":"Energy","EOG":"Energy","PSX":"Energy",
    # Industrials
    "BA":"Industrials","GE":"Industrials","CAT":"Industrials","HON":"Industrials",
    "RTX":"Industrials","LMT":"Industrials","FDX":"Industrials","UPS":"Industrials",
    "DE":"Industrials","UAL":"Industrials","NOC":"Industrials","GD":"Industrials",
    "ETN":"Industrials","EMR":"Industrials",
    # Communication Services
    "DIS":"Communication Services","CMCSA":"Communication Services",
    "VZ":"Communication Services","T":"Communication Services",
    "META":"Communication Services","NFLX":"Communication Services",
    "GOOGL":"Communication Services","SPOT":"Communication Services",
    "SNAP":"Communication Services","PINS":"Communication Services",
    # Real Estate
    "AMT":"Real Estate","PLD":"Real Estate","EQIX":"Real Estate",
    # High-beta
    "MARA":"High Beta","RIOT":"High Beta",
    # ── ETFs (granular mapping — sector guard active) ──────────────────────
    "SPY":"Broad Market","IWM":"Small Cap","DIA":"Large Cap Value",
    "QQQ":"Technology","XLK":"Technology",
    "XLF":"Financials","XLE":"Energy","XLI":"Industrials",
    "XLV":"Healthcare","XLY":"Consumer Discretionary",
    "XLP":"Consumer Staples","XLC":"Communication Services",
    "XLRE":"Real Estate","XLU":"Utilities","XLB":"Materials",
    "GLD":"Gold","SLV":"Silver","USO":"Oil",
    "TLT":"Treasury Bonds","IEF":"Intermediate Treasuries",
    "SHY":"Short Treasuries","LQD":"Investment Grade Bonds","HYG":"High Yield Bonds",
    "EFA":"Developed Markets","EEM":"Emerging Markets","FXI":"China","EWJ":"Japan",
    "ARKK":"Innovation","SMH":"Semiconductors","SOXX":"Semiconductors",
    "IBB":"Biotechnology","KRE":"Regional Banks","XBI":"Biotechnology",
}

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
    # General-purpose cache (regime, etc.)
    "cache": {
        "regime": None,   # SPY SMA-200, per-stock 20d returns — refreshed at connect + every 4h
    },
    # Daily near-miss log — tickers that were evaluated but not selected for CSP/LEAP
    "near_miss_log": {
        "date":    None,   # "YYYY-MM-DD" — reset each trading day
        "tickers": {},     # ticker → {type, score, iv_rank, reasons, last_seen, seen_count}
        "digest_sent": False,  # True once end-of-day digest has been emitted today
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
            # Tape sentiment filter — when False: scores collected but never block trades
            "tape_filter_enabled": True,
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
    # Stock Trader state (breakout momentum equity trades — separate from CSP/LEAP AT)
    "stock_trader": {
        "enabled": False,
        "config": {
            "position_size":        3000,    # fixed $ per trade
            "max_positions":        8,       # concurrent cap (90th pct = 5, 18yr backtest)
            "hard_stop_pct":        7.0,     # phase 1 GTC stop (days 1-5)
            "trail_pct":            5.0,     # phase 2 IBKR TRAIL order (days 5-30)
            "max_hold_days":        30,      # force-close at 30 trading days
            "signal_freshness_min": 30,      # skip alerts older than this
            "limit_buffer_pct":     0.10,    # LIMIT = last_price × (1 + buffer/100)
            "rotation_enabled":     False,   # auto-rotate weak positions on new breakout
        },
        "positions":    {},   # ticker → position dict
        "closed_today": [],   # list of closed trade summaries (reset each day)
        "decisions":    [],   # plain-English decision log (last 200)
        "rotation_log": [],   # outcome-tracking for rotation decisions
    },
    "day_trader": {
        "enabled": False,
        "config": {
            "position_size":        2000,    # fixed $ per trade (smaller for day trades)
            "max_positions":        10,      # intraday concurrent cap
            "hard_stop_pct":        7.0,     # intraday stop loss (DAY order)
            "profit_target_pct":    1.5,     # take profit at +1.5% (intraday)
            "force_close_time":     "15:45", # force MKT sell all positions at HH:MM ET
            "signal_freshness_min": 30,      # skip alerts older than this
            "limit_buffer_pct":     0.10,    # LIMIT entry = last_price × (1 + buffer/100)
            "daily_profit_target":  500.0,   # $ goal for the day (for goal calculator)
            "expected_return_pct":  0.8,     # assumed avg win % per trade (for goal math)
            "win_rate_est":         0.56,    # assumed win rate per trade (for goal math)
        },
        "positions":    {},   # ticker → position dict
        "closed_today": [],   # closed trade summaries (reset each day)
        "decisions":    [],   # decision log (last 200)
    },
    "spx_0dte": {
        "enabled": False,
        "config": {
            "daily_profit_target":   200.0,  # $ daily goal
            "spread_width":          25,     # points per spread leg (25 = $2500 max risk/contract)
            "otm_pct":               0.5,    # % OTM for the short strike (~35 pts OTM at SPX 7000)
            "profit_pct":            50.0,   # close IC when this % of credit is realised
            "stop_loss_mult":        2.0,    # close when loss = stop_loss_mult × credit
            "entry_start_time":     "09:45", # earliest IC entry
            "entry_cutoff_time":    "14:00", # no new entries after this (gamma risk after ~14:00)
            "force_close_time":     "15:45", # MKT close all legs regardless
            "max_attempts":          5,      # max IC entries per day
            "max_margin":        20000.0,    # max notional margin to deploy
            "min_credit_per_spread": 0.20,   # skip if net credit < this per spread
        },
        "spreads":        {},   # spread_id → spread dict
        "closed_today":   [],
        "decisions":      [],
        "attempts_today": 0,
        "today_pnl":      0.0,
        "last_stop_time": None,
    },
    # CVD tape sentiment (populated by reqMktData "233,375" subscriptions)
    "tape_sentiment": {},  # ticker → per-ticker CVD state dict
    "tape_subs":      {},  # ticker → ib_insync Ticker object from reqMktData
    # VIX live price (separate Index contract subscription)
    "vix_live":       {"price": None, "prev_close": None, "updated": None},
    # Breakout-scanner watchlist (populated via POST /watchlist/alert)
    "watchlist": {},   # ticker → {signal_type, timestamp_et, price_at_alert, pct_b, rsi, vol_ratio}
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

    # ── Tape sentiment subscription ─────────────────────────────────────────
    # Cap at TAPE_SENTIMENT_MAX_TICKERS to stay within standard market-data lines.
    # Bootstrap avg_vol_per_bar from the historical bars already fetched above.
    if ticker not in state["tape_subs"] and len(state["tape_subs"]) < TAPE_SENTIMENT_MAX_TICKERS:
        try:
            df_vol = _bars_to_df(bars)
            avg_5min = float(df_vol["volume"].mean()) if not df_vol.empty else 0.0
        except Exception:
            avg_5min = 0.0
        avg_per_min = avg_5min / 5.0   # BAR_SIZE is "5 mins"

        if ticker not in state["tape_sentiment"]:
            state["tape_sentiment"][ticker] = _tape_initial_state()
        state["tape_sentiment"][ticker]["avg_vol_per_bar"] = avg_per_min

        tape_tkr = ib.reqMktData(contract, "233,375", False, False)
        state["tape_subs"][ticker]              = tape_tkr
        state["tape_sentiment"][ticker]["sub_active"] = True
        tape_tkr.updateEvent += _make_tape_callback(ticker)
        log.info("Tape sentiment subscribed for %s (avg_vol_per_bar=%.0f)", ticker, avg_per_min)


# ── CVD Tape Sentiment helpers ─────────────────────────────────────────────

def _tape_initial_state() -> dict:
    """Return a zeroed tape sentiment state for a new ticker subscription."""
    return {
        "_last_rtv":          math.nan,  # last tkr.rtTradeVolume processed; nan != nan guards first tick
        "session_vwap":       0.0,
        "session_vol":        0,
        "last_price":         None,
        "last_dir":           0,         # +1 / -1 carry-forward for tick rule
        "bars":               [],        # closed 1-min bars: [{delta, vol, open, close}], max 20
        "cur_bar":            {"delta": 0, "vol": 0, "open": 0.0, "last": 0.0, "minute": -1},
        "price_history":      [],        # last 100 trade prices (for VWAP std-dev)
        "avg_vol_per_bar":    0.0,       # seeded from 5-day bars at subscribe time
        "session_buy_vol":    0,         # cumulative session buy volume (for bar DB rows)
        "session_sell_vol":   0,         # cumulative session sell volume
        "score":              0.0,
        "label":              "NEUTRAL",
        "components":         {"cvd": 0.0, "vwap_z": 0.0, "vol_mag": 0.0, "div_mult": 1.0},
        "last_updated":       None,      # ISO UTC string; None → treated as stale
        "sub_active":         False,
        "_first_tick_logged": False,
    }


def _tape_is_fresh(sent: dict) -> bool:
    """True when tape sentiment data was updated within TAPE_STALENESS_SECS (30 min)."""
    lu = sent.get("last_updated")
    if not lu:
        return False
    try:
        age = (datetime.utcnow() - datetime.fromisoformat(lu)).total_seconds()
        return age < TAPE_STALENESS_SECS
    except Exception:
        return False


def _update_tape_bar(sym: str, price: float, size: int, direction: int) -> None:
    """Accumulate a trade into the current 1-minute bar; close the bar on minute rollover."""
    from zoneinfo import ZoneInfo
    sent = state["tape_sentiment"].get(sym)
    if sent is None:
        return
    try:
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_et = datetime.now()
    minute = now_et.hour * 60 + now_et.minute
    cur = sent["cur_bar"]

    if cur["minute"] != minute:                 # minute boundary — close current bar
        if cur["vol"] > 0:
            closed_bar = {
                "delta": cur["delta"],
                "vol":   cur["vol"],
                "open":  cur["open"],
                "close": cur["last"],
            }
            sent["bars"].append(closed_bar)
            if len(sent["bars"]) > 20:          # rolling window: keep last 20 bars
                sent["bars"].pop(0)
            # Persist the completed bar to tape_data.db for historical analysis
            try:
                bar_score, bar_label, bar_comps = _compute_tape_score(sym)
                bar_start_str = now_et.replace(minute=cur["minute"] % 60,
                                               hour=cur["minute"] // 60,
                                               second=0, microsecond=0).isoformat()
                sess_date = now_et.strftime("%Y-%m-%d")
                bar_bvol  = sent.get("session_buy_vol", 0)
                bar_svol  = sent.get("session_sell_vol", 0)
                _tape_db_insert_bar(
                    sym, closed_bar, bar_start_str, sess_date,
                    bar_bvol, bar_svol,
                    round(bar_score, 4),
                    round(bar_comps.get("vwap_z", 0.0), 4),
                    bar_label,
                )
            except Exception:
                pass
        sent["cur_bar"] = {
            "delta": direction * size, "vol": size,
            "open": price, "last": price, "minute": minute,
        }
    else:
        cur["delta"] += direction * size
        cur["vol"]   += size
        cur["last"]   = price
        if cur["open"] == 0.0:
            cur["open"] = price

    ph = sent["price_history"]
    ph.append(price)
    if len(ph) > 100:
        ph.pop(0)


def _compute_tape_score(sym: str) -> tuple:
    """Compute the 4-component CVD composite score. Returns (score, label, components_dict)."""
    sent = state["tape_sentiment"].get(sym)
    if sent is None:
        return 0.0, "NEUTRAL", {}

    bars = sent["bars"]

    # Component 1: Normalized rolling CVD (45%)
    # Rolling 15 bars; need ≥3 bars — guard fires at open before enough history
    last15    = bars[-15:] if len(bars) >= 3 else []
    total_vol = sum(b["vol"] for b in last15)
    cvd_score = (sum(b["delta"] for b in last15) / total_vol
                 if last15 and total_vol > 0 else 0.0)
    cvd_score = float(np.clip(cvd_score, -1.0, 1.0))

    # Component 2: VWAP deviation z-score (30%)
    # Guard: std < 0.01 or < 10 price samples → 0.0 (prevents divide-by-zero at open)
    ph   = sent["price_history"]
    vwap = sent.get("session_vwap", 0.0)
    lp   = sent.get("last_price") or vwap
    if len(ph) >= 10 and vwap > 0 and lp:
        std = float(np.std(ph[-20:])) if len(ph) >= 20 else float(np.std(ph))
        vwap_z = (float(np.clip((lp - vwap) / std, -1.0, 1.0))
                  if std >= 0.01 else 0.0)
    else:
        vwap_z = 0.0

    # Component 3: Volume-weighted delta magnitude (15%)
    # TOD normalization: 3-bucket multiplier adjusts for intraday volume bias
    # v1: 3 buckets (open/close 2×, lunch 0.6×, normal 1×). Graduated curve deferred to v2.
    try:
        from zoneinfo import ZoneInfo
        hour = datetime.now(ZoneInfo("America/New_York")).hour
    except Exception:
        hour = datetime.now().hour
    tod = 2.0 if (hour < 10 or hour >= 15) else 0.6 if (12 <= hour < 13) else 1.0
    adj_avg   = sent.get("avg_vol_per_bar", 0.0) * tod
    cur       = sent["cur_bar"]
    cur_ratio = cur["delta"] / cur["vol"] if cur["vol"] > 0 else 0.0
    vol_mag   = (float(np.clip(cur_ratio * min(2.0, cur["vol"] / adj_avg), -1.0, 1.0))
                 if adj_avg > 0 else 0.0)

    # Component 4: Divergence multiplier (binary, v1)
    # Price up + CVD down = buyer exhaustion. Price down + CVD up = seller absorption.
    # Guard: fewer than 5 bars (market just opened) → no divergence penalty
    # v1: binary 0.5/1.0. Graduated version using corr(price_returns, bar_deltas) deferred to v2.
    if len(bars) < 5:
        div_mult = 1.0
    else:
        price_up = bars[-1]["close"] > bars[-5]["close"]
        cvd_running, cvd_series = 0, []
        for b in bars[-5:]:
            cvd_running += b["delta"]
            cvd_series.append(cvd_running)
        cvd_up   = cvd_series[-1] > cvd_series[0]
        div_mult = 0.5 if (price_up != cvd_up) else 1.0

    composite = 0.45 * cvd_score + 0.30 * vwap_z + 0.15 * vol_mag
    score     = float(np.clip(composite * div_mult, -1.0, 1.0))

    if   score >  0.50: label = "STRONGLY BULLISH"
    elif score >  0.25: label = "BULLISH"
    elif score < -0.50: label = "STRONGLY BEARISH"
    elif score < -0.25: label = "BEARISH"
    else:               label = "NEUTRAL"

    return round(score, 4), label, {
        "cvd":      round(cvd_score, 4),
        "vwap_z":   round(vwap_z,   4),
        "vol_mag":  round(vol_mag,  4),
        "div_mult": div_mult,
    }


def _make_tape_callback(sym: str):
    """Factory: returns an on_tape_update closure bound to sym for updateEvent registration."""
    def on_tape_update(tkr) -> None:
        # ── Dedup on rtTradeVolume sentinel ──────────────────────────────────
        # nan != nan in Python — so first tick (cur_rtv is a real number,
        # _last_rtv is nan) always passes through. This is the correct behaviour.
        cur_rtv = tkr.rtTradeVolume
        if cur_rtv is None or math.isnan(cur_rtv):
            return
        sent = state["tape_sentiment"].get(sym)
        if sent is None:
            return
        if cur_rtv == sent.get("_last_rtv", math.nan):
            return                          # same trade; callback fired for another field
        sent["_last_rtv"] = cur_rtv

        # ── Price and size ────────────────────────────────────────────────────
        price = tkr.last
        size  = tkr.lastSize          # typed float in ib_insync Ticker; cast to int below
        if price is None or math.isnan(price) or price <= 0:
            return
        if size  is None or math.isnan(size)  or size  <= 0:
            # size <= 0 also filters odd-lot prints (lastSize == 0 for < 100-share fills)
            return

        # ── Session VWAP and volume from IBKR's own calculation (tick 233) ──
        if tkr.vwap and not math.isnan(tkr.vwap) and tkr.vwap > 0:
            sent["session_vwap"] = tkr.vwap
        if tkr.volume and not math.isnan(tkr.volume):
            sent["session_vol"] = int(tkr.volume)

        # ── Live NBBO — read directly from Ticker, no stale prev_bid/ask lag ─
        bid = tkr.bid if tkr.bid and not math.isnan(tkr.bid) and tkr.bid > 0 else None
        ask = tkr.ask if tkr.ask and not math.isnan(tkr.ask) and tkr.ask > 0 else None

        # ── Quote rule → tick rule fallback ──────────────────────────────────
        lp = sent.get("last_price")
        if ask and price >= ask:
            direction = 1
        elif bid and price <= bid:
            direction = -1
        elif lp and price > lp:
            direction = 1;  sent["last_dir"] = 1
        elif lp and price < lp:
            direction = -1; sent["last_dir"] = -1
        else:
            direction = sent.get("last_dir", 0)

        sent["last_price"] = price
        if direction >= 0:
            sent["session_buy_vol"]  += int(size)
        else:
            sent["session_sell_vol"] += int(size)

        # ── First-tick runtime field-mapping verification ────────────────────
        if not sent.get("_first_tick_logged"):
            log.debug(
                "Tape first tick [%s]: last=%.2f lastSize=%d vwap=%.4f vol=%d rtv=%.0f",
                sym, price, int(size),
                sent.get("session_vwap", 0), sent.get("session_vol", 0), cur_rtv,
            )
            sent["_first_tick_logged"] = True

        # ── Update 1-minute bar buffer and recompute score ────────────────────
        _update_tape_bar(sym, price, int(size), direction)
        score, label, components = _compute_tape_score(sym)
        sent["score"]      = score
        sent["label"]      = label
        sent["components"] = components
        sent["last_updated"] = datetime.utcnow().isoformat()

        # ── Real-time block capture for institutional flow report ─────────────
        if int(size) >= TAPE_BLOCK_MIN_SHARES:
            _tape_db_insert_block_immediate(
                sym, price, int(size), direction,
                sent.get("session_vwap", 0.0), score,
            )

    return on_tape_update


async def _subscribe_pending(ib: IB, known: set) -> set:
    with _tickers_lock:
        current = set(TICKERS)
    for ticker in current - known:
        try:
            await subscribe_ticker(ib, ticker)
        except Exception as e:
            log.warning(f"Subscribe failed [{ticker}]: {e}")
    return current


async def _tape_preseed_subscribe(ib: IB, ticker: str) -> None:
    """Tape-only subscription for pre-seeding — no keepUpToDate bars, just CVD sentiment.
    Does NOT add to TICKERS or state['subscriptions']; only fills a tape slot."""
    if ticker in state["tape_subs"] or len(state["tape_subs"]) >= TAPE_SENTIMENT_MAX_TICKERS:
        return
    contract = Stock(ticker, "SMART", "USD")
    await ib.qualifyContractsAsync(contract)

    # Snapshot history for avg_vol_per_bar bootstrap (keepUpToDate=False — no persistent sub)
    avg_per_min = 0.0
    try:
        snap = await ib.reqHistoricalDataAsync(
            contract, endDateTime="", durationStr="5 D",
            barSizeSetting="5 mins", whatToShow="TRADES",
            useRTH=True, formatDate=1, keepUpToDate=False,
        )
        if snap:
            df_v = _bars_to_df(snap)
            avg_per_min = float(df_v["volume"].mean()) / 5.0 if not df_v.empty else 0.0
    except Exception:
        pass

    if ticker not in state["tape_sentiment"]:
        state["tape_sentiment"][ticker] = _tape_initial_state()
    state["tape_sentiment"][ticker]["avg_vol_per_bar"] = avg_per_min

    tape_tkr = ib.reqMktData(contract, "233,375", False, False)
    state["tape_subs"][ticker]                    = tape_tkr
    state["tape_sentiment"][ticker]["sub_active"] = True
    tape_tkr.updateEvent += _make_tape_callback(ticker)
    log.info("Tape pre-seeded [%s]  avg_vol_per_bar=%.0f", ticker, avg_per_min)


async def _subscribe_vix(ib: IB) -> None:
    """Subscribe to live VIX index price via IBKR Index contract.
    Stores price in state['vix_live']. Falls back silently if CBOE data is unavailable."""
    try:
        vix_contract = Index("VIX", "CBOE")
        await ib.qualifyContractsAsync(vix_contract)

        # Fetch previous close from 2-day history (snapshot only)
        prev_close = None
        try:
            snap = await ib.reqHistoricalDataAsync(
                vix_contract, endDateTime="", durationStr="2 D",
                barSizeSetting="1 day", whatToShow="TRADES",
                useRTH=True, formatDate=1, keepUpToDate=False,
            )
            if snap and len(snap) >= 2:
                prev_close = round(float(snap[-2].close), 2)
            elif snap:
                prev_close = round(float(snap[-1].close), 2)
        except Exception:
            pass

        vix_tkr = ib.reqMktData(vix_contract, "")

        def _on_vix(t):
            price = None
            for attr in ("last", "close", "marketPrice"):
                v = getattr(t, attr, None)
                if v is not None:
                    try:
                        f = float(v)
                        if not math.isnan(f) and f > 0:
                            price = round(f, 2)
                            break
                    except (TypeError, ValueError):
                        pass
            if price:
                state["vix_live"]["price"]      = price
                state["vix_live"]["updated"]    = datetime.utcnow().isoformat()
                if prev_close:
                    state["vix_live"]["prev_close"] = prev_close

        vix_tkr.updateEvent += _on_vix
        state["vix_live"]["_sub"] = vix_tkr   # keep reference alive
        log.info("VIX Index subscribed via IBKR (CBOE)")
    except Exception as exc:
        log.warning("VIX IBKR subscription failed (will use yfinance fallback): %s", exc)


async def _tape_preseed(ib: IB) -> None:
    """Fill remaining tape subscription slots from a priority-ordered candidate list:

      0. Index ETFs (SPY, QQQ, DIA, IWM) — always subscribed first for the banner
      1. Active auto-trader positions     — tickers with live trades
      2. Breakout scanner watchlist       — recently alerted candidates
      3. Top-ranked universe tickers      — sorted by XGBoost score

    Called once after initial connect and again after each 1102 reconnect so block
    flow capture starts at market open without waiting for the scanner to run.
    Is a no-op when all TAPE_SENTIMENT_MAX_TICKERS slots are already filled.
    """
    slots_free = TAPE_SENTIMENT_MAX_TICKERS - len(state["tape_subs"])
    if slots_free <= 0:
        return

    seen:    set  = set(state["tape_subs"].keys())
    ordered: list = []

    def _enqueue(ticker: str) -> None:
        t = (ticker or "").strip().upper()
        if t and t not in seen:
            seen.add(t)
            ordered.append(t)

    # 0 — index ETFs always first (live banner prices)
    for t in INDEX_ETF_TICKERS:
        _enqueue(t)

    # 1 — active positions (money on the line — always monitor these)
    for info in state["autotrader"].get("positions", {}).values():
        _enqueue(info.get("ticker", ""))

    # 2 — breakout watchlist (dict keyed by ticker string)
    for tk in state.get("watchlist", {}).keys():
        _enqueue(tk)

    # 3 — universe top-N by score (already sorted desc when screened)
    for row in state.get("universe_scores", []):
        _enqueue(row.get("ticker", "") if isinstance(row, dict) else str(row))

    candidates = ordered[:slots_free]
    if not candidates:
        log.info("Tape pre-seed: no new candidates (positions=%d watchlist=%d universe=%d)",
                 len(state["autotrader"].get("positions", {})),
                 len(state.get("watchlist", {})),
                 len(state.get("universe_scores", [])))
        return

    log.info("Tape pre-seed: %d slot(s) free → subscribing %s", slots_free, candidates)
    for ticker in candidates:
        if len(state["tape_subs"]) >= TAPE_SENTIMENT_MAX_TICKERS:
            break
        try:
            await _tape_preseed_subscribe(ib, ticker)
        except Exception as exc:
            log.warning("Tape pre-seed failed [%s]: %s", ticker, exc)


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
                    # Tape subscriptions are on the same connection — mark inactive so
                    # subscribe_ticker re-subscribes them as tickers are re-added.
                    state["tape_subs"].clear()
                    for _ts in state["tape_sentiment"].values():
                        _ts["sub_active"] = False
                    state["vix_live"]["_sub"] = None   # VIX sub lost on reconnect
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
            # Pre-seed tape CVD slots from positions → watchlist → universe on connect.
            # Runs before first scan so block flow capture starts at market open.
            await _tape_preseed(ib)
            # Subscribe VIX as a live Index contract for the index banner.
            await _subscribe_vix(ib)
            # Retry once — occasionally the first snapshot arrives before OPRA feed is warm
            if not state["opra_active"]:
                await asyncio.sleep(3)
                state["opra_active"] = await _check_opra_subscription(ib)
            # Prime regime cache on connect so filters are active from first scan.
            _loop = asyncio.get_event_loop()
            await _loop.run_in_executor(None, _update_regime_cache_sync)

            # Heartbeat: keep usopt farm alive + re-check OPRA every 4 min.
            # Without this, the options data farm idles and the first scan
            # after a quiet period gets NaN on every contract.
            _heartbeat_tick = 0
            while ib.isConnected():
                ctx["known"] = await _subscribe_pending(ib, ctx["known"])
                # After 1102 reconnect tape_subs is cleared — re-preseed open slots.
                # Also fills slots lazily as universe scores update throughout the day.
                if len(state["tape_subs"]) < TAPE_SENTIMENT_MAX_TICKERS:
                    await _tape_preseed(ib)
                # Re-subscribe VIX if the 1102 reconnect cleared it.
                if state["vix_live"].get("_sub") is None:
                    await _subscribe_vix(ib)
                _heartbeat_tick += 1
                if _heartbeat_tick % 24 == 0:   # every 24 × 10 s = 4 min
                    state["opra_active"] = await _check_opra_subscription(ib)
                if _heartbeat_tick % 1440 == 120:  # every ~4 h (staggered 20 min after OPRA)
                    await _loop.run_in_executor(None, _update_regime_cache_sync)
                # End-of-day near-miss digest at 16:05 ET (once per day)
                from zoneinfo import ZoneInfo as _ZI
                _now_et = datetime.now(_ZI("America/New_York"))
                if (_now_et.hour == 16 and _now_et.minute == 5
                        and not state["near_miss_log"].get("digest_sent", False)):
                    _emit_eod_digest()
                if _now_et.hour == 16 and _now_et.minute == 15:
                    _today_str = _now_et.strftime("%Y-%m-%d")
                    if not state.get("_perf_enrich_date") == _today_str:
                        state["_perf_enrich_date"] = _today_str
                        asyncio.create_task(_enrich_then_notify(_today_str))
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

def _watchlist_save() -> None:
    try:
        with open(WATCHLIST_PATH, "w") as f:
            json.dump(state["watchlist"], f, indent=2, default=str)
    except Exception as e:
        log.warning("Watchlist save failed: %s", e)


def _watchlist_load() -> None:
    if not os.path.exists(WATCHLIST_PATH):
        return
    try:
        with open(WATCHLIST_PATH, "r") as f:
            state["watchlist"] = json.load(f)
        log.info("Watchlist restored: %d entries", len(state["watchlist"]))
        _sync_watchlist_to_alert_history()
    except Exception as e:
        log.warning("Watchlist load failed: %s", e)


def _sync_watchlist_to_alert_history() -> None:
    """Backfill alert_history from watchlist.json entries missed due to restarts."""
    try:
        wl = state.get("watchlist", {})
        if not wl:
            return
        from zoneinfo import ZoneInfo as _ZI
        from datetime import timezone as _utctz
        et = _ZI("America/New_York")
        con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
        existing = {
            f"{r[0]}|{r[1]}"
            for r in con.execute("SELECT session_date, ticker FROM alert_history").fetchall()
        }
        added = 0
        for entry in wl.values():
            tk = entry.get("ticker", "")
            added_iso = entry.get("added_iso", "")
            if not tk or not added_iso:
                continue
            try:
                dt = datetime.fromisoformat(added_iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_utctz.utc)
                session_date = dt.astimezone(et).strftime("%Y-%m-%d")
            except Exception:
                session_date = added_iso[:10]
            key = f"{session_date}|{tk}"
            if key in existing:
                continue
            con.execute(
                """INSERT INTO alert_history
                   (fired_at, session_date, ticker, signal_type, price,
                    pct_b, rsi, vol_ratio, tape_score, tape_label)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (added_iso, session_date, tk,
                 entry.get("signal_type"), entry.get("price_at_alert"),
                 entry.get("pct_b"), entry.get("rsi"), entry.get("vol_ratio"),
                 entry.get("tape_score"), entry.get("tape_label")),
            )
            existing.add(key)
            added += 1
        con.commit()
        con.close()
        if added:
            log.info("Synced %d watchlist entries → alert_history", added)
    except Exception as exc:
        log.warning("_sync_watchlist_to_alert_history failed: %s", exc)


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
    """Background task: refresh universe daily at 08:30 ET on weekdays.

    If the backend restarts after 08:30 ET and the universe cache is empty or
    stale (no screen today), trigger an immediate screen so the auto-trader
    has candidates without waiting until the next morning.
    """
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")

    # On startup: if today's screen was missed (cache empty or dated before today),
    # run one immediately in the background rather than sleeping until tomorrow.
    await asyncio.sleep(30)   # brief delay so IBKR connection is stable first
    _now = datetime.now(ET)
    _today_str = _now.strftime("%Y-%m-%d")
    _last_screened = state.get("universe_last_screened") or ""
    if _now.weekday() < 5 and (not CSP_UNIVERSE or _today_str not in _last_screened):
        log.info("Universe scheduler: missed today's 08:30 screen — running now")
        try:
            tickers = await _screen_universe()
            if tickers:
                CSP_UNIVERSE.clear()
                CSP_UNIVERSE.extend(tickers)
                state["scan_cache"]["csp"]   = None
                state["scan_cache"]["leaps"] = None
                _universe_save(tickers)
                log.info("Catch-up universe screen complete: %d tickers", len(CSP_UNIVERSE))
        except Exception as exc:
            log.error("Catch-up universe screen failed: %s", exc)

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


# ── Market regime cache ────────────────────────────────────────────────────

def _update_regime_cache_sync() -> None:
    """Compute SPY SMA-200, 20d returns (SPY + universe stocks) and store in state.

    Called once at connect and every 4 h from the streaming heartbeat.
    All filters read state['cache']['regime'] — stale beats missing.
    """
    try:
        spy_hist  = yf.Ticker("SPY").history(period="1y")
        spy_c     = spy_hist["Close"].astype(float)
        spy_sma200 = float(spy_c.rolling(200).mean().iloc[-1])
        spy_price  = float(spy_c.iloc[-1])
        spy_ret20  = float((spy_c.iloc[-1] / spy_c.iloc[-21] - 1) * 100) \
                     if len(spy_c) >= 21 else 0.0
    except Exception as exc:
        log.warning("Regime cache: SPY download failed: %s", exc)
        return   # leave existing cache intact rather than clobber with bad data

    # Bulk-download 30d of closes for every universe ticker to get 20d returns
    tickers = list(CSP_UNIVERSE)
    stock_ret20: dict[str, float] = {}
    try:
        bulk = yf.download(tickers, period="30d", auto_adjust=True,
                           progress=False, threads=True)
        close_df = bulk["Close"] if isinstance(bulk.columns, pd.MultiIndex) else bulk
        for t in tickers:
            if t not in close_df.columns:
                continue
            c = close_df[t].dropna()
            if len(c) >= 21:
                stock_ret20[t] = round(float((c.iloc[-1] / c.iloc[-21] - 1) * 100), 2)
    except Exception as exc:
        log.warning("Regime cache: bulk stock download failed: %s", exc)

    state["cache"]["regime"] = {
        "spy_price":        round(spy_price, 2),
        "spy_sma200":       round(spy_sma200, 2),
        "spy_above_sma200": spy_price > spy_sma200,
        "spy_ret20":        round(spy_ret20, 2),
        "stock_ret20":      stock_ret20,
        "updated":          datetime.now().isoformat(),
    }
    bull = "BULL" if spy_price > spy_sma200 else "BEAR"
    log.info("Regime cache refreshed: SPY=%.2f SMA200=%.2f [%s] spy_ret20=%.2f%% %d stocks",
             spy_price, spy_sma200, bull, spy_ret20, len(stock_ret20))


# ── Near-miss tracking ────────────────────────────────────────────────────

def _record_near_miss(ticker: str, strategy: str, score: float, iv_rank: float,
                      reasons: list[str]) -> None:
    """Accumulate per-ticker CSP/LEAP rejection reasons for the daily near-miss log.

    Called from both filter functions every scan cycle. The same ticker may appear
    multiple times across cycles — reasons are merged (unique), counts incremented.
    Resets automatically at the start of each new trading day.
    """
    from zoneinfo import ZoneInfo
    now_et = datetime.now(ZoneInfo("America/New_York"))
    today  = now_et.strftime("%Y-%m-%d")
    nm     = state["near_miss_log"]

    if nm.get("date") != today:
        nm["date"]         = today
        nm["tickers"]      = {}
        nm["digest_sent"]  = False

    entry = nm["tickers"].setdefault(ticker, {
        "type":       strategy,
        "score":      score,
        "iv_rank":    iv_rank,
        "reasons":    [],
        "last_seen":  None,
        "seen_count": 0,
    })
    # Merge reasons (keep unique, preserve order)
    for r in reasons:
        if r not in entry["reasons"]:
            entry["reasons"].append(r)
    entry["last_seen"]  = now_et.strftime("%H:%M ET")
    entry["seen_count"] += 1
    if score > entry.get("score", 0):
        entry["score"]   = score
        entry["iv_rank"] = iv_rank
        entry["type"]    = strategy


def _emit_eod_digest() -> None:
    """Log the end-of-day near-miss digest and mark it sent for today."""
    from zoneinfo import ZoneInfo
    nm = state["near_miss_log"]
    if not nm.get("tickers"):
        _at_log("EOD-DIGEST", "No near-miss tickers today — all scans were empty or positions full.")
        nm["digest_sent"] = True
        return

    tickers = nm["tickers"]
    lines = ["End-of-day near-miss digest — tickers evaluated but not selected:\n"]
    for tk, d in sorted(tickers.items(), key=lambda x: x[1].get("score", 0), reverse=True):
        reason_str = " | ".join(d["reasons"]) if d["reasons"] else "passed filters — no capital slot"
        lines.append(
            f"  {tk:6s} [{d['type'].upper():4s}] "
            f"score={d['score']:.0f}  iv={d['iv_rank']:.0f}  "
            f"seen={d['seen_count']}×  last={d['last_seen']}  "
            f"→ {reason_str}"
        )
    _at_log("EOD-DIGEST", "\n".join(lines))
    log.info("[EOD-DIGEST] %d near-miss tickers logged", len(tickers))
    nm["digest_sent"] = True


# ── Recommendation filters (mirrors frontend JS) ──────────────────────────

def _filter_csp_recommended(candidates: list, log_diag: bool = False) -> list:
    """Filter scan candidates to recommended CSPs. Set log_diag=True to emit AT diagnostics."""
    def _excl(r): return any("excl. recommended" in w for w in r.get("warnings", []))

    # ── Per-ticker rejection pass — builds reasons for near-miss log ──────────
    clean = []
    for r in candidates:
        tk      = r["ticker"]
        score   = r.get("score", 0)
        iv_rank = r.get("iv_rank", 0)
        reasons: list[str] = []

        iv_min = IV_RANK_MIN_CSP_ETF if tk in _CSP_ETF_TICKERS else IV_RANK_MIN_CSP
        if _excl(r):
            w = next((w for w in r.get("warnings", []) if "excl. recommended" in w), "screening warning")
            reasons.append(f"warning: {w[:80]}")
        elif r["liquidity_score"] < 50:
            reasons.append(f"liquidity={r['liquidity_score']:.0f} (< 50)")
        elif iv_rank < iv_min:
            reasons.append(f"iv_rank={iv_rank:.0f} (< {iv_min} — not enough premium)")
        elif score < 70:
            reasons.append(f"score={score:.0f} (< 70)")
        elif r.get("above_sma50") is False or r.get("above_sma20") is False or r.get("above_sma200") is False:
            below = [n for n, k in [("SMA20","above_sma20"),("SMA50","above_sma50"),("SMA200","above_sma200")]
                     if r.get(k) is False]
            reasons.append(f"below {'+'.join(below)} — downtrend")
        elif r.get("earnings_days_out") is not None and r["earnings_days_out"] <= EARNINGS_BLOCK_DAYS + 7:
            reasons.append(f"earnings in {r['earnings_days_out']}d (block ≤ {EARNINGS_BLOCK_DAYS+7}d)")

        if reasons:
            if log_diag:
                _at_log("NEAR-MISS",
                    f"CSP skip {tk}: score={score:.0f} iv={iv_rank:.0f} liq={r['liquidity_score']:.0f}"
                    f" → {' | '.join(reasons)}")
            _record_near_miss(tk, "csp", score, iv_rank, reasons)
        else:
            clean.append(r)

    if log_diag:
        _at_log("SCAN",
            f"CSP filter: {len(candidates)} raw → {len(clean)} passed "
            f"({len(candidates)-len(clean)} skipped — see NEAR-MISS log for details)")

    # ── Tape sentiment filter (CSP: block when score < −0.30 and data is fresh) ─
    if state.get("autotrader", {}).get("config", {}).get("tape_filter_enabled", True):
        tape_clean = []
        for r in clean:
            sent       = state["tape_sentiment"].get(r["ticker"], {})
            tape_score = sent.get("score", 0.0)
            if _tape_is_fresh(sent) and tape_score < -0.30:
                reason = f"tape={tape_score:+.2f} {sent.get('label','?')} (< -0.30, net sellers)"
                if log_diag:
                    _at_log("NEAR-MISS", f"CSP skip {r['ticker']}: {reason}")
                _record_near_miss(r["ticker"], "csp", r.get("score", 0), r.get("iv_rank", 0), [reason])
            else:
                tape_clean.append(r)
        clean = tape_clean

    # ── Market regime gate (F1): SPY > SMA-200 AND VIX < 25 ──────────────────
    # Selling puts in a bear market or panic spike carries outsized assignment risk.
    # Gate is advisory when regime data is missing — never block on absence of data.
    regime = state["cache"].get("regime") or {}
    if regime:
        spy_above = regime.get("spy_above_sma200")
        if spy_above is False:
            if log_diag:
                _at_log("REGIME",
                    f"CSP gate: SPY ${regime['spy_price']} below SMA-200 ${regime['spy_sma200']:.0f} "
                    f"— bear market confirmed, no new CSP entries")
            return []

    vix_price = state["vix_live"].get("price") or regime.get("vix_price")
    if vix_price is not None and vix_price >= 25:
        if log_diag:
            _at_log("REGIME",
                f"CSP gate: VIX={vix_price:.1f} ≥ 25 — fear spike, no new CSP entries")
        return []

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
    # ── Per-ticker rejection pass ─────────────────────────────────────────────
    clean = []
    for r in candidates:
        tk      = r["ticker"]
        score   = r.get("score", 0)
        iv_rank = r.get("iv_rank", 50)
        reasons: list[str] = []

        if any("excl. recommended" in w for w in r.get("warnings", [])):
            w = next((w for w in r.get("warnings", []) if "excl. recommended" in w), "screening warning")
            reasons.append(f"warning: {w[:80]}")
        elif r["liquidity_score"] < 60:
            reasons.append(f"liquidity={r['liquidity_score']:.0f} (< 60 for LEAP)")
        elif iv_rank > 75:
            reasons.append(f"iv_rank={iv_rank:.0f} (> 75 — IV too high to buy calls)")

        if reasons:
            log.info("NEAR-MISS LEAP skip %s: score=%.0f iv=%.0f → %s",
                     tk, score, iv_rank, " | ".join(reasons))
            _record_near_miss(tk, "leap", score, iv_rank, reasons)
        else:
            clean.append(r)

    # ── Tape sentiment filter (LEAP: block when score < 0.10 and data is fresh) ─
    if state.get("autotrader", {}).get("config", {}).get("tape_filter_enabled", True):
        tape_clean = []
        for r in clean:
            sent       = state["tape_sentiment"].get(r["ticker"], {})
            tape_score = sent.get("score", 0.0)
            if _tape_is_fresh(sent) and tape_score < 0.10:
                reason = f"tape={tape_score:+.2f} {sent.get('label','?')} (< 0.10, needs bullish tape)"
                log.info("NEAR-MISS LEAP skip %s: %s", r["ticker"], reason)
                _record_near_miss(r["ticker"], "leap", r.get("score", 0), r.get("iv_rank", 50), [reason])
            else:
                tape_clean.append(r)
        clean = tape_clean

    # ── Market regime gate (F1): SPY > SMA-200 AND VIX < 25 ──────────────────
    # Buying long-dated calls in a bear market burns premium into a falling tape.
    regime = state["cache"].get("regime") or {}
    if regime:
        spy_above = regime.get("spy_above_sma200")
        if spy_above is False:
            reason = (f"regime: SPY ${regime['spy_price']:.2f} below SMA-200 "
                      f"${regime['spy_sma200']:.0f} — bear market")
            log.info("LEAP regime gate: %s — no new LEAP entries", reason)
            for r in clean:
                _record_near_miss(r["ticker"], "leap", r.get("score",0), r.get("iv_rank",50), [reason])
            return []

    vix_price = state["vix_live"].get("price") or regime.get("vix_price")
    if vix_price is not None and vix_price >= 25:
        reason = f"VIX={vix_price:.1f} ≥ 25 — fear spike"
        log.info("LEAP VIX gate: %s — no new LEAP entries", reason)
        for r in clean:
            _record_near_miss(r["ticker"], "leap", r.get("score",0), r.get("iv_rank",50), [reason])
        return []

    # ── Relative strength gate (F2): stock 20d return must beat SPY 20d return ─
    stock_ret20 = regime.get("stock_ret20", {})
    spy_ret20   = regime.get("spy_ret20")
    if stock_ret20 and spy_ret20 is not None:
        rs_clean = []
        for r in clean:
            s_ret = stock_ret20.get(r["ticker"])
            if s_ret is not None and s_ret < spy_ret20:
                reason = (f"rel-strength: {r['ticker']} 20d={s_ret:.1f}% "
                          f"< SPY 20d={spy_ret20:.1f}%")
                log.info("NEAR-MISS LEAP skip %s: %s", r["ticker"], reason)
                _record_near_miss(r["ticker"], "leap", r.get("score",0), r.get("iv_rank",50), [reason])
            else:
                rs_clean.append(r)
        clean = rs_clean

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
        # Build fresh Option contracts with exchange="SMART" to avoid error 321
        # ("Please enter exchange") on portfolio contracts that come back with
        # exchange="" after a reconnect.  Preserve conId for unambiguous resolution.
        def _fresh_contract(c):
            fc = Option(
                c.symbol,
                c.lastTradeDateOrContractMonth,
                float(c.strike), c.right,
                "SMART", "USD", "100",
            )
            if c.conId:
                fc.conId = c.conId
            return fc
        _iv_fresh = {_k: _fresh_contract(_c) for _k, _c in iv_refresh_pairs}
        _iv_tks   = {_k: ib.reqMktData(fc, "106", False, False)
                     for _k, fc in _iv_fresh.items()}
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
            ib.cancelMktData(_iv_fresh[_k])
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

        # ── 1.5. LEAP partial profit: lock in 50% at 100% gain (2× cost) ────
        # max_profit stores entry_price × qty × 100 (the cost paid for the LEAP).
        # When unrealized P&L equals that cost the position has doubled — close
        # half to bank the gain, let the other half ride the directional thesis.
        # partial_taken flag prevents re-triggering once the partial is placed.
        if (action == "BUY"
                and info.get("strategy_type") != "hedge"
                and not info.get("partial_taken")
                and upnl >= max_profit
                and max_profit > 0):
            _pos_qty = max(1, abs(int(item.position)))
            half_qty = _pos_qty // 2 if _pos_qty > 1 else 0
            if half_qty:
                _at_log("PARTIAL",
                        f"{key}: LEAP doubled — partial close {half_qty}/{_pos_qty} contracts "
                        f"(upnl=${upnl:.0f} >= cost=${max_profit:.0f})")
                try:
                    _c_p = item.contract
                    _con_p = Option(
                        symbol=_c_p.symbol,
                        lastTradeDateOrContractMonth=_c_p.lastTradeDateOrContractMonth,
                        strike=float(_c_p.strike), right=_c_p.right,
                        exchange="SMART", currency="USD", multiplier="100",
                    )
                    await ib.qualifyContractsAsync(_con_p)
                    _tk_p = ib.reqMktData(_con_p, "", False, False)
                    await asyncio.sleep(2)
                    _bid_p = float(_tk_p.bid or 0)
                    _ask_p = float(_tk_p.ask or 0)
                    ib.cancelMktData(_con_p)
                    if _bid_p > 0 or _ask_p > 0:
                        _lmt_p = round((_bid_p - 0.01) if _bid_p > 0 else _ask_p * 0.90, 2)
                        ib.placeOrder(_con_p, LimitOrder("SELL", half_qty, _lmt_p))
                        info["partial_taken"] = True
                        _at_save_state()
                        _at_log("PARTIAL",
                                f"{key}: SELL {half_qty}x @ ${_lmt_p:.2f} placed — "
                                f"{_pos_qty - half_qty} contracts remain")
                    else:
                        _at_log("WARN", f"{key}: partial close skipped — no market data")
                except Exception as _pe:
                    _at_log("WARN", f"{key}: partial close failed: {_pe}")
                continue   # skip stop-loss check this cycle; next cycle manages the remainder

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
        if expiry_date <= today and key not in live_keys:
            # <= catches same-day expiries: options are delisted at settlement
            # so they disappear from portfolio() on expiration day itself
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

    # ── #15: CSP assignment handler ───────────────────────────────────────────
    # If a short put was assigned, IBKR delivers 100 shares per contract and the
    # option disappears from portfolio().  The resulting STK position won't match
    # any at["positions"] entry.  We log the event once per ticker per session
    # and record it in the decisions log — manual covered-call entry is the next step.
    _live_stk = {
        item.contract.symbol: int(item.position)
        for item in ib.portfolio()
        if getattr(item.contract, "secType", "") == "STK"
    }
    _at_tickers = {
        info.get("ticker", key.split("_")[0])
        for key, info in at["positions"].items()
    }
    for _stk, _qty in _live_stk.items():
        _assign_key = f"_assigned_{_stk}"
        if _stk not in _at_tickers and not at.get(_assign_key):
            at[_assign_key] = datetime.utcnow().isoformat()
            _at_log("ASSIGN",
                    f"{_stk}: {_qty} shares in IBKR portfolio not tracked by AT — "
                    f"likely CSP assignment. Review in TWS and consider selling a covered call.")
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
        for tk in cooled:
            _record_near_miss(tk, next((c["_type"] for c in candidates if c["ticker"]==tk), "csp"),
                              next((c.get("score",0) for c in candidates if c["ticker"]==tk), 0),
                              next((c.get("iv_rank",0) for c in candidates if c["ticker"]==tk), 0),
                              ["48h stop-loss cooldown"])

    already_active = [c["ticker"] for c in candidates if c["ticker"] in active_t]
    if already_active:
        _at_log("SCAN", f"Already active — skipping: {already_active}")

    candidates = [c for c in candidates
                  if c["ticker"] not in active_t and not _in_cooldown(c["ticker"])]
    _at_log("SCAN", f"Found {len(candidates)} qualifying candidates after dedup + cooldown")

    # Tickers that passed ALL filters but exceed available slots → record as no-slot near-miss
    if candidates[slots:]:
        no_slot = candidates[slots:]
        _at_log("SCAN", f"No capital slot for {len(no_slot)} qualified candidates: "
                        f"{[c['ticker'] for c in no_slot]}")
        for c in no_slot:
            _record_near_miss(c["ticker"], c.get("_type", "csp"),
                              c.get("score", 0), c.get("iv_rank", 0),
                              ["passed all filters — no capital slot available"])

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
            except _FlowAbort:
                pass   # already logged as SKIP by _autotrader_place_coro
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
                except _FlowAbort:
                    pass   # already logged as SKIP
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
    # ── Tape sentiment gate — re-check CVD score at execution time ────────────
    # The scan snapshot can be minutes old; tape sentiment reflects real-time order flow.
    # Tape never blocks when data is absent or stale — only blocks on confirmed bearish flow.
    if state["autotrader"].get("config", {}).get("tape_filter_enabled", True):
        _sent      = state["tape_sentiment"].get(ticker, {})
        _tape_score = _sent.get("score", None)
        if _tape_score is not None and _tape_is_fresh(_sent):
            _tape_label = _sent.get("label", "NEUTRAL")
            if t == "csp" and _tape_score < -0.30:
                _abort_msg = (
                    f"Tape bearish at execution: CVD score {_tape_score:+.4f} ({_tape_label}) "
                    f"— net sellers in control. Aborting CSP {ticker} ${strike}P"
                )
                _at_log("SKIP", f"Tape gate: {_abort_msg}")
                raise _FlowAbort(_abort_msg)
            elif t == "leap" and _tape_score < 0.10:
                _abort_msg = (
                    f"Tape not bullish at execution: CVD score {_tape_score:+.4f} ({_tape_label}) "
                    f"— LEAP requires bullish tape. Aborting LEAP {ticker} ${strike}C"
                )
                _at_log("SKIP", f"Tape gate: {_abort_msg}")
                raise _FlowAbort(_abort_msg)
            else:
                log.info("Tape gate [%s %s]: score=%+.4f (%s) — PASS",
                         ticker, t.upper(), _tape_score, _tape_label)

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
        raise _FlowAbort(_flow_abort)

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

    _tape_sent  = state["tape_sentiment"].get(ticker, {})
    _tape_fresh = _tape_is_fresh(_tape_sent)
    _tape_line  = (
        f"Tape CVD score at entry: {_tape_sent['score']:+.2f} ({_tape_sent.get('label','NO DATA')}) "
        f"[cvd={_tape_sent.get('components',{}).get('cvd',0):+.2f}, "
        f"vwap_z={_tape_sent.get('components',{}).get('vwap_z',0):+.2f}, "
        f"vol_mag={_tape_sent.get('components',{}).get('vol_mag',0):+.2f}]."
    ) if _tape_fresh else "Tape sentiment: no fresh data at entry time."

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
            + earn_note + f"\n\n**{_tape_line}**"
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
            f"(IV-adjusted: tighter in high-IV to limit crush risk). {earn_note}\n\n**{_tape_line}**"
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
    at = state["autotrader"]
    existing_hedge_key = next(
        (k for k, p in at["positions"].items() if p.get("strategy_type") == "hedge"),
        None,
    )

    if abs(net_delta) < threshold:
        # Delta back in range — close the hedge if one is open
        if existing_hedge_key:
            _at_log("HEDGE",
                    f"Delta normalised ({net_delta:+.1f} inside ±{threshold:.0f}) — "
                    f"closing hedge {existing_hedge_key}")
            hedge_info = at["positions"][existing_hedge_key]
            for item in ib.portfolio():
                if _at_contract_key(item.contract) == existing_hedge_key:
                    hedge_info["exit_reason"] = "delta_normalized"
                    await _autotrader_close_coro(ib, item, hedge_info, existing_hedge_key)
                    break
        return

    # Delta still above threshold — skip if a hedge is already open
    if existing_hedge_key:
        _at_log("HEDGE",
                f"Hedge already open ({existing_hedge_key}, delta={net_delta:+.1f}) — skipping new order")
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
            # Pay 1% above ask to ensure fill (hedge is protective, not premium-seeking)
            lmt = round(ask_live * 1.01, 2) if ask_live > 0 else None
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

    # Track in at["positions"] so the monitor handles profit-close and expiry cleanup.
    # strategy_type="hedge" lets the monitor treat it like a LEAP BUY (profit target + 21 DTE).
    from zoneinfo import ZoneInfo
    hedge_key = _at_contract_key(contract)
    state["autotrader"]["positions"][hedge_key] = {
        "ticker":        "SPY",
        "expiry":        expiry_k,
        "strike":        float(strike),
        "right":         hedge_right,
        "action":        "BUY",
        "qty":           1,
        "entry_price":   lmt,
        "max_profit":    round(lmt * 100, 2),   # cost basis (monitor treats as LEAP BUY)
        "order_id":      trade.order.orderId,
        "placed_at":     datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M ET"),
        "score":         0,
        "journal_id":    None,
        "live_iv":       None,
        "roll_count":    0,
        "strategy_type": "hedge",
    }
    _at_save_state()


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


def _tape_db_init() -> None:
    con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS tape_prints (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ts             TEXT    NOT NULL,
            session_date   TEXT    NOT NULL,
            ticker         TEXT    NOT NULL,
            price          REAL    NOT NULL,
            size           INTEGER NOT NULL,
            direction      INTEGER,
            exchange       TEXT,
            is_block       INTEGER,
            is_after_hours INTEGER,
            vwap           REAL,
            cum_vol        INTEGER,
            buy_vol        INTEGER,
            sell_vol       INTEGER,
            net_delta      INTEGER,
            pct_from_open  REAL,
            cvd_score      REAL
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_tp_ticker_date ON tape_prints (ticker, session_date)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS tape_bars (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            bar_start    TEXT    NOT NULL,
            session_date TEXT    NOT NULL,
            ticker       TEXT    NOT NULL,
            open         REAL,
            close        REAL,
            delta        INTEGER,
            vol          INTEGER,
            buy_vol      INTEGER,
            sell_vol     INTEGER,
            cvd_score    REAL,
            vwap_z       REAL,
            label        TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_tb_ticker_date ON tape_bars (ticker, session_date)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS alert_history (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            fired_at            TEXT NOT NULL,
            session_date        TEXT NOT NULL,
            ticker              TEXT NOT NULL,
            signal_type         TEXT NOT NULL,
            price               REAL,
            pct_b               REAL,
            rsi                 REAL,
            vol_ratio           REAL,
            tape_score          REAL,
            tape_label          TEXT,
            prev_state          TEXT,
            mins_in_pre_breakout INTEGER,
            state_path          TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_ah_ticker ON alert_history (ticker, session_date)")
    con.execute("""
        CREATE TABLE IF NOT EXISTS alert_performance (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_id             INTEGER UNIQUE,
            session_date         TEXT NOT NULL,
            ticker               TEXT NOT NULL,
            signal_type          TEXT NOT NULL,
            alert_time_et        TEXT,
            alert_price          REAL,
            eod_price            REAL,
            eod_return_pct       REAL,
            is_win               INTEGER,
            pct_b                REAL,
            rsi                  REAL,
            vol_ratio            REAL,
            tape_score           REAL,
            tape_label           TEXT,
            enriched_at          TEXT,
            prev_state           TEXT,
            mins_in_pre_breakout INTEGER,
            state_path           TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_ap_date ON alert_performance (session_date)")
    # State transitions table — one row per %B state change, all 112 tickers, all day
    con.execute("""
        CREATE TABLE IF NOT EXISTS state_transitions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date        TEXT NOT NULL,
            ticker              TEXT NOT NULL,
            prev_state          TEXT NOT NULL,
            new_state           TEXT NOT NULL,
            pct_b               REAL,
            rsi                 REAL,
            transition_time_et  TEXT NOT NULL,
            mins_in_prev_state  INTEGER
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_st_date ON state_transitions (session_date, ticker)")
    # Migrate state_transitions — add close_price for intraday return analysis
    try:
        con.execute("ALTER TABLE state_transitions ADD COLUMN close_price REAL")
    except Exception:
        pass
    # Migrate existing alert_history rows — add new columns if absent (safe for existing DBs)
    for col, typedef in [("prev_state", "TEXT"), ("mins_in_pre_breakout", "INTEGER"), ("state_path", "TEXT")]:
        try:
            con.execute(f"ALTER TABLE alert_history ADD COLUMN {col} {typedef}")
        except Exception:
            pass   # column already exists
    for col, typedef in [("prev_state", "TEXT"), ("mins_in_pre_breakout", "INTEGER"), ("state_path", "TEXT")]:
        try:
            con.execute(f"ALTER TABLE alert_performance ADD COLUMN {col} {typedef}")
        except Exception:
            pass
    con.commit()
    con.close()
    log.info("Tape DB initialised at %s", TAPE_DB_PATH)


def _alert_history_insert(ticker: str, signal_type: str, price: Optional[float],
                           pct_b: Optional[float], rsi: Optional[float],
                           vol_ratio: Optional[float], tape_score: Optional[float],
                           tape_label: Optional[str],
                           prev_state: Optional[str] = None,
                           mins_in_pre_breakout: Optional[int] = None,
                           state_path: Optional[str] = None) -> None:
    """Persist one scanner alert to alert_history (fire-and-forget, never raises)."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
        con.execute(
            """INSERT INTO alert_history
               (fired_at, session_date, ticker, signal_type, price,
                pct_b, rsi, vol_ratio, tape_score, tape_label,
                prev_state, mins_in_pre_breakout, state_path)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.utcnow().isoformat(), now_et.strftime("%Y-%m-%d"),
             ticker, signal_type, price, pct_b, rsi, vol_ratio, tape_score, tape_label,
             prev_state, mins_in_pre_breakout, state_path),
        )
        con.commit()
        con.close()
    except Exception as exc:
        log.debug("alert_history insert failed: %s", exc)


async def _enrich_day_performance(session_date: str) -> int:
    from zoneinfo import ZoneInfo
    from datetime import timezone as _tz
    con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    existing_ids = {r[0] for r in con.execute(
        "SELECT alert_id FROM alert_performance WHERE session_date=?", (session_date,)
    ).fetchall()}
    rows = con.execute(
        "SELECT * FROM alert_history WHERE session_date=?", (session_date,)
    ).fetchall()
    con.row_factory = None
    to_enrich = [dict(r) for r in rows if r["id"] not in existing_ids]
    if not to_enrich:
        con.close()
        return 0
    tickers = list({r["ticker"] for r in to_enrich})
    try:
        next_date = (datetime.strptime(session_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: yf.download(tickers, start=session_date, end=next_date,
                                 interval="1d", auto_adjust=True, progress=False)
        )
        closes: dict = {}
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                close_df = raw["Close"] if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
            else:
                close_df = raw[["Close"]] if "Close" in raw.columns else pd.DataFrame()
            if not close_df.empty:
                for t in tickers:
                    try:
                        val = close_df[t].iloc[0] if t in close_df.columns else close_df.iloc[0, 0]
                        if pd.notna(val):
                            closes[t] = float(val)
                    except Exception:
                        pass
    except Exception as exc:
        log.warning("_enrich_day_performance yfinance error: %s", exc)
        closes = {}
    et_zone = ZoneInfo("America/New_York")
    enriched_at = datetime.utcnow().isoformat()
    count = 0
    for r in to_enrich:
        ticker = r["ticker"]
        alert_price = r.get("price")
        eod_price = closes.get(ticker)
        eod_return_pct = None
        is_win = None
        if eod_price is not None and alert_price and alert_price > 0:
            eod_return_pct = (eod_price - alert_price) / alert_price * 100
            is_win = 1 if eod_return_pct > 0 else 0
        alert_time_et = None
        fired_at = r.get("fired_at")
        if fired_at:
            try:
                dt_utc = datetime.fromisoformat(fired_at.replace("Z", "+00:00"))
                if dt_utc.tzinfo is None:
                    dt_utc = dt_utc.replace(tzinfo=_tz.utc)
                alert_time_et = dt_utc.astimezone(et_zone).strftime("%H:%M")
            except Exception:
                pass
        con.execute(
            """INSERT OR IGNORE INTO alert_performance
               (alert_id, session_date, ticker, signal_type, alert_time_et, alert_price,
                eod_price, eod_return_pct, is_win, pct_b, rsi, vol_ratio, tape_score, tape_label,
                enriched_at, prev_state, mins_in_pre_breakout, state_path)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r["id"], r["session_date"], ticker, r["signal_type"], alert_time_et, alert_price,
             eod_price, eod_return_pct, is_win, r.get("pct_b"), r.get("rsi"),
             r.get("vol_ratio"), r.get("tape_score"), r.get("tape_label"), enriched_at,
             r.get("prev_state"), r.get("mins_in_pre_breakout"), r.get("state_path"))
        )
        count += 1
    con.commit()
    con.close()
    log.info("_enrich_day_performance %s: enriched %d rows", session_date, count)
    return count


def _compute_indicator_analysis(days: int = 30) -> dict:
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
    df = pd.read_sql_query(
        "SELECT * FROM alert_performance WHERE session_date >= ? AND eod_return_pct IS NOT NULL",
        con, params=(cutoff,)
    )
    con.close()
    if df.empty:
        return {
            "total_alerts": 0, "days_analyzed": 0,
            "overall": {"win_rate": 0.0, "avg_return": 0.0, "median_return": 0.0},
            "by_signal_type": [], "by_rsi": [], "by_pct_b": [],
            "by_vol_ratio": [], "by_tape": [], "by_hour": [], "by_ticker": [],
        }

    # SQLite None → numpy NaN so pd.cut / comparisons work correctly
    for col in ("rsi", "pct_b", "vol_ratio", "tape_score", "eod_return_pct", "is_win",
                "alert_price", "eod_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def _agg(grp):
        return pd.Series({
            "count": len(grp),
            "win_rate": round(grp["is_win"].mean() * 100, 1) if grp["is_win"].notna().any() else 0.0,
            "avg_return": round(grp["eod_return_pct"].mean(), 2),
        })

    _apply_kw = {"include_groups": False} if pd.__version__ >= "2.2" else {}

    days_analyzed = df["session_date"].nunique()
    overall_wr = round(df["is_win"].mean() * 100, 1) if df["is_win"].notna().any() else 0.0

    by_signal = (
        df.groupby("signal_type").apply(_agg, **_apply_kw).reset_index()
          .rename(columns={"signal_type": "label"})
          .to_dict(orient="records")
    )

    rsi_bins = [0, 65, 70, 75, 80, float("inf")]
    rsi_labels = ["<65", "65-70", "70-75", "75-80", ">80"]
    df["_rsi_bin"] = pd.cut(df["rsi"], bins=rsi_bins, labels=rsi_labels, right=False)
    by_rsi = (
        df.dropna(subset=["rsi"]).groupby("_rsi_bin", observed=True)
          .apply(_agg, **_apply_kw).reset_index()
          .rename(columns={"_rsi_bin": "bin"})
          .to_dict(orient="records")
    )

    pctb_bins = [0, 75, 85, 95, 105, float("inf")]
    pctb_labels = ["<75", "75-85", "85-95", "95-105", ">105"]
    df["_pctb_bin"] = pd.cut(df["pct_b"], bins=pctb_bins, labels=pctb_labels, right=False)
    by_pct_b = (
        df.dropna(subset=["pct_b"]).groupby("_pctb_bin", observed=True)
          .apply(_agg, **_apply_kw).reset_index()
          .rename(columns={"_pctb_bin": "bin"})
          .to_dict(orient="records")
    )

    vr_bins = [0, 1.0, 1.5, 2.0, 3.0, float("inf")]
    vr_labels = ["<1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", ">3.0"]
    df["_vr_bin"] = pd.cut(df["vol_ratio"], bins=vr_bins, labels=vr_labels, right=False)
    by_vol_ratio = (
        df.dropna(subset=["vol_ratio"]).groupby("_vr_bin", observed=True)
          .apply(_agg, **_apply_kw).reset_index()
          .rename(columns={"_vr_bin": "bin"})
          .to_dict(orient="records")
    )

    by_tape = []
    if "tape_label" in df.columns:
        by_tape = (
            df.dropna(subset=["tape_label"]).groupby("tape_label")
              .apply(_agg, **_apply_kw).reset_index()
              .rename(columns={"tape_label": "label"})
              .to_dict(orient="records")
        )

    by_hour = []
    if "alert_time_et" in df.columns:
        df["_hour"] = df["alert_time_et"].str[:2].apply(
            lambda h: f"{h}:xx" if pd.notna(h) and h != "" else None
        )
        by_hour = (
            df.dropna(subset=["_hour"]).groupby("_hour")
              .apply(_agg, **_apply_kw).reset_index()
              .rename(columns={"_hour": "hour"})
              .sort_values("hour")
              .to_dict(orient="records")
        )

    by_ticker = (
        df.groupby("ticker").apply(_agg, **_apply_kw).reset_index()
          .sort_values("count", ascending=False)
          .head(20)
          .to_dict(orient="records")
    )

    return {
        "total_alerts": len(df),
        "days_analyzed": days_analyzed,
        "overall": {
            "win_rate": overall_wr,
            "avg_return": round(df["eod_return_pct"].mean(), 2),
            "median_return": round(df["eod_return_pct"].median(), 2),
        },
        "by_signal_type": by_signal,
        "by_rsi": by_rsi,
        "by_pct_b": by_pct_b,
        "by_vol_ratio": by_vol_ratio,
        "by_tape": by_tape,
        "by_hour": by_hour,
        "by_ticker": by_ticker,
    }


def _load_telegram_creds() -> tuple[str, str]:
    """Read telegram_token/telegram_chat_id from scanner_config.json (shared with breakout_scanner.py)."""
    try:
        with open("scanner_config.json", "r") as f:
            cfg = json.load(f)
        return cfg.get("telegram_token", ""), cfg.get("telegram_chat_id", "")
    except Exception:
        return "", ""


def _send_telegram_sync(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        log.warning("Telegram not configured — EOD digest suppressed")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if not r.ok:
            log.warning("Telegram send failed [HTTP %d] → %s", r.status_code, r.text)
            return False
        return True
    except Exception as exc:
        log.warning("Telegram send failed: %s", exc)
        return False


def _format_eod_performance_digest(session_date: str) -> str | None:
    con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM alert_performance WHERE session_date=? AND eod_return_pct IS NOT NULL "
        "ORDER BY eod_return_pct DESC",
        (session_date,)
    ).fetchall()]
    con.close()
    if not rows:
        return None

    wins      = sum(1 for r in rows if r["is_win"] == 1)
    total     = len(rows)
    win_rate  = round(wins / total * 100, 1)
    avg_ret   = round(sum(r["eod_return_pct"] for r in rows) / total, 2)
    day_label = datetime.strptime(session_date, "%Y-%m-%d").strftime("%a %b %d")

    top_winners = rows[:2]
    top_losers  = rows[-2:][::-1]   # worst first

    lines = [
        f"📊 <b>EOD Alert Performance</b> — {day_label}\n",
        f"Total alerts: {total}",
        f"Win rate: {win_rate}% ({wins}W / {total - wins}L)",
        f"Avg return: {avg_ret:+.2f}%\n",
        "🏆 <b>Top winners:</b>",
    ]
    for r in top_winners:
        lines.append(f"  {r['ticker']:6s} {r['signal_type']:14s} {r['eod_return_pct']:+.2f}%")
    lines.append("\n📉 <b>Top losers:</b>")
    for r in top_losers:
        lines.append(f"  {r['ticker']:6s} {r['signal_type']:14s} {r['eod_return_pct']:+.2f}%")

    return "\n".join(lines)


async def _send_eod_performance_digest(session_date: str) -> bool:
    msg = await asyncio.get_event_loop().run_in_executor(
        None, _format_eod_performance_digest, session_date
    )
    if not msg:
        log.info("EOD performance digest: no enriched alerts for %s — skipped", session_date)
        return False
    token, chat_id = _load_telegram_creds()
    sent = await asyncio.get_event_loop().run_in_executor(
        None, _send_telegram_sync, token, chat_id, msg
    )
    if sent:
        log.info("EOD performance digest sent to Telegram for %s", session_date)
    return sent


async def _enrich_then_notify(session_date: str) -> None:
    """EOD heartbeat hook: enrich today's alerts with closing prices, then push the Telegram digest."""
    await _enrich_day_performance(session_date)
    await _send_eod_performance_digest(session_date)


def _tape_db_flush_prints(ticker: str, rows: list) -> None:
    """Batch-insert a list of tape print dicts into tape_prints. Safe to call from any thread."""
    if not rows:
        return
    try:
        con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
        con.executemany("""
            INSERT INTO tape_prints
            (ts, session_date, ticker, price, size, direction, exchange, is_block,
             is_after_hours, vwap, cum_vol, buy_vol, sell_vol, net_delta, pct_from_open, cvd_score)
            VALUES (:ts,:session_date,:ticker,:price,:size,:direction,:exchange,:is_block,
                    :is_after_hours,:vwap,:cum_vol,:buy_vol,:sell_vol,:net_delta,:pct_from_open,:cvd_score)
        """, rows)
        con.commit()
        con.close()
    except Exception as exc:
        log.debug("tape_db flush prints error: %s", exc)


def _tape_db_insert_block_immediate(sym: str, price: float, size: int, direction: int,
                                     session_vwap: float, cvd_score: float) -> None:
    """Write a single block print immediately from the CVD callback (always-on capture)."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        now_et = datetime.now()
    ts_str      = now_et.strftime("%Y-%m-%dT%H:%M:%S")
    sess_date   = now_et.strftime("%Y-%m-%d")
    sent        = state["tape_sentiment"].get(sym, {})
    buy_vol_now = sent.get("session_buy_vol", 0)
    sell_vol_now= sent.get("session_sell_vol", 0)
    try:
        con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
        con.execute("""
            INSERT INTO tape_prints
            (ts, session_date, ticker, price, size, direction, exchange, is_block,
             is_after_hours, vwap, cum_vol, buy_vol, sell_vol, net_delta, pct_from_open, cvd_score)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (ts_str, sess_date, sym, price, size, direction, "SMART", 1,
              0, session_vwap, sent.get("session_vol", 0),
              buy_vol_now, sell_vol_now, buy_vol_now - sell_vol_now,
              None, round(cvd_score, 4)))
        con.commit()
        con.close()
    except Exception as exc:
        log.debug("tape_db block immediate write error [%s]: %s", sym, exc)


def _tape_db_insert_bar(ticker: str, bar: dict, bar_start: str, session_date: str,
                         buy_vol: int, sell_vol: int, cvd_score: float,
                         vwap_z: float, label: str) -> None:
    """Insert a completed 1-minute CVD bar into tape_bars. Safe to call from any thread."""
    try:
        con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
        con.execute("""
            INSERT INTO tape_bars
            (bar_start, session_date, ticker, open, close, delta, vol,
             buy_vol, sell_vol, cvd_score, vwap_z, label)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (bar_start, session_date, ticker,
              bar.get("open"), bar.get("close"), bar.get("delta"), bar.get("vol"),
              buy_vol, sell_vol, cvd_score, vwap_z, label))
        con.commit()
        con.close()
    except Exception as exc:
        log.debug("tape_db insert bar error: %s", exc)


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
# roll_close is always win=0 (buying back the put at a loss is WHY you rolled —
# it's not a final trade outcome). Exclude it from Kelly calibration so the win
# rate reflects terminal exits only.
_KELLY_EXIT_REASONS = (
    "'profit_target','stop_loss',"
    "'roll_max','roll_no_credit','21dte','manual','rotation'"
)


def _update_kelly_from_journal() -> None:
    """EMA-blend actual win rate from final trade exits into assumed_win_rate.

    Uses _KELLY_EXIT_REASONS which excludes 'roll_close'.  roll_close is always
    win=0 (that's why you rolled — you bought back at a loss) and is NOT a final
    trade outcome.  Including it systematically underestimates the win rate and
    produces negative Kelly fractions that block new entries.

    Requires at least 5 final-outcome trades.  Blended rate is floored at 0.50
    so Kelly never turns negative from statistical noise in a small sample.
    """
    con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
    rows = con.execute(
        f"SELECT win FROM trade_journal "
        f"WHERE closed_at IS NOT NULL AND win IS NOT NULL "
        f"  AND exit_reason IN ({_KELLY_EXIT_REASONS})"
    ).fetchall()
    con.close()
    if len(rows) < 5:
        return
    actual_wr = sum(r[0] for r in rows) / len(rows)
    old       = state["autotrader"]["config"].get("assumed_win_rate", 0.80)
    blended   = round(0.70 * old + 0.30 * actual_wr, 4)
    new_wr    = max(0.50, blended)   # floor: Kelly must stay positive
    state["autotrader"]["config"]["assumed_win_rate"] = new_wr
    _at_log("LEARN",
            f"Kelly win rate {old:.1%} → {new_wr:.1%} "
            f"(actual {actual_wr:.1%} over {len(rows)} final exits, floor=50%)")
    _at_save_state()   # persist so the updated rate survives a restart


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
                        # Target mid-range delta: (LEAP_MIN_DELTA + LEAP_MAX_DELTA) / 2 = 0.75.
                        # Old target was 0.60, which is *below* LEAP_MIN_DELTA (0.65) — meaning
                        # the formula always rewarded the lowest allowed delta.  Deep-ITM LEAP
                        # strategy prefers higher delta (more intrinsic, less extrinsic decay).
                        _delta_target = (LEAP_MIN_DELTA + LEAP_MAX_DELTA) / 2  # 0.75
                        row["score"] = round(
                            sq * 55
                            + (1 - abs(delta - _delta_target)) * 35
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
                                rw["score"]=round(sq*55+(1-abs(dlt2-(LEAP_MIN_DELTA+LEAP_MAX_DELTA)/2))*35+ib2+(liq2/100)*10,2)
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


# ── Stock Trader helpers ────────────────────────────────────────────────────

def _st_save_state() -> None:
    st = state["stock_trader"]
    try:
        with open(ST_STATE_PATH, "w") as f:
            json.dump({
                "enabled":      st["enabled"],
                "config":       st["config"],
                "positions":    st["positions"],
                "closed_today": st.get("closed_today", [])[-100:],
                "decisions":    st.get("decisions", [])[-200:],
                "rotation_log": st.get("rotation_log", [])[-50:],
            }, f, indent=2, default=str)
    except Exception as e:
        log.warning("Stock trader state save failed: %s", e)


def _st_load_state() -> None:
    if not os.path.exists(ST_STATE_PATH):
        return
    try:
        with open(ST_STATE_PATH, "r") as f:
            saved = json.load(f)
        st = state["stock_trader"]
        st["enabled"]       = saved.get("enabled", False)
        st["config"].update(saved.get("config", {}))
        st["positions"]     = saved.get("positions", {})
        st["closed_today"]  = saved.get("closed_today", [])
        st["decisions"]     = saved.get("decisions", [])
        st["rotation_log"]  = saved.get("rotation_log", [])
        log.info("Stock trader state restored: enabled=%s, positions=%d",
                 st["enabled"], len(st["positions"]))
    except Exception as e:
        log.warning("Stock trader state load failed (starting fresh): %s", e)


def _st_log(action: str, ticker: str, detail: str) -> None:
    from zoneinfo import ZoneInfo
    et_now = datetime.now(ZoneInfo("America/New_York"))
    entry  = {
        "time":   et_now.strftime("%H:%M:%S ET"),
        "action": action,
        "ticker": ticker,
        "detail": detail,
    }
    st = state["stock_trader"]
    st["decisions"].append(entry)
    st["decisions"] = st["decisions"][-200:]
    log.info("[StockTrader] %s %s: %s", action, ticker, detail)


def _st_trading_days_held(entry_date_str: str) -> int:
    """Count trading days from entry date up to (not including) today."""
    try:
        entry = datetime.fromisoformat(entry_date_str).date()
        today = date.today()
        if entry >= today:
            return 0
        return int(np.busday_count(entry.isoformat(), today.isoformat()))
    except Exception:
        return 0


def _close_st_position(ticker: str, pos: dict, exit_px: float,
                        exit_type: str, pnl: float, days_held: int) -> None:
    """Record a closed stock trade to closed_today + trade_journal."""
    st       = state["stock_trader"]
    entry_px = pos.get("entry_price", exit_px)
    pnl_pct  = round((exit_px - entry_px) / entry_px * 100, 3) if entry_px else 0.0

    record = {
        "ticker":     ticker,
        "entry_date": pos.get("entry_date"),
        "exit_date":  date.today().isoformat(),
        "entry_price": round(entry_px, 4),
        "exit_price":  round(exit_px, 4),
        "shares":      pos.get("shares", 0),
        "pnl":         round(pnl, 2),
        "pnl_pct":     pnl_pct,
        "exit_type":   exit_type,
        "days_held":   days_held,
        "win":         pnl > 0,
    }
    st["closed_today"].append(record)
    st["closed_today"] = st["closed_today"][-100:]

    # Persist to trade_journal for history / stats
    try:
        con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
        con.execute("""
            INSERT INTO trade_journal
                (opened_at, closed_at, ticker, action, qty,
                 entry_price, exit_price, pnl, pnl_pct, win,
                 exit_reason, strategy_type)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pos.get("entry_date"),
            date.today().isoformat(),
            ticker, "BUY", pos.get("shares", 0),
            round(entry_px, 4), round(exit_px, 4),
            round(pnl, 2), pnl_pct, 1 if pnl > 0 else 0,
            exit_type, "STOCK_BREAKOUT",
        ))
        con.commit()
        con.close()
    except Exception as exc:
        log.warning("Stock trade journal insert failed: %s", exc)

    _st_log(exit_type.upper(), ticker,
            f"exit={exit_px:.2f} pnl={'+'if pnl>=0 else''}{pnl:.2f} "
            f"({pnl_pct:+.2f}%) day {days_held}")
    _st_save_state()


def _st_avg_score(st: dict) -> float:
    """Average composite_score across all active (phase 1|2) positions. Returns 50 if unknown."""
    scores = [
        pos["composite_score"]
        for pos in st["positions"].values()
        if pos.get("phase", 0) in (1, 2) and pos.get("composite_score") is not None
    ]
    return round(sum(scores) / len(scores), 1) if scores else 50.0


def _st_find_rotation_candidate(new_ticker: str, new_score: float, st: dict):
    """
    Find the weakest incumbent position eligible for eviction.

    Rules (all must pass):
    - Phase 1 or 2 (active, not pending or closing)
    - trading_days_held >= 3
    - live_pnl < 0 (only evict losers, never winners)
    - new_score >= 70 AND > portfolio average score
    - Sector of candidate != sector of new_ticker (sector guard)

    Returns (ticker, reason_str) or (None, skip_reason).
    """
    MIN_SCORE_FLOOR   = 70.0
    MIN_DAYS_HELD     = 3

    # Score gate: new signal must be strong enough
    avg_score = _st_avg_score(st)
    if new_score < MIN_SCORE_FLOOR:
        return None, f"score {new_score:.0f} below floor {MIN_SCORE_FLOOR:.0f}"
    if new_score <= avg_score:
        return None, f"score {new_score:.0f} not above portfolio avg {avg_score:.0f}"

    new_sector = STOCK_SECTOR_MAP.get(new_ticker, "Unknown")

    best_ticker   = None
    best_pnl_pct  = 0.0   # only candidates with pnl_pct < 0 qualify
    best_detail   = ""
    sector_blocked = 0

    for ticker, pos in st["positions"].items():
        if ticker == new_ticker:
            continue
        if pos.get("phase", 0) not in (1, 2):
            continue
        days_held = pos.get("trading_days_held", 0) or _st_trading_days_held(pos.get("entry_date", ""))
        if days_held < MIN_DAYS_HELD:
            continue
        live_pnl = pos.get("live_pnl", 0) or 0
        if live_pnl >= 0:
            continue   # never evict a profitable position

        # Sector guard
        cand_sector = STOCK_SECTOR_MAP.get(ticker, "Unknown")
        if cand_sector != "Unknown" and new_sector != "Unknown" and cand_sector == new_sector:
            sector_blocked += 1
            continue

        entry_price = pos.get("entry_price") or 0
        shares      = pos.get("shares") or 0
        cost        = entry_price * shares
        pnl_pct     = (live_pnl / cost * 100) if cost > 0 else 0.0

        if pnl_pct < best_pnl_pct:
            best_pnl_pct  = pnl_pct
            best_ticker   = ticker
            best_detail   = (
                f"held {days_held}d, P&L ${live_pnl:+.2f} ({pnl_pct:.1f}%), "
                f"sector={cand_sector}, deepest loser among eligible candidates"
            )

    if best_ticker is None:
        if sector_blocked > 0 and best_pnl_pct == 0.0:
            return None, f"sector guard blocked all {sector_blocked} eligible loser(s) — {new_sector} overlap"
        return None, "no losing positions held >= 3d in a different sector"

    return best_ticker, best_detail


async def _stock_monitor_coro(ib) -> None:
    """One monitor cycle: detect fills, phase transitions, exits, force closes."""
    from zoneinfo import ZoneInfo
    st  = state["stock_trader"]
    cfg = st["config"]

    if not st["positions"]:
        return

    open_trades_by_oid = {t.order.orderId: t for t in ib.openTrades()}
    fills_by_oid: dict = {}
    for f in ib.fills():
        oid = getattr(f.execution, "orderId", 0)
        if oid and oid > 0:
            fills_by_oid[oid] = f

    to_remove: list[str] = []

    for ticker, pos in list(st["positions"].items()):
        days_held = _st_trading_days_held(pos.get("entry_date", ""))
        pos["trading_days_held"] = days_held
        phase = pos.get("phase", 1)

        # ── Phase 0: waiting for buy fill ─────────────────────────────────
        if phase == 0:
            buy_oid = pos.get("buy_order_id")
            if not buy_oid:
                continue
            if buy_oid in open_trades_by_oid:
                # Cancel stale limit buys that have been pending too long
                try:
                    alert_ts = pos.get("alert_fired_at", "")
                    at = datetime.fromisoformat(alert_ts) if alert_ts else None
                    if at is not None:
                        if at.tzinfo is None:
                            at = at.replace(tzinfo=ZoneInfo("America/New_York"))
                        age_min = (datetime.now(ZoneInfo("America/New_York")) - at).total_seconds() / 60
                    else:
                        age_min = 0
                except Exception:
                    age_min = 0
                if age_min > cfg.get("signal_freshness_min", 30):
                    ib.cancelOrder(open_trades_by_oid[buy_oid].order)
                    await asyncio.sleep(0.5)
                    _st_log("BUY_CANCELLED", ticker,
                            f"limit buy stale after {age_min:.0f}min — cancelled, slot freed")
                    to_remove.append(ticker)
                continue  # still pending (and not yet stale)
            fill = fills_by_oid.get(buy_oid)
            if fill:
                fill_px = round(float(fill.execution.avgPrice), 4)
                pos["entry_price"] = fill_px
                pos["phase"] = 1
                stop_px = round(fill_px * (1 - cfg["hard_stop_pct"] / 100), 2)
                contract  = Stock(ticker, "SMART", "USD")
                stop_ord  = Order()
                stop_ord.orderType      = "STP"
                stop_ord.action         = "SELL"
                stop_ord.totalQuantity  = pos["shares"]
                stop_ord.auxPrice       = stop_px
                stop_ord.tif            = "GTC"
                stop_ord.outsideRth     = False
                trade = ib.placeOrder(contract, stop_ord)
                await asyncio.sleep(0.5)
                pos["stop_order_id"] = trade.order.orderId
                pos["stop_type"]     = "STOP"
                pos["stop_price"]    = stop_px
                _st_log("FILLED", ticker,
                        f"fill={fill_px:.2f} x{pos['shares']}sh stop@{stop_px:.2f} "
                        f"(stp ord#{trade.order.orderId})")
                _st_save_state()
            else:
                _st_log("BUY_LAPSED", ticker, "buy limit not filled / cancelled — removing")
                to_remove.append(ticker)
            continue

        stop_oid = pos.get("stop_order_id")

        # ── Phase 1: hard stop active (days 1-5) ──────────────────────────
        if phase == 1:
            if stop_oid and stop_oid not in open_trades_by_oid:
                fill    = fills_by_oid.get(stop_oid)
                exit_px = round(float(fill.execution.avgPrice), 4) if fill else pos.get("stop_price", pos["entry_price"])
                pnl     = round((exit_px - pos["entry_price"]) * pos["shares"], 2)
                _close_st_position(ticker, pos, exit_px, "hard_stop", pnl, days_held)
                to_remove.append(ticker)
                continue

            # Day 5: phase 1 → phase 2 transition
            if days_held >= 5:
                contract = Stock(ticker, "SMART", "USD")
                if stop_oid and stop_oid in open_trades_by_oid:
                    ib.cancelOrder(open_trades_by_oid[stop_oid].order)
                    await asyncio.sleep(1)
                trail_ord = Order()
                trail_ord.orderType     = "TRAIL"
                trail_ord.action        = "SELL"
                trail_ord.totalQuantity = pos["shares"]
                trail_ord.trailingPercent = cfg["trail_pct"]
                trail_ord.tif           = "GTC"
                trail_ord.outsideRth    = False
                trade = ib.placeOrder(contract, trail_ord)
                await asyncio.sleep(0.5)
                pos["stop_order_id"] = trade.order.orderId
                pos["stop_type"]     = "TRAIL"
                pos.pop("stop_price", None)
                pos["phase"] = 2
                _st_log("PHASE2", ticker,
                        f"day {days_held}: hard stop cancelled, "
                        f"TRAIL {cfg['trail_pct']}% placed (ord#{trade.order.orderId})")
                _st_save_state()

        # ── Phase 2: trailing stop active (days 5-30) ─────────────────────
        elif phase == 2:
            if stop_oid and stop_oid not in open_trades_by_oid:
                fill    = fills_by_oid.get(stop_oid)
                exit_px = round(float(fill.execution.avgPrice), 4) if fill else pos["entry_price"]
                pnl     = round((exit_px - pos["entry_price"]) * pos["shares"], 2)
                _close_st_position(ticker, pos, exit_px, "trail_stop", pnl, days_held)
                to_remove.append(ticker)
                continue

            # Day 30: force close
            if days_held >= cfg["max_hold_days"]:
                if stop_oid and stop_oid in open_trades_by_oid:
                    ib.cancelOrder(open_trades_by_oid[stop_oid].order)
                    await asyncio.sleep(1)
                contract = Stock(ticker, "SMART", "USD")
                mkt_ord = Order()
                mkt_ord.orderType      = "MKT"
                mkt_ord.action         = "SELL"
                mkt_ord.totalQuantity  = pos["shares"]
                mkt_ord.tif            = "DAY"
                trade = ib.placeOrder(contract, mkt_ord)
                pos["phase"]          = 3
                pos["stop_order_id"]  = trade.order.orderId
                _st_log("FORCE_CLOSE", ticker,
                        f"day {days_held}: max hold reached, MKT sell placed (ord#{trade.order.orderId})")
                _st_save_state()

        # ── Phase 3: market force-close in flight ─────────────────────────
        elif phase == 3:
            sell_oid = pos.get("stop_order_id")
            if sell_oid and sell_oid not in open_trades_by_oid:
                fill    = fills_by_oid.get(sell_oid)
                exit_px = round(float(fill.execution.avgPrice), 4) if fill else pos["entry_price"]
                pnl     = round((exit_px - pos["entry_price"]) * pos["shares"], 2)
                _close_st_position(ticker, pos, exit_px, "max_hold", pnl, days_held)
                to_remove.append(ticker)

    # ── Live price / P&L from IBKR portfolio ──────────────────────────────
    try:
        portfolio_map = {
            item.contract.symbol: item
            for item in ib.portfolio()
            if hasattr(item.contract, "symbol")
        }
        for ticker, pos in st["positions"].items():
            pi = portfolio_map.get(ticker)
            if pi is not None:
                try:
                    mpx = float(pi.marketPrice or 0)
                    if mpx > 0:
                        pos["live_price"] = round(mpx, 4)
                        pos["live_pnl"] = round(float(pi.unrealizedPNL or 0), 2)
                except Exception:
                    pass
    except Exception as _lpe:
        log.debug("Stock trader live price fetch failed: %s", _lpe)

    for ticker in to_remove:
        st["positions"].pop(ticker, None)
    if to_remove:
        _st_save_state()

    # ── Rotation log outcome enrichment ───────────────────────────────────
    rot_log = st.get("rotation_log", [])
    enriched = False
    for rot in rot_log:
        if rot.get("outcome_5d") is not None:
            continue
        try:
            ts = datetime.fromisoformat(rot["ts"])
            days_since = int(np.busday_count(ts.date().isoformat(), date.today().isoformat()))
        except Exception:
            continue
        if days_since < 5:
            continue
        incoming = rot.get("incoming")
        if not incoming:
            continue
        # Look up current price from IBKR portfolio or live_price in open positions
        incoming_pos = st["positions"].get(incoming)
        cur_price = None
        if incoming_pos and incoming_pos.get("live_price"):
            cur_price = incoming_pos["live_price"]
        if cur_price is None:
            try:
                pi = next((i for i in ib.portfolio() if i.contract.symbol == incoming), None)
                if pi and pi.marketPrice:
                    cur_price = float(pi.marketPrice)
            except Exception:
                pass
        if cur_price and rot.get("evicted_entry_price"):
            entry_px = next(
                (p["entry_price"] for t, p in st["positions"].items() if t == incoming),
                rot.get("evicted_entry_price")
            )
            incoming_entry = next(
                (p["entry_price"] for t, p in st["positions"].items() if t == incoming),
                None
            )
            if incoming_entry:
                rot["outcome_5d"] = round((cur_price - incoming_entry) / incoming_entry * 100, 2)
                enriched = True
    if enriched:
        _st_save_state()


async def _stock_monitor_loop() -> None:
    """Background task: monitor stock trader positions every 60 seconds."""
    await asyncio.sleep(25)   # let server finish starting
    while True:
        await asyncio.sleep(60)
        st = state["stock_trader"]
        if not st["enabled"]:
            continue
        if not state.get("connected") or not state.get("ib"):
            continue
        ib = state["ib"]
        if not ib.isConnected():
            continue
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: _run_in_streaming_loop(_stock_monitor_coro(ib), timeout=50),
            )
        except Exception as exc:
            log.error("Stock monitor loop error: %s", exc, exc_info=True)


# ── FastAPI app ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        state["model"] = joblib.load(MODEL_PATH)
        log.info("Loaded cached model from disk")

    state["iv_history"] = _load_iv_history()
    log.info(f"Loaded IV history for {len(state['iv_history'])} tickers")
    _tape_db_init()
    _journal_init()

    # Restore auto-trader positions + config from last shutdown
    _at_load_state()

    # Restore stock trader state from last shutdown
    _st_load_state()

    # Restore day trader state from last shutdown
    _dt_load_state()

    # Restore SPX 0DTE state from last shutdown
    _spx_load_state()

    # Restore breakout watchlist
    _watchlist_load()

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
    asyncio.create_task(_stock_monitor_loop())
    log.info("Stock trader monitor loop started")
    asyncio.create_task(_day_trader_monitor_loop())
    log.info("Day trader monitor loop started")
    asyncio.create_task(_spx_monitor_loop())
    log.info("SPX 0DTE monitor loop started")

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
    asyncio.create_task(_watchlist_price_refresh_loop())
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
if _technicals_router_ok and technicals_router is not None:
    app.include_router(technicals_router)   # GET /technicals/{ticker}
else:
    import logging as _lg
    _lg.getLogger("main").warning("ibkr_technicals router failed to load — /technicals/{ticker} unavailable")


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
    auto_hedge:           bool    = False
    hedge_threshold:      float   = 100.0
    tape_filter_enabled:  bool    = True


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
        # IBKR returns account totals with currency="" (base) on paper accounts
        # and "BASE" or "USD" on live accounts. Accept all three; prefer non-zero
        # so a later USD-tagged entry doesn't overwrite a valid base-currency value.
        if key and av.currency in ("USD", "", "BASE"):
            try:
                v = round(float(av.value), 2)
                if result[key] == 0.0 or av.currency == "USD":
                    result[key] = v
            except (ValueError, TypeError):
                pass

    # Fallback: if IBKR account-level UnrealizedPnL is still 0 (tag absent or zeroed
    # at market close on paper accounts), compute from portfolio item.unrealizedPNL.
    # This is accurate as long as marketPrice is populated (which it is from closing marks).
    if result["unrealized_pnl"] == 0.0:
        try:
            portfolio_upnl = 0.0
            for item in ib.portfolio():
                v = item.unrealizedPNL
                if v is not None:
                    try:
                        f = float(v)
                        if not math.isnan(f):
                            portfolio_upnl += f
                    except (TypeError, ValueError):
                        pass
            if portfolio_upnl != 0.0:
                result["unrealized_pnl"] = round(portfolio_upnl, 2)
        except Exception:
            pass

    return result


@app.get("/account/raw-values")
def account_raw_values():
    """Debug: return all raw IBKR accountValues so we can see exact tag+currency pairs."""
    _require_connection()
    ib = state["ib"]
    accounts = ib.managedAccounts()
    account_id = accounts[0] if accounts else ""
    rows = []
    for av in ib.accountValues(account_id):
        rows.append({"tag": av.tag, "value": av.value, "currency": av.currency, "account": av.account})
    # Return only P&L and value-related tags to keep the response readable
    pnl_tags = [r for r in rows if any(k in r["tag"] for k in ("PnL","Pnl","pnl","Cash","Liq","Margin","Fund","Power","Position","Value"))]
    return {"account": account_id, "pnl_tags": pnl_tags, "total_tags": len(rows)}


# ── Watchlist endpoints (breakout scanner integration) ────────────────────

class WatchlistAlertRequest(BaseModel):
    ticker:               str
    signal_type:          str              # "BREAKOUT" or "PRE-BREAKOUT"
    price_at_alert:       float
    pct_b:                float
    rsi:                  Optional[float] = None
    vol_ratio:            Optional[float] = None
    timestamp_et:         Optional[str]   = None   # "HH:MM ET YYYY-MM-DD"
    tape_score:           Optional[float] = None   # CVD score from scanner's /tape/sentiment call
    tape_label:           Optional[str]   = None   # e.g. "BULLISH", "BEARISH", "NEUTRAL"
    # State lifecycle context — populated by breakout_scanner when _ticker_states has history
    prev_state:           Optional[str]   = None   # state before entering current signal zone
    mins_in_pre_breakout: Optional[int]   = None   # minutes spent in PRE-BREAKOUT before BREAKOUT
    state_path:           Optional[str]   = None   # e.g. "NEUTRAL→PRE-BREAKOUT→BREAKOUT"
    # S/R levels from scanner (swing-based, 60-bar lookback)
    sr_resistance:        Optional[float] = None   # nearest swing high above price
    sr_support:           Optional[float] = None   # nearest swing low below price


@app.post("/watchlist/alert")
def watchlist_add_alert(req: WatchlistAlertRequest):
    """Called by the breakout scanner on every cycle for every detected signal.

    New tickers are inserted with the current timestamp and price as the alert baseline.
    Existing tickers get live metrics (pct_b, vol_ratio) refreshed each cycle so the
    watchlist reflects the current scan state, while preserving the original alert
    timestamp and price_at_alert so % change is always vs. the first alert price.
    PRE-BREAKOUT is never allowed to overwrite an existing BREAKOUT entry.

    Backend gate (F1 — real-time VIX + SPY regime):
    New entries are blocked when VIX ≥ 25 (live IBKR feed) or SPY is below its
    SMA-200 (regime cache). Refreshes of existing entries always pass through so the
    watchlist stays current. Returns action='blocked' so the scanner can suppress Telegram.
    """
    from zoneinfo import ZoneInfo
    now_et  = datetime.now(ZoneInfo("America/New_York"))
    tk      = req.ticker.upper()
    existing = state["watchlist"].get(tk)

    # ── Backend gate: only for NEW entries (refreshes always pass through) ──
    if not existing:
        vix_live = state["vix_live"].get("price")
        if vix_live is not None and vix_live >= 25:
            return {"ok": False, "action": "blocked",
                    "reason": f"VIX={vix_live:.1f} ≥ 25 (live IBKR feed)"}
        regime = state["cache"].get("regime") or {}
        if regime.get("spy_above_sma200") is False:
            return {"ok": False, "action": "blocked",
                    "reason": (f"SPY ${regime.get('spy_price', 0):.2f} "
                               f"below SMA-200 ${regime.get('spy_sma200', 0):.0f}")}

    def _resolve_tape(req_score, req_label):
        """Pick best available tape: scanner-provided → backend live state → NO DATA."""
        if req_label and req_label not in ("NO DATA", "NEUTRAL"):
            return req_score, req_label
        _sent  = state["tape_sentiment"].get(tk, {})
        _fresh = _tape_is_fresh(_sent)
        if _fresh:
            return round(_sent["score"], 4), _sent.get("label", "NO DATA")
        # Fall back to what scanner sent even if neutral
        return req_score, req_label or "NO DATA"

    _today_date = now_et.strftime("%Y-%m-%d")

    def _log_if_new_day(entry_ts: str, sig: str, price: float, resolved_ts, resolved_tl):
        """Log to alert_history when a carry-over watchlist ticker fires on a new day."""
        if entry_ts[:10] != _today_date:
            _alert_history_insert(tk, sig, price,
                                  req.pct_b, req.rsi, req.vol_ratio,
                                  resolved_ts, resolved_tl,
                                  prev_state=req.prev_state,
                                  mins_in_pre_breakout=req.mins_in_pre_breakout,
                                  state_path=req.state_path)
            return True
        return False

    # Don't downgrade BREAKOUT → PRE-BREAKOUT
    if existing and existing["signal_type"] == "BREAKOUT" and req.signal_type == "PRE-BREAKOUT":
        # Still refresh live metrics so pct_b / vol_ratio / tape stay current
        existing["pct_b"]      = round(req.pct_b, 1)
        if req.rsi          is not None: existing["rsi"]          = round(req.rsi, 1)
        if req.vol_ratio    is not None: existing["vol_ratio"]    = round(req.vol_ratio, 2)
        if req.sr_resistance is not None: existing["sr_resistance"] = round(req.sr_resistance, 2)
        if req.sr_support    is not None: existing["sr_support"]    = round(req.sr_support, 2)
        ts, tl = _resolve_tape(req.tape_score, req.tape_label)
        existing["tape_score"]     = ts
        existing["tape_label"]     = tl
        existing["tape_confirmed"] = bool(ts is not None and ts > 0.20)
        if _log_if_new_day(existing.get("added_iso", ""), req.signal_type,
                           req.price_at_alert, ts, tl):
            existing["added_iso"]      = now_et.isoformat()
            existing["price_at_alert"] = round(req.price_at_alert, 2)
        _watchlist_save()
        return {"ok": True, "action": "refreshed"}

    if existing:
        # Preserve original alert timestamp and price; refresh live scan metrics
        existing["signal_type"] = req.signal_type
        existing["pct_b"]       = round(req.pct_b, 1)
        if req.rsi           is not None: existing["rsi"]          = round(req.rsi, 1)
        if req.vol_ratio     is not None: existing["vol_ratio"]    = round(req.vol_ratio, 2)
        if req.sr_resistance is not None: existing["sr_resistance"] = round(req.sr_resistance, 2)
        if req.sr_support    is not None: existing["sr_support"]    = round(req.sr_support, 2)
        ts, tl = _resolve_tape(req.tape_score, req.tape_label)
        existing["tape_score"]     = ts
        existing["tape_label"]     = tl
        existing["tape_confirmed"] = bool(ts is not None and ts > 0.20)
        if _log_if_new_day(existing.get("added_iso", ""), req.signal_type,
                           req.price_at_alert, ts, tl):
            existing["added_iso"]      = now_et.isoformat()
            existing["price_at_alert"] = round(req.price_at_alert, 2)
        _watchlist_save()
        return {"ok": True, "action": "refreshed"}

    # New entry
    ts, tl = _resolve_tape(req.tape_score, req.tape_label)
    state["watchlist"][tk] = {
        "ticker":         tk,
        "signal_type":    req.signal_type,
        "price_at_alert": round(req.price_at_alert, 2),
        "pct_b":          round(req.pct_b, 1),
        "rsi":            round(req.rsi, 1) if req.rsi is not None else None,
        "vol_ratio":      round(req.vol_ratio, 2) if req.vol_ratio is not None else None,
        "timestamp_et":   req.timestamp_et or now_et.strftime("%H:%M ET %Y-%m-%d"),
        "added_iso":      now_et.isoformat(),
        "tape_score":     ts,
        "tape_label":     tl,
        "tape_confirmed": bool(ts is not None and ts > 0.20),
        "sr_resistance":  round(req.sr_resistance, 2) if req.sr_resistance is not None else None,
        "sr_support":     round(req.sr_support, 2)    if req.sr_support    is not None else None,
    }
    # Persist every new alert to alert_history for backtesting (includes state path context)
    _alert_history_insert(tk, req.signal_type, req.price_at_alert,
                          req.pct_b, req.rsi, req.vol_ratio, ts, tl,
                          prev_state=req.prev_state,
                          mins_in_pre_breakout=req.mins_in_pre_breakout,
                          state_path=req.state_path)
    _watchlist_save()
    return {"ok": True, "action": "added"}


# Watchlist price cache — populated by _watchlist_price_refresh_loop() background task.
# The endpoint reads from this dict; no yfinance call ever happens inside a request handler.
_watchlist_price_cache: dict = {"prices": {}, "ts": 0.0}


async def _watchlist_price_refresh_loop() -> None:
    """Background task: refresh watchlist prices via yfinance every 30 s.

    Running yfinance outside the request handler means /watchlist always returns
    instantly from the in-memory cache (~1 ms) instead of blocking for 4-6 s.
    """
    import time as _time

    def _fetch(tickers: list) -> dict:
        if not tickers:
            return {}
        try:
            data = yf.download(tickers, period="1d", interval="5m",
                               group_by="ticker", progress=False,
                               auto_adjust=True, threads=True)
            prices = {}
            for tk in tickers:
                try:
                    col = (data["Close"] if len(tickers) == 1
                           else data[tk]["Close"] if tk in data else None)
                    prices[tk] = float(col.dropna().iloc[-1]) if col is not None and not col.dropna().empty else None
                except Exception:
                    prices[tk] = None
            return prices
        except Exception:
            return {}

    while True:
        try:
            await asyncio.sleep(30)
            tickers = [e["ticker"] for e in state["watchlist"].values()]
            if tickers:
                prices = await asyncio.get_event_loop().run_in_executor(None, _fetch, tickers)
                if prices:
                    _watchlist_price_cache["prices"].update(prices)
                    _watchlist_price_cache["ts"] = _time.monotonic()
        except asyncio.CancelledError:
            break
        except Exception:
            pass


@app.get("/watchlist")
async def watchlist_get():
    """Return watchlist entries enriched with current prices.

    Prices come from the background-refreshed _watchlist_price_cache (~1 ms).
    The first call before the background task fires returns prices as None.
    """
    entries = list(state["watchlist"].values())
    if not entries:
        return {"entries": []}

    current_prices = _watchlist_price_cache["prices"]

    enriched = []
    for e in entries:
        cp = current_prices.get(e["ticker"])
        ap = e["price_at_alert"]
        pct_chg = round((cp - ap) / ap * 100, 2) if cp and ap else None
        enriched.append({**e, "current_price": round(cp, 2) if cp else None, "pct_change": pct_chg})

    enriched.sort(key=lambda x: x.get("added_iso", ""), reverse=True)
    return {"entries": enriched}


@app.delete("/watchlist/{ticker}")
def watchlist_remove(ticker: str):
    """Remove a ticker from the watchlist."""
    removed = state["watchlist"].pop(ticker.upper(), None)
    if removed:
        _watchlist_save()
    return {"ok": True, "removed": removed is not None}


@app.delete("/watchlist")
def watchlist_clear():
    """Clear all watchlist entries."""
    state["watchlist"].clear()
    _watchlist_save()
    return {"ok": True}


@app.post("/scanner/trigger-eod-watchlist")
def scanner_trigger_eod_watchlist():
    """Write a trigger file that causes the scanner to run the EOD watchlist scan
    on its next loop iteration (within 3 minutes).

    The scanner checks for this file at the top of each cycle and deletes it after
    firing, so it is safe to call multiple times — only one scan will fire per file.
    """
    trigger = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trigger_eod_watchlist")
    try:
        open(trigger, "w").close()
        return {"ok": True, "message": "Trigger file written — EOD watchlist will fire within 3 min"}
    except OSError as e:
        raise HTTPException(500, f"Could not write trigger file: {e}")


@app.post("/market/regime/refresh")
async def market_regime_refresh():
    """Force-refresh the regime cache immediately, bypassing the 4-hour TTL.

    Called by the breakout scanner at 9:08 ET so the pre-market scan reflects
    overnight SPY moves rather than yesterday's stale cache.
    """
    try:
        await asyncio.get_event_loop().run_in_executor(None, _update_regime_cache_sync)
        regime = state["cache"].get("regime", {})
        return {
            "ok": True,
            "spy_above_sma200": regime.get("spy_above_sma200"),
            "spy_price":        regime.get("spy_price"),
            "spy_sma200":       regime.get("spy_sma200"),
            "updated":          regime.get("updated"),
        }
    except Exception as e:
        raise HTTPException(503, str(e))


# ── Auto-trader endpoints ──────────────────────────────────────────────────

@app.get("/autotrader/near-miss")
def autotrader_near_miss():
    """Daily log of tickers evaluated but not selected for CSP or LEAP.

    Resets each trading day. Reasons are merged across all 5-min scan cycles so
    the same ticker can accumulate multiple reasons if the blocking criteria changed
    during the day (e.g. started below score threshold, later fell into earnings window).
    """
    nm      = state["near_miss_log"]
    tickers = nm.get("tickers", {})
    rows    = sorted(
        [
            {
                "ticker":     tk,
                "type":       d["type"],
                "score":      round(d["score"], 1),
                "iv_rank":    round(d["iv_rank"], 1),
                "reasons":    d["reasons"],
                "last_seen":  d["last_seen"],
                "seen_count": d["seen_count"],
            }
            for tk, d in tickers.items()
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
    return {
        "date":         nm.get("date"),
        "total":        len(rows),
        "digest_sent":  nm.get("digest_sent", False),
        "tickers":      rows,
    }


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
                # Guard against None and nan — IBKR returns None when it hasn't
                # computed the value yet (e.g. after hours, data farm not warm).
                # `None or 0` would silently give $0, so we check explicitly.
                def _to_float_or_none(v):
                    if v is None: return None
                    try:
                        f = float(v)
                        return None if math.isnan(f) else f
                    except (TypeError, ValueError):
                        return None

                upnl_raw   = _to_float_or_none(item.unrealizedPNL)
                mprice_raw = _to_float_or_none(item.marketPrice)
                live_price = round(mprice_raw, 4) if mprice_raw is not None else None

                if k in positions_enriched:
                    pos_info = positions_enriched[k]
                    if upnl_raw is not None:
                        # IBKR computed it — use directly
                        live_pnl = round(upnl_raw, 2)
                    elif mprice_raw is not None and mprice_raw > 0:
                        # Compute from last known market price vs our stored entry price
                        entry = float(pos_info.get("entry_price", 0))
                        qty   = int(pos_info.get("qty", 1))
                        if pos_info.get("action") == "SELL":
                            live_pnl = round((entry - mprice_raw) * qty * 100, 2)
                        else:
                            live_pnl = round((mprice_raw - entry) * qty * 100, 2)
                    else:
                        live_pnl = None   # no data — UI shows "—" not "$0"
                    pos_info["live_pnl"]  = live_pnl
                    pos_info["live_price"] = live_price
                else:
                    # Option in portfolio but not tracked by auto-trader (manual trade)
                    untracked_positions.append({
                        "ticker":     getattr(c, "symbol", ""),
                        "strike":     float(getattr(c, "strike", 0)),
                        "right":      getattr(c, "right", ""),
                        "expiry":     (getattr(c, "lastTradeDateOrContractMonth", "") or "")[:8],
                        "action":     "BUY" if float(item.position or 0) > 0 else "SELL",
                        "qty":        abs(int(item.position or 0)),
                        "live_pnl":   round(upnl_raw, 2) if upnl_raw is not None else None,
                        "live_price": live_price,
                        "market_value": round(float(item.marketValue or 0), 2),
                        "avg_cost":   round(float(item.averageCost or 0), 4),
                    })
        except Exception:
            pass

    # ── Mark positions with active open orders (not yet filled) ─────────────────
    # These have no P&L yet — show "order pending" in the UI instead of "awaiting data".
    if ib and state.get("connected"):
        try:
            pending_order_ids = {
                t.order.orderId
                for t in ib.openTrades()
                if t.orderStatus.status in ("Submitted", "PreSubmitted", "PendingSubmit")
            }
            for info in positions_enriched.values():
                oid = info.get("order_id")
                if oid is not None and oid in pending_order_ids:
                    info["order_status"] = "pending"
        except Exception:
            pass

    # ── Fallback: direct option quote for positions missing from ib.portfolio() ──
    # ib.portfolio() sometimes omits CSP positions (e.g. after reconnect or when
    # IBKR hasn't reconciled the account yet).  For any tracked position that still
    # has no live_pnl, request a snapshot quote directly and compute P&L from mid.
    # Skip positions with pending orders — no real P&L exists until the order fills.
    if ib and state.get("connected"):
        missing = [
            (k, v) for k, v in positions_enriched.items()
            if "live_pnl" not in v and v.get("order_status") != "pending"
        ]
        if missing:
            try:
                async def _batch_mid(ib_conn, pos_list):
                    contracts = []
                    for _, info in pos_list:
                        tk     = info.get("ticker", "")
                        expiry = (info.get("expiry") or "")[:8]
                        strike = float(info.get("strike") or 0)
                        right  = info.get("right", "P").upper()
                        if tk and expiry and strike:
                            contracts.append(Option(tk, expiry, strike, right, "SMART"))
                    if not contracts:
                        return []
                    await ib_conn.qualifyContractsAsync(*contracts)
                    valid = [c for c in contracts if c.conId]
                    if not valid:
                        return []
                    return await ib_conn.reqTickersAsync(*valid)

                tickers_data = _run_in_streaming_loop(
                    _batch_mid(ib, missing), timeout=15
                )
                # Build conId → Ticker map for quick lookup
                ticker_map = {td.contract.conId: td for td in tickers_data}
                for _, info in missing:
                    tk     = info.get("ticker", "")
                    expiry = (info.get("expiry") or "")[:8]
                    strike = float(info.get("strike") or 0)
                    right  = info.get("right", "P").upper()
                    # Find the matching ticker by symbol+strike+expiry
                    td = next(
                        (t for t in tickers_data
                         if getattr(t.contract, "symbol", "") == tk
                         and abs(float(getattr(t.contract, "strike", 0)) - strike) < 0.01
                         and getattr(t.contract, "lastTradeDateOrContractMonth", "")[:8] == expiry),
                        None,
                    )
                    if td is None:
                        continue
                    bid = td.bid if td.bid and not math.isnan(td.bid) else None
                    ask = td.ask if td.ask and not math.isnan(td.ask) else None
                    mid = (bid + ask) / 2 if bid and ask else (ask or bid)
                    if mid is None:
                        continue
                    entry = float(info.get("entry_price") or 0)
                    qty   = int(info.get("qty") or 1)
                    action = info.get("action", "SELL")
                    live_pnl = round(
                        (entry - mid) * qty * 100 if action == "SELL"
                        else (mid - entry) * qty * 100,
                        2,
                    )
                    info["live_pnl"]   = live_pnl
                    info["live_price"] = round(mid, 4)
            except Exception as _e:
                log.debug("Fallback quote fetch failed: %s", _e)

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
        "auto_hedge":          req.auto_hedge,
        "hedge_threshold":     req.hedge_threshold,
        "tape_filter_enabled": req.tape_filter_enabled,
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


# ── Breakout Backtest — 6-Filter Analysis ───────────────────────────────────

_STOP_PCT = 0.05  # hard stop fraction for with_stops layer


def _norm_idx(s: pd.Series) -> pd.Series:
    """Strip timezone and normalise DatetimeIndex to date precision for cross-series alignment."""
    try:
        idx = pd.DatetimeIndex([pd.Timestamp(str(d)[:10]) for d in s.index])
        return pd.Series(s.values, index=idx)
    except Exception:
        return s


def _backtest_breakout_filtered_one(
    ticker: str,
    spy_close: pd.Series,
    spy_sma200: pd.Series,
    spy_ret20: pd.Series,
    vix_close: pd.Series,
) -> Optional[dict]:
    """Walk-forward breakout backtest with 6 cumulative filter layers.

    Layers (each cumulative):
      baseline      — %B > 95 AND vol_ratio >= rolling vol_90pct (no look-ahead)
      regime        — SPY > SMA-200 AND VIX < 25
      rel_strength  — stock 20d return > SPY 20d return (prior window, no look-ahead)
      pullback      — %B dipped below 50 in prior 10 days (coiled-spring setup)
      cvd_proxy     — breakout candle closes above open (net buy-pressure proxy)
      close_quality — close in top 25% of the day's high-low range
      with_stops    — same signals as close_quality; returns capped at -5% if
                      any intraday Low triggers the stop during the holding period
    """
    try:
        df = yf.Ticker(ticker).history(period="2y")
    except Exception as e:
        log.debug("Filtered BT fetch %s: %s", ticker, e)
        return None
    if len(df) < 252:
        return None

    # Normalise index to date-only so cross-series reindex works cleanly
    df.index = pd.DatetimeIndex([pd.Timestamp(str(d)[:10]) for d in df.index])

    closes  = df["Close"].astype(float)
    highs   = df["High"].astype(float)
    lows    = df["Low"].astype(float)
    opens   = df["Open"].astype(float)
    volumes = df["Volume"].astype(float)
    n = len(closes)

    # ── Bollinger Bands / %B ─────────────────────────────────────────────
    sma20    = closes.rolling(20).mean()
    bb_std   = closes.rolling(20).std()
    bb_upper = sma20 + 2 * bb_std
    bb_lower = sma20 - 2 * bb_std
    pct_b    = ((closes - bb_lower) / (bb_upper - bb_lower).replace(0, float("nan"))) * 100

    # ── Volume 90th-pct threshold ────────────────────────────────────────
    roll_avg  = volumes.rolling(20).mean().shift(1)
    vol_ratio = volumes / roll_avg.replace(0, float("nan"))
    vol_90pct = vol_ratio.rolling(252, min_periods=60).quantile(0.90)

    # ── Relative strength: prior 20d stock return (shift=1, no look-ahead) ─
    stock_ret20 = closes.pct_change(20).shift(1)

    # ── Close position within day range (0=low, 100=high) ───────────────
    day_range = (highs - lows).replace(0, float("nan"))
    close_pos = (closes - lows) / day_range * 100

    # ── Min %B in prior 10 days (shift=1 excludes today) ────────────────
    pct_b_min10 = pct_b.shift(1).rolling(10, min_periods=3).min()

    # ── Align SPY / VIX reference series to ticker dates ────────────────
    spy_c  = spy_close.reindex(closes.index, method="ffill")
    spy_s  = spy_sma200.reindex(closes.index, method="ffill")
    spy_r  = spy_ret20.reindex(closes.index, method="ffill")
    vix_c  = vix_close.reindex(closes.index, method="ffill")

    # ── Build cumulative signal masks ────────────────────────────────────
    m0 = (
        (pct_b > 95)
        & (vol_ratio >= vol_90pct)
        & vol_ratio.notna()
        & vol_90pct.notna()
    )
    m1 = m0 & spy_c.notna() & spy_s.notna() & vix_c.notna() \
             & (spy_c > spy_s) & (vix_c < 25)
    m2 = m1 & stock_ret20.notna() & spy_r.notna() \
             & (stock_ret20 > spy_r)
    m3 = m2 & pct_b_min10.notna() & (pct_b_min10 < 50)
    m4 = m3 & (closes > opens)
    m5 = m4 & close_pos.notna() & (close_pos >= 75)

    # ── Win-rate + expectancy calculator ─────────────────────────────────
    close_arr = closes.to_numpy()
    low_arr   = lows.to_numpy()

    def _stats(mask: pd.Series, apply_stops: bool = False) -> dict:
        idxs = [i for i, v in enumerate(mask.to_numpy()) if v]
        if not idxs:
            return {
                "signals": 0,
                "win_rate_5d": None, "win_rate_10d": None, "win_rate_20d": None,
                "avg_ret_5d":  None, "avg_ret_10d":  None, "avg_ret_20d":  None,
                "expectancy_10d": None,
            }

        rets5: list[float] = []
        rets10: list[float] = []
        rets20: list[float] = []

        for idx in idxs:
            p0 = close_arr[idx]
            if p0 <= 0:
                continue
            stop_px = p0 * (1 - _STOP_PCT) if apply_stops else None

            for horizon, bucket in ((5, rets5), (10, rets10), (20, rets20)):
                end = idx + horizon
                if end >= n:
                    continue
                if apply_stops:
                    ret = (close_arr[end] / p0 - 1) * 100
                    for j in range(1, horizon + 1):
                        if idx + j >= n:
                            break
                        if low_arr[idx + j] <= stop_px:
                            ret = -_STOP_PCT * 100  # stopped out
                            break
                    bucket.append(ret)
                else:
                    bucket.append((close_arr[end] / p0 - 1) * 100)

        def _wr(rs):  return round(sum(1 for r in rs if r > 0) / len(rs) * 100, 1) if rs else None
        def _avg(rs): return round(sum(rs) / len(rs), 2) if rs else None

        # Expectancy at 10d: wr * avg_win + (1-wr) * avg_loss
        pos10 = [r for r in rets10 if r > 0]
        neg10 = [r for r in rets10 if r <= 0]
        exp10: Optional[float] = None
        if rets10:
            wr_f  = len(pos10) / len(rets10)
            avg_w = sum(pos10) / len(pos10) if pos10 else 0.0
            avg_l = sum(neg10) / len(neg10) if neg10 else 0.0
            exp10 = round(wr_f * avg_w + (1 - wr_f) * avg_l, 2)

        return {
            "signals":        len(idxs),
            "win_rate_5d":    _wr(rets5),
            "win_rate_10d":   _wr(rets10),
            "win_rate_20d":   _wr(rets20),
            "avg_ret_5d":     _avg(rets5),
            "avg_ret_10d":    _avg(rets10),
            "avg_ret_20d":    _avg(rets20),
            "expectancy_10d": exp10,
        }

    return {
        "ticker":        ticker,
        "baseline":      _stats(m0),
        "regime":        _stats(m1),
        "rel_strength":  _stats(m2),
        "pullback":      _stats(m3),
        "cvd_proxy":     _stats(m4),
        "close_quality": _stats(m5),
        "with_stops":    _stats(m5, apply_stops=True),
    }


def _run_breakout_backtest_filtered() -> dict:
    """Download SPY + VIX once; run 6-layer filtered backtest on all universe tickers."""
    try:
        spy_raw  = yf.Ticker("SPY").history(period="2y")
        spy_c    = _norm_idx(spy_raw["Close"].astype(float))
        spy_s200 = spy_c.rolling(200).mean()
        spy_r20  = spy_c.pct_change(20).shift(1)
    except Exception as exc:
        log.warning("Filtered BT: SPY download failed: %s", exc)
        spy_c = spy_s200 = spy_r20 = pd.Series(dtype=float)

    try:
        vix_raw = yf.Ticker("^VIX").history(period="2y")
        vix_c   = _norm_idx(vix_raw["Close"].astype(float))
    except Exception as exc:
        log.warning("Filtered BT: VIX download failed: %s", exc)
        vix_c = pd.Series(dtype=float)

    tickers = list(CSP_UNIVERSE)
    rows: list[dict] = []
    for ticker in tickers:
        try:
            r = _backtest_breakout_filtered_one(ticker, spy_c, spy_s200, spy_r20, vix_c)
            if r:
                rows.append(r)
        except Exception as exc:
            log.warning("Filtered BT %s: %s", ticker, exc)

    # ── Aggregate summary per layer ───────────────────────────────────────
    _LAYERS = ("baseline", "regime", "rel_strength", "pullback",
               "cvd_proxy", "close_quality", "with_stops")

    def _agg_layer(lname: str) -> dict:
        layer_rows = [r[lname] for r in rows if lname in r]
        total_sig  = sum(x["signals"] for x in layer_rows)

        # Signal-count-weighted average win rates
        def _wagg(key: str) -> Optional[float]:
            pairs = [(x[key], x["signals"]) for x in layer_rows
                     if x.get(key) is not None and x["signals"] > 0]
            if not pairs:
                return None
            tw = sum(w for _, w in pairs)
            return round(sum(v * w for v, w in pairs) / tw, 1) if tw else None

        def _savg(key: str) -> Optional[float]:
            vals = [x[key] for x in layer_rows if x.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        return {
            "signals":        total_sig,
            "tickers_active": sum(1 for x in layer_rows if x["signals"] > 0),
            "win_rate_5d":    _wagg("win_rate_5d"),
            "win_rate_10d":   _wagg("win_rate_10d"),
            "win_rate_20d":   _wagg("win_rate_20d"),
            "avg_ret_10d":    _savg("avg_ret_10d"),
            "avg_ret_20d":    _savg("avg_ret_20d"),
            "expectancy_10d": _savg("expectancy_10d"),
        }

    summary = {lname: _agg_layer(lname) for lname in _LAYERS}

    return {
        "results": rows,
        "summary": summary,
        "tickers": tickers,
        "meta": {
            "filters": [
                "Baseline: %B > 95 AND volume ≥ rolling 90th-percentile threshold",
                "F1 Regime: SPY > SMA-200 AND VIX < 25 (healthy market environment)",
                "F2 Relative Strength: stock 20d return > SPY 20d return",
                "F3 Pullback: %B dipped below 50 in prior 10 days (coiled-spring setup)",
                "F4 CVD proxy: breakout candle closes above open (net buy-pressure)",
                "F5 Close Quality: close in top 25% of the day's high-low range",
                "F5 + Stops: 5% hard stop scanned against daily Low each holding day",
            ],
            "stop_pct": _STOP_PCT * 100,
        },
    }


@app.get("/backtest/breakout/filtered")
async def backtest_breakout_filtered_endpoint():
    """Breakout backtest with 6 cumulative filter layers + with-stops variant.

    Downloads SPY + ^VIX once, then runs per-ticker with all filter layers in order.
    Returns win rates at 5d / 10d / 20d and expectancy (expected return per signal)
    for each layer so you can see exactly how much each filter moves the needle.
    Uses yfinance only — no IBKR connection required.  Takes ~30-60 s on 20 tickers.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_breakout_backtest_filtered)
    return result


# ── Watchlist alert history + backtest ─────────────────────────────────────

@app.get("/alerts/history")
def alerts_history_endpoint(limit: int = 500):
    """Return persisted breakout alert history from alert_history table.

    Each row is one unique alert-fire event (only new tickers logged, not refreshes).
    Returns most-recent first.
    """
    try:
        con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
        rows = con.execute(
            """SELECT id, fired_at, session_date, ticker, signal_type, price,
                      pct_b, rsi, vol_ratio, tape_score, tape_label
               FROM alert_history
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        con.close()
        cols = ["id", "fired_at", "session_date", "ticker", "signal_type",
                "price", "pct_b", "rsi", "vol_ratio", "tape_score", "tape_label"]
        return {"alerts": [dict(zip(cols, r)) for r in rows], "total": len(rows)}
    except Exception as exc:
        return {"alerts": [], "total": 0, "error": str(exc)}


def _run_watchlist_backtest() -> dict:
    """Two-part watchlist backtest:

    Part 1 — Actual outcomes: for every alert in alert_history, fetch forward prices
    from yfinance and compute the real return at 5d / 10d / 20d from the alert price.
    This is a true paper-trading simulation of past alerts.

    Part 2 — Filter analysis: run the same 6-filter walk-forward analysis as
    /backtest/breakout/filtered but on the alerted tickers specifically (not just the
    CSP universe). Shows which filters would have improved the alerted signal set.
    """
    # ── Load alert history ──────────────────────────────────────────────────
    alerted_tickers: list[str] = []
    alert_rows: list[dict] = []
    try:
        con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
        raw = con.execute(
            """SELECT id, fired_at, session_date, ticker, signal_type,
                      price, pct_b, rsi, vol_ratio, tape_score, tape_label
               FROM alert_history ORDER BY fired_at""",
        ).fetchall()
        con.close()
        cols = ["id", "fired_at", "session_date", "ticker", "signal_type",
                "price", "pct_b", "rsi", "vol_ratio", "tape_score", "tape_label"]
        alert_rows = [dict(zip(cols, r)) for r in raw]
        alerted_tickers = list({r["ticker"] for r in alert_rows})
    except Exception as exc:
        log.warning("Watchlist BT: alert_history read failed: %s", exc)

    # Merge with current watchlist tickers (may include tickers from before DB existed)
    wl_tickers = list(state["watchlist"].keys())
    all_tickers = list({*alerted_tickers, *wl_tickers})

    # ── Part 1: Actual outcomes ─────────────────────────────────────────────
    # Group alerts by ticker, bulk-download price history, compute forward returns
    actual_outcomes: list[dict] = []
    if alert_rows:
        ticker_groups: dict[str, list[dict]] = {}
        for r in alert_rows:
            ticker_groups.setdefault(r["ticker"], []).append(r)

        for ticker, alerts in ticker_groups.items():
            try:
                hist = yf.Ticker(ticker).history(period="2y")
                if hist.empty:
                    continue
                closes = hist["Close"].astype(float)
                # Normalise dates for lookup
                date_to_idx: dict[str, int] = {
                    str(d)[:10]: i for i, d in enumerate(hist.index)
                }
                n = len(closes)
                close_arr = closes.to_numpy()

                for al in alerts:
                    alert_date = al["session_date"]
                    entry_price = al["price"]
                    if not entry_price or entry_price <= 0:
                        continue
                    idx = date_to_idx.get(alert_date)
                    if idx is None:
                        # Try T+1 (alert fired after close or on a holiday)
                        for offset in (1, 2):
                            try:
                                d = (datetime.strptime(alert_date, "%Y-%m-%d")
                                     + timedelta(days=offset)).strftime("%Y-%m-%d")
                                if d in date_to_idx:
                                    idx = date_to_idx[d]
                                    break
                            except Exception:
                                pass
                    if idx is None:
                        continue

                    def _fwd(h):
                        end = idx + h
                        if end >= n:
                            return None
                        return round((close_arr[end] / entry_price - 1) * 100, 2)

                    actual_outcomes.append({
                        "id":           al["id"],
                        "ticker":       ticker,
                        "signal_type":  al["signal_type"],
                        "session_date": alert_date,
                        "fired_at":     al["fired_at"],
                        "price":        entry_price,
                        "pct_b":        al["pct_b"],
                        "rsi":          al["rsi"],
                        "tape_label":   al["tape_label"],
                        "ret_5d":       _fwd(5),
                        "ret_10d":      _fwd(10),
                        "ret_20d":      _fwd(20),
                        "win_5d":       _fwd(5) is not None and _fwd(5) > 0,
                        "win_10d":      _fwd(10) is not None and _fwd(10) > 0,
                        "win_20d":      _fwd(20) is not None and _fwd(20) > 0,
                    })
            except Exception as exc:
                log.warning("Watchlist BT actual outcomes %s: %s", ticker, exc)

    # Aggregate actual outcome stats
    def _agg_outcomes(rows: list[dict], horizon: str) -> dict:
        rets = [r[f"ret_{horizon}"] for r in rows if r.get(f"ret_{horizon}") is not None]
        wins = [r for r in rets if r > 0]
        return {
            "n":        len(rets),
            "win_rate": round(len(wins) / len(rets) * 100, 1) if rets else None,
            "avg_ret":  round(sum(rets) / len(rets), 2) if rets else None,
        }

    breakout_outcomes  = [r for r in actual_outcomes if r["signal_type"] == "BREAKOUT"]
    pre_bo_outcomes    = [r for r in actual_outcomes if r["signal_type"] == "PRE-BREAKOUT"]

    outcome_summary = {
        "all": {
            "5d":  _agg_outcomes(actual_outcomes, "5d"),
            "10d": _agg_outcomes(actual_outcomes, "10d"),
            "20d": _agg_outcomes(actual_outcomes, "20d"),
        },
        "breakout": {
            "5d":  _agg_outcomes(breakout_outcomes, "5d"),
            "10d": _agg_outcomes(breakout_outcomes, "10d"),
            "20d": _agg_outcomes(breakout_outcomes, "20d"),
        },
        "pre_breakout": {
            "5d":  _agg_outcomes(pre_bo_outcomes, "5d"),
            "10d": _agg_outcomes(pre_bo_outcomes, "10d"),
            "20d": _agg_outcomes(pre_bo_outcomes, "20d"),
        },
    }

    # ── Part 2: 6-filter walk-forward analysis on alerted tickers ──────────
    filter_results: list[dict] = []
    filter_summary: dict = {}
    if all_tickers:
        try:
            spy_raw  = yf.Ticker("SPY").history(period="2y")
            spy_c    = _norm_idx(spy_raw["Close"].astype(float))
            spy_s200 = spy_c.rolling(200).mean()
            spy_r20  = spy_c.pct_change(20).shift(1)
        except Exception:
            spy_c = spy_s200 = spy_r20 = pd.Series(dtype=float)
        try:
            vix_c = _norm_idx(yf.Ticker("^VIX").history(period="2y")["Close"].astype(float))
        except Exception:
            vix_c = pd.Series(dtype=float)

        for ticker in all_tickers:
            try:
                r = _backtest_breakout_filtered_one(ticker, spy_c, spy_s200, spy_r20, vix_c)
                if r:
                    filter_results.append(r)
            except Exception as exc:
                log.warning("Watchlist filter BT %s: %s", ticker, exc)

        _LAYERS = ("baseline", "regime", "rel_strength", "pullback",
                   "cvd_proxy", "close_quality", "with_stops")

        def _agg_layer(lname: str) -> dict:
            layer_rows = [r[lname] for r in filter_results if lname in r]
            total_sig  = sum(x["signals"] for x in layer_rows)
            def _wagg(key):
                pairs = [(x[key], x["signals"]) for x in layer_rows
                         if x.get(key) is not None and x["signals"] > 0]
                if not pairs: return None
                tw = sum(w for _, w in pairs)
                return round(sum(v * w for v, w in pairs) / tw, 1) if tw else None
            def _savg(key):
                vals = [x[key] for x in layer_rows if x.get(key) is not None]
                return round(sum(vals) / len(vals), 2) if vals else None
            return {
                "signals":        total_sig,
                "tickers_active": sum(1 for x in layer_rows if x["signals"] > 0),
                "win_rate_5d":    _wagg("win_rate_5d"),
                "win_rate_10d":   _wagg("win_rate_10d"),
                "win_rate_20d":   _wagg("win_rate_20d"),
                "avg_ret_10d":    _savg("avg_ret_10d"),
                "expectancy_10d": _savg("expectancy_10d"),
            }

        filter_summary = {lname: _agg_layer(lname) for lname in _LAYERS}

    return {
        "tickers":         all_tickers,
        "n_alerted":       len(alerted_tickers),
        "n_watchlist":     len(wl_tickers),
        "actual_outcomes": actual_outcomes,
        "outcome_summary": outcome_summary,
        "filter_results":  filter_results,
        "filter_summary":  filter_summary,
    }


@app.get("/backtest/watchlist")
async def backtest_watchlist_endpoint():
    """Two-part watchlist backtest.

    Part 1 — Actual outcomes: each alert in alert_history is looked up in yfinance
    price data. Returns real 5d / 10d / 20d forward return from the alert price —
    a true paper-trading simulation of every alert that was generated.

    Part 2 — Filter analysis: 6-layer filtered walk-forward backtest run on all
    ever-alerted tickers so you can see which filters improve the specific signal set
    the breakout scanner has been generating.

    Takes ~45–90 s depending on how many unique tickers are in alert history.
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_watchlist_backtest)
    return result


# ── Alert Performance endpoints ────────────────────────────────────────────

@app.get("/performance/daily")
async def performance_daily(date: Optional[str] = Query(None)):
    from zoneinfo import ZoneInfo
    if date is None:
        date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    if date == today:
        await _enrich_day_performance(date)
    con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM alert_performance WHERE session_date=? ORDER BY eod_return_pct DESC NULLS LAST",
        (date,)
    ).fetchall()
    con.close()
    records = [dict(r) for r in rows]
    enriched = [r for r in records if r.get("eod_return_pct") is not None]
    wins = sum(1 for r in enriched if r.get("is_win") == 1)
    summary = {
        "date": date,
        "total": len(records),
        "enriched": len(enriched),
        "wins": wins,
        "losses": len(enriched) - wins,
        "win_rate": round(wins / len(enriched) * 100, 1) if enriched else None,
        "avg_return": round(sum(r["eod_return_pct"] for r in enriched) / len(enriched), 2) if enriched else None,
        "best": max(enriched, key=lambda r: r["eod_return_pct"], default=None),
        "worst": min(enriched, key=lambda r: r["eod_return_pct"], default=None),
    }
    return {"date": date, "rows": records, "summary": summary}


@app.get("/performance/summary")
async def performance_summary(days: int = Query(30)):
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM alert_performance WHERE session_date >= ? AND eod_return_pct IS NOT NULL ORDER BY session_date",
        (cutoff,)
    ).fetchall()]
    con.close()
    if not rows:
        return {"days": days, "total_alerts": 0, "rows": [], "daily": [], "by_ticker_count": [], "by_ticker_return": []}
    wins = sum(1 for r in rows if r.get("is_win") == 1)
    avg_ret = sum(r["eod_return_pct"] for r in rows) / len(rows)
    by_date: dict = {}
    for r in rows:
        d = r["session_date"]
        if d not in by_date:
            by_date[d] = {"date": d, "count": 0, "wins": 0, "returns": []}
        by_date[d]["count"] += 1
        if r.get("is_win") == 1:
            by_date[d]["wins"] += 1
        by_date[d]["returns"].append(r["eod_return_pct"])
    daily = []
    for d, v in sorted(by_date.items()):
        daily.append({
            "date": d,
            "count": v["count"],
            "win_rate": round(v["wins"] / v["count"] * 100, 1),
            "avg_return": round(sum(v["returns"]) / len(v["returns"]), 2),
        })
    ticker_stats: dict = {}
    for r in rows:
        t = r["ticker"]
        if t not in ticker_stats:
            ticker_stats[t] = {"ticker": t, "count": 0, "wins": 0, "returns": []}
        ticker_stats[t]["count"] += 1
        if r.get("is_win") == 1:
            ticker_stats[t]["wins"] += 1
        ticker_stats[t]["returns"].append(r["eod_return_pct"])
    ticker_list = []
    for t, v in ticker_stats.items():
        ticker_list.append({
            "ticker": t,
            "count": v["count"],
            "win_rate": round(v["wins"] / v["count"] * 100, 1),
            "avg_return": round(sum(v["returns"]) / len(v["returns"]), 2),
        })
    return {
        "days": days,
        "total_alerts": len(rows),
        "wins": wins,
        "losses": len(rows) - wins,
        "win_rate": round(wins / len(rows) * 100, 1),
        "avg_return": round(avg_ret, 2),
        "daily": daily,
        "by_ticker_count": sorted(ticker_list, key=lambda x: x["count"], reverse=True)[:20],
        "by_ticker_return": sorted(ticker_list, key=lambda x: x["avg_return"], reverse=True)[:20],
    }


@app.get("/performance/indicators")
async def performance_indicators(days: int = Query(30)):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: _compute_indicator_analysis(days))
    return result


@app.get("/performance/state-outcomes")
def performance_state_outcomes(days: int = Query(30)):
    """Analyze breakout state machine entry/exit outcomes.

    Conversion funnel: how many PRE-BREAKOUT setups convert to BREAKOUT vs fail.
    Intraday return: when close_price is available at both entry and exit transitions,
    compute entry→exit return. Historical rows pre-dating the close_price migration
    will show counts but no return stats.
    """
    from collections import defaultdict
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        """SELECT ticker, session_date, prev_state, new_state, pct_b, rsi,
                  transition_time_et, mins_in_prev_state, close_price
           FROM state_transitions WHERE session_date >= ?
           ORDER BY session_date, ticker, transition_time_et""",
        (cutoff,)
    ).fetchall()]
    # EOD returns for correlation: ticker+date → eod_return_pct
    eod_map = {}
    for r in con.execute(
        "SELECT ticker, session_date, eod_return_pct, is_win FROM alert_performance WHERE session_date >= ?",
        (cutoff,)
    ).fetchall():
        eod_map[(r["ticker"], r["session_date"])] = {
            "eod_return_pct": r["eod_return_pct"], "is_win": r["is_win"]
        }
    con.close()

    _BULLISH = {"PRE-BREAKOUT", "BREAKOUT", "EXTENDED"}

    def _ret_stats(returns):
        if not returns:
            return {"win_rate": None, "avg_return": None, "best": None, "worst": None, "with_price_data": 0}
        wins = sum(1 for r in returns if r > 0)
        return {
            "win_rate":       round(wins / len(returns) * 100, 1),
            "avg_return":     round(sum(returns) / len(returns), 2),
            "best":           round(max(returns), 2),
            "worst":          round(min(returns), 2),
            "with_price_data": len(returns),
        }

    # Group transitions by (ticker, session_date)
    by_day: dict = defaultdict(list)
    for r in rows:
        by_day[(r["ticker"], r["session_date"])].append(r)

    # ── Setup funnel: non-bullish → PRE-BREAKOUT ──────────────────────────────
    setup_groups: dict = defaultdict(list)   # key = "PREV→PRE-BREAKOUT"
    breakout_warm: list = []   # BREAKOUT via PRE-BREAKOUT path
    breakout_cold: list = []   # BREAKOUT direct from NEUTRAL/WEAKENING

    for (ticker, session_date), trans in by_day.items():
        trans.sort(key=lambda x: x["transition_time_et"])
        eod = eod_map.get((ticker, session_date), {})

        for i, t in enumerate(trans):
            subsequent = trans[i + 1:]

            # ── Setup entry: → PRE-BREAKOUT from non-bullish ──────────────
            if t["new_state"] == "PRE-BREAKOUT" and t["prev_state"] not in _BULLISH:
                reached_bo = any(s["new_state"] == "BREAKOUT" for s in subsequent)
                failed = any(
                    s["prev_state"] == "PRE-BREAKOUT" and s["new_state"] not in _BULLISH
                    for s in subsequent
                )
                # First exit (BREAKOUT conversion OR fade)
                exit_t = None
                for s in subsequent:
                    if s["new_state"] == "BREAKOUT" or (
                        s["prev_state"] == "PRE-BREAKOUT" and s["new_state"] not in _BULLISH
                    ):
                        exit_t = s
                        break

                entry_price = t.get("close_price")
                exit_price  = exit_t.get("close_price") if exit_t else None
                intra_ret   = round((exit_price - entry_price) / entry_price * 100, 2) \
                              if (entry_price and exit_price and entry_price > 0) else None

                setup_groups[f"{t['prev_state']}→PRE-BREAKOUT"].append({
                    "ticker":          ticker,
                    "session_date":    session_date,
                    "entry_time":      t["transition_time_et"],
                    "reached_breakout": reached_bo,
                    "failed":          failed,
                    "still_active":    not reached_bo and not failed,
                    "exit_type":       exit_t["new_state"] if exit_t else None,
                    "intra_ret":       intra_ret,
                    "eod_ret":         eod.get("eod_return_pct"),
                    "is_win":          eod.get("is_win"),
                })

            # ── Breakout entry ─────────────────────────────────────────────
            if t["new_state"] == "BREAKOUT":
                went_ext = any(s["new_state"] == "EXTENDED" for s in subsequent)
                faded    = any(
                    s["prev_state"] in {"BREAKOUT", "EXTENDED"} and
                    s["new_state"] not in _BULLISH
                    for s in subsequent
                )
                exit_t = None
                for s in subsequent:
                    if (s["prev_state"] in {"BREAKOUT", "EXTENDED"}
                            and s["new_state"] not in _BULLISH):
                        exit_t = s
                        break

                entry_price = t.get("close_price")
                exit_price  = exit_t.get("close_price") if exit_t else None
                intra_ret   = round((exit_price - entry_price) / entry_price * 100, 2) \
                              if (entry_price and exit_price and entry_price > 0) else None

                rec = {
                    "ticker":       ticker,
                    "session_date": session_date,
                    "prev_state":   t["prev_state"],
                    "went_extended": went_ext,
                    "faded":        faded,
                    "exit_type":    exit_t["new_state"] if exit_t else None,
                    "intra_ret":    intra_ret,
                    "eod_ret":      eod.get("eod_return_pct"),
                    "is_win":       eod.get("is_win"),
                }
                if t["prev_state"] == "PRE-BREAKOUT":
                    breakout_warm.append(rec)
                else:
                    breakout_cold.append(rec)

    # ── Build funnel summary ──────────────────────────────────────────────────
    funnel = []
    for entry_label, entries in sorted(setup_groups.items(),
                                        key=lambda x: -len(x[1])):
        n = len(entries)
        converted = sum(1 for e in entries if e["reached_breakout"])
        failed    = sum(1 for e in entries if e["failed"])
        intra_rets = [e["intra_ret"] for e in entries if e["intra_ret"] is not None]
        eod_rets   = [e["eod_ret"] for e in entries
                      if e["eod_ret"] is not None and e["reached_breakout"]]
        funnel.append({
            "entry":             entry_label,
            "count":             n,
            "pct_converted":     round(converted / n * 100, 1),
            "pct_failed":        round(failed    / n * 100, 1),
            "pct_still_active":  round((n - converted - failed) / n * 100, 1),
            **_ret_stats(intra_rets),
            "eod_win_rate":      round(sum(1 for r in eod_rets if r > 0) / len(eod_rets) * 100, 1)
                                 if eod_rets else None,
            "eod_avg_return":    round(sum(eod_rets) / len(eod_rets), 2) if eod_rets else None,
        })

    # ── Breakout quality (warm vs cold) ──────────────────────────────────────
    def _bo_block(entries):
        n = len(entries)
        if n == 0:
            return {"count": 0}
        went_ext = sum(1 for e in entries if e["went_extended"])
        faded    = sum(1 for e in entries if e["faded"])
        intra_rets = [e["intra_ret"] for e in entries if e["intra_ret"] is not None]
        eod_rets   = [e["eod_ret"]   for e in entries if e["eod_ret"]   is not None]
        return {
            "count":          n,
            "pct_extended":   round(went_ext / n * 100, 1),
            "pct_faded":      round(faded    / n * 100, 1),
            "pct_held":       round((n - faded) / n * 100, 1),
            **_ret_stats(intra_rets),
            "eod_win_rate":   round(sum(1 for r in eod_rets if r > 0) / len(eod_rets) * 100, 1)
                              if eod_rets else None,
            "eod_avg_return": round(sum(eod_rets) / len(eod_rets), 2) if eod_rets else None,
        }

    # ── Exit type analysis ────────────────────────────────────────────────────
    exit_buckets: dict = defaultdict(list)
    for e in list(setup_groups.values())[0:0]:   # just to initialise pattern
        pass
    for entries in setup_groups.values():
        for e in entries:
            if e["exit_type"]:
                exit_buckets[e["exit_type"]].append(e)
    for e in breakout_warm + breakout_cold:
        if e["exit_type"]:
            exit_buckets[e["exit_type"]].append(e)

    exit_summary = []
    for exit_type, entries in sorted(exit_buckets.items()):
        intra = [e["intra_ret"] for e in entries if e["intra_ret"] is not None]
        exit_summary.append({
            "exit_type": exit_type,
            "count":     len(entries),
            **_ret_stats(intra),
        })

    return {
        "days":               days,
        "total_transitions":  len(rows),
        "setup_entries":      sum(len(v) for v in setup_groups.values()),
        "breakout_entries":   len(breakout_warm) + len(breakout_cold),
        "funnel":             funnel,
        "breakout": {
            "via_pre_breakout": _bo_block(breakout_warm),
            "cold_jump":        _bo_block(breakout_cold),
        },
        "exits": exit_summary,
    }


@app.post("/performance/enrich")
async def performance_enrich(date: Optional[str] = Query(None)):
    from zoneinfo import ZoneInfo
    if date is None:
        date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    count = await _enrich_day_performance(date)
    return {"date": date, "enriched": count}


@app.post("/performance/telegram-digest")
async def performance_telegram_digest(date: Optional[str] = Query(None)):
    """Manually (re)send the EOD win-rate/avg-return/top-winners-losers digest to Telegram."""
    from zoneinfo import ZoneInfo
    if date is None:
        date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    sent = await _send_eod_performance_digest(date)
    return {"date": date, "sent": sent}


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

        # Auto-close open journal entries not present in IBKR portfolio.
        # Guard: only orphan when portfolio_items is non-empty. If the feed is
        # temporarily empty (post-reconnect, TWS restart), ibkr_keys = {} and
        # every open entry would be falsely orphaned, corrupting the journal.
        # Also guard: skip entries that have an active open order (Submitted /
        # PreSubmitted) — unfilled limit orders are not in the portfolio yet
        # and should not be marked orphaned before they have a chance to fill.
        active_order_jkeys: set = set()
        try:
            for t in state["ib"].openTrades():
                if t.orderStatus.status in ("Submitted", "PreSubmitted", "PendingSubmit"):
                    c = t.contract
                    active_order_jkeys.add(_jkey(
                        getattr(c, "symbol", ""),
                        getattr(c, "strike", None),
                        getattr(c, "right", None),
                        (getattr(c, "lastTradeDateOrContractMonth", "") or "")[:8],
                    ))
        except Exception:
            pass

        orphan_ids = [
            t["id"] for t in open_trades
            if _jkey(t.get("ticker",""), t.get("strike"), t.get("right",""), t.get("expiry",""))
            not in ibkr_keys
            and _jkey(t.get("ticker",""), t.get("strike"), t.get("right",""), t.get("expiry",""))
            not in active_order_jkeys
        ]
        if orphan_ids and portfolio_items:
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


# ── Tape Sentiment REST endpoints ─────────────────────────────────────────
@app.get("/tape/sentiment")
def tape_sentiment_all():
    """Return CVD tape sentiment for all subscribed tickers."""
    out = {}
    for sym, sent in state["tape_sentiment"].items():
        out[sym] = {
            "score":       sent.get("score", 0.0),
            "label":       sent.get("label", "NEUTRAL"),
            "components":  sent.get("components", {}),
            "session_vwap": sent.get("session_vwap", 0.0),
            "session_vol": sent.get("session_vol", 0),
            "last_updated": sent.get("last_updated"),
            "sub_active":  sent.get("sub_active", False),
            "fresh":       _tape_is_fresh(sent),
        }
    return out


@app.get("/tape/sentiment/{ticker}")
def tape_sentiment_single(ticker: str):
    """Return CVD tape sentiment for a single ticker."""
    sym  = ticker.upper()
    sent = state["tape_sentiment"].get(sym)
    if sent is None:
        raise HTTPException(404, f"No tape sentiment data for {sym} — not subscribed or not in watchlist")
    return {
        "ticker":       sym,
        "score":        sent.get("score", 0.0),
        "label":        sent.get("label", "NEUTRAL"),
        "components":   sent.get("components", {}),
        "session_vwap": sent.get("session_vwap", 0.0),
        "session_vol":  sent.get("session_vol", 0),
        "last_updated": sent.get("last_updated"),
        "sub_active":   sent.get("sub_active", False),
        "fresh":        _tape_is_fresh(sent),
    }


@app.get("/tape/prints/{ticker}")
def tape_prints_history(
    ticker: str,
    date: Optional[str] = Query(None, description="Session date YYYY-MM-DD (default: today)"),
    blocks_only: bool   = Query(False, description="Return only block prints (is_block=1)"),
    limit: int          = Query(1000, description="Max rows to return"),
):
    """Historical tick-by-tick prints for a ticker from tape_data.db."""
    sym          = ticker.upper()
    session_date = date or datetime.utcnow().strftime("%Y-%m-%d")
    try:
        con   = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
        where = "ticker=? AND session_date=?"
        args  = [sym, session_date]
        if blocks_only:
            where += " AND is_block=1"
        rows = con.execute(
            f"SELECT ts,price,size,direction,exchange,is_block,is_after_hours,"
            f"vwap,cum_vol,buy_vol,sell_vol,net_delta,pct_from_open,cvd_score "
            f"FROM tape_prints WHERE {where} ORDER BY ts DESC LIMIT ?",
            args + [limit],
        ).fetchall()
        con.close()
        cols = ["ts","price","size","direction","exchange","is_block","is_after_hours",
                "vwap","cum_vol","buy_vol","sell_vol","net_delta","pct_from_open","cvd_score"]
        return {"ticker": sym, "date": session_date, "count": len(rows),
                "prints": [dict(zip(cols, r)) for r in rows]}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/tape/bars/{ticker}")
def tape_bars_history(
    ticker: str,
    date: Optional[str] = Query(None, description="Session date YYYY-MM-DD (default: today)"),
    limit: int          = Query(500, description="Max bars to return"),
):
    """Historical 1-minute CVD bars for a ticker from tape_data.db."""
    sym          = ticker.upper()
    session_date = date or datetime.utcnow().strftime("%Y-%m-%d")
    try:
        con  = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
        rows = con.execute(
            "SELECT bar_start,open,close,delta,vol,buy_vol,sell_vol,cvd_score,vwap_z,label "
            "FROM tape_bars WHERE ticker=? AND session_date=? ORDER BY bar_start LIMIT ?",
            (sym, session_date, limit),
        ).fetchall()
        con.close()
        cols = ["bar_start","open","close","delta","vol","buy_vol","sell_vol",
                "cvd_score","vwap_z","label"]
        return {"ticker": sym, "date": session_date, "count": len(rows),
                "bars": [dict(zip(cols, r)) for r in rows]}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/tape/sessions")
def tape_sessions(
    ticker: Optional[str] = Query(None, description="Filter by ticker (optional)"),
):
    """List all session dates available in tape_data.db."""
    try:
        con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)
        if ticker:
            rows = con.execute(
                "SELECT ticker, session_date, COUNT(*) as prints "
                "FROM tape_prints WHERE ticker=? GROUP BY ticker, session_date ORDER BY session_date DESC",
                (ticker.upper(),),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT ticker, session_date, COUNT(*) as prints "
                "FROM tape_prints GROUP BY ticker, session_date ORDER BY session_date DESC, ticker",
            ).fetchall()
        con.close()
        return {"sessions": [{"ticker": r[0], "date": r[1], "prints": r[2]} for r in rows]}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/tape/blocks")
def tape_block_report(
    date:  Optional[str] = Query(None, description="Session date YYYY-MM-DD (default: today ET)"),
    limit: int           = Query(500,  description="Max individual block rows to return"),
):
    """
    Institutional block flow report for a given session date.
    Returns both individual block prints (sorted by size DESC) and a per-ticker summary.
    Blocks are captured in real-time from the CVD sentiment subscription for all tracked
    tickers, plus any WS sessions the user opened that day.
    """
    try:
        from zoneinfo import ZoneInfo
        sess_date = date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        sess_date = date or datetime.utcnow().strftime("%Y-%m-%d")
    try:
        con = sqlite3.connect(TAPE_DB_PATH, check_same_thread=False)

        # Individual block prints — largest first
        rows = con.execute("""
            SELECT ts, ticker, price, size, direction, exchange,
                   is_after_hours, vwap, net_delta, pct_from_open, cvd_score
            FROM tape_prints
            WHERE session_date=? AND is_block=1
            ORDER BY size DESC LIMIT ?
        """, (sess_date, limit)).fetchall()

        # Per-ticker aggregates
        agg_rows = con.execute("""
            SELECT ticker,
                   COUNT(*)                                               AS block_count,
                   SUM(size)                                              AS total_shares,
                   SUM(CASE WHEN direction >= 0 THEN size ELSE 0 END)    AS buy_shares,
                   SUM(CASE WHEN direction  < 0 THEN size ELSE 0 END)    AS sell_shares,
                   MAX(size)                                              AS largest_print,
                   ROUND(SUM(price * size) / NULLIF(SUM(size), 0), 2)    AS vwap_block,
                   MIN(ts)                                                AS first_seen,
                   MAX(ts)                                                AS last_seen
            FROM tape_prints
            WHERE session_date=? AND is_block=1
            GROUP BY ticker
            ORDER BY total_shares DESC
        """, (sess_date,)).fetchall()
        con.close()

        # Build print dicts with computed fields
        pcols = ["ts","ticker","price","size","direction","exchange",
                 "is_after_hours","vwap","net_delta","pct_from_open","cvd_score"]
        prints = []
        for i, r in enumerate(rows, 1):
            p = dict(zip(pcols, r))
            p["rank"]        = i
            p["dollar_value"]= round(float(p["price"]) * int(p["size"]))
            p["side"]        = "BUY" if (p["direction"] or 0) >= 0 else "SELL"
            prints.append(p)

        # Build ticker summary dicts
        acols = ["ticker","block_count","total_shares","buy_shares","sell_shares",
                 "largest_print","vwap_block","first_seen","last_seen"]
        by_ticker = []
        for r in agg_rows:
            s = dict(zip(acols, r))
            s["buy_shares"]  = s["buy_shares"]  or 0
            s["sell_shares"] = s["sell_shares"] or 0
            s["net_flow"]    = s["buy_shares"] - s["sell_shares"]
            s["dollar_vol"]  = round(float(s["vwap_block"] or 0) * int(s["total_shares"] or 0))
            s["bias"]        = ("STRONGLY BULLISH" if s["net_flow"] > s["total_shares"] * 0.6 else
                                "BULLISH"          if s["net_flow"] > 0                  else
                                "STRONGLY BEARISH" if s["net_flow"] < -s["total_shares"] * 0.6 else
                                "BEARISH")
            by_ticker.append(s)

        total_shares = sum(p["size"] for p in prints)
        buy_shares   = sum(p["size"] for p in prints if p["side"] == "BUY")
        sell_shares  = total_shares - buy_shares
        total_dolval = sum(p["dollar_value"] for p in prints)

        return {
            "date":          sess_date,
            "total_blocks":  len(prints),
            "total_shares":  total_shares,
            "buy_shares":    buy_shares,
            "sell_shares":   sell_shares,
            "total_dollar_vol": total_dolval,
            "by_ticker":     by_ticker,
            "prints":        prints,
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/market/history/bulk")
async def market_history_bulk(tickers: List[str], days: int = Query(5, ge=1, le=365)):
    """Fetch last N trading days of daily OHLCV for multiple tickers via IBKR.

    Used by the breakout scanner to refresh its historical cache each cycle.
    Falls back to empty data for any ticker IBKR cannot serve (not a subscriber,
    not connected, etc.) — the scanner falls back to yfinance for those.

    Paced per IBKR rules: up to 50 simultaneous requests, 2s between groups.
    """
    ib = state.get("ib")
    if not ib or not ib.isConnected():
        raise HTTPException(503, "IBKR not connected — scanner should use yfinance fallback")

    duration_str = f"{max(days * 2, 10)} D"   # extra buffer for weekends/holidays

    async def _one(ticker: str) -> tuple[str, list]:
        try:
            from ib_insync import Stock
            contract = Stock(ticker, "SMART", "USD")
            bars = await asyncio.wait_for(
                ib.reqHistoricalDataAsync(
                    contract, endDateTime="", durationStr=duration_str,
                    barSizeSetting="1 day", whatToShow="TRADES",
                    useRTH=True, keepUpToDate=False,
                ),
                timeout=15,
            )
            if not bars:
                return ticker, []
            return ticker, [
                {"date": str(b.date), "open": b.open, "high": b.high,
                 "low": b.low, "close": b.close, "volume": b.volume}
                for b in bars[-days:]      # trim to exactly the requested days
            ]
        except Exception:
            return ticker, []

    results: dict[str, list] = {}
    # IBKR allows ~50 simultaneous historical requests; batch with 2s pause
    batch_size = 40
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_results = await asyncio.gather(*[_one(tk) for tk in batch])
        for tk, bars in batch_results:
            if bars:
                results[tk] = bars
        if i + batch_size < len(tickers):
            await asyncio.sleep(2)   # IBKR pacing between batches

    return {"data": results, "tickers_requested": len(tickers), "tickers_returned": len(results)}


@app.get("/market/regime")
def market_regime():
    """Market regime snapshot for the breakout scanner.

    Returns the cached SPY SMA-200 regime + live IBKR VIX so the scanner can
    apply F1 (regime/VIX) filters without duplicating the computation.
    regime_ok=False means the scanner should suppress all new alerts this cycle.
    """
    regime   = state["cache"].get("regime") or {}
    vix_live = state["vix_live"].get("price")
    spy_ok   = regime.get("spy_above_sma200", True)  # True = benefit of doubt when cache cold
    vix_ok   = vix_live is None or vix_live < 25

    reason: list[str] = []
    if spy_ok is False:
        reason.append(
            f"SPY ${regime.get('spy_price', 0):.2f} below SMA-200 ${regime.get('spy_sma200', 0):.0f}"
        )
    if not vix_ok:
        reason.append(f"VIX={vix_live:.1f} ≥ 25")

    return {
        "regime_ok":        spy_ok is not False and vix_ok,
        "spy_price":        regime.get("spy_price"),
        "spy_sma200":       regime.get("spy_sma200"),
        "spy_above_sma200": spy_ok,
        "spy_ret20":        regime.get("spy_ret20"),
        "stock_ret20":      regime.get("stock_ret20", {}),
        "vix_live":         vix_live,
        "vix_threshold":    25,
        "reason":           " | ".join(reason) if reason else "ok",
        "updated":          regime.get("updated"),
    }


@app.get("/market/indexes")
def market_indexes():
    """
    Live prices for the five main market indexes (S&P 500, Nasdaq, Dow, Russell, VIX).

    Price source priority for each ticker:
      1. IBKR tape_sentiment last_price — real-time for SPY/QQQ/DIA/IWM (Stock ETF subs)
      2. IBKR vix_live                  — real-time for VIX (Index contract sub)
      3. yfinance fast_info             — fallback prev_close + price when IBKR unavailable
    yfinance results cached 15 s; IBKR prices overlaid on every call (no extra latency).
    """
    import time as _time, copy

    now = _time.time()
    # Rebuild yfinance baseline only when cache is stale
    if _index_cache["data"] is None or now - _index_cache["ts"] > INDEX_CACHE_TTL:
        fresh: list = []
        for cfg in INDEX_CONFIG:
            entry = {
                "sym":        cfg["sym"],
                "name":       cfg["name"],
                "price":      None,
                "prev_close": None,
                "change":     None,
                "change_pct": None,
                "is_live":    False,
            }
            yf_sym = "^VIX" if cfg["sym"] == "VIX" else cfg["sym"]
            try:
                fi = yf.Ticker(yf_sym).fast_info
                lp = getattr(fi, "last_price",    None)
                pc = getattr(fi, "previous_close", None)
                if lp:  entry["price"]      = round(float(lp), 2)
                if pc:  entry["prev_close"] = round(float(pc), 2)
            except Exception:
                pass
            fresh.append(entry)
        _index_cache["data"] = fresh
        _index_cache["ts"]   = now

    # Deep-copy cached data then overlay live IBKR prices
    result = copy.deepcopy(_index_cache["data"])
    for entry in result:
        sym = entry["sym"]

        if sym == "VIX":
            # VIX from dedicated Index contract subscription
            vl = state.get("vix_live", {})
            if vl.get("price"):
                entry["price"]   = vl["price"]
                entry["is_live"] = True
            if vl.get("prev_close") and not entry["prev_close"]:
                entry["prev_close"] = vl["prev_close"]
        else:
            # ETF from tape_sentiment (last_price field populated by CVD callback)
            sent = state["tape_sentiment"].get(sym)
            if sent and sent.get("last_price") and _tape_is_fresh(sent):
                entry["price"]   = round(float(sent["last_price"]), 2)
                entry["is_live"] = True

        # Recompute change after price overlay
        if entry["price"] is not None and entry["prev_close"]:
            entry["change"]     = round(entry["price"] - entry["prev_close"], 2)
            entry["change_pct"] = round(
                (entry["price"] - entry["prev_close"]) / entry["prev_close"] * 100, 2
            )

    return result


# ── Stock Trader REST API ──────────────────────────────────────────────────

class StockSignalRequest(BaseModel):
    ticker:          str
    price:           float
    alert_fired_at:  Optional[str]   = None   # ISO timestamp from scanner
    composite_score: Optional[float] = None   # 0-100 percentile from breakout scanner


class StockConfigRequest(BaseModel):
    position_size:        Optional[float] = None
    max_positions:        Optional[int]   = None
    hard_stop_pct:        Optional[float] = None
    trail_pct:            Optional[float] = None
    max_hold_days:        Optional[int]   = None
    signal_freshness_min: Optional[int]   = None
    limit_buffer_pct:     Optional[float] = None
    rotation_enabled:     Optional[bool]  = None


@app.post("/stock-trader/signal")
async def stock_trader_signal(req: StockSignalRequest):
    """Fast-path entry point called directly by breakout_scanner.py on BREAKOUT.

    Eliminates the 5-minute AT polling delay — scanner fires this immediately,
    backend places a LIMIT buy within seconds of signal detection.
    """
    from zoneinfo import ZoneInfo
    st  = state["stock_trader"]
    cfg = st["config"]

    if not st["enabled"]:
        return {"status": "skipped", "reason": "disabled"}

    # Market hours check (09:30–15:50 ET, leaves 10 min buffer before close)
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return {"status": "skipped", "reason": "weekend"}
    mkt_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    mkt_close = now_et.replace(hour=15, minute=50, second=0, microsecond=0)
    if not (mkt_open <= now_et <= mkt_close):
        return {"status": "skipped", "reason": "outside_hours"}

    # Signal freshness check
    if req.alert_fired_at:
        try:
            fired_at = datetime.fromisoformat(req.alert_fired_at)
            # Make both timezone-aware for comparison
            if fired_at.tzinfo is None:
                fired_at = fired_at.replace(tzinfo=ZoneInfo("America/New_York"))
            now_aware = now_et
            age_min = (now_aware - fired_at).total_seconds() / 60
            if age_min > cfg["signal_freshness_min"]:
                _st_log("SKIPPED", req.ticker.upper(),
                        f"stale signal ({age_min:.0f} min old, limit {cfg['signal_freshness_min']} min)")
                return {"status": "skipped", "reason": "stale_signal", "age_min": round(age_min, 1)}
        except Exception:
            pass

    ticker = req.ticker.upper()

    # Duplicate check
    if ticker in st["positions"]:
        _st_log("SKIPPED", ticker, "already have open position")
        return {"status": "skipped", "reason": "already_open"}

    # Capacity check — with optional rotation
    if len(st["positions"]) >= cfg["max_positions"]:
        score = req.composite_score
        if cfg.get("rotation_enabled") and score is not None:
            candidate, reason = _st_find_rotation_candidate(ticker, score, st)
            if candidate:
                cand_pos   = st["positions"][candidate]
                evict_pnl  = cand_pos.get("live_pnl", 0) or 0
                evict_days = cand_pos.get("trading_days_held", 0) or _st_trading_days_held(cand_pos.get("entry_date", ""))
                evict_sec  = STOCK_SECTOR_MAP.get(candidate, "Unknown")
                new_sec    = STOCK_SECTOR_MAP.get(ticker, "Unknown")
                avg_score  = _st_avg_score(st)

                # Place MKT sell for evicted position (reuse close endpoint logic)
                _ib = state.get("ib")
                if _ib and _ib.isConnected():
                    stop_oid = cand_pos.get("stop_order_id")
                    async def _do_evict(_ib=_ib, _ticker=candidate, _pos=cand_pos, _stop=stop_oid):
                        open_by_oid = {t.order.orderId: t for t in _ib.openTrades()}
                        if _stop and _stop in open_by_oid:
                            _ib.cancelOrder(open_by_oid[_stop].order)
                            await asyncio.sleep(1)
                        if _pos.get("phase", 1) > 0:
                            contract = Stock(_ticker, "SMART", "USD")
                            mkt_ord = Order()
                            mkt_ord.orderType     = "MKT"
                            mkt_ord.action        = "SELL"
                            mkt_ord.totalQuantity = _pos["shares"]
                            mkt_ord.tif           = "DAY"
                            trade = _ib.placeOrder(contract, mkt_ord)
                            await asyncio.sleep(0.5)
                            return trade.order.orderId
                        return None
                    try:
                        loop = asyncio.get_event_loop()
                        sell_oid = await loop.run_in_executor(
                            None,
                            lambda: _run_in_streaming_loop(_do_evict(), timeout=15),
                        )
                        cand_pos["phase"] = 3
                        if sell_oid:
                            cand_pos["stop_order_id"] = sell_oid
                    except Exception as _exc:
                        _st_log("ERROR", candidate, f"rotation evict order failed: {_exc}")
                        return {"status": "skipped", "reason": "rotation_evict_failed"}

                decision_note = (
                    f"Evicted {candidate} ({reason}) "
                    f"→ {ticker} score={score:.0f} (portfolio avg={avg_score:.0f}, sector={new_sec})"
                )
                _st_log("ROTATION", ticker, decision_note)

                # Record rotation for outcome tracking
                rot_entry = {
                    "ts":                           now_et.isoformat(),
                    "evicted":                      candidate,
                    "evicted_entry_price":          cand_pos.get("entry_price"),
                    "evicted_pnl_at_rotation":      evict_pnl,
                    "evicted_days_held":            evict_days,
                    "evicted_sector":               evict_sec,
                    "incoming":                     ticker,
                    "incoming_score":               score,
                    "incoming_sector":              new_sec,
                    "portfolio_avg_score_at_rotation": avg_score,
                    "outcome_5d":                   None,
                }
                st.setdefault("rotation_log", []).append(rot_entry)
                st["rotation_log"] = st["rotation_log"][-50:]

                st["positions"].pop(candidate, None)
                _st_save_state()
                # fall through to normal entry below
            else:
                _st_log("SKIPPED", ticker,
                        f"at capacity, rotation blocked: {reason}")
                return {"status": "skipped", "reason": f"rotation_blocked: {reason}"}
        else:
            _st_log("SKIPPED", ticker,
                    f"at capacity ({len(st['positions'])}/{cfg['max_positions']} positions)")
            return {"status": "skipped", "reason": "at_capacity"}

    # IBKR connection check
    ib = state.get("ib")
    if not ib or not ib.isConnected():
        raise HTTPException(503, "IBKR not connected")

    price   = req.price
    shares  = max(1, int(cfg["position_size"] / price))
    lmt_px  = round(price * (1 + cfg["limit_buffer_pct"] / 100), 2)
    cost    = round(shares * price, 2)

    async def _do_buy(ib):
        contract = Stock(ticker, "SMART", "USD")
        await ib.qualifyContractsAsync(contract)
        if not contract.conId:
            raise ValueError(f"Cannot qualify {ticker}")
        buy_ord = LimitOrder("BUY", shares, lmt_px)
        buy_ord.tif = "DAY"
        trade = ib.placeOrder(contract, buy_ord)
        await asyncio.sleep(0.5)
        return trade.order.orderId

    try:
        loop   = asyncio.get_event_loop()
        buy_id = await loop.run_in_executor(
            None,
            lambda: _run_in_streaming_loop(_do_buy(ib), timeout=15),
        )
    except (ValueError, TimeoutError, RuntimeError) as exc:
        _st_log("ERROR", ticker, f"order placement failed: {exc}")
        raise HTTPException(500, str(exc))

    # Record pending position (phase 0 = waiting for fill)
    st["positions"][ticker] = {
        "entry_date":        date.today().isoformat(),
        "entry_price":       price,        # will be updated to actual fill price
        "shares":            shares,
        "cost":              cost,
        "buy_order_id":      buy_id,
        "stop_order_id":     None,
        "stop_type":         None,
        "stop_price":        None,
        "trading_days_held": 0,
        "phase":             0,
        "alert_fired_at":    req.alert_fired_at or now_et.isoformat(),
        "composite_score":   req.composite_score,
    }
    score_str = f" score={req.composite_score:.0f}" if req.composite_score is not None else ""
    _st_log("ENTERED", ticker,
            f"LIMIT BUY {shares}sh @ {lmt_px:.2f} (signal={price:.2f} "
            f"cost=${cost:,.0f} ord#{buy_id}){score_str}")
    _st_save_state()

    return {
        "status":      "ordered",
        "ticker":      ticker,
        "shares":      shares,
        "limit_price": lmt_px,
        "cost":        cost,
        "order_id":    buy_id,
    }


@app.get("/stock-trader/status")
def stock_trader_status():
    """All positions, config, recent decisions, and today's closed trades."""
    st  = state["stock_trader"]
    cfg = st["config"]
    closed = st.get("closed_today", [])

    capital_deployed = sum(
        p.get("shares", 0) * p.get("entry_price", 0)
        for p in st["positions"].values()
    )
    open_pnl = sum(
        p.get("live_pnl", 0) or 0
        for p in st["positions"].values()
        if p.get("phase", 0) >= 1
    )
    closed_pnl = sum(r.get("pnl", 0) for r in closed)
    today_pnl  = closed_pnl + open_pnl

    n            = len(closed)
    wins         = [r for r in closed if r.get("win")]
    losses       = [r for r in closed if not r.get("win")]
    gross_profit = sum(r["pnl"] for r in wins)
    gross_loss   = sum(r["pnl"] for r in losses)
    avg_ret_pct  = (sum(r.get("pnl_pct", 0) for r in closed) / n) if n else 0
    avg_win_pct  = (sum(r.get("pnl_pct", 0) for r in wins)   / len(wins))   if wins   else 0
    avg_loss_pct = (sum(r.get("pnl_pct", 0) for r in losses) / len(losses)) if losses else 0
    best  = max(closed, key=lambda r: r.get("pnl", 0), default=None)
    worst = min(closed, key=lambda r: r.get("pnl", 0), default=None)
    avg_days = (sum(r.get("days_held", 0) for r in closed) / n) if n else 0
    total_capital_traded = sum(
        r.get("entry_price", 0) * r.get("shares", 0) for r in closed
    ) + capital_deployed
    exit_breakdown = {}
    for r in closed:
        et = r.get("exit_type", "unknown")
        exit_breakdown[et] = exit_breakdown.get(et, 0) + 1

    eod_summary = {
        "total_trades":         n,
        "wins":                 len(wins),
        "losses":               len(losses),
        "win_rate":             round(len(wins) / n * 100, 1) if n else 0,
        "avg_return_pct":       round(avg_ret_pct, 3),
        "avg_win_pct":          round(avg_win_pct, 3),
        "avg_loss_pct":         round(avg_loss_pct, 3),
        "avg_days_held":        round(avg_days, 1),
        "gross_profit":         round(gross_profit, 2),
        "gross_loss":           round(gross_loss, 2),
        "profit_factor":        round(gross_profit / abs(gross_loss), 2) if gross_loss else None,
        "best_trade":           {"ticker": best["ticker"],  "pnl": best["pnl"],  "pnl_pct": best["pnl_pct"]}  if best  else None,
        "worst_trade":          {"ticker": worst["ticker"], "pnl": worst["pnl"], "pnl_pct": worst["pnl_pct"]} if worst else None,
        "total_capital_traded": round(total_capital_traded, 2),
        "open_pnl":             round(open_pnl, 2),
        "closed_pnl":           round(closed_pnl, 2),
        "exit_breakdown":       exit_breakdown,
        "hard_stop_pct":        cfg.get("hard_stop_pct", 7.0),
        "trail_pct":            cfg.get("trail_pct", 5.0),
        "position_size":        cfg.get("position_size", 2000),
    }

    return {
        "enabled":          st["enabled"],
        "config":           cfg,
        "positions":        st["positions"],
        "closed_today":     closed,
        "decisions":        st.get("decisions", [])[-50:],
        "eod_summary":      eod_summary,
        "rotation_log":     list(reversed(st.get("rotation_log", [])))[:10],
        "summary": {
            "open_positions":    len(st["positions"]),
            "capital_deployed":  round(capital_deployed, 2),
            "today_pnl":         round(today_pnl, 2),
            "today_trades":      n,
        },
    }


@app.post("/stock-trader/enable")
def stock_trader_enable(enabled: bool = True):
    st = state["stock_trader"]
    st["enabled"] = enabled
    _st_log("CONFIG", "—", f"{'enabled' if enabled else 'disabled'} by user")
    _st_save_state()
    return {"enabled": st["enabled"]}


@app.post("/stock-trader/config")
def stock_trader_config(req: StockConfigRequest):
    """Update any stock trader config keys."""
    st  = state["stock_trader"]
    cfg = st["config"]
    updates: dict = req.model_dump(exclude_none=True)
    cfg.update(updates)
    _st_log("CONFIG", "—", f"updated: {updates}")
    _st_save_state()
    return {"config": cfg}


@app.post("/stock-trader/close/{ticker}")
async def stock_trader_close(ticker: str):
    """Manually market-close a stock position immediately."""
    ticker = ticker.upper()
    st     = state["stock_trader"]
    pos    = st["positions"].get(ticker)
    if not pos:
        raise HTTPException(404, f"{ticker} not in open stock positions")

    ib = state.get("ib")
    if not ib or not ib.isConnected():
        raise HTTPException(503, "IBKR not connected")

    # Phase 0: cancel the pending buy — no stock was purchased, do NOT sell
    if pos.get("phase", 1) == 0:
        buy_oid = pos.get("buy_order_id")
        async def _do_cancel_buy(ib):
            open_trades = {t.order.orderId: t for t in ib.openTrades()}
            if buy_oid and buy_oid in open_trades:
                ib.cancelOrder(open_trades[buy_oid].order)
                await asyncio.sleep(0.5)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: _run_in_streaming_loop(_do_cancel_buy(ib), timeout=15)
        )
        st["positions"].pop(ticker, None)
        _st_log("MANUAL_CANCEL", ticker, f"pending buy ord#{buy_oid} cancelled — slot freed")
        _st_save_state()
        return {"status": "cancelled", "ticker": ticker, "order_id": buy_oid}

    stop_oid = pos.get("stop_order_id")

    async def _do_close(ib):
        open_trades_by_oid = {t.order.orderId: t for t in ib.openTrades()}
        # Cancel any open stop/trail order first
        if stop_oid and stop_oid in open_trades_by_oid:
            ib.cancelOrder(open_trades_by_oid[stop_oid].order)
            await asyncio.sleep(1)
        # Place market sell
        contract = Stock(ticker, "SMART", "USD")
        mkt_ord = Order()
        mkt_ord.orderType     = "MKT"
        mkt_ord.action        = "SELL"
        mkt_ord.totalQuantity = pos["shares"]
        mkt_ord.tif           = "DAY"
        trade = ib.placeOrder(contract, mkt_ord)
        await asyncio.sleep(0.5)
        return trade.order.orderId

    loop   = asyncio.get_event_loop()
    sell_id = await loop.run_in_executor(
        None,
        lambda: _run_in_streaming_loop(_do_close(ib), timeout=15),
    )
    pos["phase"]         = 3
    pos["stop_order_id"] = sell_id
    _st_log("MANUAL_CLOSE", ticker,
            f"MKT SELL {pos['shares']}sh placed (ord#{sell_id})")
    _st_save_state()

    return {"status": "closing", "ticker": ticker, "order_id": sell_id}


@app.get("/stock-trader/history")
def stock_trader_history(days: int = Query(30, ge=1, le=365)):
    """Closed stock breakout trades from trade_journal (STOCK_BREAKOUT strategy)."""
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).date().isoformat()
        con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
        con.row_factory = sqlite3.Row
        rows = con.execute("""
            SELECT opened_at, closed_at, ticker, qty, entry_price, exit_price,
                   pnl, pnl_pct, win, exit_reason, strategy_type
            FROM trade_journal
            WHERE strategy_type = 'STOCK_BREAKOUT'
              AND closed_at IS NOT NULL
              AND closed_at >= ?
            ORDER BY closed_at DESC
        """, (cutoff,)).fetchall()
        con.close()
        trades = [dict(r) for r in rows]
        total  = len(trades)
        wins   = sum(1 for t in trades if t.get("win"))
        total_pnl = sum(t.get("pnl", 0) or 0 for t in trades)
        return {
            "trades":    trades,
            "total":     total,
            "wins":      wins,
            "win_rate":  round(wins / total * 100, 1) if total else 0,
            "total_pnl": round(total_pnl, 2),
            "days":      days,
        }
    except Exception as exc:
        raise HTTPException(500, str(exc))


# ══════════════════════════════════════════════════════════════════════════════
# DAY TRADER — intraday breakout positions, force-close by 15:45 ET
# Same signals as Stock Trader; exits via profit target, hard stop, or EOD close.
# ══════════════════════════════════════════════════════════════════════════════

def _dt_log(action: str, ticker: str, detail: str) -> None:
    from zoneinfo import ZoneInfo
    entry = {
        "time":   datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M:%S ET"),
        "action": action,
        "ticker": ticker,
        "detail": detail,
    }
    dt = state["day_trader"]
    dt["decisions"].append(entry)
    dt["decisions"] = dt["decisions"][-200:]
    log.info("[DayTrader] %s %s: %s", action, ticker, detail)


def _dt_save_state() -> None:
    dt = state["day_trader"]
    try:
        with open(DT_STATE_PATH, "w") as f:
            json.dump({
                "enabled":      dt["enabled"],
                "config":       dt["config"],
                "positions":    dt["positions"],
                "closed_today": dt.get("closed_today", [])[-100:],
                "decisions":    dt.get("decisions", [])[-200:],
            }, f, indent=2, default=str)
    except Exception as e:
        log.warning("Day trader state save failed: %s", e)


def _dt_load_state() -> None:
    if not os.path.exists(DT_STATE_PATH):
        return
    try:
        with open(DT_STATE_PATH, "r") as f:
            saved = json.load(f)
        dt = state["day_trader"]
        if "config" in saved:
            dt["config"].update(saved["config"])
        if "enabled" in saved:
            dt["enabled"] = saved["enabled"]
        # Only restore same-day positions to avoid stale overnight entries
        today = date.today().isoformat()
        restored = {
            tk: pos for tk, pos in saved.get("positions", {}).items()
            if pos.get("entry_date") == today
        }
        # Clear stale live prices on load — monitor will repopulate them fresh
        for pos in restored.values():
            pos.pop("live_price", None)
            pos.pop("live_pnl", None)
        dt["positions"]    = restored
        dt["closed_today"] = [r for r in saved.get("closed_today", [])
                               if r.get("exit_date") == today]
        dt["decisions"]    = saved.get("decisions", [])
        log.info("Day trader state restored: %d open, %d closed today",
                 len(restored), len(dt["closed_today"]))
    except Exception as exc:
        log.warning("Day trader state load failed: %s", exc)


def _close_dt_position(ticker: str, pos: dict, exit_px: float,
                        exit_type: str, pnl: float) -> None:
    """Record a closed day trade to closed_today + trade journal."""
    dt = state["day_trader"]
    entry_px = pos.get("entry_price", exit_px)
    pnl_pct  = round((exit_px - entry_px) / entry_px * 100, 3) if entry_px else 0.0
    record = {
        "ticker":      ticker,
        "entry_date":  pos.get("entry_date"),
        "exit_date":   date.today().isoformat(),
        "entry_price": round(entry_px, 4),
        "exit_price":  round(exit_px, 4),
        "shares":      pos.get("shares", 0),
        "pnl":         round(pnl, 2),
        "pnl_pct":     pnl_pct,
        "exit_type":   exit_type,
        "win":         pnl > 0,
    }
    dt["closed_today"].append(record)
    dt["closed_today"] = dt["closed_today"][-100:]
    try:
        con = sqlite3.connect(JOURNAL_DB_PATH, check_same_thread=False)
        con.execute("""
            INSERT INTO trade_journal
                (opened_at, closed_at, ticker, action, qty,
                 entry_price, exit_price, pnl, pnl_pct, win,
                 exit_reason, strategy_type)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pos.get("entry_date"), date.today().isoformat(),
            ticker, "BUY", pos.get("shares", 0),
            round(entry_px, 4), round(exit_px, 4),
            round(pnl, 2), pnl_pct, 1 if pnl > 0 else 0,
            exit_type, "DAY_BREAKOUT",
        ))
        con.commit()
        con.close()
    except Exception as exc:
        log.warning("Day trade journal insert failed: %s", exc)
    _dt_log(exit_type.upper(), ticker,
            f"exit={exit_px:.2f} pnl={'+'if pnl>=0 else''}{pnl:.2f} ({pnl_pct:+.2f}%)")
    _dt_save_state()


async def _day_trader_monitor_coro(ib) -> None:
    """One monitor cycle: fill detection, profit target, stop, EOD force-close."""
    from zoneinfo import ZoneInfo
    dt  = state["day_trader"]
    cfg = dt["config"]

    if not dt["positions"]:
        return

    now_et = datetime.now(ZoneInfo("America/New_York"))

    # Parse force_close_time ("HH:MM") into today's datetime
    try:
        fc_h, fc_m = map(int, cfg["force_close_time"].split(":"))
        force_close_dt = now_et.replace(hour=fc_h, minute=fc_m, second=0, microsecond=0)
    except Exception:
        force_close_dt = now_et.replace(hour=15, minute=45, second=0, microsecond=0)

    open_trades_by_oid = {t.order.orderId: t for t in ib.openTrades()}
    fills_by_oid: dict = {}
    for f in ib.fills():
        oid = getattr(f.execution, "orderId", 0)
        if oid and oid > 0:
            fills_by_oid[oid] = f

    # Live prices via reqTickersAsync for all phase-1 positions.
    # We intentionally avoid portfolio.marketPrice — it can bleed in other lots
    # (e.g., auto-hedge SPY puts showing as SPY price) and is unreliable in paper.
    ticker_snapshot: dict = {}
    phase1_tickers = [t for t, p in dt["positions"].items() if p.get("phase", 0) == 1]
    if phase1_tickers:
        try:
            from ib_insync import Stock as IbStock
            contracts = [IbStock(t, "SMART", "USD") for t in phase1_tickers]
            tickers = await ib.reqTickersAsync(*contracts)
            for tk in tickers:
                sym = tk.contract.symbol
                mid = None
                if tk.ask > 0 and tk.bid > 0:
                    mid = (tk.bid + tk.ask) / 2
                elif tk.close > 0:
                    # prefer official EOD close over last (last can be option cross-contamination)
                    mid = tk.close
                elif tk.last > 1.0:
                    # only use last if it's plausibly a stock price (> $1)
                    mid = tk.last
                if mid:
                    ticker_snapshot[sym] = round(mid, 4)
        except Exception as ex:
            log.debug("Day trader reqTickers failed: %s", ex)

    to_remove: list[str] = []

    for ticker, pos in list(dt["positions"].items()):
        phase = pos.get("phase", 0)

        # Refresh live price / pnl — portfolio first, then ticker snapshot
        if ticker in ticker_snapshot:
            pos["live_price"] = ticker_snapshot[ticker]

        # Always compute P&L from our specific entry and shares — isolates this
        # day-trade lot from any other lots (hedges, long-term holds) in the account
        if pos.get("live_price") and pos.get("entry_price"):
            pos["live_pnl"] = round(
                (pos["live_price"] - pos["entry_price"]) * pos.get("shares", 0), 2
            )

        # ── Phase 0: waiting for buy fill ─────────────────────────────────
        if phase == 0:
            buy_oid = pos.get("buy_order_id")
            if not buy_oid:
                continue
            if buy_oid in open_trades_by_oid:
                # Cancel stale pending buys
                try:
                    alert_ts = pos.get("alert_fired_at", "")
                    at = datetime.fromisoformat(alert_ts) if alert_ts else None
                    if at is not None:
                        if at.tzinfo is None:
                            at = at.replace(tzinfo=ZoneInfo("America/New_York"))
                        age_min = (now_et - at).total_seconds() / 60
                    else:
                        age_min = 0
                except Exception:
                    age_min = 0
                if age_min > cfg.get("signal_freshness_min", 30):
                    ib.cancelOrder(open_trades_by_oid[buy_oid].order)
                    await asyncio.sleep(0.5)
                    _dt_log("BUY_CANCELLED", ticker,
                            f"stale after {age_min:.0f}min — cancelled, slot freed")
                    to_remove.append(ticker)
                continue
            fill = fills_by_oid.get(buy_oid)
            if fill:
                fill_px = round(float(fill.execution.avgPrice), 4)
                pos["entry_price"]  = fill_px
                pos["phase"]        = 1
                stop_px   = round(fill_px * (1 - cfg["hard_stop_pct"] / 100), 2)
                profit_px = round(fill_px * (1 + cfg["profit_target_pct"] / 100), 2)
                pos["stop_price"]   = stop_px
                pos["profit_price"] = profit_px
                contract = Stock(ticker, "SMART", "USD")
                oca_group = f"DT_{ticker}_{trade.order.orderId if 'trade' in dir() else buy_oid}"

                # Stop-loss: native STP order — fires immediately at IBKR
                stop_ord = Order()
                stop_ord.orderType     = "STP"
                stop_ord.action        = "SELL"
                stop_ord.totalQuantity = pos["shares"]
                stop_ord.auxPrice      = stop_px
                stop_ord.tif           = "DAY"
                stop_ord.outsideRth    = False
                stop_ord.ocaGroup      = oca_group
                stop_ord.ocaType       = 1   # cancel remaining on fill
                stp_trade = ib.placeOrder(contract, stop_ord)

                # Profit target: native LMT order — fires immediately at IBKR
                # No polling needed; IBKR cancels the STP when this fills
                lmt_ord = Order()
                lmt_ord.orderType     = "LMT"
                lmt_ord.action        = "SELL"
                lmt_ord.totalQuantity = pos["shares"]
                lmt_ord.lmtPrice      = profit_px
                lmt_ord.tif           = "DAY"
                lmt_ord.outsideRth    = False
                lmt_ord.ocaGroup      = oca_group
                lmt_ord.ocaType       = 1
                lmt_trade = ib.placeOrder(contract, lmt_ord)

                await asyncio.sleep(0.5)
                pos["stop_order_id"]   = stp_trade.order.orderId
                pos["target_order_id"] = lmt_trade.order.orderId
                _dt_log("FILLED", ticker,
                        f"fill={fill_px:.2f} x{pos['shares']}sh "
                        f"stop@{stop_px:.2f} (ord#{stp_trade.order.orderId}) "
                        f"target@{profit_px:.2f} (ord#{lmt_trade.order.orderId})")
                _dt_save_state()
            else:
                _dt_log("BUY_LAPSED", ticker, "buy limit not filled — removing")
                to_remove.append(ticker)
            continue

        # ── Phase 3: MKT sell in flight ────────────────────────────────────
        if phase == 3:
            sell_oid = pos.get("stop_order_id")
            if sell_oid and sell_oid not in open_trades_by_oid:
                fill    = fills_by_oid.get(sell_oid)
                exit_px = round(float(fill.execution.avgPrice), 4) if fill else pos.get("entry_price", 0)
                pnl     = round((exit_px - pos["entry_price"]) * pos["shares"], 2)
                exit_type = pos.get("pending_exit_type", "force_close")
                _close_dt_position(ticker, pos, exit_px, exit_type, pnl)
                to_remove.append(ticker)
            continue

        # ── Phase 1: active intraday position ─────────────────────────────
        stop_oid   = pos.get("stop_order_id")
        target_oid = pos.get("target_order_id")

        # Profit target LMT filled? (IBKR OCA cancels the STP automatically)
        if target_oid and target_oid not in open_trades_by_oid:
            fill    = fills_by_oid.get(target_oid)
            exit_px = round(float(fill.execution.avgPrice), 4) if fill else pos.get("profit_price", pos["entry_price"])
            pnl     = round((exit_px - pos["entry_price"]) * pos["shares"], 2)
            _close_dt_position(ticker, pos, exit_px, "profit_target", pnl)
            to_remove.append(ticker)
            continue

        # Stop loss STP filled? (IBKR OCA cancels the LMT automatically)
        if stop_oid and stop_oid not in open_trades_by_oid:
            fill    = fills_by_oid.get(stop_oid)
            exit_px = round(float(fill.execution.avgPrice), 4) if fill else pos.get("stop_price", pos["entry_price"])
            pnl     = round((exit_px - pos["entry_price"]) * pos["shares"], 2)
            _close_dt_position(ticker, pos, exit_px, "hard_stop", pnl)
            to_remove.append(ticker)
            continue

        # Fallback poll for positions that pre-date the OCA bracket (no target_order_id).
        # New positions exit via the native LMT order above; this catches legacy ones.
        if not target_oid:
            live_px = pos.get("live_price") or 0
            if live_px and live_px >= pos.get("profit_price", float("inf")):
                for oid in (stop_oid,):
                    if oid and oid in open_trades_by_oid:
                        ib.cancelOrder(open_trades_by_oid[oid].order)
                await asyncio.sleep(0.5)
                contract = Stock(ticker, "SMART", "USD")
                mkt_ord = Order()
                mkt_ord.orderType = "MKT"; mkt_ord.action = "SELL"
                mkt_ord.totalQuantity = pos["shares"]; mkt_ord.tif = "DAY"
                t = ib.placeOrder(contract, mkt_ord)
                await asyncio.sleep(0.5)
                pos["phase"] = 3; pos["stop_order_id"] = t.order.orderId
                pos["pending_exit_type"] = "profit_target"
                _dt_log("PROFIT_TARGET", ticker,
                        f"poll: live={live_px:.2f} >= target={pos['profit_price']:.2f} (ord#{t.order.orderId})")
                _dt_save_state()
                continue

        # Force-close time reached?
        if now_et >= force_close_dt:
            # Cancel both OCA legs before placing MKT override
            for oid in (stop_oid, target_oid):
                if oid and oid in open_trades_by_oid:
                    ib.cancelOrder(open_trades_by_oid[oid].order)
            await asyncio.sleep(0.5)
            contract = Stock(ticker, "SMART", "USD")
            mkt_ord = Order()
            mkt_ord.orderType     = "MKT"
            mkt_ord.action        = "SELL"
            mkt_ord.totalQuantity = pos["shares"]
            mkt_ord.tif           = "DAY"
            trade = ib.placeOrder(contract, mkt_ord)
            await asyncio.sleep(0.5)
            pos["phase"]             = 3
            pos["stop_order_id"]     = trade.order.orderId
            pos["pending_exit_type"] = "force_close"
            _dt_log("FORCE_CLOSE", ticker,
                    f"EOD force-close @ {cfg['force_close_time']} ET, MKT SELL (ord#{trade.order.orderId})")
            _dt_save_state()

    for ticker in to_remove:
        dt["positions"].pop(ticker, None)
    if to_remove:
        _dt_save_state()


async def _day_trader_monitor_loop() -> None:
    """Background task: monitor day trader positions every 30 seconds."""
    await asyncio.sleep(30)   # let server finish starting
    while True:
        await asyncio.sleep(30)
        dt = state["day_trader"]
        if not dt["enabled"] or not dt["positions"]:
            continue
        if not state.get("connected") or not state.get("ib"):
            continue
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        # Only run during market hours + 15 min buffer after close for fill detection
        if now_et.weekday() >= 5:
            continue
        mkt_open  = now_et.replace(hour=9,  minute=25, second=0, microsecond=0)
        mkt_close = now_et.replace(hour=16, minute=15, second=0, microsecond=0)
        if not (mkt_open <= now_et <= mkt_close):
            continue
        ib = state["ib"]
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: _run_in_streaming_loop(_day_trader_monitor_coro(ib), timeout=25),
            )
        except Exception as exc:
            log.warning("Day trader monitor error: %s", exc)


# ── Day Trader request models ──────────────────────────────────────────────────

class DayConfigRequest(BaseModel):
    position_size:        Optional[float] = None
    max_positions:        Optional[int]   = None
    hard_stop_pct:        Optional[float] = None
    profit_target_pct:    Optional[float] = None
    force_close_time:     Optional[str]   = None   # "HH:MM"
    signal_freshness_min: Optional[int]   = None
    limit_buffer_pct:     Optional[float] = None
    daily_profit_target:  Optional[float] = None
    expected_return_pct:  Optional[float] = None
    win_rate_est:         Optional[float] = None


# ── Day Trader endpoints ───────────────────────────────────────────────────────

@app.post("/day-trader/signal")
async def day_trader_signal(req: StockSignalRequest):
    """Called by breakout_scanner on BREAKOUT — same payload as /stock-trader/signal."""
    from zoneinfo import ZoneInfo
    dt  = state["day_trader"]
    cfg = dt["config"]

    if not dt["enabled"]:
        return {"status": "skipped", "reason": "disabled"}

    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return {"status": "skipped", "reason": "weekend"}
    mkt_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    # Stop new entries 30 min before force-close so position has time to work
    try:
        fc_h, fc_m = map(int, cfg["force_close_time"].split(":"))
        entry_cutoff = now_et.replace(hour=fc_h, minute=max(0, fc_m - 30),
                                      second=0, microsecond=0)
    except Exception:
        entry_cutoff = now_et.replace(hour=15, minute=15, second=0, microsecond=0)
    if not (mkt_open <= now_et <= entry_cutoff):
        return {"status": "skipped", "reason": "outside_hours"}

    if req.alert_fired_at:
        try:
            fired_at = datetime.fromisoformat(req.alert_fired_at)
            if fired_at.tzinfo is None:
                fired_at = fired_at.replace(tzinfo=ZoneInfo("America/New_York"))
            age_min = (now_et - fired_at).total_seconds() / 60
            if age_min > cfg["signal_freshness_min"]:
                return {"status": "skipped", "reason": "stale_signal", "age_min": round(age_min, 1)}
        except Exception:
            pass

    ticker = req.ticker.upper()
    if ticker in dt["positions"]:
        _dt_log("SKIPPED", ticker, "already have open position")
        return {"status": "skipped", "reason": "already_open"}
    if len(dt["positions"]) >= cfg["max_positions"]:
        _dt_log("SKIPPED", ticker,
                f"at capacity ({len(dt['positions'])}/{cfg['max_positions']} positions)")
        return {"status": "skipped", "reason": "at_capacity"}

    ib = state.get("ib")
    if not ib or not ib.isConnected():
        raise HTTPException(503, "IBKR not connected")

    price  = req.price
    shares = max(1, int(cfg["position_size"] / price))
    lmt_px = round(price * (1 + cfg["limit_buffer_pct"] / 100), 2)
    cost   = round(shares * price, 2)

    async def _do_buy(ib):
        contract = Stock(ticker, "SMART", "USD")
        await ib.qualifyContractsAsync(contract)
        if not contract.conId:
            raise ValueError(f"Cannot qualify {ticker}")
        buy_ord = LimitOrder("BUY", shares, lmt_px)
        buy_ord.tif = "DAY"
        trade = ib.placeOrder(contract, buy_ord)
        await asyncio.sleep(0.5)
        return trade.order.orderId

    try:
        loop   = asyncio.get_event_loop()
        buy_id = await loop.run_in_executor(
            None,
            lambda: _run_in_streaming_loop(_do_buy(ib), timeout=15),
        )
    except (ValueError, TimeoutError, RuntimeError) as exc:
        _dt_log("ERROR", ticker, f"order placement failed: {exc}")
        raise HTTPException(500, str(exc))

    profit_px = round(price * (1 + cfg["profit_target_pct"] / 100), 2)
    stop_px   = round(price * (1 - cfg["hard_stop_pct"] / 100), 2)

    dt["positions"][ticker] = {
        "entry_date":        date.today().isoformat(),
        "entry_price":       price,
        "shares":            shares,
        "cost":              cost,
        "buy_order_id":      buy_id,
        "stop_order_id":     None,
        "stop_price":        stop_px,
        "profit_price":      profit_px,
        "phase":             0,
        "alert_fired_at":    req.alert_fired_at or now_et.isoformat(),
        "live_price":        None,
        "live_pnl":          None,
    }
    _dt_log("ENTERED", ticker,
            f"LIMIT BUY {shares}sh @ {lmt_px:.2f} "
            f"(signal={price:.2f} cost=${cost:,.0f} ord#{buy_id}) "
            f"target={profit_px:.2f} stop={stop_px:.2f}")
    _dt_save_state()
    return {"status": "ordered", "ticker": ticker, "shares": shares,
            "limit_price": lmt_px, "cost": cost, "order_id": buy_id}


@app.get("/day-trader/status")
def day_trader_status():
    dt  = state["day_trader"]
    cfg = dt["config"]
    open_positions = dt["positions"]
    closed         = dt.get("closed_today", [])

    capital_deployed = sum(
        p.get("shares", 0) * p.get("entry_price", 0)
        for p in open_positions.values()
    )
    closed_pnl = sum(r.get("pnl", 0) for r in closed)
    open_pnl   = sum(
        p.get("live_pnl", 0) or 0
        for p in open_positions.values()
        if p.get("phase", 0) == 1
    )
    today_pnl = closed_pnl + open_pnl

    # EOD stats from closed trades
    n          = len(closed)
    wins       = [r for r in closed if r.get("win")]
    losses     = [r for r in closed if not r.get("win")]
    gross_profit = sum(r["pnl"] for r in wins)
    gross_loss   = sum(r["pnl"] for r in losses)
    avg_ret_pct  = (sum(r.get("pnl_pct", 0) for r in closed) / n) if n else 0
    avg_win_pct  = (sum(r.get("pnl_pct", 0) for r in wins)   / len(wins))   if wins   else 0
    avg_loss_pct = (sum(r.get("pnl_pct", 0) for r in losses) / len(losses)) if losses else 0
    best  = max(closed, key=lambda r: r.get("pnl", 0), default=None)
    worst = min(closed, key=lambda r: r.get("pnl", 0), default=None)
    total_capital_traded = sum(
        r.get("entry_price", 0) * r.get("shares", 0) for r in closed
    ) + capital_deployed
    exit_breakdown = {}
    for r in closed:
        et = r.get("exit_type", "unknown")
        exit_breakdown[et] = exit_breakdown.get(et, 0) + 1

    eod_summary = {
        "total_trades":          n,
        "wins":                  len(wins),
        "losses":                len(losses),
        "win_rate":              round(len(wins) / n * 100, 1) if n else 0,
        "avg_return_pct":        round(avg_ret_pct, 3),
        "avg_win_pct":           round(avg_win_pct, 3),
        "avg_loss_pct":          round(avg_loss_pct, 3),
        "gross_profit":          round(gross_profit, 2),
        "gross_loss":            round(gross_loss, 2),
        "profit_factor":         round(gross_profit / abs(gross_loss), 2) if gross_loss else None,
        "best_trade":            {"ticker": best["ticker"],  "pnl": best["pnl"],  "pnl_pct": best["pnl_pct"]}  if best  else None,
        "worst_trade":           {"ticker": worst["ticker"], "pnl": worst["pnl"], "pnl_pct": worst["pnl_pct"]} if worst else None,
        "total_capital_traded":  round(total_capital_traded, 2),
        "exit_breakdown":        exit_breakdown,
        "profit_target_pct":     cfg["profit_target_pct"],
        "hard_stop_pct":         cfg["hard_stop_pct"],
        "daily_profit_target":   cfg["daily_profit_target"],
        "goal_achieved":         today_pnl >= cfg["daily_profit_target"],
    }

    return {
        "enabled":      dt["enabled"],
        "config":       cfg,
        "positions":    open_positions,
        "closed_today": closed,
        "decisions":    dt.get("decisions", [])[-50:],
        "eod_summary":  eod_summary,
        "summary": {
            "open_positions":   len(open_positions),
            "capital_deployed": round(capital_deployed, 2),
            "closed_pnl":       round(closed_pnl, 2),
            "open_pnl":         round(open_pnl, 2),
            "today_pnl":        round(today_pnl, 2),
            "today_trades":     n,
            "goal_pct":         round(today_pnl / cfg["daily_profit_target"] * 100, 1)
                                if cfg["daily_profit_target"] > 0 else 0,
        },
    }


@app.post("/day-trader/enable")
def day_trader_enable(enabled: bool = True):
    dt = state["day_trader"]
    dt["enabled"] = enabled
    _dt_log("CONFIG", "—", f"{'enabled' if enabled else 'disabled'} by user")
    _dt_save_state()
    return {"enabled": dt["enabled"]}


@app.post("/day-trader/config")
def day_trader_config(req: DayConfigRequest):
    dt  = state["day_trader"]
    cfg = dt["config"]
    updates: dict = req.model_dump(exclude_none=True)
    cfg.update(updates)

    # Retroactively reprice open positions when profit_target_pct changes
    if "profit_target_pct" in updates:
        new_pct = updates["profit_target_pct"]
        repriced = []
        for ticker, pos in dt["positions"].items():
            if pos.get("phase", 0) in (0, 1) and pos.get("entry_price"):
                old_target = pos.get("profit_price")
                pos["profit_price"] = round(pos["entry_price"] * (1 + new_pct / 100), 2)
                repriced.append(f"{ticker}: ${old_target}->${pos['profit_price']}")
        if repriced:
            _dt_log("REPRICE", "—", f"profit_target→{new_pct}%: {', '.join(repriced)}")

    # Log max_positions change prominently so operator knows new capacity
    if "max_positions" in updates:
        open_n = len(dt["positions"])
        _dt_log("CONFIG", "—",
                f"max_positions→{updates['max_positions']} (currently {open_n} open)")

    _dt_log("CONFIG", "—", f"updated: {updates}")
    _dt_save_state()
    return {"config": cfg}


@app.post("/day-trader/close/{ticker}")
async def day_trader_close(ticker: str):
    """Manually close a day trader position immediately."""
    ticker = ticker.upper()
    dt  = state["day_trader"]
    pos = dt["positions"].get(ticker)
    if not pos:
        raise HTTPException(404, f"{ticker} not in open day trader positions")
    ib = state.get("ib")
    if not ib or not ib.isConnected():
        raise HTTPException(503, "IBKR not connected")

    if pos.get("phase", 0) == 0:
        buy_oid = pos.get("buy_order_id")
        async def _cancel_buy(ib):
            ot = {t.order.orderId: t for t in ib.openTrades()}
            if buy_oid and buy_oid in ot:
                ib.cancelOrder(ot[buy_oid].order)
                await asyncio.sleep(0.5)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: _run_in_streaming_loop(_cancel_buy(ib), timeout=15))
        dt["positions"].pop(ticker, None)
        _dt_log("MANUAL_CANCEL", ticker, f"pending buy ord#{buy_oid} cancelled")
        _dt_save_state()
        return {"status": "cancelled", "ticker": ticker}

    stop_oid   = pos.get("stop_order_id")
    target_oid = pos.get("target_order_id")
    async def _do_close(ib):
        ot = {t.order.orderId: t for t in ib.openTrades()}
        for oid in (stop_oid, target_oid):
            if oid and oid in ot:
                ib.cancelOrder(ot[oid].order)
        await asyncio.sleep(1)
        contract = Stock(ticker, "SMART", "USD")
        mkt_ord = Order()
        mkt_ord.orderType     = "MKT"
        mkt_ord.action        = "SELL"
        mkt_ord.totalQuantity = pos["shares"]
        mkt_ord.tif           = "DAY"
        trade = ib.placeOrder(contract, mkt_ord)
        await asyncio.sleep(0.5)
        return trade.order.orderId

    loop    = asyncio.get_event_loop()
    sell_id = await loop.run_in_executor(None, lambda: _run_in_streaming_loop(_do_close(ib), timeout=15))
    pos["phase"]             = 3
    pos["stop_order_id"]     = sell_id
    pos["pending_exit_type"] = "manual_close"
    _dt_log("MANUAL_CLOSE", ticker, f"MKT SELL {pos['shares']}sh (ord#{sell_id})")
    _dt_save_state()
    return {"status": "closing", "ticker": ticker, "order_id": sell_id}


@app.get("/day-trader/goal")
def day_trader_goal():
    """Calculate capital required to hit the daily profit target."""
    cfg      = state["day_trader"]["config"]
    target   = cfg["daily_profit_target"]
    win_rate = cfg["win_rate_est"]
    exp_ret  = cfg["expected_return_pct"]
    pos_size = cfg["position_size"]
    ev_per   = pos_size * win_rate * (exp_ret / 100)
    if ev_per <= 0:
        return {"error": "Invalid win_rate_est or expected_return_pct"}
    req_pos     = int(np.ceil(target / ev_per))
    req_capital = round(req_pos * pos_size, 2)
    today_pnl   = sum(r.get("pnl", 0) for r in state["day_trader"].get("closed_today", []))
    remaining   = max(0.0, target - today_pnl)
    req_pos_remaining = int(np.ceil(remaining / ev_per)) if remaining > 0 else 0
    return {
        "daily_profit_target":   target,
        "win_rate_est":          win_rate,
        "expected_return_pct":   exp_ret,
        "position_size":         pos_size,
        "ev_per_position":       round(ev_per, 2),
        "required_positions":    req_pos,
        "required_capital":      req_capital,
        "today_pnl":             round(today_pnl, 2),
        "remaining_target":      round(remaining, 2),
        "remaining_positions":   req_pos_remaining,
        "current_max_positions": cfg["max_positions"],
        "positions_gap":         max(0, req_pos - cfg["max_positions"]),
    }


# ── SPX 0DTE Trader ──────────────────────────────────────────────────────────

def _spx_log(action: str, detail: str, spread_id: str = "—") -> None:
    from zoneinfo import ZoneInfo
    t = datetime.now(ZoneInfo("America/New_York")).strftime("%H:%M:%S ET")
    entry = {"time": t, "action": action, "spread_id": spread_id, "detail": detail}
    sx = state["spx_0dte"]
    sx["decisions"].append(entry)
    if len(sx["decisions"]) > 200:
        sx["decisions"] = sx["decisions"][-200:]
    log.info("SPX0DTE [%s] %s — %s", action, spread_id, detail)


def _spx_save_state() -> None:
    sx = state["spx_0dte"]
    try:
        with open(SPX_STATE_PATH, "w") as f:
            json.dump({
                "enabled":        sx["enabled"],
                "config":         sx["config"],
                "spreads":        sx["spreads"],
                "closed_today":   sx["closed_today"],
                "decisions":      sx["decisions"][-50:],
                "attempts_today": sx["attempts_today"],
                "today_pnl":      sx["today_pnl"],
                "last_stop_time": sx.get("last_stop_time"),
                "date":           date.today().isoformat(),
            }, f, default=str)
    except Exception as e:
        log.warning("SPX 0DTE state save failed: %s", e)


def _spx_load_state() -> None:
    if not os.path.exists(SPX_STATE_PATH):
        return
    try:
        with open(SPX_STATE_PATH, "r") as f:
            saved = json.load(f)
        sx    = state["spx_0dte"]
        today = date.today().isoformat()
        if "config" in saved:
            sx["config"].update(saved["config"])
        sx["enabled"] = saved.get("enabled", False)
        sx["spreads"]  = {
            k: v for k, v in saved.get("spreads", {}).items()
            if v.get("date") == today
        }
        sx["closed_today"]   = [r for r in saved.get("closed_today", []) if r.get("date") == today]
        sx["decisions"]      = saved.get("decisions", [])
        same_day             = saved.get("date") == today
        sx["attempts_today"] = saved.get("attempts_today", 0) if same_day else 0
        sx["today_pnl"]      = saved.get("today_pnl", 0.0)    if same_day else 0.0
        sx["last_stop_time"] = saved.get("last_stop_time")
        log.info("SPX 0DTE state restored: %d open spreads, $%.2f P&L today",
                 len(sx["spreads"]), sx["today_pnl"])
    except Exception as e:
        log.warning("SPX 0DTE state load failed: %s", e)


async def _spx_get_spot(ib) -> float:
    """SPX spot: SPY bars × 10 (primary), live reqMktData for SPX index (fallback)."""
    spy_bars = state["bars"].get("SPY", [])
    if spy_bars:
        return round(float(spy_bars[-1]["close"]) * 10, 2)
    try:
        from ib_insync import Index
        spx_c = Index("SPX", "CBOE")
        await ib.qualifyContractsAsync(spx_c)
        td = ib.reqMktData(spx_c, "", False, False)
        await asyncio.sleep(2)
        px = float(td.last or td.close or 0)
        ib.cancelMktData(spx_c)
        if px > 0:
            return px
    except Exception as e:
        log.warning("SPX 0DTE: spot unavailable via IB: %s", e)
    return 0.0


def _spx_round_strike(spot: float, offset_pct: float, right: str, increment: int = 5) -> float:
    raw = spot * (1 - offset_pct / 100) if right == "P" else spot * (1 + offset_pct / 100)
    return float(round(raw / increment) * increment)


def _spx_safe_px(v) -> float:
    """Return a positive finite float from an ib_insync tick value, else 0.0."""
    import math
    try:
        f = float(v)
        return f if (f > 0 and not math.isnan(f) and not math.isinf(f)) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _spx_mid(td) -> float:
    """Best-effort mid price from a Ticker: mid → last → 0."""
    b = _spx_safe_px(td.bid)
    a = _spx_safe_px(td.ask)
    if b > 0 and a > 0:
        return (b + a) / 2
    # Fall back to last only if it's clearly reasonable (not stale zero)
    last = _spx_safe_px(td.last)
    return last


async def _spx_quote_vertical(ib, expiry: str, right: str,
                               short_k: float, long_k: float):
    """
    Qualify two SPX option legs and return (net_credit, short_conid, long_conid).
    Uses mid-price for both legs so stale bids/asks don't invert the spread.
    net_credit = short_mid - long_mid > 0 means we collect premium.
    """
    from ib_insync import Option as IbOpt
    # tradingClass="SPXW" = PM-settled weekly (0DTE Mon/Wed/Fri). Required by IBKR.
    sc = IbOpt("SPX", expiry, short_k, right, "SMART", "100", "USD")
    lc = IbOpt("SPX", expiry, long_k,  right, "SMART", "100", "USD")
    sc.tradingClass = "SPXW"
    lc.tradingClass = "SPXW"
    await ib.qualifyContractsAsync(sc, lc)
    if not sc.conId or not lc.conId:
        raise ValueError(f"Cannot qualify SPX {right} {short_k}/{long_k} exp {expiry} (SPXW)")

    td_s = ib.reqMktData(sc, "100,101,106", False, False)
    td_l = ib.reqMktData(lc, "100,101,106", False, False)
    await asyncio.sleep(5)          # paper needs extra time for quotes to populate

    s_mid = _spx_mid(td_s)
    l_mid = _spx_mid(td_l)
    ib.cancelMktData(sc)
    ib.cancelMktData(lc)

    if s_mid <= 0 or l_mid <= 0:
        raise ValueError(
            f"No market for SPX {right} {short_k}/{long_k}: "
            f"short_mid={s_mid:.3f} long_mid={l_mid:.3f}"
        )
    if s_mid <= l_mid:
        raise ValueError(
            f"SPX {right} spread inverted: short@{short_k}={s_mid:.3f} >= long@{long_k}={l_mid:.3f} "
            f"(strikes too far OTM or bad quotes)"
        )

    net = round(s_mid - l_mid, 2)
    return net, sc.conId, lc.conId


async def _spx_place_bag(ib, expiry: str, right: str,
                          short_k: float, long_k: float,
                          short_conid: int, long_conid: int,
                          qty: int, credit: float) -> int:
    """Place a vertical spread as a CBOE BAG (combo) order. Returns orderId."""
    from ib_insync import Contract as IbCont, ComboLeg
    bag = IbCont()
    bag.symbol   = "SPX"
    bag.secType  = "BAG"
    bag.exchange = "CBOE"
    bag.currency = "USD"
    l1 = ComboLeg(); l1.conId = short_conid; l1.ratio = 1; l1.action = "SELL"; l1.exchange = "CBOE"
    l2 = ComboLeg(); l2.conId = long_conid;  l2.ratio = 1; l2.action = "BUY";  l2.exchange = "CBOE"
    bag.comboLegs = [l1, l2]
    ord_ = LimitOrder("SELL", qty, max(0.01, round(credit, 2)))
    ord_.tif = "DAY"
    trade = ib.placeOrder(bag, ord_)
    await asyncio.sleep(0.5)
    return trade.order.orderId


async def _spx_close_spread(ib, spread_id: str, exit_type: str) -> None:
    """Market-close all legs of an open IC and record result."""
    from ib_insync import Option as IbOpt
    sx = state["spx_0dte"]
    sp = sx["spreads"].get(spread_id)
    if not sp:
        return
    sp["phase"] = 2
    qty    = sp["qty"]
    expiry = sp["expiry"]

    # Cancel pending spread orders
    open_oids = {t.order.orderId: t for t in ib.openTrades()}
    for key in ("put_order_id", "call_order_id"):
        oid = sp.get(key)
        if oid and oid in open_oids:
            ib.cancelOrder(open_oids[oid].order)
    await asyncio.sleep(1)

    # Leg out: cover shorts, sell longs — only for legs that were actually entered
    legs = []
    if sp.get("put_conids") and sp.get("put_short_k") is not None:
        legs += [
            ("P", sp["put_short_k"], sp["put_conids"][0], "BUY"),
            ("P", sp["put_long_k"],  sp["put_conids"][1], "SELL"),
        ]
    if sp.get("call_conids") and sp.get("call_short_k") is not None:
        legs += [
            ("C", sp["call_short_k"], sp["call_conids"][0], "BUY"),
            ("C", sp["call_long_k"],  sp["call_conids"][1], "SELL"),
        ]
    for right, strike, conid, action in legs:
        try:
            c = IbOpt("SPX", expiry, strike, right, "SMART", "100", "USD")
            c.tradingClass = "SPXW"
            c.conId = conid
            mkt = Order()
            mkt.orderType = "MKT"; mkt.action = action
            mkt.totalQuantity = qty; mkt.tif = "DAY"
            ib.placeOrder(c, mkt)
        except Exception as ex:
            log.warning("SPX 0DTE close leg %s%s failed: %s", right, strike, ex)
    await asyncio.sleep(1)

    pnl = round(sp.get("live_pnl") or 0, 2)
    sx["today_pnl"] = round(sx.get("today_pnl", 0) + pnl, 2)
    credit = sp["total_credit"]
    strategy = sp.get("strategy", "iron_condor")
    put_str  = (f"{int(sp['put_short_k'])}/{int(sp['put_long_k'])}P"
                if sp.get("put_short_k") is not None else "—")
    call_str = (f"{int(sp['call_short_k'])}/{int(sp['call_long_k'])}C"
                if sp.get("call_short_k") is not None else "—")
    sx["closed_today"].append({
        "spread_id":    spread_id,
        "strategy":     strategy,
        "date":         sp["date"],
        "placed_at":    sp["placed_at"],
        "put_strikes":  put_str,
        "call_strikes": call_str,
        "qty":          qty,
        "total_credit": credit,
        "pnl":          pnl,
        "pnl_pct":      round(pnl / credit * 100, 1) if credit else 0,
        "exit_type":    exit_type,
    })
    sx["spreads"].pop(spread_id, None)
    _spx_log("CLOSED", f"exit={exit_type}  P&L=${pnl:.2f}  day_total=${sx['today_pnl']:.2f}", spread_id)
    _spx_save_state()


async def _spx_entry_coro(ib) -> None:
    """Check all conditions and place an Iron Condor if appropriate."""
    from zoneinfo import ZoneInfo
    ET  = ZoneInfo("America/New_York")
    sx  = state["spx_0dte"]
    cfg = sx["config"]
    now = datetime.now(ET)

    def _t(hhmm):
        h, m = map(int, hhmm.split(":"))
        return now.replace(hour=h, minute=m, second=0, microsecond=0)

    if not (_t(cfg["entry_start_time"]) <= now <= _t(cfg["entry_cutoff_time"])):
        return
    if sx["today_pnl"] >= cfg["daily_profit_target"]:
        return
    if sx["attempts_today"] >= cfg["max_attempts"]:
        return
    if sx["spreads"]:
        return
    if sx.get("last_stop_time"):
        last_stop = datetime.fromisoformat(sx["last_stop_time"])
        if (now - last_stop).total_seconds() < 30 * 60:
            return

    spot = await _spx_get_spot(ib)
    if spot <= 0:
        _spx_log("SKIP", "SPX spot price unavailable")
        return

    expiry  = now.strftime("%Y%m%d")
    otm_pct = float(cfg["otm_pct"])
    width   = int(cfg["spread_width"])

    put_short_k  = _spx_round_strike(spot, otm_pct, "P")
    put_long_k   = put_short_k - width
    call_short_k = _spx_round_strike(spot, otm_pct, "C")
    call_long_k  = call_short_k + width

    try:
        p_credit, p_s_cid, p_l_cid = await _spx_quote_vertical(ib, expiry, "P", put_short_k, put_long_k)
        c_credit, c_s_cid, c_l_cid = await _spx_quote_vertical(ib, expiry, "C", call_short_k, call_long_k)
    except Exception as e:
        _spx_log("SKIP", f"Chain unavailable: {e}")
        return

    min_cred = float(cfg.get("min_credit_per_spread", 0.20))

    # Allow one-sided entry: skip a leg only if its credit is below minimum.
    # Entering a 25-pt spread for $0.10 risks $2,490 to make $10 — terrible R/R.
    use_put  = p_credit >= min_cred
    use_call = c_credit >= min_cred

    if not use_put and not use_call:
        _spx_log("SKIP",
                 f"Both legs thin: P={p_credit:.2f} C={c_credit:.2f} min={min_cred:.2f}")
        return

    strategy      = "iron_condor" if (use_put and use_call) else ("call_spread" if use_call else "put_spread")
    active_credit = (p_credit if use_put else 0.0) + (c_credit if use_call else 0.0)

    if not use_put:
        _spx_log("INFO", f"Put leg thin ({p_credit:.2f}) — entering call spread only (C={c_credit:.2f})")
    elif not use_call:
        _spx_log("INFO", f"Call leg thin ({c_credit:.2f}) — entering put spread only (P={p_credit:.2f})")

    profit_pct    = float(cfg["profit_pct"]) / 100
    ev_per_spread = active_credit * 100 * profit_pct
    remaining     = max(0.0, cfg["daily_profit_target"] - sx["today_pnl"])
    max_by_margin = max(1, int(cfg["max_margin"] / (width * 100)))
    qty           = min(max_by_margin, max(1, int(np.ceil(remaining / ev_per_spread))))

    _spx_log("ENTRY",
             f"SPX {spot:.0f}  strategy={strategy} | "
             + (f"P {int(put_short_k)}/{int(put_long_k)} cr={p_credit:.2f} | " if use_put else "")
             + (f"C {int(call_short_k)}/{int(call_long_k)} cr={c_credit:.2f} | " if use_call else "")
             + f"qty={qty}  total_cr=${qty*active_credit*100:.0f}")

    p_oid = c_oid = None
    try:
        if use_put:
            p_oid = await _spx_place_bag(ib, expiry, "P",
                                          put_short_k, put_long_k, p_s_cid, p_l_cid, qty, p_credit)
        if use_call:
            c_oid = await _spx_place_bag(ib, expiry, "C",
                                          call_short_k, call_long_k, c_s_cid, c_l_cid, qty, c_credit)
    except Exception as e:
        _spx_log("ERROR", f"Order placement failed: {e}")
        return

    total_credit_dollar = round(qty * active_credit * 100, 2)
    spread_id = f"{strategy[:2].upper()}_{now.strftime('%H%M')}"
    sx["spreads"][spread_id] = {
        "spread_id":     spread_id,
        "strategy":      strategy,
        "date":          date.today().isoformat(),
        "expiry":        expiry,
        "spot_at_entry": spot,
        "qty":           qty,
        "put_short_k":   put_short_k  if use_put  else None,
        "put_long_k":    put_long_k   if use_put  else None,
        "call_short_k":  call_short_k if use_call else None,
        "call_long_k":   call_long_k  if use_call else None,
        "put_conids":    [p_s_cid, p_l_cid] if use_put  else [],
        "call_conids":   [c_s_cid, c_l_cid] if use_call else [],
        "put_order_id":  p_oid,
        "call_order_id": c_oid,
        "put_credit":    p_credit  if use_put  else 0.0,
        "call_credit":   c_credit  if use_call else 0.0,
        "total_credit":  total_credit_dollar,
        "profit_target": round(total_credit_dollar * profit_pct, 2),
        "max_loss":      round(qty * (width - active_credit) * 100, 2),
        "phase":         1,
        "live_pnl":      None,
        "placed_at":     now.strftime("%H:%M ET"),
    }
    sx["attempts_today"] += 1
    _spx_save_state()


async def _spx_monitor_coro(ib) -> None:
    """One cycle: compute live P&L by re-quoting spreads (portfolio.unrealizedPNL is
    unreliable in paper trading for SPX index options — IBKR marks at 0 if no subscription)."""
    from zoneinfo import ZoneInfo
    from ib_insync import Option as IbOpt
    ET  = ZoneInfo("America/New_York")
    sx  = state["spx_0dte"]
    cfg = sx["config"]
    now = datetime.now(ET)

    def _t(hhmm):
        h, m = map(int, hhmm.split(":"))
        return now.replace(hour=h, minute=m, second=0, microsecond=0)

    for sid, sp in list(sx["spreads"].items()):
        if sp.get("phase", 1) != 1:
            continue

        qty    = sp["qty"]
        expiry = sp["expiry"]
        live_pnl = 0.0

        # Re-quote each live leg: short leg value lost = profit for us
        legs_to_quote = []
        if sp.get("put_conids") and sp.get("put_short_k") is not None:
            legs_to_quote.append(("P", sp["put_short_k"], sp["put_long_k"],
                                  sp["put_credit"], sp["put_conids"]))
        if sp.get("call_conids") and sp.get("call_short_k") is not None:
            legs_to_quote.append(("C", sp["call_short_k"], sp["call_long_k"],
                                  sp["call_credit"], sp["call_conids"]))

        for right, sk, lk, entry_credit, conids in legs_to_quote:
            try:
                sc = IbOpt("SPX", expiry, sk, right, "SMART", "100", "USD")
                lc = IbOpt("SPX", expiry, lk, right, "SMART", "100", "USD")
                sc.tradingClass = "SPXW"; sc.conId = conids[0]
                lc.tradingClass = "SPXW"; lc.conId = conids[1]
                td_s = ib.reqMktData(sc, "100,101,106", False, False)
                td_l = ib.reqMktData(lc, "100,101,106", False, False)
                await asyncio.sleep(4)
                s_mid = _spx_mid(td_s)
                l_mid = _spx_mid(td_l)
                ib.cancelMktData(sc)
                ib.cancelMktData(lc)
                if s_mid > 0 and l_mid > 0:
                    current_credit = round(s_mid - l_mid, 2)
                    # Profit = credit collected at entry minus current cost to close
                    live_pnl += (entry_credit - current_credit) * 100 * qty
            except Exception as ex:
                log.warning("SPX 0DTE monitor quote failed %s %s/%s: %s", right, sk, lk, ex)

        sp["live_pnl"] = round(live_pnl, 2)
        _spx_log("MONITOR", f"spread={sid} live_pnl=${live_pnl:.2f} target=${sp['profit_target']:.2f}")

        profit_target = sp["profit_target"]
        # Stop loss = stop_loss_mult × credit collected (e.g. 2× = lose back 2× what we took in).
        # sp["max_loss"] is the theoretical full-width loss — do NOT use it as the stop threshold.
        stop_threshold = sp["total_credit"] * float(cfg["stop_loss_mult"])

        if live_pnl >= profit_target:
            _spx_log("PROFIT_TARGET",
                     f"P&L=${live_pnl:.2f} >= target=${profit_target:.2f}", sid)
            await _spx_close_spread(ib, sid, "profit_target")
            continue

        if live_pnl <= -stop_threshold:
            _spx_log("STOP_LOSS",
                     f"P&L=${live_pnl:.2f} <= -stop=-${stop_threshold:.2f} ({cfg['stop_loss_mult']}× credit)", sid)
            sx["last_stop_time"] = now.isoformat()
            await _spx_close_spread(ib, sid, "stop_loss")
            continue

        if now >= _t(cfg["force_close_time"]):
            _spx_log("FORCE_CLOSE",
                     f"Force-close at {cfg['force_close_time']} ET  P&L=${live_pnl:.2f}", sid)
            await _spx_close_spread(ib, sid, "force_close")
            continue

    _spx_save_state()


async def _spx_monitor_loop() -> None:
    """Background loop: entry + monitor every 30s during market hours (9:25–16:15 ET)."""
    await asyncio.sleep(40)
    while True:
        try:
            from zoneinfo import ZoneInfo
            ET  = ZoneInfo("America/New_York")
            now = datetime.now(ET)
            if now.weekday() >= 5:
                await asyncio.sleep(300)
                continue
            mkt_open  = now.replace(hour=9,  minute=25, second=0, microsecond=0)
            mkt_close = now.replace(hour=16, minute=15, second=0, microsecond=0)
            if not (mkt_open <= now <= mkt_close):
                await asyncio.sleep(60)
                continue
            sx = state["spx_0dte"]
            if not sx["enabled"]:
                await asyncio.sleep(30)
                continue
            ib = state.get("ib")
            if not ib or not ib.isConnected():
                await asyncio.sleep(30)
                continue
            loop = asyncio.get_event_loop()
            if sx["spreads"]:
                await loop.run_in_executor(
                    None,
                    lambda: _run_in_streaming_loop(_spx_monitor_coro(ib), timeout=25))
            if (not sx["spreads"]
                    and sx["today_pnl"] < sx["config"]["daily_profit_target"]
                    and sx["attempts_today"] < sx["config"]["max_attempts"]):
                await loop.run_in_executor(
                    None,
                    lambda: _run_in_streaming_loop(_spx_entry_coro(ib), timeout=35))
        except Exception as exc:
            log.warning("SPX 0DTE loop error: %s", exc)
        await asyncio.sleep(30)


# ── SPX 0DTE endpoints ─────────────────────────────────────────────────────────

class SPXConfigRequest(BaseModel):
    daily_profit_target:   Optional[float] = None
    spread_width:          Optional[int]   = None
    otm_pct:               Optional[float] = None
    profit_pct:            Optional[float] = None
    stop_loss_mult:        Optional[float] = None
    entry_start_time:      Optional[str]   = None
    entry_cutoff_time:     Optional[str]   = None
    force_close_time:      Optional[str]   = None
    max_attempts:          Optional[int]   = None
    max_margin:            Optional[float] = None
    min_credit_per_spread: Optional[float] = None


@app.get("/spx-0dte/status")
def spx_0dte_status():
    sx  = state["spx_0dte"]
    cfg = sx["config"]
    pnl = round(sx.get("today_pnl", 0.0), 2)
    return {
        "enabled":        sx["enabled"],
        "config":         cfg,
        "spreads":        sx["spreads"],
        "closed_today":   sx.get("closed_today", []),
        "decisions":      sx.get("decisions", [])[-50:],
        "attempts_today": sx.get("attempts_today", 0),
        "summary": {
            "open_spreads":   len(sx["spreads"]),
            "today_pnl":      pnl,
            "goal_pct":       round(pnl / cfg["daily_profit_target"] * 100, 1)
                              if cfg["daily_profit_target"] > 0 else 0,
            "attempts_today": sx.get("attempts_today", 0),
            "closed_trades":  len(sx.get("closed_today", [])),
        },
    }


@app.post("/spx-0dte/enable")
def spx_0dte_enable(enabled: bool = True):
    sx = state["spx_0dte"]
    sx["enabled"] = enabled
    _spx_log("CONFIG", f"{'enabled' if enabled else 'disabled'} by user")
    _spx_save_state()
    return {"enabled": sx["enabled"]}


@app.post("/spx-0dte/config")
def spx_0dte_config(req: SPXConfigRequest):
    sx  = state["spx_0dte"]
    cfg = sx["config"]
    updates = req.model_dump(exclude_none=True)
    cfg.update(updates)
    _spx_log("CONFIG", f"updated: {updates}")
    _spx_save_state()
    return {"config": cfg}


@app.post("/spx-0dte/close/{spread_id}")
async def spx_0dte_close(spread_id: str):
    sx = state["spx_0dte"]
    if spread_id not in sx["spreads"]:
        raise HTTPException(404, f"{spread_id} not found in open spreads")
    ib = state.get("ib")
    if not ib or not ib.isConnected():
        raise HTTPException(503, "IBKR not connected")
    _spx_log("MANUAL_CLOSE", "manual close requested", spread_id)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: _run_in_streaming_loop(_spx_close_spread(ib, spread_id, "manual_close"), timeout=20))
    return {"status": "closing", "spread_id": spread_id}


# ── Live Tape WebSocket ─────────────────────────────────────────────────────
@app.websocket("/ws/tape/{ticker}")
async def ws_tape(websocket: WebSocket, ticker: str, block: int = Query(default=5000)):
    """
    Stream real-time tick-by-tick trade data (Time & Sales) for a stock.
    Each message is a JSON object with the raw tick plus plain-English explanations.
    Uses a separate IB client ID (pool 20-29) so it never conflicts with the main app.
    """
    await websocket.accept()

    cid = _acquire_tape_cid()
    if cid is None:
        await websocket.send_json({
            "type": "error",
            "message": "Too many concurrent tape connections (max 10). Please try again shortly.",
        })
        await websocket.close()
        return

    tape_ib = IB()
    contract = None

    try:
        await tape_ib.connectAsync(TWS_HOST, TWS_PORT, clientId=cid, timeout=15)

        contract = Stock(ticker.upper(), "SMART", "USD")
        await tape_ib.qualifyContractsAsync(contract)

        # Per-session accumulators
        cum_vol:    int   = 0
        cum_pv:     float = 0.0
        buy_vol:    int   = 0
        sell_vol:   int   = 0
        last_price: Optional[float] = None
        last_dir:   int   = 0
        open_price: Optional[float] = None
        block_count: int  = 0
        tick_idx:   int   = 0

        tick_queue: asyncio.Queue = asyncio.Queue()
        tape_print_buffer: list  = []   # accumulated this session; flushed to DB on close

        def _direction(price: float) -> int:
            nonlocal last_price, last_dir
            if last_price is None:      return 0
            if price > last_price:      last_dir = 1
            elif price < last_price:    last_dir = -1
            return last_dir

        def on_update(t):
            nonlocal tick_idx
            new = t.tickByTicks[tick_idx:]
            tick_idx = len(t.tickByTicks)
            for tk in new:
                tick_queue.put_nowait(tk)

        ticker_sub = tape_ib.reqTickByTickData(
            contract, "AllLast", numberOfTicks=0, ignoreSize=False
        )
        ticker_sub.updateEvent += on_update

        await websocket.send_json({"type": "connected", "ticker": ticker.upper(), "client_id": cid})

        while True:
            try:
                tk = await asyncio.wait_for(tick_queue.get(), timeout=20)
            except asyncio.TimeoutError:
                # keep the WS alive during quiet periods (pre-market, AH)
                await websocket.send_json({"type": "heartbeat"})
                if not tape_ib.isConnected():
                    break
                continue

            price = float(tk.price)
            size  = int(tk.size)
            if size == 0 or price <= 0:
                continue

            d = _direction(price)
            if open_price is None:
                open_price = price

            cum_vol  += size
            cum_pv   += price * size
            vwap      = cum_pv / cum_vol
            if d >= 0: buy_vol  += size
            else:      sell_vol += size
            net_delta = buy_vol - sell_vol
            is_block  = size >= block
            if is_block:
                block_count += 1
            last_price = price

            # ── Plain-English explanations ──────────────────────────────────
            side     = "bought" if d >= 0 else "sold"
            dir_word = "UP" if d > 0 else "DOWN" if d < 0 else "flat"

            what_happened = (
                f"{size:,} shares {side} at ${price:.2f}"
                + (f" — price ticked {dir_word}" if d != 0 else "")
            )

            vwap_diff = price - vwap
            if abs(vwap_diff) < 0.03:
                vwap_story = (
                    f"Trading right at the session average (VWAP ${vwap:.2f}) — "
                    f"neither side has an edge right now"
                )
            elif vwap_diff > 0:
                vwap_story = (
                    f"${vwap_diff:.2f} above today's average — buyers are willing to pay up "
                    f"(VWAP ${vwap:.2f})"
                )
            else:
                vwap_story = (
                    f"${abs(vwap_diff):.2f} below today's average — price looks cheap "
                    f"relative to where most trading happened today (VWAP ${vwap:.2f})"
                )

            if net_delta > 0:
                delta_story = (
                    f"Buyers are in control — {_tape_fmt_vol(net_delta)} more shares "
                    f"bought than sold so far today"
                )
            elif net_delta < 0:
                delta_story = (
                    f"Sellers are in control — {_tape_fmt_vol(abs(net_delta))} more shares "
                    f"sold than bought so far today"
                )
            else:
                delta_story = "Perfectly balanced — equal buying and selling pressure so far"

            block_story = None
            if is_block:
                block_story = (
                    f"Large institutional print: {size:,} shares {side} at ${price:.2f}. "
                    f"Block trades (≥ {block:,} shares) come from funds or trading desks "
                    f"moving a large position — they rarely trade like this by accident."
                )

            pct_from_open = None
            if open_price and open_price > 0:
                pct_from_open = round((price - open_price) / open_price * 100, 2)

            ts_str    = tk.time.strftime("%Y-%m-%dT%H:%M:%S")
            sess_date = tk.time.strftime("%Y-%m-%d")
            cvd_score_now = (state["tape_sentiment"].get(ticker.upper(), {})
                             .get("score"))

            await websocket.send_json({
                "type":           "tick",
                "time":           tk.time.strftime("%H:%M:%S.%f")[:-3],
                "price":          price,
                "size":           size,
                "direction":      d,
                "cum_vol":        cum_vol,
                "vwap":           round(vwap, 2),
                "buy_vol":        buy_vol,
                "sell_vol":       sell_vol,
                "delta":          net_delta,
                "exchange":       tk.exchange or "—",
                "is_block":       is_block,
                "is_after_hours": tk.tickAttribLast.pastLimit,
                "block_count":    block_count,
                "open_price":     open_price,
                "pct_from_open":  pct_from_open,
                "what_happened":  what_happened,
                "vwap_story":     vwap_story,
                "delta_story":    delta_story,
                "block_story":    block_story,
            })

            # Buffer this print for DB persistence (flushed in batch on session close)
            tape_print_buffer.append({
                "ts":             ts_str,
                "session_date":   sess_date,
                "ticker":         ticker.upper(),
                "price":          price,
                "size":           size,
                "direction":      d,
                "exchange":       tk.exchange or "",
                "is_block":       int(is_block),
                "is_after_hours": int(bool(tk.tickAttribLast.pastLimit)),
                "vwap":           round(vwap, 4),
                "cum_vol":        cum_vol,
                "buy_vol":        buy_vol,
                "sell_vol":       sell_vol,
                "net_delta":      net_delta,
                "pct_from_open":  pct_from_open,
                "cvd_score":      cvd_score_now,
            })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.error("Tape WS error for %s: %s", ticker, exc)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            if contract and tape_ib.isConnected():
                tape_ib.cancelTickByTickData(contract, "AllLast")
        except Exception:
            pass
        try:
            tape_ib.disconnect()
        except Exception:
            pass
        _release_tape_cid(cid)
        # Persist all buffered prints to tape_data.db in one batch write
        if tape_print_buffer:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(
                None, _tape_db_flush_prints, ticker.upper(), tape_print_buffer
            )
            log.info("Tape WS closed for %s (cid=%d) — flushing %d prints to tape_data.db",
                     ticker, cid, len(tape_print_buffer))
        else:
            log.info("Tape WS closed for %s (cid=%d)", ticker, cid)


# ── Shared guard ───────────────────────────────────────────────────────────
def _require_connection():
    if not state["connected"] or not state["ib"]:
        raise HTTPException(503, "IBKR not connected — ensure TWS/IB Gateway is running")


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
