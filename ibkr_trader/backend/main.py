"""
XGBoost Day Trading Signal Engine — IBKR Backend  (v2 — Live Streaming)
FastAPI server that:
  1. Connects to TWS / IB Gateway via ib_insync
  2. Subscribes to 5-min OHLCV bars with keepUpToDate=True (event-driven streaming)
  3. Includes extended / after-hours data (useRTH=False)
  4. Engineers 9 features on every bar event
  5. Scores with XGBoost (or trains on-the-fly from seeded history)
  6. Serves /signal, /bars, /status endpoints to the React dashboard
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ib_insync import IB, Stock, util
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

# ── Config ─────────────────────────────────────────────────────────────────
TWS_HOST = "127.0.0.1"
TWS_PORT = 7496          # 7497=TWS paper | 7496=TWS live | 4002=IB Gateway paper
TWS_CLIENT_ID = 10
MODEL_PATH = "model.joblib"
BAR_SIZE = "5 mins"
HISTORY_DURATION = "5 D"   # Seed with 5 days so RSI/SMA have enough history
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45

FEATURE_COLS = [
    "rsi", "sma5", "sma14", "momentum",
    "vol_ratio", "body_pct", "upper_wick", "lower_wick", "volatility"
]

# ── Global state ───────────────────────────────────────────────────────────
state: Dict = {
    "ib": None,
    "connected": False,
    "model": None,
    "model_accuracy": None,
    "bars": {},            # ticker -> list[dict]  (last 80 bars)
    "signals": {},         # ticker -> latest signal dict
    "last_update": {},     # ticker -> ISO datetime str
    "subscriptions": {},   # ticker -> BarDataList (live handle)
    "error": None,
}

# Thread-safe mutable ticker list (API writes, streaming thread reads)
TICKERS: List[str] = ["AAPL", "MSFT", "NVDA", "SPY"]
_tickers_lock = threading.Lock()

# ── Feature engineering ────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["close", "open", "high", "low", "volume"]:
        df[col] = df[col].astype(float)

    df["sma5"]     = df["close"].rolling(5).mean()
    df["sma14"]    = df["close"].rolling(14).mean()
    df["momentum"] = (df["close"] - df["sma14"]) / (df["sma14"] + 1e-9)
    df["vol_ratio"]= df["volume"] / (df["volume"].rolling(14).mean() + 1e-9)

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


def on_bar_update(ticker: str, bars, has_new_bar: bool) -> None:
    """
    Called by ib_insync on every bar event — both partial (in-flight 5-min
    bar updating) and completed (new bar appended).  has_new_bar=True means
    a bar just closed and a fresh one started.
    """
    try:
        df = _bars_to_df(bars)
        if df.empty:
            return

        # Cache raw bars (keep last 80)
        bars_list = df.tail(80).to_dict(orient="records")
        for b in bars_list:
            t = b.get("time")
            if hasattr(t, "isoformat"):
                b["time"] = t.isoformat()
        state["bars"][ticker] = bars_list

        # Lazy model training on first ticker that has enough history
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
            else:
                log.debug(f"BAR UPD  {ticker}  close={sig['close']}")

    except Exception as e:
        log.warning(f"on_bar_update error [{ticker}]: {e}", exc_info=True)


def _session_label(ts: pd.Timestamp) -> str:
    """Return PRE / RTH / POST / CLOSED based on Eastern time."""
    try:
        et = ts.tz_convert("America/New_York") if ts.tzinfo else ts
        h = et.hour + et.minute / 60
        if 4.0 <= h < 9.5:
            return "PRE"
        if 9.5 <= h < 16.0:
            return "RTH"
        if 16.0 <= h < 20.0:
            return "POST"
        return "CLOSED"
    except Exception:
        return "UNKNOWN"


# ── Streaming subscription ─────────────────────────────────────────────────
async def subscribe_ticker(ib: IB, ticker: str) -> None:
    """Open a keepUpToDate streaming subscription for one ticker."""
    contract = Stock(ticker, "SMART", "USD")
    await ib.qualifyContractsAsync(contract)

    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime="",
        durationStr=HISTORY_DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow="TRADES",
        useRTH=False,           # Extended hours included
        formatDate=1,
        keepUpToDate=True,      # Event-driven streaming
    )

    state["subscriptions"][ticker] = bars

    # Wire the live callback
    bars.updateEvent += lambda b, h: on_bar_update(ticker, b, h)

    # Seed state immediately from historical data already in `bars`
    on_bar_update(ticker, bars, False)
    log.info(f"Streaming  {ticker}  ({len(bars)} bars seeded, extended hours ON)")


async def _subscribe_pending(ib: IB, known: set) -> set:
    """Subscribe any tickers added to TICKERS since last check."""
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
            await ib.connectAsync(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID, timeout=15)
            log.info(f"Connected to IBKR  {TWS_HOST}:{TWS_PORT}")
            state["ib"] = ib
            state["connected"] = True
            state["error"] = None

            known: set = set()
            known = await _subscribe_pending(ib, known)

            # Keep the event loop alive; check for newly added tickers every 10 s
            while ib.isConnected():
                known = await _subscribe_pending(ib, known)
                await asyncio.sleep(10)

            state["connected"] = False
            log.warning("IBKR disconnected — retrying in 15 s")

        except Exception as e:
            state["connected"] = False
            state["error"] = str(e)
            log.error(f"IBKR error: {e}  — retrying in 15 s")
            await asyncio.sleep(15)


def streaming_loop() -> None:
    asyncio.run(streaming_loop_async())


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


app = FastAPI(title="XGBoost IBKR Trader — Live Streaming", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────
class AddTickerRequest(BaseModel):
    ticker: str


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/status")
def get_status():
    return {
        "connected": state["connected"],
        "error": state["error"],
        "model_accuracy": state["model_accuracy"],
        "tickers": list(state["signals"].keys()),
        "subscriptions": list(state["subscriptions"].keys()),
        "last_updates": state["last_update"],
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


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
