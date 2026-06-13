"""
XGBoost Day Trading Signal Engine — IBKR Backend
FastAPI server that:
  1. Connects to TWS / IB Gateway via ib_insync
  2. Pulls live 5-min OHLCV bars
  3. Engineers 9 features
  4. Scores with a trained XGBoost model (or trains one on-the-fly)
  5. Serves /signal, /bars, /status endpoints to the React dashboard
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ib_insync import IB, Stock, util, BarData
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
TWS_PORT = 7496          # 7497 = TWS paper | 7496 = TWS live | 4002 = IB Gateway paper
TWS_CLIENT_ID = 10       # any unused client ID
MODEL_PATH = "model.joblib"
BAR_SIZE = "5 mins"
HISTORY_DURATION = "3 D"
POLL_INTERVAL = 30       # seconds between live bar refreshes
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
    "bars": {},          # ticker -> list[dict]
    "signals": {},       # ticker -> latest signal dict
    "last_update": {},   # ticker -> datetime
    "error": None,
}

# ── Feature engineering ────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features in-place. Returns df with feature cols + target."""
    df = df.copy()
    df["close"] = df["close"].astype(float)
    df["open"]  = df["open"].astype(float)
    df["high"]  = df["high"].astype(float)
    df["low"]   = df["low"].astype(float)
    df["volume"]= df["volume"].astype(float)

    # SMA
    df["sma5"]  = df["close"].rolling(5).mean()
    df["sma14"] = df["close"].rolling(14).mean()

    # Momentum = deviation from SMA14
    df["momentum"] = (df["close"] - df["sma14"]) / (df["sma14"] + 1e-9)

    # Volume ratio vs 14-bar avg
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(14).mean() + 1e-9)

    # Candle geometry
    rng = (df["high"] - df["low"]).replace(0, 1e-6)
    df["body_pct"]   = (df["close"] - df["open"]).abs() / rng
    df["upper_wick"] = (df["high"] - df[["open","close"]].max(axis=1)) / rng
    df["lower_wick"] = (df[["open","close"]].min(axis=1) - df["low"]) / rng

    # RSI (14)
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / (loss + 1e-9)
    df["rsi"] = 100 - 100 / (1 + rs)

    # Volatility = rolling std of returns
    df["volatility"] = df["close"].pct_change().rolling(14).std()

    # Target: did close go up next bar?
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    return df.dropna()


# ── XGBoost model ──────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    """Train XGBoost on the given feature dataframe. Returns (model, accuracy)."""
    df = build_features(df)
    if len(df) < 60:
        raise ValueError(f"Not enough data to train: {len(df)} rows (need ≥60)")

    X = df[FEATURE_COLS].values
    y = df["target"].values

    tscv = TimeSeriesSplit(n_splits=3)
    accs = []
    model = None
    for train_idx, val_idx in tscv.split(X):
        m = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=42,
        )
        m.fit(X[train_idx], y[train_idx])
        preds = m.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], preds))
        model = m

    accuracy = float(np.mean(accs))
    log.info(f"Model trained  accuracy={accuracy:.3f}  rows={len(df)}")
    return model, accuracy


def predict(model, df: pd.DataFrame) -> dict:
    """Score the last row. Returns signal dict."""
    df = build_features(df)
    if df.empty:
        return {"label": "HOLD", "prob": 0.5, "confidence": 0.0}

    last = df[FEATURE_COLS].iloc[[-1]].values
    prob = float(model.predict_proba(last)[0][1])
    confidence = abs(prob - 0.5) * 2
    label = "BUY" if prob > BUY_THRESHOLD else "SELL" if prob < SELL_THRESHOLD else "HOLD"
    feat  = df[FEATURE_COLS].iloc[-1].to_dict()

    return {
        "label": label,
        "prob": round(prob, 4),
        "confidence": round(confidence, 4),
        "features": {k: round(float(v), 4) for k, v in feat.items()},
        "close": round(float(df["close"].iloc[-1]), 4),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ── IBKR helpers ──────────────────────────────────────────────────────────
async def fetch_bars(ib: IB, ticker: str) -> pd.DataFrame:
    """Async version — never blocks the event loop."""
    contract = Stock(ticker, "SMART", "USD")
    await ib.qualifyContractsAsync(contract)
    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime="",
        durationStr=HISTORY_DURATION,
        barSizeSetting=BAR_SIZE,
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        raise ValueError(f"No bars returned for {ticker} — market may be closed or contract invalid")
    df = util.df(bars)[["date", "open", "high", "low", "close", "volume"]]
    df.rename(columns={"date": "time"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"])
    log.info(f"Fetched {len(df)} bars for {ticker}")
    return df


# ── Background polling loop ────────────────────────────────────────────────
async def polling_loop_async(tickers: List[str]):
    """Async polling loop — owns its own event loop via asyncio.run()."""
    while True:
        try:
            ib = IB()
            await ib.connectAsync(TWS_HOST, TWS_PORT, clientId=TWS_CLIENT_ID, timeout=15)
            log.info(f"Connected to IBKR  host={TWS_HOST}  port={TWS_PORT}")
            state["ib"] = ib
            state["connected"] = True
            state["error"] = None

            while ib.isConnected():
                for ticker in tickers:
                    try:
                        df = await fetch_bars(ib, ticker)
                        bars_list = df.tail(80).to_dict(orient="records")
                        # convert timestamps to strings
                        for b in bars_list:
                            if hasattr(b["time"], "isoformat"):
                                b["time"] = b["time"].isoformat()

                        state["bars"][ticker] = bars_list

                        # train / retrain model if not loaded (run in thread to avoid blocking)
                        if state["model"] is None:
                            loop = asyncio.get_event_loop()
                            model, acc = await loop.run_in_executor(None, train_model, df)
                            state["model"] = model
                            state["model_accuracy"] = acc
                            joblib.dump(model, MODEL_PATH)

                        sig = predict(state["model"], df)
                        sig["ticker"] = ticker
                        state["signals"][ticker] = sig
                        state["last_update"][ticker] = datetime.utcnow().isoformat()
                        log.info(f"{ticker}  {sig['label']}  prob={sig['prob']}  close={sig['close']}")

                    except Exception as e:
                        log.warning(f"Error fetching {ticker}: {e}", exc_info=True)

                await asyncio.sleep(POLL_INTERVAL)

        except Exception as e:
            state["connected"] = False
            state["error"] = str(e)
            log.error(f"IBKR connection lost: {e}  — retrying in 15s")
            await asyncio.sleep(15)


def polling_loop(tickers: List[str]):
    """Entry point for the daemon thread — creates its own event loop."""
    asyncio.run(polling_loop_async(tickers))


# ── FastAPI app ────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "SPY"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load cached model if available
    if os.path.exists(MODEL_PATH):
        state["model"] = joblib.load(MODEL_PATH)
        log.info("Loaded cached model from disk")

    # Start IBKR polling thread
    t = threading.Thread(
        target=polling_loop,
        args=(DEFAULT_TICKERS,),
        daemon=True,
    )
    t.start()
    log.info("Polling thread started")
    yield
    if state["ib"] and state["ib"].isConnected():
        state["ib"].disconnect()


app = FastAPI(title="XGBoost IBKR Trader", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────
class TrainRequest(BaseModel):
    ticker: str


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
        "last_updates": state["last_update"],
    }


@app.get("/signal/{ticker}")
def get_signal(ticker: str):
    ticker = ticker.upper()
    if ticker not in state["signals"]:
        raise HTTPException(404, f"{ticker} not found. Add it via POST /add_ticker first.")
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
    if ticker not in DEFAULT_TICKERS:
        DEFAULT_TICKERS.append(ticker)
        log.info(f"Added ticker {ticker}")
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
