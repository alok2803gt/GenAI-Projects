"""
Verbatim reimplementation of main.py's build_features()/train_model()/
predict() (main.py:781-856 as of 2026-09-07) -- same 9 FEATURE_COLS, same
XGBClassifier hyperparameters, same BUY/SELL thresholds, same 3-fold
TimeSeriesSplit-then-keep-last-fold training procedure. No shared helper
exists in this codebase for this feature set (confirmed by exhaustive
grep across breakout_research/, daytrader_research/, evc_research/, and
every top-level *_backtest.py), so this is a deliberate, convention-
following duplication -- the point of this backtest is testing what's
actually deployed, not a different or improved model.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

FEATURE_COLS = [
    "rsi", "sma5", "sma14", "momentum",
    "vol_ratio", "body_pct", "upper_wick", "lower_wick", "volatility",
]
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["close", "open", "high", "low", "volume"]:
        df[col] = df[col].astype(float)

    df["sma5"] = df["close"].rolling(5).mean()
    df["sma14"] = df["close"].rolling(14).mean()
    df["momentum"] = (df["close"] - df["sma14"]) / (df["sma14"] + 1e-9)
    df["vol_ratio"] = df["volume"] / (df["volume"].rolling(14).mean() + 1e-9)

    rng = (df["high"] - df["low"]).replace(0, 1e-6)
    df["body_pct"] = (df["close"] - df["open"]).abs() / rng
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / rng

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    df["volatility"] = df["close"].pct_change().rolling(14).std()
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    return df.dropna()


def train_model(df: pd.DataFrame):
    df = build_features(df)
    if len(df) < 60:
        raise ValueError(f"Need >=60 rows, got {len(df)}")

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

    acc = float(np.mean(accs)) if accs else None
    return model, acc


def predict_label(model, df: pd.DataFrame) -> dict | None:
    """Score the single most-recent bar in df. Returns None if not enough
    history to compute features (mirrors predict()'s df.empty guard)."""
    feat_df = build_features(df)
    if feat_df.empty:
        return None

    last = feat_df[FEATURE_COLS].iloc[[-1]].values
    prob = float(model.predict_proba(last)[0][1])
    label = "BUY" if prob > BUY_THRESHOLD else "SELL" if prob < SELL_THRESHOLD else "HOLD"
    return {"label": label, "prob": prob, "close": float(feat_df["close"].iloc[-1])}
