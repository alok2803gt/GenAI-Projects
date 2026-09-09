"""
Thread 1 experiments: variants of build_features() that differ from the
live model in one deliberate, isolated way at a time, so each experiment
can attribute a result to the specific thing changed rather than a
tangle of simultaneous changes.

build_features_horizon(df, horizon): identical to signal_model.py's
build_features() (same 9 FEATURE_COLS, same computation) except the
target label looks `horizon` bars ahead instead of hardcoding 1 bar --
main.py's live model predicts the very next 5-min bar, the single
noisiest possible horizon. Testing horizon in {1 (baseline), 3, 6, 12}
(15/30/60 min ahead) checks whether these same 9 features carry any real
signal at a longer, plausibly less noise-dominated horizon.
"""
import pandas as pd

from signal_model import FEATURE_COLS  # noqa: F401 -- re-exported for callers


def build_features_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
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
    df["target"] = (df["close"].shift(-horizon) > df["close"]).astype(int)

    return df.dropna()
