"""
Thread 2 / Experiment 2: does _stock_quality_score() (main.py:1807-1835),
the OTHER mechanism CSP/LEAP use alongside the XGBoost label, actually
correlate with what happens to the stock afterward?

Unlike the XGBoost label (experiment_forward_returns.py), this score
never touches the model at all -- it's a hand-coded rule straight off 3
of the 9 raw features (rsi, momentum, volatility), reimplemented here
verbatim:
  - RSI 45-65 -> +0.40, RSI 35-45 or 65-72 -> +0.20 (else +0)
  - momentum > 0.002 -> +0.35, momentum > 0 -> +0.20 (else +0)
  - volatility < 0.005 -> +0.25, volatility < 0.010 -> +0.15 (else +0)
  capped at 1.0.

No model training needed -- this is a pure function of build_features()'s
output at each bar, so this experiment is cheap: compute the score at
every real bar in the full 2.5y history, bucket into terciles (low/mid/
high quality), and compare forward-return distributions the same way as
the label experiment (5-day ~ CSP weekly horizon, 63-day ~ LEAP horizon).
If quality score doesn't differentiate forward returns any better than
chance, CSP's 15-point weighting and LEAP's 0.3 cutoff have no
statistical basis either, same conclusion as if the XGBoost label fails.
"""
import json
import numpy as np
import pandas as pd

from fetch_data import fetch_and_cache
from signal_model import build_features

FORWARD_HORIZONS_BARS = {"CSP_5d": 5 * 78, "LEAP_63d": 63 * 78}


def stock_quality_score(rsi: float, momentum: float, volatility: float) -> float:
    score = 0.0
    if 45 <= rsi <= 65:
        score += 0.40
    elif 35 <= rsi < 45 or 65 < rsi <= 72:
        score += 0.20
    if momentum > 0.002:
        score += 0.35
    elif momentum > 0:
        score += 0.20
    if volatility < 0.005:
        score += 0.25
    elif volatility < 0.010:
        score += 0.15
    return round(min(score, 1.0), 3)


def compute_scores_and_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    feat = build_features(df)
    scores = feat.apply(lambda r: stock_quality_score(r["rsi"], r["momentum"], r["volatility"]), axis=1)
    out = pd.DataFrame({"quality_score": scores, "close": feat["close"]}, index=feat.index)

    closes = feat["close"]
    for name, bars_ahead in FORWARD_HORIZONS_BARS.items():
        fwd = closes.shift(-bars_ahead)
        out[f"fwd_ret_{name}_pct"] = (fwd - closes) / closes * 100

    # Tercile buckets computed over this ticker's own full score distribution
    out["bucket"] = pd.qcut(out["quality_score"], 3, labels=["LOW", "MID", "HIGH"], duplicates="drop")
    return out


def summarize_by_bucket(df: pd.DataFrame, horizon_col: str) -> dict:
    out = {}
    for bucket in ("LOW", "MID", "HIGH"):
        sub = df.loc[df["bucket"] == bucket, horizon_col].dropna()
        if sub.empty:
            out[bucket] = {"n": 0}
            continue
        out[bucket] = {
            "n": int(len(sub)),
            "mean_pct": round(float(sub.mean()), 4),
            "median_pct": round(float(sub.median()), 4),
            "pct_positive": round(float((sub > 0).mean() * 100), 2),
        }
    return out


def main():
    data = fetch_and_cache()
    results = {}
    for tk, df in data.items():
        print(f"=== {tk} ===")
        scored = compute_scores_and_forward_returns(df)
        tk_result = {"n_bars": len(scored), "score_distribution": scored["quality_score"].describe().to_dict()}
        for horizon_name in FORWARD_HORIZONS_BARS:
            col = f"fwd_ret_{horizon_name}_pct"
            summary = summarize_by_bucket(scored, col)
            tk_result[horizon_name] = summary
            print(f"  {horizon_name}: {summary}")
        results[tk] = tk_result
        scored.to_csv(f"quality_score_{tk}.csv")

    with open("quality_score_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved quality_score_summary.json and per-ticker CSVs")


if __name__ == "__main__":
    main()
