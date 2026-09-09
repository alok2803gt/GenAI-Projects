"""
Thread 2 / Experiment 1: does the live model's actual label (the one
LEAP's scanner hard-filters on -- main.py:6816-6818, "if sig.get('label')
== 'SELL': return []") have any real correlation with what happens to the
stock afterward?

This is a DIFFERENT question from the main backtest (RESEARCH_LOG.md),
which tested "does reversal-trading on this label make money." Here we
ask something more directly relevant to how CSP/LEAP actually use the
signal: not "would you make money trading it," but "does a SELL label
actually precede below-average returns, and a BUY label above-average
returns" -- since that's the entire statistical premise LEAP's hard
filter and CSP's use of the underlying features rest on.

Note: _stock_quality_score() (main.py:1807-1835), the OTHER mechanism
CSP/LEAP use, does NOT consume the XGBoost model's label/probability at
all -- it's a separate hand-coded RSI/momentum/volatility heuristic. This
experiment only speaks to the label-based hard filter, not that
heuristic. That's a distinct, separate follow-up if this one doesn't
settle the question.

Method: reuse the exact same walk-forward retraining as the main
backtest (per-ticker variant, 20-day window -- picked because the main
grid already showed window size barely matters, so no need to re-run all
3), but instead of simulating reversal trades, record the label assigned
at every bar alongside that bar's ACTUAL forward return over two
horizons: 5 trading days (~CSP's typical weekly-expiry horizon) and 63
trading days (~LEAP's typical multi-month horizon). Bucket forward
returns by label and compare means/medians -- if the label carries real
information, BUY should show a better forward-return distribution than
SELL. If they're statistically indistinguishable, the label has no basis
for the role it's currently playing in LEAP's filter.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from fetch_data import fetch_and_cache
from signal_model import build_features, FEATURE_COLS, BUY_THRESHOLD, SELL_THRESHOLD
from backtest_engine import train_model_from_features, _trading_dates

WINDOW_SIZE_DAYS = 20
RETRAIN_EVERY_DAYS = 5
FORWARD_HORIZONS_BARS = {"CSP_5d": 5 * 78, "LEAP_63d": 63 * 78}  # ~78 5-min bars/trading day incl. extended hours


def label_and_forward_returns(df: pd.DataFrame, log=print) -> pd.DataFrame:
    feat = build_features(df)
    dates = _trading_dates(feat)

    records = []
    i = WINDOW_SIZE_DAYS
    while i + RETRAIN_EVERY_DAYS <= len(dates):
        train_start, train_end = dates[i - WINDOW_SIZE_DAYS], dates[i - 1]
        score_start, score_end = dates[i], dates[min(i + RETRAIN_EVERY_DAYS - 1, len(dates) - 1)]

        train_slice = feat.loc[str(train_start):str(train_end)]
        if len(train_slice) < 60:
            i += RETRAIN_EVERY_DAYS
            continue
        model, acc = train_model_from_features(train_slice)

        score_slice = feat.loc[str(score_start):str(score_end)]
        if score_slice.empty:
            i += RETRAIN_EVERY_DAYS
            continue

        X = score_slice[FEATURE_COLS].values
        prob = model.predict_proba(X)[:, 1]
        labels = np.where(prob > BUY_THRESHOLD, "BUY", np.where(prob < SELL_THRESHOLD, "SELL", "HOLD"))

        for t, label, p in zip(score_slice.index, labels, prob):
            records.append({"time": t, "label": label, "prob": float(p), "close": float(feat.loc[t, "close"])})

        log(f"  retrain {train_start}..{train_end} (n={len(train_slice)}, acc={acc:.3f}) "
            f"-> labeled {score_start}..{score_end} ({len(score_slice)} bars)")
        i += RETRAIN_EVERY_DAYS

    rec_df = pd.DataFrame(records).set_index("time")

    # Attach forward returns at each horizon using the FULL feat series (not
    # just the labeled slice) so a label near the end of a scoring window can
    # still look far enough ahead.
    closes = feat["close"]
    idx_pos = {t: n for n, t in enumerate(closes.index)}
    for name, bars_ahead in FORWARD_HORIZONS_BARS.items():
        fwd = []
        for t in rec_df.index:
            pos = idx_pos.get(t)
            if pos is None or pos + bars_ahead >= len(closes):
                fwd.append(np.nan)
                continue
            fwd.append((closes.iloc[pos + bars_ahead] - closes.iloc[pos]) / closes.iloc[pos] * 100)
        rec_df[f"fwd_ret_{name}_pct"] = fwd

    return rec_df


def summarize_by_label(rec_df: pd.DataFrame, horizon_col: str) -> dict:
    out = {}
    for label in ("BUY", "SELL", "HOLD"):
        sub = rec_df.loc[rec_df["label"] == label, horizon_col].dropna()
        if sub.empty:
            out[label] = {"n": 0}
            continue
        out[label] = {
            "n": int(len(sub)),
            "mean_pct": round(float(sub.mean()), 4),
            "median_pct": round(float(sub.median()), 4),
            "std_pct": round(float(sub.std()), 4),
            "pct_positive": round(float((sub > 0).mean() * 100), 2),
        }
    return out


def main():
    data = fetch_and_cache()
    all_results = {}
    for tk, df in data.items():
        print(f"=== {tk} ===")
        rec_df = label_and_forward_returns(df)
        tk_result = {"n_labeled_bars": len(rec_df)}
        for horizon_name in FORWARD_HORIZONS_BARS:
            col = f"fwd_ret_{horizon_name}_pct"
            summary = summarize_by_label(rec_df, col)
            tk_result[horizon_name] = summary
            print(f"  {horizon_name}: {summary}")
        all_results[tk] = tk_result
        rec_df.to_csv(f"forward_returns_{tk}.csv")

    with open("forward_returns_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nSaved forward_returns_summary.json and per-ticker CSVs")


if __name__ == "__main__":
    main()
