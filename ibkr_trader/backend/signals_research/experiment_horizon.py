"""
Thread 1 / Experiment A: does predicting further ahead than the live
model's next-5-min-bar target give these same 9 features any real edge?

Per-ticker variant only, fixed 20-day training window (the main grid
already showed window size barely moves the result, so this experiment
isolates ONE new variable -- horizon -- instead of re-running the full
grid again). Same walk-forward retraining, same reversal trade rule, same
6bps round-trip cost as the main backtest, so results are directly
comparable to RESEARCH_LOG.md's per_ticker_20d row (horizon=1 there was
-0.0551%/trade, 32.09% win rate -- included below as horizon=1 for a
same-script apples-to-apples check, not assumed from the other run).

Horizons tested: 1 (baseline), 3, 6, 12 bars = 5/15/30/60 min ahead.
"""
import json
import numpy as np
import pandas as pd

from fetch_data import fetch_and_cache
from feature_variants import build_features_horizon, FEATURE_COLS
from backtest_engine import (
    train_model_from_features, _trading_dates, _simulate_positions,
    _regime_tags, _spy_realized_vol, summarize, RETRAIN_EVERY_DAYS,
)
from signal_model import BUY_THRESHOLD, SELL_THRESHOLD

WINDOW_SIZE_DAYS = 20
HORIZONS = [1, 3, 6, 12]


def run_for_horizon(data: dict, horizon: int, log=print) -> list:
    feat = {tk: build_features_horizon(df, horizon) for tk, df in data.items()}
    spy_vol = _spy_realized_vol(feat["SPY"]) if "SPY" in feat else None

    all_trades = []
    for tk, fdf in feat.items():
        dates = _trading_dates(fdf)
        i = WINDOW_SIZE_DAYS
        while i + RETRAIN_EVERY_DAYS <= len(dates):
            train_start, train_end = dates[i - WINDOW_SIZE_DAYS], dates[i - 1]
            score_start, score_end = dates[i], dates[min(i + RETRAIN_EVERY_DAYS - 1, len(dates) - 1)]

            train_slice = fdf.loc[str(train_start):str(train_end)]
            if len(train_slice) < 60:
                i += RETRAIN_EVERY_DAYS
                continue
            model, acc = train_model_from_features(train_slice)

            score_slice = fdf.loc[str(score_start):str(score_end)]
            if not score_slice.empty:
                X = score_slice[FEATURE_COLS].values
                prob = model.predict_proba(X)[:, 1]
                labels = pd.Series(
                    np.where(prob > BUY_THRESHOLD, "BUY", np.where(prob < SELL_THRESHOLD, "SELL", "HOLD")),
                    index=score_slice.index,
                )
                trades = _simulate_positions(tk, labels, score_slice["close"])
                all_trades.extend(trades)

            i += RETRAIN_EVERY_DAYS
        log(f"  [horizon={horizon}] {tk} done, cumulative trades so far: {len(all_trades)}")

    if spy_vol is not None:
        _regime_tags(all_trades, spy_vol)
    return all_trades


def main():
    data = fetch_and_cache()
    results = {}
    for h in HORIZONS:
        print(f"=== horizon={h} bars ({h*5} min ahead) ===")
        trades = run_for_horizon(data, h)
        by_ticker = {}
        for tk in data.keys():
            by_ticker[tk] = summarize([t for t in trades if t.ticker == tk])
        results[f"horizon_{h}"] = {
            "horizon_bars": h,
            "horizon_minutes": h * 5,
            "overall": summarize(trades),
            "by_ticker": by_ticker,
        }
        print(f"  overall: {results[f'horizon_{h}']['overall']}")

    with open("horizon_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved horizon_experiment_results.json")


if __name__ == "__main__":
    main()
