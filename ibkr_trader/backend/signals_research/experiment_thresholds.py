"""
Thread 1 / Experiment B: does trading only on higher-conviction signals
fix anything? The live model's BUY_THRESHOLD=0.55/SELL_THRESHOLD=0.45
are close to the 0.5 midpoint -- the main backtest's extreme turnover
(43k-92k trades across 4 tickers over 2.5 years) is consistent with the
reversal rule firing on the model's probability wobbling across those
loose thresholds, not on any real directional persistence. Widening the
gap should cut turnover a lot; the open question is whether the trades
that DO clear a stricter bar are actually better, or whether the signal
has no edge at any conviction level.

Isolates ONE variable (threshold width) at the ORIGINAL 1-bar horizon
and 20-day window -- same reasoning as experiment_horizon.py for why
those are held fixed rather than re-grid-searched.

Threshold pairs tested: (0.55, 0.45) baseline, (0.60, 0.40), (0.65, 0.35),
(0.70, 0.30).
"""
import json
import numpy as np
import pandas as pd

from fetch_data import fetch_and_cache
from signal_model import build_features, FEATURE_COLS
from backtest_engine import (
    train_model_from_features, _trading_dates, _simulate_positions,
    _regime_tags, _spy_realized_vol, summarize, RETRAIN_EVERY_DAYS,
)

WINDOW_SIZE_DAYS = 20
THRESHOLD_PAIRS = [(0.55, 0.45), (0.60, 0.40), (0.65, 0.35), (0.70, 0.30)]


def run_for_thresholds(data: dict, buy_th: float, sell_th: float, log=print) -> list:
    feat = {tk: build_features(df) for tk, df in data.items()}
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
                    np.where(prob > buy_th, "BUY", np.where(prob < sell_th, "SELL", "HOLD")),
                    index=score_slice.index,
                )
                trades = _simulate_positions(tk, labels, score_slice["close"])
                all_trades.extend(trades)

            i += RETRAIN_EVERY_DAYS
        log(f"  [buy>{buy_th}/sell<{sell_th}] {tk} done, cumulative trades: {len(all_trades)}")

    if spy_vol is not None:
        _regime_tags(all_trades, spy_vol)
    return all_trades


def main():
    data = fetch_and_cache()
    results = {}
    for buy_th, sell_th in THRESHOLD_PAIRS:
        key = f"buy{buy_th}_sell{sell_th}"
        print(f"=== {key} ===")
        trades = run_for_thresholds(data, buy_th, sell_th)
        by_ticker = {tk: summarize([t for t in trades if t.ticker == tk]) for tk in data.keys()}
        results[key] = {
            "buy_threshold": buy_th, "sell_threshold": sell_th,
            "overall": summarize(trades),
            "by_ticker": by_ticker,
        }
        print(f"  overall: {results[key]['overall']}")

    with open("threshold_experiment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved threshold_experiment_results.json")


if __name__ == "__main__":
    main()
