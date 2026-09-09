"""
Walk-forward simulation of the Signals-tab model as a trading rule.

Methodology (every approximation stated explicitly, per this account's
research convention):

- "Trading day" = a calendar date with at least one real 5-min bar for the
  ticker in question (weekends/holidays fall out naturally).
- Walk-forward: retrain every RETRAIN_EVERY_DAYS (5) trading days, using
  the trailing WINDOW_SIZE_DAYS of bars as the training set, then use that
  frozen model to score every bar in the following week until the next
  retrain point. This assumes a periodically-maintained deployment --
  notably BETTER-maintained than what's live today, which trains once at
  startup and never auto-retrains except a manual /retrain/{ticker} call.
  That's a real, stated divergence from current production, not an
  oversight: it answers "does this signal have edge under reasonable
  upkeep," which is the more useful question before deciding whether to
  build anything further.
- "shared" variant: one model, retrained each cycle from a single
  reference ticker's (AAPL, first alphabetically) own trailing window,
  applied to score all 4 tickers that week -- this is the closest a
  backtest can get to faithfully replicating main.py's actual "one global
  model reused across every ticker" architecture (main.py:892-899,
  9683-9696), since the live system's own retrain trigger (whichever
  ticker a human happens to call /retrain on) isn't something a backtest
  can honestly simulate -- AAPL-as-reference is a stated, arbitrary but
  consistent choice.
- "per_ticker" variant: each ticker gets its own independently retrained
  model on its own trailing window.
- Trade rule: reversal-style position tracking (user decision, 2026-09-07).
  While flat or short, BUY closes any short and opens long. While flat or
  long, SELL closes any long and opens short. HOLD never changes position.
  Any position still open at the end of the ticker's data is force-closed
  at the last available bar (flagged as forced_close=True in the trade
  record) so no P&L silently goes unaccounted.
- Transaction cost: ROUND_TRIP_COST_PCT (0.06% = 6bps) subtracted from
  every closed trade's raw return. This is a documented approximation --
  no live execution exists yet to measure real slippage/commission from --
  chosen as a conservative-but-not-punitive estimate for spread+commission
  on four of the most liquid large-cap names traded, roughly in line with
  this account's other backtests' stated cost assumptions.
- Regime split: SPY's own rolling 20-trading-day realized volatility
  (std of 5-min log returns) is computed across the whole window; each
  closed trade is tagged HIGH_VOL or LOW_VOL by whether SPY's realized
  vol at that trade's close time was above or below the whole window's
  median. This is an objective, data-driven regime split rather than an
  assumed calendar date/event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from signal_model import build_features, FEATURE_COLS, BUY_THRESHOLD, SELL_THRESHOLD

RETRAIN_EVERY_DAYS = 5
ROUND_TRIP_COST_PCT = 0.0006
REFERENCE_TICKER = "AAPL"


@dataclass
class BacktestConfig:
    variant: str          # "shared" or "per_ticker"
    window_size_days: int  # 5, 20, or 60


@dataclass
class Trade:
    ticker: str
    side: str            # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    raw_ret_pct: float
    net_ret_pct: float
    forced_close: bool
    regime: str = ""


def _trading_dates(df: pd.DataFrame) -> list:
    return sorted(set(idx.date() for idx in df.index))


def _label_series(model, feat_df: pd.DataFrame) -> pd.Series:
    """Vectorized equivalent of calling predict_label() at every row --
    safe because build_features()'s rolling windows are purely backward-
    looking (no lookahead), so scoring every row at once with the frozen
    model gives identical results to scoring bar-by-bar as they arrive."""
    X = feat_df[FEATURE_COLS].values
    prob = model.predict_proba(X)[:, 1]
    labels = np.where(prob > BUY_THRESHOLD, "BUY", np.where(prob < SELL_THRESHOLD, "SELL", "HOLD"))
    return pd.Series(labels, index=feat_df.index)


def _simulate_positions(ticker: str, labels: pd.Series, closes: pd.Series) -> list[Trade]:
    trades: list[Trade] = []
    position = None  # None | "LONG" | "SHORT"
    entry_time = entry_price = None

    for t, label in labels.items():
        px = closes.loc[t]
        if label == "BUY":
            if position == "SHORT":
                raw = (entry_price - px) / entry_price
                trades.append(Trade(ticker, "SHORT", entry_time, t, entry_price, px,
                                     raw * 100, (raw - ROUND_TRIP_COST_PCT) * 100, False))
                position = None
            if position is None:
                position, entry_time, entry_price = "LONG", t, px
        elif label == "SELL":
            if position == "LONG":
                raw = (px - entry_price) / entry_price
                trades.append(Trade(ticker, "LONG", entry_time, t, entry_price, px,
                                     raw * 100, (raw - ROUND_TRIP_COST_PCT) * 100, False))
                position = None
            if position is None:
                position, entry_time, entry_price = "SHORT", t, px
        # HOLD: no-op

    if position is not None:
        last_t = labels.index[-1]
        px = closes.loc[last_t]
        raw = (px - entry_price) / entry_price if position == "LONG" else (entry_price - px) / entry_price
        trades.append(Trade(ticker, position, entry_time, last_t, entry_price, px,
                             raw * 100, (raw - ROUND_TRIP_COST_PCT) * 100, True))

    return trades


def _regime_tags(trades: list[Trade], spy_vol: pd.Series) -> None:
    if spy_vol.empty:
        for tr in trades:
            tr.regime = "UNKNOWN"
        return
    median_vol = spy_vol.median()
    vol_at = spy_vol.reindex(spy_vol.index.union([t.exit_time for t in trades])).sort_index().ffill()
    for tr in trades:
        v = vol_at.asof(tr.exit_time)
        tr.regime = "HIGH_VOL" if (pd.notna(v) and v >= median_vol) else "LOW_VOL"


def _spy_realized_vol(spy_feat: pd.DataFrame) -> pd.Series:
    ret = np.log(spy_feat["close"]).diff()
    return ret.rolling(20 * 78).std()  # ~78 5-min bars/day incl. extended hours, 20 trading days


def run_walkforward(data: dict[str, pd.DataFrame], config: BacktestConfig,
                     log=print) -> list[Trade]:
    feat = {tk: build_features(df) for tk, df in data.items()}
    spy_vol = _spy_realized_vol(feat["SPY"]) if "SPY" in feat else pd.Series(dtype=float)

    all_trades: list[Trade] = []

    if config.variant == "shared":
        ref = feat[REFERENCE_TICKER]
        dates = _trading_dates(ref)
        tickers_to_score = list(feat.keys())

        i = config.window_size_days
        while i + RETRAIN_EVERY_DAYS <= len(dates):
            train_start, train_end = dates[i - config.window_size_days], dates[i - 1]
            score_start, score_end = dates[i], dates[min(i + RETRAIN_EVERY_DAYS - 1, len(dates) - 1)]

            train_slice = ref.loc[str(train_start):str(train_end)]
            if len(train_slice) < 60:
                i += RETRAIN_EVERY_DAYS
                continue
            model, acc = train_model_from_features(train_slice)
            log(f"[shared/{config.window_size_days}d] retrain {train_start}..{train_end} "
                f"(n={len(train_slice)}, acc={acc:.3f}) -> score {score_start}..{score_end}")

            for tk in tickers_to_score:
                score_slice = feat[tk].loc[str(score_start):str(score_end)]
                if score_slice.empty:
                    continue
                labels = _label_series(model, score_slice)
                trades = _simulate_positions(tk, labels, score_slice["close"])
                all_trades.extend(trades)

            i += RETRAIN_EVERY_DAYS

    elif config.variant == "per_ticker":
        for tk, fdf in feat.items():
            dates = _trading_dates(fdf)
            i = config.window_size_days
            while i + RETRAIN_EVERY_DAYS <= len(dates):
                train_start, train_end = dates[i - config.window_size_days], dates[i - 1]
                score_start, score_end = dates[i], dates[min(i + RETRAIN_EVERY_DAYS - 1, len(dates) - 1)]

                train_slice = fdf.loc[str(train_start):str(train_end)]
                if len(train_slice) < 60:
                    i += RETRAIN_EVERY_DAYS
                    continue
                model, acc = train_model_from_features(train_slice)
                log(f"[per_ticker/{tk}/{config.window_size_days}d] retrain "
                    f"{train_start}..{train_end} (n={len(train_slice)}, acc={acc:.3f})")

                score_slice = fdf.loc[str(score_start):str(score_end)]
                if not score_slice.empty:
                    labels = _label_series(model, score_slice)
                    trades = _simulate_positions(tk, labels, score_slice["close"])
                    all_trades.extend(trades)

                i += RETRAIN_EVERY_DAYS
    else:
        raise ValueError(f"unknown variant {config.variant}")

    _regime_tags(all_trades, spy_vol)
    return all_trades


def train_model_from_features(feat_slice: pd.DataFrame):
    """Same 3-fold-TimeSeriesSplit-then-keep-last-fold procedure as
    signal_model.train_model(), but operating on an already-feature-built
    slice (avoids recomputing rolling features on every retrain -- they're
    computed once per ticker up front in run_walkforward)."""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier

    X, y = feat_slice[FEATURE_COLS].values, feat_slice["target"].values
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
    return model, float(np.mean(accs)) if accs else None


def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"n_trades": 0}
    rets = np.array([t.net_ret_pct for t in trades])
    wins = (rets > 0).sum()
    equity = np.cumsum(rets)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    return {
        "n_trades": len(trades),
        "win_rate_pct": round(100 * wins / len(trades), 2),
        "avg_ret_pct": round(float(rets.mean()), 4),
        "total_ret_sum_pct": round(float(rets.sum()), 2),
        "best_pct": round(float(rets.max()), 4),
        "worst_pct": round(float(rets.min()), 4),
        "max_drawdown_pct": round(float(drawdown.min()), 2),
    }
