"""
Day Trader position-sizing backtest.

Answers: given the SAME candidate-selection logic daytrader_scanner.py
actually uses live (ATR% floor, composite score >= min_composite_score,
sector cap, max_positions), is fixed-10%-of-equity per trade the best
sizing scheme, or does something else do better?

Two-phase design so the expensive part (candidate selection + trade
simulation) runs ONCE and every sizing scheme is then just cheap arithmetic
over the same trade log -- selection is independent of sizing, so scheme
comparisons must all see identical trades to be a fair comparison.

Phase 1 -- trade selection + outcome simulation (5y, ~500 tickers):
  For every historical trading day, replay EXACTLY what daytrader_scanner.py
  would have selected that morning (imports compute_dt_scores/apply_sector_cap
  directly from it, not a re-derived copy, so this can't silently drift from
  the live logic) using the real live config (min_atr_pct=2.5, min_composite_score=75,
  max_positions=10). For each selected candidate, simulate the trade outcome
  from that day's actual High/Low (unlike sp500_daytrade_study.py, which only
  kept the engineered features and discarded H/L, this script keeps them
  specifically to simulate stop/target touches, not just the close):
    - Low <= entry*(1-stop_pct)   -> stopped out at -stop_pct   (checked first,
      i.e. conservative -- if both stop and target were touched the same day,
      this assumes the worse outcome, same convention as main.py's existing
      "with_stops" backtest layer)
    - High >= entry*(1+target_pct) -> target hit at +target_pct
    - neither                      -> force-closed at day's actual Close

Phase 2 -- sizing schemes over the same trade log:
  Fixed-% schemes at several levels, score-weighted allocation, and a
  Kelly-fraction scheme derived from the trade log's own empirical win/loss
  stats. Reports CAGR, max drawdown, and the underlying win rate against the
  breakeven win rate the target/stop RATIO itself requires -- sizing cannot
  fix a negative-expectancy payoff ratio, only change how a positive one
  compounds and how a negative one erodes.

Output: daytrader_sizing_results.json next to this script.
"""
import sys, io, json, time
from datetime import datetime

if hasattr(sys.stdout, 'buffer') and (sys.stdout.encoding or '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import yfinance as yf

from daytrader_scanner import (
    load_universe, compute_dt_scores, apply_sector_cap, MAX_PER_SECTOR,
)

HERE = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]
RESULTS_PATH = f"{HERE}/daytrader_sizing_results.json"

# Live config (main.py's "day_trader" defaults / current overrides, 2026-08-07)
PROFIT_TARGET_PCT = 0.25
HARD_STOP_PCT     = 1.0
MIN_ATR_PCT       = 2.5     # DT-VOL gate
MIN_COMPOSITE     = 75.0    # min_composite_score gate
MAX_POSITIONS     = 10
STARTING_EQUITY   = 10_000.0


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def download_all(tickers: list[str]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    batch_size = 40
    n_batches = (len(tickers) + batch_size - 1) // batch_size
    for bi, batch in enumerate(chunked(tickers, batch_size), 1):
        for attempt in range(3):
            try:
                data = yf.download(batch, period="5y", interval="1d", group_by="ticker",
                                    threads=True, auto_adjust=False, progress=False)
                break
            except Exception as exc:
                print(f"  batch {bi}/{n_batches} download error (attempt {attempt+1}): {exc}")
                time.sleep(3)
        else:
            continue
        if len(batch) == 1:
            tk = batch[0]
            if not data.empty:
                out[tk] = data
        else:
            for tk in batch:
                try:
                    sub = data[tk].dropna(how="all")
                    if not sub.empty and len(sub) > 260:
                        out[tk] = sub
                except Exception:
                    continue
        print(f"  batch {bi}/{n_batches} done ({len(out)} tickers so far)")
    return out


def build_ticker_frame(ticker: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """Same no-look-ahead features as daytrader_scanner.compute_dt_features,
    vectorized across the WHOLE history instead of just 'today', plus keeps
    Open/High/Low/Close for outcome simulation (the one thing the original
    sp500_daytrade_study.py panel discarded)."""
    df = df.copy()
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            return None
    df = df[["Open", "High", "Low", "Close"]].astype(float).dropna()
    if len(df) < 40:
        return None

    opens, highs, lows, closes = df["Open"], df["High"], df["Low"], df["Close"]

    tr = pd.concat([
        (highs - lows),
        (highs - closes.shift()).abs(),
        (lows - closes.shift()).abs(),
    ], axis=1).max(axis=1)
    atr14   = tr.rolling(14).mean().shift(1)          # through yesterday only
    atr_pct = atr14 / closes.shift(1) * 100

    gap_pct           = (opens - closes.shift(1)) / closes.shift(1) * 100
    prior_day_ret_pct = (closes.shift(1) - opens.shift(1)) / opens.shift(1) * 100
    ret5d_prior        = closes.shift(1).pct_change(5) * 100

    out = pd.DataFrame({
        "ticker": ticker,
        "open": opens, "high": highs, "low": lows, "close": closes,
        "atr_pct": atr_pct, "gap_pct": gap_pct,
        "prior_day_ret_pct": prior_day_ret_pct, "ret5d_prior": ret5d_prior,
    }, index=df.index)
    out = out.dropna(subset=["atr_pct", "gap_pct", "prior_day_ret_pct"])
    out["date"] = out.index.astype(str)
    return out.reset_index(drop=True)


def simulate_trade(row) -> float:
    """Stop-checked-first outcome (conservative), matching main.py's existing
    with_stops convention."""
    entry = row["open"]
    if entry <= 0:
        return 0.0
    stop_px   = entry * (1 - HARD_STOP_PCT / 100)
    target_px = entry * (1 + PROFIT_TARGET_PCT / 100)
    if row["low"] <= stop_px:
        return -HARD_STOP_PCT
    if row["high"] >= target_px:
        return PROFIT_TARGET_PCT
    return (row["close"] - entry) / entry * 100


def simulate_trade_optimistic(row) -> float:
    """Same as simulate_trade but checks TARGET first -- the opposite
    ordering assumption. Daily bars can't reveal which was actually touched
    first intraday, so the true outcome lies somewhere between this and the
    conservative stop-first version; reporting both gives an honest range
    instead of false precision from one arbitrary ordering choice."""
    entry = row["open"]
    if entry <= 0:
        return 0.0
    stop_px   = entry * (1 - HARD_STOP_PCT / 100)
    target_px = entry * (1 + PROFIT_TARGET_PCT / 100)
    if row["high"] >= target_px:
        return PROFIT_TARGET_PCT
    if row["low"] <= stop_px:
        return -HARD_STOP_PCT
    return (row["close"] - entry) / entry * 100


def run_phase1() -> list[dict]:
    print("Phase 1: candidate selection + trade simulation")
    tickers, sector_map = load_universe()
    print(f"  universe: {len(tickers)} tickers")

    raw = download_all(tickers)
    print(f"  downloaded {len(raw)}/{len(tickers)} tickers")

    frames = []
    for tk, df in raw.items():
        f = build_ticker_frame(tk, df)
        if f is not None and len(f) > 50:
            frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    print(f"  panel: {len(panel):,} ticker-days")

    panel = panel[panel["atr_pct"] >= MIN_ATR_PCT].copy()
    print(f"  after ATR floor ({MIN_ATR_PCT}%): {len(panel):,} ticker-days")

    trade_log: list[dict] = []
    dates = sorted(panel["date"].unique())
    for d in dates:
        day_rows = panel[panel["date"] == d]
        candidates = day_rows.to_dict("records")
        if not candidates:
            continue
        for c in candidates:   # pandas NaN -> None, matching compute_dt_features' own convention
            if pd.isna(c.get("ret5d_prior")):
                c["ret5d_prior"] = None
        compute_dt_scores(candidates)   # in-place, same formula as live scanner
        candidates = [c for c in candidates if c["composite_score"] >= MIN_COMPOSITE]
        if not candidates:
            continue
        candidates.sort(key=lambda c: c["composite_score"], reverse=True)
        capped = apply_sector_cap(candidates, sector_map, MAX_PER_SECTOR)[:MAX_POSITIONS]
        for c in capped:
            trade_log.append({
                "date": d, "ticker": c["ticker"], "score": c["composite_score"],
                "atr_pct": c["atr_pct"], "gap_pct": c["gap_pct"],
                "trade_ret_pct": simulate_trade(c),
                "trade_ret_pct_optimistic": simulate_trade_optimistic(c),
            })
    print(f"  {len(trade_log):,} simulated trades across {len(dates)} trading days")
    return trade_log


# ── Phase 2: sizing schemes over the fixed trade log ─────────────────────────

def sim_fixed_pct(trade_log: list[dict], pct: float, starting_equity: float = STARTING_EQUITY) -> dict:
    equity = starting_equity
    curve = [equity]
    by_date: dict[str, list[dict]] = {}
    for t in trade_log:
        by_date.setdefault(t["date"], []).append(t)
    for d in sorted(by_date):
        day_start_equity = equity
        day_pnl = 0.0
        for t in by_date[d]:
            size = day_start_equity * (pct / 100)
            day_pnl += size * (t["trade_ret_pct"] / 100)
        equity += day_pnl
        curve.append(equity)
    return _metrics(curve, trade_log, starting_equity)


def sim_score_weighted(trade_log: list[dict], total_alloc_pct: float,
                        starting_equity: float = STARTING_EQUITY) -> dict:
    """Total capital deployed per day = total_alloc_pct% of equity, split
    across that day's selected candidates PROPORTIONAL to composite_score
    (higher-conviction names get a bigger slice) instead of equal shares."""
    equity = starting_equity
    curve = [equity]
    by_date: dict[str, list[dict]] = {}
    for t in trade_log:
        by_date.setdefault(t["date"], []).append(t)
    for d in sorted(by_date):
        day_start_equity = equity
        day_trades = by_date[d]
        total_score = sum(t["score"] for t in day_trades) or 1.0
        day_pnl = 0.0
        for t in day_trades:
            weight = t["score"] / total_score
            size = day_start_equity * (total_alloc_pct / 100) * weight
            day_pnl += size * (t["trade_ret_pct"] / 100)
        equity += day_pnl
        curve.append(equity)
    return _metrics(curve, trade_log, starting_equity)


def sim_kelly(trade_log: list[dict], kelly_fraction: float = 0.5,
              starting_equity: float = STARTING_EQUITY) -> dict:
    """Half-Kelly (or kelly_fraction-Kelly) sized off the trade log's OWN
    empirical win/loss stats -- a simple two-outcome approximation (win = hit
    target, everything else pooled as 'loss' at its own average size)."""
    wins   = [t["trade_ret_pct"] for t in trade_log if t["trade_ret_pct"] > 0]
    losses = [t["trade_ret_pct"] for t in trade_log if t["trade_ret_pct"] <= 0]
    p = len(wins) / len(trade_log) if trade_log else 0
    avg_win  = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = abs(sum(losses) / len(losses)) if losses else 1e-9
    b = avg_win / avg_loss if avg_loss > 0 else 0.0
    q = 1 - p
    f_star = (p * b - q) / b if b > 0 else 0.0
    f_star = max(0.0, min(f_star, 1.0))   # clip to [0, 100%]
    pct = f_star * kelly_fraction * 100

    equity = starting_equity
    curve = [equity]
    by_date: dict[str, list[dict]] = {}
    for t in trade_log:
        by_date.setdefault(t["date"], []).append(t)
    for d in sorted(by_date):
        day_start_equity = equity
        day_pnl = 0.0
        for t in by_date[d]:
            size = day_start_equity * (pct / 100)
            day_pnl += size * (t["trade_ret_pct"] / 100)
        equity += day_pnl
        curve.append(equity)
    m = _metrics(curve, trade_log, starting_equity)
    m["kelly_full_pct"] = round(f_star * 100, 2)
    m["kelly_used_pct"] = round(pct, 2)
    return m


def _metrics(curve: list[float], trade_log: list[dict], starting_equity: float) -> dict:
    arr = np.array(curve)
    total_return_pct = (arr[-1] / arr[0] - 1) * 100
    n_days = len(arr) - 1
    years = n_days / 252 if n_days > 0 else 1
    cagr = ((arr[-1] / arr[0]) ** (1 / years) - 1) * 100 if arr[-1] > 0 and years > 0 else -100.0
    running_max = np.maximum.accumulate(arr)
    drawdown = (arr - running_max) / running_max * 100
    max_dd = float(drawdown.min())
    daily_rets = np.diff(arr) / arr[:-1]
    sharpe_like = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)) if np.std(daily_rets) > 0 else 0.0
    wins = sum(1 for t in trade_log if t["trade_ret_pct"] > 0)
    hit_stop = sum(1 for t in trade_log if t["trade_ret_pct"] == -HARD_STOP_PCT)
    hit_target = sum(1 for t in trade_log if t["trade_ret_pct"] == PROFIT_TARGET_PCT)
    neither = len(trade_log) - hit_stop - hit_target
    ruined = bool(arr.min() <= starting_equity * 0.5)   # ever drew down 50%+
    return {
        "ending_equity": round(float(arr[-1]), 2),
        "total_return_pct": round(float(total_return_pct), 1),
        "cagr_pct": round(float(cagr), 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_like": round(float(sharpe_like), 2),
        "n_trades": len(trade_log),
        "win_rate_pct": round(wins / len(trade_log) * 100, 1) if trade_log else None,
        "hit_target_pct": round(hit_target / len(trade_log) * 100, 1) if trade_log else None,
        "hit_stop_pct": round(hit_stop / len(trade_log) * 100, 1) if trade_log else None,
        "neither_close_based_pct": round(neither / len(trade_log) * 100, 1) if trade_log else None,
        "ever_drew_down_50pct": ruined,
    }


def breakeven_analysis(trade_log: list[dict]) -> dict:
    """The payoff ratio (target/stop) sets a breakeven win rate BEFORE any
    sizing is even considered: p*target - (1-p)*stop = 0 => p = stop/(target+stop).
    Reports both the conservative (stop-checked-first) and optimistic
    (target-checked-first) orderings -- the true number is somewhere between,
    since daily bars can't reveal actual intraday sequencing."""
    b = PROFIT_TARGET_PCT / HARD_STOP_PCT
    breakeven_p = HARD_STOP_PCT / (PROFIT_TARGET_PCT + HARD_STOP_PCT)

    def _stats(key):
        wins = sum(1 for t in trade_log if t[key] > 0)
        p = wins / len(trade_log) if trade_log else 0
        avg_ret = sum(t[key] for t in trade_log) / len(trade_log) if trade_log else 0
        return round(p * 100, 1), round(avg_ret, 4)

    cons_wr, cons_avg = _stats("trade_ret_pct")
    opt_wr, opt_avg   = _stats("trade_ret_pct_optimistic")

    return {
        "profit_target_pct": PROFIT_TARGET_PCT,
        "hard_stop_pct": HARD_STOP_PCT,
        "reward_risk_ratio": round(b, 3),
        "breakeven_win_rate_pct": round(breakeven_p * 100, 1),
        "conservative_stop_first": {"win_rate_pct": cons_wr, "avg_trade_ret_pct": cons_avg,
                                     "edge_is_positive": cons_avg > 0},
        "optimistic_target_first": {"win_rate_pct": opt_wr, "avg_trade_ret_pct": opt_avg,
                                     "edge_is_positive": opt_avg > 0},
    }


def main():
    t0 = time.monotonic()
    trade_log = run_phase1()
    if not trade_log:
        print("No trades simulated -- aborting.")
        return

    print("\nPhase 2: sizing scheme comparison")
    schemes = {
        "fixed_5pct":         sim_fixed_pct(trade_log, 5.0),
        "fixed_10pct_current": sim_fixed_pct(trade_log, 10.0),
        "fixed_15pct":        sim_fixed_pct(trade_log, 15.0),
        "fixed_20pct":        sim_fixed_pct(trade_log, 20.0),
        "score_weighted_50pct_total":  sim_score_weighted(trade_log, 50.0),
        "score_weighted_100pct_total": sim_score_weighted(trade_log, 100.0),
        "half_kelly":  sim_kelly(trade_log, 0.5),
        "quarter_kelly": sim_kelly(trade_log, 0.25),
        "full_kelly":  sim_kelly(trade_log, 1.0),
    }
    for name, m in schemes.items():
        print(f"  {name:28s} CAGR={m['cagr_pct']:+8.2f}%  maxDD={m['max_drawdown_pct']:7.2f}%  "
              f"end=${m['ending_equity']:,.0f}  sharpe~{m['sharpe_like']:.2f}")

    be = breakeven_analysis(trade_log)
    print(f"\nBreakeven analysis: need {be['breakeven_win_rate_pct']}% win rate to break even "
          f"at {PROFIT_TARGET_PCT}%/{HARD_STOP_PCT}% target/stop")
    print(f"  conservative (stop-first): win={be['conservative_stop_first']['win_rate_pct']}% "
          f"avg_ret={be['conservative_stop_first']['avg_trade_ret_pct']}% "
          f"edge_positive={be['conservative_stop_first']['edge_is_positive']}")
    print(f"  optimistic (target-first): win={be['optimistic_target_first']['win_rate_pct']}% "
          f"avg_ret={be['optimistic_target_first']['avg_trade_ret_pct']}% "
          f"edge_positive={be['optimistic_target_first']['edge_is_positive']}")

    results = {
        "computed_at": datetime.now().astimezone().isoformat(),
        "config": {
            "profit_target_pct": PROFIT_TARGET_PCT, "hard_stop_pct": HARD_STOP_PCT,
            "min_atr_pct": MIN_ATR_PCT, "min_composite_score": MIN_COMPOSITE,
            "max_positions": MAX_POSITIONS, "starting_equity": STARTING_EQUITY,
        },
        "n_trades": len(trade_log),
        "breakeven_analysis": be,
        "schemes": schemes,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}  (total runtime {time.monotonic()-t0:.0f}s)")


if __name__ == "__main__":
    main()
