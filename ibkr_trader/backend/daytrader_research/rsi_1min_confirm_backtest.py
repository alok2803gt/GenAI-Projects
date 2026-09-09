"""
Tests whether adding an RSI(14)-on-1-minute-bars condition to Day Trader's
real, LIVE confirmation gate (0.35% price move + volume >= running median,
within the first 60min) improves on the gate alone -- CEO request
2026-09-05 ("I think ATR% alone is not enough, we should also add RSI on
1 minute bar").

Does NOT touch day_trader_agent.py or any live config. Reuses the SAME
real 820-candidate (ticker,date) 1-minute-bar dataset already fetched via
real IBKR data for the original confirmation-gate validation
(minute_bars.json, 82 trading days, 100 unique tickers, real ATR%-selected
Day Trader candidates -- see RESEARCH_LOG.md iteration 1/2). Reusing it
here means the RSI test runs on the EXACT same opportunity set as the
account's own most-validated baseline (52.7% win rate, +0.154%/trade,
n=581, confirmed in strategy_comparison.py) -- differences in results
reflect the RSI condition alone, not a different sample.

Relevant prior work, why this is a NEW question and not a rerun:
  - indicator_combo_backtest.py already tested RSI(14) in 50-70, but on
    DAILY bars, for a totally different trade shape (buy-the-open/sell-
    the-close) -- not the live intraday confirmation-gate mechanism at
    all. Its own follow-up (target_combo_multiregime_backtest.py) found
    "adding RSI/ADX/ROC on top changed nothing" for THAT daily-bar
    question. Neither result speaks to RSI computed on 1-MINUTE bars at
    the exact moment of intraday confirmation, which is what's being
    asked here.
  - daytrader_atr_confirm_backtest.py tested an ATR-RELATIVE confirmation
    threshold (still a price-move-size question) and found it did not
    beat the flat 0.35%. Also a different question from RSI (a momentum-
    state indicator, not a move-size threshold).

Methodology:
  1. Baseline: EXACT replica of simulate_confirm_trail from
     daytrader_intraday_backtest.py (0.35% confirm, 60min window, volume
     >= running median, 0.3% trailing stop -- current live config) run
     fresh on this dataset for a same-sample reference point.
  2. RSI(14) computed on 1-minute CLOSES via Wilder smoothing (ewm
     alpha=1/14), same formula already used in indicator_combo_backtest.py
     for methodological consistency. Evaluated AT the confirming minute
     (using only bars up to and including that minute -- no lookahead).
  3. Real gap stated plainly: RSI(14) needs real lookback. A confirmation
     that fires in the first ~14 minutes has an unstable/undefined RSI --
     these are tracked and reported separately (n_no_rsi_data), not
     silently dropped into either bucket.
  4. Tests a real threshold grid, not one setting: RSI >= 50, RSI >= 60,
     RSI >= 70, RSI in [50,70] (mirrors the daily-bar test's own band,
     applied intraday this time), RSI < 80 (pure overbought exclusion --
     the opposite hypothesis, momentum strategies sometimes benefit from
     NOT chasing an already-exhausted spike).
  5. Every variant uses the SAME 0.3% trailing stop as the baseline --
     isolates the RSI condition as the only variable, per this account's
     own established one-variable-at-a-time discipline.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
BARS_PATH = HERE / "minute_bars.json"

CONFIRM_PCT = 0.35
CONFIRM_WINDOW_MIN = 60
TRAIL_PCT = 0.3          # current live trailing_stop_pct
RSI_PERIOD = 14

RSI_VARIANTS = {
    "RSI_ge_50": lambda r: r is not None and r >= 50,
    "RSI_ge_60": lambda r: r is not None and r >= 60,
    "RSI_ge_70": lambda r: r is not None and r >= 70,
    "RSI_50_70": lambda r: r is not None and 50 <= r <= 70,
    "RSI_lt_80_overbought_excl": lambda r: r is not None and r < 80,
}


def rsi_series(closes: list[float], period: int = RSI_PERIOD) -> list[float | None]:
    """Wilder-smoothed RSI, same formula as indicator_combo_backtest.py's
    daily-bar version, applied here to 1-min closes. Returns None for
    indices before a real period-length lookback exists (no fabricated
    early-index values)."""
    n = len(closes)
    out: list[float | None] = [None] * n
    if n < 2:
        return out
    gains = [0.0] * n
    losses = [0.0] * n
    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gains[i] = max(delta, 0.0)
        losses[i] = max(-delta, 0.0)
    avg_gain = avg_loss = None
    alpha = 1 / period
    for i in range(1, n):
        if avg_gain is None:
            if i < period:
                continue
            avg_gain = sum(gains[1:i + 1]) / i
            avg_loss = sum(losses[1:i + 1]) / i
        else:
            avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
            avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        if avg_gain is not None:
            rs = avg_gain / avg_loss if avg_loss > 1e-9 else float("inf")
            out[i] = 100 - 100 / (1 + rs) if rs != float("inf") else 100.0
    return out


def simulate_confirm_trail(bars: list[dict], rsi_ok) -> dict | None:
    """Real intraday sequencing, same mechanics as
    daytrader_intraday_backtest.py's simulate_confirm_trail, plus an
    optional RSI gate at the confirming minute. rsi_ok: callable(rsi_or_None)
    -> bool, or None to skip the RSI check entirely (pure baseline)."""
    day_open = bars[0]["open"]
    confirm_px = day_open * (1 + CONFIRM_PCT / 100)
    closes = [b["close"] for b in bars]
    rsis = rsi_series(closes)

    vols_so_far: list[float] = []
    entry_idx = None
    entry_px = None
    price_vol_qualified_count = 0
    real_rsi_evaluated_count = 0
    for i, b in enumerate(bars[:CONFIRM_WINDOW_MIN]):
        vols_so_far.append(b["volume"])
        median_vol = sorted(vols_so_far)[len(vols_so_far) // 2]
        if b["close"] >= confirm_px and b["volume"] >= median_vol:
            price_vol_qualified_count += 1
            r = rsis[i]
            if rsi_ok is not None:
                if r is None:
                    continue   # not enough lookback yet -- keep scanning for a later confirming minute
                real_rsi_evaluated_count += 1
                if not rsi_ok(r):
                    continue   # confirmed on price+volume but RSI condition failed -- keep scanning
            entry_idx, entry_px = i, b["close"]
            break

    if entry_idx is None:
        # Only worth flagging as a "data gap" case if price+volume qualified at
        # least once but RSI was NEVER real (every qualifying minute was too
        # early for a real RSI(14)) -- if a real RSI was evaluated and simply
        # failed the threshold, that's a normal, correctly-rejected non-trade,
        # not a data gap.
        no_rsi_data_only = (price_vol_qualified_count > 0 and real_rsi_evaluated_count == 0
                             and rsi_ok is not None)
        return {"no_rsi_data_only": True} if no_rsi_data_only else None

    running_high = entry_px
    for b in bars[entry_idx + 1:]:
        running_high = max(running_high, b["high"])
        stop_px = running_high * (1 - TRAIL_PCT / 100)
        if b["low"] <= stop_px:
            exit_px = stop_px
            return {"ret_pct": (exit_px / entry_px - 1) * 100, "outcome": "trailed_out"}
    exit_px = bars[-1]["close"]
    return {"ret_pct": (exit_px / entry_px - 1) * 100, "outcome": "eod_close"}


OVERSOLD_THRESHOLDS = [20.0, 25.0, 30.0, 35.0]


def simulate_oversold_bounce(bars: list[dict], rsi_threshold: float) -> dict | None:
    """A DIFFERENT entry mechanism, not a filter on the existing one --
    CEO follow-up 2026-09-05: "try RSI under 30 on 1 minute bar + ATR% +
    volume". Enter LONG the first minute RSI(1min,14) drops below
    rsi_threshold (oversold, betting on a bounce) AND that minute's volume
    is >= the running median (same real-interest confirmation convention
    used everywhere else in this account, not a new invented rule).

    ATR% is NOT separately simulated per-minute here: every candidate in
    this dataset was selected by the live scanner's real ATR%-floor gate
    at the time, so it's already satisfied by construction for every row
    -- there's nothing new to test at the simulation level for that piece.

    Same 60min window and 0.3% trailing-stop exit as every other variant
    in this file, for a fair, one-variable-at-a-time comparison. This is a
    genuinely different bet than the baseline/RSI-filter variants above:
    those buy STRENGTH (price already up 0.35%); this buys WEAKNESS
    (price dipped, RSI oversold), betting on reversion -- much closer in
    spirit to the VWAP mean-reversion archetype strategy_comparison.py
    already tested on this exact same candidate population and found to
    be a clean loser (-0.089%/trade, worst win rate of the 4 archetypes
    tested) for a stated, specific reason: these candidates are selected
    BECAUSE they're already moving/trending, so betting on reversion
    fights the very quality they were chosen for.
    """
    closes = [b["close"] for b in bars]
    rsis = rsi_series(closes)
    vols_so_far: list[float] = []
    entry_idx = entry_px = None
    for i, b in enumerate(bars[:CONFIRM_WINDOW_MIN]):
        vols_so_far.append(b["volume"])
        median_vol = sorted(vols_so_far)[len(vols_so_far) // 2]
        r = rsis[i]
        if r is not None and r < rsi_threshold and b["volume"] >= median_vol:
            entry_idx, entry_px = i, b["close"]
            break
    if entry_idx is None:
        return None

    running_high = entry_px
    for b in bars[entry_idx + 1:]:
        running_high = max(running_high, b["high"])
        stop_px = running_high * (1 - TRAIL_PCT / 100)
        if b["low"] <= stop_px:
            return {"ret_pct": (stop_px / entry_px - 1) * 100, "outcome": "trailed_out"}
    exit_px = bars[-1]["close"]
    return {"ret_pct": (exit_px / entry_px - 1) * 100, "outcome": "eod_close"}


def _agg(results: list[dict]) -> dict:
    if not results:
        return {"n": 0}
    rets = [r["ret_pct"] for r in results]
    wins = [r for r in rets if r > 0]
    return {
        "n": len(results),
        "win_rate_pct": round(len(wins) / len(results) * 100, 1),
        "avg_ret_pct": round(sum(rets) / len(results), 4),
        "median_ret_pct": round(sorted(rets)[len(rets) // 2], 4),
        "total_ret_sum_pct": round(sum(rets), 2),
        "best_pct": round(max(rets), 2),
        "worst_pct": round(min(rets), 2),
    }


def main():
    with open(BARS_PATH) as f:
        bar_data = json.load(f)
    print(f"Loaded {len(bar_data)} real (ticker,date) minute-bar series from {BARS_PATH}")

    variants = {"BASELINE_no_rsi": None, **RSI_VARIANTS}
    results: dict[str, list] = {name: [] for name in variants}
    no_data_only_counts: dict[str, int] = {name: 0 for name in variants}
    n_usable = 0

    for key, bars in bar_data.items():
        if not bars or len(bars) < 10:
            continue
        n_usable += 1
        for name, rsi_ok in variants.items():
            r = simulate_confirm_trail(bars, rsi_ok)
            if r is None:
                continue
            if r.get("no_rsi_data_only"):
                no_data_only_counts[name] += 1
                continue
            results[name].append(r)

    print(f"\nUsable candidates (>=10 real bars): {n_usable}\n")
    aggs = {}
    for name in variants:
        agg = _agg(results[name])
        agg["confirm_rate_pct"] = round(agg.get("n", 0) / n_usable * 100, 1) if n_usable else None
        agg["never_confirmed_rsi_data_gap_only"] = no_data_only_counts[name]
        aggs[name] = agg
        print(f"{name:28s} n={agg.get('n',0):4d}  win={agg.get('win_rate_pct','-'):>5}%  "
              f"avg={agg.get('avg_ret_pct','-'):>8}%  median={agg.get('median_ret_pct','-'):>7}%  "
              f"total={agg.get('total_ret_sum_pct','-'):>8}%  "
              f"confirm_rate={agg['confirm_rate_pct']}%  "
              f"rsi_data_gap_only={no_data_only_counts[name]}")

    # ── Oversold-bounce entry (CEO follow-up 2026-09-05): a DIFFERENT entry
    # mechanism, not a filter layered on the existing one -- see
    # simulate_oversold_bounce's docstring for why this is philosophically
    # closer to VWAP mean-reversion than to the momentum-confirmation gate.
    print(f"\n--- Oversold-bounce entry: RSI(1min,14) < threshold + volume>=median "
          f"(replaces the price-up-0.35% trigger entirely) ---\n")
    bounce_results: dict[float, list] = {t: [] for t in OVERSOLD_THRESHOLDS}
    for key, bars in bar_data.items():
        if not bars or len(bars) < 10:
            continue
        for t in OVERSOLD_THRESHOLDS:
            r = simulate_oversold_bounce(bars, t)
            if r is not None:
                bounce_results[t].append(r)

    bounce_aggs = {}
    for t in OVERSOLD_THRESHOLDS:
        agg = _agg(bounce_results[t])
        agg["confirm_rate_pct"] = round(agg.get("n", 0) / n_usable * 100, 1) if n_usable else None
        bounce_aggs[t] = agg
        label = f"RSI_lt_{t:.0f}_oversold_bounce"
        print(f"{label:28s} n={agg.get('n',0):4d}  win={agg.get('win_rate_pct','-'):>5}%  "
              f"avg={agg.get('avg_ret_pct','-'):>8}%  median={agg.get('median_ret_pct','-'):>7}%  "
              f"total={agg.get('total_ret_sum_pct','-'):>8}%  "
              f"confirm_rate={agg['confirm_rate_pct']}%")

    out = {
        "config": {"confirm_pct": CONFIRM_PCT, "confirm_window_min": CONFIRM_WINDOW_MIN,
                   "trail_pct": TRAIL_PCT, "rsi_period": RSI_PERIOD,
                   "oversold_thresholds_tested": OVERSOLD_THRESHOLDS},
        "n_usable_candidates": n_usable,
        "results": aggs,
        "oversold_bounce_results": {str(t): bounce_aggs[t] for t in OVERSOLD_THRESHOLDS},
    }
    out_path = HERE / "rsi_1min_confirm_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
