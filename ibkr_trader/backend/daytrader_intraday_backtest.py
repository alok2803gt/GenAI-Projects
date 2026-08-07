"""
Day Trader intraday validation: confirmation gate + trailing stop, vs the
current blind-entry/fixed-target baseline -- using REAL 1-minute bars, not
daily OHLC. This is the follow-up to daytrader_sizing_backtest.py, which
found the current 0.25% target / 1.0% stop pair is so small relative to
these stocks' typical range that daily bars can't tell whether target or
stop got touched first (conservative ordering: 26.7% win rate, deeply
negative edge; optimistic ordering: 92.7% win rate, positive edge -- an
unresolvable coin flip from daily data alone).

This script resolves that ambiguity for real, for a recent sample, and
tests the proposed fix: don't buy blind at the open -- wait for the move to
actually show up (price + volume confirmation), then protect it with a
trailing stop instead of a fixed target too small to mean anything.

Two-part run:
  1. BASELINE (current live mechanics, but with REAL intraday sequencing
     instead of an assumed ordering): enter at day's Open, fixed 0.25%
     target / 1.0% stop, walk 1-min bars forward and take whichever is
     ACTUALLY touched first.
  2. CONFIRMATION + TRAILING: only enter once price has moved
     >= CONFIRM_PCT from the open AND that minute's volume is at/above the
     median per-minute volume seen so far that morning (rules out a
     move on a single thin print). Confirmation window: first
     CONFIRM_WINDOW_MIN minutes only -- if it never confirms, no trade
     that day. Once entered, a trailing stop (tested at several widths)
     replaces the fixed target entirely; running high ratchets the stop
     up, force-close at 15:59 ET if never stopped out.

Sample: the most recent ~35 trading days of candidates from
daytrader_sizing_backtest.py's exact live selection logic (same ATR%/score/
sector-cap/max-positions gates) -- real candidates the live system would
actually have picked, not a hand-picked list.
"""
import sys, io, json, time
from datetime import datetime, timedelta

if hasattr(sys.stdout, 'buffer') and (sys.stdout.encoding or '').lower() != 'utf-8':
    # Guard against double-wrapping: this module imports daytrader_sizing_backtest,
    # which does the same re-encode at its own module level. Wrapping the same
    # underlying buffer twice creates two TextIOWrapper objects over one buffer --
    # when the first gets garbage-collected its __del__ closes the shared buffer,
    # breaking the second ("ValueError: I/O operation on closed file", hit live
    # 2026-08-07). Checking the encoding first makes this idempotent.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests

from daytrader_sizing_backtest import run_phase1, PROFIT_TARGET_PCT, HARD_STOP_PCT

HERE = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]
RESULTS_PATH = f"{HERE}/daytrader_intraday_results.json"
BACKEND_URL  = "http://localhost:8000"

RECENT_TRADING_DAYS = 15     # how far back the sample goes -- kept smaller than the first attempt
                               # since /market/history/minute now paces at 5 req/15s (IBKR's stricter
                               # sub-daily-bar limit, discovered 2026-08-07 when 40-concurrent/2s only
                               # returned 40/350), so runtime scales directly with sample size
CONFIRM_PCT          = 0.35   # price must be up this much from the open to "confirm" -- roughly midway
                               # between the old 0.25% target and a level clearly above noise
CONFIRM_WINDOW_MIN    = 60     # only look for confirmation in the first N minutes (9:30-10:30 ET)
TRAIL_WIDTHS_PCT      = [0.3, 0.4, 0.5, 0.6, 0.8]   # let the data pick the best


def fetch_minute_bars(pairs: list[tuple[str, str]]) -> dict[str, list]:
    payload = [{"ticker": tk, "date": d} for tk, d in pairs]
    # ~5 req/15s server-side pacing -- e.g. 150 pairs takes ~7.5min; generous client timeout
    r = requests.post(f"{BACKEND_URL}/market/history/minute", json=payload, timeout=1800)
    r.raise_for_status()
    d = r.json()
    print(f"  minute bars: requested {d['requested']}, returned {d['returned']}")
    return d["data"]


def simulate_baseline(bars: list[dict]) -> dict:
    """Real intraday sequencing -- entry at Open, walk bars forward, whichever
    of target/stop is ACTUALLY touched first wins. No ambiguity this time."""
    entry = bars[0]["open"]
    stop_px   = entry * (1 - HARD_STOP_PCT / 100)
    target_px = entry * (1 + PROFIT_TARGET_PCT / 100)
    for b in bars:
        if b["low"] <= stop_px:
            return {"ret_pct": -HARD_STOP_PCT, "outcome": "stop", "entry": entry}
        if b["high"] >= target_px:
            return {"ret_pct": PROFIT_TARGET_PCT, "outcome": "target", "entry": entry}
    exit_px = bars[-1]["close"]
    return {"ret_pct": (exit_px / entry - 1) * 100, "outcome": "eod_close", "entry": entry}


def simulate_confirm_trail(bars: list[dict], confirm_pct: float, confirm_window_min: int,
                            trail_pct: float) -> dict | None:
    """Returns None if never confirmed (no trade taken that day)."""
    day_open = bars[0]["open"]
    confirm_px = day_open * (1 + confirm_pct / 100)

    vols_so_far: list[float] = []
    entry_idx = None
    entry_px  = None
    for i, b in enumerate(bars[:confirm_window_min]):
        vols_so_far.append(b["volume"])
        median_vol = sorted(vols_so_far)[len(vols_so_far) // 2]
        if b["close"] >= confirm_px and b["volume"] >= median_vol:
            entry_idx, entry_px = i, b["close"]
            break

    if entry_idx is None:
        return None   # never confirmed -- no trade

    running_high = entry_px
    for b in bars[entry_idx + 1:]:
        running_high = max(running_high, b["high"])
        stop_px = running_high * (1 - trail_pct / 100)
        if b["low"] <= stop_px:
            exit_px = stop_px
            return {"ret_pct": (exit_px / entry_px - 1) * 100, "outcome": "trailed_out",
                    "entry": entry_px, "confirm_minute": entry_idx}
    exit_px = bars[-1]["close"]
    return {"ret_pct": (exit_px / entry_px - 1) * 100, "outcome": "eod_close",
            "entry": entry_px, "confirm_minute": entry_idx}


def _agg(results: list[dict]) -> dict:
    if not results:
        return {"n": 0}
    rets = [r["ret_pct"] for r in results]
    wins = [r for r in rets if r > 0]
    return {
        "n": len(results),
        "win_rate_pct": round(len(wins) / len(results) * 100, 1),
        "avg_ret_pct": round(sum(rets) / len(results), 4),
        "total_ret_sum_pct": round(sum(rets), 2),
        "best_pct": round(max(rets), 2),
        "worst_pct": round(min(rets), 2),
    }


def main():
    print("Step 1: regenerating the real trade-selection sample (5yr, live logic)...")
    trade_log = run_phase1()   # from daytrader_sizing_backtest -- exact live selection logic
    if not trade_log:
        print("No trades in selection sample -- aborting.")
        return

    dates_all = sorted({t["date"] for t in trade_log})
    recent_dates = set(dates_all[-RECENT_TRADING_DAYS:])
    recent = [t for t in trade_log if t["date"] in recent_dates]
    pairs = sorted({(t["ticker"], t["date"]) for t in recent})
    print(f"Step 2: recent sample = {len(recent_dates)} trading days ({min(recent_dates)} to "
          f"{max(recent_dates)}), {len(pairs)} (ticker,date) candidates")

    print("Step 3: pulling real 1-min IBKR bars for each candidate...")
    bar_data = fetch_minute_bars(pairs)

    print("Step 4: simulating baseline (real sequencing) + confirmation/trailing variants...")
    baseline_results = []
    confirm_results: dict[float, list] = {w: [] for w in TRAIL_WIDTHS_PCT}
    n_confirmed_any = 0

    for tk, d in pairs:
        bars = bar_data.get(f"{tk}:{d}")
        if not bars or len(bars) < 10:
            continue
        baseline_results.append(simulate_baseline(bars))

        confirmed_this_one = False
        for w in TRAIL_WIDTHS_PCT:
            res = simulate_confirm_trail(bars, CONFIRM_PCT, CONFIRM_WINDOW_MIN, w)
            if res is not None:
                confirm_results[w].append(res)
                confirmed_this_one = True
        if confirmed_this_one:
            n_confirmed_any += 1

    print(f"\n  candidates with usable bars: {len(baseline_results)}")
    print(f"  confirmed (price+volume within {CONFIRM_WINDOW_MIN}min): {n_confirmed_any} "
          f"({n_confirmed_any/len(baseline_results)*100:.1f}%)" if baseline_results else "")

    baseline_agg = _agg(baseline_results)
    print(f"\n  BASELINE (blind entry, real sequencing): {baseline_agg}")

    trail_aggs = {}
    for w in TRAIL_WIDTHS_PCT:
        agg = _agg(confirm_results[w])
        trail_aggs[w] = agg
        print(f"  CONFIRM+TRAIL {w}%: {agg}")

    best_width = max(TRAIL_WIDTHS_PCT, key=lambda w: trail_aggs[w].get("avg_ret_pct", -999))

    results = {
        "computed_at": datetime.now().astimezone().isoformat(),
        "sample": {"n_trading_days": len(recent_dates), "date_range": [min(recent_dates), max(recent_dates)],
                   "n_candidates": len(pairs), "n_usable": len(baseline_results)},
        "config": {"confirm_pct": CONFIRM_PCT, "confirm_window_min": CONFIRM_WINDOW_MIN,
                   "trail_widths_tested": TRAIL_WIDTHS_PCT,
                   "baseline_target_pct": PROFIT_TARGET_PCT, "baseline_stop_pct": HARD_STOP_PCT},
        "baseline": baseline_agg,
        "confirm_trail_by_width": {str(w): trail_aggs[w] for w in TRAIL_WIDTHS_PCT},
        "best_trail_width_pct": best_width,
        "n_confirmed_any_width": n_confirmed_any,
        "confirm_rate_pct": round(n_confirmed_any / len(baseline_results) * 100, 1) if baseline_results else None,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
