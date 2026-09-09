"""
Answers a question this account's own daytrader_scanner.py explicitly
flagged as open (see its SCAN_TIME_ET comment: "If a precise, statistically
grounded answer matters, that requires a separate intraday (minute-bar)
backtest -- not something this daily-bar study can produce") -- CEO
follow-up 2026-09-05, investigating why real live Day Trader win rate
(21%) is so far below the backtested confirmation-gate design (52.7%).

Real finding that motivated this: the scanner is deliberately scheduled
for 9:35 ET (not 9:30), and the real 2026-09-03 log shows the actual
scan+dispatch pipeline doesn't finish sending watch signals until
~9:36:02-9:36:50 -- for fast "gap-up momentum" names, price has often
already blown past the 0.35% confirm threshold before watching even
starts, so confirmation doesn't catch an early move, it just rubber-
stamps an already-extended price. This script quantifies, with the same
real 820-candidate 1-minute-bar dataset used throughout this research
project, exactly how much a delayed watch-start costs -- i.e., is a
faster (pre-market-shortlist-driven) pipeline actually worth building.

Method: run the EXACT same real confirmation+trailing-stop mechanics
(0.35% + volume>=median, 60min window, 0.3% trail) but start the watching
clock at bars[delay_min] instead of bars[0], keeping day_open anchored to
the REAL bars[0]["open"] (the true session open the backtest and the gap%
metric both reference) -- exactly replicating what "watching starts N
minutes late" means live. Window length stays 60min from whenever
watching actually starts, matching day_trader_agent.py's real
age_min > confirm_window check (relative to registration time, not a
fixed clock cutoff).
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
BARS_PATH = HERE / "minute_bars.json"

CONFIRM_PCT = 0.35
CONFIRM_WINDOW_MIN = 60
TRAIL_PCT = 0.3
DELAYS_TESTED = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]


def simulate_delayed_confirm_trail(bars: list[dict], delay_min: int) -> dict | None:
    day_open = bars[0]["open"]   # true session open -- unchanged regardless of delay
    confirm_px = day_open * (1 + CONFIRM_PCT / 100)

    watch_start = delay_min
    watch_end = min(delay_min + CONFIRM_WINDOW_MIN, len(bars))
    if watch_start >= len(bars):
        return None

    vols_so_far: list[float] = []
    entry_idx = entry_px = None
    for i in range(watch_start, watch_end):
        b = bars[i]
        vols_so_far.append(b["volume"])
        median_vol = sorted(vols_so_far)[len(vols_so_far) // 2]
        if b["close"] >= confirm_px and b["volume"] >= median_vol:
            entry_idx, entry_px = i, b["close"]
            break
    if entry_idx is None:
        return None

    running_high = entry_px
    for b in bars[entry_idx + 1:]:
        running_high = max(running_high, b["high"])
        stop_px = running_high * (1 - TRAIL_PCT / 100)
        if b["low"] <= stop_px:
            return {"ret_pct": (stop_px / entry_px - 1) * 100, "outcome": "trailed_out",
                    "entry_minute_from_open": entry_idx,
                    "entry_pct_above_open": (entry_px / day_open - 1) * 100}
    exit_px = bars[-1]["close"]
    return {"ret_pct": (exit_px / entry_px - 1) * 100, "outcome": "eod_close",
            "entry_minute_from_open": entry_idx,
            "entry_pct_above_open": (entry_px / day_open - 1) * 100}


def _agg(results):
    if not results:
        return {"n": 0}
    rets = [r["ret_pct"] for r in results]
    wins = [r for r in rets if r > 0]
    avg_entry_pct_above_open = sum(r["entry_pct_above_open"] for r in results) / len(results)
    avg_entry_minute = sum(r["entry_minute_from_open"] for r in results) / len(results)
    return {
        "n": len(results),
        "win_rate_pct": round(len(wins) / len(results) * 100, 1),
        "avg_ret_pct": round(sum(rets) / len(results), 4),
        "total_ret_sum_pct": round(sum(rets), 2),
        "avg_entry_pct_above_open": round(avg_entry_pct_above_open, 3),
        "avg_entry_minute_from_open": round(avg_entry_minute, 1),
    }


def main():
    with open(BARS_PATH) as f:
        bar_data = json.load(f)

    print(f"Loaded {len(bar_data)} real candidates. Testing watch-start delay = 0..10 real minutes "
          f"past the TRUE 9:30 open, same 0.35%+volume/60min/0.3%-trail mechanics throughout.\n")

    all_results = {}
    for delay in DELAYS_TESTED:
        results = []
        for key, bars in bar_data.items():
            if not bars or len(bars) < 10:
                continue
            r = simulate_delayed_confirm_trail(bars, delay)
            if r is not None:
                results.append(r)
        agg = _agg(results)
        all_results[delay] = agg
        print(f"delay={delay:2d}min:  n={agg.get('n',0):4d}  win={agg.get('win_rate_pct','-'):>5}%  "
              f"avg={agg.get('avg_ret_pct','-'):>8}%  total={agg.get('total_ret_sum_pct','-'):>8}%  "
              f"avg_entry_vs_open={agg.get('avg_entry_pct_above_open','-'):>7}%  "
              f"avg_entry_minute={agg.get('avg_entry_minute_from_open','-')}")

    out_path = HERE / "scan_delay_results.json"
    with open(out_path, "w") as f:
        json.dump({"config": {"confirm_pct": CONFIRM_PCT, "confirm_window_min": CONFIRM_WINDOW_MIN,
                               "trail_pct": TRAIL_PCT},
                   "results_by_delay_min": all_results}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
