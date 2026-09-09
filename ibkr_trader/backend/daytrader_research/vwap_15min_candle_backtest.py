"""
Backtests the "15-minute VWAP candle rule" (CEO request, 2026-08-27) on the
EXACT SAME 820-candidate, 82-trading-day sample as strategy_comparison.py
(candidates.csv + minute_bars.json, real IBKR minute bars) -- same real
bars, same live-scanner-selected opportunity set as every other archetype
already tested in this directory, so results are directly comparable.

Rule, as confirmed with the CEO (2026-08-27):
  1. Aggregate 1-min bars into 15-min candles, tracking running VWAP
     bar-by-bar (typical price * volume, cumulative all day).
  2. Signal candle = the FIRST 15-min candle that CLOSES above VWAP as a
     green candle (close > open) -- long only, matching this account's
     standing no-margin/no-shorting constraint (the rule conceptually
     supports shorts too, not tested here).
  3. Entry = break of the signal candle's high, in a later bar.
  4. Stop = signal candle's low. Target = entry + 2x(entry - stop) --
     fixed 2:1 reward:risk, the CEO-confirmed exit rule.
  5. Whichever of stop/target is touched first wins (real bar-by-bar walk,
     same convention as every other strategy here); force-close at the
     day's last bar if neither triggers.

This is a genuinely different archetype from the VWAP MEAN-REVERSION
strategy already tested in strategy_comparison.py (-0.089%/trade, a real
loser) -- that one buys DIPS below VWAP (contrarian); this one waits for a
candle to close on one side of VWAP and trades WITH that momentum
(trend-following), so the prior negative finding does not apply here.
"""
import json
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
candidates = pd.read_csv(HERE / "candidates.csv")
with open(HERE / "minute_bars.json") as f:
    minute_bars = json.load(f)

print(f"{len(candidates)} candidates, {len(minute_bars)} bar series loaded")


def get_bars(ticker, date):
    return minute_bars.get(f"{ticker}:{date}")


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


def sim_baseline(bars, confirm_pct=0.35, confirm_window=60, trail_pct=0.3):
    """Unchanged from strategy_comparison.py -- re-run here for a same-
    script, same-run reference point."""
    day_open = bars[0]["open"]
    confirm_px = day_open * (1 + confirm_pct / 100)
    vols, entry_idx, entry_px = [], None, None
    for i, b in enumerate(bars[:confirm_window]):
        vols.append(b["volume"])
        median_vol = sorted(vols)[len(vols) // 2]
        if b["close"] >= confirm_px and b["volume"] >= median_vol:
            entry_idx, entry_px = i, b["close"]
            break
    if entry_idx is None:
        return None
    running_high = entry_px
    for b in bars[entry_idx + 1:]:
        running_high = max(running_high, b["high"])
        stop_px = running_high * (1 - trail_pct / 100)
        if b["low"] <= stop_px:
            return {"ret_pct": (stop_px / entry_px - 1) * 100, "outcome": "trailed_out"}
    return {"ret_pct": (bars[-1]["close"] / entry_px - 1) * 100, "outcome": "eod_close"}


def sim_vwap_15min_candle(bars, candle_min=15, r_multiple=2.0):
    cum_pv, cum_vol = 0.0, 0.0
    n = len(bars)
    candle_start = 0
    while candle_start + candle_min <= n:
        candle_bars = bars[candle_start:candle_start + candle_min]
        for b in candle_bars:
            typical = (b["high"] + b["low"] + b["close"]) / 3
            cum_pv += typical * b["volume"]
            cum_vol += b["volume"]
        if cum_vol == 0:
            candle_start += candle_min
            continue
        vwap = cum_pv / cum_vol
        c_open  = candle_bars[0]["open"]
        c_close = candle_bars[-1]["close"]
        c_high  = max(b["high"] for b in candle_bars)
        c_low   = min(b["low"] for b in candle_bars)

        if c_close > vwap and c_close > c_open:
            remaining = bars[candle_start + candle_min:]
            entry_idx, entry_px = None, None
            for i, b in enumerate(remaining):
                if b["high"] >= c_high:
                    entry_idx, entry_px = i, max(c_high, b["open"])
                    break
            if entry_idx is None:
                return None  # signal fired but breakout never happened that day

            stop_px = c_low
            risk = entry_px - stop_px
            if risk <= 0:
                return None  # degenerate candle (low >= entry), skip
            target_px = entry_px + r_multiple * risk

            for b in remaining[entry_idx + 1:]:
                if b["low"] <= stop_px:
                    return {"ret_pct": (stop_px / entry_px - 1) * 100, "outcome": "stopped"}
                if b["high"] >= target_px:
                    return {"ret_pct": (target_px / entry_px - 1) * 100, "outcome": "target_hit"}
            return {"ret_pct": (bars[-1]["close"] / entry_px - 1) * 100, "outcome": "eod_close"}

        candle_start += candle_min
    return None  # no qualifying signal candle all day


results = {"baseline": [], "vwap_15candle": []}
matched = 0
for _, row in candidates.iterrows():
    bars = get_bars(row["ticker"], row["date"])
    if not bars or len(bars) < 30:
        continue
    matched += 1

    r = sim_baseline(bars)
    if r: results["baseline"].append(r)

    r = sim_vwap_15min_candle(bars)
    if r: results["vwap_15candle"].append(r)

print(f"\n{matched}/{len(candidates)} candidates had usable real minute bars\n")
print("=== VWAP 15-min candle rule vs baseline (real minute bars, same candidate set) ===")
for name, res in results.items():
    agg = _agg(res)
    if agg["n"] == 0:
        print(f"  {name:16} n=0 (no trades taken -- entry condition never triggered)")
        continue
    print(f"  {name:16} n={agg['n']:4} win_rate={agg['win_rate_pct']:5.1f}% "
          f"avg_ret={agg['avg_ret_pct']:+.4f}% total={agg['total_ret_sum_pct']:+.2f}% "
          f"best={agg['best_pct']:+.2f}% worst={agg['worst_pct']:+.2f}%")

with open(HERE / "vwap_15min_candle_results.json", "w") as f:
    json.dump({name: _agg(res) for name, res in results.items()}, f, indent=2)
print("\nSaved vwap_15min_candle_results.json")
