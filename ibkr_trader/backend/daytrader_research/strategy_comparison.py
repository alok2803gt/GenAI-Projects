"""
Compares 4 real, long-only intraday strategies on the EXACT SAME 820-
candidate, 82-trading-day sample (candidates.csv + minute_bars.json) --
same real IBKR minute bars, same live-scanner-selected opportunity set, so
differences reflect entry/exit mechanics, not different stocks/days. All
long-only (no margin, no shorting -- matches this account's standing
constraints).

1. BASELINE -- replicates the account's ALREADY-LIVE, already-validated
   Day Trader config exactly (confirm_pct=0.35% within 60min + volume
   confirmation, then 0.3% trailing stop), run fresh on THIS sample rather
   than citing the old backtest's numbers from a different period -- a
   real apples-to-apples reference point for everything else here.

2. OPENING RANGE BREAKOUT (ORB) -- a genuinely different archetype: define
   the opening range as the high/low of the first N minutes, enter long
   only on a real breakout above the range high, stop at the range low,
   trail once in profit. Classic, distinct from momentum-continuation.

3. VWAP MEAN-REVERSION (long only) -- enter on a real dip below the
   day's running VWAP (buying weakness, not strength -- opposite
   philosophy from the other 3), exit at VWAP reversion or a stop.

4. GAP-FADE (long only) -- for names that gapped DOWN at the open
   (gap_pct already computed in candidates.csv), bet on reversion toward
   the prior close rather than continuation of the gap.

Same outcome convention throughout: real bar-by-bar walk, whichever of
target/stop is ACTUALLY touched first (no daily-bar ambiguity -- this is
exactly why daytrader_intraday_backtest.py moved to minute bars in the
first place), force-close at day's last bar if neither triggers.
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
    # Real backend key format (main.py /market/history/minute): f"{ticker}:{date}"
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


def sim_orb(bars, range_min=15, stop_at_range_low=True, trail_pct=0.4):
    if len(bars) < range_min + 5:
        return None
    opening_range = bars[:range_min]
    range_high = max(b["high"] for b in opening_range)
    range_low = min(b["low"] for b in opening_range)
    if range_high <= range_low:
        return None
    entry_idx, entry_px = None, None
    for i, b in enumerate(bars[range_min:], start=range_min):
        if b["high"] >= range_high:
            entry_idx, entry_px = i, max(range_high, b["open"])
            break
    if entry_idx is None:
        return None
    stop_px = range_low if stop_at_range_low else entry_px * (1 - trail_pct / 100)
    running_high = entry_px
    for b in bars[entry_idx + 1:]:
        if b["low"] <= stop_px:
            return {"ret_pct": (stop_px / entry_px - 1) * 100, "outcome": "stopped"}
        running_high = max(running_high, b["high"])
        trail_stop = running_high * (1 - trail_pct / 100)
        if trail_stop > stop_px:
            stop_px = trail_stop
    return {"ret_pct": (bars[-1]["close"] / entry_px - 1) * 100, "outcome": "eod_close"}


def sim_vwap_reversion(bars, dip_pct=0.5, stop_pct=0.5):
    cum_pv, cum_vol = 0.0, 0.0
    entry_idx, entry_px = None, None
    for i, b in enumerate(bars[:-5]):  # leave room for an exit after entry
        typical = (b["high"] + b["low"] + b["close"]) / 3
        cum_pv += typical * b["volume"]
        cum_vol += b["volume"]
        if cum_vol == 0:
            continue
        vwap = cum_pv / cum_vol
        dip_level = vwap * (1 - dip_pct / 100)
        if i >= 5 and b["low"] <= dip_level and b["close"] > bars[i - 1]["close"]:
            entry_idx, entry_px, entry_vwap = i, b["close"], vwap
            break
    if entry_idx is None:
        return None
    stop_px = entry_px * (1 - stop_pct / 100)
    for b in bars[entry_idx + 1:]:
        if b["low"] <= stop_px:
            return {"ret_pct": (stop_px / entry_px - 1) * 100, "outcome": "stopped"}
        if b["high"] >= entry_vwap:
            return {"ret_pct": (entry_vwap / entry_px - 1) * 100, "outcome": "vwap_reversion"}
    return {"ret_pct": (bars[-1]["close"] / entry_px - 1) * 100, "outcome": "eod_close"}


def sim_gap_fade(bars, gap_pct, prior_close, stop_pct=1.0, min_gap_down=1.5):
    if gap_pct is None or gap_pct > -min_gap_down:
        return None  # only fade real, meaningful gap-downs
    entry_px = bars[0]["open"]
    stop_px = entry_px * (1 - stop_pct / 100)
    target_px = prior_close
    for b in bars:
        if b["low"] <= stop_px:
            return {"ret_pct": (stop_px / entry_px - 1) * 100, "outcome": "stopped"}
        if b["high"] >= target_px:
            return {"ret_pct": (target_px / entry_px - 1) * 100, "outcome": "gap_filled"}
    return {"ret_pct": (bars[-1]["close"] / entry_px - 1) * 100, "outcome": "eod_close"}


results = {"baseline": [], "orb_15": [], "orb_30": [], "vwap_reversion": [], "gap_fade": []}
matched = 0
for _, row in candidates.iterrows():
    bars = get_bars(row["ticker"], row["date"])
    if not bars or len(bars) < 30:
        continue
    matched += 1

    r = sim_baseline(bars)
    if r: results["baseline"].append(r)

    r = sim_orb(bars, range_min=15)
    if r: results["orb_15"].append(r)

    r = sim_orb(bars, range_min=30)
    if r: results["orb_30"].append(r)

    r = sim_vwap_reversion(bars)
    if r: results["vwap_reversion"].append(r)

    prior_close = row["open"] / (1 + row["gap_pct"] / 100) if pd.notna(row["gap_pct"]) else None
    if prior_close:
        r = sim_gap_fade(bars, row["gap_pct"], prior_close)
        if r: results["gap_fade"].append(r)

print(f"\n{matched}/{len(candidates)} candidates had usable real minute bars\n")
print("=== Strategy comparison (real minute bars, same candidate set) ===")
for name, res in results.items():
    agg = _agg(res)
    if agg["n"] == 0:
        print(f"  {name:16} n=0 (no trades taken -- entry condition never triggered)")
        continue
    print(f"  {name:16} n={agg['n']:4} win_rate={agg['win_rate_pct']:5.1f}% "
          f"avg_ret={agg['avg_ret_pct']:+.4f}% total={agg['total_ret_sum_pct']:+.2f}% "
          f"best={agg['best_pct']:+.2f}% worst={agg['worst_pct']:+.2f}%")

with open(HERE / "strategy_comparison_results.json", "w") as f:
    json.dump({name: _agg(res) for name, res in results.items()}, f, indent=2)
print("\nSaved strategy_comparison_results.json")
