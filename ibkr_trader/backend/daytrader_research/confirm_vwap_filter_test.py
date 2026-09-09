"""
CEO question (2026-09-07): would requiring price above VWAP, as an
ADDITIONAL confirmation condition, improve Day Trader's real entry gate?
Sourced from a generic Level-2/tape-reading education dump the CEO
reviewed -- most of that content (Level 2 depth reading, spotting
spoofing/iceberg orders, "the Ax") is discretionary human pattern-
recognition that doesn't map onto an automated, rules-based system without
a lot of unproven new infrastructure. VWAP is the one piece that's a
clean, well-defined, already-computable number -- worth a real test,
same standard as every other confirmation-gate change this session.

Method: hold the REAL, currently-live confirmation exactly fixed (0.35%
price above day_open + RVOL>=1.2x, the exact gate deployed today) and add
ONE incremental condition: close >= VWAP at the moment of confirmation.
Same full 820-candidate real dataset, same 0.2% real live trailing stop.
Isolates ONLY the VWAP filter's own effect, same discipline as the
RVOL-vs-median-of-polls test earlier today.

VWAP computed the standard intraday way: cumulative(typical_price x
volume) / cumulative(volume), typical_price = (high+low+close)/3,
reset from the true session open (bar 0) -- resets daily, matches how
VWAP is used in practice.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
BARS_PATH = HERE / "minute_bars.json"
CANDIDATES_PATH = HERE / "candidates.csv"
VOLCACHE_PATH = HERE / "confirm_volume_avgvol_cache.json"

CONFIRM_PCT = 0.35
CONFIRM_WINDOW_MIN = 60
TRAIL_PCT = 0.2          # today's real live trailing_stop_pct
RVOL_THRESHOLD = 1.2     # today's real live rvol_threshold


def load_full_cohort() -> list[dict]:
    rows = list(csv.DictReader(open(CANDIDATES_PATH)))
    with open(BARS_PATH) as f:
        bar_data = json.load(f)
    out = []
    for r in rows:
        key = f"{r['ticker']}:{r['date']}"
        bars = bar_data.get(key)
        if bars and len(bars) >= 65:
            out.append({"ticker": r["ticker"], "date": r["date"], "bars": bars})
    return out


def simulate(bars: list[dict], avg_daily_vol: float, require_vwap: bool) -> dict | None:
    day_open = bars[0]["open"]
    confirm_price = day_open * (1 + CONFIRM_PCT / 100)
    watch_end = min(CONFIRM_WINDOW_MIN, len(bars))

    cum_vol = 0.0
    cum_pv = 0.0
    entry_idx = entry_px = None
    for i in range(0, watch_end):
        b = bars[i]
        cum_vol += b["volume"]
        typical_price = (b["high"] + b["low"] + b["close"]) / 3
        cum_pv += typical_price * b["volume"]
        vwap = cum_pv / cum_vol if cum_vol > 0 else b["close"]

        expected_by_now = avg_daily_vol * (i + 1) / 390.0
        rvol = cum_vol / expected_by_now if expected_by_now > 0 else 0.0

        price_ok = b["close"] >= confirm_price
        vol_ok = rvol >= RVOL_THRESHOLD
        vwap_ok = (b["close"] >= vwap) if require_vwap else True

        if price_ok and vol_ok and vwap_ok:
            entry_idx, entry_px = i, b["close"]
            break
    if entry_idx is None:
        return None
    return _trail_from(bars, entry_idx, entry_px)


def _trail_from(bars: list[dict], entry_idx: int, entry_px: float) -> dict:
    running_high = entry_px
    for b in bars[entry_idx + 1:]:
        running_high = max(running_high, b["high"])
        stop_px = running_high * (1 - TRAIL_PCT / 100)
        if b["low"] <= stop_px:
            return {"ret_pct": (stop_px / entry_px - 1) * 100, "entry_minute": entry_idx}
    exit_px = bars[-1]["close"]
    return {"ret_pct": (exit_px / entry_px - 1) * 100, "entry_minute": entry_idx}


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
        "avg_entry_minute": round(sum(r["entry_minute"] for r in results) / len(results), 1),
    }


def main():
    cohort = load_full_cohort()
    avgvol = json.loads(VOLCACHE_PATH.read_text())
    print(f"Full real population with usable minute bars: {len(cohort)}")
    n_missing = sum(1 for c in cohort if avgvol.get(c["ticker"]) is None)
    if n_missing:
        print(f"WARNING: {n_missing} candidates have no avg_daily_vol cached (excluded).\n")

    baseline, with_vwap = [], []
    for c in cohort:
        av = avgvol.get(c["ticker"])
        if av is None:
            continue
        r0 = simulate(c["bars"], av, require_vwap=False)
        if r0 is not None:
            baseline.append(r0)
        r1 = simulate(c["bars"], av, require_vwap=True)
        if r1 is not None:
            with_vwap.append(r1)

    agg0 = _agg(baseline)
    agg1 = _agg(with_vwap)
    print(f"\nCURRENT LIVE (0.35% + RVOL>=1.2x, no VWAP):")
    print(f"  n={agg0['n']}  win={agg0['win_rate_pct']}%  avg={agg0['avg_ret_pct']}%  "
          f"total={agg0['total_ret_sum_pct']}%  avg_entry_min={agg0['avg_entry_minute']}")
    print(f"\n+ VWAP filter (close>=VWAP added):")
    print(f"  n={agg1['n']}  win={agg1['win_rate_pct']}%  avg={agg1['avg_ret_pct']}%  "
          f"total={agg1['total_ret_sum_pct']}%  avg_entry_min={agg1['avg_entry_minute']}")

    out_path = HERE / "confirm_vwap_filter_results.json"
    out_path.write_text(json.dumps({"baseline": agg0, "with_vwap": agg1}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
