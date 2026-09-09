"""
CEO asked for genuine SAME-DAY S/R (not the prior-day-only pivot/52w/20d
range factors already tested for premarket scoring). This tests it as an
ADDITIONAL confirmation filter, not a trigger replacement -- the earlier
ORB test (orb_rvol_dipbuy_backtest.py) already showed REPLACING the
0.35%-above-open trigger with an opening-range-high trigger hurts, because
it delays and raises the effective entry level. This isolates a different
question: does ALSO requiring price to clear the opening-range high (on
top of the existing 0.35%+RVOL trigger, not instead of it) change
anything, using the SAME real full population and real live mechanics
(0.2% trail).

Same real 820-candidate dataset used throughout this session.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
CANDIDATES_PATH = HERE / "candidates.csv"
VOLCACHE_PATH = HERE / "confirm_volume_avgvol_cache.json"

CONFIRM_PCT = 0.35
RVOL_THRESHOLD = 1.2
CONFIRM_WINDOW_MIN = 60
TRAIL_PCT = 0.2
ORB_MINUTES_TESTED = [5, 15]


def load_all_bars() -> dict:
    with open(HERE / "minute_bars.json") as f:
        bars = json.load(f)
    for extra in ["minute_bars_incremental.json"]:
        p = HERE / extra
        if p.exists():
            with open(p) as f:
                bars.update(json.load(f))
    return bars


def load_full_cohort(bar_data: dict) -> list[dict]:
    rows = list(csv.DictReader(open(CANDIDATES_PATH)))
    out = []
    for r in rows:
        key = f"{r['ticker']}:{r['date']}"
        bars = bar_data.get(key)
        if bars and len(bars) >= 65:
            out.append({"ticker": r["ticker"], "date": r["date"], "bars": bars})
    return out


def simulate(bars: list[dict], avg_daily_vol: float, orb_minutes: int | None) -> dict | None:
    day_open = bars[0]["open"]
    confirm_price = day_open * (1 + CONFIRM_PCT / 100)
    watch_end = min(CONFIRM_WINDOW_MIN, len(bars))
    orb_high = max(b["high"] for b in bars[:orb_minutes]) if orb_minutes else None

    cum_vol = 0.0
    entry_idx = entry_px = None
    for i in range(0, watch_end):
        b = bars[i]
        cum_vol += b["volume"]
        expected_by_now = avg_daily_vol * (i + 1) / 390.0
        rvol = cum_vol / expected_by_now if expected_by_now > 0 else 0.0

        price_ok = b["close"] >= confirm_price
        vol_ok = rvol >= RVOL_THRESHOLD
        sr_ok = (b["close"] >= orb_high) if (orb_minutes and i >= orb_minutes) else (orb_minutes is None)

        if price_ok and vol_ok and sr_ok:
            entry_idx, entry_px = i, b["close"]
            break
    if entry_idx is None:
        return None

    running_high = entry_px
    for b in bars[entry_idx + 1:]:
        running_high = max(running_high, b["high"])
        stop_px = running_high * (1 - TRAIL_PCT / 100)
        if b["low"] <= stop_px:
            return {"ret_pct": (stop_px / entry_px - 1) * 100, "entry_minute": entry_idx}
    exit_px = bars[-1]["close"]
    return {"ret_pct": (exit_px / entry_px - 1) * 100, "entry_minute": entry_idx}


def _agg(results):
    if not results:
        return {"n": 0}
    rets = [r["ret_pct"] for r in results]
    wins = [r for r in rets if r > 0]
    return {
        "n": len(results),
        "win_rate_pct": round(len(wins) / len(results) * 100, 1),
        "avg_ret_pct": round(sum(rets) / len(results), 4),
        "avg_entry_minute": round(sum(r["entry_minute"] for r in results) / len(results), 1),
    }


def main():
    bar_data = load_all_bars()
    cohort = load_full_cohort(bar_data)
    avgvol = json.loads(VOLCACHE_PATH.read_text())
    print(f"Full real population with usable minute bars: {len(cohort)}")

    baseline = []
    for c in cohort:
        av = avgvol.get(c["ticker"])
        if av is None:
            continue
        r = simulate(c["bars"], av, orb_minutes=None)
        if r is not None:
            baseline.append(r)
    agg0 = _agg(baseline)
    print(f"\nCURRENT LIVE (0.35% + RVOL>=1.2x, no same-day S/R):")
    print(f"  n={agg0['n']}  win={agg0['win_rate_pct']}%  avg={agg0['avg_ret_pct']}%  "
          f"avg_entry_min={agg0['avg_entry_minute']}")

    for orb_min in ORB_MINUTES_TESTED:
        results = []
        for c in cohort:
            av = avgvol.get(c["ticker"])
            if av is None:
                continue
            r = simulate(c["bars"], av, orb_minutes=orb_min)
            if r is not None:
                results.append(r)
        agg = _agg(results)
        print(f"\n+ same-day S/R (also clear {orb_min}-min opening range high):")
        print(f"  n={agg.get('n',0)}  win={agg.get('win_rate_pct','-')}%  avg={agg.get('avg_ret_pct','-')}%  "
              f"avg_entry_min={agg.get('avg_entry_minute','-')}")


if __name__ == "__main__":
    main()
