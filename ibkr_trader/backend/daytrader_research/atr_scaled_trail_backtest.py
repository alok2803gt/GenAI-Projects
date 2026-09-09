"""
CEO proposal (2026-09-06): replace Day Trader's flat 0.3% trailing stop
with a dynamic buffer scaled to each ticker's own (real, already-computed)
daily ATR% -- the idea being a low-volatility name gets stopped out on
ordinary noise at a flat 0.3%, while a high-volatility name's real edge
might need more room than 0.3% to breathe.

Real, honest scoping note: daily ATR% in this dataset ranges 3.9%-46.7%
(median 6.5%) -- that's a full SESSION's typical range. The live trail is
an INTRADAY distance-from-peak (minutes, not a full day), so testing
trail_pct = 1x daily ATR% would be wildly wider than anything currently
used (would almost never trigger before EOD). Tested a realistic range of
small fractions of daily ATR% instead (2%-10% of it), which brackets the
current flat 0.3% around the dataset's median ATR%.

Method: SAME real 820-candidate/minute-bar dataset and SAME confirmation
entry mechanics (0.35% above open + volume>=running median, 60min window)
as every other Day Trader confirmation-gate study this session
(scan_delay_backtest.py's delay=0 case is this exact baseline) -- only the
EXIT (trailing-stop distance) changes, so this isolates the ATR-scaling
question cleanly from everything already tested about the entry side.
Real atr_pct per candidate from candidates.csv (the account's own
compute_dt_features output), joined on ticker+date.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
BARS_PATH = HERE / "minute_bars.json"
CANDIDATES_PATH = HERE / "candidates.csv"

CONFIRM_PCT = 0.35
CONFIRM_WINDOW_MIN = 60
FLAT_TRAIL_PCT = 0.3                       # current live value, baseline
ATR_MULTIPLIERS_TESTED = [0.02, 0.04, 0.06, 0.08, 0.10]
TRAIL_FLOOR_PCT = 0.15                     # sanity floor so a very-low-ATR name doesn't get an unrealistically tight stop
TRAIL_CEIL_PCT = 1.0                       # sanity ceiling so a very-high-ATR outlier doesn't get an absurd stop


def load_cohort() -> list[dict]:
    rows = list(csv.DictReader(open(CANDIDATES_PATH)))
    with open(BARS_PATH) as f:
        bar_data = json.load(f)
    out = []
    for r in rows:
        key = f"{r['ticker']}:{r['date']}"
        bars = bar_data.get(key)
        if bars and len(bars) >= 65:
            out.append({"ticker": r["ticker"], "date": r["date"], "atr_pct": float(r["atr_pct"]), "bars": bars})
    return out


def simulate(bars: list[dict], trail_pct: float) -> dict | None:
    day_open = bars[0]["open"]
    confirm_price = day_open * (1 + CONFIRM_PCT / 100)
    watch_end = min(CONFIRM_WINDOW_MIN, len(bars))

    vols_so_far: list[float] = []
    entry_idx = entry_px = None
    for i in range(0, watch_end):
        b = bars[i]
        vols_so_far.append(b["volume"])
        median_vol = sorted(vols_so_far)[len(vols_so_far) // 2]
        if b["close"] >= confirm_price and b["volume"] >= median_vol:
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
    exit_px = bars[-1]["close"]
    return {"ret_pct": (exit_px / entry_px - 1) * 100, "outcome": "eod_close"}


def _agg(results: list[dict]) -> dict:
    if not results:
        return {"n": 0}
    rets = [r["ret_pct"] for r in results]
    wins = [r for r in rets if r > 0]
    n_trailed = sum(1 for r in results if r["outcome"] == "trailed_out")
    return {
        "n": len(results),
        "win_rate_pct": round(len(wins) / len(results) * 100, 1),
        "avg_ret_pct": round(sum(rets) / len(results), 4),
        "total_ret_sum_pct": round(sum(rets), 2),
        "pct_trailed_out": round(n_trailed / len(results) * 100, 1),
    }


def main():
    cohort = load_cohort()
    print(f"Real candidates with usable minute bars + real atr_pct: {len(cohort)}\n")

    flat_results = [simulate(c["bars"], FLAT_TRAIL_PCT) for c in cohort]
    flat_results = [r for r in flat_results if r is not None]
    flat_agg = _agg(flat_results)
    print(f"BASELINE flat {FLAT_TRAIL_PCT}% trail (current live value):")
    print(f"  n={flat_agg['n']}  win={flat_agg['win_rate_pct']}%  avg={flat_agg['avg_ret_pct']}%  "
          f"total={flat_agg['total_ret_sum_pct']}%  trailed_out={flat_agg['pct_trailed_out']}%\n")

    for k in ATR_MULTIPLIERS_TESTED:
        results = []
        trail_pcts = []
        for c in cohort:
            trail_pct = max(TRAIL_FLOOR_PCT, min(TRAIL_CEIL_PCT, k * c["atr_pct"]))
            trail_pcts.append(trail_pct)
            r = simulate(c["bars"], trail_pct)
            if r is not None:
                results.append(r)
        agg = _agg(results)
        avg_trail = sum(trail_pcts) / len(trail_pcts)
        label = f"ATR x{k:.2f}"
        print(f"{label:12s} (avg trail={avg_trail:.3f}%)  n={agg.get('n',0):4d}  "
              f"win={agg.get('win_rate_pct','-'):>5}%  avg={agg.get('avg_ret_pct','-'):>8}%  "
              f"total={agg.get('total_ret_sum_pct','-'):>8}%  trailed_out={agg.get('pct_trailed_out','-')}%")

    # Decompose: does ATR-scaling help specifically the HIGH-ATR subset (top quartile),
    # where the "flat stop is too tight" theory should show up most clearly if real?
    atrs = sorted(c["atr_pct"] for c in cohort)
    p75 = atrs[3 * len(atrs) // 4]
    high_atr_cohort = [c for c in cohort if c["atr_pct"] >= p75]
    print(f"\n=== High-ATR subset only (top quartile, atr_pct>={p75:.2f}%, n={len(high_atr_cohort)}) ===")
    flat_hi = [simulate(c["bars"], FLAT_TRAIL_PCT) for c in high_atr_cohort]
    flat_hi = [r for r in flat_hi if r is not None]
    agg = _agg(flat_hi)
    print(f"  flat {FLAT_TRAIL_PCT}%:      n={agg.get('n',0):4d}  win={agg.get('win_rate_pct','-'):>5}%  "
          f"avg={agg.get('avg_ret_pct','-'):>8}%  trailed_out={agg.get('pct_trailed_out','-')}%")
    for k in ATR_MULTIPLIERS_TESTED:
        results = []
        for c in high_atr_cohort:
            trail_pct = max(TRAIL_FLOOR_PCT, min(TRAIL_CEIL_PCT, k * c["atr_pct"]))
            r = simulate(c["bars"], trail_pct)
            if r is not None:
                results.append(r)
        agg = _agg(results)
        print(f"  ATR x{k:.2f}:      n={agg.get('n',0):4d}  win={agg.get('win_rate_pct','-'):>5}%  "
              f"avg={agg.get('avg_ret_pct','-'):>8}%  trailed_out={agg.get('pct_trailed_out','-')}%")


if __name__ == "__main__":
    main()
