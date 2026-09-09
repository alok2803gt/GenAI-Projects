"""
Final comparison: real entry/exit simulation (SAME 0.35%+RVOL>=1.2x confirm,
0.2% real live trailing stop mechanics used throughout this session) across
three real candidate selections built by build_candidates_variants.py:
  (a) baseline        -- reproduction of the current live method
  (b) no_sector_cap    -- same scoring, sector cap removed
  (c) with_52w         -- composite blended with a real 52-week-range factor

Reuses minute_bars.json (original) + minute_bars_incremental.json (the
fetch this script depends on) so every candidate in all three variants has
real intraday data. RVOL needs each ticker's real 20-day avg daily volume
-- reuses confirm_volume_avgvol_cache.json where already cached, fetches
any remaining via yfinance.
"""
import csv
import json
import time
from pathlib import Path

import yfinance as yf

HERE = Path(__file__).parent
CONFIRM_PCT = 0.35
RVOL_THRESHOLD = 1.2
CONFIRM_WINDOW_MIN = 60
TRAIL_PCT = 0.2
VOLCACHE_PATH = HERE / "confirm_volume_avgvol_cache.json"


def load_all_bars() -> dict:
    with open(HERE / "minute_bars.json") as f:
        bars = json.load(f)
    inc_path = HERE / "minute_bars_incremental.json"
    if inc_path.exists():
        with open(inc_path) as f:
            bars.update(json.load(f))
    return bars


def load_avg_daily_vol(tickers: list[str]) -> dict:
    cache = json.loads(VOLCACHE_PATH.read_text()) if VOLCACHE_PATH.exists() else {}
    missing = [t for t in tickers if t not in cache]
    if missing:
        print(f"Fetching avg_daily_vol for {len(missing)} tickers not yet cached...")
        for i, t in enumerate(missing):
            try:
                hist = yf.Ticker(t).history(period="60d", interval="1d")
                cache[t] = float(hist["Volume"].tail(21).head(20).mean()) if len(hist) >= 20 else None
            except Exception:
                cache[t] = None
            if (i + 1) % 20 == 0:
                VOLCACHE_PATH.write_text(json.dumps(cache))
                print(f"  ...{i+1}/{len(missing)}")
            time.sleep(0.1)
        VOLCACHE_PATH.write_text(json.dumps(cache))
    return cache


def simulate(bars: list[dict], avg_daily_vol: float) -> dict | None:
    day_open = bars[0]["open"]
    confirm_price = day_open * (1 + CONFIRM_PCT / 100)
    watch_end = min(CONFIRM_WINDOW_MIN, len(bars))

    cum_vol = 0.0
    entry_idx = entry_px = None
    for i in range(0, watch_end):
        b = bars[i]
        cum_vol += b["volume"]
        expected_by_now = avg_daily_vol * (i + 1) / 390.0
        rvol = cum_vol / expected_by_now if expected_by_now > 0 else 0.0
        if b["close"] >= confirm_price and rvol >= RVOL_THRESHOLD:
            entry_idx, entry_px = i, b["close"]
            break
    if entry_idx is None:
        return None

    running_high = entry_px
    for b in bars[entry_idx + 1:]:
        running_high = max(running_high, b["high"])
        stop_px = running_high * (1 - TRAIL_PCT / 100)
        if b["low"] <= stop_px:
            return {"ret_pct": (stop_px / entry_px - 1) * 100}
    exit_px = bars[-1]["close"]
    return {"ret_pct": (exit_px / entry_px - 1) * 100}


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
    }


def run_variant(csv_path: Path, bar_data: dict, avgvol: dict, label: str) -> dict:
    rows = list(csv.DictReader(open(csv_path)))
    results = []
    n_no_bars = n_no_vol = 0
    for r in rows:
        key = f"{r['ticker']}:{r['date']}"
        bars = bar_data.get(key)
        if not bars or len(bars) < 65:
            n_no_bars += 1
            continue
        av = avgvol.get(r["ticker"])
        if av is None:
            n_no_vol += 1
            continue
        res = simulate(bars, av)
        if res is not None:
            results.append(res)
    agg = _agg(results)
    print(f"\n{label}: rows={len(rows)}  missing_bars={n_no_bars}  missing_vol={n_no_vol}  "
          f"confirmed_trades={agg['n']}")
    print(f"  win={agg.get('win_rate_pct','-')}%  avg={agg.get('avg_ret_pct','-')}%  "
          f"total={agg.get('total_ret_sum_pct','-')}%")
    return agg


def main():
    bar_data = load_all_bars()
    print(f"Total real minute-bar keys available: {len(bar_data)}")

    all_tickers = set()
    for name in ["candidates_baseline_repro.csv", "candidates_no_sector_cap.csv", "candidates_with_52w.csv"]:
        rows = list(csv.DictReader(open(HERE / name)))
        all_tickers.update(r["ticker"] for r in rows)
    avgvol = load_avg_daily_vol(sorted(all_tickers))

    print("\n" + "=" * 70)
    agg_a = run_variant(HERE / "candidates_baseline_repro.csv", bar_data, avgvol, "BASELINE (current live method)")
    agg_b = run_variant(HERE / "candidates_no_sector_cap.csv", bar_data, avgvol, "NO SECTOR CAP")
    agg_c = run_variant(HERE / "candidates_with_52w.csv", bar_data, avgvol, "WITH 52-WEEK RANGE FACTOR")

    out = {"baseline": agg_a, "no_sector_cap": agg_b, "with_52w": agg_c}
    (HERE / "variant_comparison_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote variant_comparison_results.json")


if __name__ == "__main__":
    main()
