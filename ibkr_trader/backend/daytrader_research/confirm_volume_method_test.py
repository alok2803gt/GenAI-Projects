"""
CEO caught a real gap (2026-09-07): orb_rvol_dipbuy_backtest.py never
actually isolated "median-of-recent-polls vs real RVOL" as a clean,
apples-to-apples swap. That script changed the PRICE level (flat 0.35%
above open -> ORB-high) and the VOLUME check (median-of-this-watch's-own-
polls -> real RVOL) at the same time, and only on the 355-trade gap-down
cohort. The one decomposition run held ORB-high fixed and toggled RVOL on
vs off -- it never compared RVOL against the median-of-polls method at the
ACTUAL live price threshold (0.35% flat, day_trader_agent.py's real node
G), and never across the full population (every setup, not just gap-down).

This script does the clean version: confirm_price = day_open*(1+0.35%)
EXACTLY as live, on the FULL real 820-candidate dataset (all setups, not
just gap-down reversion) -- the ONLY thing that changes is the volume
check:
  (a) CURRENT (live): interval_vol >= median(this watch's own interval
      volumes seen so far) -- exact reproduction of day_trader_agent.py's
      real logic.
  (b) RVOL: cumulative volume so far today / (avg_daily_vol_20d *
      elapsed_fraction_of_session) >= threshold -- a real, external
      baseline instead of a self-referential one.
"""
import csv
import json
import time
from pathlib import Path

import yfinance as yf

HERE = Path(__file__).parent
BARS_PATH = HERE / "minute_bars.json"
CANDIDATES_PATH = HERE / "candidates.csv"
VOLCACHE_PATH = HERE / "confirm_volume_avgvol_cache.json"

CONFIRM_PCT = 0.35            # exactly the live confirm_pct
CONFIRM_WINDOW_MIN = 60       # exactly the live confirm_window_min
TRAIL_PCT = 0.2                # exactly today's live trailing_stop_pct
RVOL_THRESHOLDS_TESTED = [0.8, 1.0, 1.2, 1.5, 2.0]


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


def load_avg_daily_volumes(tickers: list[str]) -> dict[str, float | None]:
    cache = json.loads(VOLCACHE_PATH.read_text()) if VOLCACHE_PATH.exists() else {}
    # Reuse the earlier gap-down-cohort cache if present -- same real data, no need to re-fetch.
    old_cache_path = HERE / "orb_rvol_dipbuy_avgvol_cache.json"
    if old_cache_path.exists():
        old_cache = json.loads(old_cache_path.read_text())
        for tk, v in old_cache.items():
            cache.setdefault(tk, v)
    missing = [t for t in tickers if t not in cache]
    print(f"Fetching 20-day avg daily volume for {len(missing)} tickers not in cache "
          f"({len(tickers) - len(missing)} already cached)...")
    for i, t in enumerate(missing):
        try:
            hist = yf.Ticker(t).history(period="60d", interval="1d")
            cache[t] = float(hist["Volume"].tail(21).head(20).mean()) if len(hist) >= 20 else None
        except Exception as exc:
            print(f"  {t}: fetch failed ({exc})")
            cache[t] = None
        if (i + 1) % 15 == 0:
            VOLCACHE_PATH.write_text(json.dumps(cache))
            print(f"  ...{i+1}/{len(missing)}")
        time.sleep(0.15)
    VOLCACHE_PATH.write_text(json.dumps(cache))
    return cache


def simulate_median_of_polls(bars: list[dict]) -> dict | None:
    """Exact reproduction of day_trader_agent.py's live confirmation gate."""
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
    return _trail_from(bars, entry_idx, entry_px)


def simulate_rvol(bars: list[dict], rvol_threshold: float, avg_daily_vol: float) -> dict | None:
    """Same 0.35% price confirm, real RVOL swapped in for the volume check."""
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
        if b["close"] >= confirm_price and rvol >= rvol_threshold:
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
    print(f"Full real population with usable minute bars: {len(cohort)}\n")

    tickers = sorted(set(c["ticker"] for c in cohort))
    avgvol = load_avg_daily_volumes(tickers)
    n_no_vol = sum(1 for t in tickers if avgvol.get(t) is None)
    if n_no_vol:
        print(f"WARNING: {n_no_vol}/{len(tickers)} tickers have no usable avg-volume baseline "
              f"(excluded from RVOL variants).\n")

    current_results = [simulate_median_of_polls(c["bars"]) for c in cohort]
    current_results = [r for r in current_results if r is not None]
    current_agg = _agg(current_results)
    print(f"CURRENT LIVE (0.35% price + median-of-this-watch's-polls volume), full population:")
    print(f"  n={current_agg['n']}  win={current_agg['win_rate_pct']}%  avg={current_agg['avg_ret_pct']}%  "
          f"total={current_agg['total_ret_sum_pct']}%  avg_entry_min={current_agg['avg_entry_minute']}\n")

    for thr in RVOL_THRESHOLDS_TESTED:
        results = []
        for c in cohort:
            av = avgvol.get(c["ticker"])
            if av is None:
                continue
            r = simulate_rvol(c["bars"], thr, av)
            if r is not None:
                results.append(r)
        agg = _agg(results)
        label = f"RVOL>={thr}x (same 0.35% price)"
        print(f"{label:32s}  n={agg.get('n',0):4d}  win={agg.get('win_rate_pct','-'):>5}%  "
              f"avg={agg.get('avg_ret_pct','-'):>8}%  total={agg.get('total_ret_sum_pct','-'):>8}%  "
              f"avg_entry_min={agg.get('avg_entry_minute','-')}")


if __name__ == "__main__":
    main()
