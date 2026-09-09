"""
CEO proposal (2026-09-06): the live day_trader_agent.py confirmation gate
(day_open * (1+0.35%) + volume >= median-of-this-watch's-own-intervals) is
applied identically to every setup, including "gap-down reversion" (gap_pct
<= -1.0%, the account's OWN best-tested setup per daytrader_scanner.py's
comment: "gap-down setups are effectively a same-day dip-buy / mean-
reversion bet ... empirically the single best setup tested"). The concern:
a heavy gap-down on market-wide panic often has correlated downside
continuation in the first 15min, and a flat 0.35%-above-open + self-
referential volume check is too crude a filter to distinguish "dip is
bottoming" from "still falling, about to blow through a naive threshold."

Proposed fix: require price to break above the REAL opening-range high
(first 1 or 5 one-minute bars) on real above-average RVOL (volume relative
to a genuine historical baseline), instead of an arbitrary %-above-open
level with a self-referential volume comparison.

Method: real 355-candidate gap-down-reversion subset (gap_pct<=-1.0%) of
the SAME 820-candidate/minute-bar dataset (minute_bars.json + candidates.csv)
used throughout this research project, so results are directly comparable
to every prior confirmation-gate study. Real per-ticker 20-day trailing
average DAILY volume pulled from yfinance (cached), used to build a
cumulative-RVOL proxy: RVOL_cum(i) = cum_volume_through_minute_i /
(avg_daily_vol_20d * (i+1)/390). This assumes volume is uniformly
distributed across the session, which is an approximation -- real volume is
front/back-loaded (U-shaped), so this UNDERSTATES true early-session RVOL,
making the RVOL filter, if anything, slightly easier to pass than a proper
time-of-day-adjusted baseline would allow. Stated explicitly per this
account's standing discipline to never hide an approximation in code.

Same 0.3% trailing-stop-from-entry mechanics as every other backtest in
this project, so the ONLY thing being isolated is the entry-confirmation
rule itself.
"""
import csv
import json
import time
from pathlib import Path

import yfinance as yf

HERE = Path(__file__).parent
BARS_PATH = HERE / "minute_bars.json"
CANDIDATES_PATH = HERE / "candidates.csv"
VOLCACHE_PATH = HERE / "orb_rvol_avgvol_cache.json"

TRAIL_PCT = 0.3
CONFIRM_WINDOW_MIN = 60          # unchanged from the live gate -- isolates the entry rule only
CURRENT_CONFIRM_PCT = 0.35       # the live gate's flat threshold, for baseline comparison
ORB_MINUTES_TESTED = [1, 5]
RVOL_THRESHOLDS_TESTED = [1.0, 1.2, 1.5, 2.0]


def load_gap_down_cohort() -> list[dict]:
    rows = list(csv.DictReader(open(CANDIDATES_PATH)))
    cohort = [r for r in rows if float(r["gap_pct"]) <= -1.0]
    with open(BARS_PATH) as f:
        bar_data = json.load(f)
    out = []
    for r in cohort:
        key = f"{r['ticker']}:{r['date']}"
        bars = bar_data.get(key)
        if bars and len(bars) >= 65:
            out.append({"ticker": r["ticker"], "date": r["date"], "gap_pct": float(r["gap_pct"]), "bars": bars})
    return out


def load_avg_daily_volumes(tickers: list[str]) -> dict[str, float | None]:
    if VOLCACHE_PATH.exists():
        cache = json.loads(VOLCACHE_PATH.read_text())
    else:
        cache = {}
    missing = [t for t in tickers if t not in cache]
    print(f"Fetching 20-day avg daily volume for {len(missing)} tickers not in cache "
          f"({len(tickers) - len(missing)} already cached)...")
    for i, t in enumerate(missing):
        try:
            hist = yf.Ticker(t).history(period="60d", interval="1d")
            if len(hist) >= 20:
                cache[t] = float(hist["Volume"].tail(21).head(20).mean())
            else:
                cache[t] = None
        except Exception as exc:
            print(f"  {t}: fetch failed ({exc})")
            cache[t] = None
        if (i + 1) % 15 == 0:
            VOLCACHE_PATH.write_text(json.dumps(cache))
            print(f"  ...{i+1}/{len(missing)}")
        time.sleep(0.15)
    VOLCACHE_PATH.write_text(json.dumps(cache))
    return cache


def simulate_current_gate(bars: list[dict]) -> dict | None:
    day_open = bars[0]["open"]
    confirm_price = day_open * (1 + CURRENT_CONFIRM_PCT / 100)
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
    return _trail_from(bars, entry_idx, entry_px, day_open)


def simulate_orb_rvol_gate(bars: list[dict], orb_minutes: int, rvol_threshold: float,
                            avg_daily_vol: float) -> dict | None:
    day_open = bars[0]["open"]
    if len(bars) <= orb_minutes:
        return None
    orb_high = max(b["high"] for b in bars[:orb_minutes])
    watch_end = min(CONFIRM_WINDOW_MIN, len(bars))

    cum_vol = sum(b["volume"] for b in bars[:orb_minutes])
    entry_idx = entry_px = None
    for i in range(orb_minutes, watch_end):
        b = bars[i]
        cum_vol += b["volume"]
        expected_by_now = avg_daily_vol * (i + 1) / 390.0
        rvol = cum_vol / expected_by_now if expected_by_now > 0 else 0.0
        if b["close"] >= orb_high and rvol >= rvol_threshold:
            entry_idx, entry_px = i, b["close"]
            break
    if entry_idx is None:
        return None
    return _trail_from(bars, entry_idx, entry_px, day_open)


def _trail_from(bars: list[dict], entry_idx: int, entry_px: float, day_open: float) -> dict:
    running_high = entry_px
    for b in bars[entry_idx + 1:]:
        running_high = max(running_high, b["high"])
        stop_px = running_high * (1 - TRAIL_PCT / 100)
        if b["low"] <= stop_px:
            return {"ret_pct": (stop_px / entry_px - 1) * 100, "outcome": "trailed_out",
                    "entry_minute": entry_idx, "entry_pct_above_open": (entry_px / day_open - 1) * 100}
    exit_px = bars[-1]["close"]
    return {"ret_pct": (exit_px / entry_px - 1) * 100, "outcome": "eod_close",
            "entry_minute": entry_idx, "entry_pct_above_open": (entry_px / day_open - 1) * 100}


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
        "avg_entry_pct_above_open": round(sum(r["entry_pct_above_open"] for r in results) / len(results), 3),
        "avg_entry_minute": round(sum(r["entry_minute"] for r in results) / len(results), 1),
    }


def main():
    cohort = load_gap_down_cohort()
    print(f"Real gap-down-reversion cohort with usable minute bars: {len(cohort)}\n")

    tickers = sorted(set(c["ticker"] for c in cohort))
    avgvol = load_avg_daily_volumes(tickers)
    n_no_vol = sum(1 for t in tickers if avgvol.get(t) is None)
    if n_no_vol:
        print(f"WARNING: {n_no_vol}/{len(tickers)} tickers have no usable avg-volume baseline "
              f"(excluded from RVOL variants).\n")

    # Baseline: current live gate, restricted to this cohort
    current_results = [simulate_current_gate(c["bars"]) for c in cohort]
    current_results = [r for r in current_results if r is not None]
    current_agg = _agg(current_results)
    print(f"CURRENT gate (0.35% above open + self-referential median volume), gap-down cohort only:")
    print(f"  n={current_agg.get('n')}  win={current_agg.get('win_rate_pct')}%  "
          f"avg={current_agg.get('avg_ret_pct')}%  total={current_agg.get('total_ret_sum_pct')}%  "
          f"avg_entry_vs_open={current_agg.get('avg_entry_pct_above_open')}%  "
          f"avg_entry_min={current_agg.get('avg_entry_minute')}\n")

    all_results = {"current_gate": current_agg, "orb_rvol_variants": {}}

    for orb_min in ORB_MINUTES_TESTED:
        for thr in RVOL_THRESHOLDS_TESTED:
            results = []
            for c in cohort:
                av = avgvol.get(c["ticker"])
                if av is None:
                    continue
                r = simulate_orb_rvol_gate(c["bars"], orb_min, thr, av)
                if r is not None:
                    results.append(r)
            agg = _agg(results)
            label = f"orb={orb_min}min_rvol>={thr}x"
            all_results["orb_rvol_variants"][label] = agg
            print(f"{label:22s}  n={agg.get('n',0):4d}  win={agg.get('win_rate_pct','-'):>5}%  "
                  f"avg={agg.get('avg_ret_pct','-'):>8}%  total={agg.get('total_ret_sum_pct','-'):>8}%  "
                  f"avg_entry_vs_open={agg.get('avg_entry_pct_above_open','-'):>7}%  "
                  f"avg_entry_min={agg.get('avg_entry_minute','-')}")

    out_path = HERE / "orb_rvol_dipbuy_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
