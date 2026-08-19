"""
Day Trader confirmation gate: ATR-relative threshold vs the current flat
0.35%. Reviewer catch (2026-08-11): Day Trader Scanner ranks candidates by
ATR% (55% of composite score) -- i.e. deliberately selects the MOST
volatile names in the universe -- then gates entry on a flat 0.35% move.
For a high-ATR% name that's trivial noise, not meaningful confirmation;
the filter may barely discriminate for exactly the candidates most likely
to be selected. This tests confirm_pct = atr_mult * atr_pct instead of a
fixed number, same real 1-min-bar methodology as
daytrader_intraday_backtest.py (same sample, same trailing-stop mechanics),
so the flat-0.35% and ATR-relative results are a fair apples-to-apples
comparison, not two different backtests.
"""
import sys, io, json
from datetime import datetime

if hasattr(sys.stdout, 'buffer') and (sys.stdout.encoding or '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
from daytrader_sizing_backtest import run_phase1, PROFIT_TARGET_PCT, HARD_STOP_PCT

HERE = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]
RESULTS_PATH = f"{HERE}/daytrader_atr_confirm_results.json"
BACKEND_URL  = "http://localhost:8000"

RECENT_TRADING_DAYS = 15
FLAT_CONFIRM_PCT     = 0.35                      # current live baseline
ATR_MULTS            = [0.10, 0.125, 0.15, 0.20]  # candidate ATR-relative thresholds to test
CONFIRM_WINDOW_MIN   = 60
TRAIL_WIDTHS_PCT     = [0.3, 0.4, 0.5, 0.6, 0.8]


def fetch_minute_bars(pairs: list[tuple[str, str]]) -> dict[str, list]:
    payload = [{"ticker": tk, "date": d} for tk, d in pairs]
    r = requests.post(f"{BACKEND_URL}/market/history/minute", json=payload, timeout=1800)
    r.raise_for_status()
    d = r.json()
    print(f"  minute bars: requested {d['requested']}, returned {d['returned']}")
    return d["data"]


def simulate_confirm_trail(bars: list[dict], confirm_pct: float, confirm_window_min: int,
                            trail_pct: float) -> dict | None:
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
    return {
        "n": len(results),
        "win_rate_pct": round(len(wins) / len(results) * 100, 1),
        "avg_ret_pct": round(sum(rets) / len(results), 4),
        "total_ret_sum_pct": round(sum(rets), 2),
        "best_pct": round(max(rets), 2),
        "worst_pct": round(min(rets), 2),
    }


def _best_trail(results_by_width: dict[float, list]) -> tuple[float, dict]:
    aggs = {w: _agg(results_by_width[w]) for w in results_by_width}
    best_w = max(aggs, key=lambda w: aggs[w].get("avg_ret_pct", -999))
    return best_w, aggs


def main():
    print("Step 1: regenerating the real trade-selection sample (5yr, live logic)...")
    trade_log = run_phase1()
    if not trade_log:
        print("No trades in selection sample -- aborting.")
        return

    dates_all = sorted({t["date"] for t in trade_log})
    recent_dates = set(dates_all[-RECENT_TRADING_DAYS:])
    recent = [t for t in trade_log if t["date"] in recent_dates]
    # carry atr_pct through per (ticker,date) -- needed for the ATR-relative variants
    atr_by_pair = {(t["ticker"], t["date"]): t["atr_pct"] for t in recent}
    pairs = sorted(atr_by_pair.keys())
    print(f"Step 2: recent sample = {len(recent_dates)} trading days ({min(recent_dates)} to "
          f"{max(recent_dates)}), {len(pairs)} (ticker,date) candidates")
    print(f"  ATR%% range in sample: min={min(atr_by_pair.values()):.2f} "
          f"median={sorted(atr_by_pair.values())[len(atr_by_pair)//2]:.2f} "
          f"max={max(atr_by_pair.values()):.2f}")

    print("Step 3: pulling real 1-min IBKR bars for each candidate...")
    bar_data = fetch_minute_bars(pairs)

    print("Step 4: simulating flat-0.35%% baseline + ATR-relative variants (same bars, same trail widths)...")
    flat_results: dict[float, list] = {w: [] for w in TRAIL_WIDTHS_PCT}
    atr_results: dict[float, dict[float, list]] = {m: {w: [] for w in TRAIL_WIDTHS_PCT} for m in ATR_MULTS}
    n_confirmed_flat = 0
    n_confirmed_atr = {m: 0 for m in ATR_MULTS}
    n_usable = 0

    for tk, d in pairs:
        bars = bar_data.get(f"{tk}:{d}")
        if not bars or len(bars) < 10:
            continue
        n_usable += 1
        atr_pct = atr_by_pair[(tk, d)]

        confirmed_flat_this_one = False
        for w in TRAIL_WIDTHS_PCT:
            res = simulate_confirm_trail(bars, FLAT_CONFIRM_PCT, CONFIRM_WINDOW_MIN, w)
            if res is not None:
                flat_results[w].append(res)
                confirmed_flat_this_one = True
        if confirmed_flat_this_one:
            n_confirmed_flat += 1

        for m in ATR_MULTS:
            confirm_pct = m * atr_pct
            confirmed_this_mult = False
            for w in TRAIL_WIDTHS_PCT:
                res = simulate_confirm_trail(bars, confirm_pct, CONFIRM_WINDOW_MIN, w)
                if res is not None:
                    atr_results[m][w].append(res)
                    confirmed_this_mult = True
            if confirmed_this_mult:
                n_confirmed_atr[m] += 1

    print(f"\n  candidates with usable bars: {n_usable}")
    print(f"  FLAT 0.35%% confirm rate: {n_confirmed_flat}/{n_usable} "
          f"({n_confirmed_flat/n_usable*100:.1f}%%)" if n_usable else "")

    flat_best_w, flat_aggs = _best_trail(flat_results)
    print(f"\n  === FLAT 0.35% (current live baseline) ===")
    for w, agg in flat_aggs.items():
        marker = " <- best" if w == flat_best_w else ""
        print(f"    trail={w}%: {agg}{marker}")

    atr_summary = {}
    for m in ATR_MULTS:
        best_w, aggs = _best_trail(atr_results[m])
        confirm_rate = round(n_confirmed_atr[m] / n_usable * 100, 1) if n_usable else None
        print(f"\n  === ATR-relative, confirm_pct = {m} x ATR% (confirm rate {confirm_rate}%) ===")
        for w, agg in aggs.items():
            marker = " <- best" if w == best_w else ""
            print(f"    trail={w}%: {agg}{marker}")
        atr_summary[str(m)] = {"best_trail_width_pct": best_w, "confirm_rate_pct": confirm_rate,
                                "results_by_width": {str(w): aggs[w] for w in aggs}}

    results = {
        "computed_at": datetime.now().astimezone().isoformat(),
        "sample": {"n_trading_days": len(recent_dates), "date_range": [min(recent_dates), max(recent_dates)],
                   "n_candidates": len(pairs), "n_usable": n_usable,
                   "atr_pct_min": min(atr_by_pair.values()), "atr_pct_median": sorted(atr_by_pair.values())[len(atr_by_pair)//2],
                   "atr_pct_max": max(atr_by_pair.values())},
        "flat_baseline": {"confirm_pct": FLAT_CONFIRM_PCT, "confirm_rate_pct": round(n_confirmed_flat/n_usable*100,1) if n_usable else None,
                           "best_trail_width_pct": flat_best_w,
                           "results_by_width": {str(w): flat_aggs[w] for w in flat_aggs}},
        "atr_relative": atr_summary,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
