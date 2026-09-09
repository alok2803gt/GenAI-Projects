"""
CEO question 2026-09-05: "the watch time... what are you watching during
that time, can you backtest that and see if we are losing" -- i.e., of the
real candidates that NEVER confirm within the live 60-minute watch window
(WATCH_EXPIRED, dropped with zero P&L), are we actually walking away from
good trades, or correctly filtering out bad ones? Same real 820-candidate
1-minute-bar dataset as every other test in this directory.

Three real questions, each with its own simulation:
  1. Of the candidates that never confirm within 60min, what would have
     happened if we'd given up waiting and bought anyway at the minute-60
     mark, holding with the SAME 0.3% trailing stop as every confirmed
     trade? This directly measures the opportunity cost (or lack of one)
     of walking away.
  2. Of those same never-confirmed candidates, how many DID eventually
     clear the 0.35% price threshold later in the day (just without ever
     getting real interval-volume confirmation) -- are we cutting off a
     price move that was already coming, just missing the volume half?
  3. What if the window were longer (90min, 120min, full day) instead of
     60min -- does extending it rescue any of these, and if so, how would
     THOSE later entries have performed?
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from rsi_1min_confirm_backtest import CONFIRM_PCT, TRAIL_PCT, _agg

HERE = Path(__file__).parent
BARS_PATH = HERE / "minute_bars.json"
CONFIRM_WINDOW_MIN = 60
EXTENDED_WINDOWS = [90, 120, 390]  # 390 = full session (9:30-15:59 ET, no cutoff)


def real_confirm_entry(bars, window_min):
    """Exact live mechanics: price >= open*1.0035 AND this-interval volume
    >= running median of intervals seen so far, within window_min minutes.
    Returns (entry_idx, entry_px) or (None, None) if never confirmed."""
    day_open = bars[0]["open"]
    confirm_px = day_open * (1 + CONFIRM_PCT / 100)
    vols_so_far = []
    for i, b in enumerate(bars[:window_min]):
        vols_so_far.append(b["volume"])
        median_vol = sorted(vols_so_far)[len(vols_so_far) // 2]
        if b["close"] >= confirm_px and b["volume"] >= median_vol:
            return i, b["close"]
    return None, None


def trail_from(bars, entry_idx, entry_px):
    running_high = entry_px
    for b in bars[entry_idx + 1:]:
        running_high = max(running_high, b["high"])
        stop_px = running_high * (1 - TRAIL_PCT / 100)
        if b["low"] <= stop_px:
            return {"ret_pct": (stop_px / entry_px - 1) * 100, "outcome": "trailed_out"}
    exit_px = bars[-1]["close"]
    return {"ret_pct": (exit_px / entry_px - 1) * 100, "outcome": "eod_close"}


def main():
    with open(BARS_PATH) as f:
        bar_data = json.load(f)

    confirmed_60 = []
    never_confirmed_60 = []
    for key, bars in bar_data.items():
        if not bars or len(bars) < 10:
            continue
        idx, px = real_confirm_entry(bars, CONFIRM_WINDOW_MIN)
        if idx is not None:
            confirmed_60.append((key, bars, idx, px))
        else:
            never_confirmed_60.append((key, bars))

    print(f"Real, live-mechanics confirm within {CONFIRM_WINDOW_MIN}min: "
          f"{len(confirmed_60)} confirmed, {len(never_confirmed_60)} never confirmed "
          f"(of {len(confirmed_60)+len(never_confirmed_60)} usable candidates)\n")

    # ── Q1: buy anyway at minute-60 mark for the never-confirmed group ─────
    forced_entry_results = []
    for key, bars in never_confirmed_60:
        if len(bars) <= CONFIRM_WINDOW_MIN:
            continue
        entry_px = bars[CONFIRM_WINDOW_MIN - 1]["close"]
        forced_entry_results.append(trail_from(bars, CONFIRM_WINDOW_MIN - 1, entry_px))

    real_confirmed_results = [trail_from(bars, idx, px) for key, bars, idx, px in confirmed_60]

    print("=== Q1: 'Buy anyway at minute 60' for candidates that never confirmed ===")
    agg_forced = _agg(forced_entry_results)
    agg_real = _agg(real_confirmed_results)
    print(f"  Real CONFIRMED trades (what we actually take):     {agg_real}")
    print(f"  Forced entry on NEVER-CONFIRMED (what we skip):    {agg_forced}")
    print(f"  --> Walking away from these {len(forced_entry_results)} candidates cost us: "
          f"{'a REAL, positive opportunity we are missing' if agg_forced.get('avg_ret_pct',0) > 0 else 'NOTHING -- they were net losers, correctly filtered'}")

    # ── Q2: of the never-confirmed, how many eventually cleared price alone ──
    day_open_cleared_late = 0
    for key, bars in never_confirmed_60:
        day_open = bars[0]["open"]
        confirm_px = day_open * (1 + CONFIRM_PCT / 100)
        if any(b["close"] >= confirm_px for b in bars[CONFIRM_WINDOW_MIN:]):
            day_open_cleared_late += 1
    print(f"\n=== Q2: Of {len(never_confirmed_60)} never-confirmed, "
          f"{day_open_cleared_late} ({day_open_cleared_late/len(never_confirmed_60)*100:.1f}%) "
          f"DID eventually clear the 0.35% price level later in the day "
          f"(just never got real volume confirmation) ===")

    # ── Q3: extend the window -- does more time rescue any of these? ───────
    print(f"\n=== Q3: Extending the confirmation window beyond {CONFIRM_WINDOW_MIN}min ===")
    for w in EXTENDED_WINDOWS:
        rescued = []
        for key, bars in never_confirmed_60:
            idx, px = real_confirm_entry(bars, min(w, len(bars)))
            if idx is not None and idx >= CONFIRM_WINDOW_MIN:
                rescued.append(trail_from(bars, idx, px))
        agg = _agg(rescued)
        print(f"  Window={w:3d}min: rescues {len(rescued):3d} of {len(never_confirmed_60)} "
              f"never-confirmed-at-60min candidates -- {agg}")

    out = {
        "confirm_window_min": CONFIRM_WINDOW_MIN,
        "n_confirmed_60": len(confirmed_60), "n_never_confirmed_60": len(never_confirmed_60),
        "real_confirmed_agg": agg_real, "forced_entry_at_60_agg": agg_forced,
        "never_confirmed_but_cleared_price_late": day_open_cleared_late,
    }
    with open(HERE / "watch_expiry_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {HERE / 'watch_expiry_results.json'}")


if __name__ == "__main__":
    main()
