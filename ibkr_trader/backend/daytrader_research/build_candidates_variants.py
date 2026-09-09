"""
CEO asked (2026-09-07) for two more backtests on the premarket shortlist
step: (1) does adding a 52-week-range factor to the scoring find better
candidates, (2) does removing the sector cap (MAX_PER_SECTOR=3) find
better results. Both require re-deriving candidate SELECTION from the
full universe, not just reusing the already-sector-capped candidates.csv
-- this script does that, reusing the exact same real inputs
build_candidates.py already established (load_universe, compute_dt_scores,
apply_sector_cap imported directly, not re-derived, same discipline).

Produces THREE parallel (date, ticker) selections from the SAME real
scored universe on each real day:
  (a) baseline   -- exact reproduction of the current live method (sector
                     cap applied, top MAX_POSITIONS). Should closely match
                     the existing candidates.csv as a sanity check.
  (b) no_sector_cap -- same scoring, sector cap skipped, straight top
                     MAX_POSITIONS by composite_score.
  (c) with_52w   -- composite_score blended with a real 52-week-range
                     position (computed from the SAME 5y daily panel
                     download_all already pulls, no extra fetch needed),
                     THEN sector-capped exactly like baseline -- isolates
                     the scoring-factor question from the sector-cap
                     question.

Real, stated approximation: 52w range position = (yesterday_close - 252
trading-day trailing low) / (252-day trailing high - trailing low),
0=at the low, 1=at the high -- no lookahead, uses the SAME t-1-close-only
convention as every other feature in this pipeline.
"""
import sys
sys.path.insert(0, r"C:\Projects\GenAI-Projects\ibkr_trader\backend")

import pandas as pd
from datetime import datetime, timedelta

from daytrader_scanner import load_universe, compute_dt_scores, apply_sector_cap, MAX_PER_SECTOR
from daytrader_sizing_backtest import download_all, build_ticker_frame

MIN_ATR_PCT = 2.5
MIN_COMPOSITE = 75.0
MAX_POSITIONS = 10
RECENT_DAYS = 120
RANGE_WINDOW = 252  # ~52 trading weeks

HERE = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]


def main():
    tickers, sector_map = load_universe()
    print(f"Universe: {len(tickers)} tickers")

    raw = download_all(tickers)
    print(f"Downloaded {len(raw)}/{len(tickers)} tickers")

    frames = []
    range_lookup = {}  # (ticker, date_str) -> 52w range position, computed once per ticker
    for tk, df in raw.items():
        f = build_ticker_frame(tk, df)
        if f is None or len(f) <= 50:
            continue
        frames.append(f)

        # Real 52w range position from the SAME raw daily panel, no lookahead:
        # trailing 252-day window ending at t-1 (yesterday), matching every
        # other feature's t-1-close-only convention.
        d2 = df.copy()
        d2.columns = [c if isinstance(c, str) else c[0] for c in d2.columns]
        closes = d2["Close"]
        roll_high = closes.rolling(RANGE_WINDOW, min_periods=60).max().shift(1)
        roll_low = closes.rolling(RANGE_WINDOW, min_periods=60).min().shift(1)
        prior_close = closes.shift(1)
        span = (roll_high - roll_low).replace(0, pd.NA)
        range_pos = (prior_close - roll_low) / span
        for dt_idx, val in range_pos.items():
            if pd.notna(val):
                range_lookup[(tk, dt_idx.strftime("%Y-%m-%d"))] = float(val)

    panel = pd.concat(frames, ignore_index=True)
    print(f"Panel: {len(panel):,} ticker-days")

    cutoff = pd.Timestamp(datetime.now() - timedelta(days=RECENT_DAYS))
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= cutoff].copy()
    print(f"After {RECENT_DAYS}-day recent-window cutoff: {len(panel):,} ticker-days")

    panel = panel[panel["atr_pct"] >= MIN_ATR_PCT].copy()
    print(f"After ATR floor ({MIN_ATR_PCT}%): {len(panel):,} ticker-days")

    baseline_log, no_cap_log, with_52w_log = [], [], []
    dates = sorted(panel["date"].unique())
    n_missing_range = 0
    for d in dates:
        day_rows = panel[panel["date"] == d]
        candidates = day_rows.to_dict("records")
        if not candidates:
            continue
        for c in candidates:
            if pd.isna(c.get("ret5d_prior")):
                c["ret5d_prior"] = None

        # ---- (a) baseline: exact reproduction ----
        cands_a = [dict(c) for c in candidates]
        compute_dt_scores(cands_a)
        cands_a = [c for c in cands_a if c["composite_score"] >= MIN_COMPOSITE]
        cands_a.sort(key=lambda c: c["composite_score"], reverse=True)
        capped_a = apply_sector_cap(cands_a, sector_map, MAX_PER_SECTOR)[:MAX_POSITIONS]
        for c in capped_a:
            baseline_log.append({"date": d.strftime("%Y-%m-%d"), "ticker": c["ticker"],
                                  "score": c["composite_score"], "atr_pct": c["atr_pct"],
                                  "gap_pct": c.get("gap_pct")})

        # ---- (b) no sector cap ----
        cands_b = [dict(c) for c in candidates]
        compute_dt_scores(cands_b)
        cands_b = [c for c in cands_b if c["composite_score"] >= MIN_COMPOSITE]
        cands_b.sort(key=lambda c: c["composite_score"], reverse=True)
        capped_b = cands_b[:MAX_POSITIONS]
        for c in capped_b:
            no_cap_log.append({"date": d.strftime("%Y-%m-%d"), "ticker": c["ticker"],
                                "score": c["composite_score"], "atr_pct": c["atr_pct"],
                                "gap_pct": c.get("gap_pct")})

        # ---- (c) with 52-week range blended in, then sector-capped like baseline ----
        cands_c = [dict(c) for c in candidates]
        compute_dt_scores(cands_c)
        d_str = d.strftime("%Y-%m-%d")
        for c in cands_c:
            rp = range_lookup.get((c["ticker"], d_str))
            if rp is None:
                n_missing_range += 1
                c["range_pos"] = 0.5  # neutral if not enough history yet
            else:
                c["range_pos"] = rp
            # 85% original composite, 15% real 52w-range position (favors
            # names near their highs -- the classic momentum-breakout
            # reading of 52w range; tested as a real, disclosed weighting,
            # not tuned to any known outcome)
            c["composite_score_52w"] = 0.85 * c["composite_score"] + 0.15 * (c["range_pos"] * 100)
        cands_c = [c for c in cands_c if c["composite_score"] >= MIN_COMPOSITE]
        cands_c.sort(key=lambda c: c["composite_score_52w"], reverse=True)
        capped_c = apply_sector_cap(cands_c, sector_map, MAX_PER_SECTOR)[:MAX_POSITIONS]
        for c in capped_c:
            with_52w_log.append({"date": d_str, "ticker": c["ticker"],
                                  "score": c["composite_score"], "score_52w": c["composite_score_52w"],
                                  "range_pos": c["range_pos"], "atr_pct": c["atr_pct"],
                                  "gap_pct": c.get("gap_pct")})

    print(f"\n(missing 52w-range history for {n_missing_range} candidate evaluations -- defaulted neutral)")

    df_a = pd.DataFrame(baseline_log)
    df_b = pd.DataFrame(no_cap_log)
    df_c = pd.DataFrame(with_52w_log)
    df_a.to_csv(f"{HERE}/candidates_baseline_repro.csv", index=False)
    df_b.to_csv(f"{HERE}/candidates_no_sector_cap.csv", index=False)
    df_c.to_csv(f"{HERE}/candidates_with_52w.csv", index=False)

    print(f"\nbaseline repro:   {len(df_a)} rows, {df_a['ticker'].nunique()} unique tickers")
    print(f"no sector cap:    {len(df_b)} rows, {df_b['ticker'].nunique()} unique tickers")
    print(f"with 52w range:   {len(df_c)} rows, {df_c['ticker'].nunique()} unique tickers")

    # Real overlap check vs the ORIGINAL candidates.csv (sanity: baseline
    # repro should closely match, since it's the same real method)
    try:
        orig = pd.read_csv(f"{HERE}/candidates.csv")
        orig_pairs = set(zip(orig["date"], orig["ticker"]))
        repro_pairs = set(zip(df_a["date"], df_a["ticker"]))
        overlap = len(orig_pairs & repro_pairs)
        print(f"\nSanity check vs original candidates.csv: {overlap}/{len(orig_pairs)} original pairs "
              f"reproduced ({overlap/len(orig_pairs)*100:.1f}%) -- expect high overlap, not necessarily "
              f"100% (fresh yfinance pull, different day, minor data revisions possible)")
    except Exception as exc:
        print(f"Could not compare against original candidates.csv: {exc}")

    # Incremental (ticker,date) pairs not already covered by minute_bars.json
    import json
    with open(f"{HERE}/minute_bars.json") as f:
        existing_keys = set(json.load(f).keys())
    for name, df in [("no_sector_cap", df_b), ("with_52w", df_c)]:
        pairs = set(f"{tk}:{d}" for tk, d in zip(df["ticker"], df["date"]))
        new_pairs = pairs - existing_keys
        print(f"{name}: {len(new_pairs)} (ticker,date) pairs need fresh minute-bar fetch "
              f"(of {len(pairs)} total)")


if __name__ == "__main__":
    main()
