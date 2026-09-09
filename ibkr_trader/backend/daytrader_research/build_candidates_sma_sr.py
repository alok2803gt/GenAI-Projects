"""
CEO asked (2026-09-07) to also test support/resistance proximity and
various SMAs/EMAs in the premarket shortlist scoring. GEX explicitly
skipped -- no existing cache, and the gex-vex-calculator skill's own
runtime (~2min/ticker near-the-money options scan) would take 15-20+
hours across the full 503-ticker universe; it's a once-daily batch job
for a small curated universe, not built to scale to a full-market scan.

Re-downloads the universe panel (SAME real inputs as build_candidates_
variants.py) and this time SAVES it to a pickle for reuse across further
iteration. Computes, from the SAME real daily panel (no extra fetch):
  - 20-day rolling range position (near-term S/R proxy, same math as the
    52-week-range factor already tested, shorter window)
  - Classic daily pivot-point distance: pivot=(prior_high+prior_low+
    prior_close)/3, distance = (prior_close-pivot)/pivot*100 -- a
    different, complementary S/R concept (immediate day-to-day levels,
    not a rolling extreme)
  - SMA20/50/200 relative position: (prior_close-SMA)/SMA*100
  - EMA9/21 relative position: (prior_close-EMA)/EMA*100
All t-1-close-only, no lookahead, same convention as every other feature
in this pipeline.

Each new factor is tested INDIVIDUALLY first (blended into the existing
composite_score at a modest weight, same 85/15 split used for the 52-week
test, sector-capped exactly like baseline) -- isolates one variable at a
time rather than a combinatorial grid, matching this session's established
discipline and avoiding overfitting an 820-trade sample.
"""
import pickle
import sys
sys.path.insert(0, r"C:\Projects\GenAI-Projects\ibkr_trader\backend")

import pandas as pd
from datetime import datetime, timedelta

from daytrader_scanner import load_universe, compute_dt_scores, apply_sector_cap, MAX_PER_SECTOR

MIN_ATR_PCT = 2.5
MIN_COMPOSITE = 75.0
MAX_POSITIONS = 10
RECENT_DAYS = 120

HERE = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]


def build_ticker_frame_with_factors(ticker: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """Same base features as daytrader_sizing_backtest.build_ticker_frame
    (imported inline below to avoid a second yfinance download module
    diverging), PLUS the new real S/R and SMA/EMA factors."""
    sys.path.insert(0, r"C:\Projects\GenAI-Projects\ibkr_trader\backend")
    from daytrader_sizing_backtest import build_ticker_frame
    base = build_ticker_frame(ticker, df)
    if base is None:
        return None

    d2 = df.copy()
    d2.columns = [c if isinstance(c, str) else c[0] for c in d2.columns]
    closes = d2["Close"]
    highs = d2["High"]
    lows = d2["Low"]

    prior_close = closes.shift(1)
    prior_high = highs.shift(1)
    prior_low = lows.shift(1)

    # 20-day rolling range position (near-term S/R proxy)
    roll_high20 = closes.rolling(20, min_periods=15).max().shift(1)
    roll_low20 = closes.rolling(20, min_periods=15).min().shift(1)
    span20 = (roll_high20 - roll_low20).replace(0, pd.NA)
    range_pos_20d = (prior_close - roll_low20) / span20

    # Classic daily pivot point distance
    pivot = (prior_high + prior_low + prior_close) / 3
    pivot_dist_pct = (prior_close - pivot) / pivot.replace(0, pd.NA) * 100

    # SMA relative position
    sma20 = closes.rolling(20, min_periods=15).mean().shift(1)
    sma50 = closes.rolling(50, min_periods=35).mean().shift(1)
    sma200 = closes.rolling(200, min_periods=150).mean().shift(1)
    sma20_dist_pct = (prior_close - sma20) / sma20.replace(0, pd.NA) * 100
    sma50_dist_pct = (prior_close - sma50) / sma50.replace(0, pd.NA) * 100
    sma200_dist_pct = (prior_close - sma200) / sma200.replace(0, pd.NA) * 100

    # EMA relative position
    ema9 = closes.ewm(span=9, adjust=False).mean().shift(1)
    ema21 = closes.ewm(span=21, adjust=False).mean().shift(1)
    ema9_dist_pct = (prior_close - ema9) / ema9.replace(0, pd.NA) * 100
    ema21_dist_pct = (prior_close - ema21) / ema21.replace(0, pd.NA) * 100

    factors = pd.DataFrame({
        "date": d2.index,
        "range_pos_20d": range_pos_20d.values,
        "pivot_dist_pct": pivot_dist_pct.values,
        "sma20_dist_pct": sma20_dist_pct.values,
        "sma50_dist_pct": sma50_dist_pct.values,
        "sma200_dist_pct": sma200_dist_pct.values,
        "ema9_dist_pct": ema9_dist_pct.values,
        "ema21_dist_pct": ema21_dist_pct.values,
    })
    factors["date"] = pd.to_datetime(factors["date"])
    base["date"] = pd.to_datetime(base["date"])
    return base.merge(factors, on="date", how="left")


def main():
    from daytrader_sizing_backtest import download_all

    tickers, sector_map = load_universe()
    print(f"Universe: {len(tickers)} tickers")

    raw = download_all(tickers)
    print(f"Downloaded {len(raw)}/{len(tickers)} tickers")

    # Save the raw panel for reuse -- avoids a third redundant yfinance download if more iteration is needed
    with open(f"{HERE}/_raw_universe_panel.pkl", "wb") as f:
        pickle.dump(raw, f)
    print("Saved _raw_universe_panel.pkl for reuse")

    frames = []
    for tk, df in raw.items():
        f = build_ticker_frame_with_factors(tk, df)
        if f is not None and len(f) > 50:
            frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    print(f"Panel with new factors: {len(panel):,} ticker-days")

    cutoff = pd.Timestamp(datetime.now() - timedelta(days=RECENT_DAYS))
    panel = panel[panel["date"] >= cutoff].copy()
    panel = panel[panel["atr_pct"] >= MIN_ATR_PCT].copy()
    print(f"After recent-window + ATR floor: {len(panel):,} ticker-days")

    NEW_FACTORS = ["range_pos_20d", "pivot_dist_pct", "sma20_dist_pct",
                   "sma50_dist_pct", "sma200_dist_pct", "ema9_dist_pct", "ema21_dist_pct"]

    variant_logs = {name: [] for name in NEW_FACTORS}
    baseline_log = []
    dates = sorted(panel["date"].unique())

    for d in dates:
        day_rows = panel[panel["date"] == d]
        candidates = day_rows.to_dict("records")
        if not candidates:
            continue
        for c in candidates:
            if pd.isna(c.get("ret5d_prior")):
                c["ret5d_prior"] = None

        # baseline (for this run's own sanity/consistency)
        cands_base = [dict(c) for c in candidates]
        compute_dt_scores(cands_base)
        cands_base = [c for c in cands_base if c["composite_score"] >= MIN_COMPOSITE]
        cands_base.sort(key=lambda c: c["composite_score"], reverse=True)
        capped_base = apply_sector_cap(cands_base, sector_map, MAX_PER_SECTOR)[:MAX_POSITIONS]
        for c in capped_base:
            baseline_log.append({"date": d.strftime("%Y-%m-%d"), "ticker": c["ticker"]})

        # each new factor individually: normalize to a 0-100 rank within the
        # day's pool, blend 85/15 with the existing composite (same weighting
        # already used and reported for the 52-week test)
        for factor in NEW_FACTORS:
            vals = [c.get(factor) for c in candidates if pd.notna(c.get(factor))]
            if len(vals) < 10:
                continue
            lo, hi = min(vals), max(vals)
            span = (hi - lo) or 1.0
            cands_f = [dict(c) for c in candidates]
            compute_dt_scores(cands_f)
            for c in cands_f:
                raw_val = c.get(factor)
                pct = ((raw_val - lo) / span * 100) if pd.notna(raw_val) else 50.0
                c["composite_score_f"] = 0.85 * c["composite_score"] + 0.15 * pct
            cands_f = [c for c in cands_f if c["composite_score"] >= MIN_COMPOSITE]
            cands_f.sort(key=lambda c: c["composite_score_f"], reverse=True)
            capped_f = apply_sector_cap(cands_f, sector_map, MAX_PER_SECTOR)[:MAX_POSITIONS]
            for c in capped_f:
                variant_logs[factor].append({"date": d.strftime("%Y-%m-%d"), "ticker": c["ticker"]})

    pd.DataFrame(baseline_log).to_csv(f"{HERE}/candidates_baseline_repro2.csv", index=False)
    print(f"\nbaseline repro2: {len(baseline_log)} rows")
    for factor in NEW_FACTORS:
        df_f = pd.DataFrame(variant_logs[factor])
        df_f.to_csv(f"{HERE}/candidates_{factor}.csv", index=False)
        print(f"{factor}: {len(df_f)} rows, {df_f['ticker'].nunique() if len(df_f) else 0} unique tickers")

    # Incremental minute-bar pairs needed across ALL new-factor variants combined
    import json
    with open(f"{HERE}/minute_bars.json") as f:
        existing_keys = set(json.load(f).keys())
    inc_path = f"{HERE}/minute_bars_incremental.json"
    try:
        with open(inc_path) as f:
            existing_keys |= set(json.load(f).keys())
    except FileNotFoundError:
        pass

    union_new = set()
    for factor in NEW_FACTORS:
        df_f = pd.DataFrame(variant_logs[factor])
        if len(df_f) == 0:
            continue
        pairs = set(f"{tk}:{d}" for tk, d in zip(df_f["ticker"], df_f["date"]))
        union_new |= (pairs - existing_keys)
    print(f"\nTotal NEW (ticker,date) pairs needed across all 7 factor variants: {len(union_new)}")
    pd.DataFrame([{"ticker": p.split(":")[0], "date": p.split(":")[1]} for p in union_new]) \
        .to_csv(f"{HERE}/_incremental_pairs_needed_2.csv", index=False)
    print("Saved _incremental_pairs_needed_2.csv")


if __name__ == "__main__":
    main()
