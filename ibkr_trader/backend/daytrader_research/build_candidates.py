"""
Real candidate selection for the day-trading strategy comparison -- reuses
the LIVE scanner's own selection logic directly (load_universe,
compute_dt_scores, apply_sector_cap from daytrader_scanner.py; download_all/
build_ticker_frame from daytrader_sizing_backtest.py), same pattern that
script already established: "imports compute_dt_scores/apply_sector_cap
directly from it, not a re-derived copy, so this can't silently drift from
the live logic." Read-only imports -- never modifies either file.

Produces the SAME set of (date, ticker) candidates the live Day Trader
scanner would have selected on each real historical day, restricted to a
RECENT window (not the full 5y) since minute-bar data is the real
constraint for testing entry/exit mechanics -- matches
daytrader_intraday_backtest.py's own "RECENT_TRADING_DAYS" convention.

Output: candidates.csv (date, ticker, score, atr_pct, gap_pct, open, high,
low, close from THAT DAY's daily bar -- used by build_candidates only for
context/sanity, minute-level entry/exit simulation happens in a separate
script against real intraday bars).
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
RECENT_DAYS = 120  # calendar days back; minute-bar fetch is the real constraint downstream

HERE = __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0]


def main():
    tickers, sector_map = load_universe()
    print(f"Universe: {len(tickers)} tickers")

    raw = download_all(tickers)
    print(f"Downloaded {len(raw)}/{len(tickers)} tickers")

    frames = []
    for tk, df in raw.items():
        f = build_ticker_frame(tk, df)
        if f is not None and len(f) > 50:
            frames.append(f)
    panel = pd.concat(frames, ignore_index=True)
    print(f"Panel: {len(panel):,} ticker-days")

    cutoff = pd.Timestamp(datetime.now() - timedelta(days=RECENT_DAYS))
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[panel["date"] >= cutoff].copy()
    print(f"After {RECENT_DAYS}-day recent-window cutoff: {len(panel):,} ticker-days")

    panel = panel[panel["atr_pct"] >= MIN_ATR_PCT].copy()
    print(f"After ATR floor ({MIN_ATR_PCT}%): {len(panel):,} ticker-days")

    trade_log = []
    dates = sorted(panel["date"].unique())
    for d in dates:
        day_rows = panel[panel["date"] == d]
        candidates = day_rows.to_dict("records")
        if not candidates:
            continue
        for c in candidates:
            if pd.isna(c.get("ret5d_prior")):
                c["ret5d_prior"] = None
        compute_dt_scores(candidates)
        candidates = [c for c in candidates if c["composite_score"] >= MIN_COMPOSITE]
        if not candidates:
            continue
        candidates.sort(key=lambda c: c["composite_score"], reverse=True)
        capped = apply_sector_cap(candidates, sector_map, MAX_PER_SECTOR)[:MAX_POSITIONS]
        for c in capped:
            trade_log.append({
                "date": d.strftime("%Y-%m-%d"), "ticker": c["ticker"], "score": c["composite_score"],
                "atr_pct": c["atr_pct"], "gap_pct": c.get("gap_pct"),
                "open": c.get("open"), "high": c.get("high"), "low": c.get("low"), "close": c.get("close"),
            })

    df = pd.DataFrame(trade_log)
    df.to_csv(f"{HERE}/candidates.csv", index=False)
    print(f"\nSaved {len(df)} candidate (date, ticker) selections to candidates.csv, "
          f"{df['date'].nunique()} trading days, {df['ticker'].nunique()} unique tickers")


if __name__ == "__main__":
    main()
