"""
Fetches the 4 tickers not already in breakout_research/universe_5y_ohlcv.pkl
that this sub-sector rotation study needs (TSM, WDC, STX, SNDK), matching
the same 5y daily window, then builds a combined pickle covering every
ticker this study's baskets use. Reuses the existing universe pickle for
everything already there rather than re-fetching 100+ tickers we don't need.
"""
import pickle
from pathlib import Path

import yfinance as yf

HERE = Path(__file__).parent
BACKEND_DIR = HERE.parent
UNIVERSE_PKL = BACKEND_DIR / "breakout_research" / "universe_5y_ohlcv.pkl"
OUT_PKL = HERE / "sector_rotation_ohlcv.pkl"

NEEDED_TICKERS = [
    # Hyperscalers (AI infra buyers)
    "MSFT", "GOOGL", "AMZN", "META", "ORCL",
    # Chipmakers (AI compute)
    "NVDA", "AMD", "AVGO", "MRVL", "TSM",
    # Memory (HBM/DRAM, per real business-model check 2026-09-09 -- MU and
    # SNDK genuinely provide HBM; STX/WDC do not, kept below as a control)
    "MU", "SNDK",
    # Storage (non-HBM control group -- HDD makers, NOT memory-chip makers)
    "STX", "WDC",
    # Semi-cap equipment (adjacent 5th group, makes the tools that make chips)
    "AMAT", "LRCX", "KLAC",
]


def main():
    with open(UNIVERSE_PKL, "rb") as f:
        universe = pickle.load(f)

    combined = {}
    missing = []
    for t in NEEDED_TICKERS:
        if t in universe:
            combined[t] = universe[t]
        else:
            missing.append(t)

    print(f"{len(combined)}/{len(NEEDED_TICKERS)} already in universe pickle. Fetching {len(missing)}: {missing}")

    for t in missing:
        hist = yf.Ticker(t).history(period="5y", interval="1d")
        if hist.empty:
            print(f"  {t}: EMPTY -- yfinance returned nothing, excluding from study")
            continue
        combined[t] = hist
        print(f"  {t}: fetched {len(hist)} rows, {hist.index[0].date()} -> {hist.index[-1].date()}")

    with open(OUT_PKL, "wb") as f:
        pickle.dump(combined, f)
    print(f"\nSaved {len(combined)} tickers to {OUT_PKL}")


if __name__ == "__main__":
    main()
